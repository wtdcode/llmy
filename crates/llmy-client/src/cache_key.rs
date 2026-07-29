//! Automatic `prompt_cache_key` management.
//!
//! `prompt_cache_key` is a routing hint: requests sharing a key are steered to
//! the machine that already holds their cached prefix. Picking one by hand for
//! every conversation is tedious and easy to get wrong, so when the caller does
//! not supply one we pick it here, from what this client has already sent.
//!
//! The model is a chain of rolling hashes, one link per prompt block (the tool
//! definitions, then each message), each hashed as the prompt text it renders
//! to. Link `i` covers the whole prefix `0..=i`, so two requests share a link
//! exactly when they share that stretch of prompt.
//!
//! Selection walks the chain from the longest link down and takes the first one
//! already claimed by a live key with request budget left, so the deepest shared
//! prefix wins and a saturated key falls through to the next-best one. Claiming
//! covers only the links the provider will really store a cache entry at, which
//! is where the two [`CachePolicy`] flavours differ — see [`cache_points`].
//!
//! A claim is an assertion that a prefix is cached under a key, so it is only
//! settled once a request carrying it has actually been answered. Until then it
//! is held *in flight*: visible to concurrent siblings, so a fan-out sharing a
//! prefix still converges on one key, but dropped if the request never lands.
//! In-flight claims are refcounted, so one sibling failing cannot pull a claim
//! out from under another that is still going. See [`CacheKeyClaim`].

use std::collections::hash_map::{DefaultHasher, Entry};
use std::collections::{HashMap, VecDeque};
use std::hash::Hasher;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, Instant};

use crate::debug::completion_to_string;
use crate::model::{CachePolicy, OpenAIModel};
use crate::req::{PromptCacheMode, RawExtensibleChatCompletionRequest};

/// Rolling hash of a prompt prefix. A collision only ever costs a suboptimal
/// routing hint, never correctness, so 64 bits is plenty.
type PrefixHash = u64;

/// Opaque handle for one minted key.
type KeyId = u64;

/// The window the `req/min` budget is measured over.
const RATE_WINDOW: Duration = Duration::from_secs(60);

/// OpenAI steers one cache key to one machine, and warns that sustaining more
/// than ~15 requests/minute on a single key spills over to more machines and
/// costs you hit rate. So that is the default ceiling before we spread out.
pub const DEFAULT_MAX_RPM: u32 = 15;

/// Below this share of the expected prefix actually coming back cached, routing
/// is not buying what it should and the prompt shape is the likely culprit — a
/// prefix that moves between turns, a timestamp in the system prompt, a toolbox
/// rebuilt per request.
const CACHE_HIT_WARN_RATIO: f64 = 0.50;

/// OpenAI only caches prompts from this length up, so a shorter prefix coming
/// back uncached is normal and not worth complaining about.
const MIN_CACHEABLE_TOKENS: u64 = 1024;

/// Default idle lifetime of a key.
///
/// Deliberately generous, because expiry is a memory bound and nothing more. We
/// cannot see whether the provider still holds a cached prefix, and reusing a
/// key whose entry has lapsed costs nothing — the next request simply caches it
/// again. Dropping a key too early is the expensive direction: it mints a new
/// one, which is a new machine, and that is a guaranteed miss even when the old
/// entry was still live.
pub const DEFAULT_TTL_SECS: u64 = 4 * 60 * 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheKeyConfig {
    /// When false, requests without a caller-supplied key are sent without one.
    pub enabled: bool,
    /// How long a key survives without being used. See [`DEFAULT_TTL_SECS`] —
    /// this exists to keep a long-lived client from growing forever, not to
    /// track the provider's own cache lifetime.
    pub ttl: Duration,
    /// Requests per minute one key is allowed before we spread to another.
    pub max_rpm: u32,
}

impl Default for CacheKeyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl: Duration::from_secs(DEFAULT_TTL_SECS),
            max_rpm: DEFAULT_MAX_RPM,
        }
    }
}

#[derive(Debug)]
struct KeyState {
    key: String,
    /// Send times inside the trailing [`RATE_WINDOW`], for the req/min budget.
    sends: VecDeque<Instant>,
    last_used: Instant,
}

impl KeyState {
    fn drop_stale_sends(&mut self, now: Instant) {
        while let Some(oldest) = self.sends.front() {
            if now.duration_since(*oldest) >= RATE_WINDOW {
                self.sends.pop_front();
            } else {
                break;
            }
        }
    }

    fn has_budget(&mut self, now: Instant, max_rpm: u32) -> bool {
        self.drop_stale_sends(now);
        (self.sends.len() as u32) < max_rpm
    }

    fn record_send(&mut self, now: Instant) {
        self.sends.push_back(now);
        self.last_used = now;
    }

    fn expired(&self, now: Instant, ttl: Duration) -> bool {
        now.duration_since(self.last_used) >= ttl
    }
}

/// A prompt prefix claimed for a key: how big it is, and how far along the claim
/// has got.
#[derive(Debug, Clone, Copy)]
struct Claim {
    key: KeyId,
    /// Prompt tokens this prefix covers, measured once when it is first claimed
    /// so a later match reads the number rather than re-counting it.
    tokens: u64,
    /// Requests still in flight behind this claim. Refcounted so one sibling
    /// failing cannot strand another that is still going.
    holders: u32,
    /// Whether a request carrying this prefix has actually been answered. Until
    /// then the claim is provisional — visible to concurrent siblings so a
    /// fan-out converges on one key, but gone again if nothing lands.
    settled: bool,
}

/// Per-client registry mapping prompt prefixes to the cache key that owns them.
///
/// Shared across every [`crate::client::LLM`] handle cut from the same client
/// (billing scopes included), so a scoped sub-agent continuing a conversation
/// still lands on the machine holding its prefix.
#[derive(Debug)]
pub struct CacheKeyRegistry {
    config: CacheKeyConfig,
    keys: HashMap<KeyId, KeyState>,
    /// One prefix to one key — whoever claims it first owns it until that key
    /// expires, so a lineage keeps its original key as it grows.
    claims: HashMap<PrefixHash, Claim>,
    next_id: KeyId,
    pid: u32,
}

impl CacheKeyRegistry {
    pub fn new(config: CacheKeyConfig) -> Self {
        Self {
            config,
            keys: HashMap::new(),
            claims: HashMap::new(),
            next_id: 0,
            pid: std::process::id(),
        }
    }

    pub fn config(&self) -> CacheKeyConfig {
        self.config
    }

    /// Pick the cache key for `req` under `policy`, take its prefixes in flight,
    /// and charge the send against that key's budget.
    ///
    /// The claims are provisional until [`Self::confirm`]; see [`CacheKeyClaim`].
    /// Returns `None` only when auto keys are disabled.
    fn select(
        &mut self,
        req: &RawExtensibleChatCompletionRequest,
        model: &OpenAIModel,
    ) -> Option<(String, KeyId, Vec<PrefixHash>, u64)> {
        if !self.config.enabled {
            return None;
        }
        let blocks = prompt_blocks(req);
        let claimable = cache_points(req, model.cache_policy(), blocks.len());

        let now = Instant::now();
        self.expire(now);

        // Longest link first: the deepest shared prefix is the biggest cache
        // hit. A key that is out of budget is skipped, so we fall through to the
        // next-best (shorter) prefix rather than piling onto one machine.
        //
        // Whatever the matched prefix was measured at when it was claimed is
        // what we are betting comes back as a cache read; a fresh key bets on
        // nothing.
        let max_rpm = self.config.max_rpm;
        let mut chosen = None;
        for block in blocks.iter().rev() {
            let Some(&claim) = self.claims.get(&block.hash) else {
                continue;
            };
            let Some(state) = self.keys.get_mut(&claim.key) else {
                continue;
            };
            if state.has_budget(now, max_rpm) {
                chosen = Some(claim);
                break;
            }
        }

        let (id, expected_cached_tokens) = match chosen {
            Some(claim) => (claim.key, claim.tokens),
            None => (self.mint(now), 0),
        };

        // Only real cache points get claimed, or a key would advertise cached
        // content that was never stored.
        let mut held = Vec::with_capacity(claimable.len());
        // Blocks `[0, priced)` are already accounted for in `tokens`. Prefixes
        // somebody else already measured are adopted wholesale, so a growing
        // conversation only ever tokenizes the messages it just added.
        let mut priced = 0usize;
        let mut tokens = 0u64;

        for link in claimable {
            let Some(block) = blocks.get(link) else {
                continue;
            };
            match self.claims.get(&block.hash) {
                Some(claim) => {
                    tokens = claim.tokens;
                }
                None => {
                    for unpriced in &blocks[priced..=link] {
                        tokens += model.config.count_tokens_lossy(&unpriced.text) as u64;
                    }
                }
            }
            priced = link + 1;

            match self.claims.entry(block.hash) {
                Entry::Occupied(mut claimed) => {
                    // Already settled, or in flight under a different key: either
                    // way it belongs to someone else, so leave it alone.
                    if !claimed.get().settled && claimed.get().key == id {
                        claimed.get_mut().holders += 1;
                        held.push(block.hash);
                    }
                }
                Entry::Vacant(slot) => {
                    slot.insert(Claim {
                        key: id,
                        tokens,
                        holders: 1,
                        settled: false,
                    });
                    held.push(block.hash);
                }
            }
        }

        let state = self.keys.get_mut(&id)?;
        state.record_send(now);
        Some((state.key.clone(), id, held, expected_cached_tokens))
    }

    /// Charge one request against a key's budget. The provider counts total
    /// traffic per key, so every send costs a slot — including a retry, which is
    /// real extra traffic on the same machine.
    fn charge(&mut self, key: KeyId) {
        if let Some(state) = self.keys.get_mut(&key) {
            state.record_send(Instant::now());
        }
    }

    /// Settle the claims of a request that landed: those prefixes really were
    /// sent and answered, so later requests may route by them.
    fn confirm(&mut self, held: &[PrefixHash]) {
        for hash in held {
            if let Some(claim) = self.claims.get_mut(hash) {
                claim.settled = true;
                claim.holders = claim.holders.saturating_sub(1);
            }
        }
    }

    /// Drop the claims of a request that never landed — unless a sibling settled
    /// them meanwhile, or is still going.
    fn abandon(&mut self, held: &[PrefixHash]) {
        for hash in held {
            let Entry::Occupied(mut claim) = self.claims.entry(*hash) else {
                continue;
            };
            claim.get_mut().holders = claim.get().holders.saturating_sub(1);
            if claim.get().holders == 0 && !claim.get().settled {
                claim.remove();
            }
        }
    }

    fn mint(&mut self, now: Instant) -> KeyId {
        self.next_id += 1;
        let id = self.next_id;
        self.keys.insert(
            id,
            KeyState {
                // The pid keeps concurrent processes off each other's machines.
                key: format!("llmy-{}-{}", self.pid, id),
                sends: VecDeque::new(),
                last_used: now,
            },
        );
        id
    }

    /// Drop idle-expired keys, and any prefix claims they were holding. This is
    /// housekeeping against unbounded growth; see [`DEFAULT_TTL_SECS`] for why
    /// it is not trying to mirror the provider's cache lifetime.
    fn expire(&mut self, now: Instant) {
        let ttl = self.config.ttl;
        let before = self.keys.len();
        self.keys.retain(|_, state| !state.expired(now, ttl));
        if self.keys.len() != before {
            let live = &self.keys;
            self.claims.retain(|_, claim| live.contains_key(&claim.key));
        }
    }
}

/// Shared, cloneable handle to one client's [`CacheKeyRegistry`].
///
/// Every [`crate::client::LLM`] cut from the same client — billing scopes
/// included — holds a clone, so a scoped sub-agent continuing a conversation
/// routes to the machine already holding its prefix.
#[derive(Debug, Clone)]
pub struct CacheKeys(Arc<StdRwLock<CacheKeyRegistry>>);

impl CacheKeys {
    pub fn new(config: CacheKeyConfig) -> Self {
        Self(Arc::new(StdRwLock::new(CacheKeyRegistry::new(config))))
    }

    pub fn config(&self) -> CacheKeyConfig {
        self.0.read().expect("cache keys poisoned").config()
    }

    /// Pick the cache key for `req`, holding its prefixes in flight until the
    /// returned claim settles. `None` when auto keys are disabled.
    ///
    /// Selection and the rate-limit charge happen under one lock, so concurrent
    /// callers cannot both take the last slot on a key.
    pub fn select(
        &self,
        req: &RawExtensibleChatCompletionRequest,
        model: &OpenAIModel,
    ) -> Option<CacheKeyClaim> {
        let (key, id, held, expected_cached_tokens) = self
            .0
            .write()
            .expect("cache keys poisoned")
            .select(req, model)?;
        Some(CacheKeyClaim(Arc::new(HeldClaim {
            keys: self.clone(),
            key,
            id,
            held,
            expected_cached_tokens,
            settled: AtomicBool::new(false),
        })))
    }
}

#[cfg(test)]
impl CacheKeys {
    fn key_count(&self) -> usize {
        self.0.read().unwrap().keys.len()
    }
    fn settled(&self) -> usize {
        self.0
            .read()
            .unwrap()
            .claims
            .values()
            .filter(|claim| claim.settled)
            .count()
    }
    fn in_flight(&self) -> usize {
        self.0
            .read()
            .unwrap()
            .claims
            .values()
            .filter(|claim| claim.holders > 0)
            .count()
    }
}

/// A cache key, plus the prefix claims the request carrying it holds until it
/// lands.
///
/// Cheap to clone, like a semaphore permit: one logical request takes a claim
/// and hands a clone to each of its attempts. [`Self::confirm`] settles it —
/// those prefixes really were sent under this key and answered, so later
/// requests may route by them — and is idempotent, so whichever attempt lands
/// first settles it. When the *last* clone drops unconfirmed the claim is
/// abandoned instead, so a request that never lands, an early return or a panic
/// leaves no prefix pointing at a key with nothing behind it.
///
/// Note this refcount is per logical request. The registry keeps its own count
/// per prefix, because *different* concurrent requests can share a prefix while
/// holding separate claims.
#[derive(Debug, Clone)]
pub struct CacheKeyClaim(Arc<HeldClaim>);

#[derive(Debug)]
struct HeldClaim {
    keys: CacheKeys,
    key: String,
    id: KeyId,
    held: Vec<PrefixHash>,
    /// Tokens the matched prefix covers — what we are betting the provider will
    /// serve from cache. Zero when this key is fresh and nothing is expected.
    expected_cached_tokens: u64,
    settled: AtomicBool,
}

impl CacheKeyClaim {
    /// The `prompt_cache_key` to send.
    pub fn key(&self) -> &str {
        &self.0.key
    }

    /// Tokens the matched prefix covers, i.e. what routing to this key is
    /// betting comes back as a cache read.
    pub fn expected_cached_tokens(&self) -> u64 {
        self.0.expected_cached_tokens
    }

    /// Charge one more request against this key's budget.
    ///
    /// The *first* send is already paid for by the selection that handed out
    /// this claim — picking a key and taking its slot happen under one lock, so
    /// a fan-out cannot all grab the same last slot. Later sends are retries,
    /// which the provider counts as traffic on the key just the same, so the
    /// retry loop charges them here.
    pub fn charge_resend(&self) {
        if let Ok(mut registry) = self.0.keys.0.write() {
            registry.charge(self.0.id);
        }
    }

    /// Settle the claims — call once the request has been answered — and report
    /// how the bet went against the `cached_tokens` the provider reported.
    ///
    /// Idempotent: later calls, from a retry that also landed or another clone,
    /// do nothing.
    pub fn confirm(&self, cached_tokens: u64) {
        if self.0.settled.swap(true, Ordering::Relaxed) {
            return;
        }
        self.0.report(cached_tokens);
        if let Ok(mut registry) = self.0.keys.0.write() {
            registry.confirm(&self.0.held);
        }
    }
}

impl HeldClaim {
    /// Compare what the matched prefix promised against what came back, so a
    /// prompt whose prefix is quietly unstable shows up in the logs instead of
    /// just costing money.
    fn report(&self, cached_tokens: u64) {
        // Nothing was expected (fresh key), or the prefix is too short for the
        // provider to cache at all — either way there is no bet to grade.
        if self.expected_cached_tokens < MIN_CACHEABLE_TOKENS {
            return;
        }
        let hit_rate = cached_tokens as f64 / self.expected_cached_tokens as f64;
        if hit_rate < CACHE_HIT_WARN_RATIO {
            tracing::warn!(
                "prompt cache key {} expected ~{} cached tokens but got {} ({:.0}%); \
                 the prompt prefix may not be stable across requests",
                self.key,
                self.expected_cached_tokens,
                cached_tokens,
                hit_rate * 100.0
            );
        } else {
            tracing::debug!(
                "prompt cache key {}: {}/{} prefix tokens cached ({:.0}%)",
                self.key,
                cached_tokens,
                self.expected_cached_tokens,
                hit_rate * 100.0
            );
        }
    }
}

impl Drop for HeldClaim {
    fn drop(&mut self) {
        if self.settled.load(Ordering::Relaxed) || self.held.is_empty() {
            return;
        }
        // Best-effort and panic-free, as required in `Drop`.
        if let Ok(mut registry) = self.keys.0.write() {
            registry.abandon(&self.held);
        }
    }
}

// ---------------------------------------------------------------------------
// Reading a request's cache shape
// ---------------------------------------------------------------------------

/// One block of the rendered prompt: its text, and the rolling hash of the whole
/// prefix ending at it.
struct Block {
    hash: PrefixHash,
    text: String,
}

/// Split a request into prompt blocks: the tool definitions first — they head
/// the rendered prompt, and changing them invalidates everything after — then
/// one per message. Block `i`'s hash covers the prefix `0..=i`, so two requests
/// share a hash exactly when they share that stretch of prompt.
///
/// Blocks are hashed as the **prompt text** they render to, via
/// [`completion_to_string`], not as the JSON we happen to hold. What the
/// provider caches is tokens, so anything that never reaches the model — a
/// `prompt_cache_breakpoint` marker, the lone-text array that
/// [`crate::req::RawExtensibleChatRequestMessage::breakpoint`] promotes content
/// into, a vendor extension field — must not look like a different conversation.
/// The bias is deliberate: a false match only wastes a lookup on the wrong
/// machine, while a false miss throws away a real cache hit.
///
/// The rendered text is kept rather than dropped, so pricing a newly claimed
/// prefix does not have to render it a second time.
fn prompt_blocks(req: &RawExtensibleChatCompletionRequest) -> Vec<Block> {
    let mut blocks = Vec::with_capacity(req.messages.len() + 1);
    // One hasher fed block by block. `Hasher::finish` reads the running value
    // without resetting it, so block `i` comes out carrying the hash of the
    // whole prefix `0..=i` for a single pass over the prompt.
    let mut hasher = DefaultHasher::new();

    // Tool definitions go into the prompt as their schemas, so hash them so.
    let tools = serde_json::to_string(&req.tools).unwrap_or_default();
    hasher.write(tools.as_bytes());
    blocks.push(Block {
        hash: hasher.finish(),
        text: tools,
    });

    for message in req.messages.iter() {
        let text = completion_to_string(message);
        hasher.write(text.as_bytes());
        blocks.push(Block {
            hash: hasher.finish(),
            text,
        });
    }
    blocks
}

/// Which links of [`prompt_blocks`] the provider will really store a cache entry
/// at — the only ones a key may claim. Lookup can match on *any* shared prefix,
/// but claiming one that never becomes an entry would have a key advertise
/// content that was never cached.
///
/// Under [`CachePolicy::PartialPrefix`] the provider transparently caches the
/// longest matching prefix, so every message link qualifies. Under
/// [`CachePolicy::Breakpoint`] only the declared breakpoints do — plus, in the
/// default `implicit` mode, the end of the request, because the API adds a
/// breakpoint of its own on the latest message.
///
/// Link 0 is never a cache point. It covers the tool definitions alone, which is
/// not a prompt prefix any provider stores; it exists only to poison every later
/// link when the tools change. Claiming it would collapse every conversation
/// sharing a toolbox — including all the ones with no tools at all — onto a
/// single key.
fn cache_points(
    req: &RawExtensibleChatCompletionRequest,
    policy: CachePolicy,
    chain_len: usize,
) -> Vec<usize> {
    if !policy.needs_breakpoints() {
        return (1..chain_len).collect();
    }

    // chain[0] is the tools block, so message `i` lives at chain[i + 1].
    let mut points: Vec<usize> = req
        .messages
        .iter()
        .enumerate()
        .filter(|(_, message)| message.has_cache_breakpoint())
        .map(|(index, _)| index + 1)
        .collect();

    let mode = req
        .prompt_cache_options
        .as_ref()
        .and_then(|options| options.mode)
        .unwrap_or_default();
    let last = chain_len - 1;
    if mode == PromptCacheMode::Implicit && last > 0 && points.last() != Some(&last) {
        points.push(last);
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::req::{
        ChatCompletionRequestMessageRaw, ChatCompletionRequestUserMessageRaw,
        CreateChatCompletionRequestRaw, PromptCacheOptionsRaw,
    };
    use llmy_types::other::WithOtherFields;
    use std::str::FromStr;

    fn request(messages: &[&str]) -> RawExtensibleChatCompletionRequest {
        RawExtensibleChatCompletionRequest::new(CreateChatCompletionRequestRaw {
            messages: messages
                .iter()
                .map(|text| {
                    WithOtherFields::new(ChatCompletionRequestMessageRaw::User(
                        ChatCompletionRequestUserMessageRaw::new_text(*text),
                    ))
                })
                .collect(),
            ..Default::default()
        })
    }

    fn keys() -> CacheKeys {
        CacheKeys::new(CacheKeyConfig::default())
    }

    fn chain_of(messages: &[&str]) -> Vec<PrefixHash> {
        hashes(&request(messages))
    }

    fn hashes(req: &RawExtensibleChatCompletionRequest) -> Vec<PrefixHash> {
        prompt_blocks(req).into_iter().map(|b| b.hash).collect()
    }

    /// A model whose cache is the transparent prefix kind, the common case.
    fn prefix_model() -> OpenAIModel {
        OpenAIModel::from_str("openai/gpt-4o").expect("built-in model")
    }

    fn breakpoint_model() -> OpenAIModel {
        OpenAIModel::from_str("openai/gpt-5.6-sol").expect("built-in model")
    }

    fn select(keys: &CacheKeys, messages: &[&str]) -> Option<CacheKeyClaim> {
        keys.select(&request(messages), &prefix_model())
    }

    /// A request that was sent *and* answered. These prompts are far below the
    /// cacheable minimum, so nothing is expected back and nothing is graded.
    fn sent(keys: &CacheKeys, messages: &[&str]) -> Option<String> {
        let claim = select(keys, messages)?;
        let key = claim.key().to_string();
        claim.confirm(0);
        Some(key)
    }

    /// What `PartialPrefix` claims for a request of `n` messages.
    fn claims_of(messages: &[&str]) -> Vec<usize> {
        let req = request(messages);
        cache_points(&req, CachePolicy::PartialPrefix, messages.len() + 1)
    }

    #[test]
    fn chain_shares_links_with_a_common_prefix() {
        let first = chain_of(&["a", "b"]);
        let second = chain_of(&["a", "b", "c"]);

        assert_eq!(first.len(), 3); // tools + 2 messages
        assert_eq!(second.len(), 4);
        assert_eq!(first[..], second[..3]);

        // Diverging at the second message breaks every link from there on.
        let other = chain_of(&["a", "z"]);
        assert_eq!(first[0], other[0]);
        assert_eq!(first[1], other[1]);
        assert_ne!(first[2], other[2]);
    }

    #[test]
    fn chain_ignores_fields_that_never_reach_the_model() {
        let baseline = chain_of(&["a"]);

        // A vendor extension on the message is not prompt content.
        let mut extended = request(&["a"]);
        extended.messages[0]
            .other
            .insert("vendor_hint".into(), serde_json::json!(1));
        assert_eq!(hashes(&extended), baseline);

        // Neither is the array form the promotion produces.
        let mut promoted = request(&["a"]);
        promoted.messages[0].toggle_cache_breakpoint(true);
        promoted.messages[0].toggle_cache_breakpoint(false);
        assert_eq!(hashes(&promoted), baseline);

        // Changing the prompt itself, of course, does move it.
        assert_ne!(chain_of(&["b"]), baseline);
    }

    #[test]
    fn chain_ignores_breakpoints_but_not_tools() {
        let mut marked = request(&["a"]);
        marked.messages[0].toggle_cache_breakpoint(true);
        // A breakpoint marks where to cut the cache, not what is in it — and it
        // promotes the content to an array, which must not matter either.
        assert_eq!(hashes(&marked), chain_of(&["a"]));

        let mut with_tools = request(&["a"]);
        with_tools.tools = Some(vec![]);
        assert_ne!(hashes(&with_tools)[0], chain_of(&["a"])[0]);
    }

    #[test]
    fn a_growing_conversation_keeps_its_key() {
        let keys = keys();
        let key = sent(&keys, &["a"]).unwrap();
        assert_eq!(sent(&keys, &["a", "b", "c"]).as_deref(), Some(&*key));
        assert_eq!(keys.key_count(), 1);
    }

    #[test]
    fn an_unrelated_conversation_gets_its_own_key() {
        let keys = keys();
        // Both have no tools, so they share link 0 — which must not be enough to
        // put them on the same machine.
        assert_eq!(chain_of(&["a"])[0], chain_of(&["b"])[0]);

        let key_a = sent(&keys, &["a"]).unwrap();
        let key_b = sent(&keys, &["b"]).unwrap();
        assert_ne!(key_a, key_b);
        assert_eq!(keys.key_count(), 2);
    }

    #[test]
    fn branches_share_the_key_of_their_common_prefix() {
        let keys = keys();
        let key = sent(&keys, &["a"]).unwrap();

        // Two sub-agents forking off the same trunk both want that machine.
        for branch in [["a", "x"], ["a", "y"]] {
            assert_eq!(sent(&keys, &branch).as_deref(), Some(&*key));
        }
        assert_eq!(keys.key_count(), 1);
    }

    #[test]
    fn a_saturated_key_spreads_to_a_new_one() {
        let keys = CacheKeys::new(CacheKeyConfig {
            max_rpm: 2,
            ..CacheKeyConfig::default()
        });
        let key = sent(&keys, &["a"]).unwrap();
        assert_eq!(sent(&keys, &["a"]).as_deref(), Some(&*key));
        // Third send inside the minute: the key is full, so we spread out.
        let overflow = sent(&keys, &["a"]).unwrap();
        assert_ne!(overflow, key);
        assert_eq!(keys.key_count(), 2);
        // The original prefix still belongs to the original key, so once its
        // budget frees up the lineage goes home rather than fragmenting.
        assert_eq!(keys.settled(), claims_of(&["a"]).len());
    }

    #[test]
    fn a_matched_prefix_reports_the_size_it_was_claimed_at() {
        let keys = keys();
        let model = prefix_model();
        // A prompt long enough to actually be worth measuring.
        let head = "lorem ipsum dolor sit amet ".repeat(64);
        let turn1 = request(&[&head]);

        // Turn 1 is a fresh key: nothing is expected back from it.
        let claim = keys.select(&turn1, &model).unwrap();
        assert_eq!(claim.expected_cached_tokens(), 0);
        claim.confirm(0);

        // Turn 2 matches turn 1's prefix, so the expectation is exactly what that
        // prefix was measured at when it was claimed — read back, not recomputed.
        let turn2 = request(&[&head, "and a follow-up"]);
        let claim = keys.select(&turn2, &model).unwrap();
        let expected = claim.expected_cached_tokens();
        assert!(expected > 0);
        assert_eq!(
            expected,
            model
                .config
                .count_tokens_lossy(&format!("null{}", completion_to_string(&turn1.messages[0])))
                as u64,
            "should be the tools block plus the first message"
        );
        claim.confirm(expected);

        // Turn 3 matches turn 2's longer prefix, so it expects strictly more.
        let turn3 = request(&[&head, "and a follow-up", "more"]);
        let claim = keys.select(&turn3, &model).unwrap();
        assert!(claim.expected_cached_tokens() > expected);
    }

    #[test]
    fn an_abandoned_claim_settles_nothing() {
        let keys = keys();
        // Selected, never answered: the prefix was never cached anywhere, so it
        // must not be left pointing at a key.
        drop(select(&keys, &["a"]).unwrap());
        assert_eq!(keys.settled(), 0);
        assert_eq!(keys.in_flight(), 0);

        // Which means the retry has to start a fresh key rather than inherit a
        // machine that has nothing on it.
        let key = sent(&keys, &["a"]).unwrap();
        assert_eq!(keys.key_count(), 2); // the burnt one plus this
        assert_eq!(sent(&keys, &["a"]).as_deref(), Some(&*key));
    }

    #[test]
    fn a_failing_sibling_does_not_strand_the_others() {
        let keys = keys();
        // Three requests fan out from one prefix, none answered yet.
        let first = select(&keys, &["a"]).unwrap();
        let key = first.key().to_string();
        let second = select(&keys, &["a", "x"]).unwrap();
        let third = select(&keys, &["a", "y"]).unwrap();
        assert_eq!(second.key(), key);
        assert_eq!(third.key(), key);
        assert_eq!(keys.key_count(), 1);

        // The first one fails. The claim is refcounted, so the other two keep it
        // alive and a latecomer still finds the shared machine.
        drop(first);
        assert_eq!(select(&keys, &["a", "z"]).unwrap().key(), key);

        second.confirm(0);
        third.confirm(0);
        assert!(keys.settled() > 0);
    }

    #[test]
    fn a_retry_costs_another_slot_of_the_key_s_budget() {
        // The provider counts total traffic per key, so a retried request is two
        // requests on that key, not one.
        let keys = CacheKeys::new(CacheKeyConfig {
            max_rpm: 2,
            ..CacheKeyConfig::default()
        });

        let claim = select(&keys, &["a"]).unwrap();
        let key = claim.key().to_string();
        claim.charge_resend(); // one attempt failed and was re-sent
        claim.confirm(0);

        // Two sends spent the whole minute's budget, so the next request finds
        // the prefix but no room and spreads to another machine.
        assert_ne!(sent(&keys, &["a"]).unwrap(), key);
        assert_eq!(keys.key_count(), 2);
    }

    #[test]
    fn disabled_registry_never_supplies_a_key() {
        let keys = CacheKeys::new(CacheKeyConfig {
            enabled: false,
            ..CacheKeyConfig::default()
        });
        assert!(select(&keys, &["a"]).is_none());
        assert_eq!(keys.key_count(), 0);
    }

    #[test]
    fn expired_keys_release_their_prefixes() {
        let keys = CacheKeys::new(CacheKeyConfig {
            ttl: Duration::ZERO,
            ..CacheKeyConfig::default()
        });
        let first = sent(&keys, &["a"]).unwrap();
        // A zero TTL expires the key before the next lookup even reads it.
        let second = sent(&keys, &["a"]).unwrap();
        assert_ne!(first, second);
        assert_eq!(keys.key_count(), 1);
        assert_eq!(keys.settled(), claims_of(&["a"]).len());
    }

    // --- cache points ------------------------------------------------------

    #[test]
    fn prefix_cached_models_claim_every_message_link() {
        assert_eq!(claims_of(&["a", "b"]), vec![1, 2]);
        // Never the tools-only link, or every toolless conversation would share
        // one key.
        assert!(claims_of(&[]).is_empty());
    }

    #[test]
    fn breakpoint_models_claim_breakpoints_plus_the_implicit_one() {
        let mut req = request(&["a", "b", "c"]);
        req.messages[0].toggle_cache_breakpoint(true);

        // Implicit is the API default: the declared breakpoint, plus the end of
        // the request, which the API breaks on by itself.
        assert_eq!(cache_points(&req, CachePolicy::Breakpoint, 4), vec![1, 3]);

        // Explicit suppresses that, leaving only what the caller declared.
        req.prompt_cache_options = Some(PromptCacheOptionsRaw::explicit());
        assert_eq!(cache_points(&req, CachePolicy::Breakpoint, 4), vec![1]);
    }

    #[test]
    fn a_breakpoint_on_the_last_message_is_not_claimed_twice() {
        let mut req = request(&["a", "b"]);
        req.messages[1].toggle_cache_breakpoint(true);
        assert_eq!(cache_points(&req, CachePolicy::Breakpoint, 3), vec![2]);
    }

    #[test]
    fn breakpoint_models_only_reuse_a_key_at_a_real_cache_point() {
        let keys = keys();

        // Turn 1: system-ish head marked, so the cache entry sits at link 1.
        let mut turn1 = request(&["head", "u1"]);
        turn1.messages[0].toggle_cache_breakpoint(true);
        turn1.prompt_cache_options = Some(PromptCacheOptionsRaw::explicit());
        let claim = keys.select(&turn1, &breakpoint_model()).unwrap();
        let key = claim.key().to_string();
        claim.confirm(0);

        // Turn 2 keeps the same head, so it hits that entry and the same key,
        // even though nothing else about the request matches.
        let mut turn2 = request(&["head", "u1", "a1", "u2"]);
        turn2.messages[0].toggle_cache_breakpoint(true);
        turn2.prompt_cache_options = Some(PromptCacheOptionsRaw::explicit());
        let claim = keys.select(&turn2, &breakpoint_model()).unwrap();
        assert_eq!(claim.key(), key);
        claim.confirm(0);
        assert_eq!(keys.key_count(), 1);

        // A different head shares no cache point, so it gets its own machine.
        let mut other = request(&["other-head", "u1"]);
        other.messages[0].toggle_cache_breakpoint(true);
        other.prompt_cache_options = Some(PromptCacheOptionsRaw::explicit());
        let claim = keys.select(&other, &breakpoint_model()).unwrap();
        assert_ne!(claim.key(), key);
    }
}
