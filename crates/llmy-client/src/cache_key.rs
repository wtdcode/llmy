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
use std::sync::{Arc, RwLock as StdRwLock};
use std::time::{Duration, Instant};

use crate::debug::completion_to_string;
use crate::model::CachePolicy;
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

/// A prefix claimed by requests that have not landed yet.
#[derive(Debug)]
struct InFlight {
    key: KeyId,
    /// How many in-flight requests are holding this claim. Refcounted so one
    /// failing sibling cannot strand another that is still going.
    holders: u32,
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
    /// Settled claims: this prefix really was sent under this key and answered.
    /// One prefix to one key — whoever settles it first owns it until that key
    /// expires, so a lineage keeps its original key as it grows.
    by_prefix: HashMap<PrefixHash, KeyId>,
    /// Claims held by requests still in flight. Looked up alongside `by_prefix`
    /// so concurrent siblings converge, but never promoted until confirmed.
    in_flight: HashMap<PrefixHash, InFlight>,
    next_id: KeyId,
    pid: u32,
}

impl CacheKeyRegistry {
    pub fn new(config: CacheKeyConfig) -> Self {
        Self {
            config,
            keys: HashMap::new(),
            by_prefix: HashMap::new(),
            in_flight: HashMap::new(),
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
        policy: CachePolicy,
    ) -> Option<(String, KeyId, Vec<PrefixHash>)> {
        if !self.config.enabled {
            return None;
        }
        let chain = prefix_chain(req);
        let claimable = cache_points(req, policy, chain.len());

        let now = Instant::now();
        self.expire(now);

        // Longest link first: the deepest shared prefix is the biggest cache
        // hit. A key that is out of budget is skipped, so we fall through to the
        // next-best (shorter) prefix rather than piling onto one machine.
        let max_rpm = self.config.max_rpm;
        let mut chosen = None;
        for hash in chain.iter().rev() {
            let Some(id) = self.claimant(hash) else {
                continue;
            };
            let Some(state) = self.keys.get_mut(&id) else {
                continue;
            };
            if state.has_budget(now, max_rpm) {
                chosen = Some(id);
                break;
            }
        }

        let id = match chosen {
            Some(id) => id,
            None => self.mint(now),
        };

        // Only real cache points, or a key would advertise cached content that
        // was never stored.
        let mut held = Vec::with_capacity(claimable.len());
        for hash in claimable.iter().filter_map(|link| chain.get(*link)) {
            if self.by_prefix.contains_key(hash) {
                continue; // already settled, by us or by an earlier lineage
            }
            match self.in_flight.entry(*hash) {
                Entry::Occupied(mut held_by) => {
                    // A sibling already has it in flight. Join it only if we are
                    // on the same key; otherwise leave their claim alone.
                    if held_by.get().key == id {
                        held_by.get_mut().holders += 1;
                        held.push(*hash);
                    }
                }
                Entry::Vacant(slot) => {
                    slot.insert(InFlight {
                        key: id,
                        holders: 1,
                    });
                    held.push(*hash);
                }
            }
        }

        let state = self.keys.get_mut(&id)?;
        state.record_send(now);
        Some((state.key.clone(), id, held))
    }

    /// The key currently claiming `hash`, settled or in flight.
    fn claimant(&self, hash: &PrefixHash) -> Option<KeyId> {
        self.by_prefix
            .get(hash)
            .copied()
            .or_else(|| self.in_flight.get(hash).map(|held| held.key))
    }

    /// Settle the claims of a request that landed: those prefixes really were
    /// sent under `id` and answered.
    fn confirm(&mut self, id: KeyId, held: &[PrefixHash]) {
        for hash in held {
            self.by_prefix.entry(*hash).or_insert(id);
            self.release(hash);
        }
    }

    /// Drop the claims of a request that never landed.
    fn abandon(&mut self, held: &[PrefixHash]) {
        for hash in held {
            self.release(hash);
        }
    }

    fn release(&mut self, hash: &PrefixHash) {
        let Entry::Occupied(mut held) = self.in_flight.entry(*hash) else {
            return;
        };
        held.get_mut().holders = held.get().holders.saturating_sub(1);
        if held.get().holders == 0 {
            held.remove();
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
            self.by_prefix.retain(|_, id| live.contains_key(id));
            self.in_flight
                .retain(|_, held| live.contains_key(&held.key));
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
        policy: CachePolicy,
    ) -> Option<CacheKeyClaim> {
        let (key, id, held) = self
            .0
            .write()
            .expect("cache keys poisoned")
            .select(req, policy)?;
        Some(CacheKeyClaim {
            keys: self.clone(),
            key,
            id,
            held,
        })
    }
}

#[cfg(test)]
impl CacheKeys {
    fn key_count(&self) -> usize {
        self.0.read().unwrap().keys.len()
    }
    fn settled(&self) -> usize {
        self.0.read().unwrap().by_prefix.len()
    }
    fn in_flight(&self) -> usize {
        self.0.read().unwrap().in_flight.len()
    }
}

/// A cache key, plus the prefix claims the request carrying it holds until it
/// lands.
///
/// [`Self::confirm`] settles them: those prefixes really were sent under this
/// key and answered, so later requests may route by them. Dropping unconfirmed
/// abandons them instead — a failed request, an early return or a panic never
/// leaves a prefix pointing at a key with nothing cached behind it.
#[derive(Debug)]
pub struct CacheKeyClaim {
    keys: CacheKeys,
    key: String,
    id: KeyId,
    held: Vec<PrefixHash>,
}

impl CacheKeyClaim {
    /// The `prompt_cache_key` to send.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Settle the claims — call once the request has been answered.
    pub fn confirm(mut self) {
        // Emptied first, so the `Drop` that follows has nothing to abandon.
        let held = std::mem::take(&mut self.held);
        if let Ok(mut registry) = self.keys.0.write() {
            registry.confirm(self.id, &held);
        }
    }
}

impl Drop for CacheKeyClaim {
    fn drop(&mut self) {
        if self.held.is_empty() {
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

/// Roll a request up into one hash per prompt block: the tool definitions first
/// — they head the rendered prompt, and changing them invalidates everything
/// after — then one per message. Link `i` covers the prefix `0..=i`, so two
/// requests share a link exactly when they share that stretch of prompt.
///
/// Blocks are hashed as the **prompt text** they render to, via
/// [`completion_to_string`], not as the JSON we happen to hold. What the
/// provider caches is tokens, so anything that never reaches the model — a
/// `prompt_cache_breakpoint` marker, the lone-text array that
/// [`crate::req::RawExtensibleChatRequestMessage::breakpoint`] promotes content
/// into, a vendor extension field — must not look like a different conversation.
/// The bias is deliberate: a false match only wastes a lookup on the wrong
/// machine, while a false miss throws away a real cache hit.
fn prefix_chain(req: &RawExtensibleChatCompletionRequest) -> Vec<PrefixHash> {
    let mut chain = Vec::with_capacity(req.messages.len() + 1);
    // One hasher fed block by block. `Hasher::finish` reads the running value
    // without resetting it, so link `i` comes out as the hash of the whole
    // prefix `0..=i` for a single pass over the prompt.
    let mut hasher = DefaultHasher::new();

    // Tool definitions go into the prompt as their schemas, so hash them so.
    let tools = serde_json::to_string(&req.tools).unwrap_or_default();
    hasher.write(tools.as_bytes());
    chain.push(hasher.finish());

    for message in req.messages.iter() {
        hasher.write(completion_to_string(message).as_bytes());
        chain.push(hasher.finish());
    }
    chain
}

/// Which links of [`prefix_chain`] the provider will really store a cache entry
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
        prefix_chain(&request(messages))
    }

    /// Select under the transparent-prefix policy, the common case.
    fn select(keys: &CacheKeys, messages: &[&str]) -> Option<CacheKeyClaim> {
        keys.select(&request(messages), CachePolicy::PartialPrefix)
    }

    /// A request that was sent *and* answered.
    fn sent(keys: &CacheKeys, messages: &[&str]) -> Option<String> {
        let claim = select(keys, messages)?;
        let key = claim.key().to_string();
        claim.confirm();
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
        assert_eq!(prefix_chain(&extended), baseline);

        // Neither is the array form the promotion produces.
        let mut promoted = request(&["a"]);
        promoted.messages[0].toggle_cache_breakpoint(true);
        promoted.messages[0].toggle_cache_breakpoint(false);
        assert_eq!(prefix_chain(&promoted), baseline);

        // Changing the prompt itself, of course, does move it.
        assert_ne!(chain_of(&["b"]), baseline);
    }

    #[test]
    fn chain_ignores_breakpoints_but_not_tools() {
        let mut marked = request(&["a"]);
        marked.messages[0].toggle_cache_breakpoint(true);
        // A breakpoint marks where to cut the cache, not what is in it — and it
        // promotes the content to an array, which must not matter either.
        assert_eq!(prefix_chain(&marked), chain_of(&["a"]));

        let mut with_tools = request(&["a"]);
        with_tools.tools = Some(vec![]);
        assert_ne!(prefix_chain(&with_tools)[0], chain_of(&["a"])[0]);
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

        second.confirm();
        third.confirm();
        assert!(keys.settled() > 0);
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
        let claim = keys.select(&turn1, CachePolicy::Breakpoint).unwrap();
        let key = claim.key().to_string();
        claim.confirm();

        // Turn 2 keeps the same head, so it hits that entry and the same key,
        // even though nothing else about the request matches.
        let mut turn2 = request(&["head", "u1", "a1", "u2"]);
        turn2.messages[0].toggle_cache_breakpoint(true);
        turn2.prompt_cache_options = Some(PromptCacheOptionsRaw::explicit());
        let claim = keys.select(&turn2, CachePolicy::Breakpoint).unwrap();
        assert_eq!(claim.key(), key);
        claim.confirm();
        assert_eq!(keys.key_count(), 1);

        // A different head shares no cache point, so it gets its own machine.
        let mut other = request(&["other-head", "u1"]);
        other.messages[0].toggle_cache_breakpoint(true);
        other.prompt_cache_options = Some(PromptCacheOptionsRaw::explicit());
        let claim = keys.select(&other, CachePolicy::Breakpoint).unwrap();
        assert_ne!(claim.key(), key);
    }
}
