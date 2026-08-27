use std::{
    collections::BTreeMap,
    ops::Deref,
    sync::{Arc, RwLock as StdRwLock},
    time::Duration,
};

use crate::model::OpenAIModel;
use async_openai::{
    Client,
    config::{AzureConfig, OpenAIConfig},
    error::OpenAIError,
};
use color_eyre::eyre::eyre;
use llmy_types::error::LLMYError;
use llmy_types::other::WithOtherFields;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use serde::de::DeserializeOwned;
use tokio_stream::StreamExt;

use crate::req::{
    ChatCompletionMessageToolCallRaw, ChatCompletionMessageToolCalls,
    ChatCompletionMessageToolCallsRaw, ChatCompletionRequestMessageRaw,
    ChatCompletionRequestSystemMessageRaw, ChatCompletionRequestUserMessageRaw,
    ChatCompletionStreamOptionsRaw, ChatCompletionToolChoiceOptionRaw, ChatCompletionTools,
    CreateChatCompletionRequestRaw, FunctionCallRaw, Role, ToolChoiceOptions,
};
use crate::resp::{
    ChatChoice, ChatChoiceRaw, ChatCompletionResponseMessageRaw, CompletionUsage,
    CreateChatCompletionResponseRaw, CreateChatCompletionStreamResponse, FinishReason,
};

use crate::cache_key::{CacheKeyClaim, CacheKeys};
use crate::debug::{self, DebugBackend, DebugRowContext, DebugUsage, PrefixBilling};
pub use crate::filters::{
    GoogleContentFilter, MarkdownTagFilter, MiMoContentFilter, NoFilter, OpenAIContentFilter,
};
pub use crate::req::{RawExtensibleChatCompletionRequest, RawExtensibleChatRequestMessage};
pub use crate::resp::{RawExtensibleChatChoice, RawExtensibleChatCompletionResponse};
use crate::{
    billing::{BillingTree, NodeId, ROOT, TokenBilling, TokenUsage},
    settings::{LLMSettings, Reasoning},
};

#[derive(Clone, Debug, Default)]
struct ToolCallAcc {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone)]
pub enum SupportedConfig {
    Azure {
        config: AzureConfig,
        deployment_id: String,
    },
    OpenAI(OpenAIConfig),
}

impl SupportedConfig {
    pub fn new_azure(endpoint: &str, key: &str, deployment: &str, api_version: &str) -> Self {
        let cfg = AzureConfig::new()
            .with_api_base(endpoint)
            .with_api_key(key)
            .with_deployment_id(deployment)
            .with_api_version(api_version);
        Self::Azure {
            config: cfg,
            deployment_id: deployment.to_string(),
        }
    }

    pub fn new(endpoint: &str, key: &str) -> Self {
        let cfg = OpenAIConfig::new()
            .with_api_base(endpoint)
            .with_api_key(key);
        Self::OpenAI(cfg)
    }

    pub fn endpoint_kind(&self) -> &'static str {
        match self {
            Self::Azure { .. } => "azure",
            Self::OpenAI(_) => "openai",
        }
    }

    /// The HTTP base URL the underlying client will hit. For Azure this is
    /// the resource endpoint; for OpenAI it's the configured `api_base`.
    pub fn endpoint_url(&self) -> &str {
        use async_openai::config::Config;
        match self {
            Self::Azure { config, .. } => config.api_base(),
            Self::OpenAI(cfg) => cfg.api_base(),
        }
    }

    pub fn azure_deployment(&self) -> Option<&str> {
        match self {
            Self::Azure { deployment_id, .. } => Some(deployment_id),
            Self::OpenAI(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum LLMClient {
    Azure(Client<AzureConfig>),
    OpenAI(Client<OpenAIConfig>),
}

/// SSE stream of [`CreateChatCompletionStreamResponse`] chunks until the
/// upstream server emits `[DONE]`. Mirrors `async_openai`'s
/// `ChatCompletionResponseStream` shape using our own response type so we
/// don't lose unknown fields per chunk.
pub type ChatCompletionResponseStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<CreateChatCompletionStreamResponse, OpenAIError>> + Send>,
>;

impl LLMClient {
    pub fn new(config: SupportedConfig) -> Self {
        match config {
            SupportedConfig::Azure { config, .. } => Self::Azure(Client::with_config(config)),
            SupportedConfig::OpenAI(cfg) => Self::OpenAI(Client::with_config(cfg)),
        }
    }

    pub async fn create_chat_extensible(
        &self,
        req: &RawExtensibleChatCompletionRequest,
    ) -> Result<RawExtensibleChatCompletionResponse, OpenAIError> {
        match self {
            Self::Azure(cl) => cl.chat().create_byot(req).await,
            Self::OpenAI(cl) => cl.chat().create_byot(req).await,
        }
    }

    pub async fn create_chat_stream_extensible(
        &self,
        req: &RawExtensibleChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        match self {
            Self::Azure(cl) => cl.chat().create_stream_byot(req).await,
            Self::OpenAI(cl) => cl.chat().create_stream_byot(req).await,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LLM {
    llm: Arc<LLMInner>,
}

impl LLM {
    /// Build an LLM with a pre-constructed debug backend (or `None` to disable).
    /// Use this directly when you already own a [`DebugBackend`]; otherwise see
    /// [`LLM::new_async`] which dispatches on `LLM_DEBUG`-style strings.
    pub fn new(
        config: SupportedConfig,
        model: OpenAIModel,
        cap: Decimal,
        settings: LLMSettings,
        debug_backend: Option<DebugBackend>,
    ) -> Self {
        let billing = Arc::new(StdRwLock::new(BillingTree::new(cap)));

        let endpoint = config.endpoint_url().to_string();
        let azure_deployment = config.azure_deployment().map(|s| s.to_string());

        let content_filter: Box<dyn OpenAIContentFilter> = if model.is_mimo() {
            Box::new(MiMoContentFilter::default())
        } else if model.is_google() {
            Box::new(GoogleContentFilter)
        } else {
            Box::new(NoFilter)
        };

        match debug_backend.as_ref() {
            Some(DebugBackend::Folder(folder)) => {
                tracing::info!(
                    "LLM debug enabled: folder backend at {}",
                    folder.folder().display()
                );
            }
            Some(DebugBackend::Sqlite3(db)) => {
                tracing::info!(
                    "LLM debug enabled: sqlite3 backend at {} (client_id={:?})",
                    db.path(),
                    db.client_id()
                );
            }
            None => {}
        }

        let cache_keys = CacheKeys::new(settings.cache_key_config());

        LLM {
            llm: Arc::new(LLMInner {
                node: ROOT,
                client: LLMClient::new(config),
                model,
                billing,
                debug_backend: debug_backend.map(Arc::new),
                endpoint,
                azure_deployment,
                default_settings: settings,
                content_filter: Arc::new(StdRwLock::new(content_filter)),
                cache_keys,
            }),
        }
    }

    /// Convenience constructor that dispatches `LLM_DEBUG`-style strings to a
    /// concrete [`DebugBackend`]. An empty / `None` `debug_target` disables
    /// debug entirely.
    pub async fn new_async(
        config: SupportedConfig,
        model: OpenAIModel,
        cap: Decimal,
        settings: LLMSettings,
        debug_prefix: Option<String>,
        debug_target: Option<String>,
    ) -> Result<Self, LLMYError> {
        let backend = match debug_target {
            Some(s) if !s.is_empty() => {
                Some(DebugBackend::from_env_string(&s, debug_prefix.as_deref()).await?)
            }
            _ => None,
        };
        Ok(Self::new(config, model, cap, settings, backend))
    }

    /// A handle pointing back at the root scope (the whole-LLM budget), sharing
    /// the same underlying client and billing tree.
    pub fn root(&self) -> LLM {
        LLM {
            llm: Arc::new(self.llm.rescope(ROOT)),
        }
    }

    /// Open a child billing scope under this handle's current node. The child's
    /// `cap` is clamped to the parent's remaining budget (and, if `None`, equals
    /// it). Returns a new `LLM` (sharing the underlying client and billing tree)
    /// that can be passed anywhere an `LLM` is expected; every call made through
    /// it bills the child scope, bubbling up to root.
    pub fn scope(&self, name: Option<String>, cap: Option<Decimal>) -> LLM {
        let child = {
            let mut tree = self.llm.billing.write().unwrap();
            tree.alloc_child(self.llm.node, name, cap)
        };
        LLM {
            llm: Arc::new(self.llm.rescope(child)),
        }
    }

    /// Snapshot of this handle's current scope: usage, spend, and the cap that
    /// bounds it (its own if set, otherwise the global cap).
    pub fn node_snapshot(&self) -> TokenBilling {
        self.llm
            .billing
            .read()
            .unwrap()
            .node_snapshot(self.llm.node)
    }

    /// This handle's current-scope accumulated token usage (shorthand for
    /// `node_snapshot().tokens`).
    pub fn usage(&self) -> TokenUsage {
        self.node_snapshot().tokens
    }

    /// This handle's current-scope accumulated spend (shorthand for
    /// `node_snapshot().current`).
    pub fn cost(&self) -> Decimal {
        self.node_snapshot().current
    }

    /// The global per-`debug_prefix` usage breakdown (orthogonal to scopes;
    /// `""` = calls made with no prefix).
    pub fn usage_by_prefix(&self) -> BTreeMap<String, TokenUsage> {
        self.llm.billing.read().unwrap().usage_by_prefix()
    }

    /// Usage for a single `debug_prefix` bucket.
    pub fn usage_for_prefix(&self, prefix: &str) -> TokenUsage {
        self.llm.billing.read().unwrap().usage_for_prefix(prefix)
    }
}

impl Deref for LLM {
    type Target = LLMInner;

    fn deref(&self) -> &Self::Target {
        &self.llm
    }
}

#[derive(Debug)]
pub struct LLMInner {
    /// This handle's current billing scope; defaults to [`ROOT`]. [`LLM::scope`]
    /// clones the handle with a different node while sharing everything else via
    /// `Arc`, so billing reads `self.node` with zero parameter threading.
    node: NodeId,
    pub client: LLMClient,
    pub model: OpenAIModel,
    pub billing: Arc<StdRwLock<BillingTree>>,
    pub debug_backend: Option<Arc<DebugBackend>>,
    pub endpoint: String,
    pub azure_deployment: Option<String>,
    pub default_settings: LLMSettings,
    content_filter: Arc<StdRwLock<Box<dyn OpenAIContentFilter>>>,
    /// Auto `prompt_cache_key` selection, shared with every scope cut from this
    /// client so a sub-agent continuing a conversation keeps its routing.
    cache_keys: CacheKeys,
}

impl Drop for LLMInner {
    fn drop(&mut self) {
        // Each `scope()` builds a distinct `Arc<LLMInner>`, so its refcount tracks
        // how many handles point at that scope. When the last one drops we land
        // here and auto-close the scope: prune its node (logging its final usage)
        // and re-parent its children to keep the chain to root intact. Root is
        // never pruned. Best-effort and panic-free, as required in `Drop`.
        if self.node != ROOT
            && let Ok(mut tree) = self.billing.write()
        {
            tree.remove(self.node);
        }
    }
}

impl LLMInner {
    /// Flat snapshot of the whole-LLM (root) billing totals.
    pub fn billing_snapshot(&self) -> TokenBilling {
        self.billing.read().unwrap().root_snapshot()
    }

    /// Clone this handle pointing at a different scope `node`. The mutable shared
    /// state (`billing` tree, `content_filter`) is shared via `Arc`; the rest is
    /// cheap config.
    fn rescope(&self, node: NodeId) -> LLMInner {
        LLMInner {
            node,
            client: self.client.clone(),
            model: self.model.clone(),
            billing: self.billing.clone(),
            debug_backend: self.debug_backend.clone(),
            endpoint: self.endpoint.clone(),
            azure_deployment: self.azure_deployment.clone(),
            default_settings: self.default_settings.clone(),
            content_filter: self.content_filter.clone(),
            cache_keys: self.cache_keys.clone(),
        }
    }

    /// Replace the content filter applied to every request and response. Defaults to
    /// `MiMoContentFilter` for mimo models, `GoogleContentFilter` for google models,
    /// `NoFilter` otherwise.
    pub fn set_content_filter(&self, filter: Box<dyn OpenAIContentFilter>) {
        *self
            .content_filter
            .write()
            .expect("content_filter poisoned") = filter;
    }

    fn apply_filter_input(&self, req: &mut RawExtensibleChatCompletionRequest) {
        let guard = self.content_filter.read().expect("content_filter poisoned");
        guard.filter_input(req);
    }

    /// Take the auto cache key for one logical request: the key whose prompt
    /// prefix best matches what this client has already sent, so consecutive
    /// turns of a conversation keep landing on the machine holding their cached
    /// prefix.
    ///
    /// The claim is handed (cloned) to every attempt of that request, settled by
    /// whichever one lands, and abandoned when the caller drops the last clone.
    /// A newly minted key carries `debug_prefix` in its name, so the keys in the
    /// logs and the debug DB say which workload opened them. A caller-supplied
    /// key is never second-guessed.
    fn auto_cache_key(
        &self,
        req: &RawExtensibleChatCompletionRequest,
        debug_prefix: Option<&str>,
    ) -> Option<CacheKeyClaim> {
        if req.prompt_cache_key.is_some() {
            return None;
        }
        self.cache_keys.select(req, &self.model, debug_prefix)
    }

    /// Cached prompt tokens the provider reported, or zero if it reported none.
    fn reported_cached_tokens(resp: &RawExtensibleChatCompletionResponse) -> u64 {
        resp.usage
            .as_ref()
            .and_then(|usage| usage.prompt_tokens_details.as_ref())
            .and_then(|details| details.cached_tokens)
            .unwrap_or_default() as u64
    }

    /// Whether this request's billing line should go to INFO rather than DEBUG:
    /// true once every `billing_log_tokens` of the client's traffic.
    ///
    /// `running` is the client's total after this request, `just_billed` this
    /// request's own slice of it, so the request owns the interval
    /// `(running - just_billed, running]` and reports if a multiple of the
    /// threshold falls inside it. That needs no counter of its own: the
    /// intervals of all requests tile the total exactly once, whatever order
    /// concurrent scopes finish in, so every boundary is reported exactly once.
    fn billing_line_due(&self, running: TokenUsage, just_billed: TokenUsage) -> bool {
        let every = self.default_settings.billing_log_tokens;
        if every == 0 {
            return true;
        }
        let total = running.total();
        total / every != total.saturating_sub(just_billed.total()) / every
    }

    /// Snapshot of the auto cache key policy in force for this client.
    pub fn cache_key_config(&self) -> crate::cache_key::CacheKeyConfig {
        self.cache_keys.config()
    }

    fn apply_filter_output(&self, resp: &mut RawExtensibleChatCompletionResponse) {
        let guard = self.content_filter.read().expect("content_filter poisoned");
        guard.filter_output(resp);
    }

    fn debug_row_context(&self, cache_key: Option<&str>) -> DebugRowContext {
        // The debug DB stores USD as a SQLite REAL column; SQLite has no native
        // decimal type, so collapse to f64 only at this boundary. The cap logged
        // is the global budget (matching the root-level `current_usage_usd`).
        let cap_usd = self
            .billing
            .read()
            .unwrap()
            .cap()
            .to_f64()
            .unwrap_or_default();
        DebugRowContext {
            model_name: self.model.model_id_str().to_string(),
            endpoint: self.endpoint.clone(),
            azure_deployment: self.azure_deployment.clone(),
            cache_key: cache_key.map(|s| s.to_string()),
            cap_usd,
        }
    }

    // we use t/s to estimate a timeout to avoid infinite repeating
    pub async fn prompt_once_with_retry(
        &self,
        sys_msg: &str,
        user_msg: &str,
        debug_prefix: Option<&str>,
        cache_key: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let sys = ChatCompletionRequestSystemMessageRaw::new_text(sys_msg);
        let user = ChatCompletionRequestUserMessageRaw::new_text(user_msg);
        self.prompt_messages_once(
            vec![
                ChatCompletionRequestMessageRaw::System(sys),
                ChatCompletionRequestMessageRaw::User(user),
            ],
            debug_prefix,
            cache_key,
            settings,
            None,
        )
        .await
    }

    /// Like [`Self::prompt_once_with_retry`], but deserializes the first-choice
    /// content into `T` (retried up to `settings.llm_retry` on failure) and returns
    /// the parsed value.
    pub async fn prompt_once_with_retry_typed<T: DeserializeOwned>(
        &self,
        sys_msg: &str,
        user_msg: &str,
        debug_prefix: Option<&str>,
        cache_key: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<T, LLMYError> {
        let sys = ChatCompletionRequestSystemMessageRaw::new_text(sys_msg);
        let user = ChatCompletionRequestUserMessageRaw::new_text(user_msg);
        self.prompt_messages_once_typed::<T>(
            vec![
                ChatCompletionRequestMessageRaw::System(sys),
                ChatCompletionRequestMessageRaw::User(user),
            ],
            debug_prefix,
            cache_key,
            settings,
            None,
        )
        .await
    }

    /// Prompt and deserialize the first choice's content into `T`. The typed path
    /// parses (and returns) `T` directly, so there is no second deserialize here;
    /// a malformed response is retried, and (when `auto_strip` is set) a markdown
    /// code fence is stripped before parsing.
    pub async fn prompt_json_with_retry<T: DeserializeOwned>(
        &self,
        sys_msg: &str,
        user_msg: &str,
        debug_prefix: Option<&str>,
        cache_key: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<T, LLMYError> {
        self.prompt_once_with_retry_typed::<T>(sys_msg, user_msg, debug_prefix, cache_key, settings)
            .await
    }

    pub async fn complete_once_with_retry(
        &self,
        req: &CreateChatCompletionRequestRaw,
        debug_prefix: Option<&str>,
        timeout: Option<Duration>,
        retry: Option<u64>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let req = RawExtensibleChatCompletionRequest::new(req.clone());
        self.complete_extensible_once_with_retry(&req, debug_prefix, timeout, retry)
            .await
    }

    pub async fn complete_extensible_once_with_retry(
        &self,
        req: &RawExtensibleChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout: Option<Duration>,
        retry: Option<u64>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let retry = retry.unwrap_or(u64::MAX);
        // One claim for the whole logical request; every attempt gets a clone,
        // and giving up drops the last one, which abandons it.
        let claim = self.auto_cache_key(req, debug_prefix);

        let mut last = None;
        for idx in 0..retry {
            // Selection paid for the first send; a retry is extra traffic on the
            // same key and costs another slot of its budget.
            if let (true, Some(claim)) = (idx > 0, claim.as_ref()) {
                claim.charge_resend();
            }
            match self
                .complete_extensible_attempt(req.clone(), claim.clone(), debug_prefix, timeout)
                .await
            {
                Ok(r) => return Ok(r),
                // A billing/cap error is deterministic — retrying can't recover it
                // (and would keep tripping the pre-flight check), so fail fast.
                Err(e @ LLMYError::Billing(_)) => return Err(e),
                Err(e) => {
                    tracing::warn!("Having an error {} during {} retry", e, idx);
                    last = Some(Err(e));
                }
            }
        }

        last.ok_or_else(|| eyre!("no response after {} retries?!", retry))?
    }

    /// Like [`Self::complete_extensible_once_with_retry`], but each attempt also
    /// deserializes the first-choice content into `T` (with markdown auto-strip);
    /// a malformed response is retried like any other error, and the parsed `T` is
    /// returned directly.
    pub async fn complete_extensible_once_with_retry_typed<T: DeserializeOwned>(
        &self,
        req: &RawExtensibleChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout: Option<Duration>,
        retry: Option<u64>,
    ) -> Result<T, LLMYError> {
        let retry = retry.unwrap_or(u64::MAX);
        // One claim for the whole logical request — see the untyped variant.
        let claim = self.auto_cache_key(req, debug_prefix);

        let mut last = None;
        for idx in 0..retry {
            if let (true, Some(claim)) = (idx > 0, claim.as_ref()) {
                claim.charge_resend();
            }
            match self
                .complete_extensible_typed_attempt::<T>(
                    req.clone(),
                    claim.clone(),
                    debug_prefix,
                    timeout,
                )
                .await
            {
                Ok(value) => return Ok(value),
                // A billing/cap error is deterministic — retrying can't recover it
                // (and would keep tripping the pre-flight check), so fail fast.
                Err(e @ LLMYError::Billing(_)) => return Err(e),
                Err(e) => {
                    tracing::warn!("Having an error {} during {} retry", e, idx);
                    last = Some(Err(e));
                }
            }
        }

        last.ok_or_else(|| eyre!("no response after {} retries?!", retry))?
    }

    pub async fn complete(
        &self,
        req: CreateChatCompletionRequestRaw,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let req = RawExtensibleChatCompletionRequest::new(req);
        self.complete_extensible(req, debug_prefix, timeout_overwrite)
            .await
    }

    /// Deserialize the first-choice content into `T`. On a JSON parse error — and
    /// only when `auto_strip` is enabled — retry the parse after stripping a
    /// markdown code fence (the common ` ```json {…} ``` ` wrapper) from the
    /// content. Absent content is an error.
    fn parse_first_choice<T: DeserializeOwned>(
        &self,
        resp: &RawExtensibleChatCompletionResponse,
    ) -> Result<T, LLMYError> {
        let content = resp
            .choices
            .first()
            .and_then(|c| c.inner.message.content.as_deref())
            .ok_or_else(|| {
                eyre!("completion has no content to deserialize into the requested type")
            })?;
        match serde_json::from_str::<T>(content) {
            Ok(value) => Ok(value),
            Err(err) => {
                if self.default_settings.auto_strip
                    && let Some(stripped) = crate::filters::strip_markdown_fence(content)
                {
                    return Ok(serde_json::from_str::<T>(&stripped)?);
                }
                Err(err.into())
            }
        }
    }

    /// A single completion attempt that deserializes the first-choice content into
    /// `T` and returns it (so callers don't deserialize a second time). A malformed
    /// response surfaces as an error so the caller's retry loop re-issues it; the
    /// retry itself lives in [`Self::complete_extensible_once_with_retry_typed`].
    pub async fn complete_extensible_typed<T: DeserializeOwned>(
        &self,
        req: RawExtensibleChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<T, LLMYError> {
        let claim = self.auto_cache_key(&req, debug_prefix);
        self.complete_extensible_typed_attempt::<T>(req, claim, debug_prefix, timeout_overwrite)
            .await
    }

    /// One attempt of a typed logical request; see
    /// [`Self::complete_extensible_attempt`] for how `given_claim` travels.
    async fn complete_extensible_typed_attempt<T: DeserializeOwned>(
        &self,
        req: RawExtensibleChatCompletionRequest,
        given_claim: Option<CacheKeyClaim>,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<T, LLMYError> {
        let resp = self
            .complete_extensible_attempt(req, given_claim, debug_prefix, timeout_overwrite)
            .await?;
        self.parse_first_choice::<T>(&resp)
    }

    pub async fn complete_extensible(
        &self,
        req: RawExtensibleChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        // A one-shot call is a logical request of exactly one attempt, so the
        // claim is taken and dropped here — abandoned if the call does not land.
        let claim = self.auto_cache_key(&req, debug_prefix);
        self.complete_extensible_attempt(req, claim, debug_prefix, timeout_overwrite)
            .await
    }

    /// One attempt of a logical request, carrying that request's cache key
    /// claim — see [`Self::auto_cache_key`]. The attempt only ever uses the
    /// claim it is given; it never takes one of its own, so retrying cannot cost
    /// a second claim or a second slot of the key's request budget.
    async fn complete_extensible_attempt(
        &self,
        mut req: RawExtensibleChatCompletionRequest,
        given_claim: Option<CacheKeyClaim>,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        // Fail fast: if we're already over budget, don't issue the request.
        self.billing.read().unwrap().check_cap(self.node)?;

        if let Some(claim) = given_claim.as_ref() {
            tracing::debug!("auto prompt cache key {}", claim.key());
            req.prompt_cache_key = Some(claim.key().to_string());
        }
        self.apply_filter_input(&mut req);

        // Keep the raw prefix (None => "") for the per-prefix billing dimension,
        // before it gets defaulted to "llm" for the debug backend below.
        let billing_prefix = debug_prefix;

        let use_stream = self.default_settings.llm_stream;
        let debug_prefix = if let Some(debug_prefix) = debug_prefix {
            debug_prefix.to_string()
        } else {
            "llm".to_string()
        };

        let dbg_handle = if let Some(backend) = self.debug_backend.as_ref() {
            let cache_key = req.prompt_cache_key.clone();
            let ctx = self.debug_row_context(cache_key.as_deref());
            backend.start(&debug_prefix, ctx, &req).await
        } else {
            None
        };

        let estimated_tokens = {
            let text = debug::extract_raw_text_with_other(&req);
            tracing::trace!("Text is {:?}", text);
            self.model.config.count_tokens(&text)
        };

        tracing::trace!(
            "Sending completion request: {:?}",
            &serde_json::to_string(&req)
        );

        let now = std::time::SystemTime::now();
        let llm_fut = async {
            if use_stream {
                self.complete_streaming(&mut req).await
            } else {
                self.client
                    .create_chat_extensible(&req)
                    .await
                    .map_err(|e| e.into())
            }
        };

        let timeout = timeout_overwrite.unwrap_or_else(|| self.default_settings.timeout());
        let resp = if timeout == Duration::MAX {
            llm_fut.await
        } else {
            tokio::time::timeout(timeout, llm_fut)
                .await
                .unwrap_or_else(|_| {
                    Err(LLMYError::Other(eyre!(
                        "LLM request timed out after {:?}",
                        timeout
                    )))
                })
        };

        let mut resp = match resp {
            Ok(resp) => resp,
            Err(e) => {
                if let (Some(backend), Some(handle)) =
                    (self.debug_backend.as_ref(), dbg_handle.as_ref())
                {
                    backend.record_error(handle, &e).await;
                }
                // This attempt's clone drops here without settling; the caller
                // still holds the claim for a retry, or drops it to abandon.
                return Err(e);
            }
        };
        // The provider answered, so it really did cache this prompt's prefixes.
        // Settle before the billing record below, which can still bail out.
        if let Some(claim) = given_claim.as_ref() {
            claim.confirm(Self::reported_cached_tokens(&resp));
        }
        self.apply_filter_output(&mut resp);
        if let (Some(backend), Some(handle)) = (self.debug_backend.as_ref(), dbg_handle.as_ref()) {
            backend.record_response(handle, &req, &resp).await;
        }

        let mut billed: Option<(TokenBilling, TokenUsage)> = None;
        let output_tokens = if let Some(usage) = &resp.usage {
            let prompt_details = usage.prompt_tokens_details.as_ref();
            let cached = prompt_details
                .and_then(|v| v.cached_tokens)
                .unwrap_or_default();
            // GPT-5.6+ only; absent (=> 0) on every earlier model.
            let cache_write = prompt_details
                .and_then(|v| v.cache_write_tokens)
                .unwrap_or_default();
            // Saturating: a provider reporting inconsistent counts must not panic.
            let input_without_cached = usage.prompt_tokens.saturating_sub(cached);
            let reasoning = usage
                .completion_tokens_details
                .as_ref()
                .and_then(|v| v.reasoning_tokens)
                .unwrap_or_default() as u64;
            let output_without_reasoning =
                (usage.completion_tokens as u64).saturating_sub(reasoning);

            let delta = TokenUsage {
                input_tokens: usage.prompt_tokens as u64,
                cache_tokens: cached as u64,
                cache_write_tokens: cache_write as u64,
                output_tokens: usage.completion_tokens as u64,
                reasoning_tokens: reasoning,
            };

            // Tight critical section (no `.await` while the std guard is held):
            // bill the scope tree + prefix bucket, then snapshot root and the
            // per-prefix breakdown for the debug backend.
            let debug_snapshot = {
                let mut tree = self.billing.write().unwrap();
                tree.record(self.node, billing_prefix, &self.model, delta)?;
                // Snapshot under the same lock as the record, so this request's
                // slice of the running total is exactly `total - billed` even
                // when other scopes are recording concurrently.
                let snapshot = tree.root_snapshot();
                billed = Some((snapshot, delta));
                self.debug_backend
                    .as_ref()
                    .map(|_| (snapshot, tree.usage_by_prefix()))
            };

            if let (Some(backend), Some(handle), Some((snapshot, prefix_usage))) = (
                self.debug_backend.as_ref(),
                dbg_handle.as_ref(),
                debug_snapshot.as_ref(),
            ) {
                let usage_for_debug = DebugUsage {
                    input_without_cached_tokens: input_without_cached as u64,
                    cached_tokens: cached as u64,
                    cache_write_tokens: cache_write as u64,
                    output_without_reasoning_tokens: output_without_reasoning,
                    reasoning_tokens: reasoning,
                };
                backend
                    .record_billing(handle, snapshot, &usage_for_debug)
                    .await;

                // Persist the cumulative per-debug_prefix breakdown (cost computed
                // at this LLM's single model). Not tied to the per-request handle.
                let prefix_rows: Vec<PrefixBilling> = prefix_usage
                    .iter()
                    .map(|(prefix, tokens)| PrefixBilling {
                        prefix: prefix.clone(),
                        tokens: *tokens,
                        cost_usd: tokens.cost(&self.model).to_f64().unwrap_or_default(),
                    })
                    .collect();
                backend.record_prefix_billing(&prefix_rows).await;
            }
            if let Some(est) = estimated_tokens {
                let actual = usage.prompt_tokens as f64;
                let diff = (est as f64 - actual).abs();
                let pct = if actual > 0.0 {
                    diff / actual * 100.0
                } else {
                    0.0
                };
                // A small drift is just tokenizer noise; a large one means this
                // model's tokenizer config is wrong and is worth surfacing.
                if pct > self.default_settings.token_estimate_pct {
                    tracing::info!(
                        "Token estimate: {} estimated vs {} actual (diff {:.1}%)",
                        est,
                        usage.prompt_tokens,
                        pct
                    );
                } else {
                    tracing::debug!(
                        "Token estimate: {} estimated vs {} actual (diff {:.1}%)",
                        est,
                        usage.prompt_tokens,
                        pct
                    );
                }
            }

            usage.completion_tokens
        } else {
            tracing::warn!("No usage from {:?}?!", &resp);
            0
        };

        let delta = std::time::SystemTime::now()
            .duration_since(now)
            .map(|v| v.as_secs_f64())
            .unwrap_or_default();
        let (billing_snapshot, promote) = match billed {
            Some((snapshot, just_billed)) => (
                snapshot,
                self.billing_line_due(snapshot.tokens, just_billed),
            ),
            // No usage came back, so nothing was billed: still report the
            // running total, but don't spend an INFO slot on it.
            None => (self.billing.read().unwrap().root_snapshot(), false),
        };
        let speed = if delta.is_normal() && delta.is_sign_positive() {
            output_tokens as f64 / delta
        } else {
            0.0f64
        };
        let client = self.debug_backend.as_ref().and_then(|v| v.client_id());
        if promote {
            tracing::info!(
                "Usage: {}, Speed: {:.2} tok/s (client={:?})",
                billing_snapshot,
                speed,
                client
            );
        } else {
            tracing::debug!(
                "Usage: {}, Speed: {:.2} tok/s (client={:?})",
                billing_snapshot,
                speed,
                client
            );
        }
        Ok(resp)
    }

    #[allow(deprecated)]
    async fn complete_streaming(
        &self,
        req: &mut RawExtensibleChatCompletionRequest,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        req.stream = Some(true);

        if req.stream_options.is_none() {
            req.stream_options = Some(WithOtherFields::new(ChatCompletionStreamOptionsRaw {
                include_usage: Some(true),
                include_obfuscation: None,
            }));
        }

        let mut stream = self.client.create_chat_stream_extensible(&*req).await?;

        let mut id: Option<String> = None;
        let mut created: Option<u32> = None;
        let mut model: Option<String> = None;
        let mut service_tier = None;
        let mut system_fingerprint = None;
        let mut usage: Option<CompletionUsage> = None;

        let mut contents: Vec<String> = Vec::new();
        let mut finish_reasons: Vec<Option<FinishReason>> = Vec::new();
        let mut tool_calls: Vec<Vec<ToolCallAcc>> = Vec::new();

        while let Some(item) = stream.next().await {
            let chunk: CreateChatCompletionStreamResponse = item?;
            // Take ownership of the inner raw chunk so we can move fields out of `WithOtherFields`.
            let chunk = chunk.inner;
            if id.is_none() {
                id = Some(chunk.id.clone());
            }
            created = Some(chunk.created);
            model = Some(chunk.model.clone());
            service_tier = chunk.service_tier.clone();
            #[allow(deprecated)]
            {
                system_fingerprint = chunk.system_fingerprint.clone();
            }
            if let Some(u) = chunk.usage.clone() {
                usage = Some(u);
            }

            for ch in chunk.choices.into_iter() {
                let ch = ch.inner;
                let idx = ch.index as usize;
                if contents.len() <= idx {
                    contents.resize_with(idx + 1, String::new);
                    finish_reasons.resize_with(idx + 1, || None);
                    tool_calls.resize_with(idx + 1, Vec::new);
                }
                let delta = ch.delta.inner;
                if let Some(content) = delta.content {
                    contents[idx].push_str(&content);
                }
                if let Some(tcs) = delta.tool_calls {
                    for tc in tcs.into_iter() {
                        let tc = tc.inner;
                        let tc_idx = tc.index as usize;
                        if tool_calls[idx].len() <= tc_idx {
                            tool_calls[idx].resize_with(tc_idx + 1, ToolCallAcc::default);
                        }
                        let acc = &mut tool_calls[idx][tc_idx];
                        if let Some(id) = tc.id {
                            acc.id = id;
                        }
                        if let Some(func) = tc.function {
                            let func = func.inner;
                            if let Some(name) = func.name {
                                acc.name = name;
                            }
                            if let Some(args) = func.arguments {
                                acc.arguments.push_str(&args);
                            }
                        }
                    }
                }
                if ch.finish_reason.is_some() {
                    finish_reasons[idx] = ch.finish_reason;
                }
            }
        }

        let mut choices: Vec<ChatChoice> = Vec::new();
        for (idx, content) in contents.into_iter().enumerate() {
            let finish_reason = finish_reasons.get(idx).cloned().unwrap_or(None);
            let built_tool_calls: Vec<ChatCompletionMessageToolCalls> = tool_calls
                .get(idx)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|t| !t.name.trim().is_empty() || !t.arguments.trim().is_empty())
                .map(|t| {
                    let raw = ChatCompletionMessageToolCallRaw {
                        id: if t.id.trim().is_empty() {
                            format!("toolcall-{}", idx)
                        } else {
                            t.id
                        },
                        function: WithOtherFields::new(FunctionCallRaw {
                            name: t.name,
                            arguments: t.arguments,
                        }),
                    };
                    WithOtherFields::new(ChatCompletionMessageToolCallsRaw::Function(
                        WithOtherFields::new(raw),
                    ))
                })
                .collect();
            let tool_calls_opt = if built_tool_calls.is_empty() {
                None
            } else {
                Some(built_tool_calls)
            };
            #[allow(deprecated)]
            let message = WithOtherFields::new(ChatCompletionResponseMessageRaw {
                content: if content.is_empty() {
                    None
                } else {
                    Some(content)
                },
                refusal: None,
                tool_calls: tool_calls_opt,
                annotations: None,
                role: Role::Assistant,
                function_call: None,
                audio: None,
            });
            choices.push(WithOtherFields::new(ChatChoiceRaw {
                index: idx as u32,
                message,
                finish_reason,
                logprobs: None,
            }));
        }
        if choices.is_empty() {
            #[allow(deprecated)]
            let message = WithOtherFields::new(ChatCompletionResponseMessageRaw {
                content: Some(String::new()),
                refusal: None,
                tool_calls: None,
                annotations: None,
                role: Role::Assistant,
                function_call: None,
                audio: None,
            });
            choices.push(WithOtherFields::new(ChatChoiceRaw {
                index: 0,
                message,
                finish_reason: None,
                logprobs: None,
            }));
        }

        #[allow(deprecated)]
        let resp_raw = CreateChatCompletionResponseRaw {
            id: id.unwrap_or_else(|| "stream".to_string()),
            choices,
            created: created.unwrap_or(0),
            model: model.unwrap_or_else(|| self.model.api_model_name().to_string()),
            service_tier,
            system_fingerprint,
            object: "chat.completion".to_string(),
            usage,
        };
        Ok(RawExtensibleChatCompletionResponse::new(resp_raw))
    }

    /// Build an extensible chat request with this model's settings, tools, and provider quirks
    /// applied. Per-message extras (e.g. mimo `reasoning_content`) are preserved from the
    /// supplied wrapped messages.
    pub fn build_chat_request(
        &self,
        messages: Vec<RawExtensibleChatRequestMessage>,
        cache_key: Option<&str>,
        settings: &LLMSettings,
        tools: Option<Vec<ChatCompletionTools>>,
    ) -> Result<RawExtensibleChatCompletionRequest, LLMYError> {
        let mut raw = CreateChatCompletionRequestRaw::default();
        raw.model = self.model.api_model_name().to_string();

        if let Some(tools) = tools {
            raw.tools = Some(tools);
        }

        if let Some(tc) = settings.llm_tool_choice.clone() {
            raw.tool_choice = Some(tc.0);
        } else if self.model.is_mimo() {
            // This ensures mimo generates tool calls correctly, only god knows why.
            raw.tool_choice = Some(WithOtherFields::new(
                ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Auto),
            ));
        }

        if let Some(effort) = settings.reasoning_effort.clone()
            && !self.model.is_mimo()
        {
            raw.reasoning_effort = Some(effort.0);
        }

        if let Some(cache_key) = cache_key {
            raw.prompt_cache_key = Some(cache_key.to_string());
        }
        if let Some(temperature) = settings.llm_temperature {
            raw.temperature = Some(temperature);
        }
        if let Some(presence_penalty) = settings.llm_presence_penalty {
            raw.presence_penalty = Some(presence_penalty);
        }
        if let Some(max_completion_tokens) = settings.llm_max_completion_tokens {
            raw.max_completion_tokens = Some(max_completion_tokens);
        }
        if let Some(top_p) = settings.top_p {
            raw.top_p = Some(top_p);
        }

        // Use the extensible message wrappers directly so per-message extras survive.
        raw.messages = messages.iter().map(|m| m.0.clone()).collect();
        let mut req = RawExtensibleChatCompletionRequest::new(raw);
        apply_provider_request_extensions(&self.model, settings, &mut req);
        Ok(req)
    }

    pub async fn prompt_messages_once(
        &self,
        messages: Vec<ChatCompletionRequestMessageRaw>,
        debug_prefix: Option<&str>,
        cache_key: Option<&str>,
        settings: Option<LLMSettings>,
        tools: Option<Vec<ChatCompletionTools>>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let settings = settings.unwrap_or_else(|| self.default_settings.clone());
        let timeout = settings.timeout();
        let retry = settings.llm_retry;
        let wrapped = messages
            .into_iter()
            .map(RawExtensibleChatRequestMessage::new)
            .collect();
        let req = self.build_chat_request(wrapped, cache_key, &settings, tools)?;
        self.complete_extensible_once_with_retry(&req, debug_prefix, Some(timeout), Some(retry))
            .await
    }

    /// Like [`Self::prompt_messages_once`], but deserializes the first-choice
    /// content into `T` (retrying on failure up to `settings.llm_retry`) and
    /// returns the parsed value.
    pub async fn prompt_messages_once_typed<T: DeserializeOwned>(
        &self,
        messages: Vec<ChatCompletionRequestMessageRaw>,
        debug_prefix: Option<&str>,
        cache_key: Option<&str>,
        settings: Option<LLMSettings>,
        tools: Option<Vec<ChatCompletionTools>>,
    ) -> Result<T, LLMYError> {
        let settings = settings.unwrap_or_else(|| self.default_settings.clone());
        let timeout = settings.timeout();
        let retry = settings.llm_retry;
        let wrapped = messages
            .into_iter()
            .map(RawExtensibleChatRequestMessage::new)
            .collect();
        let req: RawExtensibleChatCompletionRequest =
            self.build_chat_request(wrapped, cache_key, &settings, tools)?;
        self.complete_extensible_once_with_retry_typed::<T>(
            &req,
            debug_prefix,
            Some(timeout),
            Some(retry),
        )
        .await
    }

    pub async fn prompt_once(
        &self,
        sys_msg: &str,
        user_msg: &str,
        debug_prefix: Option<&str>,
        cache_key: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let sys = ChatCompletionRequestSystemMessageRaw::new_text(sys_msg);
        let user = ChatCompletionRequestUserMessageRaw::new_text(user_msg);
        self.prompt_messages_once(
            vec![
                ChatCompletionRequestMessageRaw::System(sys),
                ChatCompletionRequestMessageRaw::User(user),
            ],
            debug_prefix,
            cache_key,
            settings,
            None,
        )
        .await
    }
}

fn apply_provider_request_extensions(
    model: &OpenAIModel,
    settings: &LLMSettings,
    req: &mut RawExtensibleChatCompletionRequest,
) {
    if model.is_mimo()
        && settings
            .reasoning_effort
            .as_ref()
            .is_some_and(Reasoning::is_none)
    {
        req.extra_mut().insert(
            "thinking".to_string(),
            serde_json::json!({
                "type": "disabled"
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::req::ReasoningEffort;
    use crate::settings::{LLMToolChoice, Reasoning};
    use std::str::FromStr;

    fn test_settings(reasoning_effort: Option<Reasoning>) -> LLMSettings {
        LLMSettings {
            llm_temperature: None,
            llm_presence_penalty: None,
            llm_prompt_timeout: 0,
            llm_retry: 1,
            llm_max_completion_tokens: None,
            llm_tool_choice: None::<LLMToolChoice>,
            llm_stream: false,
            top_p: None,
            reasoning_effort,
            auto_strip: true,
            auto_cache_key: true,
            cache_key_ttl: crate::cache_key::DEFAULT_TTL_SECS,
            cache_key_rpm: crate::cache_key::DEFAULT_MAX_RPM,
            billing_log_tokens: 100_000,
            token_estimate_pct: 10.0,
        }
    }

    fn user_request(content: &str) -> RawExtensibleChatCompletionRequest {
        let user = ChatCompletionRequestMessageRaw::User(
            ChatCompletionRequestUserMessageRaw::new_text(content),
        );
        let mut raw = CreateChatCompletionRequestRaw::default();
        raw.model = "mimo-v2.5-pro".to_string();
        raw.messages = vec![WithOtherFields::new(user)];
        RawExtensibleChatCompletionRequest::new(raw)
    }

    fn test_llm_with(settings: LLMSettings) -> LLM {
        // No network is touched by construction or by scope/usage/parse helpers.
        let config = SupportedConfig::new("http://localhost:0", "test-key");
        let model = OpenAIModel::from_str("captest,1000000,1000000").unwrap();
        LLM::new(config, model, rust_decimal::dec!(100), settings, None)
    }

    fn test_llm() -> LLM {
        test_llm_with(test_settings(None))
    }

    #[test]
    fn parse_first_choice_parses_plain_json() {
        let llm = test_llm();
        let resp = crate::filters::build_resp(Some(r#"{"a": 2}"#), FinishReason::Stop);
        let v: serde_json::Value = llm.parse_first_choice(&resp).unwrap();
        assert_eq!(v["a"], 2);
    }

    #[test]
    fn parse_first_choice_auto_strips_markdown_fence() {
        let llm = test_llm(); // auto_strip = true
        let resp = crate::filters::build_resp(Some("```json\n{\"a\": 1}\n```"), FinishReason::Stop);
        let v: serde_json::Value = llm.parse_first_choice(&resp).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn parse_first_choice_without_auto_strip_errors_on_fenced_json() {
        let mut settings = test_settings(None);
        settings.auto_strip = false;
        let llm = test_llm_with(settings);
        let resp = crate::filters::build_resp(Some("```json\n{\"a\": 1}\n```"), FinishReason::Stop);
        let parsed: Result<serde_json::Value, _> = llm.parse_first_choice(&resp);
        assert!(matches!(parsed, Err(LLMYError::STDJSON(_))));
    }

    #[test]
    fn parse_first_choice_errors_when_no_content() {
        let llm = test_llm();
        let resp = crate::filters::build_resp(None, FinishReason::Stop);
        let parsed: Result<serde_json::Value, _> = llm.parse_first_choice(&resp);
        assert!(parsed.is_err());
    }

    #[tokio::test]
    async fn over_cap_returns_billing_through_retry_without_network() {
        // A negative cap means "already over budget": the pre-flight check_cap in
        // complete_extensible rejects before any network call, and the retry loop
        // must surface that Billing error (breaking instead of looping/hanging).
        let config = SupportedConfig::new("http://localhost:0", "k");
        let model = OpenAIModel::from_str("captest,1000000,1000000").unwrap();
        let llm = LLM::new(
            config,
            model,
            rust_decimal::dec!(-1),
            test_settings(None),
            None,
        );
        let req = user_request("hi");
        let err = llm
            .complete_extensible_once_with_retry(&req, None, Some(Duration::from_secs(5)), Some(8))
            .await
            .unwrap_err();
        assert!(matches!(err, LLMYError::Billing(_)), "got {err:?}");
    }

    #[test]
    fn dropping_a_scope_auto_prunes_its_node() {
        let llm = test_llm();
        let count = || llm.llm.billing.read().unwrap().node_count();
        let before = count(); // just root
        {
            let sub = llm.scope(Some("sub".into()), None);
            assert_eq!(count(), before + 1);
            // A clone of the same scope shares the node; only the last drop prunes.
            let clone = sub.clone();
            assert_eq!(count(), before + 1);
            drop(clone);
            assert_eq!(count(), before + 1);
        }
        // Last handle to the scope is gone => node pruned.
        assert_eq!(count(), before);
    }

    #[test]
    fn extensible_chat_completion_request_flattens_extra_fields() {
        let mut request = user_request("hello");

        request.extra_mut().insert(
            "thinking".to_string(),
            serde_json::json!({
                "type": "disabled"
            }),
        );

        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["model"], "mimo-v2.5-pro");
        assert_eq!(value["thinking"]["type"], "disabled");
    }

    #[test]
    fn extensible_chat_completion_request_flattens_message_extra_fields() {
        let mut request = user_request("hello");
        // mutate the wrapped message via Deref→WithOtherFields
        request.messages[0]
            .other
            .insert("reasoning_content".to_string(), "I need the tool.".into());

        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["messages"][0]["content"], "hello");
        assert_eq!(
            value["messages"][0]["reasoning_content"],
            "I need the tool."
        );
    }

    #[test]
    fn extensible_chat_completion_request_serializes_inner_changes() {
        let mut request = user_request("hello");
        request.stream = Some(true);

        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["stream"], true);
    }

    #[test]
    fn extensible_chat_completion_response_preserves_extra_fields() {
        let response: RawExtensibleChatCompletionResponse =
            serde_json::from_value(serde_json::json!({
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "call the tool",
                            "reasoning_content": "I need the tool."
                        },
                        "finish_reason": "tool_calls",
                        "provider_choice_id": "choice-123"
                    }
                ],
                "created": 1,
                "model": "mimo-v2.5-pro",
                "object": "chat.completion",
                "provider_trace_id": "trace-123"
            }))
            .unwrap();

        assert_eq!(response.id, "chatcmpl-test");
        assert_eq!(response.extra()["provider_trace_id"], "trace-123");
        assert_eq!(
            response.choices[0].other["provider_choice_id"],
            "choice-123"
        );
        assert_eq!(
            response.choices[0]
                .inner
                .message
                .other
                .get("reasoning_content")
                .and_then(|v| v.as_str()),
            Some("I need the tool.")
        );

        let value = serde_json::to_value(&response).unwrap();
        assert_eq!(value["model"], "mimo-v2.5-pro");
        assert_eq!(value["provider_trace_id"], "trace-123");
        assert_eq!(
            value["choices"][0]["message"]["reasoning_content"],
            "I need the tool."
        );
    }

    #[test]
    #[allow(deprecated)]
    fn extensible_chat_completion_response_serializes_base_mut_changes() {
        let raw = CreateChatCompletionResponseRaw {
            id: "chatcmpl-test".to_string(),
            choices: Vec::new(),
            created: 1,
            model: "old-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion".to_string(),
            usage: None,
        };
        let mut response = RawExtensibleChatCompletionResponse::new(raw);
        response.model = "new-model".to_string();
        response.extra_mut().insert(
            "provider_trace_id".to_string(),
            serde_json::json!("trace-123"),
        );

        let value = serde_json::to_value(&response).unwrap();
        assert_eq!(value["model"], "new-model");
        assert_eq!(value["provider_trace_id"], "trace-123");
    }

    #[test]
    fn mimo_reasoning_none_adds_thinking_disabled() {
        let mut request = user_request("hello");
        let model = OpenAIModel::from_str("mimo-v2.5-pro").unwrap();

        apply_provider_request_extensions(
            &model,
            &test_settings(Some(Reasoning(ReasoningEffort::None))),
            &mut request,
        );

        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["thinking"]["type"], "disabled");
    }

    /// Locks the Gemini `thought_signature` round-trip. Gemini's OpenAI-compat
    /// endpoint attaches `thought_signature` to each tool_call in the response,
    /// and rejects subsequent assistant messages whose tool_calls drop it. This
    /// test asserts that the field survives deserialize → clone-into-assistant
    /// → reserialize without any provider-specific code.
    #[test]
    fn gemini_thought_signature_round_trips_through_tool_calls() {
        use crate::req::{
            ChatCompletionMessageToolCallsRaw, ChatCompletionRequestAssistantMessageRaw,
        };

        let response: RawExtensibleChatCompletionResponse =
            serde_json::from_value(serde_json::json!({
                "id": "chatcmpl-gemini",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                            "thought_signature": "ZGVlcC10aG91Z2h0LXNpZw=="
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "created": 1,
                "model": "google/gemini-3.1-pro-preview",
                "object": "chat.completion"
            }))
            .unwrap();

        // The tool_call JSON has the same flat shape as the variant payload
        // (id/type/function/thought_signature live on the same level), so the
        // unknown field is captured inside the variant's WithOtherFields, not
        // on the outer enum wrapper.
        let tcs = response.choices[0]
            .inner
            .message
            .inner
            .tool_calls
            .as_ref()
            .expect("tool_calls present");
        assert_eq!(tcs.len(), 1);
        let tc = &tcs[0];
        match &tc.inner {
            ChatCompletionMessageToolCallsRaw::Function(f) => {
                assert_eq!(f.function.name, "lookup");
                assert_eq!(
                    f.other.get("thought_signature").and_then(|v| v.as_str()),
                    Some("ZGVlcC10aG91Z2h0LXNpZw==")
                );
            }
            _ => panic!("expected function tool call"),
        }

        // Cloning the tool_calls into an assistant request message and
        // re-serializing must preserve `thought_signature` verbatim.
        #[allow(deprecated)]
        let assistant = WithOtherFields::new(ChatCompletionRequestAssistantMessageRaw {
            content: None,
            refusal: None,
            name: None,
            audio: None,
            tool_calls: response.choices[0].inner.message.inner.tool_calls.clone(),
            function_call: None,
        });
        let echoed = serde_json::to_value(&assistant).unwrap();
        assert_eq!(
            echoed["tool_calls"][0]["thought_signature"],
            "ZGVlcC10aG91Z2h0LXNpZw=="
        );
        assert_eq!(echoed["tool_calls"][0]["id"], "call_1");
        assert_eq!(echoed["tool_calls"][0]["function"]["name"], "lookup");
    }

    // --- auto prompt cache key ---------------------------------------------

    /// Resolve a key for `req` and settle it, i.e. a request that was answered.
    fn landed(llm: &LLM, req: &mut RawExtensibleChatCompletionRequest) {
        if let Some(claim) = llm.auto_cache_key(req, None) {
            req.prompt_cache_key = Some(claim.key().to_string());
            claim.confirm(0);
        }
    }

    fn extended(base: &str, more: &str) -> RawExtensibleChatCompletionRequest {
        let mut req = user_request(base);
        req.messages
            .push(WithOtherFields::new(ChatCompletionRequestMessageRaw::User(
                ChatCompletionRequestUserMessageRaw::new_text(more),
            )));
        req
    }

    #[test]
    fn auto_cache_key_is_stable_across_turns_and_scopes() {
        let llm = test_llm();

        let mut turn1 = user_request("hello");
        landed(&llm, &mut turn1);
        let key = turn1.prompt_cache_key.clone().expect("auto key");

        // A scope shares the client's registry, so a sub-agent continuing the
        // same conversation stays on the same machine.
        let scoped = llm.scope(Some("sub".into()), None);
        let mut turn2 = extended("hello", "and more");
        landed(&scoped, &mut turn2);
        assert_eq!(turn2.prompt_cache_key.as_deref(), Some(&*key));

        // An unrelated conversation must not be pinned to it.
        let mut other = user_request("something else entirely");
        landed(&llm, &mut other);
        assert_ne!(other.prompt_cache_key.as_deref(), Some(&*key));
    }

    #[test]
    fn a_request_that_never_landed_leaves_no_claim_behind() {
        let llm = test_llm();

        // Claim taken, then dropped unconfirmed — the request failed, so nothing
        // was cached and the prefix must not point anywhere.
        let failed = user_request("hello");
        let stale = llm
            .auto_cache_key(&failed, None)
            .expect("auto key")
            .key()
            .to_string();

        let mut retry = user_request("hello");
        landed(&llm, &mut retry);
        assert_ne!(retry.prompt_cache_key.as_deref(), Some(&*stale));
    }

    #[test]
    fn concurrent_siblings_share_an_in_flight_claim() {
        let llm = test_llm();

        // Two sub-agents fan out from the same prefix at once. Neither has an
        // answer yet, but the second must still land on the first one's machine.
        let first = user_request("shared");
        let first_claim = llm.auto_cache_key(&first, None).expect("auto key");
        let second = extended("shared", "branch");
        let second_claim = llm.auto_cache_key(&second, None).expect("auto key");
        assert_eq!(second_claim.key(), first_claim.key());

        // The first one failing must not strand the second, which is still going.
        let shared_key = first_claim.key().to_string();
        drop(first_claim);
        let mut third = extended("shared", "other branch");
        landed(&llm, &mut third);
        assert_eq!(third.prompt_cache_key.as_deref(), Some(&*shared_key));
        second_claim.confirm(0);
    }

    #[test]
    fn retries_share_one_claim() {
        // One request per key per minute, so taking a second claim would be
        // forced onto a fresh key and show up immediately.
        let llm = test_llm_with(LLMSettings {
            cache_key_rpm: 1,
            ..test_settings(None)
        });

        // What a retry loop owns: one claim, cloned into every attempt.
        let req = user_request("hello");
        let claim = llm.auto_cache_key(&req, None).expect("auto key");
        for _ in 0..3 {
            let attempt = claim.clone();
            assert_eq!(attempt.key(), claim.key());
        }
        claim.confirm(0);

        // A caller-supplied key short-circuits selection entirely.
        let mut given = user_request("hello");
        given.prompt_cache_key = Some("mine".into());
        assert!(llm.auto_cache_key(&given, None).is_none());
    }

    #[test]
    fn the_billing_line_is_promoted_once_per_interval_crossed() {
        let llm = test_llm_with(LLMSettings {
            billing_log_tokens: 1000,
            ..test_settings(None)
        });
        // Each request owns the interval (running - billed, running] and reports
        // if a multiple of 1000 lands inside it.
        assert!(!llm.billing_line_due(tokens(500), tokens(500))); // (0, 500]
        assert!(llm.billing_line_due(tokens(1200), tokens(700))); // (500, 1200] crosses 1000
        assert!(!llm.billing_line_due(tokens(1900), tokens(700))); // (1200, 1900]
        assert!(llm.billing_line_due(tokens(2100), tokens(200))); // (1900, 2100] crosses 2000

        // One request spanning several intervals still reports just once.
        assert!(llm.billing_line_due(tokens(9000), tokens(6900)));
        // Nothing billed, nothing to report.
        assert!(!llm.billing_line_due(tokens(9000), tokens(0)));
    }

    #[test]
    fn interleaved_requests_cover_every_boundary_exactly_once() {
        let llm = test_llm_with(LLMSettings {
            billing_log_tokens: 1000,
            ..test_settings(None)
        });
        // Concurrent scopes land in whatever order; their intervals still tile
        // the running total, which is what makes the counter unnecessary.
        let mut running = 0;
        let mut promoted = Vec::new();
        for billed in [300u64, 1500, 900, 50, 2250] {
            let before = running;
            running += billed;
            if llm.billing_line_due(tokens(running), tokens(billed)) {
                promoted.push((before, running));
            }
        }
        assert_eq!(running, 5000);

        // Every boundary in the run is inside exactly one promoted interval —
        // none missed, none reported twice.
        for boundary in (1000..=running).step_by(1000) {
            let covering = promoted
                .iter()
                .filter(|(start, end)| *start < boundary && boundary <= *end)
                .count();
            assert_eq!(covering, 1, "boundary {boundary}");
        }
        // And nothing was promoted that had no boundary to report. Fewer lines
        // than boundaries because the last request swallowed three at once.
        assert_eq!(promoted.len(), 3);
    }

    /// A usage carrying `n` tokens, for the billing-line interval maths.
    fn tokens(n: u64) -> TokenUsage {
        TokenUsage {
            input_tokens: n,
            ..Default::default()
        }
    }

    /// Deterministic xorshift, so the sweep below is reproducible without a
    /// `rand` dev-dependency.
    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn every_boundary_is_covered_exactly_once_for_any_split() {
        let mut seed = 0x5eed_1234_9abc_def0u64;
        // Sweep interval sizes against request sizes that are far smaller,
        // comparable and far larger than the interval, plus zero-token requests.
        for every in [1u64, 7, 1000, 100_000] {
            let llm = test_llm_with(LLMSettings {
                billing_log_tokens: every,
                ..test_settings(None)
            });
            for spread in [1u64, every.max(1), every.saturating_mul(4)] {
                let mut running = 0u64;
                let mut promoted: Vec<(u64, u64)> = Vec::new();

                for _ in 0..200 {
                    // Zero-token requests are real: a provider can answer
                    // without reporting usage of its own.
                    let billed = xorshift(&mut seed) % (spread + 1);
                    let before = running;
                    running += billed;
                    if llm.billing_line_due(tokens(running), tokens(billed)) {
                        promoted.push((before, running));
                    }
                }

                // Each promoted line must own at least one boundary...
                for (start, end) in &promoted {
                    assert!(
                        (start / every) != (end / every),
                        "every={every} spread={spread}: promoted [{start}, {end}) with no boundary"
                    );
                }
                // ...and every boundary in the run must be owned by exactly one.
                let mut boundary = every;
                while boundary <= running {
                    let owners = promoted
                        .iter()
                        .filter(|(start, end)| *start < boundary && boundary <= *end)
                        .count();
                    assert_eq!(
                        owners, 1,
                        "every={every} spread={spread}: boundary {boundary} owned by {owners}"
                    );
                    boundary += every;
                }
            }
        }
    }

    #[test]
    fn a_minted_key_carries_the_debug_prefix() {
        let llm = test_llm();

        let planner = llm
            .auto_cache_key(&user_request("plan this"), Some("planner"))
            .expect("auto key");
        assert!(
            planner.key().contains("planner"),
            "{} should name the workload that opened it",
            planner.key()
        );
        planner.confirm(0);

        // A blank prefix must not leave a dangling separator in the key.
        let blank = llm
            .auto_cache_key(&user_request("something else"), Some("  "))
            .expect("auto key");
        assert!(!blank.key().contains("--"), "{}", blank.key());
        blank.confirm(0);
    }

    #[test]
    fn a_caller_supplied_cache_key_is_never_overridden() {
        let llm = test_llm();
        let mut req = user_request("hello");
        req.prompt_cache_key = Some("mine".to_string());
        landed(&llm, &mut req);
        assert_eq!(req.prompt_cache_key.as_deref(), Some("mine"));
    }

    #[test]
    fn auto_cache_key_can_be_switched_off() {
        let llm = test_llm_with(LLMSettings {
            auto_cache_key: false,
            ..test_settings(None)
        });
        let mut req = user_request("hello");
        landed(&llm, &mut req);
        assert!(req.prompt_cache_key.is_none());
        assert!(!llm.cache_key_config().enabled);
    }
}
