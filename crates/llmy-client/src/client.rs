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

use crate::debug::{self, DebugBackend, DebugRowContext, DebugUsage};
pub use crate::filter::{GoogleContentFilter, MiMoContentFilter, NoFilter, OpenAIContentFilter};
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

    /// Like [`Self::prompt_once_with_retry`], but a response whose first-choice
    /// content fails to deserialize into `T` is treated as a transient failure
    /// and retried (up to `settings.llm_retry`).
    pub async fn prompt_once_with_retry_typed<T: DeserializeOwned>(
        &self,
        sys_msg: &str,
        user_msg: &str,
        debug_prefix: Option<&str>,
        cache_key: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
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

    // Note only consider the `content` of the first choice
    pub async fn prompt_json_with_retry<T: DeserializeOwned>(
        &self,
        sys_msg: &str,
        user_msg: &str,
        debug_prefix: Option<&str>,
        cache_key: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<Option<T>, LLMYError> {
        // The typed prompt already retries until the content parses as `T`, so by
        // the time we get here the first-choice content is known to deserialize
        // (or be absent); this final parse is just to hand back the value.
        let msg = self
            .prompt_once_with_retry_typed::<T>(sys_msg, user_msg, debug_prefix, cache_key, settings)
            .await?;
        Ok(msg
            .choices
            .first()
            .and_then(|v| v.inner.message.content.as_ref())
            .map(|k| serde_json::from_str::<T>(k))
            .transpose()?)
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

        let mut last = None;
        for idx in 0..retry {
            match self
                .complete_extensible(req.clone(), debug_prefix, timeout)
                .await
            {
                Ok(r) => return Ok(r),
                // A billing/cap error is deterministic — retrying can't recover it
                // (and would keep tripping the pre-flight check), so fail fast.
                Err(e @ LLMYError::Billing { .. }) => return Err(e),
                Err(e) => {
                    tracing::warn!("Having an error {} during {} retry", e, idx);
                    last = Some(Err(e));
                }
            }
        }

        last.ok_or_else(|| eyre!("no response after {} retries?!", retry))?
    }

    /// Like [`Self::complete_extensible_once_with_retry`], but each attempt also
    /// requires the first-choice content to deserialize into `T`; a malformed
    /// response is retried like any other error (reusing the same `retry` budget).
    pub async fn complete_extensible_once_with_retry_typed<T: DeserializeOwned>(
        &self,
        req: &RawExtensibleChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout: Option<Duration>,
        retry: Option<u64>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let retry = retry.unwrap_or(u64::MAX);

        let mut last = None;
        for idx in 0..retry {
            match self
                .complete_extensible_typed::<T>(req.clone(), debug_prefix, timeout)
                .await
            {
                Ok(r) => return Ok(r),
                // A billing/cap error is deterministic — retrying can't recover it
                // (and would keep tripping the pre-flight check), so fail fast.
                Err(e @ LLMYError::Billing { .. }) => return Err(e),
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

    /// A single completion attempt that additionally requires the first-choice
    /// content to deserialize into `T`, surfacing a malformed response as an
    /// error so the caller's retry loop re-issues it. The retry itself lives in
    /// [`Self::complete_extensible_once_with_retry_typed`].
    pub async fn complete_extensible_typed<T: DeserializeOwned>(
        &self,
        req: RawExtensibleChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let resp = self
            .complete_extensible(req, debug_prefix, timeout_overwrite)
            .await?;
        // Gate success on the first-choice content deserializing into `T`: a
        // malformed response becomes an error so the caller's retry loop re-issues
        // it. Absent content is not an error (mirrors `prompt_json_with_retry`).
        if let Some(content) = resp
            .choices
            .first()
            .and_then(|c| c.inner.message.content.as_ref())
        {
            serde_json::from_str::<T>(content)?;
        }
        Ok(resp)
    }

    pub async fn complete_extensible(
        &self,
        mut req: RawExtensibleChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        // Fail fast: if we're already over budget, don't issue the request.
        self.billing.read().unwrap().check_cap(self.node)?;

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
                    let err = format!("{:?}", e);
                    backend.record_error(handle, &err).await;
                }
                return Err(e);
            }
        };
        self.apply_filter_output(&mut resp);
        if let (Some(backend), Some(handle)) = (self.debug_backend.as_ref(), dbg_handle.as_ref()) {
            backend.record_response(handle, &req, &resp).await;
        }

        let output_tokens = if let Some(usage) = &resp.usage {
            let cached = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|v| v.cached_tokens)
                .unwrap_or_default();
            let input_without_cached = usage.prompt_tokens - cached;
            let reasoning = usage
                .completion_tokens_details
                .as_ref()
                .and_then(|v| v.reasoning_tokens)
                .unwrap_or_default() as u64;
            let output_without_reasoning = usage.completion_tokens as u64 - reasoning;

            let delta = TokenUsage {
                input_tokens: usage.prompt_tokens as u64,
                cache_tokens: cached as u64,
                output_tokens: usage.completion_tokens as u64,
                reasoning_tokens: reasoning,
            };

            // Tight critical section (no `.await` while the std guard is held):
            // bill the scope tree + prefix bucket, then snapshot root for debug.
            let root_snapshot = {
                let mut tree = self.billing.write().unwrap();
                tree.record(self.node, billing_prefix, &self.model, delta)?;
                self.debug_backend.as_ref().map(|_| tree.root_snapshot())
            };

            if let (Some(backend), Some(handle), Some(snapshot)) = (
                self.debug_backend.as_ref(),
                dbg_handle.as_ref(),
                root_snapshot.as_ref(),
            ) {
                let usage_for_debug = DebugUsage {
                    input_without_cached_tokens: input_without_cached as u64,
                    cached_tokens: cached as u64,
                    output_without_reasoning_tokens: output_without_reasoning,
                    reasoning_tokens: reasoning,
                };
                backend
                    .record_billing(handle, snapshot, &usage_for_debug)
                    .await;
            }
            if let Some(est) = estimated_tokens {
                let actual = usage.prompt_tokens as f64;
                let diff = (est as f64 - actual).abs();
                let pct = if actual > 0.0 {
                    diff / actual * 100.0
                } else {
                    0.0
                };
                tracing::info!(
                    "Token estimate: {} estimated vs {} actual (diff {:.1}%)",
                    est,
                    usage.prompt_tokens,
                    pct
                );
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
        let billing_snapshot = self.billing.read().unwrap().root_snapshot();
        tracing::info!(
            "Usage: {}, Speed: {:.2} tok/s (client={:?})",
            billing_snapshot,
            if delta.is_normal() && delta.is_sign_positive() {
                output_tokens as f64 / delta
            } else {
                0.0f64
            },
            self.debug_backend.as_ref().and_then(|v| v.client_id())
        );
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

    /// Like [`Self::prompt_messages_once`], but a response whose first-choice
    /// content fails to deserialize into `T` is retried (up to `settings.llm_retry`).
    pub async fn prompt_messages_once_typed<T: DeserializeOwned>(
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

    fn test_llm() -> LLM {
        // No network is touched by construction or by scope/usage/drop.
        let config = SupportedConfig::new("http://localhost:0", "test-key");
        let model = OpenAIModel::from_str("captest,1000000,1000000").unwrap();
        LLM::new(
            config,
            model,
            rust_decimal::dec!(100),
            test_settings(None),
            None,
        )
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
        assert!(matches!(err, LLMYError::Billing { .. }), "got {err:?}");
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
}
