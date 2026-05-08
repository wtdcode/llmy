use std::{
    ops::Deref,
    sync::{Arc, RwLock as StdRwLock},
    time::Duration,
};

use crate::model::OpenAIModel;
use async_openai::{
    Client,
    config::{AzureConfig, OpenAIConfig},
    error::OpenAIError,
    types::chat::{
        ChatChoice, ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionResponseMessage,
        ChatCompletionResponseStream, ChatCompletionStreamOptions, ChatCompletionToolChoiceOption,
        ChatCompletionTools, CompletionUsage, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
        CreateChatCompletionStreamResponse, FinishReason, FunctionCall, Role, ToolChoiceOptions,
    },
};
use color_eyre::eyre::eyre;
use llmy_types::error::LLMYError;
use serde::de::DeserializeOwned;
use tokio::sync::RwLock;
use tokio_stream::StreamExt;

use crate::debug::{self, DebugBackend, DebugRowContext, DebugUsage};
pub use crate::filter::{GoogleContentFilter, MiMoContentFilter, NoFilter, OpenAIContentFilter};
pub use crate::req::{RawExtensibleChatCompletionRequest, RawExtensibleChatRequestMessage};
pub use crate::resp::{RawExtensibleChatChoice, RawExtensibleChatCompletionResponse};
use crate::{
    billing::ModelBilling,
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

impl LLMClient {
    pub fn new(config: SupportedConfig) -> Self {
        match config {
            SupportedConfig::Azure { config, .. } => Self::Azure(Client::with_config(config)),
            SupportedConfig::OpenAI(cfg) => Self::OpenAI(Client::with_config(cfg)),
        }
    }

    pub async fn create_chat(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        match self {
            Self::Azure(cl) => cl.chat().create(req).await,
            Self::OpenAI(cl) => cl.chat().create(req).await,
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

    pub async fn create_chat_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        match self {
            Self::Azure(cl) => cl.chat().create_stream(req).await,
            Self::OpenAI(cl) => cl.chat().create_stream(req).await,
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
        cap: f64,
        settings: LLMSettings,
        debug_backend: Option<DebugBackend>,
    ) -> Self {
        let billing = RwLock::new(ModelBilling::new(cap));

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
                client: LLMClient::new(config),
                model,
                billing,
                debug_backend,
                endpoint,
                azure_deployment,
                cap,
                default_settings: settings,
                content_filter: StdRwLock::new(content_filter),
            }),
        }
    }

    /// Convenience constructor that dispatches `LLM_DEBUG`-style strings to a
    /// concrete [`DebugBackend`]. An empty / `None` `debug_target` disables
    /// debug entirely.
    pub async fn new_async(
        config: SupportedConfig,
        model: OpenAIModel,
        cap: f64,
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
}

impl Deref for LLM {
    type Target = LLMInner;

    fn deref(&self) -> &Self::Target {
        &self.llm
    }
}

#[derive(Debug)]
pub struct LLMInner {
    pub client: LLMClient,
    pub model: OpenAIModel,
    pub billing: RwLock<ModelBilling>,
    pub debug_backend: Option<DebugBackend>,
    pub endpoint: String,
    pub azure_deployment: Option<String>,
    pub cap: f64,
    pub default_settings: LLMSettings,
    content_filter: StdRwLock<Box<dyn OpenAIContentFilter>>,
}

impl LLMInner {
    pub async fn billing_snapshot(&self) -> ModelBilling {
        self.billing.read().await.clone()
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

    fn debug_row_context<'a>(&'a self, cache_key: Option<&'a str>) -> DebugRowContext<'a> {
        DebugRowContext {
            model_name: self.model.model_id(),
            endpoint: &self.endpoint,
            azure_deployment: self.azure_deployment.as_deref(),
            cache_key,
            cap_usd: self.cap,
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
        let sys = ChatCompletionRequestSystemMessageArgs::default()
            .content(sys_msg)
            .build()?;

        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(user_msg)
            .build()?;
        self.prompt_messages_once(
            vec![sys.into(), user.into()],
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
        let msg = self
            .prompt_once_with_retry(sys_msg, user_msg, debug_prefix, cache_key, settings)
            .await?;
        Ok(msg
            .choices
            .first()
            .map(|v| {
                v.inner
                    .message
                    .content
                    .as_ref()
                    .map(|k| serde_json::from_str::<T>(&k))
            })
            .flatten()
            .transpose()?)
    }

    pub async fn complete_once_with_retry(
        &self,
        req: &CreateChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout: Option<Duration>,
        retry: Option<u64>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let req = RawExtensibleChatCompletionRequest::from(req.clone());
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
        req: CreateChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let req = RawExtensibleChatCompletionRequest::from(req);
        self.complete_extensible(req, debug_prefix, timeout_overwrite)
            .await
    }

    pub async fn complete_extensible(
        &self,
        mut req: RawExtensibleChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        self.apply_filter_input(&mut req);

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
            let mut billing = self.billing.write().await;

            let cached = usage
                .prompt_tokens_details
                .as_ref()
                .and_then(|v| v.cached_tokens)
                .unwrap_or_default();
            let input_without_cached = usage.prompt_tokens - cached;
            billing.input_tokens(&self.model, input_without_cached as _, cached as _)?;
            let reasoning = usage
                .completion_tokens_details
                .as_ref()
                .and_then(|v| v.reasoning_tokens)
                .unwrap_or_default() as u64;
            let output_without_reasoning = usage.completion_tokens as u64 - reasoning;

            billing.output_tokens(&self.model, output_without_reasoning, reasoning)?;

            if let (Some(backend), Some(handle)) =
                (self.debug_backend.as_ref(), dbg_handle.as_ref())
            {
                let billing_clone = billing.clone();
                drop(billing);
                let usage_for_debug = DebugUsage {
                    input_without_cached_tokens: input_without_cached as u64,
                    cached_tokens: cached as u64,
                    output_without_reasoning_tokens: output_without_reasoning,
                    reasoning_tokens: reasoning,
                };
                backend
                    .record_billing(handle, &billing_clone, &usage_for_debug)
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
        tracing::info!(
            "Usage: {}, Speed: {:.2} tok/s",
            &self.billing.read().await,
            if delta.is_normal() && delta.is_sign_positive() {
                output_tokens as f64 / delta
            } else {
                0.0f64
            }
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
            req.stream_options = Some(ChatCompletionStreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            });
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
            if id.is_none() {
                id = Some(chunk.id.clone());
            }
            created = Some(chunk.created);
            model = Some(chunk.model.clone());
            service_tier = chunk.service_tier.clone();
            system_fingerprint = chunk.system_fingerprint.clone();
            if let Some(u) = chunk.usage.clone() {
                usage = Some(u);
            }

            for ch in chunk.choices.into_iter() {
                let idx = ch.index as usize;
                if contents.len() <= idx {
                    contents.resize_with(idx + 1, String::new);
                    finish_reasons.resize_with(idx + 1, || None);
                    tool_calls.resize_with(idx + 1, Vec::new);
                }
                if let Some(delta) = ch.delta.content {
                    contents[idx].push_str(&delta);
                }
                if let Some(tcs) = ch.delta.tool_calls {
                    for tc in tcs.into_iter() {
                        let tc_idx = tc.index as usize;
                        if tool_calls[idx].len() <= tc_idx {
                            tool_calls[idx].resize_with(tc_idx + 1, ToolCallAcc::default);
                        }
                        let acc = &mut tool_calls[idx][tc_idx];
                        if let Some(id) = tc.id {
                            acc.id = id;
                        }
                        if let Some(func) = tc.function {
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

        let mut choices = Vec::new();
        for (idx, content) in contents.into_iter().enumerate() {
            let finish_reason = finish_reasons.get(idx).cloned().unwrap_or(None);
            let built_tool_calls = tool_calls
                .get(idx)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|t| !t.name.trim().is_empty() || !t.arguments.trim().is_empty())
                .map(|t| {
                    ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                        id: if t.id.trim().is_empty() {
                            format!("toolcall-{}", idx)
                        } else {
                            t.id
                        },
                        function: FunctionCall {
                            name: t.name,
                            arguments: t.arguments,
                        },
                    })
                })
                .collect::<Vec<_>>();
            let tool_calls_opt = if built_tool_calls.is_empty() {
                None
            } else {
                Some(built_tool_calls)
            };
            choices.push(ChatChoice {
                index: idx as u32,
                message: ChatCompletionResponseMessage {
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
                },
                finish_reason,
                logprobs: None,
            });
        }
        if choices.is_empty() {
            choices.push(ChatChoice {
                index: 0,
                message: ChatCompletionResponseMessage {
                    content: Some(String::new()),
                    refusal: None,
                    tool_calls: None,
                    annotations: None,
                    role: Role::Assistant,
                    function_call: None,
                    audio: None,
                },
                finish_reason: None,
                logprobs: None,
            });
        }

        Ok(RawExtensibleChatCompletionResponse::new(
            CreateChatCompletionResponse {
                id: id.unwrap_or_else(|| "stream".to_string()),
                choices,
                created: created.unwrap_or(0),
                model: model.unwrap_or_else(|| self.model.to_string()),
                service_tier,
                system_fingerprint,
                object: "chat.completion".to_string(),
                usage,
            },
        ))
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
        let mut req = CreateChatCompletionRequestArgs::default();

        if let Some(tools) = tools {
            req.tools(tools);
        }

        if let Some(tc) = settings.llm_tool_choice.clone() {
            req.tool_choice(tc);
        } else if self.model.is_mimo() {
            // This ensures mimo to generate tool calls correctly, only god knows why
            req.tool_choice(ChatCompletionToolChoiceOption::Mode(
                ToolChoiceOptions::Auto,
            ));
        }

        if let Some(effort) = settings.reasoning_effort.clone()
            && !self.model.is_mimo()
        {
            req.reasoning_effort(effort.0);
        }

        if let Some(cache_key) = cache_key {
            req.prompt_cache_key(cache_key.to_string());
        }
        if let Some(temperature) = settings.llm_temperature {
            req.temperature(temperature);
        }

        if let Some(presence_penalty) = settings.llm_presence_penalty {
            req.presence_penalty(presence_penalty);
        }

        if let Some(max_completion_tokens) = settings.llm_max_completion_tokens {
            req.max_completion_tokens(max_completion_tokens);
        }

        if let Some(top_p) = settings.top_p {
            req.top_p(top_p);
        }

        let raw_messages: Vec<ChatCompletionRequestMessage> =
            messages.iter().map(|m| m.inner.clone()).collect();
        let raw_req = req
            .messages(raw_messages)
            .model(self.model.to_string())
            .build()?;
        let mut req = RawExtensibleChatCompletionRequest::from(raw_req);
        req.messages = messages;
        apply_provider_request_extensions(&self.model, settings, &mut req);
        Ok(req)
    }

    pub async fn prompt_messages_once(
        &self,
        messages: Vec<ChatCompletionRequestMessage>,
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

    pub async fn prompt_once(
        &self,
        sys_msg: &str,
        user_msg: &str,
        debug_prefix: Option<&str>,
        cache_key: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let sys = ChatCompletionRequestSystemMessageArgs::default()
            .content(sys_msg)
            .build()?;

        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(user_msg)
            .build()?;
        self.prompt_messages_once(
            vec![sys.into(), user.into()],
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
        req.other.insert(
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
    use crate::settings::{LLMToolChoice, Reasoning};
    use async_openai::types::chat::ReasoningEffort;
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

    #[test]
    fn extensible_chat_completion_request_flattens_extra_fields() {
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content("hello")
            .build()
            .unwrap();
        let request = CreateChatCompletionRequestArgs::default()
            .messages(vec![user.into()])
            .model("mimo-v2.5-pro")
            .build()
            .unwrap();
        let mut request = RawExtensibleChatCompletionRequest::from(request);

        request.other.insert(
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
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content("hello")
            .build()
            .unwrap();
        let request = CreateChatCompletionRequestArgs::default()
            .messages(vec![user.into()])
            .model("mimo-v2.5-pro")
            .build()
            .unwrap();
        let mut request = RawExtensibleChatCompletionRequest::from(request);
        request.messages[0].insert_extra_string("reasoning_content", "I need the tool.");

        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["messages"][0]["content"], "hello");
        assert_eq!(
            value["messages"][0]["reasoning_content"],
            "I need the tool."
        );
    }

    #[test]
    fn extensible_chat_completion_request_serializes_inner_changes() {
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content("hello")
            .build()
            .unwrap();
        let request = CreateChatCompletionRequestArgs::default()
            .messages(vec![user.into()])
            .model("mimo-v2.5-pro")
            .build()
            .unwrap();
        let mut request = RawExtensibleChatCompletionRequest::from(request);

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
            response.choices[0].extra()["provider_choice_id"],
            "choice-123"
        );
        assert_eq!(
            response.choices[0].reasoning_content(),
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
        let mut response = RawExtensibleChatCompletionResponse::new(CreateChatCompletionResponse {
            id: "chatcmpl-test".to_string(),
            choices: Vec::new(),
            created: 1,
            model: "old-model".to_string(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion".to_string(),
            usage: None,
        });
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
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content("hello")
            .build()
            .unwrap();
        let request = CreateChatCompletionRequestArgs::default()
            .messages(vec![user.into()])
            .model("mimo-v2.5-pro")
            .build()
            .unwrap();
        let mut request = RawExtensibleChatCompletionRequest::from(request);
        let model = OpenAIModel::from_str("mimo-v2.5-pro").unwrap();

        apply_provider_request_extensions(
            &model,
            &test_settings(Some(Reasoning(ReasoningEffort::None))),
            &mut request,
        );

        let value = serde_json::to_value(&request).unwrap();
        assert_eq!(value["thinking"]["type"], "disabled");
    }
}
