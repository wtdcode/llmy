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
    ChatCompletionStreamOptionsRaw, ChatCompletionTools, CreateChatCompletionRequestRaw,
    FunctionCallRaw, Role,
};
use crate::resp::{
    ChatChoice, ChatChoiceRaw, ChatCompletionResponseMessageRaw, CompletionUsage,
    CreateChatCompletionResponseRaw, CreateChatCompletionStreamResponse, FinishReason,
};

pub use crate::anthropic::{AnthropicConfig, DEFAULT_ANTHROPIC_VERSION};
use crate::anthropic::{
    AnthropicMessagesRequest, AnthropicMessagesRequestRaw, AnthropicMessagesResponse,
    AnthropicStreamAccumulator, AnthropicStreamEvent,
};
use crate::cache_key::{CacheKeyClaim, CacheKeys, CacheShape};
use crate::debug::{self, DebugBackend, DebugRequest, DebugRowContext, DebugUsage, PrefixBilling};
pub use crate::filters::{GoogleContentFilter, MarkdownTagFilter, NoFilter, OpenAIContentFilter};
pub use crate::message::{Message, MessagePart, MessageRole};
pub use crate::req::{RawExtensibleChatCompletionRequest, RawExtensibleChatRequestMessage};
pub use crate::resp::{RawExtensibleChatChoice, RawExtensibleChatCompletionResponse};
use crate::responses::{
    ResponsesRequest, ResponsesRequestRaw, ResponsesResponse, ResponsesStreamEvent,
};
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
    /// Anthropic Messages protocol at an OpenAI-style base URL (e.g.
    /// `https://api.anthropic.com/v1`).
    Anthropic(AnthropicConfig),
    /// OpenAI Responses protocol; same auth and base URL shape as
    /// [`Self::OpenAI`], different wire format.
    OpenAIResponses(OpenAIConfig),
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

    /// Anthropic Messages protocol. `endpoint` should include the version
    /// segment (e.g. `https://api.anthropic.com/v1`); `version` is the
    /// `anthropic-version` header value, see [`DEFAULT_ANTHROPIC_VERSION`].
    pub fn new_anthropic(endpoint: &str, key: &str, version: &str) -> Self {
        Self::Anthropic(AnthropicConfig::new(endpoint, key, version))
    }

    /// [`Self::new_anthropic`] with bearer auth (`Authorization: Bearer`)
    /// instead of `x-api-key` — the scheme behind `ANTHROPIC_AUTH_TOKEN`.
    pub fn new_anthropic_bearer(endpoint: &str, token: &str, version: &str) -> Self {
        Self::Anthropic(AnthropicConfig::new_bearer(endpoint, token, version))
    }

    /// OpenAI Responses protocol at an OpenAI-style base URL.
    pub fn new_responses(endpoint: &str, key: &str) -> Self {
        let cfg = OpenAIConfig::new()
            .with_api_base(endpoint)
            .with_api_key(key);
        Self::OpenAIResponses(cfg)
    }

    pub fn endpoint_kind(&self) -> &'static str {
        match self {
            Self::Azure { .. } => "azure",
            Self::OpenAI(_) => "openai",
            Self::Anthropic(_) => "anthropic",
            Self::OpenAIResponses(_) => "openai-responses",
        }
    }

    /// The HTTP base URL the underlying client will hit. For Azure this is
    /// the resource endpoint; for OpenAI it's the configured `api_base`.
    pub fn endpoint_url(&self) -> &str {
        use async_openai::config::Config;
        match self {
            Self::Azure { config, .. } => config.api_base(),
            Self::OpenAI(cfg) => cfg.api_base(),
            Self::Anthropic(cfg) => cfg.api_base(),
            Self::OpenAIResponses(cfg) => cfg.api_base(),
        }
    }

    pub fn azure_deployment(&self) -> Option<&str> {
        match self {
            Self::Azure { deployment_id, .. } => Some(deployment_id),
            Self::OpenAI(_) | Self::Anthropic(_) | Self::OpenAIResponses(_) => None,
        }
    }
}

/// The protocol-speaking inner client. Everything above it (billing, debug
/// records, filters, cache keys) works on chat-completion types; each variant
/// converts to and from its own wire format at this boundary.
#[derive(Debug, Clone)]
pub enum LLMClient {
    Azure(Client<AzureConfig>),
    OpenAI(Client<OpenAIConfig>),
    Anthropic(Client<AnthropicConfig>),
    Responses(Client<OpenAIConfig>),
}

/// SSE stream of [`CreateChatCompletionStreamResponse`] chunks until the
/// upstream server emits `[DONE]`. Mirrors `async_openai`'s
/// `ChatCompletionResponseStream` shape using our own response type so we
/// don't lose unknown fields per chunk.
pub type ChatCompletionResponseStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<CreateChatCompletionStreamResponse, OpenAIError>> + Send>,
>;

/// SSE stream of one protocol's own event objects (Anthropic / Responses),
/// aggregated back into a full chat-completion response by
/// [`LLMClient::complete_streaming_extensible`].
type ProtocolEventStream<E> =
    std::pin::Pin<Box<dyn futures::Stream<Item = Result<E, OpenAIError>> + Send>>;

impl LLMClient {
    pub fn new(config: SupportedConfig) -> Self {
        match config {
            SupportedConfig::Azure { config, .. } => Self::Azure(Client::with_config(config)),
            SupportedConfig::OpenAI(cfg) => Self::OpenAI(Client::with_config(cfg)),
            SupportedConfig::Anthropic(cfg) => Self::Anthropic(Client::with_config(cfg)),
            SupportedConfig::OpenAIResponses(cfg) => Self::Responses(Client::with_config(cfg)),
        }
    }

    /// Which wire protocol this client speaks.
    pub fn protocol(&self) -> &'static str {
        match self {
            Self::Azure(_) | Self::OpenAI(_) => "chat-completion",
            Self::Anthropic(_) => "anthropic",
            Self::Responses(_) => "responses",
        }
    }

    /// Resolve a request against this client's protocol. Only one path is
    /// unconditional: a request already in the backend's wire format passes
    /// through verbatim. Everything else — chat structs included — is an
    /// implicit conversion through the chat hub, allowed only when
    /// `allow_implicit_convert` is set (`LLMY_ALLOW_IMPLICIT_CONVERT`):
    /// rewriting a request into another protocol should be a conscious
    /// choice. Callers holding protocol-neutral conversation state rather
    /// than a wire request build natively via [`Self::lower_chat_request`]
    /// instead, which needs no opt-in. `default_max_output_tokens` backs the
    /// Anthropic protocol's mandatory `max_tokens` when a converted request
    /// carries no bound of its own.
    pub fn resolve_request(
        &self,
        req: LLMRequest,
        default_max_output_tokens: u32,
        allow_implicit_convert: bool,
    ) -> Result<LLMRequest, LLMYError> {
        let req = match (self, req) {
            // Already in the backend's own format: send exactly as given.
            (Self::Azure(_) | Self::OpenAI(_), req @ LLMRequest::Chat(_))
            | (Self::Anthropic(_), req @ LLMRequest::Anthropic(_))
            | (Self::Responses(_), req @ LLMRequest::Responses(_)) => return Ok(req),
            (_, req) => req,
        };
        if !allow_implicit_convert {
            return Err(eyre!(
                "a {} request cannot be sent over the {} protocol; set \
                 LLMY_ALLOW_IMPLICIT_CONVERT (--allow-implicit-convert) to convert it through \
                 the chat form",
                req.protocol(),
                self.protocol()
            )
            .into());
        }
        let chat = match req {
            LLMRequest::Chat(chat) => chat,
            LLMRequest::Anthropic(req) => req.into_inner().into_chat_request()?,
            LLMRequest::Responses(req) => req.into_inner().into_chat_request()?,
        };
        self.lower_chat_request(chat, default_max_output_tokens)
    }

    /// Lower a chat-typed request into this backend's wire format — the
    /// *explicit* counterpart of the conversion inside
    /// [`Self::resolve_request`]. Callers holding protocol-neutral
    /// conversation state (the agent harness, the message-level prompt APIs)
    /// build their native request through here deliberately, so the
    /// `allow_implicit_convert` opt-in does not apply.
    pub fn lower_chat_request(
        &self,
        chat: RawExtensibleChatCompletionRequest,
        default_max_output_tokens: u32,
    ) -> Result<LLMRequest, LLMYError> {
        match self {
            Self::Azure(_) | Self::OpenAI(_) => Ok(LLMRequest::Chat(chat)),
            Self::Anthropic(_) => Ok(LLMRequest::Anthropic(
                AnthropicMessagesRequestRaw::from_chat(&chat, default_max_output_tokens)?,
            )),
            Self::Responses(_) => Ok(LLMRequest::Responses(ResponsesRequestRaw::from_chat(
                &chat,
            )?)),
        }
    }

    /// Send one resolved, non-streaming request verbatim; the response comes
    /// back in the protocol's own wire format.
    pub async fn send(&self, req: &LLMRequest) -> Result<LLMResponse, LLMYError> {
        match (self, req) {
            (Self::Azure(cl), LLMRequest::Chat(req)) => {
                let resp: RawExtensibleChatCompletionResponse = cl.chat().create_byot(req).await?;
                Ok(LLMResponse::Chat(resp))
            }
            (Self::OpenAI(cl), LLMRequest::Chat(req)) => {
                let resp: RawExtensibleChatCompletionResponse = cl.chat().create_byot(req).await?;
                Ok(LLMResponse::Chat(resp))
            }
            (Self::Anthropic(cl), LLMRequest::Anthropic(req)) => {
                let resp: AnthropicMessagesResponse = cl.chat().create_byot(req).await?;
                Ok(LLMResponse::Anthropic(resp))
            }
            (Self::Responses(cl), LLMRequest::Responses(req)) => {
                let resp: ResponsesResponse = cl.responses().create_byot(req).await?;
                Ok(LLMResponse::Responses(resp))
            }
            (client, req) => Err(eyre!(
                "a {} request reached the {} protocol unsent; resolve_request must run first",
                req.protocol(),
                client.protocol()
            )
            .into()),
        }
    }

    /// Raw chat-completion chunk stream. Only the chat protocols emit these
    /// chunks; on Anthropic/Responses use [`Self::send_streaming`], which
    /// aggregates each protocol's own event stream instead.
    pub async fn create_chat_stream_extensible(
        &self,
        req: &RawExtensibleChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        match self {
            Self::Azure(cl) => cl.chat().create_stream_byot(req).await,
            Self::OpenAI(cl) => cl.chat().create_stream_byot(req).await,
            Self::Anthropic(_) | Self::Responses(_) => Err(OpenAIError::InvalidArgument(
                "chat-completion chunk streams exist only on the chat protocols".to_string(),
            )),
        }
    }

    /// Send one resolved streaming request and aggregate the protocol's event
    /// stream back into a full response. `fallback_model` fills the chat
    /// response's model field when the stream never named one.
    pub async fn send_streaming(
        &self,
        req: &mut LLMRequest,
        fallback_model: &str,
    ) -> Result<LLMResponse, LLMYError> {
        match (self, req) {
            (Self::Azure(_) | Self::OpenAI(_), LLMRequest::Chat(chat)) => Ok(LLMResponse::Chat(
                self.chat_streaming(chat, fallback_model).await?,
            )),
            (Self::Anthropic(cl), LLMRequest::Anthropic(converted)) => {
                converted.stream = Some(true);
                let mut stream: ProtocolEventStream<AnthropicStreamEvent> =
                    cl.chat().create_stream_byot(&*converted).await?;
                let mut acc = AnthropicStreamAccumulator::new();
                while let Some(event) = stream.next().await {
                    acc.push(event?)?;
                }
                Ok(LLMResponse::Anthropic(acc.finish()?))
            }
            (Self::Responses(cl), LLMRequest::Responses(converted)) => {
                converted.stream = Some(true);
                let mut stream: ProtocolEventStream<ResponsesStreamEvent> =
                    cl.responses().create_stream_byot(&*converted).await?;
                // The terminal events carry the entire final response object,
                // so aggregation is just waiting for one of them. A `failed`
                // response is kept: its error field surfaces when the caller
                // takes the chat view.
                let mut terminal: Option<ResponsesResponse> = None;
                while let Some(event) = stream.next().await {
                    match event? {
                        ResponsesStreamEvent::Completed { response }
                        | ResponsesStreamEvent::Incomplete { response }
                        | ResponsesStreamEvent::Failed { response } => terminal = Some(response),
                        ResponsesStreamEvent::Error { code, message } => {
                            return Err(
                                eyre!("responses stream error {:?}: {}", code, message).into()
                            );
                        }
                        ResponsesStreamEvent::Other => {}
                    }
                }
                Ok(LLMResponse::Responses(terminal.ok_or_else(|| {
                    eyre!("responses stream ended without a terminal response event")
                })?))
            }
            (client, req) => Err(eyre!(
                "a {} request reached the {} protocol unsent; resolve_request must run first",
                req.protocol(),
                client.protocol()
            )
            .into()),
        }
    }

    /// Aggregate a chat-completion chunk stream into one full response.
    #[allow(deprecated)]
    async fn chat_streaming(
        &self,
        req: &mut RawExtensibleChatCompletionRequest,
        fallback_model: &str,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        req.stream = Some(true);

        if req.stream_options.is_none() {
            req.stream_options = Some(WithOtherFields::new(ChatCompletionStreamOptionsRaw {
                include_usage: Some(true),
                include_obfuscation: None,
            }));
        }

        let mut stream = self.create_chat_stream_extensible(&*req).await?;

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
            model: model.unwrap_or_else(|| fallback_model.to_string()),
            service_tier,
            system_fingerprint,
            object: "chat.completion".to_string(),
            usage,
        };
        Ok(RawExtensibleChatCompletionResponse::new(resp_raw))
    }
}

/// A request in the wire format it will be sent in. The rule is transparency:
/// whatever the caller hands over is sent as-is when the backend speaks that
/// protocol; any cross-protocol send is an implicit conversion through the
/// chat hub, gated by `allow_implicit_convert` (see
/// [`LLMClient::resolve_request`]).
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum LLMRequest {
    Chat(RawExtensibleChatCompletionRequest),
    Anthropic(AnthropicMessagesRequest),
    Responses(ResponsesRequest),
}

impl LLMRequest {
    /// Which wire protocol this request is in.
    pub fn protocol(&self) -> &'static str {
        match self {
            Self::Chat(_) => "chat-completion",
            Self::Anthropic(_) => "anthropic",
            Self::Responses(_) => "responses",
        }
    }

    fn prompt_cache_key(&self) -> Option<&str> {
        match self {
            Self::Chat(req) => req.prompt_cache_key.as_deref(),
            Self::Responses(req) => req.prompt_cache_key.as_deref(),
            Self::Anthropic(_) => None,
        }
    }

    /// Set the routing key on the protocols that have one; the Anthropic
    /// protocol routes by content, so there is nothing to set.
    fn set_prompt_cache_key(&mut self, key: &str) {
        match self {
            Self::Chat(req) => req.prompt_cache_key = Some(key.to_string()),
            Self::Responses(req) => req.prompt_cache_key = Some(key.to_string()),
            Self::Anthropic(_) => {}
        }
    }

    /// The cache-key shape, for the protocols that route by
    /// `prompt_cache_key`; `None` means auto keys do not apply.
    fn cache_shape(&self) -> Option<CacheShape> {
        match self {
            Self::Chat(req) => Some(CacheShape::from_chat(req)),
            Self::Responses(req) => Some(req.cache_shape()),
            Self::Anthropic(_) => None,
        }
    }

    /// Raw text for token estimation.
    fn estimate_text(&self) -> String {
        match self {
            Self::Chat(req) => debug::extract_raw_text_with_other(req),
            Self::Anthropic(req) => req.estimate_text(),
            Self::Responses(req) => req.estimate_text(),
        }
    }

    /// Render this request for the debug record — the JSON is the wire truth,
    /// whichever protocol it goes out in.
    fn debug_request(&self) -> DebugRequest {
        match self {
            Self::Chat(req) => DebugRequest::from_chat(req),
            Self::Anthropic(req) => DebugRequest {
                json: serde_json::to_value(req).unwrap_or(serde_json::Value::Null),
                conversation: req.conversation_text(),
                tools_text: req.tools_text(),
            },
            Self::Responses(req) => DebugRequest {
                json: serde_json::to_value(req).unwrap_or(serde_json::Value::Null),
                conversation: req.conversation_text(),
                tools_text: req.tools_text(),
            },
        }
    }
}

/// A response in the wire format it arrived in.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(untagged)]
pub enum LLMResponse {
    Chat(RawExtensibleChatCompletionResponse),
    Anthropic(AnthropicMessagesResponse),
    Responses(ResponsesResponse),
}

impl LLMResponse {
    /// The normalized chat-completion view every caller-facing API returns;
    /// billing and content filters read it too. A chat response passes through
    /// untouched; a native response that is itself a failure surfaces here as
    /// an error.
    pub fn into_chat_view(self) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        match self {
            Self::Chat(resp) => Ok(resp),
            Self::Anthropic(resp) => Ok(resp.into_inner().into_chat_response()),
            Self::Responses(resp) => resp.into_inner().into_chat_response(),
        }
    }

    /// The assistant turn of this response in protocol-neutral form, typed
    /// parts preserved (signed thinking, encrypted reasoning, tool-call
    /// extras) — what conversation-state callers push into their context.
    pub fn to_message(&self) -> Message {
        match self {
            Self::Chat(resp) => resp
                .choices
                .first()
                .map(Message::from_chat_choice)
                .unwrap_or_else(|| Message::new(MessageRole::Assistant, vec![])),
            Self::Anthropic(resp) => resp.to_message(),
            Self::Responses(resp) => resp.to_message(),
        }
    }
}

impl From<RawExtensibleChatCompletionRequest> for LLMRequest {
    fn from(req: RawExtensibleChatCompletionRequest) -> Self {
        Self::Chat(req)
    }
}

impl From<CreateChatCompletionRequestRaw> for LLMRequest {
    fn from(req: CreateChatCompletionRequestRaw) -> Self {
        Self::Chat(RawExtensibleChatCompletionRequest::new(req))
    }
}

impl From<AnthropicMessagesRequest> for LLMRequest {
    fn from(req: AnthropicMessagesRequest) -> Self {
        Self::Anthropic(req)
    }
}

impl From<AnthropicMessagesRequestRaw> for LLMRequest {
    fn from(req: AnthropicMessagesRequestRaw) -> Self {
        Self::Anthropic(WithOtherFields::new(req))
    }
}

impl From<ResponsesRequest> for LLMRequest {
    fn from(req: ResponsesRequest) -> Self {
        Self::Responses(req)
    }
}

impl From<ResponsesRequestRaw> for LLMRequest {
    fn from(req: ResponsesRequestRaw) -> Self {
        Self::Responses(WithOtherFields::new(req))
    }
}

impl From<RawExtensibleChatCompletionResponse> for LLMResponse {
    fn from(resp: RawExtensibleChatCompletionResponse) -> Self {
        Self::Chat(resp)
    }
}

impl From<AnthropicMessagesResponse> for LLMResponse {
    fn from(resp: AnthropicMessagesResponse) -> Self {
        Self::Anthropic(resp)
    }
}

impl From<ResponsesResponse> for LLMResponse {
    fn from(resp: ResponsesResponse) -> Self {
        Self::Responses(resp)
    }
}

impl TryFrom<LLMResponse> for RawExtensibleChatCompletionResponse {
    type Error = LLMYError;

    fn try_from(resp: LLMResponse) -> Result<Self, Self::Error> {
        resp.into_chat_view()
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

        let content_filter: Box<dyn OpenAIContentFilter> = if model.is_google() {
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
        let concurrency = LLMInner::concurrency_limiter(settings.llm_concurrent);

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
                concurrency,
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
    /// Global in-flight request limiter (`llm_concurrent`), shared with every
    /// scope/clone of this client; `None` when unlimited.
    concurrency: Option<Arc<tokio::sync::Semaphore>>,
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
            concurrency: self.concurrency.clone(),
        }
    }

    /// The output-token bound for protocols that demand one (the Anthropic
    /// protocol's mandatory `max_tokens`) when `llm_max_completion_tokens` is
    /// unset: the model's configured max, or 8192 when the config doesn't say
    /// (custom `name,input,output` models leave it at zero — the wire rejects
    /// `max_tokens: 0`).
    fn default_max_output_tokens(&self) -> u32 {
        match self.model.config.max_tokens {
            0 => 8192,
            max => max.try_into().unwrap_or(u32::MAX),
        }
    }

    /// Lower a chat-typed request into the backend's wire format — see
    /// [`LLMClient::lower_chat_request`]. This is how conversation-state
    /// callers (agents, the message-level prompt APIs) go native on any
    /// backend without the implicit-conversion opt-in.
    pub fn lower_request(
        &self,
        chat: RawExtensibleChatCompletionRequest,
    ) -> Result<LLMRequest, LLMYError> {
        self.client
            .lower_chat_request(chat, self.default_max_output_tokens())
    }

    /// Replace the content filter applied to every request and response. Defaults to
    /// `GoogleContentFilter` for google models, `NoFilter` otherwise.
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
    // Production traffic goes through [`Self::auto_cache_key_request`]; this
    // chat-typed shorthand remains for the tests exercising claim semantics.
    #[cfg(test)]
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

    /// Take the auto cache key for one logical request, whatever protocol it
    /// resolved to. The Anthropic protocol routes by content, so it never
    /// takes one; a caller-supplied key is never second-guessed.
    fn auto_cache_key_request(
        &self,
        req: &LLMRequest,
        debug_prefix: Option<&str>,
    ) -> Option<CacheKeyClaim> {
        if req.prompt_cache_key().is_some() {
            return None;
        }
        let shape = req.cache_shape()?;
        self.cache_keys
            .select_shape(&shape, &self.model, debug_prefix)
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

    /// The limiter an `llm_concurrent` value describes: `None` when 0
    /// (unlimited).
    fn concurrency_limiter(limit: usize) -> Option<Arc<tokio::sync::Semaphore>> {
        (limit > 0).then(|| Arc::new(tokio::sync::Semaphore::new(limit)))
    }

    /// Take a slot on the global concurrency limiter; trivially `None` when
    /// the client is unlimited. Holding the permit spans the wire round trip,
    /// so excess requests queue here instead of piling onto the endpoint.
    async fn acquire_concurrency_slot(
        &self,
    ) -> Result<Option<tokio::sync::SemaphorePermit<'_>>, LLMYError> {
        match &self.concurrency {
            Some(semaphore) => {
                Ok(Some(semaphore.acquire().await.map_err(|_| {
                    LLMYError::Other(eyre!("the concurrency limiter was closed"))
                })?))
            }
            None => Ok(None),
        }
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
        let settings = settings.unwrap_or_else(|| self.default_settings.clone());
        let req = self.build_prompt_request(sys_msg, user_msg, cache_key, &settings)?;
        self.complete_request_once_with_retry(
            req,
            debug_prefix,
            Some(settings.timeout()),
            Some(settings.llm_retry),
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
        let settings = settings.unwrap_or_else(|| self.default_settings.clone());
        let req = self.build_prompt_request(sys_msg, user_msg, cache_key, &settings)?;
        self.complete_request_once_with_retry_typed::<T>(
            req,
            debug_prefix,
            Some(settings.timeout()),
            Some(settings.llm_retry),
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
        self.complete_request_once_with_retry(
            LLMRequest::Chat(req.clone()),
            debug_prefix,
            timeout,
            retry,
        )
        .await
    }

    /// Retry wrapper over one logical request in any wire format. The request
    /// is resolved against the backend exactly once — passthrough when it is
    /// already in the backend's protocol; any cross-protocol conversion needs
    /// `allow_implicit_convert` — and every attempt then sends the same
    /// resolved request.
    pub async fn complete_request_once_with_retry(
        &self,
        req: LLMRequest,
        debug_prefix: Option<&str>,
        timeout: Option<Duration>,
        retry: Option<u64>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        self.complete_request_message_once_with_retry(req, debug_prefix, timeout, retry)
            .await
            .map(|(resp, _)| resp)
    }

    /// Like [`Self::complete_request_once_with_retry`], but also returns the
    /// protocol-faithful assistant [`Message`] parsed from the wire response —
    /// what conversation-state callers (the agent harness) push into their
    /// context, so signed thinking and encrypted reasoning survive the turn.
    pub async fn complete_request_message_once_with_retry(
        &self,
        req: LLMRequest,
        debug_prefix: Option<&str>,
        timeout: Option<Duration>,
        retry: Option<u64>,
    ) -> Result<(RawExtensibleChatCompletionResponse, Message), LLMYError> {
        let req = self.client.resolve_request(
            req,
            self.default_max_output_tokens(),
            self.default_settings.allow_implicit_convert,
        )?;
        let retry = retry.unwrap_or(u64::MAX);
        // One claim for the whole logical request; every attempt gets a clone,
        // and giving up drops the last one, which abandons it.
        let claim = self.auto_cache_key_request(&req, debug_prefix);

        let mut last = None;
        for idx in 0..retry {
            // Selection paid for the first send; a retry is extra traffic on the
            // same key and costs another slot of its budget.
            if let (true, Some(claim)) = (idx > 0, claim.as_ref()) {
                claim.charge_resend();
            }
            match self
                .complete_request_attempt(req.clone(), claim.clone(), debug_prefix, timeout)
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
        self.complete_request_once_with_retry_typed::<T>(
            LLMRequest::Chat(req.clone()),
            debug_prefix,
            timeout,
            retry,
        )
        .await
    }

    /// Like [`Self::complete_request_once_with_retry`], but each attempt also
    /// deserializes the first-choice content into `T` (with markdown
    /// auto-strip); a malformed response is retried like any other error.
    pub async fn complete_request_once_with_retry_typed<T: DeserializeOwned>(
        &self,
        req: LLMRequest,
        debug_prefix: Option<&str>,
        timeout: Option<Duration>,
        retry: Option<u64>,
    ) -> Result<T, LLMYError> {
        let req = self.client.resolve_request(
            req,
            self.default_max_output_tokens(),
            self.default_settings.allow_implicit_convert,
        )?;
        let retry = retry.unwrap_or(u64::MAX);
        // One claim for the whole logical request — see the untyped variant.
        let claim = self.auto_cache_key_request(&req, debug_prefix);

        let mut last = None;
        for idx in 0..retry {
            if let (true, Some(claim)) = (idx > 0, claim.as_ref()) {
                claim.charge_resend();
            }
            let attempt = self
                .complete_request_attempt(req.clone(), claim.clone(), debug_prefix, timeout)
                .await
                .and_then(|(resp, _)| self.parse_first_choice::<T>(&resp));
            match attempt {
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
        let resp = self
            .complete_request(LLMRequest::Chat(req), debug_prefix, timeout_overwrite)
            .await?;
        self.parse_first_choice::<T>(&resp)
    }

    pub async fn complete_extensible(
        &self,
        req: RawExtensibleChatCompletionRequest,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        self.complete_request(LLMRequest::Chat(req), debug_prefix, timeout_overwrite)
            .await
    }

    /// One-shot send of a request in any wire format: passthrough when the
    /// backend speaks it; a cross-protocol send needs `allow_implicit_convert`
    /// to convert through the chat hub. A one-shot
    /// call is a logical request of exactly one attempt, so the cache-key
    /// claim is taken and dropped here — abandoned if the call does not land.
    pub async fn complete_request(
        &self,
        req: LLMRequest,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        let req = self.client.resolve_request(
            req,
            self.default_max_output_tokens(),
            self.default_settings.allow_implicit_convert,
        )?;
        let claim = self.auto_cache_key_request(&req, debug_prefix);
        self.complete_request_attempt(req, claim, debug_prefix, timeout_overwrite)
            .await
            .map(|(resp, _)| resp)
    }

    /// One attempt of a resolved logical request, carrying that request's
    /// cache key claim — see [`Self::auto_cache_key_request`]. The attempt only
    /// ever uses the claim it is given; it never takes one of its own, so
    /// retrying cannot cost a second claim or a second slot of the key's
    /// request budget.
    async fn complete_request_attempt(
        &self,
        mut req: LLMRequest,
        given_claim: Option<CacheKeyClaim>,
        debug_prefix: Option<&str>,
        timeout_overwrite: Option<Duration>,
    ) -> Result<(RawExtensibleChatCompletionResponse, Message), LLMYError> {
        // A limited client queues right at the attempt's entry. Nothing about
        // the attempt (budget check, debug row, token estimate, wire clock)
        // happens until it owns a slot; the permit then spans the whole round
        // trip (streaming included — the stream is consumed inside `llm_fut`),
        // keeping queue wait out of the tok/s math and the request timeout.
        let _concurrency_permit = self.acquire_concurrency_slot().await?;

        // Check the budget only once the slot is owned: usage recorded by
        // requests that finished while this one queued counts against it, so
        // an over-cap request can't slip out on a pre-queue check gone stale.
        // A refused request holds its slot only for the length of this read.
        self.billing.read().unwrap().check_cap(self.node)?;

        if let Some(claim) = given_claim.as_ref() {
            tracing::debug!("auto prompt cache key {}", claim.key());
            req.set_prompt_cache_key(claim.key());
        }
        // The content-filter quirks are all chat-endpoint quirks; a native
        // request goes out exactly as built.
        if let LLMRequest::Chat(chat) = &mut req {
            self.apply_filter_input(chat);
        }

        // Keep the raw prefix (None => "") for the per-prefix billing dimension,
        // before it gets defaulted to "llm" for the debug backend below.
        let billing_prefix = debug_prefix;

        let use_stream = self.default_settings.llm_stream;
        let debug_prefix = if let Some(debug_prefix) = debug_prefix {
            debug_prefix.to_string()
        } else {
            "llm".to_string()
        };

        let dbg_req = self.debug_backend.as_ref().map(|_| req.debug_request());
        let dbg_handle = if let (Some(backend), Some(dbg_req)) =
            (self.debug_backend.as_ref(), dbg_req.as_ref())
        {
            let ctx = self.debug_row_context(req.prompt_cache_key());
            backend.start(&debug_prefix, ctx, dbg_req).await
        } else {
            None
        };

        let estimated_tokens = {
            let text = req.estimate_text();
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
                self.client
                    .send_streaming(&mut req, self.model.api_model_name())
                    .await
            } else {
                self.client.send(&req).await
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

        let resp = match resp {
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
        // The wire-truth JSON for the debug record, captured before the
        // response is normalized into its chat view.
        let resp_json = dbg_handle
            .as_ref()
            .map(|_| serde_json::to_value(&resp).unwrap_or(serde_json::Value::Null));
        // The protocol-faithful assistant turn, before normalization sheds
        // anything the chat view cannot carry.
        let assistant = resp.to_message();
        let mut resp = match resp.into_chat_view() {
            Ok(view) => view,
            Err(e) => {
                // A response that is itself a failure (e.g. a `failed`
                // responses object) is recorded like any other error.
                if let (Some(backend), Some(handle)) =
                    (self.debug_backend.as_ref(), dbg_handle.as_ref())
                {
                    backend.record_error(handle, &e).await;
                }
                return Err(e);
            }
        };
        // The provider answered, so it really did cache this prompt's prefixes.
        // Settle before the billing record below, which can still bail out.
        if let Some(claim) = given_claim.as_ref() {
            claim.confirm(Self::reported_cached_tokens(&resp));
        }
        self.apply_filter_output(&mut resp);
        if let (Some(backend), Some(handle), Some(dbg_req), Some(resp_json)) = (
            self.debug_backend.as_ref(),
            dbg_handle.as_ref(),
            dbg_req.as_ref(),
            resp_json.as_ref(),
        ) {
            backend
                .record_response(handle, dbg_req, resp_json, &resp)
                .await;
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
        Ok((resp, assistant))
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

    /// Build the protocol-native request for a plain system+user prompt:
    /// strings go straight into the backend's own wire format instead of being
    /// forced through the chat shape only to be converted back out of it.
    /// Chat-typed callers (message/struct APIs) still build chat requests and
    /// rely on [`LLMClient::resolve_request`] to convert — which, on a
    /// non-chat backend, needs `allow_implicit_convert`.
    pub fn build_prompt_request(
        &self,
        sys_msg: &str,
        user_msg: &str,
        cache_key: Option<&str>,
        settings: &LLMSettings,
    ) -> Result<LLMRequest, LLMYError> {
        match &self.client {
            LLMClient::Azure(_) | LLMClient::OpenAI(_) => {
                let sys = ChatCompletionRequestSystemMessageRaw::new_text(sys_msg);
                let user = ChatCompletionRequestUserMessageRaw::new_text(user_msg);
                let messages = vec![
                    ChatCompletionRequestMessageRaw::System(sys),
                    ChatCompletionRequestMessageRaw::User(user),
                ]
                .into_iter()
                .map(RawExtensibleChatRequestMessage::new)
                .collect();
                Ok(LLMRequest::Chat(
                    self.build_chat_request(messages, cache_key, settings, None)?,
                ))
            }
            LLMClient::Anthropic(_) => Ok(LLMRequest::Anthropic(
                AnthropicMessagesRequestRaw::from_prompt(
                    self.model.api_model_name(),
                    sys_msg,
                    user_msg,
                    settings,
                    self.default_max_output_tokens(),
                )?,
            )),
            LLMClient::Responses(_) => Ok(LLMRequest::Responses(ResponsesRequestRaw::from_prompt(
                self.model.api_model_name(),
                sys_msg,
                user_msg,
                cache_key,
                settings,
            )?)),
        }
    }

    /// Build the backend-native request for a protocol-neutral conversation:
    /// typed parts map straight onto each protocol's own constructs (signed
    /// thinking, encrypted reasoning, tool calls); on chat backends the
    /// conversation lowers via [`Message::many_to_chat`].
    pub fn build_conversation_request(
        &self,
        conversation: &[Message],
        cache_key: Option<&str>,
        settings: &LLMSettings,
        tools: Option<Vec<ChatCompletionTools>>,
    ) -> Result<LLMRequest, LLMYError> {
        match &self.client {
            LLMClient::Azure(_) | LLMClient::OpenAI(_) => {
                let messages = Message::many_to_chat(conversation);
                Ok(LLMRequest::Chat(self.build_chat_request(
                    messages, cache_key, settings, tools,
                )?))
            }
            LLMClient::Anthropic(_) => Ok(LLMRequest::Anthropic(
                AnthropicMessagesRequestRaw::from_conversation(
                    self.model.api_model_name(),
                    conversation,
                    tools.as_deref(),
                    settings,
                    self.default_max_output_tokens(),
                )?,
            )),
            LLMClient::Responses(_) => Ok(LLMRequest::Responses(
                ResponsesRequestRaw::from_conversation(
                    self.model.api_model_name(),
                    conversation,
                    tools.as_deref(),
                    cache_key,
                    settings,
                )?,
            )),
        }
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
        // Explicit lowering: message-level callers go native on any backend.
        let req = self.lower_request(req)?;
        self.complete_request_once_with_retry(req, debug_prefix, Some(timeout), Some(retry))
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
        // Explicit lowering: message-level callers go native on any backend.
        let req = self.lower_request(req)?;
        self.complete_request_once_with_retry_typed::<T>(
            req,
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
        self.prompt_once_with_retry(sys_msg, user_msg, debug_prefix, cache_key, settings)
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
            allow_implicit_convert: false,
            llm_concurrent: 0,
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
    fn a_zero_concurrency_setting_means_no_limiter() {
        assert!(LLMInner::concurrency_limiter(0).is_none());
        let limiter = LLMInner::concurrency_limiter(2).expect("limited");
        let _a = limiter.try_acquire().expect("slot 1");
        let _b = limiter.try_acquire().expect("slot 2");
        assert!(
            limiter.try_acquire().is_err(),
            "a third in-flight request must queue"
        );
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

    // --- protocol passthrough / conversion ----------------------------------

    fn anthropic_llm() -> LLM {
        let config =
            SupportedConfig::new_anthropic("http://localhost:0/v1", "k", DEFAULT_ANTHROPIC_VERSION);
        let model = OpenAIModel::from_str("captest,1000000,1000000").unwrap();
        LLM::new(
            config,
            model,
            rust_decimal::dec!(100),
            test_settings(None),
            None,
        )
    }

    fn responses_llm() -> LLM {
        let config = SupportedConfig::new_responses("http://localhost:0/v1", "k");
        let model = OpenAIModel::from_str("captest,1000000,1000000").unwrap();
        LLM::new(
            config,
            model,
            rust_decimal::dec!(100),
            test_settings(None),
            None,
        )
    }

    #[test]
    fn a_string_prompt_builds_the_backend_native_request() {
        // Chat backend: the chat request, as before.
        let req = test_llm()
            .build_prompt_request("sys", "user", None, &test_settings(None))
            .unwrap();
        assert_eq!(req.protocol(), "chat-completion");

        // Anthropic backend: the native messages request, no chat intermediate.
        let req = anthropic_llm()
            .build_prompt_request("sys", "user", None, &test_settings(None))
            .unwrap();
        assert_eq!(req.protocol(), "anthropic");
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value["system"][0]["text"], "sys");
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["messages"][0]["content"][0]["text"], "user");
        assert!(value["max_tokens"].is_number());

        // Responses backend: the native responses request.
        let req = responses_llm()
            .build_prompt_request("sys", "user", Some("key-9"), &test_settings(None))
            .unwrap();
        assert_eq!(req.protocol(), "responses");
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value["input"][0]["role"], "system");
        assert_eq!(value["input"][1]["role"], "user");
        assert_eq!(value["input"][1]["content"], "user");
        assert_eq!(value["prompt_cache_key"], "key-9");
        assert_eq!(value["store"], false);
    }

    #[test]
    fn resolve_passes_matching_requests_through_and_gates_every_conversion() {
        let anthropic = anthropic_llm();
        let settings = test_settings(None);

        // A request already in the backend's own format passes through
        // byte-for-byte — no opt-in needed.
        let native = anthropic
            .build_prompt_request("s", "u", None, &settings)
            .unwrap();
        let before = serde_json::to_value(&native).unwrap();
        let resolved = anthropic
            .client
            .resolve_request(native, 4096, false)
            .unwrap();
        assert_eq!(serde_json::to_value(&resolved).unwrap(), before);

        // Every cross-protocol send is an implicit conversion and is refused
        // by default — a chat struct included...
        let chat: LLMRequest = user_request("hello").into();
        let err = anthropic
            .client
            .resolve_request(chat.clone(), 4096, false)
            .unwrap_err();
        assert!(
            err.to_string().contains("LLMY_ALLOW_IMPLICIT_CONVERT"),
            "{err}"
        );
        let foreign = responses_llm()
            .build_prompt_request("sys", "hello", None, &settings)
            .unwrap();
        let err = anthropic
            .client
            .resolve_request(foreign.clone(), 4096, false)
            .unwrap_err();
        assert!(
            err.to_string().contains("LLMY_ALLOW_IMPLICIT_CONVERT"),
            "{err}"
        );

        // ...and converts through the chat hub once the caller opts in.
        let resolved = anthropic.client.resolve_request(chat, 4096, true).unwrap();
        assert_eq!(resolved.protocol(), "anthropic");
        let resolved = anthropic
            .client
            .resolve_request(foreign, 4096, true)
            .unwrap();
        assert_eq!(resolved.protocol(), "anthropic");
        let value = serde_json::to_value(&resolved).unwrap();
        assert_eq!(value["system"][0]["text"], "sys");
        assert_eq!(value["messages"][0]["content"][0]["text"], "hello");

        // Native folds back down to plain chat the same way.
        let native = anthropic
            .build_prompt_request("sys", "hello", None, &settings)
            .unwrap();
        let resolved = test_llm()
            .client
            .resolve_request(native, 4096, true)
            .unwrap();
        assert_eq!(resolved.protocol(), "chat-completion");
        let value = serde_json::to_value(&resolved).unwrap();
        assert_eq!(value["messages"][0]["role"], "system");
        assert_eq!(value["messages"][0]["content"], "sys");
        assert_eq!(value["messages"][1]["role"], "user");
        assert_eq!(value["messages"][1]["content"], "hello");
    }

    #[test]
    fn trivial_conversions_wrap_and_unwrap() {
        let req: LLMRequest = user_request("hi").into();
        assert_eq!(req.protocol(), "chat-completion");

        let resp: LLMResponse = crate::filters::build_resp(Some("ok"), FinishReason::Stop).into();
        let chat: RawExtensibleChatCompletionResponse = resp.try_into().unwrap();
        assert_eq!(
            chat.choices[0].inner.message.inner.content.as_deref(),
            Some("ok")
        );
    }

    #[test]
    fn conversation_state_lowers_natively_without_the_opt_in() {
        // Agents and the message-level APIs hold chat-typed conversation
        // state; lowering it is an explicit build, not an implicit conversion,
        // so it needs no allow_implicit_convert even on native backends.
        let anthropic = anthropic_llm();
        assert!(!anthropic.default_settings.allow_implicit_convert);
        let lowered = anthropic.lower_request(user_request("hello")).unwrap();
        assert_eq!(lowered.protocol(), "anthropic");

        let responses = responses_llm();
        let lowered = responses.lower_request(user_request("hello")).unwrap();
        assert_eq!(lowered.protocol(), "responses");

        let chat = test_llm().lower_request(user_request("hello")).unwrap();
        assert_eq!(chat.protocol(), "chat-completion");
    }

    #[test]
    fn native_requests_take_auto_cache_keys_only_where_the_protocol_has_them() {
        // The responses protocol routes by prompt_cache_key, so a native
        // request still gets an auto key from its own rendering.
        let responses = responses_llm();
        let req = responses
            .build_prompt_request("sys", "hello", None, &test_settings(None))
            .unwrap();
        let claim = responses
            .auto_cache_key_request(&req, None)
            .expect("auto key");
        claim.confirm(0);

        // The anthropic protocol routes by content: no key, no claim.
        let anthropic = anthropic_llm();
        let req = anthropic
            .build_prompt_request("sys", "hello", None, &test_settings(None))
            .unwrap();
        assert!(anthropic.auto_cache_key_request(&req, None).is_none());
    }
}
