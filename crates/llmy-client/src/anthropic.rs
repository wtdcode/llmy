//! Anthropic Messages protocol support.
//!
//! Three pieces live here: wire types mirroring the
//! [Messages API](https://docs.anthropic.com/en/api/messages) (wrapped in
//! [`WithOtherFields`] like the chat types, so unknown fields survive a round
//! trip), the [`AnthropicConfig`] that teaches the shared `async_openai` HTTP
//! client to speak the protocol, and the conversions between these types and
//! the chat-completion types the rest of the crate works in.
//!
//! The client-facing surface stays chat completion: `LLMClient` converts an
//! outgoing [`RawExtensibleChatCompletionRequest`] into an
//! [`AnthropicMessagesRequest`] at the wire boundary and folds the reply back
//! into a [`RawExtensibleChatCompletionResponse`], so billing, debug records,
//! content filters and cache keys all keep operating on one set of types.
//!
//! Multi-turn fidelity: conversation-state callers hold
//! [`crate::message::Message`]s, and this module maps them natively in both
//! directions — [`AnthropicMessagesRequestRaw::from_conversation`] builds the
//! request straight from typed parts (signed thinking replays verbatim), and
//! [`AnthropicMessagesResponseRaw::to_message`] parses the reply back into
//! them — so extended thinking with tool use survives multi-turn round trips.

use std::time::{SystemTime, UNIX_EPOCH};

use color_eyre::eyre::eyre;
use llmy_types::{error::LLMYError, other::WithOtherFields};
use secrecy::SecretString;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{Message, MessagePart, MessageRole};
use crate::req::{
    ChatCompletionMessageToolCallRaw, ChatCompletionMessageToolCallsRaw,
    ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestAssistantMessageContentPartRaw, ChatCompletionRequestAssistantMessageRaw,
    ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestDeveloperMessageContentPartRaw, ChatCompletionRequestMessageRaw,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestSystemMessageContentPartRaw,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPartRaw,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPartRaw,
    ChatCompletionToolChoiceOption, ChatCompletionToolChoiceOptionRaw, ChatCompletionTools,
    ChatCompletionToolsRaw, CreateChatCompletionRequestRaw, FunctionCallRaw, PromptCacheBreakpoint,
    RawExtensibleChatCompletionRequest, ReasoningEffort, Role, StopConfiguration,
    ToolChoiceOptions,
};
use crate::req::{
    ChatCompletionNamedToolChoiceRaw, ChatCompletionRequestAssistantMessageContentPart,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImageRaw,
    ChatCompletionRequestMessageContentPartTextRaw, ChatCompletionRequestSystemMessageRaw,
    ChatCompletionRequestToolMessageRaw, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionRequestUserMessageRaw, ChatCompletionToolRaw, FunctionNameRaw, FunctionObjectRaw,
    ImageUrlRaw, PromptCacheBreakpointRaw,
};
use crate::resp::{
    ChatChoiceRaw, ChatCompletionMessageToolCalls, ChatCompletionResponseMessageRaw,
    CompletionUsageRaw, CreateChatCompletionResponseRaw, FinishReason, PromptTokensDetailsRaw,
    RawExtensibleChatCompletionResponse,
};
use crate::settings::LLMSettings;

/// Default `anthropic-version` header value sent with every request.
pub const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

/// Protocol tag carried by [`MessagePart::Opaque`] parts this module produces,
/// so replay stays confined to its own protocol.
const ANTHROPIC_PROTOCOL: &str = "anthropic";

// ---------------------------------------------------------------------------
// HTTP config
// ---------------------------------------------------------------------------

/// Configuration teaching the shared `async_openai` HTTP client to speak the
/// Anthropic Messages protocol: auth goes in `x-api-key` (or `Authorization:
/// Bearer` for an auth-token credential),
/// every request carries `anthropic-version`, and the one POST path the client
/// uses (`/chat/completions`) maps onto `/messages`.
///
/// `api_base` should include the version segment, e.g.
/// `https://api.anthropic.com/v1`.
#[derive(Clone, Debug)]
pub struct AnthropicConfig {
    api_base: String,
    auth: AnthropicAuth,
    version: String,
}

/// How the endpoint authenticates: an API key sent as `x-api-key` (the
/// protocol's canonical scheme), or a bearer token in `Authorization` — the
/// scheme the Anthropic SDK ties to `ANTHROPIC_AUTH_TOKEN`.
#[derive(Clone, Debug)]
enum AnthropicAuth {
    ApiKey(SecretString),
    Bearer(SecretString),
}

impl AnthropicConfig {
    pub fn new(api_base: &str, api_key: &str, version: &str) -> Self {
        Self {
            api_base: api_base.trim_end_matches('/').to_string(),
            auth: AnthropicAuth::ApiKey(SecretString::from(api_key.to_string())),
            version: version.to_string(),
        }
    }

    /// Like [`Self::new`], but authenticating with `Authorization: Bearer`
    /// instead of `x-api-key` — what the Anthropic SDK does with a credential
    /// from `ANTHROPIC_AUTH_TOKEN`.
    pub fn new_bearer(api_base: &str, token: &str, version: &str) -> Self {
        Self {
            api_base: api_base.trim_end_matches('/').to_string(),
            auth: AnthropicAuth::Bearer(SecretString::from(token.to_string())),
            version: version.to_string(),
        }
    }

    pub fn version(&self) -> &str {
        &self.version
    }
}

impl async_openai::config::Config for AnthropicConfig {
    fn headers(&self) -> http::HeaderMap {
        use secrecy::ExposeSecret;
        let mut headers = http::HeaderMap::new();
        // An unencodable credential/version could not authenticate anyway;
        // sending the request without the header surfaces the provider's own
        // auth error instead of panicking here.
        match &self.auth {
            AnthropicAuth::ApiKey(key) => {
                if let Ok(value) = http::HeaderValue::from_str(key.expose_secret()) {
                    headers.insert("x-api-key", value);
                }
            }
            AnthropicAuth::Bearer(token) => {
                if let Ok(value) =
                    http::HeaderValue::from_str(&format!("Bearer {}", token.expose_secret()))
                {
                    headers.insert(http::header::AUTHORIZATION, value);
                }
            }
        }
        if let Ok(value) = http::HeaderValue::from_str(&self.version) {
            headers.insert("anthropic-version", value);
        }
        headers
    }

    fn url(&self, path: &str) -> String {
        // The generic client only ever posts chat completions; on this protocol
        // that endpoint is `/messages`.
        let path = if path == "/chat/completions" {
            "/messages"
        } else {
            path
        };
        format!("{}{}", self.api_base, path)
    }

    fn query(&self) -> Vec<(&str, &str)> {
        vec![]
    }

    fn api_base(&self) -> &str {
        &self.api_base
    }

    fn api_key(&self) -> &SecretString {
        // The generic client only reads this for logging/reuse; either auth
        // scheme's credential is the secret in play.
        match &self.auth {
            AnthropicAuth::ApiKey(secret) | AnthropicAuth::Bearer(secret) => secret,
        }
    }
}

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AnthropicCacheControlRaw {
    Ephemeral,
}
pub type AnthropicCacheControl = WithOtherFields<AnthropicCacheControlRaw>;

impl AnthropicCacheControlRaw {
    pub fn ephemeral() -> AnthropicCacheControl {
        WithOtherFields::new(Self::Ephemeral)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AnthropicImageSourceRaw {
    Base64 { media_type: String, data: String },
    Url { url: String },
}
pub type AnthropicImageSource = WithOtherFields<AnthropicImageSourceRaw>;

impl AnthropicImageSourceRaw {
    /// Chat images arrive as a URL that may be a `data:` URL; Anthropic wants
    /// those spelled out as an explicit base64 source.
    fn from_chat_url(url: &str) -> AnthropicImageSource {
        if let Some(rest) = url.strip_prefix("data:")
            && let Some((meta, data)) = rest.split_once(',')
            && let Some(media_type) = meta.strip_suffix(";base64")
        {
            return WithOtherFields::new(Self::Base64 {
                media_type: media_type.to_string(),
                data: data.to_string(),
            });
        }
        WithOtherFields::new(Self::Url {
            url: url.to_string(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockRaw {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Image {
        source: AnthropicImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
    },
    /// Server-side block kinds this client does not model (e.g. web search
    /// results), captured verbatim so a replayed conversation keeps them.
    #[serde(untagged)]
    Raw(Value),
}
pub type AnthropicContentBlock = WithOtherFields<AnthropicContentBlockRaw>;

impl AnthropicContentBlockRaw {
    fn text(text: String, cache_control: Option<AnthropicCacheControl>) -> AnthropicContentBlock {
        WithOtherFields::new(Self::Text {
            text,
            cache_control,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct AnthropicMessageRaw {
    pub role: AnthropicRole,
    pub content: Vec<AnthropicContentBlock>,
}
pub type AnthropicMessage = WithOtherFields<AnthropicMessageRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct AnthropicToolRaw {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicCacheControl>,
}
pub type AnthropicTool = WithOtherFields<AnthropicToolRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AnthropicToolChoiceRaw {
    Auto {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    Any {
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
    None,
    Tool {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        disable_parallel_tool_use: Option<bool>,
    },
}
pub type AnthropicToolChoice = WithOtherFields<AnthropicToolChoiceRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum AnthropicThinkingRaw {
    Enabled { budget_tokens: u32 },
    Disabled,
}
pub type AnthropicThinking = WithOtherFields<AnthropicThinkingRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct AnthropicMessagesRequestRaw {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    /// Mandatory on this protocol, unlike chat completion's optional
    /// `max_completion_tokens`.
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<AnthropicContentBlock>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<AnthropicThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}
pub type AnthropicMessagesRequest = WithOtherFields<AnthropicMessagesRequestRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct AnthropicUsageRaw {
    /// Prompt tokens *excluding* cache reads/writes — narrower than the chat
    /// protocol's `prompt_tokens`, which the conversion widens back.
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}
pub type AnthropicUsage = WithOtherFields<AnthropicUsageRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    PauseTurn,
    Refusal,
    #[serde(untagged)]
    Other(String),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct AnthropicMessagesResponseRaw {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub role: AnthropicRole,
    pub model: String,
    pub content: Vec<AnthropicContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<AnthropicStopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    #[serde(default)]
    pub usage: AnthropicUsage,
}
pub type AnthropicMessagesResponse = WithOtherFields<AnthropicMessagesResponseRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct AnthropicErrorBody {
    #[serde(rename = "type", default)]
    pub kind: String,
    #[serde(default)]
    pub message: String,
}

// ---------------------------------------------------------------------------
// Stream events
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct AnthropicMessageDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<AnthropicStopReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicContentBlockDelta {
    TextDelta {
        text: String,
    },
    InputJsonDelta {
        partial_json: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    SignatureDelta {
        signature: String,
    },
    #[serde(other)]
    Unknown,
}

/// One SSE event of a streamed Messages response. These are transient
/// aggregation inputs, so unlike the wire types above they carry no
/// `WithOtherFields` wrapper.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnthropicStreamEvent {
    MessageStart {
        message: AnthropicMessagesResponse,
    },
    ContentBlockStart {
        index: usize,
        content_block: AnthropicContentBlock,
    },
    ContentBlockDelta {
        index: usize,
        delta: AnthropicContentBlockDelta,
    },
    ContentBlockStop {
        index: usize,
    },
    MessageDelta {
        delta: AnthropicMessageDelta,
        #[serde(default)]
        usage: Option<AnthropicUsage>,
    },
    MessageStop,
    Ping,
    Error {
        error: AnthropicErrorBody,
    },
    #[serde(other)]
    Unknown,
}

/// Folds the Messages SSE event stream back into one full
/// [`AnthropicMessagesResponse`], mirroring what the chat protocol's chunk
/// aggregation does for chat completions.
#[derive(Debug, Default)]
pub struct AnthropicStreamAccumulator {
    message: Option<AnthropicMessagesResponse>,
    blocks: Vec<AnthropicContentBlock>,
    /// Per-index accumulated `input_json_delta` fragments for tool_use blocks;
    /// parsed once at [`Self::finish`].
    pending_json: Vec<String>,
    stop_reason: Option<AnthropicStopReason>,
    stop_sequence: Option<String>,
    usage_delta: Option<AnthropicUsage>,
}

impl AnthropicStreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, event: AnthropicStreamEvent) -> Result<(), LLMYError> {
        match event {
            AnthropicStreamEvent::MessageStart { message } => self.message = Some(message),
            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                if self.blocks.len() <= index {
                    self.blocks.resize_with(index + 1, || {
                        AnthropicContentBlockRaw::text(String::new(), None)
                    });
                    self.pending_json.resize_with(index + 1, String::new);
                }
                self.blocks[index] = content_block;
            }
            AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                if let AnthropicContentBlockDelta::InputJsonDelta { partial_json } = &delta {
                    let slot = self.pending_json.get_mut(index).ok_or_else(|| {
                        eyre!("input_json_delta for unopened content block {}", index)
                    })?;
                    slot.push_str(partial_json);
                    return Ok(());
                }
                let block = self
                    .blocks
                    .get_mut(index)
                    .ok_or_else(|| eyre!("delta for unopened content block {}", index))?;
                match (&mut block.inner, delta) {
                    (
                        AnthropicContentBlockRaw::Text { text, .. },
                        AnthropicContentBlockDelta::TextDelta { text: chunk },
                    ) => text.push_str(&chunk),
                    (
                        AnthropicContentBlockRaw::Thinking { thinking, .. },
                        AnthropicContentBlockDelta::ThinkingDelta { thinking: chunk },
                    ) => thinking.push_str(&chunk),
                    (
                        AnthropicContentBlockRaw::Thinking { signature, .. },
                        AnthropicContentBlockDelta::SignatureDelta { signature: sig },
                    ) => *signature = Some(sig),
                    // An unknown or mismatched delta only decorates a block; it
                    // must not kill the whole stream.
                    _ => {}
                }
            }
            AnthropicStreamEvent::MessageDelta { delta, usage } => {
                if delta.stop_reason.is_some() {
                    self.stop_reason = delta.stop_reason;
                }
                if delta.stop_sequence.is_some() {
                    self.stop_sequence = delta.stop_sequence;
                }
                if usage.is_some() {
                    self.usage_delta = usage;
                }
            }
            AnthropicStreamEvent::Error { error } => {
                return Err(
                    eyre!("anthropic stream error ({}): {}", error.kind, error.message).into(),
                );
            }
            AnthropicStreamEvent::ContentBlockStop { .. }
            | AnthropicStreamEvent::MessageStop
            | AnthropicStreamEvent::Ping
            | AnthropicStreamEvent::Unknown => {}
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<AnthropicMessagesResponse, LLMYError> {
        let mut message = self
            .message
            .ok_or_else(|| eyre!("anthropic stream ended without a message_start event"))?;
        for (idx, block) in self.blocks.iter_mut().enumerate() {
            if let AnthropicContentBlockRaw::ToolUse { input, .. } = &mut block.inner {
                let json = self.pending_json.get(idx).map(|s| s.trim()).unwrap_or("");
                if !json.is_empty() {
                    *input = serde_json::from_str(json).map_err(|e| {
                        eyre!(
                            "streamed tool_use input is not valid JSON ({}): {}",
                            e,
                            json
                        )
                    })?;
                }
            }
        }
        message.inner.content = self.blocks;
        if self.stop_reason.is_some() {
            message.inner.stop_reason = self.stop_reason;
        }
        if self.stop_sequence.is_some() {
            message.inner.stop_sequence = self.stop_sequence;
        }
        if let Some(delta) = self.usage_delta {
            let usage = &mut message.inner.usage.inner;
            usage.output_tokens = delta.inner.output_tokens;
            if delta.inner.input_tokens > 0 {
                usage.input_tokens = delta.inner.input_tokens;
            }
            if delta.inner.cache_read_input_tokens.is_some() {
                usage.cache_read_input_tokens = delta.inner.cache_read_input_tokens;
            }
            if delta.inner.cache_creation_input_tokens.is_some() {
                usage.cache_creation_input_tokens = delta.inner.cache_creation_input_tokens;
            }
        }
        Ok(message)
    }
}

// ---------------------------------------------------------------------------
// Chat request -> Anthropic request
// ---------------------------------------------------------------------------

impl AnthropicMessagesRequestRaw {
    /// Convert a chat-completion request into the Messages wire form.
    ///
    /// `default_max_output_tokens` fills the mandatory `max_tokens` field when
    /// the chat request does not bound completion length itself (callers pass
    /// the model's configured max output tokens).
    pub fn from_chat(
        req: &RawExtensibleChatCompletionRequest,
        default_max_output_tokens: u32,
    ) -> Result<AnthropicMessagesRequest, LLMYError> {
        let raw: &CreateChatCompletionRequestRaw = req;
        Self::reject_unsupported(raw)?;

        #[allow(deprecated)]
        let max_tokens = raw
            .max_completion_tokens
            .or(raw.max_tokens)
            .unwrap_or(default_max_output_tokens);

        let mut system: Vec<AnthropicContentBlock> = Vec::new();
        let mut messages: Vec<AnthropicMessage> = Vec::new();
        for msg in &raw.messages {
            match &msg.inner {
                ChatCompletionRequestMessageRaw::System(m) => {
                    Self::system_blocks(&m.inner.content, &mut system)
                }
                ChatCompletionRequestMessageRaw::Developer(m) => {
                    Self::developer_blocks(&m.inner.content, &mut system)
                }
                ChatCompletionRequestMessageRaw::User(m) => {
                    Self::user_blocks(&m.inner.content, &mut messages)?
                }
                ChatCompletionRequestMessageRaw::Assistant(m) => {
                    Self::assistant_blocks(&m.inner, &mut messages)?
                }
                ChatCompletionRequestMessageRaw::Tool(m) => {
                    let cache_control = m
                        .inner
                        .content
                        .has_cache_breakpoint()
                        .then(AnthropicCacheControlRaw::ephemeral);
                    Self::push_block(
                        &mut messages,
                        AnthropicRole::User,
                        WithOtherFields::new(AnthropicContentBlockRaw::ToolResult {
                            tool_use_id: m.inner.tool_call_id.clone(),
                            content: Self::tool_text(&m.inner.content),
                            is_error: None,
                            cache_control,
                        }),
                    );
                }
                ChatCompletionRequestMessageRaw::Function(_) => {
                    return Err(eyre!(
                        "legacy `function` messages are not supported on the anthropic protocol"
                    )
                    .into());
                }
            }
        }
        if messages.is_empty() {
            return Err(
                eyre!("the anthropic protocol needs at least one user/assistant message").into(),
            );
        }

        let tools = raw.tools.as_deref().map(Self::tools).transpose()?;
        let disable_parallel = matches!(raw.parallel_tool_calls, Some(false)).then_some(true);
        let tool_choice = Self::tool_choice(raw.tool_choice.as_ref(), disable_parallel)?;
        let stop_sequences = raw.stop.as_ref().map(|stop| match stop {
            StopConfiguration::String(s) => vec![s.clone()],
            StopConfiguration::StringArray(v) => v.clone(),
        });
        let thinking = Self::thinking(raw.reasoning_effort.as_ref(), max_tokens);

        Ok(WithOtherFields::new(Self {
            model: raw.model.clone(),
            messages,
            max_tokens,
            system: (!system.is_empty()).then_some(system),
            tools,
            tool_choice,
            temperature: raw.temperature,
            top_p: raw.top_p,
            stop_sequences,
            thinking,
            stream: raw.stream,
        }))
    }

    /// Chat-request fields this protocol has no slot for. Refusing beats
    /// silently changing the request's meaning.
    fn reject_unsupported(raw: &CreateChatCompletionRequestRaw) -> Result<(), LLMYError> {
        let mut unsupported = Vec::new();
        if raw.presence_penalty.is_some() {
            unsupported.push("presence_penalty");
        }
        if raw.frequency_penalty.is_some() {
            unsupported.push("frequency_penalty");
        }
        if raw.logit_bias.is_some() {
            unsupported.push("logit_bias");
        }
        if raw.logprobs.is_some() || raw.top_logprobs.is_some() {
            unsupported.push("logprobs");
        }
        if raw.n.is_some_and(|n| n > 1) {
            unsupported.push("n > 1");
        }
        if raw.modalities.is_some() || raw.audio.is_some() {
            unsupported.push("audio output");
        }
        if raw.prediction.is_some() {
            unsupported.push("prediction");
        }
        if raw.web_search_options.is_some() {
            unsupported.push("web_search_options");
        }
        if raw.response_format.is_some() {
            unsupported.push("response_format");
        }
        #[allow(deprecated)]
        if raw.functions.is_some() || raw.function_call.is_some() {
            unsupported.push("functions/function_call");
        }
        if unsupported.is_empty() {
            Ok(())
        } else {
            Err(eyre!(
                "the anthropic protocol does not support: {}",
                unsupported.join(", ")
            )
            .into())
        }
    }

    /// Append a block to the message list, merging into the previous message
    /// when it has the same role — the Messages API wants strictly alternating
    /// roles, while chat history can hold e.g. several `tool` results in a row.
    fn push_block(
        messages: &mut Vec<AnthropicMessage>,
        role: AnthropicRole,
        block: AnthropicContentBlock,
    ) {
        if let Some(last) = messages.last_mut()
            && last.inner.role == role
        {
            last.inner.content.push(block);
            return;
        }
        messages.push(WithOtherFields::new(AnthropicMessageRaw {
            role,
            content: vec![block],
        }));
    }

    fn cache_control(breakpoint: &Option<PromptCacheBreakpoint>) -> Option<AnthropicCacheControl> {
        breakpoint
            .as_ref()
            .map(|_| AnthropicCacheControlRaw::ephemeral())
    }

    fn system_blocks(
        content: &ChatCompletionRequestSystemMessageContent,
        out: &mut Vec<AnthropicContentBlock>,
    ) {
        match content {
            ChatCompletionRequestSystemMessageContent::Text(text) => {
                out.push(AnthropicContentBlockRaw::text(text.clone(), None))
            }
            ChatCompletionRequestSystemMessageContent::Array(parts) => {
                for part in parts {
                    let ChatCompletionRequestSystemMessageContentPartRaw::Text(text) = &part.inner;
                    out.push(AnthropicContentBlockRaw::text(
                        text.inner.text.clone(),
                        Self::cache_control(&text.inner.prompt_cache_breakpoint),
                    ));
                }
            }
        }
    }

    fn developer_blocks(
        content: &ChatCompletionRequestDeveloperMessageContent,
        out: &mut Vec<AnthropicContentBlock>,
    ) {
        match content {
            ChatCompletionRequestDeveloperMessageContent::Text(text) => {
                out.push(AnthropicContentBlockRaw::text(text.clone(), None))
            }
            ChatCompletionRequestDeveloperMessageContent::Array(parts) => {
                for part in parts {
                    let ChatCompletionRequestDeveloperMessageContentPartRaw::Text(text) =
                        &part.inner;
                    out.push(AnthropicContentBlockRaw::text(
                        text.inner.text.clone(),
                        Self::cache_control(&text.inner.prompt_cache_breakpoint),
                    ));
                }
            }
        }
    }

    fn user_blocks(
        content: &ChatCompletionRequestUserMessageContent,
        messages: &mut Vec<AnthropicMessage>,
    ) -> Result<(), LLMYError> {
        match content {
            ChatCompletionRequestUserMessageContent::Text(text) => Self::push_block(
                messages,
                AnthropicRole::User,
                AnthropicContentBlockRaw::text(text.clone(), None),
            ),
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                for part in parts {
                    let block = match &part.inner {
                        ChatCompletionRequestUserMessageContentPartRaw::Text(text) => {
                            AnthropicContentBlockRaw::text(
                                text.inner.text.clone(),
                                Self::cache_control(&text.inner.prompt_cache_breakpoint),
                            )
                        }
                        ChatCompletionRequestUserMessageContentPartRaw::ImageUrl(image) => {
                            WithOtherFields::new(AnthropicContentBlockRaw::Image {
                                source: AnthropicImageSourceRaw::from_chat_url(
                                    &image.inner.image_url.inner.url,
                                ),
                                cache_control: Self::cache_control(
                                    &image.inner.prompt_cache_breakpoint,
                                ),
                            })
                        }
                        ChatCompletionRequestUserMessageContentPartRaw::InputAudio(_) => {
                            return Err(eyre!(
                                "audio input parts are not supported on the anthropic protocol"
                            )
                            .into());
                        }
                        ChatCompletionRequestUserMessageContentPartRaw::File(_) => {
                            return Err(eyre!(
                                "file input parts are not supported on the anthropic protocol"
                            )
                            .into());
                        }
                    };
                    Self::push_block(messages, AnthropicRole::User, block);
                }
            }
        }
        Ok(())
    }

    fn assistant_blocks(
        msg: &ChatCompletionRequestAssistantMessageRaw,
        messages: &mut Vec<AnthropicMessage>,
    ) -> Result<(), LLMYError> {
        match &msg.content {
            Some(ChatCompletionRequestAssistantMessageContent::Text(text)) => {
                if !text.is_empty() {
                    Self::push_block(
                        messages,
                        AnthropicRole::Assistant,
                        AnthropicContentBlockRaw::text(text.clone(), None),
                    );
                }
            }
            Some(ChatCompletionRequestAssistantMessageContent::Array(parts)) => {
                for part in parts {
                    let (text, breakpoint) = match &part.inner {
                        ChatCompletionRequestAssistantMessageContentPartRaw::Text(text) => {
                            (&text.inner.text, &text.inner.prompt_cache_breakpoint)
                        }
                        ChatCompletionRequestAssistantMessageContentPartRaw::Refusal(refusal) => (
                            &refusal.inner.refusal,
                            &refusal.inner.prompt_cache_breakpoint,
                        ),
                    };
                    Self::push_block(
                        messages,
                        AnthropicRole::Assistant,
                        AnthropicContentBlockRaw::text(
                            text.clone(),
                            Self::cache_control(breakpoint),
                        ),
                    );
                }
            }
            None => {}
        }
        for tool_call in msg.tool_calls.iter().flatten() {
            match &tool_call.inner {
                ChatCompletionMessageToolCallsRaw::Function(call) => {
                    let arguments = call.inner.function.inner.arguments.trim();
                    let input: Value = if arguments.is_empty() {
                        Value::Object(Default::default())
                    } else {
                        serde_json::from_str(arguments).map_err(|e| {
                            eyre!(
                                "tool call {} carries non-JSON arguments: {}",
                                call.inner.id,
                                e
                            )
                        })?
                    };
                    Self::push_block(
                        messages,
                        AnthropicRole::Assistant,
                        WithOtherFields::new(AnthropicContentBlockRaw::ToolUse {
                            id: call.inner.id.clone(),
                            name: call.inner.function.inner.name.clone(),
                            input,
                            cache_control: None,
                        }),
                    );
                }
                ChatCompletionMessageToolCallsRaw::Custom(_) => {
                    return Err(eyre!(
                        "custom tool calls are not supported on the anthropic protocol"
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    fn tool_text(content: &ChatCompletionRequestToolMessageContent) -> String {
        match content {
            ChatCompletionRequestToolMessageContent::Text(text) => text.clone(),
            ChatCompletionRequestToolMessageContent::Array(parts) => parts
                .iter()
                .map(|part| {
                    let ChatCompletionRequestToolMessageContentPartRaw::Text(text) = &part.inner;
                    text.inner.text.as_str()
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    fn tools(tools: &[ChatCompletionTools]) -> Result<Vec<AnthropicTool>, LLMYError> {
        tools
            .iter()
            .map(|tool| match &tool.inner {
                ChatCompletionToolsRaw::Function(function) => {
                    let function = &function.inner.function.inner;
                    Ok(WithOtherFields::new(AnthropicToolRaw {
                        name: function.name.clone(),
                        description: function.description.clone(),
                        input_schema: function
                            .parameters
                            .clone()
                            .unwrap_or(Value::Object(Default::default())),
                        cache_control: None,
                    }))
                }
                ChatCompletionToolsRaw::Custom(custom) => Err(eyre!(
                    "custom tool {:?} is not supported on the anthropic protocol",
                    custom.inner.custom.inner.name
                )
                .into()),
            })
            .collect()
    }

    fn tool_choice(
        choice: Option<&ChatCompletionToolChoiceOption>,
        disable_parallel_tool_use: Option<bool>,
    ) -> Result<Option<AnthropicToolChoice>, LLMYError> {
        let Some(choice) = choice else {
            // `parallel_tool_calls: false` has no home of its own; it rides on
            // the (otherwise default) auto tool choice.
            return Ok(disable_parallel_tool_use.map(|disable| {
                WithOtherFields::new(AnthropicToolChoiceRaw::Auto {
                    disable_parallel_tool_use: Some(disable),
                })
            }));
        };
        let mapped = match &choice.inner {
            ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Auto) => {
                AnthropicToolChoiceRaw::Auto {
                    disable_parallel_tool_use,
                }
            }
            ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Required) => {
                AnthropicToolChoiceRaw::Any {
                    disable_parallel_tool_use,
                }
            }
            ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::None) => {
                AnthropicToolChoiceRaw::None
            }
            ChatCompletionToolChoiceOptionRaw::Function(named) => AnthropicToolChoiceRaw::Tool {
                name: named.inner.function.inner.name.clone(),
                disable_parallel_tool_use,
            },
            ChatCompletionToolChoiceOptionRaw::Custom(_)
            | ChatCompletionToolChoiceOptionRaw::AllowedTools(_) => {
                return Err(eyre!(
                    "custom/allowed-tools tool choices are not supported on the anthropic protocol"
                )
                .into());
            }
        };
        Ok(Some(WithOtherFields::new(mapped)))
    }

    /// Map chat `reasoning_effort` onto the `thinking` budget: `none` disables
    /// extended thinking, everything else buys a share of `max_tokens`
    /// (minimal 10% … xhigh 90%), clamped into the API's `1024 <= budget <
    /// max_tokens` window — staying under `max_tokens` wins when both cannot
    /// hold.
    fn thinking(effort: Option<&ReasoningEffort>, max_tokens: u32) -> Option<AnthropicThinking> {
        let effort = effort?;
        let share = match effort {
            ReasoningEffort::None => {
                return Some(WithOtherFields::new(AnthropicThinkingRaw::Disabled));
            }
            ReasoningEffort::Minimal => 0.10,
            ReasoningEffort::Low => 0.25,
            ReasoningEffort::Medium => 0.50,
            ReasoningEffort::High => 0.75,
            ReasoningEffort::Xhigh => 0.90,
        };
        let budget = ((max_tokens as f64) * share) as u32;
        let budget = budget.max(1024).min(max_tokens.saturating_sub(1)).max(1);
        Some(WithOtherFields::new(AnthropicThinkingRaw::Enabled {
            budget_tokens: budget,
        }))
    }

    /// Build a native prompt request straight from strings — nothing is forced
    /// through the chat-completion shape on its way to the wire.
    pub fn from_prompt(
        model: &str,
        sys_msg: &str,
        user_msg: &str,
        settings: &LLMSettings,
        default_max_output_tokens: u32,
    ) -> Result<AnthropicMessagesRequest, LLMYError> {
        let mut conversation = Vec::new();
        if !sys_msg.is_empty() {
            conversation.push(Message::system(sys_msg));
        }
        conversation.push(Message::user(user_msg));
        Self::from_conversation(
            model,
            &conversation,
            None,
            settings,
            default_max_output_tokens,
        )
    }

    /// Build the native request straight from a protocol-neutral conversation:
    /// typed parts map onto their own blocks — signed thinking replays
    /// verbatim — so nothing is forced through the chat shape. The caller's
    /// settings map onto the protocol's own fields; a setting with no slot
    /// here (`presence_penalty`) is refused rather than dropped.
    pub fn from_conversation(
        model: &str,
        conversation: &[Message],
        tools: Option<&[ChatCompletionTools]>,
        settings: &LLMSettings,
        default_max_output_tokens: u32,
    ) -> Result<AnthropicMessagesRequest, LLMYError> {
        if settings.llm_presence_penalty.is_some() {
            return Err(eyre!(
                "the anthropic protocol does not support presence_penalty; unset it to use this backend"
            )
            .into());
        }
        let max_tokens = settings
            .llm_max_completion_tokens
            .unwrap_or(default_max_output_tokens);
        let tool_choice = match settings.llm_tool_choice.as_ref() {
            Some(choice) => Self::tool_choice(Some(&choice.0), None)?,
            None => None,
        };
        let tools = tools.map(Self::tools).transpose()?;

        let mut system: Vec<AnthropicContentBlock> = Vec::new();
        let mut messages: Vec<AnthropicMessage> = Vec::new();
        for message in conversation {
            Self::conversation_message(message, &mut system, &mut messages)?;
        }
        if messages.is_empty() {
            return Err(
                eyre!("the anthropic protocol needs at least one user/assistant message").into(),
            );
        }

        Ok(WithOtherFields::new(Self {
            model: model.to_string(),
            messages,
            max_tokens,
            system: (!system.is_empty()).then_some(system),
            tools,
            tool_choice,
            temperature: settings.llm_temperature,
            top_p: settings.top_p,
            stop_sequences: None,
            thinking: Self::thinking(settings.reasoning_effort.as_ref().map(|r| &r.0), max_tokens),
            stream: None,
        }))
    }

    /// One neutral message into native blocks. The message-level breakpoint
    /// becomes `cache_control` on the message's last block; parts belonging to
    /// another protocol (responses reasoning, foreign opaque parts) are
    /// skipped — their payloads are only valid where they were issued.
    fn conversation_message(
        message: &Message,
        system: &mut Vec<AnthropicContentBlock>,
        messages: &mut Vec<AnthropicMessage>,
    ) -> Result<(), LLMYError> {
        if message.role == MessageRole::System {
            let cache_control = message
                .cache_breakpoint
                .then(AnthropicCacheControlRaw::ephemeral);
            system.push(AnthropicContentBlockRaw::text(
                message.text(),
                cache_control,
            ));
            return Ok(());
        }
        let role = if message.role == MessageRole::Assistant {
            AnthropicRole::Assistant
        } else {
            AnthropicRole::User
        };
        let mut blocks: Vec<AnthropicContentBlock> = Vec::new();
        for part in &message.parts {
            match part {
                MessagePart::Text { text } => {
                    blocks.push(AnthropicContentBlockRaw::text(text.clone(), None))
                }
                MessagePart::Image { url } => {
                    blocks.push(WithOtherFields::new(AnthropicContentBlockRaw::Image {
                        source: AnthropicImageSourceRaw::from_chat_url(url),
                        cache_control: None,
                    }))
                }
                MessagePart::ToolCall {
                    id,
                    name,
                    arguments,
                    extra,
                } => {
                    let arguments = arguments.trim();
                    let input: Value = if arguments.is_empty() {
                        Value::Object(Default::default())
                    } else {
                        serde_json::from_str(arguments).map_err(|e| {
                            eyre!("tool call {} carries non-JSON arguments: {}", id, e)
                        })?
                    };
                    let mut block = WithOtherFields::new(AnthropicContentBlockRaw::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input,
                        cache_control: None,
                    });
                    block.other = extra.clone();
                    blocks.push(block);
                }
                MessagePart::ToolResult { id, content } => {
                    blocks.push(WithOtherFields::new(AnthropicContentBlockRaw::ToolResult {
                        tool_use_id: id.clone(),
                        content: content.clone(),
                        is_error: None,
                        cache_control: None,
                    }))
                }
                MessagePart::Thinking {
                    thinking,
                    signature,
                } => blocks.push(WithOtherFields::new(AnthropicContentBlockRaw::Thinking {
                    thinking: thinking.clone(),
                    signature: signature.clone(),
                })),
                MessagePart::RedactedThinking { data } => blocks.push(WithOtherFields::new(
                    AnthropicContentBlockRaw::RedactedThinking { data: data.clone() },
                )),
                MessagePart::Opaque { protocol, value } if protocol == ANTHROPIC_PROTOCOL => blocks
                    .push(WithOtherFields::new(AnthropicContentBlockRaw::Raw(
                        value.clone(),
                    ))),
                MessagePart::Reasoning { .. } | MessagePart::Opaque { .. } => {}
            }
        }
        if message.cache_breakpoint
            && let Some(last) = blocks.last_mut()
        {
            match &mut last.inner {
                AnthropicContentBlockRaw::Text { cache_control, .. }
                | AnthropicContentBlockRaw::Image { cache_control, .. }
                | AnthropicContentBlockRaw::ToolUse { cache_control, .. }
                | AnthropicContentBlockRaw::ToolResult { cache_control, .. } => {
                    *cache_control = Some(AnthropicCacheControlRaw::ephemeral())
                }
                _ => {}
            }
        }
        for block in blocks {
            Self::push_block(messages, role, block);
        }
        Ok(())
    }

    /// Prompt-text rendering of this request (system + messages), tagged the
    /// same way as the chat renderer so debug records read uniformly whatever
    /// protocol went over the wire.
    pub fn conversation_text(&self) -> String {
        let mut out = String::new();
        if let Some(system) = &self.system {
            out.push_str("<SYSTEM>\n");
            for block in system {
                Self::render_block(block, &mut out);
            }
            out.push_str("\n</SYSTEM>\n");
        }
        for message in &self.messages {
            let role = match message.inner.role {
                AnthropicRole::User => "USER",
                AnthropicRole::Assistant => "ASSISTANT",
            };
            out.push_str(&format!("<{}>\n", role));
            for block in &message.inner.content {
                Self::render_block(block, &mut out);
            }
            out.push_str(&format!("\n</{}>\n", role));
        }
        out
    }

    fn render_block(block: &AnthropicContentBlock, out: &mut String) {
        match &block.inner {
            AnthropicContentBlockRaw::Text { text, .. } => out.push_str(text),
            AnthropicContentBlockRaw::Thinking { thinking, .. } => {
                out.push_str("<thinking>");
                out.push_str(thinking);
                out.push_str("</thinking>\n");
            }
            AnthropicContentBlockRaw::ToolUse {
                id, name, input, ..
            } => out.push_str(&format!(
                "<toolcall name=\"{}\" id=\"{}\">\n{}\n</toolcall>",
                name, id, input
            )),
            AnthropicContentBlockRaw::ToolResult {
                tool_use_id,
                content,
                ..
            } => out.push_str(&format!(
                "<toolresult id=\"{}\">\n{}\n</toolresult>",
                tool_use_id, content
            )),
            AnthropicContentBlockRaw::Image { source, .. } => match &source.inner {
                AnthropicImageSourceRaw::Url { url } => {
                    out.push_str(&format!("<img url=\"{}\"/>", url))
                }
                AnthropicImageSourceRaw::Base64 { media_type, .. } => {
                    out.push_str(&format!("<img media_type=\"{}\"/>", media_type))
                }
            },
            AnthropicContentBlockRaw::RedactedThinking { .. }
            | AnthropicContentBlockRaw::Raw(_) => {}
        }
    }

    /// Tool definitions rendered for the folder debug view.
    pub fn tools_text(&self) -> String {
        self.tools
            .iter()
            .flatten()
            .map(|tool| {
                format!(
                    "<tool name=\"{}\", description=\"{}\">\n{}\n</tool>",
                    tool.inner.name,
                    tool.inner.description.clone().unwrap_or_default(),
                    serde_json::to_string_pretty(&tool.inner.input_schema).unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Raw text for token estimation: the conversation plus the tool schemas.
    pub fn estimate_text(&self) -> String {
        let mut out = self.conversation_text();
        for tool in self.tools.iter().flatten() {
            out.push('\n');
            out.push_str(&serde_json::to_string(tool).unwrap_or_default());
        }
        out
    }

    /// Fold this native request back into the chat-completion form — the hub
    /// every cross-protocol conversion routes through, so a request built for
    /// this protocol can still be sent over another backend. System blocks
    /// become system messages, merged tool results split back out into `tool`
    /// messages, `cache_control` markers become breakpoints, and the thinking
    /// budget maps back onto `reasoning_effort` by its share of `max_tokens`.
    pub fn into_chat_request(self) -> Result<RawExtensibleChatCompletionRequest, LLMYError> {
        let max_tokens = self.max_tokens;
        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();
        for block in self.system.into_iter().flatten() {
            match block.inner {
                AnthropicContentBlockRaw::Text {
                    text,
                    cache_control,
                } => {
                    let mut msg = ChatCompletionRequestSystemMessageRaw::new_text(text);
                    if cache_control.is_some() {
                        msg.inner.content.toggle_cache_breakpoint(true);
                    }
                    messages.push(WithOtherFields::new(
                        ChatCompletionRequestMessageRaw::System(msg),
                    ));
                }
                other => {
                    return Err(
                        eyre!("a system block has no chat-completion form: {:?}", other).into(),
                    );
                }
            }
        }
        for message in self.messages {
            let message = message.into_inner();
            match message.role {
                AnthropicRole::User => Self::user_chat_messages(message.content, &mut messages)?,
                AnthropicRole::Assistant => {
                    Self::assistant_chat_message(message.content, &mut messages)?
                }
            }
        }

        let tools = self.tools.map(|tools| {
            tools
                .into_iter()
                .map(|tool| {
                    // `cache_control` is a caching hint with no chat slot;
                    // dropping it never changes what the model sees.
                    let tool = tool.into_inner();
                    WithOtherFields::new(ChatCompletionToolsRaw::Function(WithOtherFields::new(
                        ChatCompletionToolRaw {
                            function: WithOtherFields::new(FunctionObjectRaw {
                                name: tool.name,
                                description: tool.description,
                                parameters: Some(tool.input_schema),
                                strict: None,
                            }),
                        },
                    )))
                })
                .collect::<Vec<_>>()
        });
        let (tool_choice, parallel_tool_calls) = Self::chat_tool_choice(self.tool_choice);
        let reasoning_effort = self.thinking.map(|thinking| match thinking.into_inner() {
            AnthropicThinkingRaw::Disabled => ReasoningEffort::None,
            AnthropicThinkingRaw::Enabled { budget_tokens } => {
                Self::effort_of_budget(budget_tokens, max_tokens)
            }
        });

        let raw = CreateChatCompletionRequestRaw {
            messages,
            model: self.model,
            reasoning_effort,
            max_completion_tokens: Some(max_tokens),
            stream: self.stream,
            stop: self.stop_sequences.map(StopConfiguration::StringArray),
            temperature: self.temperature,
            top_p: self.top_p,
            tools,
            tool_choice,
            parallel_tool_calls,
            ..Default::default()
        };
        Ok(RawExtensibleChatCompletionRequest::new(raw))
    }

    /// Split one native user message back into chat messages: content blocks
    /// gather into a user message, and each `tool_result` splits out into its
    /// own `tool` message (the reverse of [`Self::push_block`]'s merging).
    fn user_chat_messages(
        blocks: Vec<AnthropicContentBlock>,
        out: &mut Vec<ChatCompletionRequestMessage>,
    ) -> Result<(), LLMYError> {
        let mut parts: Vec<ChatCompletionRequestUserMessageContentPart> = Vec::new();
        for block in blocks {
            match block.inner {
                AnthropicContentBlockRaw::Text {
                    text,
                    cache_control,
                } => parts.push(WithOtherFields::new(
                    ChatCompletionRequestUserMessageContentPartRaw::Text(WithOtherFields::new(
                        ChatCompletionRequestMessageContentPartTextRaw {
                            text,
                            prompt_cache_breakpoint: cache_control
                                .map(|_| PromptCacheBreakpointRaw::explicit()),
                        },
                    )),
                )),
                AnthropicContentBlockRaw::Image {
                    source,
                    cache_control,
                } => {
                    let url = match source.into_inner() {
                        AnthropicImageSourceRaw::Url { url } => url,
                        AnthropicImageSourceRaw::Base64 { media_type, data } => {
                            format!("data:{};base64,{}", media_type, data)
                        }
                    };
                    parts.push(WithOtherFields::new(
                        ChatCompletionRequestUserMessageContentPartRaw::ImageUrl(
                            WithOtherFields::new(ChatCompletionRequestMessageContentPartImageRaw {
                                image_url: WithOtherFields::new(ImageUrlRaw { url, detail: None }),
                                prompt_cache_breakpoint: cache_control
                                    .map(|_| PromptCacheBreakpointRaw::explicit()),
                            }),
                        ),
                    ));
                }
                AnthropicContentBlockRaw::ToolResult {
                    tool_use_id,
                    content,
                    cache_control,
                    ..
                } => {
                    Self::flush_user_parts(&mut parts, out);
                    let mut msg =
                        ChatCompletionRequestToolMessageRaw::new_text(content, tool_use_id);
                    if cache_control.is_some() {
                        msg.inner.content.toggle_cache_breakpoint(true);
                    }
                    out.push(WithOtherFields::new(ChatCompletionRequestMessageRaw::Tool(
                        msg,
                    )));
                }
                other @ (AnthropicContentBlockRaw::ToolUse { .. }
                | AnthropicContentBlockRaw::Thinking { .. }
                | AnthropicContentBlockRaw::RedactedThinking { .. }
                | AnthropicContentBlockRaw::Raw(_)) => {
                    return Err(eyre!(
                        "a user message block has no chat-completion form: {:?}",
                        other
                    )
                    .into());
                }
            }
        }
        Self::flush_user_parts(&mut parts, out);
        Ok(())
    }

    fn flush_user_parts(
        parts: &mut Vec<ChatCompletionRequestUserMessageContentPart>,
        out: &mut Vec<ChatCompletionRequestMessage>,
    ) {
        if parts.is_empty() {
            return;
        }
        let mut content = ChatCompletionRequestUserMessageContent::Array(std::mem::take(parts));
        content.compact();
        out.push(WithOtherFields::new(ChatCompletionRequestMessageRaw::User(
            WithOtherFields::new(ChatCompletionRequestUserMessageRaw {
                content,
                name: None,
            }),
        )));
    }

    fn assistant_chat_message(
        blocks: Vec<AnthropicContentBlock>,
        out: &mut Vec<ChatCompletionRequestMessage>,
    ) -> Result<(), LLMYError> {
        let mut parts: Vec<ChatCompletionRequestAssistantMessageContentPart> = Vec::new();
        let mut tool_calls: Vec<ChatCompletionMessageToolCalls> = Vec::new();
        let mut reasoning = String::new();
        for block in blocks {
            match block.inner {
                AnthropicContentBlockRaw::Text {
                    text,
                    cache_control,
                } => parts.push(WithOtherFields::new(
                    ChatCompletionRequestAssistantMessageContentPartRaw::Text(
                        WithOtherFields::new(ChatCompletionRequestMessageContentPartTextRaw {
                            text,
                            prompt_cache_breakpoint: cache_control
                                .map(|_| PromptCacheBreakpointRaw::explicit()),
                        }),
                    ),
                )),
                AnthropicContentBlockRaw::ToolUse {
                    id, name, input, ..
                } => tool_calls.push(WithOtherFields::new(
                    ChatCompletionMessageToolCallsRaw::Function(WithOtherFields::new(
                        ChatCompletionMessageToolCallRaw {
                            id,
                            function: WithOtherFields::new(FunctionCallRaw {
                                name,
                                arguments: input.to_string(),
                            }),
                        },
                    )),
                )),
                // Thinking lands in the `reasoning_content` extra, mirroring
                // how responses fold it; redacted thinking is opaque by design
                // and has nothing to carry.
                AnthropicContentBlockRaw::Thinking { thinking, .. } => {
                    reasoning.push_str(&thinking)
                }
                AnthropicContentBlockRaw::RedactedThinking { .. } => {}
                other @ (AnthropicContentBlockRaw::Image { .. }
                | AnthropicContentBlockRaw::ToolResult { .. }
                | AnthropicContentBlockRaw::Raw(_)) => {
                    return Err(eyre!(
                        "an assistant message block has no chat-completion form: {:?}",
                        other
                    )
                    .into());
                }
            }
        }
        let mut content =
            (!parts.is_empty()).then(|| ChatCompletionRequestAssistantMessageContent::Array(parts));
        if let Some(content) = content.as_mut() {
            content.compact();
        }
        #[allow(deprecated)]
        let mut msg = WithOtherFields::new(ChatCompletionRequestAssistantMessageRaw {
            content,
            refusal: None,
            name: None,
            audio: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            function_call: None,
        });
        if !reasoning.is_empty() {
            msg.other
                .insert("reasoning_content".to_string(), Value::String(reasoning));
        }
        out.push(WithOtherFields::new(
            ChatCompletionRequestMessageRaw::Assistant(msg),
        ));
        Ok(())
    }

    fn chat_tool_choice(
        choice: Option<AnthropicToolChoice>,
    ) -> (Option<ChatCompletionToolChoiceOption>, Option<bool>) {
        let Some(choice) = choice else {
            return (None, None);
        };
        let (mapped, disable) = match choice.into_inner() {
            AnthropicToolChoiceRaw::Auto {
                disable_parallel_tool_use,
            } => (
                ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Auto),
                disable_parallel_tool_use,
            ),
            AnthropicToolChoiceRaw::Any {
                disable_parallel_tool_use,
            } => (
                ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Required),
                disable_parallel_tool_use,
            ),
            AnthropicToolChoiceRaw::None => (
                ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::None),
                None,
            ),
            AnthropicToolChoiceRaw::Tool {
                name,
                disable_parallel_tool_use,
            } => (
                ChatCompletionToolChoiceOptionRaw::Function(WithOtherFields::new(
                    ChatCompletionNamedToolChoiceRaw {
                        function: WithOtherFields::new(FunctionNameRaw { name }),
                    },
                )),
                disable_parallel_tool_use,
            ),
        };
        (Some(WithOtherFields::new(mapped)), disable.map(|d| !d))
    }

    /// Reverse of [`Self::thinking`]'s buckets: the budget's share of
    /// `max_tokens` picks the nearest effort level.
    fn effort_of_budget(budget_tokens: u32, max_tokens: u32) -> ReasoningEffort {
        let share = budget_tokens as f64 / max_tokens.max(1) as f64;
        if share < 0.175 {
            ReasoningEffort::Minimal
        } else if share < 0.375 {
            ReasoningEffort::Low
        } else if share < 0.625 {
            ReasoningEffort::Medium
        } else if share < 0.825 {
            ReasoningEffort::High
        } else {
            ReasoningEffort::Xhigh
        }
    }
}

// ---------------------------------------------------------------------------
// Anthropic response -> chat response
// ---------------------------------------------------------------------------

impl AnthropicMessagesResponseRaw {
    /// The assistant turn of this response in protocol-neutral form — typed
    /// parts, signatures included, ready to go back into conversation state.
    /// A block carrying extension fields (outside `tool_use`, whose extras
    /// ride the part) is kept whole as an opaque part so nothing is shed.
    pub fn to_message(&self) -> Message {
        let mut parts = Vec::new();
        for block in &self.content {
            if !block.other.is_empty()
                && !matches!(block.inner, AnthropicContentBlockRaw::ToolUse { .. })
            {
                parts.push(MessagePart::Opaque {
                    protocol: ANTHROPIC_PROTOCOL.to_string(),
                    value: serde_json::to_value(block).unwrap_or(Value::Null),
                });
                continue;
            }
            match &block.inner {
                AnthropicContentBlockRaw::Text { text, .. } => {
                    parts.push(MessagePart::Text { text: text.clone() })
                }
                AnthropicContentBlockRaw::Thinking {
                    thinking,
                    signature,
                } => parts.push(MessagePart::Thinking {
                    thinking: thinking.clone(),
                    signature: signature.clone(),
                }),
                AnthropicContentBlockRaw::RedactedThinking { data } => {
                    parts.push(MessagePart::RedactedThinking { data: data.clone() })
                }
                AnthropicContentBlockRaw::ToolUse {
                    id, name, input, ..
                } => parts.push(MessagePart::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: input.to_string(),
                    extra: block.other.clone(),
                }),
                AnthropicContentBlockRaw::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => parts.push(MessagePart::ToolResult {
                    id: tool_use_id.clone(),
                    content: content.clone(),
                }),
                AnthropicContentBlockRaw::Image { source, .. } => parts.push(MessagePart::Image {
                    url: match &source.inner {
                        AnthropicImageSourceRaw::Url { url } => url.clone(),
                        AnthropicImageSourceRaw::Base64 { media_type, data } => {
                            format!("data:{};base64,{}", media_type, data)
                        }
                    },
                }),
                AnthropicContentBlockRaw::Raw(value) => parts.push(MessagePart::Opaque {
                    protocol: ANTHROPIC_PROTOCOL.to_string(),
                    value: value.clone(),
                }),
            }
        }
        Message::new(MessageRole::Assistant, parts)
    }

    /// Fold a Messages response back into the chat-completion shape the rest
    /// of the crate consumes: text blocks concatenate into `content`,
    /// `tool_use` blocks become `tool_calls`, thinking lands in the message's
    /// `reasoning_content` extra (mirroring providers that expose it there),
    /// and the usage split is widened to chat semantics (`prompt_tokens`
    /// includes cache reads and writes).
    pub fn into_chat_response(self) -> RawExtensibleChatCompletionResponse {
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut tool_calls: Vec<ChatCompletionMessageToolCalls> = Vec::new();
        for block in self.content {
            match block.inner {
                AnthropicContentBlockRaw::Text { text: chunk, .. } => text.push_str(&chunk),
                AnthropicContentBlockRaw::Thinking { thinking, .. } => {
                    reasoning.push_str(&thinking)
                }
                AnthropicContentBlockRaw::ToolUse {
                    id, name, input, ..
                } => {
                    tool_calls.push(WithOtherFields::new(
                        ChatCompletionMessageToolCallsRaw::Function(WithOtherFields::new(
                            ChatCompletionMessageToolCallRaw {
                                id,
                                function: WithOtherFields::new(FunctionCallRaw {
                                    name,
                                    arguments: input.to_string(),
                                }),
                            },
                        )),
                    ));
                }
                AnthropicContentBlockRaw::Image { .. }
                | AnthropicContentBlockRaw::ToolResult { .. }
                | AnthropicContentBlockRaw::RedactedThinking { .. }
                | AnthropicContentBlockRaw::Raw(_) => {}
            }
        }

        let finish_reason = self.stop_reason.as_ref().map(|reason| match reason {
            AnthropicStopReason::EndTurn
            | AnthropicStopReason::StopSequence
            | AnthropicStopReason::PauseTurn
            | AnthropicStopReason::Other(_) => FinishReason::Stop,
            AnthropicStopReason::MaxTokens => FinishReason::Length,
            AnthropicStopReason::ToolUse => FinishReason::ToolCalls,
            AnthropicStopReason::Refusal => FinishReason::ContentFilter,
        });

        #[allow(deprecated)]
        let mut message = WithOtherFields::new(ChatCompletionResponseMessageRaw {
            content: (!text.is_empty()).then_some(text),
            refusal: None,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            annotations: None,
            role: Role::Assistant,
            function_call: None,
            audio: None,
        });
        if !reasoning.is_empty() {
            message
                .other
                .insert("reasoning_content".to_string(), Value::String(reasoning));
        }

        let cached = self.usage.inner.cache_read_input_tokens.unwrap_or(0);
        let cache_write = self.usage.inner.cache_creation_input_tokens.unwrap_or(0);
        let prompt_tokens = self
            .usage
            .inner
            .input_tokens
            .saturating_add(cached)
            .saturating_add(cache_write);
        let completion_tokens = self.usage.inner.output_tokens;
        let usage = WithOtherFields::new(CompletionUsageRaw {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
            prompt_tokens_details: Some(WithOtherFields::new(PromptTokensDetailsRaw {
                audio_tokens: None,
                cached_tokens: Some(cached),
                cache_write_tokens: Some(cache_write),
            })),
            completion_tokens_details: None,
        });

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as u32)
            .unwrap_or(0);
        #[allow(deprecated)]
        let raw = CreateChatCompletionResponseRaw {
            id: self.id,
            choices: vec![WithOtherFields::new(ChatChoiceRaw {
                index: 0,
                message,
                finish_reason,
                logprobs: None,
            })],
            created,
            model: self.model,
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion".to_string(),
            usage: Some(usage),
        };
        RawExtensibleChatCompletionResponse::new(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::req::{
        ChatCompletionRequestMessageRaw, ChatCompletionRequestSystemMessageRaw,
        ChatCompletionRequestToolMessageRaw, ChatCompletionRequestUserMessageRaw,
    };
    use async_openai::config::Config;

    fn chat_request(
        messages: Vec<ChatCompletionRequestMessageRaw>,
    ) -> RawExtensibleChatCompletionRequest {
        let mut raw = CreateChatCompletionRequestRaw::default();
        raw.model = "claude-test".to_string();
        raw.messages = messages.into_iter().map(WithOtherFields::new).collect();
        RawExtensibleChatCompletionRequest::new(raw)
    }

    #[test]
    fn config_rewrites_the_chat_path_and_signs_with_the_anthropic_headers() {
        let config = AnthropicConfig::new(
            "https://api.anthropic.com/v1/",
            "sk-ant-test",
            DEFAULT_ANTHROPIC_VERSION,
        );
        assert_eq!(
            config.url("/chat/completions"),
            "https://api.anthropic.com/v1/messages"
        );
        let headers = config.headers();
        assert_eq!(headers.get("x-api-key").unwrap(), "sk-ant-test");
        assert_eq!(
            headers.get("anthropic-version").unwrap(),
            DEFAULT_ANTHROPIC_VERSION
        );
        assert!(headers.get("authorization").is_none());
        // The Debug rendering must not leak the key.
        assert!(!format!("{config:?}").contains("sk-ant-test"));
    }

    #[test]
    fn a_bearer_config_signs_with_authorization_instead_of_x_api_key() {
        let config = AnthropicConfig::new_bearer(
            "https://llm.example/v1",
            "sk-token",
            DEFAULT_ANTHROPIC_VERSION,
        );
        let headers = config.headers();
        assert_eq!(headers.get("authorization").unwrap(), "Bearer sk-token");
        assert!(headers.get("x-api-key").is_none());
        // The Debug rendering must not leak the token either.
        assert!(!format!("{config:?}").contains("sk-token"));
    }

    #[test]
    fn a_conversation_converts_to_alternating_anthropic_messages() {
        let mut req = chat_request(vec![
            ChatCompletionRequestMessageRaw::System(
                ChatCompletionRequestSystemMessageRaw::new_text("be terse"),
            ),
            ChatCompletionRequestMessageRaw::User(ChatCompletionRequestUserMessageRaw::new_text(
                "look it up",
            )),
            // An assistant tool call followed by two tool results and the next
            // user turn: the results and the turn must merge into ONE user
            // message, keeping roles alternating.
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\": 1}"}},
                    {"id": "call_2", "type": "function", "function": {"name": "lookup", "arguments": ""}}
                ]
            }))
            .map(|m: crate::req::ChatCompletionRequestAssistantMessage| {
                ChatCompletionRequestMessageRaw::Assistant(m)
            })
            .unwrap(),
            ChatCompletionRequestMessageRaw::Tool(ChatCompletionRequestToolMessageRaw::new_text(
                "found a", "call_1",
            )),
            ChatCompletionRequestMessageRaw::Tool(ChatCompletionRequestToolMessageRaw::new_text(
                "found b", "call_2",
            )),
            ChatCompletionRequestMessageRaw::User(ChatCompletionRequestUserMessageRaw::new_text(
                "so?",
            )),
        ]);
        req.max_completion_tokens = Some(300);
        req.stop = Some(StopConfiguration::String("END".to_string()));

        let converted = AnthropicMessagesRequestRaw::from_chat(&req, 4096).unwrap();
        let value = serde_json::to_value(&converted).unwrap();

        assert_eq!(value["model"], "claude-test");
        assert_eq!(value["max_tokens"], 300);
        assert_eq!(value["system"][0]["type"], "text");
        assert_eq!(value["system"][0]["text"], "be terse");
        assert_eq!(value["stop_sequences"], serde_json::json!(["END"]));

        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3, "{messages:?}");
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["content"][0]["type"], "tool_use");
        assert_eq!(
            messages[1]["content"][0]["input"],
            serde_json::json!({"q": 1})
        );
        // Empty arguments become an empty input object, not a parse error.
        assert_eq!(messages[1]["content"][1]["input"], serde_json::json!({}));
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "call_1");
        assert_eq!(messages[2]["content"][1]["tool_use_id"], "call_2");
        assert_eq!(messages[2]["content"][2]["type"], "text");
        assert_eq!(messages[2]["content"][2]["text"], "so?");
        // No chat-only leftovers on the wire.
        assert!(value.get("prompt_cache_key").is_none());
        assert!(value.get("max_completion_tokens").is_none());
    }

    #[test]
    fn tools_and_choice_map_and_breakpoints_become_cache_control() {
        let mut req = chat_request(vec![ChatCompletionRequestMessageRaw::User(
            ChatCompletionRequestUserMessageRaw::new_text("hi"),
        )]);
        req.messages[0].inner.toggle_cache_breakpoint(true);
        req.tools =
            Some(vec![serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": "lookup", "description": "d", "parameters": {"type": "object"}}
        }))
        .unwrap()]);
        req.tool_choice = Some(WithOtherFields::new(
            ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Required),
        ));
        req.parallel_tool_calls = Some(false);

        let value =
            serde_json::to_value(AnthropicMessagesRequestRaw::from_chat(&req, 4096).unwrap())
                .unwrap();
        assert_eq!(
            value["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert_eq!(value["tools"][0]["name"], "lookup");
        assert_eq!(value["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(value["tool_choice"]["type"], "any");
        assert_eq!(value["tool_choice"]["disable_parallel_tool_use"], true);
    }

    #[test]
    fn reasoning_effort_buys_a_thinking_budget_and_none_disables_it() {
        let mut req = chat_request(vec![ChatCompletionRequestMessageRaw::User(
            ChatCompletionRequestUserMessageRaw::new_text("hi"),
        )]);
        req.max_completion_tokens = Some(10_000);
        req.reasoning_effort = Some(ReasoningEffort::High);
        let value =
            serde_json::to_value(AnthropicMessagesRequestRaw::from_chat(&req, 4096).unwrap())
                .unwrap();
        assert_eq!(value["thinking"]["type"], "enabled");
        assert_eq!(value["thinking"]["budget_tokens"], 7500);

        req.reasoning_effort = Some(ReasoningEffort::None);
        let value =
            serde_json::to_value(AnthropicMessagesRequestRaw::from_chat(&req, 4096).unwrap())
                .unwrap();
        assert_eq!(value["thinking"]["type"], "disabled");
    }

    #[test]
    fn unsupported_chat_fields_are_refused_not_dropped() {
        let mut req = chat_request(vec![ChatCompletionRequestMessageRaw::User(
            ChatCompletionRequestUserMessageRaw::new_text("hi"),
        )]);
        req.presence_penalty = Some(0.5);
        let err = AnthropicMessagesRequestRaw::from_chat(&req, 4096).unwrap_err();
        assert!(err.to_string().contains("presence_penalty"), "{err}");
    }

    #[test]
    fn a_response_folds_back_into_a_chat_completion() {
        let resp: AnthropicMessagesResponse = serde_json::from_value(serde_json::json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "calling the tool"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": 1}}
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 7,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2
            }
        }))
        .unwrap();

        let chat = resp.into_inner().into_chat_response();
        assert_eq!(chat.id, "msg_1");
        let choice = &chat.choices[0].inner;
        assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(
            choice.message.inner.content.as_deref(),
            Some("calling the tool")
        );
        assert_eq!(
            choice
                .message
                .other
                .get("reasoning_content")
                .and_then(|v| v.as_str()),
            Some("hmm")
        );
        let tool_calls = choice.message.inner.tool_calls.as_ref().unwrap();
        match &tool_calls[0].inner {
            ChatCompletionMessageToolCallsRaw::Function(call) => {
                assert_eq!(call.inner.id, "toolu_1");
                assert_eq!(call.inner.function.inner.name, "lookup");
                assert_eq!(
                    serde_json::from_str::<Value>(&call.inner.function.inner.arguments).unwrap(),
                    serde_json::json!({"q": 1})
                );
            }
            other => panic!("expected function tool call, got {other:?}"),
        }

        // Chat semantics: prompt_tokens includes cache reads AND writes.
        let usage = chat.usage.as_ref().unwrap();
        assert_eq!(usage.inner.prompt_tokens, 15);
        assert_eq!(usage.inner.completion_tokens, 7);
        let details = usage.inner.prompt_tokens_details.as_ref().unwrap();
        assert_eq!(details.inner.cached_tokens, Some(3));
        assert_eq!(details.inner.cache_write_tokens, Some(2));
    }

    #[test]
    fn a_streamed_message_accumulates_to_the_same_chat_response() {
        let events: Vec<AnthropicStreamEvent> = serde_json::from_value(serde_json::json!([
            {"type": "message_start", "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
                "content": [], "usage": {"input_tokens": 10, "output_tokens": 1}
            }},
            {"type": "ping"},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hel"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block":
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"q\":"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": " 1}"}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
            {"type": "message_stop"}
        ]))
        .unwrap();

        let mut acc = AnthropicStreamAccumulator::new();
        for event in events {
            acc.push(event).unwrap();
        }
        let chat = acc.finish().unwrap().into_inner().into_chat_response();

        let choice = &chat.choices[0].inner;
        assert_eq!(choice.message.inner.content.as_deref(), Some("hello"));
        assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));
        let tool_calls = choice.message.inner.tool_calls.as_ref().unwrap();
        match &tool_calls[0].inner {
            ChatCompletionMessageToolCallsRaw::Function(call) => {
                assert_eq!(
                    serde_json::from_str::<Value>(&call.inner.function.inner.arguments).unwrap(),
                    serde_json::json!({"q": 1})
                );
            }
            other => panic!("expected function tool call, got {other:?}"),
        }
        let usage = chat.usage.as_ref().unwrap();
        assert_eq!(usage.inner.prompt_tokens, 10);
        assert_eq!(usage.inner.completion_tokens, 9);
    }

    #[test]
    fn a_stream_error_event_fails_the_request() {
        let mut acc = AnthropicStreamAccumulator::new();
        let err = acc
            .push(AnthropicStreamEvent::Error {
                error: AnthropicErrorBody {
                    kind: "overloaded_error".to_string(),
                    message: "try later".to_string(),
                },
            })
            .unwrap_err();
        assert!(err.to_string().contains("overloaded_error"), "{err}");
    }

    #[test]
    fn unknown_content_blocks_and_events_are_tolerated() {
        // A server-side block kind this client does not model must neither
        // fail deserialization nor leak into the converted chat response.
        let resp: AnthropicMessagesResponse = serde_json::from_value(serde_json::json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search"},
                {"type": "text", "text": "done"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .unwrap();
        let chat = resp.into_inner().into_chat_response();
        assert_eq!(
            chat.choices[0].inner.message.inner.content.as_deref(),
            Some("done")
        );

        let event: AnthropicStreamEvent =
            serde_json::from_value(serde_json::json!({"type": "brand_new_event"})).unwrap();
        assert_eq!(event, AnthropicStreamEvent::Unknown);
    }

    #[test]
    fn signed_thinking_survives_the_context_round_trip() {
        // The wire truth of an assistant turn: signed thinking first, then the
        // visible answer and a tool call.
        let content = serde_json::json!([
            {"type": "thinking", "thinking": "hmm", "signature": "sig-1"},
            {"type": "text", "text": "on it"},
            {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": 1}}
        ]);
        let resp: AnthropicMessagesResponse = serde_json::from_value(serde_json::json!({
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
            "content": content,
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
        .unwrap();

        // The response parses into typed parts, signature included.
        let assistant = resp.to_message();
        assert_eq!(assistant.role, MessageRole::Assistant);
        assert!(matches!(
            &assistant.parts[0],
            MessagePart::Thinking { thinking, signature: Some(signature) }
                if thinking == "hmm" && signature == "sig-1"
        ));

        // Rebuilding the next turn from conversation state replays the blocks
        // verbatim — thinking first, signature intact.
        let conversation = vec![
            Message::user("look it up"),
            assistant,
            Message::tool_result("toolu_1", "found"),
        ];
        let settings = LLMSettings {
            llm_temperature: None,
            llm_presence_penalty: None,
            llm_prompt_timeout: 0,
            llm_retry: 1,
            llm_max_completion_tokens: Some(512),
            llm_tool_choice: None,
            llm_stream: false,
            top_p: None,
            reasoning_effort: None,
            auto_strip: false,
            auto_cache_key: false,
            cache_key_ttl: 0,
            cache_key_rpm: 1,
            billing_log_tokens: 0,
            token_estimate_pct: 10.0,
            allow_implicit_convert: false,
        };
        let value = serde_json::to_value(
            AnthropicMessagesRequestRaw::from_conversation(
                "claude-test",
                &conversation,
                None,
                &settings,
                4096,
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(value["messages"][1]["content"], content);
        assert_eq!(value["messages"][2]["content"][0]["type"], "tool_result");
        assert_eq!(value["max_tokens"], 512);
    }

    #[test]
    fn a_native_request_folds_back_into_chat_for_other_backends() {
        let req: AnthropicMessagesRequest = serde_json::from_value(serde_json::json!({
            "model": "claude-test",
            "max_tokens": 512,
            "system": [{"type": "text", "text": "be terse"}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "look it up"}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "on it"},
                    {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": 1}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "found"},
                    {"type": "text", "text": "so?"}
                ]}
            ],
            "tools": [{"name": "lookup", "description": "d", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
            "stop_sequences": ["END"],
            "thinking": {"type": "enabled", "budget_tokens": 256}
        }))
        .unwrap();

        let chat = req.into_inner().into_chat_request().unwrap();
        let value = serde_json::to_value(&chat).unwrap();
        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 5, "{messages:?}");
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be terse");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "look it up");
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["content"], "on it");
        assert_eq!(messages[2]["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(messages[2]["reasoning_content"], "hmm");
        // The merged user message splits back out: the tool result, then the
        // next user turn.
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["tool_call_id"], "toolu_1");
        assert_eq!(messages[3]["content"], "found");
        assert_eq!(messages[4]["role"], "user");
        assert_eq!(messages[4]["content"], "so?");

        assert_eq!(value["max_completion_tokens"], 512);
        assert_eq!(value["stop"], serde_json::json!(["END"]));
        assert_eq!(value["tools"][0]["function"]["name"], "lookup");
        assert_eq!(value["tool_choice"], "required");
        assert_eq!(value["parallel_tool_calls"], false);
        // 256/512 = 50% of max_tokens => medium.
        assert_eq!(value["reasoning_effort"], "medium");
    }
}
