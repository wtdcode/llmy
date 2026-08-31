//! OpenAI Responses protocol support.
//!
//! Wire types mirroring the [Responses API](https://platform.openai.com/docs/api-reference/responses)
//! (wrapped in [`WithOtherFields`] like the chat types, so unknown fields
//! survive a round trip) plus the conversions between these types and the
//! chat-completion types the rest of the crate works in.
//!
//! The client-facing surface stays chat completion: `LLMClient` converts an
//! outgoing [`RawExtensibleChatCompletionRequest`] into a [`ResponsesRequest`]
//! at the wire boundary and folds the reply back into a
//! [`RawExtensibleChatCompletionResponse`], so billing, debug records, content
//! filters and cache keys all keep operating on one set of types. Requests are
//! sent with `store: false` unless the caller opts in, mirroring chat
//! completion's statelessness.
//!
//! Multi-turn fidelity: conversation-state callers hold
//! [`crate::message::Message`]s, and this module maps them natively in both
//! directions — [`ResponsesRequestRaw::from_conversation`] builds the request
//! straight from typed parts (reasoning items replay with their
//! `encrypted_content`), and [`ResponsesResponseRaw::to_message`] parses the
//! reply back into them. Stateless requests ask for
//! `reasoning.encrypted_content` back so the replay has something to carry.

use color_eyre::eyre::eyre;
use llmy_types::{error::LLMYError, other::WithOtherFields};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cache_key::CacheShape;
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
    ChatCompletionToolsRaw, CreateChatCompletionRequestRaw, FunctionCallRaw, ImageDetail, Metadata,
    RawExtensibleChatCompletionRequest, ReasoningEffort, ResponseFormatRaw, Role, ServiceTier,
    ToolChoiceOptions, Verbosity,
};
use crate::req::{
    ChatCompletionNamedToolChoiceRaw, ChatCompletionRequestDeveloperMessageRaw,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartFileRaw,
    ChatCompletionRequestMessageContentPartImageRaw,
    ChatCompletionRequestMessageContentPartTextRaw, ChatCompletionRequestSystemMessageRaw,
    ChatCompletionRequestToolMessageRaw, ChatCompletionRequestUserMessageRaw,
    ChatCompletionToolRaw, FileObjectRaw, FunctionNameRaw, FunctionObjectRaw, ImageUrlRaw,
    ResponseFormatJsonSchemaRaw,
};
use crate::req::{PromptCacheBreakpoint, PromptCacheBreakpointRaw, PromptCacheOptions};
use crate::resp::{
    ChatChoiceRaw, ChatCompletionMessageToolCalls, ChatCompletionResponseMessageRaw,
    CompletionTokensDetailsRaw, CompletionUsageRaw, CreateChatCompletionResponseRaw, FinishReason,
    PromptTokensDetailsRaw, RawExtensibleChatCompletionResponse,
};
use crate::settings::LLMSettings;

/// Protocol tag carried by [`MessagePart::Opaque`] parts this module produces,
/// so replay stays confined to its own protocol.
const RESPONSES_PROTOCOL: &str = "responses";

// ---------------------------------------------------------------------------
// Wire types: request
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ResponsesRole {
    System,
    Developer,
    User,
    Assistant,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesInputPartRaw {
    InputText {
        text: String,
        /// Explicit cache breakpoint on this block (GPT-5.6+ explicit
        /// caching) — the same extension the chat protocol carries.
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
    },
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        file_data: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    /// Part kinds this client does not model, captured verbatim so a replayed
    /// conversation keeps them.
    #[serde(untagged)]
    Raw(Value),
}
pub type ResponsesInputPart = WithOtherFields<ResponsesInputPartRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ResponsesInputContent {
    Text(String),
    Parts(Vec<ResponsesInputPart>),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesInputItemRaw {
    Message {
        role: ResponsesRole,
        content: ResponsesInputContent,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: String,
        /// Explicit cache breakpoint on this item (accepted by the API like
        /// the chat tool-message breakpoint).
        #[serde(skip_serializing_if = "Option::is_none")]
        prompt_cache_breakpoint: Option<PromptCacheBreakpoint>,
    },
    /// A replayed reasoning item; `encrypted_content` carries the model's own
    /// reasoning back on stateless requests.
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default)]
        summary: Vec<ResponsesReasoningSummary>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
    /// Item kinds this client does not model (built-in tool calls), captured
    /// verbatim so they go back on the wire untouched.
    #[serde(untagged)]
    Raw(Value),
}
pub type ResponsesInputItem = WithOtherFields<ResponsesInputItemRaw>;

impl ResponsesInputItemRaw {
    fn message(role: ResponsesRole, content: ResponsesInputContent) -> ResponsesInputItem {
        WithOtherFields::new(Self::Message { role, content })
    }
}

/// Responses function tools are flat (`name` at top level), unlike chat's
/// nested `function` object.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesToolRaw {
    Function {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        parameters: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
    /// Tool kinds this client does not model (built-in tools), captured
    /// verbatim.
    #[serde(untagged)]
    Raw(Value),
}
pub type ResponsesTool = WithOtherFields<ResponsesToolRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesNamedToolChoiceRaw {
    Function { name: String },
}
pub type ResponsesNamedToolChoice = WithOtherFields<ResponsesNamedToolChoiceRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ResponsesToolChoice {
    Mode(ToolChoiceOptions),
    Named(ResponsesNamedToolChoice),
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct ResponsesReasoningRaw {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
}
pub type ResponsesReasoning = WithOtherFields<ResponsesReasoningRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesTextFormatRaw {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        schema: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
}
pub type ResponsesTextFormat = WithOtherFields<ResponsesTextFormatRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct ResponsesTextConfigRaw {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<ResponsesTextFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,
}
pub type ResponsesTextConfig = WithOtherFields<ResponsesTextConfigRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ResponsesRequestRaw {
    pub model: String,
    /// The protocol's top-level system-instructions slot. The builders promote
    /// the leading run of system messages here — leaving it unset invites
    /// gateways to stuff their own persona into it — while a system message
    /// later in the transcript stays in `input`, keeping its position. A
    /// system message carrying a cache breakpoint also stays in `input`:
    /// `instructions` cannot hold explicit breakpoints.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    pub input: Vec<ResponsesInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ResponsesTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ResponsesToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoning>,
    /// Extra response payloads to include. `reasoning.encrypted_content` is
    /// requested automatically for stateless (`store: false`) requests, so
    /// reasoning items can be replayed on the next turn.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponsesTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// Request-wide explicit-caching policy (GPT-5.6+), same field as chat.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_options: Option<PromptCacheOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    /// The Responses API stores responses server-side by default; the
    /// conversion sends `false` unless the chat request opted in, mirroring
    /// chat completion's statelessness.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}
pub type ResponsesRequest = WithOtherFields<ResponsesRequestRaw>;

// ---------------------------------------------------------------------------
// Wire types: response
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputContentRaw {
    OutputText {
        text: String,
    },
    Refusal {
        refusal: String,
    },
    /// Content kinds this client does not model, captured verbatim.
    #[serde(untagged)]
    Raw(Value),
}
pub type ResponsesOutputContent = WithOtherFields<ResponsesOutputContentRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesReasoningSummaryRaw {
    SummaryText {
        text: String,
    },
    /// Summary kinds this client does not model, captured verbatim.
    #[serde(untagged)]
    Raw(Value),
}
pub type ResponsesReasoningSummary = WithOtherFields<ResponsesReasoningSummaryRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItemRaw {
    Message {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        role: ResponsesRole,
        content: Vec<ResponsesOutputContent>,
    },
    FunctionCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
    },
    Reasoning {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default)]
        summary: Vec<ResponsesReasoningSummary>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
    /// Output item kinds this client does not model (built-in tool calls
    /// etc.), captured verbatim so a replayed conversation keeps them.
    #[serde(untagged)]
    Raw(Value),
}
pub type ResponsesOutputItem = WithOtherFields<ResponsesOutputItemRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct ResponsesInputTokensDetailsRaw {
    #[serde(default)]
    pub cached_tokens: Option<u32>,
}
pub type ResponsesInputTokensDetails = WithOtherFields<ResponsesInputTokensDetailsRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct ResponsesOutputTokensDetailsRaw {
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}
pub type ResponsesOutputTokensDetails = WithOtherFields<ResponsesOutputTokensDetailsRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct ResponsesUsageRaw {
    /// Total prompt tokens, cache reads included — same semantics as chat's
    /// `prompt_tokens`.
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<ResponsesInputTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<ResponsesOutputTokensDetails>,
}
pub type ResponsesUsage = WithOtherFields<ResponsesUsageRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct ResponsesErrorRaw {
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub message: String,
}
pub type ResponsesError = WithOtherFields<ResponsesErrorRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct ResponsesIncompleteDetailsRaw {
    #[serde(default)]
    pub reason: Option<String>,
}
pub type ResponsesIncompleteDetails = WithOtherFields<ResponsesIncompleteDetailsRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ResponsesResponseRaw {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created_at: f64,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub output: Vec<ResponsesOutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponsesUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<ResponsesIncompleteDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ResponsesError>,
}
pub type ResponsesResponse = WithOtherFields<ResponsesResponseRaw>;

/// One SSE event of a streamed response. Only the terminal events matter for
/// aggregation — `response.completed` / `response.incomplete` carry the entire
/// final response object — so every incremental delta event lands in `Other`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
pub enum ResponsesStreamEvent {
    #[serde(rename = "response.completed")]
    Completed { response: ResponsesResponse },
    #[serde(rename = "response.incomplete")]
    Incomplete { response: ResponsesResponse },
    #[serde(rename = "response.failed")]
    Failed { response: ResponsesResponse },
    #[serde(rename = "error")]
    Error {
        #[serde(default)]
        code: Option<String>,
        #[serde(default)]
        message: String,
    },
    #[serde(other)]
    Other,
}

// ---------------------------------------------------------------------------
// Chat request -> Responses request
// ---------------------------------------------------------------------------

impl ResponsesRequestRaw {
    /// Convert a chat-completion request into the Responses wire form.
    pub fn from_chat(
        req: &RawExtensibleChatCompletionRequest,
    ) -> Result<ResponsesRequest, LLMYError> {
        let raw: &CreateChatCompletionRequestRaw = req;
        Self::reject_unsupported(raw)?;

        let mut input: Vec<ResponsesInputItem> = Vec::new();
        let mut instructions: Vec<String> = Vec::new();
        for msg in &raw.messages {
            match &msg.inner {
                // The leading run of system messages takes the protocol's own
                // `instructions` slot; one appearing after other messages —
                // or one carrying a cache breakpoint, which `instructions`
                // cannot hold — stays in `input`.
                ChatCompletionRequestMessageRaw::System(m)
                    if input.is_empty() && !Self::system_has_breakpoint(&m.inner.content) =>
                {
                    instructions.push(Self::system_text(&m.inner.content));
                }
                ChatCompletionRequestMessageRaw::System(m) => {
                    input.push(ResponsesInputItemRaw::message(
                        ResponsesRole::System,
                        Self::system_content(&m.inner.content),
                    ))
                }
                ChatCompletionRequestMessageRaw::Developer(m) => {
                    input.push(ResponsesInputItemRaw::message(
                        ResponsesRole::Developer,
                        ResponsesInputContent::Text(Self::developer_text(&m.inner.content)),
                    ))
                }
                ChatCompletionRequestMessageRaw::User(m) => {
                    input.push(ResponsesInputItemRaw::message(
                        ResponsesRole::User,
                        Self::user_content(&m.inner.content)?,
                    ))
                }
                ChatCompletionRequestMessageRaw::Assistant(m) => {
                    Self::assistant_items(&m.inner, &mut input)?
                }
                ChatCompletionRequestMessageRaw::Tool(m) => input.push(WithOtherFields::new(
                    ResponsesInputItemRaw::FunctionCallOutput {
                        call_id: m.inner.tool_call_id.clone(),
                        output: Self::tool_text(&m.inner.content),
                        prompt_cache_breakpoint: m
                            .inner
                            .content
                            .has_cache_breakpoint()
                            .then(PromptCacheBreakpointRaw::explicit),
                    },
                )),
                ChatCompletionRequestMessageRaw::Function(_) => {
                    return Err(eyre!(
                        "legacy `function` messages are not supported on the responses protocol"
                    )
                    .into());
                }
            }
        }

        let tools = raw.tools.as_deref().map(Self::tools).transpose()?;
        let tool_choice = Self::tool_choice(raw.tool_choice.as_ref())?;
        let reasoning = raw.reasoning_effort.as_ref().map(|effort| {
            WithOtherFields::new(ResponsesReasoningRaw {
                effort: Some(effort.clone()),
            })
        });
        let text = Self::text_config(raw)?;
        #[allow(deprecated)]
        let max_output_tokens = raw.max_completion_tokens.or(raw.max_tokens);

        Ok(WithOtherFields::new(Self {
            model: raw.model.clone(),
            instructions: (!instructions.is_empty()).then(|| instructions.join("\n\n")),
            input,
            prompt_cache_options: raw.prompt_cache_options.clone(),
            max_output_tokens,
            temperature: raw.temperature,
            top_p: raw.top_p,
            top_logprobs: raw.top_logprobs,
            tools,
            tool_choice,
            parallel_tool_calls: raw.parallel_tool_calls,
            reasoning,
            text,
            prompt_cache_key: raw.prompt_cache_key.clone(),
            safety_identifier: raw.safety_identifier.clone(),
            service_tier: raw.service_tier.clone(),
            metadata: raw.metadata.clone(),
            include: (!raw.store.unwrap_or(false))
                .then(|| vec!["reasoning.encrypted_content".to_string()]),
            store: Some(raw.store.unwrap_or(false)),
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
        if raw.logprobs.is_some() {
            unsupported.push("logprobs");
        }
        if raw.stop.is_some() {
            unsupported.push("stop");
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
        #[allow(deprecated)]
        if raw.functions.is_some() || raw.function_call.is_some() {
            unsupported.push("functions/function_call");
        }
        if unsupported.is_empty() {
            Ok(())
        } else {
            Err(eyre!(
                "the responses protocol does not support: {}",
                unsupported.join(", ")
            )
            .into())
        }
    }

    fn system_has_breakpoint(content: &ChatCompletionRequestSystemMessageContent) -> bool {
        match content {
            ChatCompletionRequestSystemMessageContent::Text(_) => false,
            ChatCompletionRequestSystemMessageContent::Array(parts) => {
                parts.iter().any(|part| part.inner.has_cache_breakpoint())
            }
        }
    }

    /// System content for an `input` item: plain text when no part carries a
    /// breakpoint, parts form (which can hold the markers) otherwise.
    fn system_content(
        content: &ChatCompletionRequestSystemMessageContent,
    ) -> ResponsesInputContent {
        match content {
            ChatCompletionRequestSystemMessageContent::Array(parts)
                if Self::system_has_breakpoint(content) =>
            {
                ResponsesInputContent::Parts(
                    parts
                        .iter()
                        .map(|part| {
                            let ChatCompletionRequestSystemMessageContentPartRaw::Text(text) =
                                &part.inner;
                            WithOtherFields::new(ResponsesInputPartRaw::InputText {
                                text: text.inner.text.clone(),
                                prompt_cache_breakpoint: text.inner.prompt_cache_breakpoint.clone(),
                            })
                        })
                        .collect(),
                )
            }
            content => ResponsesInputContent::Text(Self::system_text(content)),
        }
    }

    fn system_text(content: &ChatCompletionRequestSystemMessageContent) -> String {
        match content {
            ChatCompletionRequestSystemMessageContent::Text(text) => text.clone(),
            ChatCompletionRequestSystemMessageContent::Array(parts) => parts
                .iter()
                .map(|part| {
                    let ChatCompletionRequestSystemMessageContentPartRaw::Text(text) = &part.inner;
                    text.inner.text.as_str()
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    fn developer_text(content: &ChatCompletionRequestDeveloperMessageContent) -> String {
        match content {
            ChatCompletionRequestDeveloperMessageContent::Text(text) => text.clone(),
            ChatCompletionRequestDeveloperMessageContent::Array(parts) => parts
                .iter()
                .map(|part| {
                    let ChatCompletionRequestDeveloperMessageContentPartRaw::Text(text) =
                        &part.inner;
                    text.inner.text.as_str()
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
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

    fn user_content(
        content: &ChatCompletionRequestUserMessageContent,
    ) -> Result<ResponsesInputContent, LLMYError> {
        match content {
            ChatCompletionRequestUserMessageContent::Text(text) => {
                Ok(ResponsesInputContent::Text(text.clone()))
            }
            ChatCompletionRequestUserMessageContent::Array(parts) => {
                let parts = parts
                    .iter()
                    .map(|part| {
                        let mapped = match &part.inner {
                            ChatCompletionRequestUserMessageContentPartRaw::Text(text) => {
                                ResponsesInputPartRaw::InputText {
                                    text: text.inner.text.clone(),
                                    prompt_cache_breakpoint: text
                                        .inner
                                        .prompt_cache_breakpoint
                                        .clone(),
                                }
                            }
                            ChatCompletionRequestUserMessageContentPartRaw::ImageUrl(image) => {
                                ResponsesInputPartRaw::InputImage {
                                    image_url: Some(image.inner.image_url.inner.url.clone()),
                                    detail: image.inner.image_url.inner.detail.clone(),
                                }
                            }
                            ChatCompletionRequestUserMessageContentPartRaw::File(file) => {
                                ResponsesInputPartRaw::InputFile {
                                    file_id: file.inner.file.inner.file_id.clone(),
                                    file_data: file.inner.file.inner.file_data.clone(),
                                    filename: file.inner.file.inner.filename.clone(),
                                }
                            }
                            ChatCompletionRequestUserMessageContentPartRaw::InputAudio(_) => {
                                return Err(LLMYError::from(eyre!(
                                    "audio input parts are not supported on the responses protocol"
                                )));
                            }
                        };
                        Ok(WithOtherFields::new(mapped))
                    })
                    .collect::<Result<Vec<_>, LLMYError>>()?;
                Ok(ResponsesInputContent::Parts(parts))
            }
        }
    }

    fn assistant_items(
        msg: &ChatCompletionRequestAssistantMessageRaw,
        input: &mut Vec<ResponsesInputItem>,
    ) -> Result<(), LLMYError> {
        let text = match &msg.content {
            Some(ChatCompletionRequestAssistantMessageContent::Text(text)) => text.clone(),
            Some(ChatCompletionRequestAssistantMessageContent::Array(parts)) => parts
                .iter()
                .map(|part| match &part.inner {
                    ChatCompletionRequestAssistantMessageContentPartRaw::Text(text) => {
                        text.inner.text.as_str()
                    }
                    ChatCompletionRequestAssistantMessageContentPartRaw::Refusal(refusal) => {
                        refusal.inner.refusal.as_str()
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
            None => String::new(),
        };
        let has_breakpoint = matches!(
            &msg.content,
            Some(ChatCompletionRequestAssistantMessageContent::Array(parts))
                if parts.iter().any(|part| part.inner.has_cache_breakpoint())
        );
        if !text.is_empty() {
            input.push(ResponsesInputItemRaw::message(
                ResponsesRole::Assistant,
                ResponsesInputContent::Text(text),
            ));
            if has_breakpoint {
                Self::mark_last_item_breakpoint(input);
            }
        }
        for tool_call in msg.tool_calls.iter().flatten() {
            match &tool_call.inner {
                ChatCompletionMessageToolCallsRaw::Function(call) => {
                    input.push(WithOtherFields::new(ResponsesInputItemRaw::FunctionCall {
                        call_id: call.inner.id.clone(),
                        name: call.inner.function.inner.name.clone(),
                        arguments: call.inner.function.inner.arguments.clone(),
                    }))
                }
                ChatCompletionMessageToolCallsRaw::Custom(_) => {
                    return Err(eyre!(
                        "custom tool calls are not supported on the responses protocol"
                    )
                    .into());
                }
            }
        }
        Ok(())
    }

    fn tools(tools: &[ChatCompletionTools]) -> Result<Vec<ResponsesTool>, LLMYError> {
        tools
            .iter()
            .map(|tool| match &tool.inner {
                ChatCompletionToolsRaw::Function(function) => {
                    let function = &function.inner.function.inner;
                    Ok(WithOtherFields::new(ResponsesToolRaw::Function {
                        name: function.name.clone(),
                        description: function.description.clone(),
                        parameters: function.parameters.clone(),
                        strict: function.strict,
                    }))
                }
                ChatCompletionToolsRaw::Custom(custom) => Err(eyre!(
                    "custom tool {:?} is not supported on the responses protocol",
                    custom.inner.custom.inner.name
                )
                .into()),
            })
            .collect()
    }

    fn tool_choice(
        choice: Option<&ChatCompletionToolChoiceOption>,
    ) -> Result<Option<ResponsesToolChoice>, LLMYError> {
        let Some(choice) = choice else {
            return Ok(None);
        };
        let mapped = match &choice.inner {
            ChatCompletionToolChoiceOptionRaw::Mode(mode) => {
                ResponsesToolChoice::Mode(mode.clone())
            }
            ChatCompletionToolChoiceOptionRaw::Function(named) => ResponsesToolChoice::Named(
                WithOtherFields::new(ResponsesNamedToolChoiceRaw::Function {
                    name: named.inner.function.inner.name.clone(),
                }),
            ),
            ChatCompletionToolChoiceOptionRaw::Custom(_)
            | ChatCompletionToolChoiceOptionRaw::AllowedTools(_) => {
                return Err(eyre!(
                    "custom/allowed-tools tool choices are not supported on the responses protocol"
                )
                .into());
            }
        };
        Ok(Some(mapped))
    }

    fn text_config(
        raw: &CreateChatCompletionRequestRaw,
    ) -> Result<Option<ResponsesTextConfig>, LLMYError> {
        let format = match raw.response_format.as_ref().map(|format| &format.inner) {
            None | Some(ResponseFormatRaw::Text) => None,
            Some(ResponseFormatRaw::JsonObject) => {
                Some(WithOtherFields::new(ResponsesTextFormatRaw::JsonObject))
            }
            Some(ResponseFormatRaw::JsonSchema { json_schema }) => {
                Some(WithOtherFields::new(ResponsesTextFormatRaw::JsonSchema {
                    name: json_schema.inner.name.clone(),
                    schema: json_schema.inner.schema.clone(),
                    strict: json_schema.inner.strict,
                    description: json_schema.inner.description.clone(),
                }))
            }
        };
        let wanted = format.is_some() || raw.verbosity.is_some();
        Ok(wanted.then(|| {
            WithOtherFields::new(ResponsesTextConfigRaw {
                format,
                verbosity: raw.verbosity.clone(),
            })
        }))
    }

    /// Build a native prompt request straight from strings — nothing is forced
    /// through the chat-completion shape on its way to the wire.
    pub fn from_prompt(
        model: &str,
        sys_msg: &str,
        user_msg: &str,
        cache_key: Option<&str>,
        settings: &LLMSettings,
    ) -> Result<ResponsesRequest, LLMYError> {
        let mut conversation = Vec::new();
        if !sys_msg.is_empty() {
            conversation.push(Message::system(sys_msg));
        }
        conversation.push(Message::user(user_msg));
        Self::from_conversation(model, &conversation, None, cache_key, settings)
    }

    /// Build the native request straight from a protocol-neutral conversation:
    /// typed parts map onto their own items — reasoning replays with its
    /// `encrypted_content` — so nothing is forced through the chat shape. The
    /// caller's settings map onto the protocol's own fields; a setting with no
    /// slot here (`presence_penalty`) is refused rather than dropped.
    pub fn from_conversation(
        model: &str,
        conversation: &[Message],
        tools: Option<&[ChatCompletionTools]>,
        cache_key: Option<&str>,
        settings: &LLMSettings,
    ) -> Result<ResponsesRequest, LLMYError> {
        if settings.llm_presence_penalty.is_some() {
            return Err(eyre!(
                "the responses protocol does not support presence_penalty; unset it to use this backend"
            )
            .into());
        }
        let tool_choice = match settings.llm_tool_choice.as_ref() {
            Some(choice) => Self::tool_choice(Some(&choice.0))?,
            None => None,
        };
        let tools = tools.map(Self::tools).transpose()?;

        let mut input = Vec::new();
        let mut instructions: Vec<String> = Vec::new();
        for message in conversation {
            // The leading run of system messages takes the protocol's own
            // `instructions` slot (the anthropic protocol hoists system
            // content the same way); anything later stays in `input`.
            if input.is_empty()
                && matches!(message.role, MessageRole::System)
                && !message.cache_breakpoint
                && message
                    .parts
                    .iter()
                    .all(|part| matches!(part, MessagePart::Text { .. }))
            {
                instructions.extend(message.parts.iter().filter_map(|part| match part {
                    MessagePart::Text { text } => Some(text.clone()),
                    _ => None,
                }));
                continue;
            }
            Self::conversation_items(message, &mut input)?;
            // Message-level breakpoints land on the last block, mirroring the
            // chat lowering and the anthropic `cache_control` placement.
            if message.cache_breakpoint {
                Self::mark_last_item_breakpoint(&mut input);
            }
        }

        Ok(WithOtherFields::new(Self {
            model: model.to_string(),
            instructions: (!instructions.is_empty()).then(|| instructions.join("\n\n")),
            input,
            prompt_cache_options: None,
            max_output_tokens: settings.llm_max_completion_tokens,
            temperature: settings.llm_temperature,
            top_p: settings.top_p,
            top_logprobs: None,
            tools,
            tool_choice,
            parallel_tool_calls: None,
            reasoning: settings.reasoning_effort.as_ref().map(|effort| {
                WithOtherFields::new(ResponsesReasoningRaw {
                    effort: Some(effort.0.clone()),
                })
            }),
            text: None,
            prompt_cache_key: cache_key.map(str::to_string),
            safety_identifier: None,
            service_tier: None,
            metadata: None,
            include: Some(vec!["reasoning.encrypted_content".to_string()]),
            store: Some(false),
            stream: None,
        }))
    }

    /// One neutral message into native input items, in part order. Parts
    /// belonging to another protocol (anthropic thinking, foreign opaque
    /// parts) are skipped — their payloads are only valid where they were
    /// issued. The protocol has no explicit cache breakpoints, so the flag is
    /// ignored here.
    fn conversation_items(
        message: &Message,
        input: &mut Vec<ResponsesInputItem>,
    ) -> Result<(), LLMYError> {
        let role = match message.role {
            MessageRole::System => ResponsesRole::System,
            MessageRole::Assistant => ResponsesRole::Assistant,
            MessageRole::User | MessageRole::Tool => ResponsesRole::User,
        };
        let mut parts: Vec<ResponsesInputPart> = Vec::new();
        let flush = |parts: &mut Vec<ResponsesInputPart>, input: &mut Vec<ResponsesInputItem>| {
            if parts.is_empty() {
                return;
            }
            let content = if let [part] = parts.as_slice()
                && let ResponsesInputPartRaw::InputText { text, .. } = &part.inner
            {
                ResponsesInputContent::Text(text.clone())
            } else {
                ResponsesInputContent::Parts(std::mem::take(parts))
            };
            parts.clear();
            input.push(ResponsesInputItemRaw::message(role, content));
        };
        for part in &message.parts {
            match part {
                MessagePart::Text { text } => {
                    parts.push(WithOtherFields::new(ResponsesInputPartRaw::InputText {
                        text: text.clone(),
                        prompt_cache_breakpoint: None,
                    }))
                }
                MessagePart::Image { url } => {
                    parts.push(WithOtherFields::new(ResponsesInputPartRaw::InputImage {
                        image_url: Some(url.clone()),
                        detail: None,
                    }))
                }
                MessagePart::ToolCall {
                    id,
                    name,
                    arguments,
                    extra,
                } => {
                    flush(&mut parts, input);
                    let mut item = WithOtherFields::new(ResponsesInputItemRaw::FunctionCall {
                        call_id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    });
                    item.other = extra.clone();
                    input.push(item);
                }
                MessagePart::ToolResult { id, content } => {
                    flush(&mut parts, input);
                    input.push(WithOtherFields::new(
                        ResponsesInputItemRaw::FunctionCallOutput {
                            call_id: id.clone(),
                            output: content.clone(),
                            prompt_cache_breakpoint: None,
                        },
                    ));
                }
                MessagePart::Reasoning {
                    id,
                    summary,
                    encrypted_content,
                } => {
                    flush(&mut parts, input);
                    input.push(WithOtherFields::new(ResponsesInputItemRaw::Reasoning {
                        id: id.clone(),
                        summary: summary
                            .iter()
                            .map(|text| {
                                WithOtherFields::new(ResponsesReasoningSummaryRaw::SummaryText {
                                    text: text.clone(),
                                })
                            })
                            .collect(),
                        encrypted_content: encrypted_content.clone(),
                    }));
                }
                MessagePart::Opaque { protocol, value } if protocol == RESPONSES_PROTOCOL => {
                    flush(&mut parts, input);
                    input.push(WithOtherFields::new(ResponsesInputItemRaw::Raw(
                        value.clone(),
                    )));
                }
                MessagePart::Thinking { .. }
                | MessagePart::RedactedThinking { .. }
                | MessagePart::Opaque { .. } => {}
            }
        }
        flush(&mut parts, input);
        Ok(())
    }

    /// Prompt-text rendering of one input item, tagged the same way as the
    /// chat renderer so debug records read uniformly.
    fn item_text(item: &ResponsesInputItem) -> String {
        match &item.inner {
            ResponsesInputItemRaw::Message { role, content } => {
                let role = match role {
                    ResponsesRole::System => "SYSTEM",
                    ResponsesRole::Developer => "DEVELOPER",
                    ResponsesRole::User => "USER",
                    ResponsesRole::Assistant => "ASSISTANT",
                };
                let mut body = String::new();
                match content {
                    ResponsesInputContent::Text(text) => body.push_str(text),
                    ResponsesInputContent::Parts(parts) => {
                        for part in parts {
                            match &part.inner {
                                ResponsesInputPartRaw::InputText { text, .. } => {
                                    body.push_str(text)
                                }
                                ResponsesInputPartRaw::InputImage { image_url, .. } => body
                                    .push_str(&format!(
                                        "<img url=\"{}\"/>",
                                        image_url.as_deref().unwrap_or_default()
                                    )),
                                ResponsesInputPartRaw::InputFile {
                                    file_id, filename, ..
                                } => body.push_str(&format!(
                                    "<file name={:?} id={:?}/>",
                                    filename, file_id
                                )),
                                ResponsesInputPartRaw::Raw(_) => {}
                            }
                        }
                    }
                }
                format!("<{}>\n{}\n</{}>\n", role, body, role)
            }
            ResponsesInputItemRaw::FunctionCall {
                call_id,
                name,
                arguments,
            } => format!(
                "<toolcall name=\"{}\" id=\"{}\">\n{}\n</toolcall>\n",
                name, call_id, arguments
            ),
            ResponsesInputItemRaw::FunctionCallOutput {
                call_id, output, ..
            } => format!(
                "<toolresult id=\"{}\">\n{}\n</toolresult>\n",
                call_id, output
            ),
            ResponsesInputItemRaw::Reasoning { .. } | ResponsesInputItemRaw::Raw(_) => {
                String::new()
            }
        }
    }

    /// Prompt-text rendering of the whole input list.
    /// Mark an explicit cache breakpoint on the last input item, where the
    /// protocol can hold one: the final `input_text` part of a message (plain
    /// text content is upgraded to parts form to carry the marker), or a
    /// `function_call_output` directly. Items with no slot (reasoning, raw
    /// passthrough) are left untouched.
    fn mark_last_item_breakpoint(input: &mut Vec<ResponsesInputItem>) {
        let Some(item) = input.last_mut() else {
            return;
        };
        match &mut item.inner {
            ResponsesInputItemRaw::Message { content, .. } => {
                match content {
                    ResponsesInputContent::Text(text) => {
                        *content = ResponsesInputContent::Parts(vec![WithOtherFields::new(
                            ResponsesInputPartRaw::InputText {
                                text: std::mem::take(text),
                                prompt_cache_breakpoint: Some(PromptCacheBreakpointRaw::explicit()),
                            },
                        )]);
                    }
                    ResponsesInputContent::Parts(parts) => {
                        if let Some(part) = parts.iter_mut().rev().find(|part| {
                            matches!(part.inner, ResponsesInputPartRaw::InputText { .. })
                        }) && let ResponsesInputPartRaw::InputText {
                            prompt_cache_breakpoint,
                            ..
                        } = &mut part.inner
                        {
                            *prompt_cache_breakpoint = Some(PromptCacheBreakpointRaw::explicit());
                        }
                    }
                }
            }
            ResponsesInputItemRaw::FunctionCallOutput {
                prompt_cache_breakpoint,
                ..
            } => *prompt_cache_breakpoint = Some(PromptCacheBreakpointRaw::explicit()),
            _ => {}
        }
    }

    /// Whether any block of this input item carries an explicit breakpoint.
    fn item_has_breakpoint(item: &ResponsesInputItem) -> bool {
        match &item.inner {
            ResponsesInputItemRaw::Message { content, .. } => match content {
                ResponsesInputContent::Text(_) => false,
                ResponsesInputContent::Parts(parts) => parts.iter().any(|part| {
                    matches!(
                        &part.inner,
                        ResponsesInputPartRaw::InputText {
                            prompt_cache_breakpoint: Some(_),
                            ..
                        }
                    )
                }),
            },
            ResponsesInputItemRaw::FunctionCallOutput {
                prompt_cache_breakpoint,
                ..
            } => prompt_cache_breakpoint.is_some(),
            _ => false,
        }
    }

    pub fn conversation_text(&self) -> String {
        let mut out = String::new();
        if let Some(instructions) = &self.instructions {
            out.push_str(&format!("<SYSTEM>\n{instructions}\n</SYSTEM>\n"));
        }
        out.extend(self.input.iter().map(Self::item_text));
        out
    }

    /// Tool definitions rendered for the folder debug view.
    pub fn tools_text(&self) -> String {
        self.tools
            .iter()
            .flatten()
            .map(|tool| match &tool.inner {
                ResponsesToolRaw::Function {
                    name,
                    description,
                    parameters,
                    strict,
                } => format!(
                    "<tool name=\"{}\", description=\"{}\", strict={}>\n{}\n</tool>",
                    name,
                    description.clone().unwrap_or_default(),
                    strict.unwrap_or_default(),
                    parameters
                        .as_ref()
                        .and_then(|p| serde_json::to_string_pretty(p).ok())
                        .unwrap_or_default()
                ),
                ResponsesToolRaw::Raw(_) => String::new(),
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

    /// The cache-relevant shape of this request — one block per input item —
    /// so auto cache keys route native requests exactly like chat ones.
    /// Explicit breakpoints (GPT-5.6+) and the request's cache mode carry
    /// over; indices account for the leading `instructions` block.
    pub fn cache_shape(&self) -> CacheShape {
        let offset = usize::from(self.instructions.is_some());
        CacheShape {
            tools_text: serde_json::to_string(&self.tools).unwrap_or_default(),
            message_texts: self
                .instructions
                .iter()
                .map(|instructions| format!("<SYSTEM>\n{instructions}\n</SYSTEM>\n"))
                .chain(self.input.iter().map(Self::item_text))
                .collect(),
            breakpoints: self
                .input
                .iter()
                .enumerate()
                .filter(|(_, item)| Self::item_has_breakpoint(item))
                .map(|(index, _)| index + offset)
                .collect(),
            mode: self
                .prompt_cache_options
                .as_ref()
                .and_then(|options| options.mode)
                .unwrap_or_default(),
        }
    }

    /// Fold this native request back into the chat-completion form — the hub
    /// every cross-protocol conversion routes through, so a request built for
    /// this protocol can still be sent over another backend. `function_call`
    /// items merge back into the assistant message they follow, and
    /// `function_call_output` items become `tool` messages.
    pub fn into_chat_request(self) -> Result<RawExtensibleChatCompletionRequest, LLMYError> {
        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();
        if let Some(instructions) = self.instructions {
            messages.push(WithOtherFields::new(
                ChatCompletionRequestMessageRaw::System(
                    ChatCompletionRequestSystemMessageRaw::new_text(instructions),
                ),
            ));
        }
        for item in self.input {
            match item.into_inner() {
                ResponsesInputItemRaw::Message { role, content } => {
                    Self::chat_message(role, content, &mut messages)?
                }
                ResponsesInputItemRaw::FunctionCall {
                    call_id,
                    name,
                    arguments,
                } => {
                    let call = WithOtherFields::new(ChatCompletionMessageToolCallsRaw::Function(
                        WithOtherFields::new(ChatCompletionMessageToolCallRaw {
                            id: call_id,
                            function: WithOtherFields::new(FunctionCallRaw { name, arguments }),
                        }),
                    ));
                    if let Some(last) = messages.last_mut()
                        && let ChatCompletionRequestMessageRaw::Assistant(assistant) =
                            &mut last.inner
                    {
                        assistant
                            .inner
                            .tool_calls
                            .get_or_insert_with(Vec::new)
                            .push(call);
                    } else {
                        #[allow(deprecated)]
                        let assistant =
                            WithOtherFields::new(ChatCompletionRequestAssistantMessageRaw {
                                content: None,
                                refusal: None,
                                name: None,
                                audio: None,
                                tool_calls: Some(vec![call]),
                                function_call: None,
                            });
                        messages.push(WithOtherFields::new(
                            ChatCompletionRequestMessageRaw::Assistant(assistant),
                        ));
                    }
                }
                ResponsesInputItemRaw::FunctionCallOutput {
                    call_id,
                    output,
                    prompt_cache_breakpoint,
                } => {
                    let mut tool = ChatCompletionRequestMessageRaw::Tool(
                        ChatCompletionRequestToolMessageRaw::new_text(output, call_id),
                    );
                    if prompt_cache_breakpoint.is_some() {
                        tool.toggle_cache_breakpoint(true);
                    }
                    messages.push(WithOtherFields::new(tool))
                }
                // Reasoning replay has no chat slot; degrading it is the whole
                // point of asking for the chat form, so it is dropped rather
                // than refused.
                ResponsesInputItemRaw::Reasoning { .. } => {}
                ResponsesInputItemRaw::Raw(_) => {
                    return Err(eyre!("an unknown input item has no chat-completion form").into());
                }
            }
        }

        let tools = self
            .tools
            .map(|tools| {
                tools
                    .into_iter()
                    .map(|tool| match tool.into_inner() {
                        ResponsesToolRaw::Function {
                            name,
                            description,
                            parameters,
                            strict,
                        } => Ok(WithOtherFields::new(ChatCompletionToolsRaw::Function(
                            WithOtherFields::new(ChatCompletionToolRaw {
                                function: WithOtherFields::new(FunctionObjectRaw {
                                    name,
                                    description,
                                    parameters,
                                    strict,
                                }),
                            }),
                        ))),
                        ResponsesToolRaw::Raw(_) => Err(LLMYError::from(eyre!(
                            "an unknown tool type has no chat-completion form"
                        ))),
                    })
                    .collect::<Result<Vec<_>, LLMYError>>()
            })
            .transpose()?;

        let tool_choice = self.tool_choice.map(|choice| match choice {
            ResponsesToolChoice::Mode(mode) => {
                WithOtherFields::new(ChatCompletionToolChoiceOptionRaw::Mode(mode))
            }
            ResponsesToolChoice::Named(named) => {
                let ResponsesNamedToolChoiceRaw::Function { name } = named.into_inner();
                WithOtherFields::new(ChatCompletionToolChoiceOptionRaw::Function(
                    WithOtherFields::new(ChatCompletionNamedToolChoiceRaw {
                        function: WithOtherFields::new(FunctionNameRaw { name }),
                    }),
                ))
            }
        });

        let (response_format, verbosity) = match self.text {
            None => (None, None),
            Some(text) => {
                let text = text.into_inner();
                let format = text.format.map(|format| match format.into_inner() {
                    ResponsesTextFormatRaw::Text => WithOtherFields::new(ResponseFormatRaw::Text),
                    ResponsesTextFormatRaw::JsonObject => {
                        WithOtherFields::new(ResponseFormatRaw::JsonObject)
                    }
                    ResponsesTextFormatRaw::JsonSchema {
                        name,
                        schema,
                        strict,
                        description,
                    } => WithOtherFields::new(ResponseFormatRaw::JsonSchema {
                        json_schema: WithOtherFields::new(ResponseFormatJsonSchemaRaw {
                            description,
                            name,
                            schema,
                            strict,
                        }),
                    }),
                });
                (format, text.verbosity)
            }
        };

        let raw = CreateChatCompletionRequestRaw {
            messages,
            model: self.model,
            verbosity,
            reasoning_effort: self.reasoning.and_then(|r| r.into_inner().effort),
            max_completion_tokens: self.max_output_tokens,
            top_logprobs: self.top_logprobs,
            response_format,
            store: self.store,
            stream: self.stream,
            service_tier: self.service_tier,
            temperature: self.temperature,
            top_p: self.top_p,
            tools,
            tool_choice,
            parallel_tool_calls: self.parallel_tool_calls,
            safety_identifier: self.safety_identifier,
            prompt_cache_key: self.prompt_cache_key,
            prompt_cache_options: self.prompt_cache_options,
            metadata: self.metadata,
            ..Default::default()
        };
        Ok(RawExtensibleChatCompletionRequest::new(raw))
    }

    fn chat_message(
        role: ResponsesRole,
        content: ResponsesInputContent,
        out: &mut Vec<ChatCompletionRequestMessage>,
    ) -> Result<(), LLMYError> {
        let mapped = match role {
            ResponsesRole::System => ChatCompletionRequestMessageRaw::System(
                ChatCompletionRequestSystemMessageRaw::new_text(Self::text_only(
                    content, "system",
                )?),
            ),
            ResponsesRole::Developer => ChatCompletionRequestMessageRaw::Developer(
                ChatCompletionRequestDeveloperMessageRaw::new_text(Self::text_only(
                    content,
                    "developer",
                )?),
            ),
            ResponsesRole::Assistant => ChatCompletionRequestMessageRaw::Assistant(
                ChatCompletionRequestAssistantMessageRaw::new_text(Self::text_only(
                    content,
                    "assistant",
                )?),
            ),
            ResponsesRole::User => {
                let content = match content {
                    ResponsesInputContent::Text(text) => {
                        ChatCompletionRequestUserMessageContent::Text(text)
                    }
                    ResponsesInputContent::Parts(parts) => {
                        let parts = parts
                            .into_iter()
                            .map(|part| {
                                let mapped = match part.into_inner() {
                                    ResponsesInputPartRaw::InputText {
                                        text,
                                        prompt_cache_breakpoint,
                                    } => ChatCompletionRequestUserMessageContentPartRaw::Text(
                                        WithOtherFields::new(
                                            ChatCompletionRequestMessageContentPartTextRaw {
                                                text,
                                                prompt_cache_breakpoint,
                                            },
                                        ),
                                    ),
                                    ResponsesInputPartRaw::InputImage { image_url, detail } => {
                                        let url = image_url.ok_or_else(|| {
                                            eyre!(
                                                "an image without a url has no chat-completion form"
                                            )
                                        })?;
                                        ChatCompletionRequestUserMessageContentPartRaw::ImageUrl(
                                            WithOtherFields::new(
                                                ChatCompletionRequestMessageContentPartImageRaw {
                                                    image_url: WithOtherFields::new(ImageUrlRaw {
                                                        url,
                                                        detail,
                                                    }),
                                                    prompt_cache_breakpoint: None,
                                                },
                                            ),
                                        )
                                    }
                                    ResponsesInputPartRaw::InputFile {
                                        file_id,
                                        file_data,
                                        filename,
                                    } => ChatCompletionRequestUserMessageContentPartRaw::File(
                                        WithOtherFields::new(
                                            ChatCompletionRequestMessageContentPartFileRaw {
                                                file: WithOtherFields::new(FileObjectRaw {
                                                    file_data,
                                                    file_id,
                                                    filename,
                                                }),
                                                prompt_cache_breakpoint: None,
                                            },
                                        ),
                                    ),
                                    ResponsesInputPartRaw::Raw(_) => {
                                        return Err(LLMYError::from(eyre!(
                                            "an unknown input part has no chat-completion form"
                                        )));
                                    }
                                };
                                Ok(WithOtherFields::new(mapped))
                            })
                            .collect::<Result<Vec<_>, LLMYError>>()?;
                        let mut content = ChatCompletionRequestUserMessageContent::Array(parts);
                        content.compact();
                        content
                    }
                };
                ChatCompletionRequestMessageRaw::User(WithOtherFields::new(
                    ChatCompletionRequestUserMessageRaw {
                        content,
                        name: None,
                    },
                ))
            }
        };
        out.push(WithOtherFields::new(mapped));
        Ok(())
    }

    /// Join a message's parts into plain text; system/developer/assistant chat
    /// messages carry text only.
    fn text_only(content: ResponsesInputContent, role: &str) -> Result<String, LLMYError> {
        match content {
            ResponsesInputContent::Text(text) => Ok(text),
            ResponsesInputContent::Parts(parts) => parts
                .into_iter()
                .map(|part| match part.into_inner() {
                    ResponsesInputPartRaw::InputText { text, .. } => Ok(text),
                    other => Err(LLMYError::from(eyre!(
                        "a non-text part in a {} message has no chat-completion form: {:?}",
                        role,
                        other
                    ))),
                })
                .collect::<Result<Vec<_>, _>>()
                .map(|texts| texts.join("\n")),
        }
    }
}

// ---------------------------------------------------------------------------
// Responses response -> chat response
// ---------------------------------------------------------------------------

impl ResponsesResponseRaw {
    /// The assistant turn of this response in protocol-neutral form — typed
    /// parts, encrypted reasoning included, ready to go back into conversation
    /// state. Item-level extras ride the tool-call part (with the item's own
    /// `id`), so a replayed `function_call` reproduces them.
    pub fn to_message(&self) -> Message {
        let mut parts = Vec::new();
        for item in &self.output {
            match &item.inner {
                ResponsesOutputItemRaw::Message { content, .. } => {
                    for part in content {
                        if let ResponsesOutputContentRaw::OutputText { text } = &part.inner {
                            parts.push(MessagePart::Text { text: text.clone() });
                        }
                    }
                }
                ResponsesOutputItemRaw::FunctionCall {
                    id,
                    call_id,
                    name,
                    arguments,
                } => {
                    let mut extra = item.other.clone();
                    if let Some(item_id) = id {
                        extra.insert("id".to_string(), Value::String(item_id.clone()));
                    }
                    parts.push(MessagePart::ToolCall {
                        id: call_id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                        extra,
                    });
                }
                ResponsesOutputItemRaw::Reasoning {
                    id,
                    summary,
                    encrypted_content,
                } => parts.push(MessagePart::Reasoning {
                    id: id.clone(),
                    summary: summary
                        .iter()
                        .filter_map(|part| match &part.inner {
                            ResponsesReasoningSummaryRaw::SummaryText { text } => {
                                Some(text.clone())
                            }
                            ResponsesReasoningSummaryRaw::Raw(_) => None,
                        })
                        .collect(),
                    encrypted_content: encrypted_content.clone(),
                }),
                ResponsesOutputItemRaw::Raw(value) => parts.push(MessagePart::Opaque {
                    protocol: RESPONSES_PROTOCOL.to_string(),
                    value: value.clone(),
                }),
            }
        }
        Message::new(MessageRole::Assistant, parts)
    }

    /// Fold a Responses reply back into the chat-completion shape the rest of
    /// the crate consumes. `function_call` items become `tool_calls` (their
    /// `call_id` becomes the chat tool-call id, so tool results round-trip),
    /// reasoning summaries land in the message's `reasoning_content` extra,
    /// and an errored/failed response surfaces as an error instead of an empty
    /// completion.
    pub fn into_chat_response(self) -> Result<RawExtensibleChatCompletionResponse, LLMYError> {
        if let Some(error) = &self.error {
            return Err(eyre!(
                "responses api error {:?}: {}",
                error.inner.code,
                error.inner.message
            )
            .into());
        }
        if self.status == "failed" {
            return Err(eyre!("responses api request failed without error detail").into());
        }

        let mut text = String::new();
        let mut refusal: Option<String> = None;
        let mut reasoning = String::new();
        let mut tool_calls: Vec<ChatCompletionMessageToolCalls> = Vec::new();
        for item in self.output {
            match item.inner {
                ResponsesOutputItemRaw::Message { content, .. } => {
                    for part in content {
                        match part.inner {
                            ResponsesOutputContentRaw::OutputText { text: chunk } => {
                                text.push_str(&chunk)
                            }
                            ResponsesOutputContentRaw::Refusal { refusal: message } => {
                                refusal = Some(message)
                            }
                            ResponsesOutputContentRaw::Raw(_) => {}
                        }
                    }
                }
                ResponsesOutputItemRaw::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => {
                    tool_calls.push(WithOtherFields::new(
                        ChatCompletionMessageToolCallsRaw::Function(WithOtherFields::new(
                            ChatCompletionMessageToolCallRaw {
                                id: call_id,
                                function: WithOtherFields::new(FunctionCallRaw { name, arguments }),
                            },
                        )),
                    ));
                }
                ResponsesOutputItemRaw::Reasoning { summary, .. } => {
                    for part in summary {
                        if let ResponsesReasoningSummaryRaw::SummaryText { text: chunk } =
                            part.inner
                        {
                            if !reasoning.is_empty() {
                                reasoning.push('\n');
                            }
                            reasoning.push_str(&chunk);
                        }
                    }
                }
                ResponsesOutputItemRaw::Raw(_) => {}
            }
        }

        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolCalls
        } else if self.status == "incomplete" {
            match self
                .incomplete_details
                .as_ref()
                .and_then(|details| details.inner.reason.as_deref())
            {
                Some("content_filter") => FinishReason::ContentFilter,
                // `max_output_tokens` is the only other reason the API
                // documents; length is also the only sensible reading of an
                // unknown one.
                _ => FinishReason::Length,
            }
        } else {
            FinishReason::Stop
        };

        #[allow(deprecated)]
        let mut message = WithOtherFields::new(ChatCompletionResponseMessageRaw {
            content: (!text.is_empty()).then_some(text),
            refusal,
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

        let usage = self.usage.map(|usage| {
            let usage = usage.inner;
            let cached = usage
                .input_tokens_details
                .as_ref()
                .and_then(|details| details.inner.cached_tokens);
            let reasoning_tokens = usage
                .output_tokens_details
                .as_ref()
                .and_then(|details| details.inner.reasoning_tokens);
            WithOtherFields::new(CompletionUsageRaw {
                prompt_tokens: usage.input_tokens,
                completion_tokens: usage.output_tokens,
                total_tokens: if usage.total_tokens > 0 {
                    usage.total_tokens
                } else {
                    usage.input_tokens.saturating_add(usage.output_tokens)
                },
                prompt_tokens_details: Some(WithOtherFields::new(PromptTokensDetailsRaw {
                    audio_tokens: None,
                    cached_tokens: cached,
                    cache_write_tokens: None,
                })),
                completion_tokens_details: Some(WithOtherFields::new(CompletionTokensDetailsRaw {
                    accepted_prediction_tokens: None,
                    audio_tokens: None,
                    reasoning_tokens,
                    rejected_prediction_tokens: None,
                })),
            })
        });

        #[allow(deprecated)]
        let raw = CreateChatCompletionResponseRaw {
            id: self.id,
            choices: vec![WithOtherFields::new(ChatChoiceRaw {
                index: 0,
                message,
                finish_reason: Some(finish_reason),
                logprobs: None,
            })],
            created: self.created_at as u32,
            model: self.model,
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion".to_string(),
            usage,
        };
        Ok(RawExtensibleChatCompletionResponse::new(raw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::req::{
        ChatCompletionRequestMessageRaw, ChatCompletionRequestSystemMessageRaw,
        ChatCompletionRequestToolMessageRaw, ChatCompletionRequestUserMessageRaw,
    };

    fn chat_request(
        messages: Vec<ChatCompletionRequestMessageRaw>,
    ) -> RawExtensibleChatCompletionRequest {
        let mut raw = CreateChatCompletionRequestRaw::default();
        raw.model = "gpt-test".to_string();
        raw.messages = messages.into_iter().map(WithOtherFields::new).collect();
        RawExtensibleChatCompletionRequest::new(raw)
    }

    fn test_settings() -> LLMSettings {
        LLMSettings {
            llm_temperature: None,
            llm_presence_penalty: None,
            llm_prompt_timeout: 0,
            llm_retry: 1,
            llm_max_completion_tokens: None,
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
            llm_concurrent: 0,
        }
    }

    #[test]
    fn explicit_cache_fields_map_onto_the_responses_wire() {
        let req: RawExtensibleChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-test",
            "prompt_cache_options": {"mode": "explicit"},
            "messages": [
                {"role": "system", "content": [
                    {"type": "text", "text": "sys", "prompt_cache_breakpoint": {"mode": "explicit"}}
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": "hi", "prompt_cache_breakpoint": {"mode": "explicit"}}
                ]}
            ]
        }))
        .unwrap();

        let native = ResponsesRequestRaw::from_chat(&req).unwrap();
        let value = serde_json::to_value(&native).unwrap();
        assert_eq!(value["prompt_cache_options"]["mode"], "explicit");
        // A breakpointed system message cannot live in `instructions` (the
        // slot holds no markers), so it stays in `input`, in parts form.
        assert!(value.get("instructions").is_none(), "{value:?}");
        assert_eq!(value["input"][0]["role"], "system");
        assert_eq!(
            value["input"][0]["content"][0]["prompt_cache_breakpoint"]["mode"],
            "explicit"
        );
        assert_eq!(
            value["input"][1]["content"][0]["prompt_cache_breakpoint"]["mode"],
            "explicit"
        );

        // The user-part marker and the options survive the trip back to chat.
        let chat = native.into_inner().into_chat_request().unwrap();
        let chat_value = serde_json::to_value(&chat).unwrap();
        assert_eq!(chat_value["prompt_cache_options"]["mode"], "explicit");
        assert_eq!(
            chat_value["messages"][1]["content"][0]["prompt_cache_breakpoint"]["mode"],
            "explicit"
        );
    }

    #[test]
    fn a_message_breakpoint_lands_on_the_last_input_block() {
        let mut user = Message::user("hello");
        user.breakpoint();
        let conversation = vec![Message::system("sys"), user];
        let req = ResponsesRequestRaw::from_conversation(
            "gpt-test",
            &conversation,
            None,
            None,
            &test_settings(),
        )
        .unwrap();
        let value = serde_json::to_value(&req).unwrap();
        // The breakpoint-free system prompt still takes `instructions`...
        assert_eq!(value["instructions"], "sys");
        // ...and the user message's breakpoint lands on its last text block.
        assert_eq!(
            value["input"][0]["content"][0]["prompt_cache_breakpoint"]["mode"],
            "explicit"
        );

        // The cache shape sees it, indexed past the instructions block.
        let shape = req.cache_shape();
        assert_eq!(shape.breakpoints, vec![1]);
    }

    #[test]
    fn a_conversation_converts_to_responses_input_items() {
        let mut req = chat_request(vec![
            ChatCompletionRequestMessageRaw::System(
                ChatCompletionRequestSystemMessageRaw::new_text("be terse"),
            ),
            ChatCompletionRequestMessageRaw::User(ChatCompletionRequestUserMessageRaw::new_text(
                "look it up",
            )),
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": "on it",
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{\"q\": 1}"}}
                ]
            }))
            .map(|m: crate::req::ChatCompletionRequestAssistantMessage| {
                ChatCompletionRequestMessageRaw::Assistant(m)
            })
            .unwrap(),
            ChatCompletionRequestMessageRaw::Tool(ChatCompletionRequestToolMessageRaw::new_text(
                "found it", "call_1",
            )),
        ]);
        req.max_completion_tokens = Some(200);
        req.prompt_cache_key = Some("key-1".to_string());
        req.tools = Some(vec![serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {"name": "lookup", "description": "d", "parameters": {"type": "object"}, "strict": true}
        }))
        .unwrap()]);
        req.tool_choice = Some(WithOtherFields::new(
            ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Auto),
        ));
        req.reasoning_effort = Some(ReasoningEffort::Low);

        let value = serde_json::to_value(ResponsesRequestRaw::from_chat(&req).unwrap()).unwrap();

        assert_eq!(value["model"], "gpt-test");
        assert_eq!(value["max_output_tokens"], 200);
        assert_eq!(value["prompt_cache_key"], "key-1");
        assert_eq!(value["reasoning"]["effort"], "low");
        // Stateless by default, unlike the API's own default.
        assert_eq!(value["store"], false);
        assert_eq!(value["tool_choice"], "auto");
        // Responses tools are flat.
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["name"], "lookup");
        assert_eq!(value["tools"][0]["strict"], true);
        assert!(value["tools"][0].get("function").is_none());

        // The leading system message takes the `instructions` slot instead
        // of an input item.
        assert!(value["instructions"].is_string(), "{value:?}");
        let input = value["input"].as_array().unwrap();
        assert_eq!(input.len(), 4, "{input:?}");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "look it up");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"], "on it");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["output"], "found it");
    }

    #[test]
    fn unsupported_chat_fields_are_refused_not_dropped() {
        let mut req = chat_request(vec![ChatCompletionRequestMessageRaw::User(
            ChatCompletionRequestUserMessageRaw::new_text("hi"),
        )]);
        req.stop = Some(crate::req::StopConfiguration::String("END".to_string()));
        req.presence_penalty = Some(0.5);
        let err = ResponsesRequestRaw::from_chat(&req).unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.contains("stop"), "{rendered}");
        assert!(rendered.contains("presence_penalty"), "{rendered}");
    }

    #[test]
    fn a_response_folds_back_into_a_chat_completion() {
        let resp: ResponsesResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_1",
            "object": "response",
            "created_at": 1234.0,
            "status": "completed",
            "model": "gpt-test",
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "hmm"}]},
                {"type": "message", "id": "msg_1", "role": "assistant",
                 "content": [{"type": "output_text", "text": "calling"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_9", "name": "lookup", "arguments": "{\"q\": 1}"},
                {"type": "web_search_call", "id": "ws_1", "status": "completed"}
            ],
            "usage": {
                "input_tokens": 12,
                "input_tokens_details": {"cached_tokens": 4},
                "output_tokens": 6,
                "output_tokens_details": {"reasoning_tokens": 2},
                "total_tokens": 18
            }
        }))
        .unwrap();

        let chat = resp.into_inner().into_chat_response().unwrap();
        assert_eq!(chat.id, "resp_1");
        assert_eq!(chat.created, 1234);
        let choice = &chat.choices[0].inner;
        assert_eq!(choice.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(choice.message.inner.content.as_deref(), Some("calling"));
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
                // The chat id is the round-trippable call_id.
                assert_eq!(call.inner.id, "call_9");
                assert_eq!(call.inner.function.inner.name, "lookup");
            }
            other => panic!("expected function tool call, got {other:?}"),
        }
        let usage = chat.usage.as_ref().unwrap();
        assert_eq!(usage.inner.prompt_tokens, 12);
        assert_eq!(usage.inner.completion_tokens, 6);
        assert_eq!(usage.inner.total_tokens, 18);
        assert_eq!(
            usage
                .inner
                .prompt_tokens_details
                .as_ref()
                .unwrap()
                .inner
                .cached_tokens,
            Some(4)
        );
        assert_eq!(
            usage
                .inner
                .completion_tokens_details
                .as_ref()
                .unwrap()
                .inner
                .reasoning_tokens,
            Some(2)
        );
    }

    #[test]
    fn an_incomplete_response_maps_to_a_length_finish() {
        let resp: ResponsesResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_1", "object": "response", "created_at": 1.0, "status": "incomplete",
            "model": "gpt-test",
            "output": [{"type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "trunc"}]}],
            "incomplete_details": {"reason": "max_output_tokens"}
        }))
        .unwrap();
        let chat = resp.into_inner().into_chat_response().unwrap();
        assert_eq!(
            chat.choices[0].inner.finish_reason,
            Some(FinishReason::Length)
        );
    }

    #[test]
    fn an_errored_response_surfaces_as_an_error() {
        let resp: ResponsesResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_1", "object": "response", "created_at": 1.0, "status": "failed",
            "model": "gpt-test", "output": [],
            "error": {"code": "server_error", "message": "boom"}
        }))
        .unwrap();
        let err = resp.into_inner().into_chat_response().unwrap_err();
        assert!(err.to_string().contains("boom"), "{err}");
    }

    #[test]
    fn stream_terminal_events_parse_and_deltas_are_ignored() {
        let completed: ResponsesStreamEvent = serde_json::from_value(serde_json::json!({
            "type": "response.completed",
            "response": {"id": "resp_1", "object": "response", "created_at": 1.0,
                         "status": "completed", "model": "gpt-test", "output": []}
        }))
        .unwrap();
        assert!(matches!(completed, ResponsesStreamEvent::Completed { .. }));

        let delta: ResponsesStreamEvent = serde_json::from_value(serde_json::json!({
            "type": "response.output_text.delta", "delta": "hel"
        }))
        .unwrap();
        assert_eq!(delta, ResponsesStreamEvent::Other);
    }

    #[test]
    fn reasoning_items_survive_the_context_round_trip() {
        // Stateless requests ask for the encrypted reasoning payload back.
        let req = chat_request(vec![ChatCompletionRequestMessageRaw::User(
            ChatCompletionRequestUserMessageRaw::new_text("hi"),
        )]);
        let value = serde_json::to_value(ResponsesRequestRaw::from_chat(&req).unwrap()).unwrap();
        assert_eq!(
            value["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );

        // A reply carrying a reasoning item parses into typed parts...
        let resp: ResponsesResponse = serde_json::from_value(serde_json::json!({
            "id": "resp_1", "object": "response", "created_at": 1.0,
            "status": "completed", "model": "gpt-test",
            "output": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-1",
                 "summary": [{"type": "summary_text", "text": "hmm"}]},
                {"type": "message", "id": "msg_1", "role": "assistant",
                 "content": [{"type": "output_text", "text": "on it"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "lookup", "arguments": "{\"q\":1}"}
            ]
        }))
        .unwrap();
        let assistant = resp.to_message();
        assert!(matches!(
            &assistant.parts[0],
            MessagePart::Reasoning { id: Some(id), encrypted_content: Some(enc), .. }
                if id == "rs_1" && enc == "enc-1"
        ));

        // ...and rebuilding the next turn replays them, encrypted content and
        // item ids included.
        let conversation = vec![
            Message::user("look it up"),
            assistant,
            Message::tool_result("call_1", "found"),
        ];
        let settings = test_settings();
        let value = serde_json::to_value(
            ResponsesRequestRaw::from_conversation(
                "gpt-test",
                &conversation,
                None,
                None,
                &settings,
            )
            .unwrap(),
        )
        .unwrap();
        let input = value["input"].as_array().unwrap();
        assert_eq!(input.len(), 5, "{input:?}");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[1]["encrypted_content"], "enc-1");
        assert_eq!(input[1]["summary"][0]["text"], "hmm");
        assert_eq!(input[2]["type"], "message");
        assert_eq!(input[2]["role"], "assistant");
        assert_eq!(input[2]["content"], "on it");
        assert_eq!(input[3]["type"], "function_call");
        assert_eq!(input[3]["call_id"], "call_1");
        // The item's own id rode the part extras and lands back on the item.
        assert_eq!(input[3]["id"], "fc_1");
        assert_eq!(input[4]["type"], "function_call_output");
        assert_eq!(input[4]["output"], "found");
        assert_eq!(
            value["include"],
            serde_json::json!(["reasoning.encrypted_content"])
        );
    }

    #[test]
    fn a_native_request_folds_back_into_chat_for_other_backends() {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-test",
            "instructions": "stay factual",
            "input": [
                {"type": "message", "role": "system", "content": "be terse"},
                {"type": "message", "role": "user", "content": "look it up"},
                {"type": "message", "role": "assistant", "content": "on it"},
                {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\": 1}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "found"}
            ],
            "max_output_tokens": 200,
            "reasoning": {"effort": "low"},
            "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object"}, "strict": true}],
            "tool_choice": "auto",
            "prompt_cache_key": "key-1"
        }))
        .unwrap();

        let chat = req.into_inner().into_chat_request().unwrap();
        let value = serde_json::to_value(&chat).unwrap();
        let messages = value["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 5, "{messages:?}");
        // `instructions` folds back in front as a system message, ahead of
        // the input's own system item.
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "stay factual");
        assert_eq!(messages[1]["role"], "system");
        assert_eq!(messages[1]["content"], "be terse");
        assert_eq!(messages[2]["role"], "user");
        assert_eq!(messages[2]["content"], "look it up");
        // The function_call item merges back into the assistant message.
        assert_eq!(messages[3]["role"], "assistant");
        assert_eq!(messages[3]["content"], "on it");
        assert_eq!(messages[3]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[4]["role"], "tool");
        assert_eq!(messages[4]["content"], "found");

        assert_eq!(value["max_completion_tokens"], 200);
        assert_eq!(value["reasoning_effort"], "low");
        assert_eq!(value["tools"][0]["function"]["name"], "lookup");
        assert_eq!(value["tools"][0]["function"]["strict"], true);
        assert_eq!(value["tool_choice"], "auto");
        assert_eq!(value["prompt_cache_key"], "key-1");
    }
}
