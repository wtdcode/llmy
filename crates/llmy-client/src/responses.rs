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

use color_eyre::eyre::eyre;
use llmy_types::{error::LLMYError, other::WithOtherFields};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cache_key::CacheShape;
use crate::req::PromptCacheMode;
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
use crate::resp::{
    ChatChoiceRaw, ChatCompletionMessageToolCalls, ChatCompletionResponseMessageRaw,
    CompletionTokensDetailsRaw, CompletionUsageRaw, CreateChatCompletionResponseRaw, FinishReason,
    PromptTokensDetailsRaw, RawExtensibleChatCompletionResponse,
};
use crate::settings::LLMSettings;

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
    #[serde(other)]
    Unknown,
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
    },
    #[serde(other)]
    Unknown,
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
    #[serde(other)]
    Unknown,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponsesTextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
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
    #[serde(other)]
    Unknown,
}
pub type ResponsesOutputContent = WithOtherFields<ResponsesOutputContentRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesReasoningSummaryRaw {
    SummaryText {
        text: String,
    },
    #[serde(other)]
    Unknown,
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
    },
    /// Output item kinds this client does not model (built-in tool calls
    /// etc.). Tolerated on deserialize, ignored by the conversion.
    #[serde(other)]
    Unknown,
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
        for msg in &raw.messages {
            match &msg.inner {
                ChatCompletionRequestMessageRaw::System(m) => {
                    input.push(ResponsesInputItemRaw::message(
                        ResponsesRole::System,
                        ResponsesInputContent::Text(Self::system_text(&m.inner.content)),
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
            input,
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
        if !text.is_empty() {
            input.push(ResponsesInputItemRaw::message(
                ResponsesRole::Assistant,
                ResponsesInputContent::Text(text),
            ));
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
    /// through the chat-completion shape on its way to the wire. The caller's
    /// settings map onto the protocol's own fields; a setting with no slot
    /// here (`presence_penalty`) is refused rather than dropped.
    pub fn from_prompt(
        model: &str,
        sys_msg: &str,
        user_msg: &str,
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
        let mut input = Vec::new();
        if !sys_msg.is_empty() {
            input.push(ResponsesInputItemRaw::message(
                ResponsesRole::System,
                ResponsesInputContent::Text(sys_msg.to_string()),
            ));
        }
        input.push(ResponsesInputItemRaw::message(
            ResponsesRole::User,
            ResponsesInputContent::Text(user_msg.to_string()),
        ));
        Ok(WithOtherFields::new(Self {
            model: model.to_string(),
            input,
            max_output_tokens: settings.llm_max_completion_tokens,
            temperature: settings.llm_temperature,
            top_p: settings.top_p,
            top_logprobs: None,
            tools: None,
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
            store: Some(false),
            stream: None,
        }))
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
                                ResponsesInputPartRaw::InputText { text } => body.push_str(text),
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
                                ResponsesInputPartRaw::Unknown => {}
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
            ResponsesInputItemRaw::FunctionCallOutput { call_id, output } => format!(
                "<toolresult id=\"{}\">\n{}\n</toolresult>\n",
                call_id, output
            ),
            ResponsesInputItemRaw::Unknown => String::new(),
        }
    }

    /// Prompt-text rendering of the whole input list.
    pub fn conversation_text(&self) -> String {
        self.input.iter().map(Self::item_text).collect()
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
                ResponsesToolRaw::Unknown => String::new(),
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
    /// so auto cache keys route native requests exactly like chat ones. The
    /// protocol has no explicit breakpoints; the provider caches prefixes
    /// transparently, which the default implicit mode models.
    pub fn cache_shape(&self) -> CacheShape {
        CacheShape {
            tools_text: serde_json::to_string(&self.tools).unwrap_or_default(),
            message_texts: self.input.iter().map(Self::item_text).collect(),
            breakpoints: vec![],
            mode: PromptCacheMode::default(),
        }
    }

    /// Fold this native request back into the chat-completion form — the hub
    /// every cross-protocol conversion routes through, so a request built for
    /// this protocol can still be sent over another backend. `function_call`
    /// items merge back into the assistant message they follow, and
    /// `function_call_output` items become `tool` messages.
    pub fn into_chat_request(self) -> Result<RawExtensibleChatCompletionRequest, LLMYError> {
        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();
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
                ResponsesInputItemRaw::FunctionCallOutput { call_id, output } => {
                    messages.push(WithOtherFields::new(ChatCompletionRequestMessageRaw::Tool(
                        ChatCompletionRequestToolMessageRaw::new_text(output, call_id),
                    )))
                }
                ResponsesInputItemRaw::Unknown => {
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
                        ResponsesToolRaw::Unknown => Err(LLMYError::from(eyre!(
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
                                    ResponsesInputPartRaw::InputText { text } => {
                                        ChatCompletionRequestUserMessageContentPartRaw::Text(
                                            WithOtherFields::new(
                                                ChatCompletionRequestMessageContentPartTextRaw {
                                                    text,
                                                    prompt_cache_breakpoint: None,
                                                },
                                            ),
                                        )
                                    }
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
                                    ResponsesInputPartRaw::Unknown => {
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
                    ResponsesInputPartRaw::InputText { text } => Ok(text),
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
                            ResponsesOutputContentRaw::Unknown => {}
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
                ResponsesOutputItemRaw::Unknown => {}
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

        let input = value["input"].as_array().unwrap();
        assert_eq!(input.len(), 5, "{input:?}");
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[1]["role"], "user");
        assert_eq!(input[1]["content"], "look it up");
        assert_eq!(input[2]["role"], "assistant");
        assert_eq!(input[2]["content"], "on it");
        assert_eq!(input[3]["type"], "function_call");
        assert_eq!(input[3]["call_id"], "call_1");
        assert_eq!(input[4]["type"], "function_call_output");
        assert_eq!(input[4]["output"], "found it");
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
    fn a_native_request_folds_back_into_chat_for_other_backends() {
        let req: ResponsesRequest = serde_json::from_value(serde_json::json!({
            "model": "gpt-test",
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
        assert_eq!(messages.len(), 4, "{messages:?}");
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be terse");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "look it up");
        // The function_call item merges back into the assistant message.
        assert_eq!(messages[2]["role"], "assistant");
        assert_eq!(messages[2]["content"], "on it");
        assert_eq!(messages[2]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[3]["role"], "tool");
        assert_eq!(messages[3]["content"], "found");

        assert_eq!(value["max_completion_tokens"], 200);
        assert_eq!(value["reasoning_effort"], "low");
        assert_eq!(value["tools"][0]["function"]["name"], "lookup");
        assert_eq!(value["tools"][0]["function"]["strict"], true);
        assert_eq!(value["tool_choice"], "auto");
        assert_eq!(value["prompt_cache_key"], "key-1");
    }
}
