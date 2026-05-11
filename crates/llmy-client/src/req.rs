//! Mirror of the chat-completion request types from `async-openai` 0.36.1, with every non-trivial
//! struct/enum wrapped in [`WithOtherFields`] so unknown JSON fields survive a deserialize →
//! serialize round trip.
//!
//! This file is intentionally self-contained: it does not depend on `async_openai`. The wire
//! format aims to be byte-for-byte compatible with the OpenAI chat completions REST API.

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use llmy_types::other::{OtherFields, WithOtherFields};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Set of 16 key-value pairs that can be attached to an object.
/// This can be useful for storing additional information about the
/// object in a structured format, and querying for objects via API
/// or the dashboard. Keys are strings with a maximum length of 64
/// characters. Values are strings with a maximum length of 512
/// characters.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
#[serde(transparent)]
pub struct Metadata(pub serde_json::Value);

impl From<serde_json::Value> for Metadata {
    fn from(value: serde_json::Value) -> Self {
        Self(value)
    }
}

// ---------------------------------------------------------------------------
// Role
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, Copy, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    #[default]
    User,
    Assistant,
    Tool,
    Function,
}

// ---------------------------------------------------------------------------
// Reasoning effort
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    #[default]
    Medium,
    High,
    Xhigh,
}

// ---------------------------------------------------------------------------
// Verbosity
// ---------------------------------------------------------------------------

/// Constrains the verbosity of the model's response. Lower values will result in more concise responses, while higher values will result in more verbose responses. Currently supported values are `low`, `medium`, and `high`.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    Low,
    #[default]
    Medium,
    High,
}

// ---------------------------------------------------------------------------
// ResponseModalities
// ---------------------------------------------------------------------------

/// Output types that you would like the model to generate for this request.
///
/// Most models are capable of generating text, which is the default: `["text"]`
///
/// The `gpt-4o-audio-preview` model can also be used to [generate
/// audio](https://platform.openai.com/docs/guides/audio). To request that this model generate both text and audio responses, you can use: `["text", "audio"]`
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ResponseModalities {
    Text,
    Audio,
}

// ---------------------------------------------------------------------------
// Service tier
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServiceTier {
    Auto,
    Default,
    Flex,
    Scale,
    Priority,
}

// ---------------------------------------------------------------------------
// Stop configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum StopConfiguration {
    String(String),           // nullable: true
    StringArray(Vec<String>), // minItems: 1; maxItems: 4
}

// ---------------------------------------------------------------------------
// Image / detail / url
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
    Original,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ImageUrlRaw {
    /// Either a URL of the image or the base64 encoded image data.
    pub url: String,
    /// Specifies the detail level of the image. Learn more in the [Vision guide](https://platform.openai.com/docs/guides/vision/low-or-high-fidelity-image-understanding).
    pub detail: Option<ImageDetail>,
}
pub type ImageUrl = WithOtherFields<ImageUrlRaw>;

// ---------------------------------------------------------------------------
// Input audio
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum InputAudioFormat {
    Wav,
    #[default]
    Mp3,
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct InputAudioRaw {
    /// Base64 encoded audio data.
    pub data: String,
    /// The format of the encoded audio data. Currently supports "wav" and "mp3".
    pub format: InputAudioFormat,
}
pub type InputAudio = WithOtherFields<InputAudioRaw>;

// ---------------------------------------------------------------------------
// File object
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct FileObjectRaw {
    /// The base64 encoded file data, used when passing the file to the model
    /// as a string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    /// The ID of an uploaded file to use as input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    /// The name of the file, used when passing the file to the model as a
    /// string.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
}
pub type FileObject = WithOtherFields<FileObjectRaw>;

// ---------------------------------------------------------------------------
// Function object / call
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct FunctionObjectRaw {
    /// The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    pub name: String,
    /// A description of what the function does, used by the model to choose when and how to call the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The parameters the functions accepts, described as a JSON Schema object. See the [guide](https://platform.openai.com/docs/guides/text-generation/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format.
    ///
    /// Omitting `parameters` defines a function with an empty parameter list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,

    /// Whether to enable strict schema adherence when generating the function call. If set to true, the model will follow the exact schema defined in the `parameters` field. Only a subset of JSON Schema is supported when `strict` is `true`. Learn more about Structured Outputs in the [function calling guide](https://platform.openai.com/docs/guides/function-calling).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}
pub type FunctionObject = WithOtherFields<FunctionObjectRaw>;

/// The name and arguments of a function that should be called, as generated by the model.
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Default)]
pub struct FunctionCallRaw {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to call the function with, as generated by the model in JSON format. Note that the model does not always generate valid JSON, and may hallucinate parameters not defined by your function schema. Validate the arguments in your code before calling your function.
    pub arguments: String,
}
pub type FunctionCall = WithOtherFields<FunctionCallRaw>;

// ---------------------------------------------------------------------------
// Custom grammar format
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum GrammarSyntax {
    Lark,
    #[default]
    Regex,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq, Default)]
pub struct CustomGrammarFormatParamRaw {
    /// The grammar definition.
    pub definition: String,
    /// The syntax of the grammar definition. One of `lark` or `regex`.
    pub syntax: GrammarSyntax,
}
pub type CustomGrammarFormatParam = WithOtherFields<CustomGrammarFormatParamRaw>;

// ---------------------------------------------------------------------------
// Response format
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormatRaw {
    /// The type of response format being defined: `text`
    Text,
    /// The type of response format being defined: `json_object`
    JsonObject,
    /// The type of response format being defined: `json_schema`
    JsonSchema {
        json_schema: ResponseFormatJsonSchema,
    },
}
pub type ResponseFormat = WithOtherFields<ResponseFormatRaw>;

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ResponseFormatJsonSchemaRaw {
    /// A description of what the response format is for, used by the model to determine how to respond in the format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The name of the response format. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    pub name: String,
    /// The schema for the response format, described as a JSON Schema object.
    /// Learn how to build JSON schemas [here](https://json-schema.org/).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<serde_json::Value>,
    /// Whether to enable strict schema adherence when generating the output.
    /// If set to true, the model will always follow the exact schema defined
    /// in the `schema` field. Only a subset of JSON Schema is supported when
    /// `strict` is `true`. To learn more, read the [Structured Outputs
    /// guide](https://platform.openai.com/docs/guides/structured-outputs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}
pub type ResponseFormatJsonSchema = WithOtherFields<ResponseFormatJsonSchemaRaw>;

// ---------------------------------------------------------------------------
// Content parts (shared)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestMessageContentPartTextRaw {
    pub text: String,
}
pub type ChatCompletionRequestMessageContentPartText =
    WithOtherFields<ChatCompletionRequestMessageContentPartTextRaw>;

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestMessageContentPartRefusalRaw {
    /// The refusal message generated by the model.
    pub refusal: String,
}
pub type ChatCompletionRequestMessageContentPartRefusal =
    WithOtherFields<ChatCompletionRequestMessageContentPartRefusalRaw>;

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestMessageContentPartImageRaw {
    pub image_url: ImageUrl,
}
pub type ChatCompletionRequestMessageContentPartImage =
    WithOtherFields<ChatCompletionRequestMessageContentPartImageRaw>;

/// Learn about [audio inputs](https://platform.openai.com/docs/guides/audio).
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestMessageContentPartAudioRaw {
    pub input_audio: InputAudio,
}
pub type ChatCompletionRequestMessageContentPartAudio =
    WithOtherFields<ChatCompletionRequestMessageContentPartAudioRaw>;

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestMessageContentPartFileRaw {
    pub file: FileObject,
}
pub type ChatCompletionRequestMessageContentPartFile =
    WithOtherFields<ChatCompletionRequestMessageContentPartFileRaw>;

// ---------------------------------------------------------------------------
// Content-part enums (per role)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionRequestUserMessageContentPartRaw {
    Text(ChatCompletionRequestMessageContentPartText),
    ImageUrl(ChatCompletionRequestMessageContentPartImage),
    InputAudio(ChatCompletionRequestMessageContentPartAudio),
    File(ChatCompletionRequestMessageContentPartFile),
}
pub type ChatCompletionRequestUserMessageContentPart =
    WithOtherFields<ChatCompletionRequestUserMessageContentPartRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionRequestSystemMessageContentPartRaw {
    Text(ChatCompletionRequestMessageContentPartText),
}
pub type ChatCompletionRequestSystemMessageContentPart =
    WithOtherFields<ChatCompletionRequestSystemMessageContentPartRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionRequestAssistantMessageContentPartRaw {
    Text(ChatCompletionRequestMessageContentPartText),
    Refusal(ChatCompletionRequestMessageContentPartRefusal),
}
pub type ChatCompletionRequestAssistantMessageContentPart =
    WithOtherFields<ChatCompletionRequestAssistantMessageContentPartRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionRequestToolMessageContentPartRaw {
    Text(ChatCompletionRequestMessageContentPartText),
}
pub type ChatCompletionRequestToolMessageContentPart =
    WithOtherFields<ChatCompletionRequestToolMessageContentPartRaw>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionRequestDeveloperMessageContentPartRaw {
    Text(ChatCompletionRequestMessageContentPartText),
}
pub type ChatCompletionRequestDeveloperMessageContentPart =
    WithOtherFields<ChatCompletionRequestDeveloperMessageContentPartRaw>;

// ---------------------------------------------------------------------------
// Content enums (untagged)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestSystemMessageContent {
    /// The text contents of the system message.
    Text(String),
    /// An array of content parts with a defined type. For system messages, only type `text` is supported.
    Array(Vec<ChatCompletionRequestSystemMessageContentPart>),
}

impl Default for ChatCompletionRequestSystemMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestUserMessageContent {
    /// The text contents of the message.
    Text(String),
    /// An array of content parts with a defined type. Supported options differ based on the [model](https://platform.openai.com/docs/models) being used to generate the response. Can contain text, image, or audio inputs.
    Array(Vec<ChatCompletionRequestUserMessageContentPart>),
}

impl Default for ChatCompletionRequestUserMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestAssistantMessageContent {
    /// The text contents of the message.
    Text(String),
    /// An array of content parts with a defined type. Can be one or more of type `text`, or exactly one of type `refusal`.
    Array(Vec<ChatCompletionRequestAssistantMessageContentPart>),
}

impl Default for ChatCompletionRequestAssistantMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestToolMessageContent {
    /// The text contents of the tool message.
    Text(String),
    /// An array of content parts with a defined type. For tool messages, only type `text` is supported.
    Array(Vec<ChatCompletionRequestToolMessageContentPart>),
}

impl Default for ChatCompletionRequestToolMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatCompletionRequestDeveloperMessageContent {
    Text(String),
    Array(Vec<ChatCompletionRequestDeveloperMessageContentPart>),
}

impl Default for ChatCompletionRequestDeveloperMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

// ---------------------------------------------------------------------------
// Tool calls referenced by request assistant message
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionMessageToolCallsRaw {
    Function(ChatCompletionMessageToolCall),
    Custom(ChatCompletionMessageCustomToolCall),
}
pub type ChatCompletionMessageToolCalls = WithOtherFields<ChatCompletionMessageToolCallsRaw>;

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionMessageToolCallRaw {
    /// The ID of the tool call.
    pub id: String,
    /// The function that the model called.
    pub function: FunctionCall,
}
pub type ChatCompletionMessageToolCall = WithOtherFields<ChatCompletionMessageToolCallRaw>;

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionMessageCustomToolCallRaw {
    /// The ID of the tool call.
    pub id: String,
    /// The custom tool that the model called.
    pub custom_tool: CustomTool,
}
pub type ChatCompletionMessageCustomToolCall =
    WithOtherFields<ChatCompletionMessageCustomToolCallRaw>;

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct CustomToolRaw {
    /// The name of the custom tool to call.
    pub name: String,
    /// The input for the custom tool call generated by the model.
    pub input: String,
}
pub type CustomTool = WithOtherFields<CustomToolRaw>;

// ---------------------------------------------------------------------------
// Per-role request messages
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestDeveloperMessageRaw {
    /// The contents of the developer message.
    pub content: ChatCompletionRequestDeveloperMessageContent,

    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}
pub type ChatCompletionRequestDeveloperMessage =
    WithOtherFields<ChatCompletionRequestDeveloperMessageRaw>;

impl ChatCompletionRequestDeveloperMessageRaw {
    pub fn new_text(content: impl Into<String>) -> ChatCompletionRequestDeveloperMessage {
        WithOtherFields::new(Self {
            content: ChatCompletionRequestDeveloperMessageContent::Text(content.into()),
            name: None,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestSystemMessageRaw {
    /// The contents of the system message.
    pub content: ChatCompletionRequestSystemMessageContent,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}
pub type ChatCompletionRequestSystemMessage =
    WithOtherFields<ChatCompletionRequestSystemMessageRaw>;

impl ChatCompletionRequestSystemMessageRaw {
    pub fn new_text(content: impl Into<String>) -> ChatCompletionRequestSystemMessage {
        WithOtherFields::new(Self {
            content: ChatCompletionRequestSystemMessageContent::Text(content.into()),
            name: None,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestUserMessageRaw {
    /// The contents of the user message.
    pub content: ChatCompletionRequestUserMessageContent,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}
pub type ChatCompletionRequestUserMessage = WithOtherFields<ChatCompletionRequestUserMessageRaw>;

impl ChatCompletionRequestUserMessageRaw {
    pub fn new_text(content: impl Into<String>) -> ChatCompletionRequestUserMessage {
        WithOtherFields::new(Self {
            content: ChatCompletionRequestUserMessageContent::Text(content.into()),
            name: None,
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestAssistantMessageAudioRaw {
    /// Unique identifier for a previous audio response from the model.
    pub id: String,
}
pub type ChatCompletionRequestAssistantMessageAudio =
    WithOtherFields<ChatCompletionRequestAssistantMessageAudioRaw>;

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestAssistantMessageRaw {
    /// The contents of the assistant message. Required unless `tool_calls` or `function_call` is specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatCompletionRequestAssistantMessageContent>,
    /// The refusal message by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Data about a previous audio response from the model.
    /// [Learn more](https://platform.openai.com/docs/guides/audio).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionRequestAssistantMessageAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCalls>>,
    /// Deprecated and replaced by `tool_calls`. The name and arguments of a function that should be called, as generated by the model.
    #[deprecated]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<FunctionCall>,
}
pub type ChatCompletionRequestAssistantMessage =
    WithOtherFields<ChatCompletionRequestAssistantMessageRaw>;

#[allow(deprecated)]
impl ChatCompletionRequestAssistantMessageRaw {
    pub fn new_text(content: impl Into<String>) -> ChatCompletionRequestAssistantMessage {
        WithOtherFields::new(Self {
            content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                content.into(),
            )),
            refusal: None,
            name: None,
            audio: None,
            tool_calls: None,
            function_call: None,
        })
    }
}

/// Tool message
#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestToolMessageRaw {
    /// The contents of the tool message.
    pub content: ChatCompletionRequestToolMessageContent,
    pub tool_call_id: String,
}
pub type ChatCompletionRequestToolMessage = WithOtherFields<ChatCompletionRequestToolMessageRaw>;

impl ChatCompletionRequestToolMessageRaw {
    pub fn new_text(
        content: impl Into<String>,
        tool_call_id: impl Into<String>,
    ) -> ChatCompletionRequestToolMessage {
        WithOtherFields::new(Self {
            content: ChatCompletionRequestToolMessageContent::Text(content.into()),
            tool_call_id: tool_call_id.into(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, PartialEq)]
pub struct ChatCompletionRequestFunctionMessageRaw {
    /// The return value from the function call, to return to the model.
    pub content: Option<String>,
    /// The name of the function to call.
    pub name: String,
}
pub type ChatCompletionRequestFunctionMessage =
    WithOtherFields<ChatCompletionRequestFunctionMessageRaw>;

// ---------------------------------------------------------------------------
// Tagged-enum: ChatCompletionRequestMessage
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionRequestMessageRaw {
    Developer(ChatCompletionRequestDeveloperMessage),
    System(ChatCompletionRequestSystemMessage),
    User(ChatCompletionRequestUserMessage),
    Assistant(ChatCompletionRequestAssistantMessage),
    Tool(ChatCompletionRequestToolMessage),
    Function(ChatCompletionRequestFunctionMessage),
}
pub type ChatCompletionRequestMessage = WithOtherFields<ChatCompletionRequestMessageRaw>;

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionToolsRaw {
    /// A function tool that can be used to generate a response.
    Function(ChatCompletionTool),
    /// A custom tool that processes input using a specified format.
    Custom(CustomToolChatCompletions),
}
pub type ChatCompletionTools = WithOtherFields<ChatCompletionToolsRaw>;

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionToolRaw {
    pub function: FunctionObject,
}
pub type ChatCompletionTool = WithOtherFields<ChatCompletionToolRaw>;

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct CustomToolChatCompletionsRaw {
    pub custom: CustomToolProperties,
}
pub type CustomToolChatCompletions = WithOtherFields<CustomToolChatCompletionsRaw>;

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct CustomToolPropertiesRaw {
    /// The name of the custom tool, used to identify it in tool calls.
    pub name: String,

    /// Optional description of the custom tool, used to provide more context.
    pub description: Option<String>,

    /// The input format for the custom tool. Default is unconstrained text.
    pub format: CustomToolPropertiesFormat,
}
pub type CustomToolProperties = WithOtherFields<CustomToolPropertiesRaw>;

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CustomToolPropertiesFormatRaw {
    /// Unconstrained free-form text.
    Text,
    /// A grammar defined by the user.
    Grammar { grammar: CustomGrammarFormatParam },
}

impl Default for CustomToolPropertiesFormatRaw {
    fn default() -> Self {
        Self::Text
    }
}

pub type CustomToolPropertiesFormat = WithOtherFields<CustomToolPropertiesFormatRaw>;

// ---------------------------------------------------------------------------
// Tool choice
// ---------------------------------------------------------------------------

/// Specifies a tool the model should use. Use to force the model to call a specific function.
#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionNamedToolChoiceRaw {
    pub function: FunctionName,
}
pub type ChatCompletionNamedToolChoice = WithOtherFields<ChatCompletionNamedToolChoiceRaw>;

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionNamedToolChoiceCustomRaw {
    pub custom: CustomName,
}
pub type ChatCompletionNamedToolChoiceCustom =
    WithOtherFields<ChatCompletionNamedToolChoiceCustomRaw>;

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct FunctionNameRaw {
    /// The name of the function to call.
    pub name: String,
}
pub type FunctionName = WithOtherFields<FunctionNameRaw>;

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct CustomNameRaw {
    /// The name of the custom tool to call.
    pub name: String,
}
pub type CustomName = WithOtherFields<CustomNameRaw>;

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceOptions {
    #[default]
    None,
    Auto,
    Required,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ToolChoiceAllowedMode {
    Auto,
    Required,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ChatCompletionAllowedToolsRaw {
    /// Constrains the tools available to the model to a pre-defined set.
    ///
    /// `auto` allows the model to pick from among the allowed tools and generate a
    /// message.
    ///
    /// `required` requires the model to call one or more of the allowed tools.
    pub mode: ToolChoiceAllowedMode,
    /// A list of tool definitions that the model should be allowed to call.
    pub tools: Vec<serde_json::Value>,
}
pub type ChatCompletionAllowedTools = WithOtherFields<ChatCompletionAllowedToolsRaw>;

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionAllowedToolsChoiceRaw {
    pub allowed_tools: Vec<ChatCompletionAllowedTools>,
}
pub type ChatCompletionAllowedToolsChoice = WithOtherFields<ChatCompletionAllowedToolsChoiceRaw>;

/// Controls which (if any) tool is called by the model.
/// `none` means the model will not call any tool and instead generates a message.
/// `auto` means the model can pick between generating a message or calling one or more tools.
/// `required` means the model must call one or more tools.
/// Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.
///
/// `none` is the default when no tools are present. `auto` is the default if tools are present.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionToolChoiceOptionRaw {
    AllowedTools(ChatCompletionAllowedToolsChoice),
    Function(ChatCompletionNamedToolChoice),
    Custom(ChatCompletionNamedToolChoiceCustom),

    #[serde(untagged)]
    Mode(ToolChoiceOptions),
}
pub type ChatCompletionToolChoiceOption = WithOtherFields<ChatCompletionToolChoiceOptionRaw>;

// ---------------------------------------------------------------------------
// Web search
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
/// The amount of context window space to use for the search.
pub enum WebSearchContextSize {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchUserLocationType {
    Approximate,
}

/// Approximate location parameters for the search.
#[derive(Clone, Serialize, Debug, Default, Deserialize, PartialEq)]
pub struct WebSearchLocationRaw {
    ///  The two-letter [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1) of the user, e.g. `US`.
    pub country: Option<String>,
    /// Free text input for the region of the user, e.g. `California`.
    pub region: Option<String>,
    /// Free text input for the city of the user, e.g. `San Francisco`.
    pub city: Option<String>,
    /// The [IANA timezone](https://timeapi.io/documentation/iana-timezones) of the user, e.g. `America/Los_Angeles`.
    pub timezone: Option<String>,
}
pub type WebSearchLocation = WithOtherFields<WebSearchLocationRaw>;

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct WebSearchUserLocationRaw {
    //  The type of location approximation. Always `approximate`.
    pub r#type: WebSearchUserLocationType,

    pub approximate: WebSearchLocation,
}
pub type WebSearchUserLocation = WithOtherFields<WebSearchUserLocationRaw>;

/// Options for the web search tool.
#[derive(Clone, Serialize, Debug, Default, Deserialize, PartialEq)]
pub struct WebSearchOptionsRaw {
    /// High level guidance for the amount of context window space to use for the search. One of `low`, `medium`, or `high`. `medium` is the default.
    pub search_context_size: Option<WebSearchContextSize>,

    /// Approximate location parameters for the search.
    pub user_location: Option<WebSearchUserLocation>,
}
pub type WebSearchOptions = WithOtherFields<WebSearchOptionsRaw>;

// ---------------------------------------------------------------------------
// Audio (request side)
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionAudioVoice {
    Alloy,
    Ash,
    Ballad,
    Coral,
    Echo,
    Fable,
    Nova,
    Onyx,
    Sage,
    Shimmer,
    #[serde(untagged)]
    Other(String),
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionAudioFormat {
    Wav,
    Aac,
    Mp3,
    Flac,
    Opus,
    Pcm16,
}

#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionAudioRaw {
    /// The voice the model uses to respond. Supported built-in voices are `alloy`, `ash`,
    /// `ballad`, `coral`, `echo`, `fable`, `nova`, `onyx`, `sage`, `shimmer`, `marin`, and `cedar`.
    pub voice: ChatCompletionAudioVoice,
    /// Specifies the output audio format. Must be one of `wav`, `aac`, `mp3`, `flac`, `opus`, or `pcm16`.
    pub format: ChatCompletionAudioFormat,
}
pub type ChatCompletionAudio = WithOtherFields<ChatCompletionAudioRaw>;

// ---------------------------------------------------------------------------
// Predicted output
// ---------------------------------------------------------------------------

/// The content that should be matched when generating a model response. If generated tokens would match this content, the entire model response can be returned much more quickly.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum PredictionContentContent {
    /// The content used for a Predicted Output. This is often the text of a file you are regenerating with minor changes.
    Text(String),
    /// An array of content parts with a defined type. Supported options differ based on the [model](https://platform.openai.com/docs/models) being used to generate the response. Can contain text inputs.
    Array(Vec<ChatCompletionRequestMessageContentPartText>),
}

/// Static predicted output content, such as the content of a text file that is being regenerated.
#[derive(Clone, Serialize, Debug, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase", content = "content")]
pub enum PredictionContentRaw {
    /// The type of the predicted content you want to provide. This type is
    /// currently always `content`.
    Content(PredictionContentContent),
}
pub type PredictionContent = WithOtherFields<PredictionContentRaw>;

// ---------------------------------------------------------------------------
// Stream options
// ---------------------------------------------------------------------------

/// Options for streaming response. Only set this when you set `stream: true`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct ChatCompletionStreamOptionsRaw {
    /// If set, an additional chunk will be streamed before the `data: [DONE]`
    /// message. The `usage` field on this chunk shows the token usage statistics
    /// for the entire request, and the `choices` field will always be an empty
    /// array.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,

    /// When true, stream obfuscation will be enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_obfuscation: Option<bool>,
}
pub type ChatCompletionStreamOptions = WithOtherFields<ChatCompletionStreamOptionsRaw>;

// ---------------------------------------------------------------------------
// Deprecated function-call config
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum ChatCompletionFunctionCall {
    /// The model does not call a function, and responds to the end-user.
    #[serde(rename = "none")]
    None,
    /// The model can pick between an end-user or calling a function.
    #[serde(rename = "auto")]
    Auto,

    /// Forces the model to call the specified function.
    #[serde(untagged)]
    Function { name: String },
}

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct ChatCompletionFunctionsRaw {
    /// The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    pub name: String,
    /// A description of what the function does, used by the model to choose when and how to call the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The parameters the functions accepts, described as a JSON Schema object.
    pub parameters: serde_json::Value,
}
pub type ChatCompletionFunctions = WithOtherFields<ChatCompletionFunctionsRaw>;

// ---------------------------------------------------------------------------
// CreateChatCompletionRequest
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize, Default, Debug, Deserialize, PartialEq)]
pub struct CreateChatCompletionRequestRaw {
    /// A list of messages comprising the conversation so far.
    pub messages: Vec<ChatCompletionRequestMessage>, // min: 1

    /// Model ID used to generate the response.
    pub model: String,

    /// Output types that you would like the model to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<ResponseModalities>>,

    /// Constrains the verbosity of the model's response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,

    /// Constrains effort on reasoning for reasoning models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,

    /// Upper bound for the number of tokens that can be generated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on
    /// their existing frequency in the text so far.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on
    /// whether they appear in the text so far.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// This tool searches the web for relevant results to use in a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,

    /// An integer between 0 and 20 specifying the number of most likely tokens to
    /// return at each token position.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,

    /// An object specifying the format that the model must output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Parameters for audio output. Required when audio output is requested with
    /// `modalities: ["audio"]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionAudio>,

    /// Whether or not to store the output of this chat completion request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,

    /// If set to true, the model response data will be streamed to the client
    /// as it is generated using server-sent events.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Up to 4 sequences where the API will stop generating further tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopConfiguration>,

    /// Modify the likelihood of specified tokens appearing in the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, i8>>,

    /// Whether to return log probabilities of the output tokens or not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,

    /// The maximum number of tokens that can be generated in the chat completion.
    /// Deprecated in favor of `max_completion_tokens`.
    #[deprecated]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// How many chat completion choices to generate for each input message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u8>,

    /// Configuration for a Predicted Output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<PredictionContent>,

    /// Beta. If specified, our system will make a best effort to sample deterministically.
    #[deprecated]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,

    /// Specifies the processing type used for serving the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,

    /// What sampling temperature to use, between 0 and 2.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// An alternative to sampling with temperature, called nucleus sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// A list of tools the model may call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionTools>>,

    /// Controls which (if any) tool is called by the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,

    /// Whether to enable parallel function calling during tool use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,

    /// Deprecated stable identifier for end-users.
    #[deprecated]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,

    /// A stable identifier used to help detect users of your application that may be violating OpenAI's
    /// usage policies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,

    /// Used by OpenAI to cache responses for similar requests.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,

    /// Deprecated in favor of `tool_choice`.
    #[deprecated]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatCompletionFunctionCall>,

    /// Deprecated in favor of `tools`.
    #[deprecated]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<ChatCompletionFunctions>>,

    ///  Developer-defined tags and values used for filtering completions in the dashboard.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

impl CreateChatCompletionRequestRaw {
    pub fn new() -> Self {
        Self::default()
    }
}

pub type CreateChatCompletionRequest = WithOtherFields<CreateChatCompletionRequestRaw>;

// ---------------------------------------------------------------------------
// Public extensible wrappers (preserved from the old API)
// ---------------------------------------------------------------------------

/// Public wrapper around [`CreateChatCompletionRequest`] preserved for backward compatibility.
///
/// `CreateChatCompletionRequest` is itself already a `WithOtherFields` alias, so this newtype
/// just adds a few helper methods used by callers.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct RawExtensibleChatCompletionRequest(pub CreateChatCompletionRequest);

impl RawExtensibleChatCompletionRequest {
    pub fn new(request: CreateChatCompletionRequestRaw) -> Self {
        Self(WithOtherFields::new(request))
    }

    /// Returns the inner [`CreateChatCompletionRequestRaw`] (clone). The "raw" form drops any
    /// captured top-level extension fields.
    pub fn to_chat_completion_request(&self) -> CreateChatCompletionRequestRaw {
        self.0.inner.clone()
    }

    pub fn extra(&self) -> &OtherFields {
        &self.0.other
    }

    pub fn extra_mut(&mut self) -> &mut OtherFields {
        &mut self.0.other
    }

    pub fn insert_extra_string(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.0.other.insert(key.into(), Value::String(value.into()));
    }

    pub fn into_inner(self) -> CreateChatCompletionRequest {
        self.0
    }
}

impl Deref for RawExtensibleChatCompletionRequest {
    type Target = CreateChatCompletionRequest;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RawExtensibleChatCompletionRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<CreateChatCompletionRequestRaw> for RawExtensibleChatCompletionRequest {
    fn from(request: CreateChatCompletionRequestRaw) -> Self {
        Self::new(request)
    }
}

impl From<CreateChatCompletionRequest> for RawExtensibleChatCompletionRequest {
    fn from(request: CreateChatCompletionRequest) -> Self {
        Self(request)
    }
}

impl From<RawExtensibleChatCompletionRequest> for CreateChatCompletionRequest {
    fn from(request: RawExtensibleChatCompletionRequest) -> Self {
        request.0
    }
}

/// Public wrapper around [`ChatCompletionRequestMessage`] preserved for backward compatibility.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct RawExtensibleChatRequestMessage(pub ChatCompletionRequestMessage);

impl RawExtensibleChatRequestMessage {
    pub fn new(message: ChatCompletionRequestMessageRaw) -> Self {
        Self(WithOtherFields::new(message))
    }

    pub fn from_message(message: ChatCompletionRequestMessage) -> Self {
        Self(message)
    }

    pub fn extra(&self) -> &OtherFields {
        &self.0.other
    }

    pub fn extra_mut(&mut self) -> &mut OtherFields {
        &mut self.0.other
    }

    pub fn insert_extra_string(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.0.other.insert(key.into(), Value::String(value.into()));
    }

    pub fn into_inner(self) -> ChatCompletionRequestMessage {
        self.0
    }
}

impl Deref for RawExtensibleChatRequestMessage {
    type Target = ChatCompletionRequestMessage;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RawExtensibleChatRequestMessage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<ChatCompletionRequestMessageRaw> for RawExtensibleChatRequestMessage {
    fn from(message: ChatCompletionRequestMessageRaw) -> Self {
        Self::new(message)
    }
}

impl From<ChatCompletionRequestMessage> for RawExtensibleChatRequestMessage {
    fn from(message: ChatCompletionRequestMessage) -> Self {
        Self(message)
    }
}

impl From<RawExtensibleChatRequestMessage> for ChatCompletionRequestMessage {
    fn from(message: RawExtensibleChatRequestMessage) -> Self {
        message.0
    }
}
