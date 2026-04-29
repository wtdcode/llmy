use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use async_openai::types::{Metadata, chat::*};
use llmy_types::other::{OtherFields, WithOtherFields};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct RawExtensibleChatCompletionRequest(
    WithOtherFields<CreateChatCompletionRequestWithOtherFields>,
);

impl RawExtensibleChatCompletionRequest {
    pub fn new(request: CreateChatCompletionRequest) -> Self {
        Self(WithOtherFields::new(
            CreateChatCompletionRequestWithOtherFields::from(request),
        ))
    }

    pub fn to_chat_completion_request(&self) -> CreateChatCompletionRequest {
        self.0.inner.clone().into()
    }
}

impl Deref for RawExtensibleChatCompletionRequest {
    type Target = WithOtherFields<CreateChatCompletionRequestWithOtherFields>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RawExtensibleChatCompletionRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<CreateChatCompletionRequest> for RawExtensibleChatCompletionRequest {
    fn from(request: CreateChatCompletionRequest) -> Self {
        Self::new(request)
    }
}

impl From<RawExtensibleChatCompletionRequest> for CreateChatCompletionRequest {
    fn from(request: RawExtensibleChatCompletionRequest) -> Self {
        request.0.inner.into()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct RawExtensibleChatRequestMessage(WithOtherFields<ChatCompletionRequestMessage>);

impl RawExtensibleChatRequestMessage {
    pub fn new(message: ChatCompletionRequestMessage) -> Self {
        Self(WithOtherFields::new(message))
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
        self.0.into_inner()
    }
}

impl Deref for RawExtensibleChatRequestMessage {
    type Target = WithOtherFields<ChatCompletionRequestMessage>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RawExtensibleChatRequestMessage {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<ChatCompletionRequestMessage> for RawExtensibleChatRequestMessage {
    fn from(message: ChatCompletionRequestMessage) -> Self {
        Self::new(message)
    }
}

impl From<RawExtensibleChatRequestMessage> for ChatCompletionRequestMessage {
    fn from(message: RawExtensibleChatRequestMessage) -> Self {
        message.0.into_inner()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct CreateChatCompletionRequestWithOtherFields {
    pub messages: Vec<RawExtensibleChatRequestMessage>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<ResponseModalities>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verbosity: Option<Verbosity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChatCompletionAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopConfiguration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, i8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<PredictionContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionTools>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function_call: Option<ChatCompletionFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub functions: Option<Vec<ChatCompletionFunctions>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
}

#[allow(deprecated)]
impl From<CreateChatCompletionRequest> for CreateChatCompletionRequestWithOtherFields {
    fn from(request: CreateChatCompletionRequest) -> Self {
        Self {
            messages: request
                .messages
                .into_iter()
                .map(RawExtensibleChatRequestMessage::new)
                .collect(),
            model: request.model,
            modalities: request.modalities,
            verbosity: request.verbosity,
            reasoning_effort: request.reasoning_effort,
            max_completion_tokens: request.max_completion_tokens,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            web_search_options: request.web_search_options,
            top_logprobs: request.top_logprobs,
            response_format: request.response_format,
            audio: request.audio,
            store: request.store,
            stream: request.stream,
            stop: request.stop,
            logit_bias: request.logit_bias,
            logprobs: request.logprobs,
            max_tokens: request.max_tokens,
            n: request.n,
            prediction: request.prediction,
            seed: request.seed,
            stream_options: request.stream_options,
            service_tier: request.service_tier,
            temperature: request.temperature,
            top_p: request.top_p,
            tools: request.tools,
            tool_choice: request.tool_choice,
            parallel_tool_calls: request.parallel_tool_calls,
            user: request.user,
            safety_identifier: request.safety_identifier,
            prompt_cache_key: request.prompt_cache_key,
            function_call: request.function_call,
            functions: request.functions,
            metadata: request.metadata,
        }
    }
}

#[allow(deprecated)]
impl From<CreateChatCompletionRequestWithOtherFields> for CreateChatCompletionRequest {
    fn from(request: CreateChatCompletionRequestWithOtherFields) -> Self {
        Self {
            messages: request
                .messages
                .into_iter()
                .map(ChatCompletionRequestMessage::from)
                .collect(),
            model: request.model,
            modalities: request.modalities,
            verbosity: request.verbosity,
            reasoning_effort: request.reasoning_effort,
            max_completion_tokens: request.max_completion_tokens,
            frequency_penalty: request.frequency_penalty,
            presence_penalty: request.presence_penalty,
            web_search_options: request.web_search_options,
            top_logprobs: request.top_logprobs,
            response_format: request.response_format,
            audio: request.audio,
            store: request.store,
            stream: request.stream,
            stop: request.stop,
            logit_bias: request.logit_bias,
            logprobs: request.logprobs,
            max_tokens: request.max_tokens,
            n: request.n,
            prediction: request.prediction,
            seed: request.seed,
            stream_options: request.stream_options,
            service_tier: request.service_tier,
            temperature: request.temperature,
            top_p: request.top_p,
            tools: request.tools,
            tool_choice: request.tool_choice,
            parallel_tool_calls: request.parallel_tool_calls,
            user: request.user,
            safety_identifier: request.safety_identifier,
            prompt_cache_key: request.prompt_cache_key,
            function_call: request.function_call,
            functions: request.functions,
            metadata: request.metadata,
        }
    }
}
