use std::ops::{Deref, DerefMut};

use async_openai::types::chat::*;
use llmy_types::other::{OtherFields, WithOtherFields};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct RawExtensibleChatCompletionResponse(
    WithOtherFields<CreateChatCompletionResponseWithOtherFields>,
);

impl RawExtensibleChatCompletionResponse {
    pub fn new(response: CreateChatCompletionResponse) -> Self {
        Self(WithOtherFields::new(response.into()))
    }

    pub fn extra(&self) -> &OtherFields {
        &self.0.other
    }

    pub fn extra_mut(&mut self) -> &mut OtherFields {
        &mut self.0.other
    }

    pub fn into_base(self) -> CreateChatCompletionResponse {
        self.0.inner.into()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct RawExtensibleChatChoice(WithOtherFields<ChatChoiceWithOtherFields>);

impl RawExtensibleChatChoice {
    pub fn new(choice: ChatChoice) -> Self {
        Self(WithOtherFields::new(choice.into()))
    }

    pub fn extra(&self) -> &OtherFields {
        &self.0.other
    }

    pub fn extra_mut(&mut self) -> &mut OtherFields {
        &mut self.0.other
    }

    pub fn message_extra(&self) -> &OtherFields {
        &self.0.inner.message.other
    }

    pub fn message_extra_mut(&mut self) -> &mut OtherFields {
        &mut self.0.inner.message.other
    }

    pub fn reasoning_content(&self) -> Option<&str> {
        self.message_extra().get("reasoning_content")?.as_str()
    }

    pub fn into_inner(self) -> ChatChoiceWithOtherFields {
        self.0.into_inner()
    }
}

impl Deref for RawExtensibleChatChoice {
    type Target = WithOtherFields<ChatChoiceWithOtherFields>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RawExtensibleChatChoice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<ChatChoice> for RawExtensibleChatChoice {
    fn from(choice: ChatChoice) -> Self {
        Self::new(choice)
    }
}

impl From<RawExtensibleChatChoice> for ChatChoice {
    fn from(choice: RawExtensibleChatChoice) -> Self {
        choice.0.into_inner().into()
    }
}

impl Deref for RawExtensibleChatCompletionResponse {
    type Target = WithOtherFields<CreateChatCompletionResponseWithOtherFields>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RawExtensibleChatCompletionResponse {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<CreateChatCompletionResponse> for RawExtensibleChatCompletionResponse {
    fn from(response: CreateChatCompletionResponse) -> Self {
        Self::new(response)
    }
}

impl From<RawExtensibleChatCompletionResponse> for CreateChatCompletionResponse {
    fn from(response: RawExtensibleChatCompletionResponse) -> Self {
        response.into_base()
    }
}

#[allow(deprecated)]
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct ChatChoiceWithOtherFields {
    pub index: u32,
    pub message: WithOtherFields<ChatCompletionResponseMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatChoiceLogprobs>,
}

#[allow(deprecated)]
#[derive(Debug, Deserialize, Serialize, Clone, PartialEq)]
pub struct CreateChatCompletionResponseWithOtherFields {
    pub id: String,
    pub choices: Vec<RawExtensibleChatChoice>,
    pub created: u32,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    pub object: String,
    pub usage: Option<CompletionUsage>,
}

#[allow(deprecated)]
impl From<ChatChoice> for ChatChoiceWithOtherFields {
    fn from(choice: ChatChoice) -> Self {
        Self {
            index: choice.index,
            message: WithOtherFields::new(choice.message),
            finish_reason: choice.finish_reason,
            logprobs: choice.logprobs,
        }
    }
}

#[allow(deprecated)]
impl From<ChatChoiceWithOtherFields> for ChatChoice {
    fn from(choice: ChatChoiceWithOtherFields) -> Self {
        Self {
            index: choice.index,
            message: choice.message.into_inner(),
            finish_reason: choice.finish_reason,
            logprobs: choice.logprobs,
        }
    }
}

#[allow(deprecated)]
impl From<CreateChatCompletionResponse> for CreateChatCompletionResponseWithOtherFields {
    fn from(response: CreateChatCompletionResponse) -> Self {
        Self {
            id: response.id,
            choices: response
                .choices
                .into_iter()
                .map(RawExtensibleChatChoice::from)
                .collect(),
            created: response.created,
            model: response.model,
            service_tier: response.service_tier,
            system_fingerprint: response.system_fingerprint,
            object: response.object,
            usage: response.usage,
        }
    }
}

#[allow(deprecated)]
impl From<CreateChatCompletionResponseWithOtherFields> for CreateChatCompletionResponse {
    fn from(response: CreateChatCompletionResponseWithOtherFields) -> Self {
        Self {
            id: response.id,
            choices: response.choices.into_iter().map(ChatChoice::from).collect(),
            created: response.created,
            model: response.model,
            service_tier: response.service_tier,
            system_fingerprint: response.system_fingerprint,
            object: response.object,
            usage: response.usage,
        }
    }
}
