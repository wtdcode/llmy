use std::{
    convert::Infallible,
    ops::{Deref, DerefMut},
    str::FromStr,
};

use color_eyre::eyre::eyre;
use llmy_types::other::WithOtherFields;

use crate::req::{
    ChatCompletionNamedToolChoiceCustomRaw, ChatCompletionToolChoiceOption,
    ChatCompletionToolChoiceOptionRaw, CustomNameRaw, ReasoningEffort, ToolChoiceOptions,
};

#[derive(Debug, Clone)]
pub struct LLMToolChoice(pub ChatCompletionToolChoiceOption);

impl FromStr for LLMToolChoice {
    type Err = Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let raw = match s {
            "auto" => ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Auto),
            "required" => ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Required),
            "none" => ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::None),
            _ => ChatCompletionToolChoiceOptionRaw::Custom(WithOtherFields::new(
                ChatCompletionNamedToolChoiceCustomRaw {
                    custom: WithOtherFields::new(CustomNameRaw {
                        name: s.to_string(),
                    }),
                },
            )),
        };
        Ok(Self(WithOtherFields::new(raw)))
    }
}

impl Deref for LLMToolChoice {
    type Target = ChatCompletionToolChoiceOption;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LLMToolChoice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<ChatCompletionToolChoiceOption> for LLMToolChoice {
    fn from(value: ChatCompletionToolChoiceOption) -> Self {
        Self(value)
    }
}

impl From<LLMToolChoice> for ChatCompletionToolChoiceOption {
    fn from(value: LLMToolChoice) -> Self {
        value.0
    }
}

#[derive(Debug, Clone)]
pub struct Reasoning(pub ReasoningEffort);

impl Reasoning {
    pub fn is_none(&self) -> bool {
        matches!(self.0, ReasoningEffort::None)
    }
}

impl FromStr for Reasoning {
    type Err = color_eyre::Report;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self(ReasoningEffort::None)),
            "minimal" => Ok(Self(ReasoningEffort::Minimal)),
            "low" => Ok(Self(ReasoningEffort::Low)),
            "medium" => Ok(Self(ReasoningEffort::Medium)),
            "high" => Ok(Self(ReasoningEffort::High)),
            "xhigh" => Ok(Self(ReasoningEffort::Xhigh)),
            _ => Err(eyre!("unknown effort: {}", s)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LLMSettings {
    pub llm_temperature: Option<f32>,
    pub llm_presence_penalty: Option<f32>,
    pub llm_prompt_timeout: u64,
    pub llm_retry: u64,
    pub llm_max_completion_tokens: Option<u32>,
    pub llm_tool_choice: Option<LLMToolChoice>,
    pub llm_stream: bool,
    pub top_p: Option<f32>,
    pub reasoning_effort: Option<Reasoning>,
    /// When a typed/JSON completion fails to deserialize, retry the parse after
    /// stripping a markdown code fence from the content (see `MarkdownTagFilter`).
    pub auto_strip: bool,
    /// Pick a `prompt_cache_key` for requests that don't come with one, based on
    /// what this client has already sent (see [`crate::cache_key`]).
    pub auto_cache_key: bool,
    /// How long an auto cache key survives without being used, in seconds.
    pub cache_key_ttl: u64,
    /// Requests per minute one auto cache key takes before we spread to another.
    pub cache_key_rpm: u32,
}

impl LLMSettings {
    /// The auto cache key policy these settings describe.
    pub fn cache_key_config(&self) -> crate::cache_key::CacheKeyConfig {
        crate::cache_key::CacheKeyConfig {
            enabled: self.auto_cache_key,
            ttl: std::time::Duration::from_secs(self.cache_key_ttl),
            max_rpm: self.cache_key_rpm,
        }
    }

    pub fn timeout(&self) -> std::time::Duration {
        if self.llm_prompt_timeout == 0 {
            std::time::Duration::MAX
        } else {
            std::time::Duration::from_secs(self.llm_prompt_timeout)
        }
    }
}
