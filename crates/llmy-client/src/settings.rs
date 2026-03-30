use std::{
    convert::Infallible,
    ops::{Deref, DerefMut},
    str::FromStr,
};

use async_openai::types::chat::{
    ChatCompletionNamedToolChoiceCustom, ChatCompletionToolChoiceOption, CustomName,
    ReasoningEffort, ToolChoiceOptions,
};
use color_eyre::eyre::eyre;

#[derive(Debug, Clone)]
pub struct LLMToolChoice(pub ChatCompletionToolChoiceOption);

impl FromStr for LLMToolChoice {
    type Err = Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "auto" => Self(ChatCompletionToolChoiceOption::Mode(
                ToolChoiceOptions::Auto,
            )),
            "required" => Self(ChatCompletionToolChoiceOption::Mode(
                ToolChoiceOptions::Required,
            )),
            "none" => Self(ChatCompletionToolChoiceOption::Mode(
                ToolChoiceOptions::None,
            )),
            _ => Self(ChatCompletionToolChoiceOption::Custom(
                ChatCompletionNamedToolChoiceCustom {
                    custom: CustomName {
                        name: s.to_string(),
                    },
                },
            )),
        })
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
    pub llm_temperature: f32,
    pub llm_presence_penalty: f32,
    pub llm_prompt_timeout: u64,
    pub llm_retry: u64,
    pub llm_max_completion_tokens: u32,
    pub llm_tool_choice: Option<LLMToolChoice>,
    pub llm_stream: bool,
    pub reasoning_effort: Option<Reasoning>,
}

impl LLMSettings {
    pub fn timeout(&self) -> std::time::Duration {
        if self.llm_prompt_timeout == 0 {
            std::time::Duration::MAX
        } else {
            std::time::Duration::from_secs(self.llm_prompt_timeout)
        }
    }
}
