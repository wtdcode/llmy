use std::fmt::Display;

use async_openai::error::OpenAIError;
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct GeneralToolCall {
    pub tool_id: String,
    pub tool_name: String,
    pub tool_args: String,
}

impl Display for GeneralToolCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "ToolCall(name={}, args={:?}, id={})",
            self.tool_name, self.tool_args, self.tool_id
        ))
    }
}

#[derive(Error, Debug)]
pub enum LLMYError {
    #[error("io error: {0}")]
    IO(#[from] std::io::Error),
    #[error("openai error: {0}")]
    OpenAI(#[from] OpenAIError),
    #[error("json error: {0}")]
    STDJSON(#[from] serde_json::Error),
    #[error("billing error: reach cap {0}, current {1}")]
    Billing(f64, f64),
    #[error("incorrect tool call for tool {0} with args {1} given schema {2:?}")]
    IncorrectToolCall(String, String, schemars::Schema),
    #[error("nonexisting tool call {0}")]
    NonExistingToolCall(GeneralToolCall),
    #[error("toolcall {0} has nested error: {1}")]
    ToolCallError(GeneralToolCall, Box<LLMYError>),
    #[error("response filtered: {0}")]
    Filtered(String),
    #[error("no choice is returned")]
    EmptyChoice,
    #[error("reach output length limit")]
    OutputLength,
    #[error(transparent)]
    Other(#[from] color_eyre::Report),
}
