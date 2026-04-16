use async_openai::error::OpenAIError;
use thiserror::Error;

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
    #[error("incorrect tool call, schema: {0:?}, args: {1}")]
    IncorrectToolCall(schemars::Schema, String),
    #[error(transparent)]
    Other(#[from] color_eyre::Report),
}
