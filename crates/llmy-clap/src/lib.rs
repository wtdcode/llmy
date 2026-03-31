use clap::Args;
use llmy_client::{client::*, model::OpenAIModel, settings::*};
use std::path::PathBuf;

macro_rules! make_openai_args {
    ($struct_name:ident, $prefix:literal) => {
        #[derive(Args, Clone, Debug)]
        pub struct $struct_name {
            #[arg(
                long,
                env = concat!($prefix, "OPENAI_API_URL"),
                default_value = "https://api.openai.com/v1"
            )]
            pub openai_url: String,

            #[arg(long, env = concat!($prefix, "AZURE_OPENAI_ENDPOINT"))]
            pub azure_openai_endpoint: Option<String>,

            #[arg(long, env = concat!($prefix, "OPENAI_API_KEY"))]
            pub openai_key: Option<String>,

            #[arg(long, env = concat!($prefix, "AZURE_API_DEPLOYMENT"))]
            pub azure_deployment: Option<String>,

            #[arg(long, env = concat!($prefix,"AZURE_API_VERSION"), default_value = "2025-01-01-preview")]
            pub azure_api_version: String,

            #[arg(long, default_value_t = 10.0, env = concat!($prefix,"OPENAI_BILLING_CAP"))]
            pub biling_cap: f64,

            #[arg(long, env = concat!($prefix,"OPENAI_API_MODEL"), default_value = "o1")]
            pub model: OpenAIModel,

            #[arg(long, env = concat!($prefix,"LLM_DEBUG"))]
            pub llm_debug: Option<PathBuf>,

            #[arg(long, env = concat!($prefix, "LLM_TEMPERATURE"), default_value_t = 0.8)]
            pub llm_temperature: f32,

            #[arg(long, env = concat!($prefix, "LLM_PRESENCE_PENALTY"), default_value_t = 0.0)]
            pub llm_presence_penalty: f32,

            #[arg(long, env = concat!($prefix, "LLM_PROMPT_TIMEOUT"), default_value_t = 20 * 60)]
            pub llm_prompt_timeout: u64,

            #[arg(long, env = concat!($prefix, "LLM_RETRY"), default_value_t = 5)]
            pub llm_retry: u64,

            #[arg(long, env = concat!($prefix, "LLM_MAX_COMPLETION_TOKENS"), default_value_t = 16384)]
            pub llm_max_completion_tokens: u32,

            #[arg(long, env = concat!($prefix, "LLM_TOOL_CHOINCE"))]
            pub llm_tool_choice: Option<LLMToolChoice>,

            #[arg(
                long,
                env = concat!($prefix, "LLM_STREAM"),
                default_value_t = false,
                value_parser = clap::builder::BoolishValueParser::new()
            )]
            pub llm_stream: bool,

            #[arg(
                long,
                env = concat!($prefix, "LLM_REASONING_EFFORT"),
            )]
            pub reasoning_effort: Option<Reasoning>
        }

        impl $struct_name {
            pub fn settings(&self) -> LLMSettings {
                LLMSettings {
                    llm_temperature: self.llm_temperature,
                    llm_presence_penalty: self.llm_presence_penalty,
                    llm_prompt_timeout: self.llm_prompt_timeout,
                    llm_retry: self.llm_retry,
                    llm_max_completion_tokens: self.llm_max_completion_tokens,
                    llm_tool_choice: self.llm_tool_choice.clone(),
                    llm_stream: self.llm_stream,
                    reasoning_effort: self.reasoning_effort.clone()
                }
            }

            pub fn to_config(&self) -> SupportedConfig {
                if let Some(ep) = self.azure_openai_endpoint.as_ref() {
                    SupportedConfig::new_azure(
                        ep,
                        self.openai_key.clone().unwrap_or_default().as_str(),
                        self.azure_deployment
                            .as_ref()
                            .unwrap_or(&self.model.to_string())
                            .as_str(),
                        &self.azure_api_version
                    )
                } else {
                    SupportedConfig::new(&self.openai_url, self.openai_key.clone().unwrap_or_default().as_str())
                }
            }

            pub fn to_llm(self) -> LLM {
                let config = self.to_config();
                let model = self.model.clone();
                let debug_path = self.llm_debug.clone();
                LLM::new(config, model, self.biling_cap, self.settings(), Some($prefix.to_string()), debug_path)
            }
        }
    };
}

make_openai_args!(OpenAISetup, "");
make_openai_args!(OptOpenAISetup, "OPT_");
make_openai_args!(OptOptOpenAISetup, "OPT_OPT_");
