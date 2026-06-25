use clap::Args;
use color_eyre::eyre::eyre;
use llmy_client::{client::*, model::OpenAIModel, settings::*};
use llmy_types::error::LLMYError;

/// Default OpenAI-compatible base URL, used when neither `--openai-url` nor
/// `--azure-openai-endpoint` is provided.
const DEFAULT_OPENAI_URL: &str = "https://api.openai.com/v1";

macro_rules! make_openai_args {
    ($struct_name:ident, $prefix:literal, $long:literal) => {
        #[derive(Args, Clone, Debug)]
        pub struct $struct_name {
            #[arg(
                long = concat!($long, "openai-url"),
                env = concat!($prefix, "OPENAI_API_URL"),
            )]
            pub openai_url: Option<String>,

            #[arg(long = concat!($long, "azure-openai-endpoint"), env = concat!($prefix, "AZURE_OPENAI_ENDPOINT"))]
            pub azure_openai_endpoint: Option<String>,

            #[arg(long = concat!($long, "openai-key"), env = concat!($prefix, "OPENAI_API_KEY"))]
            pub openai_key: Option<String>,

            #[arg(long = concat!($long, "azure-deployment"), env = concat!($prefix, "AZURE_API_DEPLOYMENT"))]
            pub azure_deployment: Option<String>,

            #[arg(long = concat!($long, "azure-api-version"), env = concat!($prefix, "AZURE_API_VERSION"), default_value = "2025-01-01-preview")]
            pub azure_api_version: String,

            #[arg(long = concat!($long, "biling-cap"), default_value_t = rust_decimal::dec!(10.0), env = concat!($prefix, "OPENAI_BILLING_CAP"))]
            pub biling_cap: rust_decimal::Decimal,

            #[arg(long = concat!($long, "model"), env = concat!($prefix, "OPENAI_API_MODEL"))]
            pub model: Option<OpenAIModel>,

            /// Send the canonical `owner/name` model id (e.g. `openai/gpt-5.4-mini`)
            /// in chat completion requests instead of the bare model name.
            /// Required for aggregators like OpenRouter.
            #[arg(
                long = concat!($long, "use-full-model-id"),
                env = concat!($prefix, "LLM_FULL_MODEL_NAME"),
                default_value_t = false,
                value_parser = clap::builder::BoolishValueParser::new()
            )]
            pub use_full_model_id: bool,

            /// Where to dump LLM interaction logs. A directory path enables the
            /// folder backend (one xml/json pair per request); a value starting
            /// with `sqlite3://` or ending in `sqlite3` enables the SQLite
            /// backend.
            #[arg(long = concat!($long, "llm-debug"), env = concat!($prefix, "LLM_DEBUG"))]
            pub llm_debug: Option<String>,

            #[arg(long = concat!($long, "llm-temperature"), env = concat!($prefix, "LLM_TEMPERATURE"))]
            pub llm_temperature: Option<f32>,

            #[arg(long = concat!($long, "llm-presence-penalty"), env = concat!($prefix, "LLM_PRESENCE_PENALTY"))]
            pub llm_presence_penalty: Option<f32>,

            #[arg(long = concat!($long, "llm-prompt-timeout"), env = concat!($prefix, "LLM_PROMPT_TIMEOUT"), default_value_t = 20 * 60)]
            pub llm_prompt_timeout: u64,

            #[arg(long = concat!($long, "llm-retry"), env = concat!($prefix, "LLM_RETRY"), default_value_t = 5)]
            pub llm_retry: u64,

            #[arg(long = concat!($long, "llm-max-completion-tokens"), env = concat!($prefix, "LLM_MAX_COMPLETION_TOKENS"))]
            pub llm_max_completion_tokens: Option<u32>,

            #[arg(long = concat!($long, "llm-tool-choice"), env = concat!($prefix, "LLM_TOOL_CHOINCE"))]
            pub llm_tool_choice: Option<LLMToolChoice>,

            #[arg(
                long = concat!($long, "llm-stream"),
                env = concat!($prefix, "LLM_STREAM"),
                default_value_t = false,
                value_parser = clap::builder::BoolishValueParser::new()
            )]
            pub llm_stream: bool,

            #[arg(long = concat!($long, "top-p"), env = concat!($prefix, "LLM_TOP_P"))]
            pub top_p: Option<f32>,

            #[arg(
                long = concat!($long, "reasoning-effort"),
                env = concat!($prefix, "LLM_REASONING_EFFORT"),
            )]
            pub reasoning_effort: Option<Reasoning>,

            /// On a typed/JSON deserialize failure, retry the parse after stripping
            /// a markdown code fence (```json ... ```) from the content.
            #[arg(
                long = concat!($long, "llm-auto-strip"),
                env = concat!($prefix, "LLM_AUTO_STRIP"),
                default_value_t = true,
                value_parser = clap::builder::BoolishValueParser::new()
            )]
            pub auto_strip: bool
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
                    top_p: self.top_p,
                    reasoning_effort: self.reasoning_effort.clone(),
                    auto_strip: self.auto_strip,
                }
            }

            /// Build the upstream config from the URL/endpoint flags:
            /// - neither set      => OpenAI at [`DEFAULT_OPENAI_URL`]
            /// - only `openai-url`   => OpenAI at that URL
            /// - only `azure-...`    => Azure
            /// - both set            => error (ambiguous)
            pub fn to_config(&self) -> Result<SupportedConfig, LLMYError> {
                let key = self.openai_key.clone().unwrap_or_default();
                match (&self.openai_url, &self.azure_openai_endpoint) {
                    (Some(_), Some(_)) => Err(LLMYError::Other(eyre!(
                        "both --openai-url and --azure-openai-endpoint are set; provide only one"
                    ))),
                    (None, Some(ep)) => {
                        // Azure deployment names are user-chosen and almost never
                        // contain `/`; fall back to the bare model name, not the
                        // canonical `owner/name` form. The model is only needed
                        // for that fallback, so require it only when no explicit
                        // deployment is given.
                        let deployment = match self.azure_deployment.as_deref() {
                            Some(deployment) => deployment.to_string(),
                            None => self
                                .model
                                .as_ref()
                                .ok_or_else(|| {
                                    LLMYError::Other(eyre!(
                                        "azure config needs --azure-deployment or a model id to \
                                         derive it from"
                                    ))
                                })?
                                .model_name()
                                .to_string(),
                        };
                        Ok(SupportedConfig::new_azure(
                            ep,
                            key.as_str(),
                            &deployment,
                            &self.azure_api_version,
                        ))
                    }
                    (openai_url, None) => {
                        let url = openai_url.as_deref().unwrap_or(DEFAULT_OPENAI_URL);
                        Ok(SupportedConfig::new(url, key.as_str()))
                    }
                }
            }


            async fn llm_new_inner(&self, model: OpenAIModel) -> Result<LLM, LLMYError> {
                let config = self.to_config()?;
                let debug_target = self.llm_debug.clone();
                let model = model.with_full_id(self.use_full_model_id);
                LLM::new_async(
                    config,
                    model,
                    self.biling_cap,
                    self.settings(),
                    Some($prefix.to_string()),
                    debug_target,
                )
                .await
            }

            pub async fn may_llm(self) -> Result<Option<LLM>, LLMYError> {
                let Some(model) = self.model.clone() else { return Ok(None); };
                Ok(Some(self.llm_new_inner(model).await?))
            }

            pub async fn to_llm(self) -> LLM {
                let model = self.model.clone().expect("LLM model not given");
                self.llm_new_inner(model)
                    .await
                    .expect("construct LLM")
            }
        }
    };
}

make_openai_args!(OpenAISetup, "", "");
make_openai_args!(OptOpenAISetup, "OPT_", "opt-");
make_openai_args!(OptOptOpenAISetup, "OPT_OPT_", "opt-opt-");
