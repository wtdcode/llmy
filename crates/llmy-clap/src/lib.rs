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

            /// Generic API key (`LLM_API_KEY`). The deprecated
            /// `OPENAI_API_KEY` env is still honoured at resolve time, with a
            /// removal warning; the anthropic-specific key wins on that
            /// protocol.
            // `hide_env_values` so `--help` does not print the key it picked up.
            #[arg(
                long = concat!($long, "openai-key"),
                visible_alias = concat!($long, "llm-api-key"),
                env = concat!($prefix, "LLM_API_KEY"),
                hide_env_values = true,
            )]
            pub openai_key: Option<String>,

            #[arg(long = concat!($long, "azure-deployment"), env = concat!($prefix, "AZURE_API_DEPLOYMENT"))]
            pub azure_deployment: Option<String>,

            #[arg(long = concat!($long, "azure-api-version"), env = concat!($prefix, "AZURE_API_VERSION"), default_value = "2025-01-01-preview")]
            pub azure_api_version: String,

            /// Anthropic Messages protocol endpoint (e.g.
            /// `https://api.anthropic.com/v1`). Mutually exclusive with the
            /// other endpoint flags; setting it switches the wire protocol.
            #[arg(long = concat!($long, "anthropic-url"), env = concat!($prefix, "ANTHROPIC_API_URL"))]
            pub anthropic_url: Option<String>,

            /// API key for the Anthropic endpoint; falls back to the
            /// `openai-key` flag when unset.
            #[arg(
                long = concat!($long, "anthropic-key"),
                env = concat!($prefix, "ANTHROPIC_API_KEY"),
                hide_env_values = true,
            )]
            pub anthropic_key: Option<String>,

            /// `anthropic-version` header sent with every request on the
            /// Anthropic protocol.
            #[arg(long = concat!($long, "anthropic-version"), env = concat!($prefix, "ANTHROPIC_API_VERSION"), default_value = DEFAULT_ANTHROPIC_VERSION)]
            pub anthropic_version: String,

            /// OpenAI Responses protocol endpoint (e.g.
            /// `https://api.openai.com/v1`). Mutually exclusive with the other
            /// endpoint flags; auth uses the `openai-key` flag.
            #[arg(long = concat!($long, "responses-url"), env = concat!($prefix, "RESPONSES_API_URL"))]
            pub responses_url: Option<String>,

            /// Total spend cap in USD; defaults to 10. The deprecated
            /// `OPENAI_BILLING_CAP` env is still honoured at resolve time,
            /// with a removal warning.
            #[arg(
                long = concat!($long, "biling-cap"),
                visible_alias = concat!($long, "llm-billing-cap"),
                env = concat!($prefix, "LLM_BILLING_CAP"),
            )]
            pub biling_cap: Option<rust_decimal::Decimal>,

            /// Model id. The deprecated `OPENAI_API_MODEL` env is still
            /// honoured at resolve time, with a removal warning.
            #[arg(
                long = concat!($long, "model"),
                env = concat!($prefix, "LLM_MODEL"),
            )]
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
            pub auto_strip: bool,

            /// For requests sent without an explicit `prompt_cache_key`, pick the
            /// key whose prompt prefix best matches — so consecutive turns of a
            /// conversation keep hitting the machine that cached them.
            #[arg(
                long = concat!($long, "llm-auto-cache-key"),
                env = concat!($prefix, "LLM_AUTO_CACHE_KEY"),
                default_value_t = true,
                value_parser = clap::builder::BoolishValueParser::new()
            )]
            pub auto_cache_key: bool,

            /// How long an auto cache key survives without being used, in
            /// seconds. This only bounds memory: reusing a key whose cache entry
            /// has lapsed is free, while dropping one early guarantees a miss, so
            /// the default errs long (4 hours).
            #[arg(
                long = concat!($long, "llm-cache-key-ttl"),
                env = concat!($prefix, "LLM_CACHE_KEY_TTL"),
                default_value_t = llmy_client::cache_key::DEFAULT_TTL_SECS,
            )]
            pub cache_key_ttl: u64,

            /// Requests per minute one auto cache key takes before we spread to
            /// another. OpenAI steers one key to one machine and warns that
            /// sustaining more than 15/min costs hit rate.
            #[arg(
                long = concat!($long, "llm-cache-key-rpm"),
                env = concat!($prefix, "LLM_CACHE_KEY_RPM"),
                default_value_t = llmy_client::cache_key::DEFAULT_MAX_RPM,
            )]
            pub cache_key_rpm: u32,

            /// Log the running billing total at INFO once every this many
            /// tokens; the requests in between log it at DEBUG. Set to 0 to put
            /// every request back at INFO.
            #[arg(
                long = concat!($long, "llm-billing-log-tokens"),
                env = concat!($prefix, "LLM_BILLING_LOG_TOKENS"),
                default_value_t = 1_000_000,
            )]
            pub billing_log_tokens: u64,

            /// How far the local token estimate may drift from the provider's
            /// own count, in percent, before the comparison is logged at INFO
            /// rather than DEBUG. A large drift means the tokenizer config for
            /// this model is off. The estimator renders messages its own way, so
            /// it sits a few percent high even when healthy — hence the slack.
            #[arg(
                long = concat!($long, "llm-token-estimate-pct"),
                env = concat!($prefix, "LLM_TOKEN_ESTIMATE_PCT"),
                default_value_t = 10.0,
            )]
            pub token_estimate_pct: f64,

            /// Allow a request whose wire format differs from the backend's
            /// protocol to be implicitly converted through the chat form
            /// instead of being refused. Off by default: rewriting a request
            /// into another protocol should be a conscious choice.
            #[arg(
                long = concat!($long, "allow-implicit-convert"),
                env = concat!($prefix, "LLMY_ALLOW_IMPLICIT_CONVERT"),
                default_value_t = false,
                value_parser = clap::builder::BoolishValueParser::new()
            )]
            pub allow_implicit_convert: bool
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
                    auto_cache_key: self.auto_cache_key,
                    cache_key_ttl: self.cache_key_ttl,
                    cache_key_rpm: self.cache_key_rpm,
                    billing_log_tokens: self.billing_log_tokens,
                    token_estimate_pct: self.token_estimate_pct,
                    allow_implicit_convert: self.allow_implicit_convert,
                }
            }

            /// The model id: the parsed `--model`/`LLM_MODEL` value, else the
            /// deprecated `OPENAI_API_MODEL` env (with a removal warning).
            pub fn resolved_model(&self) -> Result<Option<OpenAIModel>, LLMYError> {
                if let Some(model) = &self.model {
                    return Ok(Some(model.clone()));
                }
                let Ok(legacy) = std::env::var(concat!($prefix, "OPENAI_API_MODEL")) else {
                    return Ok(None);
                };
                tracing::warn!(concat!(
                    $prefix, "OPENAI_API_MODEL is deprecated and will be removed; use ",
                    $prefix, "LLM_MODEL instead"
                ));
                let model = legacy.parse::<OpenAIModel>().map_err(|e| {
                    LLMYError::Other(eyre!(
                        concat!("invalid ", $prefix, "OPENAI_API_MODEL: {}"),
                        e
                    ))
                })?;
                Ok(Some(model))
            }

            /// The generic API key: the parsed `--openai-key`/`LLM_API_KEY`
            /// value, else the deprecated `OPENAI_API_KEY` env (with a removal
            /// warning).
            pub fn resolved_key(&self) -> Option<String> {
                if let Some(key) = &self.openai_key {
                    return Some(key.clone());
                }
                let legacy = std::env::var(concat!($prefix, "OPENAI_API_KEY")).ok()?;
                tracing::warn!(concat!(
                    $prefix, "OPENAI_API_KEY is deprecated and will be removed; use ",
                    $prefix, "LLM_API_KEY instead"
                ));
                Some(legacy)
            }

            /// The billing cap: the parsed `--biling-cap`/`LLM_BILLING_CAP`
            /// value, else the deprecated `OPENAI_BILLING_CAP` env (with a
            /// removal warning), else 10 USD.
            pub fn resolved_billing_cap(&self) -> Result<rust_decimal::Decimal, LLMYError> {
                if let Some(cap) = self.biling_cap {
                    return Ok(cap);
                }
                match std::env::var(concat!($prefix, "OPENAI_BILLING_CAP")) {
                    Ok(legacy) => {
                        tracing::warn!(concat!(
                            $prefix,
                            "OPENAI_BILLING_CAP is deprecated and will be removed; use ",
                            $prefix, "LLM_BILLING_CAP instead"
                        ));
                        legacy.parse::<rust_decimal::Decimal>().map_err(|e| {
                            LLMYError::Other(eyre!(
                                concat!("invalid ", $prefix, "OPENAI_BILLING_CAP: {}"),
                                e
                            ))
                        })
                    }
                    Err(_) => Ok(rust_decimal::dec!(10.0)),
                }
            }

            /// Build the upstream config from the endpoint flags:
            /// - none set                    => OpenAI chat completion at [`DEFAULT_OPENAI_URL`]
            /// - only `openai-url`           => OpenAI chat completion at that URL
            /// - only `azure-...-endpoint`   => Azure chat completion
            /// - only `anthropic-url`        => Anthropic Messages protocol
            /// - only `responses-url`        => OpenAI Responses protocol
            /// - more than one               => error (ambiguous)
            pub fn to_config(&self) -> Result<SupportedConfig, LLMYError> {
                let key = self.resolved_key().unwrap_or_default();
                let mut endpoints = Vec::new();
                if self.openai_url.is_some() {
                    endpoints.push(concat!("--", $long, "openai-url"));
                }
                if self.azure_openai_endpoint.is_some() {
                    endpoints.push(concat!("--", $long, "azure-openai-endpoint"));
                }
                if self.anthropic_url.is_some() {
                    endpoints.push(concat!("--", $long, "anthropic-url"));
                }
                if self.responses_url.is_some() {
                    endpoints.push(concat!("--", $long, "responses-url"));
                }
                if endpoints.len() > 1 {
                    return Err(LLMYError::Other(eyre!(
                        "conflicting endpoint flags: {}; provide only one",
                        endpoints.join(", ")
                    )));
                }
                if let Some(url) = &self.anthropic_url {
                    // A dedicated Anthropic key wins; otherwise the generic key
                    // flag serves, so proxies fronting several protocols need
                    // only one credential.
                    let key = self.anthropic_key.clone().unwrap_or(key);
                    return Ok(SupportedConfig::new_anthropic(
                        url,
                        &key,
                        &self.anthropic_version,
                    ));
                }
                if let Some(url) = &self.responses_url {
                    return Ok(SupportedConfig::new_responses(url, key.as_str()));
                }
                if let Some(ep) = &self.azure_openai_endpoint {
                    // Azure deployment names are user-chosen and almost never
                    // contain `/`; fall back to the bare model name, not the
                    // canonical `owner/name` form. The model is only needed
                    // for that fallback, so require it only when no explicit
                    // deployment is given.
                    let deployment = match self.azure_deployment.as_deref() {
                        Some(deployment) => deployment.to_string(),
                        None => self
                            .resolved_model()?
                            .ok_or_else(|| {
                                LLMYError::Other(eyre!(
                                    "azure config needs --azure-deployment or a model id to \
                                     derive it from"
                                ))
                            })?
                            .model_name()
                            .to_string(),
                    };
                    return Ok(SupportedConfig::new_azure(
                        ep,
                        key.as_str(),
                        &deployment,
                        &self.azure_api_version,
                    ));
                }
                let url = self.openai_url.as_deref().unwrap_or(DEFAULT_OPENAI_URL);
                Ok(SupportedConfig::new(url, key.as_str()))
            }


            async fn llm_new_inner(&self, model: OpenAIModel) -> Result<LLM, LLMYError> {
                let config = self.to_config()?;
                let debug_target = self.llm_debug.clone();
                let model = model.with_full_id(self.use_full_model_id);
                LLM::new_async(
                    config,
                    model,
                    self.resolved_billing_cap()?,
                    self.settings(),
                    Some($prefix.to_string()),
                    debug_target,
                )
                .await
            }

            pub async fn may_llm(self) -> Result<Option<LLM>, LLMYError> {
                let Some(model) = self.resolved_model()? else { return Ok(None); };
                Ok(Some(self.llm_new_inner(model).await?))
            }

            pub async fn to_llm(self) -> LLM {
                let model = self
                    .resolved_model()
                    .expect("resolve model")
                    .expect("LLM model not given");
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// The opt-opt flavor: its `OPT_OPT_*` env vars are essentially never set,
    /// so parsing stays deterministic no matter the developer's environment.
    #[derive(Parser, Debug)]
    struct TestCli {
        #[command(flatten)]
        llm: OptOptOpenAISetup,
    }

    fn parse(args: &[&str]) -> OptOptOpenAISetup {
        TestCli::try_parse_from(std::iter::once("prog").chain(args.iter().copied()))
            .expect("parse args")
            .llm
    }

    #[test]
    fn each_endpoint_flag_selects_its_protocol() {
        let anthropic = parse(&["--opt-opt-anthropic-url", "https://api.anthropic.com/v1"])
            .to_config()
            .unwrap();
        assert!(matches!(anthropic, SupportedConfig::Anthropic(_)));

        let responses = parse(&["--opt-opt-responses-url", "https://api.openai.com/v1"])
            .to_config()
            .unwrap();
        assert!(matches!(responses, SupportedConfig::OpenAIResponses(_)));

        let openai = parse(&["--opt-opt-openai-url", "https://example.test/v1"])
            .to_config()
            .unwrap();
        assert!(matches!(openai, SupportedConfig::OpenAI(_)));
    }

    #[test]
    fn conflicting_endpoint_flags_are_an_error() {
        let err = parse(&[
            "--opt-opt-anthropic-url",
            "https://api.anthropic.com/v1",
            "--opt-opt-responses-url",
            "https://api.openai.com/v1",
        ])
        .to_config()
        .unwrap_err();
        let rendered = err.to_string();
        assert!(rendered.contains("--opt-opt-anthropic-url"), "{rendered}");
        assert!(rendered.contains("--opt-opt-responses-url"), "{rendered}");
    }

    #[test]
    fn the_anthropic_version_has_a_sane_default() {
        let config = parse(&["--opt-opt-anthropic-url", "https://api.anthropic.com/v1"])
            .to_config()
            .unwrap();
        match config {
            SupportedConfig::Anthropic(config) => {
                assert_eq!(config.version(), DEFAULT_ANTHROPIC_VERSION)
            }
            other => panic!("expected an anthropic config, got {other:?}"),
        }
    }

    #[test]
    fn implicit_convert_is_off_unless_asked_for() {
        assert!(!parse(&[]).settings().allow_implicit_convert);
        assert!(
            parse(&["--opt-opt-allow-implicit-convert"])
                .settings()
                .allow_implicit_convert
        );
    }

    #[test]
    fn deprecated_openai_envs_still_resolve_with_a_warning() {
        // SAFETY: test-local env keys of the opt-opt flavor, removed before
        // the test ends; no other test asserts on these settings.
        unsafe {
            std::env::set_var("OPT_OPT_LLM_MODEL", "captest,1000000,1000000");
            std::env::set_var("OPT_OPT_OPENAI_API_MODEL", "other,1000,1000");
        }
        let both = parse(&[]).resolved_model();
        unsafe {
            std::env::remove_var("OPT_OPT_LLM_MODEL");
        }
        let legacy_only = parse(&[]).resolved_model();
        unsafe {
            std::env::remove_var("OPT_OPT_OPENAI_API_MODEL");
        }

        // The clap-wired name wins when both are set...
        assert!(both.unwrap().unwrap().to_string().contains("captest"));
        // ...the deprecated one still resolves (warning aside) on its own...
        assert!(legacy_only.unwrap().unwrap().to_string().contains("other"));
        // ...and nothing set resolves to nothing.
        assert!(parse(&[]).resolved_model().unwrap().is_none());

        // Key and cap follow the same rule.
        unsafe {
            std::env::set_var("OPT_OPT_OPENAI_API_KEY", "legacy-key");
            std::env::set_var("OPT_OPT_OPENAI_BILLING_CAP", "25");
        }
        let key = parse(&[]).resolved_key();
        let cap = parse(&[]).resolved_billing_cap();
        unsafe {
            std::env::remove_var("OPT_OPT_OPENAI_API_KEY");
            std::env::remove_var("OPT_OPT_OPENAI_BILLING_CAP");
        }
        assert_eq!(key.as_deref(), Some("legacy-key"));
        assert_eq!(cap.unwrap(), rust_decimal::dec!(25));
        assert_eq!(
            parse(&[]).resolved_billing_cap().unwrap(),
            rust_decimal::dec!(10.0)
        );

        // The spelled-out flag aliases land in the same fields.
        assert_eq!(
            parse(&["--opt-opt-llm-billing-cap", "25"]).biling_cap,
            Some(rust_decimal::dec!(25))
        );
        assert_eq!(
            parse(&["--opt-opt-llm-api-key", "k1"])
                .openai_key
                .as_deref(),
            Some("k1")
        );
    }
}
