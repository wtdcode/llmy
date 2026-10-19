use std::{fmt, str::FromStr};

use rust_decimal::{Decimal, dec};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub use llmy_tokenizer::{
    CachePolicy, LongContextPricing, ModelConfig, ModelId, ModelPricing, ModelTokens,
};

#[derive(Debug, Clone)]
pub struct OpenAIModel {
    model_id: ModelId,
    pub config: ModelConfig,
    /// When true, the wire/`Display` form is the canonical `owner/name` id
    /// (e.g. `"openai/gpt-5.4-mini"`) instead of the bare `name`. Aggregators
    /// like OpenRouter expect this form.
    use_full_id: bool,
}

impl OpenAIModel {
    /// The strongly-typed model identifier.
    pub fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Canonical id string (e.g. `"google/gemini-3.1-pro-preview"`).
    pub fn model_id_str(&self) -> &str {
        self.model_id.as_str()
    }

    /// Bare model name (the part after `owner/`). E.g. `"gpt-4o"`.
    pub fn model_name(&self) -> &str {
        &self.config.model_name
    }

    /// Vendor / namespace prefix (e.g. `Some("google")`).
    pub fn owner(&self) -> Option<&str> {
        self.config.owner.as_deref()
    }

    pub fn is_mimo(&self) -> bool {
        self.owner() == Some("mimo")
    }

    pub fn is_google(&self) -> bool {
        self.owner() == Some("google")
    }

    pub fn is_qwen(&self) -> bool {
        self.owner() == Some("alibaba")
    }

    /// Whether outgoing chat-completion requests send the canonical
    /// `owner/name` id (e.g. for OpenRouter) instead of the bare model name.
    pub fn use_full_id(&self) -> bool {
        self.use_full_id
    }

    /// Toggle whether outgoing chat-completion requests send the canonical
    /// `owner/name` id or the bare model name. Display/serialization is
    /// unaffected — they always emit the canonical id.
    pub fn with_full_id(mut self, flag: bool) -> Self {
        self.use_full_id = flag;
        self
    }

    /// In-place equivalent of [`Self::with_full_id`].
    pub fn set_full_id(&mut self, flag: bool) {
        self.use_full_id = flag;
    }

    /// The identifier to send in the `model` field of a chat-completion
    /// request. Defaults to the bare name; switches to the canonical
    /// `owner/name` id when [`Self::use_full_id`] is set.
    pub fn api_model_name(&self) -> &str {
        if self.use_full_id {
            self.model_id_str()
        } else {
            self.model_name()
        }
    }

    /// Per-token USD pricing. Returns zero pricing if unavailable.
    pub fn pricing(&self) -> ModelPricing {
        self.config.pricing.unwrap_or(ModelPricing {
            input: Decimal::ZERO,
            output: Decimal::ZERO,
            input_cache_read: None,
            input_cache_write: None,
            long_context: None,
            off_peak: None,
        })
    }

    /// How this model's prompt cache is addressed. Callers use this to decide
    /// whether marking breakpoints (see
    /// `RawExtensibleChatRequestMessage::break`) buys them anything.
    pub fn cache_policy(&self) -> CachePolicy {
        self.config.cache_policy
    }

    pub fn info(&self) -> (u64, u64) {
        (self.config.max_input_tokens, self.config.max_tokens)
    }

    /// Whether an answer reporting `served` as its model comes from the model
    /// a request asked for as `requested`. Providers routinely answer with a
    /// more specific name than the one asked for, so beyond a case-insensitive
    /// exact match this accepts:
    /// - an `owner/` prefix on either side, which aggregators and proxies add
    ///   or drop;
    /// - a dated snapshot of the requested name — `-2024-08-06` (OpenAI),
    ///   `-20250929` (Anthropic), `-0731` (DeepSeek, Qwen) — including a
    ///   requested `-latest` answered by any snapshot;
    /// - two names the registry resolves to one model: an alias and its
    ///   target, or two aliases of the same target. Aliases carry their
    ///   target's display name, which the registry keeps unique per model.
    pub fn is_same_model(requested: &str, served: &str) -> bool {
        let requested_lower = requested.to_ascii_lowercase();
        let served_lower = served.to_ascii_lowercase();
        let requested_bare = requested_lower
            .rsplit_once('/')
            .map_or(requested_lower.as_str(), |(_, name)| name);
        let served_bare = served_lower
            .rsplit_once('/')
            .map_or(served_lower.as_str(), |(_, name)| name);
        if requested_bare == served_bare {
            return true;
        }

        let snapshot_of = requested_bare
            .strip_suffix("-latest")
            .unwrap_or(requested_bare);
        let is_snapshot = served_bare
            .strip_prefix(snapshot_of)
            .and_then(|rest| rest.strip_prefix('-'))
            .is_some_and(|stamp| {
                let groups_are_digits = stamp
                    .split('-')
                    .all(|group| !group.is_empty() && group.bytes().all(|b| b.is_ascii_digit()));
                let digits = stamp.bytes().filter(u8::is_ascii_digit).count();
                groups_are_digits && (digits == 4 || digits == 8)
            });
        if is_snapshot {
            return true;
        }

        match (
            find_registered_model(requested_bare),
            find_registered_model(served_bare),
        ) {
            (Some((_, requested)), Some((_, served))) => requested.name == served.name,
            _ => false,
        }
    }
}

impl fmt::Display for OpenAIModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display is always the canonical `owner/name` id, independent of the
        // wire-format toggle. Use [`Self::api_model_name`] for the value to put
        // in an outgoing chat-completion request.
        f.write_str(self.model_id_str())
    }
}

impl Serialize for OpenAIModel {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for OpenAIModel {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Self::from_str(&s).map_err(serde::de::Error::custom)
    }
}

impl FromStr for OpenAIModel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Custom pricing format: name,input,output[,cache_read[,cache_write]]
        // Values are per-1M-token USD (converted to per-token for storage)
        if let Some(comma_pos) = s.find(',') {
            let name = &s[..comma_pos];
            let rest = &s[comma_pos + 1..];
            let values: Vec<Decimal> = rest
                .split(',')
                .map(|t| Decimal::from_str(t.trim()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;

            // CLI prices are per-1M-token USD; convert to per-token. Decimal
            // division by a power of ten is exact (it just shifts the scale).
            let per_million = dec!(1_000_000);
            let pricing = match values.len() {
                2 => ModelPricing {
                    input: values[0] / per_million,
                    output: values[1] / per_million,
                    input_cache_read: None,
                    input_cache_write: None,
                    long_context: None,
                    off_peak: None,
                },
                3 => ModelPricing {
                    input: values[0] / per_million,
                    output: values[1] / per_million,
                    input_cache_read: Some(values[2] / per_million),
                    input_cache_write: None,
                    long_context: None,
                    off_peak: None,
                },
                4 => ModelPricing {
                    input: values[0] / per_million,
                    output: values[1] / per_million,
                    input_cache_read: Some(values[2] / per_million),
                    input_cache_write: Some(values[3] / per_million),
                    long_context: None,
                    off_peak: None,
                },
                _ => {
                    return Err(
                        "expected: name,input,output[,cache_read[,cache_write]]".to_string()
                    );
                }
            };

            if let Some((model_id, mut config)) = find_registered_model(name.trim()) {
                config.pricing = Some(pricing);
                return Ok(Self {
                    model_id,
                    config,
                    use_full_id: false,
                });
            }

            return Ok(custom_model(name.trim(), Some(pricing)));
        }

        // Case-insensitive match against registry model short names
        if let Some((model_id, config)) = find_registered_model(s) {
            return Ok(Self {
                model_id,
                config,
                use_full_id: false,
            });
        }

        // Unknown model, zero pricing
        tracing::info!("{} is not in the model registry", s);
        Ok(custom_model(s, None))
    }
}

fn custom_model(s: &str, pricing: Option<ModelPricing>) -> OpenAIModel {
    let (owner, model_name) = match s.split_once('/') {
        Some((owner, name)) => (Some(owner.to_string()), name.to_string()),
        None => (None, s.to_string()),
    };
    OpenAIModel {
        model_id: ModelId::Custom(s.to_string()),
        config: ModelConfig {
            owner,
            model_name: model_name.clone(),
            encoding: "o200k_base".to_string(),
            tokens: ModelTokens::default(),
            name: model_name,
            // An unknown model still deserves sane bounds; 256k for both
            // context and output matches typical frontier ceilings, and the
            // `llm-max-context` / `llm-max-output` flags override it.
            max_input_tokens: 262_144,
            max_tokens: 262_144,
            pricing,
            // Unknown model: assume the conservative, do-nothing-required policy.
            cache_policy: CachePolicy::default(),
        },
        use_full_id: false,
    }
}

fn find_registered_model(name: &str) -> Option<(ModelId, ModelConfig)> {
    for id in ModelId::ALL_KNOWN {
        let canonical = id.as_str();
        let short = id.model_name();
        let config = id.to_model_config()?;
        if canonical.eq_ignore_ascii_case(name)
            || short.eq_ignore_ascii_case(name)
            || config.name.eq_ignore_ascii_case(name)
        {
            return Some((id.clone(), config));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_answer_under_a_more_specific_name_is_the_same_model() {
        for (requested, served) in [
            ("gpt-4o", "gpt-4o"),
            ("GPT-4o", "gpt-4o"),
            // Aggregators and proxies add or drop the owner.
            ("openai/gpt-4o", "gpt-4o"),
            ("gpt-4o", "openai/gpt-4o"),
            // Dated snapshots, in each provider's stamp format.
            ("gpt-4o", "gpt-4o-2024-08-06"),
            ("gpt-4", "gpt-4-0613"),
            ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
            ("claude-3-5-sonnet-latest", "claude-3-5-sonnet-20241022"),
            ("deepseek-v4-flash", "deepseek-v4-flash-0731"),
            ("qwen3.8-max", "qwen3.8-max-0902"),
            // Registry aliases: alias and target, and two aliases of one target.
            ("deepseek-v4-flash", "deepseek-flash"),
            ("deepseek-v4.1-flash", "deepseek-v4-flash-0731"),
        ] {
            assert!(
                OpenAIModel::is_same_model(requested, served),
                "{requested} answered as {served} should match"
            );
        }
    }

    #[test]
    fn an_answer_from_another_model_is_a_mismatch() {
        for (requested, served) in [
            ("gpt-4o", "gpt-4o-mini"),
            // A stamp only excuses the name it follows.
            ("gpt-4o", "gpt-4o-mini-2024-07-18"),
            ("gpt-4o", "gpt-4o-2024-08-06-preview"),
            ("gpt-4o", "gpt-4o-12"),
            ("claude-opus-4-5", "claude-sonnet-4-5-20250929"),
            // Both registered, but as different models.
            ("deepseek-v4-flash", "deepseek-v4-pro"),
            // A runtime substitution Codex's backend has been seen making.
            ("gpt-6-astra", "gpt-5.6-luna"),
            ("gpt-4o", "served-model"),
        ] {
            assert!(
                !OpenAIModel::is_same_model(requested, served),
                "{requested} answered as {served} should not match"
            );
        }
    }

    #[test]
    fn custom_pricing_reuses_registered_model_limits() {
        let model = OpenAIModel::from_str("DeepSeek V4.1 Flash,0.5,1.5,0.1").unwrap();
        let pricing = model.pricing();

        assert_eq!(model.model_id_str(), "deepseek/deepseek-flash");
        assert_eq!(model.owner(), Some("deepseek"));
        assert_eq!(model.model_name(), "deepseek-flash");
        assert_eq!(model.config.max_input_tokens, 655360);
        assert_eq!(model.config.max_tokens, 393216);
        assert_eq!(pricing.input, rust_decimal::dec!(0.0000005));
        assert_eq!(pricing.output, rust_decimal::dec!(0.0000015));
        assert_eq!(
            pricing.input_cache_read.unwrap(),
            rust_decimal::dec!(0.0000001)
        );
    }

    #[test]
    fn minimax_ids_go_out_in_minimax_s_own_casing() {
        // MiniMax rejects `minimax-m3.1-flash`; however the model is named,
        // the request must carry `MiniMax-M3.1-Flash`.
        for name in [
            "MiniMax-M3.1-Flash",
            "minimax-m3.1-flash",
            "minimax/minimax-m3.1-flash",
            "MiniMax M3.1 Flash",
        ] {
            let model = OpenAIModel::from_str(name).unwrap();
            assert_eq!(model.model_id_str(), "minimax/MiniMax-M3.1-Flash", "{name}");
            assert_eq!(model.api_model_name(), "MiniMax-M3.1-Flash", "{name}");
        }
        let model = OpenAIModel::from_str("minimax/MiniMax-M3")
            .unwrap()
            .with_full_id(true);
        assert_eq!(model.api_model_name(), "minimax/MiniMax-M3");

        // Every MiniMax text model the API still serves is registered and
        // goes out under exactly the id it is served as.
        for served in [
            "MiniMax-M2",
            "MiniMax-M2.1",
            "MiniMax-M2.1-highspeed",
            "MiniMax-M2.5",
            "MiniMax-M2.5-highspeed",
            "MiniMax-M2.7",
            "MiniMax-M2.7-highspeed",
            "MiniMax-M3",
            "MiniMax-M3.1-Flash",
        ] {
            let model = OpenAIModel::from_str(served).unwrap();
            assert!(!matches!(model.model_id(), ModelId::Custom(_)), "{served}");
            assert_eq!(model.api_model_name(), served);
        }
    }

    #[test]
    fn the_glm_and_qwen_flash_families_are_registered() {
        let model = OpenAIModel::from_str("glm-5.3").unwrap();
        assert_eq!(model.model_id_str(), "zai/glm-5.3");
        assert_eq!(model.config.max_input_tokens, 1_048_576);
        assert_eq!(model.config.max_tokens, 131_072);

        let model = OpenAIModel::from_str("GLM-5.3-Flash").unwrap();
        assert_eq!(model.model_id_str(), "zai/glm-5.3-flash");

        let model = OpenAIModel::from_str("glm-4.7").unwrap();
        assert_eq!(model.model_id_str(), "zai/glm-4.7");

        let model = OpenAIModel::from_str("qwen3.8-flash-next").unwrap();
        assert_eq!(model.model_id_str(), "alibaba/qwen3.8-flash-next");
        assert_eq!(model.config.max_input_tokens, 1_000_000);
        assert_eq!(model.config.max_tokens, 65_536);
    }

    #[test]
    fn custom_pricing_still_accepts_unknown_models() {
        let model = OpenAIModel::from_str("custom-model,2,4").unwrap();
        let pricing = model.pricing();

        assert_eq!(model.model_id_str(), "custom-model");
        assert!(matches!(model.model_id(), ModelId::Custom(_)));
        assert_eq!(model.owner(), None);
        assert_eq!(model.model_name(), "custom-model");
        assert_eq!(model.config.max_input_tokens, 262_144);
        assert_eq!(model.config.max_tokens, 262_144);
        assert_eq!(pricing.input, rust_decimal::dec!(0.000002));
        assert_eq!(pricing.output, rust_decimal::dec!(0.000004));
    }

    #[test]
    fn unknown_qualified_model_splits_owner_and_name() {
        let model = OpenAIModel::from_str("openrouter/some-foo").unwrap();
        assert_eq!(model.model_id_str(), "openrouter/some-foo");
        assert_eq!(model.owner(), Some("openrouter"));
        assert_eq!(model.model_name(), "some-foo");
    }

    #[test]
    fn custom_pricing_qualified_unknown_splits_owner_and_name() {
        let model = OpenAIModel::from_str("openrouter/some-foo,1,2").unwrap();
        assert_eq!(model.model_id_str(), "openrouter/some-foo");
        assert_eq!(model.owner(), Some("openrouter"));
        assert_eq!(model.model_name(), "some-foo");
    }

    #[test]
    fn cache_policy_comes_from_the_registry() {
        // Breakpoint-addressed caches: OpenAI from GPT-5.6 on, and Anthropic.
        for id in ["openai/gpt-5.6-sol", "anthropic/claude-opus-4.5"] {
            let model = OpenAIModel::from_str(id).unwrap();
            assert_eq!(model.cache_policy(), CachePolicy::Breakpoint, "{id}");
            assert!(model.cache_policy().needs_breakpoints(), "{id}");
        }
        // Everything else caches the longest matching prefix transparently.
        for id in ["openai/gpt-5.5", "openai/gpt-4o", "google/gemini-2.5-pro"] {
            let model = OpenAIModel::from_str(id).unwrap();
            assert_eq!(model.cache_policy(), CachePolicy::PartialPrefix, "{id}");
            assert!(!model.cache_policy().needs_breakpoints(), "{id}");
        }
    }

    #[test]
    fn unknown_models_default_to_partial_prefix_caching() {
        let model = OpenAIModel::from_str("openrouter/some-foo,1,2").unwrap();
        assert_eq!(model.cache_policy(), CachePolicy::PartialPrefix);
    }

    #[test]
    fn custom_pricing_keeps_the_registered_cache_policy() {
        // Overriding prices must not silently downgrade the caching model.
        let model = OpenAIModel::from_str("gpt-5.6-sol,5,30,0.5,6.25").unwrap();
        assert_eq!(model.cache_policy(), CachePolicy::Breakpoint);
    }

    #[test]
    fn display_always_emits_canonical_id() {
        let model = OpenAIModel::from_str("openai/gpt-4o").unwrap();
        assert!(!model.use_full_id());
        // Display is the canonical id regardless of the wire-format toggle.
        assert_eq!(model.to_string(), "openai/gpt-4o");
        let serialized = serde_json::to_string(&model).unwrap();
        assert_eq!(serialized, "\"openai/gpt-4o\"");
    }

    #[test]
    fn api_model_name_defaults_to_bare_name() {
        let model = OpenAIModel::from_str("openai/gpt-4o").unwrap();
        assert_eq!(model.api_model_name(), "gpt-4o");
    }

    #[test]
    fn api_model_name_with_full_id_uses_canonical_form() {
        let model = OpenAIModel::from_str("openai/gpt-4o")
            .unwrap()
            .with_full_id(true);
        assert!(model.use_full_id());
        assert_eq!(model.api_model_name(), "openai/gpt-4o");
        // Display is unaffected by the toggle.
        assert_eq!(model.to_string(), "openai/gpt-4o");
    }

    #[test]
    fn api_model_name_is_inert_for_unqualified_custom_models() {
        // A bare custom name has no owner; both forms collapse to the same string.
        let model = OpenAIModel::from_str("custom-model,2,4")
            .unwrap()
            .with_full_id(true);
        assert_eq!(model.api_model_name(), "custom-model");
        assert_eq!(model.to_string(), "custom-model");
    }
}
