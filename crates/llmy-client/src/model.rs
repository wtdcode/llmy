use std::{fmt, str::FromStr};

use rust_decimal::{Decimal, dec};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub use llmy_tokenizer::{CachePolicy, ModelConfig, ModelId, ModelPricing, ModelTokens};

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
                },
                3 => ModelPricing {
                    input: values[0] / per_million,
                    output: values[1] / per_million,
                    input_cache_read: Some(values[2] / per_million),
                    input_cache_write: None,
                },
                4 => ModelPricing {
                    input: values[0] / per_million,
                    output: values[1] / per_million,
                    input_cache_read: Some(values[2] / per_million),
                    input_cache_write: Some(values[3] / per_million),
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
        tracing::info!("No valid model detected for {}, assume not billed", s);
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
            max_input_tokens: 0,
            max_tokens: 0,
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
        if canonical == name
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
    fn custom_pricing_reuses_registered_model_limits() {
        let model = OpenAIModel::from_str("DeepSeek V4 Flash,0.5,1.5,0.1").unwrap();
        let pricing = model.pricing();

        assert_eq!(model.model_id_str(), "deepseek/deepseek-v4-flash");
        assert_eq!(model.owner(), Some("deepseek"));
        assert_eq!(model.model_name(), "deepseek-v4-flash");
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
    fn custom_pricing_still_accepts_unknown_models() {
        let model = OpenAIModel::from_str("custom-model,2,4").unwrap();
        let pricing = model.pricing();

        assert_eq!(model.model_id_str(), "custom-model");
        assert!(matches!(model.model_id(), ModelId::Custom(_)));
        assert_eq!(model.owner(), None);
        assert_eq!(model.model_name(), "custom-model");
        assert_eq!(model.config.max_input_tokens, 0);
        assert_eq!(model.config.max_tokens, 0);
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
