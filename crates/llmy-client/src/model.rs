use std::{fmt, str::FromStr};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub use llmy_tokenizer::{ModelConfig, ModelPricing, ModelTokens};

#[derive(Debug, Clone)]
pub struct OpenAIModel {
    model_id: String, // TODO: Have enum?
    pub config: ModelConfig,
}

impl OpenAIModel {
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn is_mimo(&self) -> bool {
        self.model_id.starts_with("mimo/")
            || self
                .model_id
                .rsplit('/')
                .next()
                .is_some_and(|name| name.starts_with("mimo-"))
    }

    /// Per-token USD pricing. Returns zero pricing if unavailable.
    pub fn pricing(&self) -> ModelPricing {
        self.config.pricing.unwrap_or(ModelPricing {
            input: 0.0,
            output: 0.0,
            input_cache_read: None,
            input_cache_write: None,
        })
    }

    pub fn info(&self) -> (u64, u64) {
        (self.config.max_input_tokens, self.config.max_tokens)
    }
}

impl fmt::Display for OpenAIModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = self.model_id.rsplit('/').next().unwrap_or(&self.model_id);
        f.write_str(name)
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
            let values: Vec<f64> = rest
                .split(',')
                .map(|t| f64::from_str(t.trim()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;

            let pricing = match values.len() {
                2 => ModelPricing {
                    input: values[0] / 1e6,
                    output: values[1] / 1e6,
                    input_cache_read: None,
                    input_cache_write: None,
                },
                3 => ModelPricing {
                    input: values[0] / 1e6,
                    output: values[1] / 1e6,
                    input_cache_read: Some(values[2] / 1e6),
                    input_cache_write: None,
                },
                4 => ModelPricing {
                    input: values[0] / 1e6,
                    output: values[1] / 1e6,
                    input_cache_read: Some(values[2] / 1e6),
                    input_cache_write: Some(values[3] / 1e6),
                },
                _ => {
                    return Err(
                        "expected: name,input,output[,cache_read[,cache_write]]".to_string()
                    );
                }
            };

            if let Some((model_id, mut config)) = find_registered_model(name.trim()) {
                config.pricing = Some(pricing);
                return Ok(Self { model_id, config });
            }

            let name = name.trim().to_string();
            return Ok(Self {
                model_id: name.clone(),
                config: ModelConfig {
                    encoding: "o200k_base".to_string(),
                    tokens: ModelTokens::default(),
                    name,
                    max_input_tokens: 0,
                    max_tokens: 0,
                    pricing: Some(pricing),
                },
            });
        }

        // Case-insensitive match against registry model short names
        if let Some((model_id, config)) = find_registered_model(s) {
            return Ok(Self { model_id, config });
        }

        // Unknown model, zero pricing
        tracing::info!("No valid model detected for {}, assume not billed", s);
        Ok(Self {
            model_id: s.to_string(),
            config: ModelConfig {
                encoding: "o200k_base".to_string(),
                tokens: ModelTokens::default(),
                name: s.to_string(),
                max_input_tokens: 0,
                max_tokens: 0,
                pricing: None,
            },
        })
    }
}

fn find_registered_model(name: &str) -> Option<(String, ModelConfig)> {
    for (id, config) in llmy_tokenizer::models() {
        let short = id.rsplit('/').next().unwrap_or(id);
        if id == &name || short.eq_ignore_ascii_case(name) || config.name.eq_ignore_ascii_case(name)
        {
            return Some((id.to_string(), config.clone()));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < f64::EPSILON);
    }

    #[test]
    fn custom_pricing_reuses_registered_model_limits() {
        let model = OpenAIModel::from_str("DeepSeek V4 Flash,0.5,1.5,0.1").unwrap();
        let pricing = model.pricing();

        assert_eq!(model.model_id(), "deepseek/deepseek-v4-flash");
        assert_eq!(model.config.max_input_tokens, 655360);
        assert_eq!(model.config.max_tokens, 393216);
        assert_close(pricing.input, 5e-07);
        assert_close(pricing.output, 1.5e-06);
        assert_close(pricing.input_cache_read.unwrap(), 1e-07);
    }

    #[test]
    fn custom_pricing_still_accepts_unknown_models() {
        let model = OpenAIModel::from_str("custom-model,2,4").unwrap();
        let pricing = model.pricing();

        assert_eq!(model.model_id(), "custom-model");
        assert_eq!(model.config.max_input_tokens, 0);
        assert_eq!(model.config.max_tokens, 0);
        assert_close(pricing.input, 2e-06);
        assert_close(pricing.output, 4e-06);
    }
}
