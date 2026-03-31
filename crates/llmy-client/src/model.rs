use std::{fmt, str::FromStr};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub use llmy_tokenizer::{ModelConfig, ModelPricing, ModelTokens};

#[derive(Debug, Clone)]
pub struct OpenAIModel {
    model_id: String,
    pub config: ModelConfig,
}

impl OpenAIModel {
    pub fn model_id(&self) -> &str {
        &self.model_id
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
        (self.config.context_window, self.config.max_tokens)
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

            return Ok(Self {
                model_id: name.to_string(),
                config: ModelConfig {
                    encoding: "o200k_base".to_string(),
                    tokens: ModelTokens::default(),
                    name: name.to_string(),
                    context_window: 0,
                    max_tokens: 0,
                    pricing: Some(pricing),
                },
            });
        }

        // Case-insensitive match against registry model short names
        for (id, config) in llmy_tokenizer::models() {
            let short = id.rsplit('/').next().unwrap_or(id);
            if id == s || short.eq_ignore_ascii_case(s) || config.name.eq_ignore_ascii_case(s) {
                return Ok(Self {
                    model_id: id.to_string(),
                    config: config.clone(),
                });
            }
        }

        // Unknown model, zero pricing
        tracing::info!("No valid model detected, assume not billed");
        Ok(Self {
            model_id: s.to_string(),
            config: ModelConfig {
                encoding: "o200k_base".to_string(),
                tokens: ModelTokens::default(),
                name: s.to_string(),
                context_window: 0,
                max_tokens: 0,
                pricing: None,
            },
        })
    }
}
