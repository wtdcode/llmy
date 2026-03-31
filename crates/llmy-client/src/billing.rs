use std::fmt::Display;

use crate::model::OpenAIModel;
use llmy_types::error::LLMYError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBilling {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_tokens: u64,
    pub reasoning_tokens: u64,
    pub current: f64,
    pub cap: f64,
}

impl Display for ModelBilling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Billing(budget={:.4}/{}, inputs={}({:.2}% cached for {}), outputs={}({:.2}% reasoning for {}))",
            self.current, self.cap, self.input_tokens, if self.input_tokens == 0 {
                0.0f64
            } else {
                self.cache_tokens as f64 / self.input_tokens as f64
            }, self.cache_tokens, self.output_tokens, if self.output_tokens == 0 {
                0.0f64
            } else {
                self.reasoning_tokens as f64 / self.output_tokens as f64
            }, self.reasoning_tokens
        ))
    }
}

impl ModelBilling {
    pub fn new(cap: f64) -> Self {
        Self {
            current: 0.0,
            cap,
            input_tokens: 0,
            output_tokens: 0,
            cache_tokens: 0,
            reasoning_tokens: 0,
        }
    }

    pub fn in_cap(&self) -> bool {
        self.current <= self.cap
    }

    pub fn input_tokens(
        &mut self,
        model: &OpenAIModel,
        input_wihout_cache_count: u64,
        cached_count: u64,
    ) -> Result<(), LLMYError> {
        let pricing = model.pricing();

        let cached_price = pricing.input_cache_read.unwrap_or(pricing.input);

        let cached_usd = cached_price * (cached_count as f64);
        let raw_input_usd = pricing.input * (input_wihout_cache_count as f64);
        self.input_tokens += input_wihout_cache_count + cached_count;
        self.cache_tokens += cached_count;
        tracing::debug!(
            "Input token usage: cached {:.4} USD, {} tokens / input: {:.4} USD, {} tokens",
            cached_usd,
            cached_count,
            raw_input_usd,
            input_wihout_cache_count
        );
        self.current += cached_usd + raw_input_usd;

        if self.in_cap() {
            Ok(())
        } else {
            Err(LLMYError::Billing(self.cap, self.current))
        }
    }

    pub fn output_tokens(
        &mut self,
        model: &OpenAIModel,
        count_without_reasoning: u64,
        reasoning: u64,
    ) -> Result<(), LLMYError> {
        let pricing = model.pricing();

        let output_usd = pricing.output * (count_without_reasoning as f64);
        let reason_usd = pricing.output * (reasoning as f64);
        tracing::debug!(
            "Output token usage: {:.4} USD, {} tokens / reason: {:.4} USD, {} tokens",
            output_usd,
            count_without_reasoning,
            reason_usd,
            reasoning
        );
        self.current += output_usd;
        self.current += reason_usd;

        self.output_tokens += count_without_reasoning + reasoning;
        self.reasoning_tokens += reasoning;
        if self.in_cap() {
            Ok(())
        } else {
            Err(LLMYError::Billing(self.cap, self.current))
        }
    }
}
