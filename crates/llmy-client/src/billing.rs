use std::fmt::Display;
use std::ops::{Add, AddAssign, Sub, SubAssign};

use crate::model::OpenAIModel;
use llmy_types::error::LLMYError;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_tokens: u64,
    pub reasoning_tokens: u64,
}

impl Display for TokenUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Usage(inputs={}({}/{:.2}% cached), outputs={}({}/{:.2}% reasoning))",
            self.input_tokens,
            self.cache_tokens,
            if self.input_tokens == 0 {
                0.0f64
            } else {
                100f64 * self.cache_tokens as f64 / self.input_tokens as f64
            },
            self.output_tokens,
            self.reasoning_tokens,
            if self.output_tokens == 0 {
                0.0f64
            } else {
                100f64 * self.reasoning_tokens as f64 / self.output_tokens as f64
            },
        ))
    }
}

impl TokenUsage {
    pub fn input_cost(&self, model: &OpenAIModel) -> Decimal {
        let pricing = model.pricing();
        let input_without_cache = self.input_tokens.saturating_sub(self.cache_tokens);
        // A model without a dedicated cache-read price bills cached tokens at the
        // normal input rate (not free) — matches the original billing logic.
        let cache_price = pricing.input_cache_read.unwrap_or(pricing.input);
        Decimal::from(input_without_cache) * pricing.input
            + Decimal::from(self.cache_tokens) * cache_price
    }
    pub fn output_cost(&self, model: &OpenAIModel) -> Decimal {
        let pricing = model.pricing();
        Decimal::from(self.output_tokens) * pricing.output
    }

    pub fn cost(&self, model: &OpenAIModel) -> Decimal {
        self.input_cost(model) + self.output_cost(model)
    }

    pub fn overflowing_add(self, rhs: Self) -> (Self, bool) {
        let (input_tokens, o1) = self.input_tokens.overflowing_add(rhs.input_tokens);
        let (output_tokens, o2) = self.output_tokens.overflowing_add(rhs.output_tokens);
        let (cache_tokens, o3) = self.cache_tokens.overflowing_add(rhs.cache_tokens);
        let (reasoning_tokens, o4) = self.reasoning_tokens.overflowing_add(rhs.reasoning_tokens);
        (
            Self {
                input_tokens,
                output_tokens,
                cache_tokens,
                reasoning_tokens,
            },
            o1 || o2 || o3 || o4,
        )
    }

    /// Field-wise overflowing subtract. The returned flag is `true` if *any*
    /// field underflowed; the value carries the wrapped result regardless.
    pub fn overflowing_sub(self, rhs: Self) -> (Self, bool) {
        let (input_tokens, o1) = self.input_tokens.overflowing_sub(rhs.input_tokens);
        let (output_tokens, o2) = self.output_tokens.overflowing_sub(rhs.output_tokens);
        let (cache_tokens, o3) = self.cache_tokens.overflowing_sub(rhs.cache_tokens);
        let (reasoning_tokens, o4) = self.reasoning_tokens.overflowing_sub(rhs.reasoning_tokens);
        (
            Self {
                input_tokens,
                output_tokens,
                cache_tokens,
                reasoning_tokens,
            },
            o1 || o2 || o3 || o4,
        )
    }

    /// Field-wise wrapping add (each field wraps on overflow).
    pub fn wrapping_add(self, rhs: Self) -> Self {
        self.overflowing_add(rhs).0
    }

    /// Field-wise wrapping subtract (each field wraps on underflow).
    pub fn wrapping_sub(self, rhs: Self) -> Self {
        self.overflowing_sub(rhs).0
    }

    /// Field-wise saturating add (each field clamps to `u64::MAX`).
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.saturating_add(rhs.input_tokens),
            output_tokens: self.output_tokens.saturating_add(rhs.output_tokens),
            cache_tokens: self.cache_tokens.saturating_add(rhs.cache_tokens),
            reasoning_tokens: self.reasoning_tokens.saturating_add(rhs.reasoning_tokens),
        }
    }

    /// Field-wise saturating subtract (each field clamps to `0`).
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.saturating_sub(rhs.input_tokens),
            output_tokens: self.output_tokens.saturating_sub(rhs.output_tokens),
            cache_tokens: self.cache_tokens.saturating_sub(rhs.cache_tokens),
            reasoning_tokens: self.reasoning_tokens.saturating_sub(rhs.reasoning_tokens),
        }
    }
}

impl Add for TokenUsage {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (sum, overflowed) = self.overflowing_add(rhs);
        assert!(!overflowed, "TokenUsage add overflowed: {self} + {rhs}");
        sum
    }
}

impl Sub for TokenUsage {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let (diff, overflowed) = self.overflowing_sub(rhs);
        assert!(!overflowed, "TokenUsage sub underflowed: {self} - {rhs}");
        diff
    }
}

impl AddAssign for TokenUsage {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for TokenUsage {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBilling {
    pub tokens: TokenUsage,
    pub current: Decimal,
    pub cap: Decimal,
}

impl Display for ModelBilling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Billing(budget={:.4}/{}, usage={})",
            self.current, self.cap, self.tokens
        ))
    }
}

impl ModelBilling {
    pub fn new(cap: Decimal) -> Self {
        Self {
            tokens: TokenUsage::default(),
            current: Decimal::ZERO,
            cap,
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
        let usage = TokenUsage {
            input_tokens: input_wihout_cache_count + cached_count,
            cache_tokens: cached_count,
            output_tokens: 0,
            reasoning_tokens: 0,
        };
        let cost = usage.cost(model);
        tracing::debug!("Input token usage: {}, cost {:.4} USD", usage, cost);
        self.tokens += usage;
        self.current += cost;

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
        let usage = TokenUsage {
            input_tokens: 0,
            cache_tokens: 0,
            output_tokens: count_without_reasoning + reasoning,
            reasoning_tokens: reasoning,
        };
        let cost = usage.cost(model);
        tracing::debug!("Output token usage: {}, cost {:.4} USD", usage, cost);
        self.tokens += usage;
        self.current += cost;
        if self.in_cap() {
            Ok(())
        } else {
            Err(LLMYError::Billing(self.cap, self.current))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usage(input: u64, output: u64, cache: u64, reasoning: u64) -> TokenUsage {
        TokenUsage {
            input_tokens: input,
            output_tokens: output,
            cache_tokens: cache,
            reasoning_tokens: reasoning,
        }
    }

    #[test]
    fn input_cost_without_cache_price_bills_cache_at_input_rate() {
        use crate::model::OpenAIModel;
        use std::str::FromStr;
        // "name,input,output" (per-1M USD) => no dedicated cache-read price.
        let model = OpenAIModel::from_str("billing-test,2,4").unwrap();
        // 100 input tokens, 40 of them cached. Cached tokens must NOT be free:
        // (60 + 40) * 2e-6 = 0.0002.
        let u = usage(100, 0, 40, 0);
        assert_eq!(u.input_cost(&model), rust_decimal::dec!(0.0002));
    }

    #[test]
    fn input_cost_with_cache_price_uses_cache_rate() {
        use crate::model::OpenAIModel;
        use std::str::FromStr;
        // "name,input,output,cache_read" => cached tokens priced at 1e-6.
        let model = OpenAIModel::from_str("billing-test,2,4,1").unwrap();
        // 60 * 2e-6 + 40 * 1e-6 = 0.00016.
        let u = usage(100, 0, 40, 0);
        assert_eq!(u.input_cost(&model), rust_decimal::dec!(0.00016));
    }

    #[test]
    fn token_usage_add_is_field_wise() {
        let a = usage(100, 40, 20, 5);
        let b = usage(30, 10, 8, 2);
        assert_eq!(a + b, usage(130, 50, 28, 7));
        // The diff is the inverse of the sum.
        assert_eq!((a + b) - b, a);
    }

    #[test]
    #[should_panic(expected = "TokenUsage sub underflowed")]
    fn token_usage_sub_panics_on_underflow() {
        let _ = usage(10, 0, 0, 0) - usage(50, 3, 1, 1);
    }

    #[test]
    #[should_panic(expected = "TokenUsage add overflowed")]
    fn token_usage_add_panics_on_overflow() {
        let _ = usage(u64::MAX, 0, 0, 0) + usage(1, 0, 0, 0);
    }

    #[test]
    fn token_usage_saturating_clamps_per_field() {
        // Underflowing fields clamp to 0, the rest subtract normally.
        assert_eq!(
            usage(10, 100, 0, 5).saturating_sub(usage(50, 30, 1, 2)),
            usage(0, 70, 0, 3)
        );
        // Overflowing fields clamp to u64::MAX.
        assert_eq!(
            usage(u64::MAX, 1, 0, 0).saturating_add(usage(1, 1, 0, 0)),
            usage(u64::MAX, 2, 0, 0)
        );
    }

    #[test]
    fn token_usage_overflowing_and_wrapping_agree_with_fields() {
        let (wrapped, overflowed) = usage(u64::MAX, 0, 0, 0).overflowing_add(usage(1, 0, 0, 0));
        assert!(overflowed);
        assert_eq!(wrapped, usage(0, 0, 0, 0));
        assert_eq!(
            usage(u64::MAX, 5, 0, 0).wrapping_add(usage(1, 0, 0, 0)),
            usage(0, 5, 0, 0)
        );

        // No out-of-range field ⇒ flag is false and the value matches `+`.
        let a = usage(7, 9, 2, 1);
        let b = usage(3, 4, 1, 1);
        assert_eq!(a.overflowing_add(b), (a + b, false));
    }
}
