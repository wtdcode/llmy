use std::{
    convert::Infallible,
    ops::{Deref, DerefMut},
    str::FromStr,
};

use color_eyre::eyre::eyre;
use llmy_types::other::WithOtherFields;

use crate::req::{
    ChatCompletionNamedToolChoiceCustomRaw, ChatCompletionToolChoiceOption,
    ChatCompletionToolChoiceOptionRaw, CustomNameRaw, ReasoningEffort, ToolChoiceOptions,
};

#[derive(Debug, Clone)]
pub struct LLMToolChoice(pub ChatCompletionToolChoiceOption);

impl FromStr for LLMToolChoice {
    type Err = Infallible;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let raw = match s {
            "auto" => ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Auto),
            "required" => ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::Required),
            "none" => ChatCompletionToolChoiceOptionRaw::Mode(ToolChoiceOptions::None),
            _ => ChatCompletionToolChoiceOptionRaw::Custom(WithOtherFields::new(
                ChatCompletionNamedToolChoiceCustomRaw {
                    custom: WithOtherFields::new(CustomNameRaw {
                        name: s.to_string(),
                    }),
                },
            )),
        };
        Ok(Self(WithOtherFields::new(raw)))
    }
}

// Hand-written instead of derived: the CLI string format has sugar the wire
// JSON shape does not — any string that is not a mode names a custom tool —
// and TOML profiles must accept exactly the CLI strings, not the wire form.
impl<'de> serde::Deserialize<'de> for LLMToolChoice {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(deserializer)?;
        raw.parse()
            .map_err(|infallible: std::convert::Infallible| match infallible {})
    }
}

impl Deref for LLMToolChoice {
    type Target = ChatCompletionToolChoiceOption;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LLMToolChoice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<ChatCompletionToolChoiceOption> for LLMToolChoice {
    fn from(value: ChatCompletionToolChoiceOption) -> Self {
        Self(value)
    }
}

impl From<LLMToolChoice> for ChatCompletionToolChoiceOption {
    fn from(value: LLMToolChoice) -> Self {
        value.0
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(transparent)]
pub struct Reasoning(pub ReasoningEffort);

impl Reasoning {
    pub fn is_none(&self) -> bool {
        matches!(self.0, ReasoningEffort::None)
    }
}

impl FromStr for Reasoning {
    type Err = color_eyre::Report;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self(ReasoningEffort::None)),
            "minimal" => Ok(Self(ReasoningEffort::Minimal)),
            "low" => Ok(Self(ReasoningEffort::Low)),
            "medium" => Ok(Self(ReasoningEffort::Medium)),
            "high" => Ok(Self(ReasoningEffort::High)),
            "xhigh" => Ok(Self(ReasoningEffort::Xhigh)),
            _ => Err(eyre!("unknown effort: {}", s)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LLMSettings {
    pub llm_temperature: Option<f32>,
    pub llm_presence_penalty: Option<f32>,
    pub llm_prompt_timeout: u64,
    pub llm_retry: u64,
    /// Pause before the first retry of a failed request, in seconds; `0` retries at once.
    /// Each further retry waits `llm_retry_backoff_factor` times longer, capped at
    /// `llm_retry_backoff_max_secs`.
    pub llm_retry_backoff_secs: f64,
    /// Multiplier applied to the pause on every retry after the first.
    pub llm_retry_backoff_factor: f64,
    /// Upper bound on the pause between retries, in seconds.
    pub llm_retry_backoff_max_secs: f64,
    /// How many times an agent step re-asks the model after its turn is
    /// discarded in validation — a malformed tool call (`IncorrectToolCall`)
    /// or a tool's own rejection (`ToolCallRejected`) — before the error
    /// surfaces to the caller. The re-ask starts from a clean context.
    pub tool_reject_retries: u64,
    /// Whether a call to a tool that does not exist discards the model's
    /// turn like a malformed call, sharing `tool_reject_retries` (the
    /// default), instead of feeding a soft "tool not defined" result back.
    /// The soft path lets the model learn the actual roster — e.g. after a
    /// tool was removed at runtime — at the cost of the failed attempt
    /// staying in context.
    pub unknown_tool_hard_reject: bool,
    /// Cap on concurrently in-flight requests through one client (all its
    /// scopes/clones share the limiter); 0 = unlimited. Applied when the
    /// client is constructed — per-request settings overrides don't resize it.
    pub llm_concurrent: usize,
    pub llm_max_completion_tokens: Option<u32>,
    pub llm_tool_choice: Option<LLMToolChoice>,
    pub llm_stream: bool,
    pub top_p: Option<f32>,
    pub reasoning_effort: Option<Reasoning>,
    /// When a typed/JSON completion fails to deserialize, retry the parse after
    /// stripping a markdown code fence from the content (see `MarkdownTagFilter`).
    pub auto_strip: bool,
    /// Pick a `prompt_cache_key` for requests that don't come with one, based on
    /// what this client has already sent (see [`crate::cache_key`]).
    pub auto_cache_key: bool,
    /// How long an auto cache key survives without being used, in seconds.
    pub cache_key_ttl: u64,
    /// Requests per minute one auto cache key takes before we spread to another.
    pub cache_key_rpm: u32,
    /// Emit the running billing line at INFO once every this many tokens; every
    /// other request logs it at DEBUG. `0` puts every request at INFO.
    pub billing_log_tokens: u64,
    /// How far the local token estimate may drift from the provider's count, in
    /// percent, before the comparison is logged at INFO instead of DEBUG.
    pub token_estimate_pct: f64,
    /// Allow a request whose wire format differs from the backend's protocol
    /// to be implicitly converted (through the chat form) instead of being
    /// refused (`LLMY_ALLOW_IMPLICIT_CONVERT`). Only requests already in the
    /// backend's own format are sent without it.
    pub allow_implicit_convert: bool,
    /// Application identity announced on every request (`User-Agent` plus
    /// app marker headers): llmy itself, a mimicked known client, or a
    /// custom value. Like `llm_concurrent`, applied when the client is
    /// constructed — per-request settings overrides don't change it.
    pub llm_app: Option<crate::app::AppIdentity>,
}

impl LLMSettings {
    /// The auto cache key policy these settings describe.
    pub fn cache_key_config(&self) -> crate::cache_key::CacheKeyConfig {
        crate::cache_key::CacheKeyConfig {
            enabled: self.auto_cache_key,
            ttl: std::time::Duration::from_secs(self.cache_key_ttl),
            max_rpm: self.cache_key_rpm,
        }
    }

    pub fn timeout(&self) -> std::time::Duration {
        if self.llm_prompt_timeout == 0 {
            std::time::Duration::MAX
        } else {
            std::time::Duration::from_secs(self.llm_prompt_timeout)
        }
    }

    /// How long to wait before retry number `retry` (1 = the first retry): the base pause
    /// grown by the factor once per earlier retry, capped at the maximum; zero when the base
    /// is zero or `retry` is zero.
    pub fn retry_backoff(&self, retry: u64) -> std::time::Duration {
        if retry == 0 || !(self.llm_retry_backoff_secs > 0.0) {
            return std::time::Duration::ZERO;
        }
        let grown = self.llm_retry_backoff_secs
            * self
                .llm_retry_backoff_factor
                .max(1.0)
                .powi(i32::try_from(retry - 1).unwrap_or(i32::MAX));
        let capped = if self.llm_retry_backoff_max_secs > 0.0 {
            grown.min(self.llm_retry_backoff_max_secs)
        } else {
            grown
        };
        std::time::Duration::from_secs_f64(capped)
    }
}

#[cfg(test)]
mod retry_backoff_tests {
    use super::LLMSettings;

    fn settings(base: f64, factor: f64, max: f64) -> LLMSettings {
        LLMSettings {
            llm_temperature: None,
            llm_presence_penalty: None,
            llm_prompt_timeout: 0,
            llm_retry: 5,
            llm_retry_backoff_secs: base,
            llm_retry_backoff_factor: factor,
            llm_retry_backoff_max_secs: max,
            tool_reject_retries: 0,
            unknown_tool_hard_reject: true,
            llm_concurrent: 0,
            llm_max_completion_tokens: None,
            llm_tool_choice: None,
            llm_stream: false,
            top_p: None,
            reasoning_effort: None,
            auto_strip: false,
            auto_cache_key: false,
            cache_key_ttl: 0,
            cache_key_rpm: 0,
            billing_log_tokens: 0,
            token_estimate_pct: 0.0,
            allow_implicit_convert: false,
            llm_app: None,
        }
    }

    #[test]
    fn retry_backoff_grows_and_caps() {
        let s = settings(1.0, 2.0, 16.0);
        assert_eq!(s.retry_backoff(0).as_secs_f64(), 0.0);
        assert_eq!(s.retry_backoff(1).as_secs_f64(), 1.0);
        assert_eq!(s.retry_backoff(2).as_secs_f64(), 2.0);
        assert_eq!(s.retry_backoff(4).as_secs_f64(), 8.0);
        assert_eq!(s.retry_backoff(9).as_secs_f64(), 16.0);
    }

    #[test]
    fn retry_backoff_zero_base_disables_and_zero_cap_is_unbounded() {
        assert!(settings(0.0, 2.0, 16.0).retry_backoff(3).is_zero());
        assert_eq!(
            settings(1.0, 2.0, 0.0).retry_backoff(9).as_secs_f64(),
            256.0
        );
    }
}
