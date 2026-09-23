use std::sync::OnceLock;

use rust_decimal::Decimal;
use tiktoken_rs::CoreBPE;

// Build-time generated data
mod generated_models {
    include!(concat!(env!("OUT_DIR"), "/models_generated.rs"));
}
mod generated_claude {
    include!(concat!(env!("OUT_DIR"), "/claude_generated.rs"));
}
mod generated_deepseek {
    include!(concat!(env!("OUT_DIR"), "/deepseek_generated.rs"));
}
mod generated_qwen {
    include!(concat!(env!("OUT_DIR"), "/qwen_generated.rs"));
}
mod generated_glm {
    include!(concat!(env!("OUT_DIR"), "/glm_generated.rs"));
}

pub use generated_models::ModelId;

// ---------------------------------------------------------------------------
// Encoding enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    Cl100kBase,
    O200kBase,
    P50kBase,
    Claude,
    DeepSeek,
    Qwen,
    Glm,
}

impl Encoding {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "cl100k_base" => Some(Self::Cl100kBase),
            "o200k_base" => Some(Self::O200kBase),
            "p50k_base" => Some(Self::P50kBase),
            "claude" => Some(Self::Claude),
            "deepseek" => Some(Self::DeepSeek),
            "qwen" => Some(Self::Qwen),
            "glm" => Some(Self::Glm),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Cl100kBase => "cl100k_base",
            Self::O200kBase => "o200k_base",
            Self::P50kBase => "p50k_base",
            Self::Claude => "claude",
            Self::DeepSeek => "deepseek",
            Self::Qwen => "qwen",
            Self::Glm => "glm",
        }
    }
}

// ---------------------------------------------------------------------------
// Model config (mirrors data/models.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ModelTokens {
    pub content_multiplier: f64,
    pub base_overhead: i32,
    pub per_message: i32,
    pub tools_exist: i32,
    pub per_tool: i32,
    pub per_desc: i32,
    pub per_first_prop: i32,
    pub per_additional_prop: i32,
    pub per_prop_desc: i32,
    pub per_enum: i32,
    pub per_nested_object: i32,
    pub per_array_of_objects: i32,
}

/// How a model's prompt cache is addressed — i.e. what a caller has to do to
/// get a cache hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CachePolicy {
    /// The provider transparently caches the longest matching prefix of the
    /// prompt. There is nothing to declare on the request, and writes are
    /// normally free. This is classic OpenAI prompt caching, and the default.
    #[default]
    PartialPrefix,
    /// Caching happens only at breakpoints the caller declares (OpenAI's
    /// `prompt_cache_breakpoint` from GPT-5.6 on, Anthropic's `cache_control`).
    /// Writes are billed at [`ModelPricing::input_cache_write`].
    Breakpoint,
}

impl CachePolicy {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PartialPrefix => "partial_prefix",
            Self::Breakpoint => "breakpoint",
        }
    }

    /// Whether the caller has to mark cache breakpoints to get a cache hit.
    pub fn needs_breakpoints(&self) -> bool {
        matches!(self, Self::Breakpoint)
    }
}

impl std::fmt::Display for CachePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Time-of-day discounted pricing — DeepSeek's peak/off-peak (峰谷)
/// mechanism. The base [`ModelPricing`] rates are the PEAK rates, in force
/// inside the provider's peak windows; at every other moment a request
/// bills at the discounted rates here.
#[derive(Debug, Clone, Copy)]
pub struct OffPeakPricing {
    /// Daily UTC windows `[start, end)` in minutes-of-day during which the
    /// base (peak) rates apply. A window wrapping midnight has start > end.
    pub peak_windows: &'static [(u32, u32)],
    /// Peak windows apply Monday–Friday (UTC) only; weekends bill off-peak
    /// around the clock. Provider public holidays are NOT modeled — a
    /// holiday falling on a weekday bills at peak in our ledger, an
    /// over-estimate but never an under-estimate.
    pub peak_weekdays_only: bool,
    /// Per-token USD price for uncached input off-peak.
    pub input: Decimal,
    /// Per-token USD price for output off-peak.
    pub output: Decimal,
    /// Per-token USD price for cache reads off-peak (falls back to `input`
    /// of this tier when absent).
    pub input_cache_read: Option<Decimal>,
    /// Per-token USD price for cache writes off-peak.
    pub input_cache_write: Option<Decimal>,
}

impl OffPeakPricing {
    /// Whether `at` falls inside a peak window, i.e. the base rates apply.
    pub fn is_peak(&self, at: chrono::DateTime<chrono::Utc>) -> bool {
        use chrono::{Datelike, Timelike, Weekday};
        if self.peak_weekdays_only && matches!(at.weekday(), Weekday::Sat | Weekday::Sun) {
            return false;
        }
        let minute = at.hour() * 60 + at.minute();
        self.peak_windows.iter().any(|(start, end)| {
            if start <= end {
                (*start..*end).contains(&minute)
            } else {
                minute >= *start || minute < *end
            }
        })
    }
}

/// The long-context pricing tier: once a request's prompt exceeds
/// `threshold` input tokens, the provider reprices the ENTIRE request at
/// these rates — OpenAI's rule for the GPT-6 series (2x input/cache, 1.5x
/// output past 272K input tokens), not a marginal rate on the overflow.
#[derive(Debug, Clone, Copy)]
pub struct LongContextPricing {
    /// Input-token count (cached tokens included) beyond which these rates
    /// apply to the whole request.
    pub threshold: u64,
    /// Per-token USD price for uncached input at long context.
    pub input: Decimal,
    /// Per-token USD price for output at long context.
    pub output: Decimal,
    /// Per-token USD price for cache reads at long context (falls back to
    /// this tier's `input` when absent).
    pub input_cache_read: Option<Decimal>,
    /// Per-token USD price for cache writes at long context.
    pub input_cache_write: Option<Decimal>,
}

#[derive(Debug, Clone, Copy)]
pub struct ModelPricing {
    /// Per-token USD price for uncached input.
    pub input: Decimal,
    /// Per-token USD price for output.
    pub output: Decimal,
    /// Per-token USD price for cache reads (falls back to `input` when absent).
    pub input_cache_read: Option<Decimal>,
    /// Per-token USD price for cache writes.
    pub input_cache_write: Option<Decimal>,
    /// Long-context tier; `None` means one flat tier at any length.
    pub long_context: Option<LongContextPricing>,
    /// Peak/off-peak time-of-day discount; `None` means the same rates
    /// around the clock.
    pub off_peak: Option<OffPeakPricing>,
}

impl ModelPricing {
    /// The rates one request bills at, given its prompt size and the moment
    /// it is billed: off-peak windows discount the whole request first, and
    /// otherwise the long-context tier reprices it once the prompt (cached
    /// tokens included) exceeds the tier's threshold. No known model has
    /// both mechanisms; if one ever does, its off-peak rates carry no
    /// length tiers here. Exact per request; summing tiered costs over
    /// several requests is NOT the same as costing their summed usage, so
    /// aggregate spend must be accumulated from per-request costs.
    pub fn for_request(
        &self,
        input_tokens: u64,
        at: chrono::DateTime<chrono::Utc>,
    ) -> ModelPricing {
        if let Some(off) = self.off_peak
            && !off.is_peak(at)
        {
            return ModelPricing {
                input: off.input,
                output: off.output,
                input_cache_read: off.input_cache_read,
                input_cache_write: off.input_cache_write,
                long_context: None,
                off_peak: None,
            };
        }
        match self.long_context {
            Some(long) if input_tokens > long.threshold => ModelPricing {
                input: long.input,
                output: long.output,
                input_cache_read: long.input_cache_read,
                input_cache_write: long.input_cache_write,
                long_context: None,
                off_peak: None,
            },
            _ => *self,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Vendor / namespace prefix (e.g. `Some("google")` for `google/gemini-3.1-pro-preview`).
    pub owner: Option<String>,
    /// Identifier sent to the upstream API (the part after `owner/`).
    pub model_name: String,
    pub encoding: String,
    pub tokens: ModelTokens,
    /// Human-friendly display name (e.g. `"Gemini 3.1 Pro Preview"`).
    pub name: String,
    pub max_input_tokens: u64,
    pub max_tokens: u64,
    pub pricing: Option<ModelPricing>,
    /// How this model's prompt cache is addressed. Defaults to
    /// [`CachePolicy::PartialPrefix`] for models that don't declare one.
    pub cache_policy: CachePolicy,
}

impl ModelConfig {
    pub fn encoding(&self) -> Option<Encoding> {
        Encoding::from_str(&self.encoding)
    }

    /// How this model's prompt cache is addressed.
    pub fn cache_policy(&self) -> CachePolicy {
        self.cache_policy
    }

    pub fn max_input(&self) -> u64 {
        self.max_input_tokens
    }

    pub fn max_output(&self) -> u64 {
        self.max_tokens
    }

    pub fn count_tokens(&self, text: &str) -> Option<usize> {
        self.encoding().map(|enc| count_tokens(text, enc))
    }

    /// Lossy counting tokens, fallback to simple len // 4 if no encoder is available
    pub fn count_tokens_lossy(&self, text: &str) -> usize {
        self.count_tokens(text).unwrap_or_else(|| text.len() / 4)
    }
}

// ---------------------------------------------------------------------------
// ModelId convenience helpers (the variants themselves are generated)
// ---------------------------------------------------------------------------

impl ModelId {
    /// Vendor / namespace prefix derived from the canonical id.
    pub fn owner(&self) -> Option<&str> {
        self.as_str().split_once('/').map(|(o, _)| o)
    }

    /// Bare model name sent to the upstream API.
    pub fn model_name(&self) -> &str {
        self.as_str()
            .split_once('/')
            .map_or_else(|| self.as_str(), |(_, m)| m)
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Registry helpers (back compat with the previous Vec/HashMap API)
// ---------------------------------------------------------------------------

/// Iterate over all built-in models with their canonical ids and configs.
pub fn models() -> Vec<(&'static str, ModelConfig)> {
    ModelId::ALL_KNOWN
        .iter()
        .filter_map(|id| id.to_model_config().map(|c| (id.as_str(), c)))
        .collect()
}

pub fn get_model(model_id: &str) -> Option<ModelConfig> {
    ModelId::parse_known(model_id).and_then(|id| id.to_model_config())
}

pub fn encoding_for_model(model_id: &str) -> Option<Encoding> {
    get_model(model_id).and_then(|m| Encoding::from_str(&m.encoding))
}

// ---------------------------------------------------------------------------
// Embedded BPEs (built from pre-decoded binary data)
// ---------------------------------------------------------------------------

fn build_embedded_bpe(
    data: &[u8],
    token_count: u32,
    special_tokens: &[(&str, u32)],
    pat_str: &str,
) -> CoreBPE {
    let encoder = {
        let mut pos = 0;
        let mut entries = Vec::with_capacity(token_count as usize);
        while pos < data.len() {
            let rank = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            entries.push((data[pos..pos + len].to_vec(), rank));
            pos += len;
        }
        entries.into_iter().collect()
    };

    let special = special_tokens
        .iter()
        .map(|&(k, v)| (k.to_string(), v))
        .collect();

    CoreBPE::new(encoder, special, pat_str).expect("build embedded BPE")
}

// ---------------------------------------------------------------------------
// BPE singletons
// ---------------------------------------------------------------------------

static CLAUDE_BPE: OnceLock<CoreBPE> = OnceLock::new();
static DEEPSEEK_BPE: OnceLock<CoreBPE> = OnceLock::new();
static QWEN_BPE: OnceLock<CoreBPE> = OnceLock::new();
static GLM_BPE: OnceLock<CoreBPE> = OnceLock::new();
static CL100K_BPE: OnceLock<CoreBPE> = OnceLock::new();
static O200K_BPE: OnceLock<CoreBPE> = OnceLock::new();
static P50K_BPE: OnceLock<CoreBPE> = OnceLock::new();

pub fn get_bpe(encoding: Encoding) -> &'static CoreBPE {
    match encoding {
        Encoding::Cl100kBase => {
            CL100K_BPE.get_or_init(|| tiktoken_rs::cl100k_base().expect("init cl100k_base"))
        }
        Encoding::O200kBase => {
            O200K_BPE.get_or_init(|| tiktoken_rs::o200k_base().expect("init o200k_base"))
        }
        Encoding::P50kBase => {
            P50K_BPE.get_or_init(|| tiktoken_rs::p50k_base().expect("init p50k_base"))
        }
        Encoding::Claude => CLAUDE_BPE.get_or_init(|| {
            build_embedded_bpe(
                generated_claude::CLAUDE_BPE_DATA,
                generated_claude::CLAUDE_TOKEN_COUNT,
                generated_claude::CLAUDE_SPECIAL_TOKENS,
                generated_claude::CLAUDE_PAT_STR,
            )
        }),
        Encoding::DeepSeek => DEEPSEEK_BPE.get_or_init(|| {
            build_embedded_bpe(
                generated_deepseek::DEEPSEEK_BPE_DATA,
                generated_deepseek::DEEPSEEK_TOKEN_COUNT,
                generated_deepseek::DEEPSEEK_SPECIAL_TOKENS,
                generated_deepseek::DEEPSEEK_PAT_STR,
            )
        }),
        Encoding::Qwen => QWEN_BPE.get_or_init(|| {
            build_embedded_bpe(
                generated_qwen::QWEN_BPE_DATA,
                generated_qwen::QWEN_TOKEN_COUNT,
                generated_qwen::QWEN_SPECIAL_TOKENS,
                generated_qwen::QWEN_PAT_STR,
            )
        }),
        Encoding::Glm => GLM_BPE.get_or_init(|| {
            build_embedded_bpe(
                generated_glm::GLM_BPE_DATA,
                generated_glm::GLM_TOKEN_COUNT,
                generated_glm::GLM_SPECIAL_TOKENS,
                generated_glm::GLM_PAT_STR,
            )
        }),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Encode text into token ids using the given encoding.
pub fn encode(text: &str, encoding: Encoding) -> Vec<u32> {
    let bpe = get_bpe(encoding);
    bpe.encode_with_special_tokens(text)
}

/// Count tokens in the text for the given encoding.
pub fn count_tokens(text: &str, encoding: Encoding) -> usize {
    encode(text, encoding).len()
}

/// Count tokens using a model identifier (e.g. "openai/gpt-4o").
/// Returns `None` if the model is unknown.
pub fn count_tokens_for_model(text: &str, model_id: &str) -> Option<usize> {
    encoding_for_model(model_id).map(|enc| count_tokens(text, enc))
}

/// List all known model IDs.
pub fn model_ids() -> Vec<&'static str> {
    let mut ids: Vec<&str> = ModelId::ALL_KNOWN.iter().map(|id| id.as_str()).collect();
    ids.sort();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl100k_hello() {
        let tokens = encode("Hello, world!", Encoding::Cl100kBase);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_o200k_hello() {
        let tokens = encode("Hello, world!", Encoding::O200kBase);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_p50k_hello() {
        let tokens = encode("Hello, world!", Encoding::P50kBase);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_claude_hello() {
        let tokens = encode("Hello, world!", Encoding::Claude);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn deepseek_and_qwen_match_the_reference_tokenizer() {
        // Golden counts computed with the upstream HF `tokenizers` runtime on
        // the published tokenizer.json files (DeepSeek-V4-Flash and
        // Qwen3.8-Flash-Next).
        let samples = [
            "hello world",
            "The quick brown fox jumps over the lazy dog.",
            "你好，世界！今天天气怎么样？",
            "fn main() { println!(\"{}\", 42); }",
            "  indented\n\nmulti-line   text with   spaces\n",
        ];
        let deepseek = [2usize, 10, 8, 12, 12];
        let qwen = [2usize, 10, 8, 11, 12];
        let glm = [2usize, 10, 8, 10, 12];
        for (s, want) in samples.iter().zip(deepseek) {
            assert_eq!(encode(s, Encoding::DeepSeek).len(), want, "{s:?}");
        }
        for (s, want) in samples.iter().zip(qwen) {
            assert_eq!(encode(s, Encoding::Qwen).len(), want, "{s:?}");
        }
        // The GLM vocab is GLM-5.3's; earlier zai generations share its base
        // (151k of the 154k entries), so it stands in for them too.
        for (s, want) in samples.iter().zip(glm) {
            assert_eq!(encode(s, Encoding::Glm).len(), want, "{s:?}");
        }
    }

    #[test]
    fn test_model_lookup() {
        assert_eq!(
            encoding_for_model("openai/gpt-4o"),
            Some(Encoding::O200kBase)
        );
        assert_eq!(
            encoding_for_model("anthropic/claude-sonnet-4"),
            Some(Encoding::Claude)
        );
        assert_eq!(encoding_for_model("nonexistent/model"), None);
    }

    #[test]
    fn test_count_tokens_for_model() {
        let count = count_tokens_for_model("Hello, world!", "openai/gpt-4o");
        assert!(count.is_some());
        assert!(count.unwrap() > 0);
    }

    #[test]
    fn test_model_config_limits() {
        let model = get_model("openai/gpt-5.1").expect("known model");

        assert_eq!(model.max_input_tokens, 272000);
        assert_eq!(model.max_tokens, 128000);
        assert_eq!(model.max_input(), 272000);
        assert_eq!(model.max_output(), 128000);
        assert_eq!(model.owner.as_deref(), Some("openai"));
        assert_eq!(model.model_name, "gpt-5.1");
    }

    #[test]
    fn test_mimo_model_config() {
        let model = get_model("mimo/mimo-v2.5-pro").expect("known model");

        assert_eq!(model.encoding(), Some(Encoding::O200kBase));
        assert_eq!(model.max_input_tokens, 917504);
        assert_eq!(model.max_tokens, 131072);
        assert_eq!(
            model.pricing.expect("mimo pricing").input,
            rust_decimal::dec!(0.000001)
        );
    }

    #[test]
    fn test_deepseek_v4_model_config() {
        // The retired names are `alias_of` entries served by DeepSeek V4.1
        // Flash and billed at the Flash price: $0.30/$1.20 per 1M at peak,
        // half of that off-peak (peak = UTC 01:00-04:00 and 06:00-10:00,
        // weekdays only).
        for key in [
            "deepseek/deepseek-v4-flash",
            "deepseek/deepseek-v4-flash-0731",
            "deepseek/deepseek-flash",
        ] {
            let model = get_model(key).expect("known model");

            assert_eq!(model.encoding(), Some(Encoding::DeepSeek), "{key}");
            assert_eq!(model.max_input_tokens, 655360, "{key}");
            assert_eq!(model.max_tokens, 393216, "{key}");
            let pricing = model.pricing.expect("deepseek pricing");
            assert_eq!(pricing.input, rust_decimal::dec!(0.0000003), "{key}");
            assert_eq!(pricing.output, rust_decimal::dec!(0.0000012), "{key}");

            let off = pricing.off_peak.expect("off-peak tier");
            assert_eq!(off.peak_windows, &[(60, 240), (360, 600)], "{key}");
            assert!(off.peak_weekdays_only, "{key}");
            assert_eq!(off.input * rust_decimal::dec!(2), pricing.input, "{key}");
            assert_eq!(off.output * rust_decimal::dec!(2), pricing.output, "{key}");
            assert_eq!(
                off.input_cache_read.map(|p| p * rust_decimal::dec!(2)),
                pricing.input_cache_read,
                "{key}"
            );
        }

        // An alias borrows the target's config but keeps its own key as the
        // wire model name — proxies serving retired names want the literal.
        let alias = get_model("deepseek/deepseek-v4-flash-0731").expect("alias");
        assert_eq!(alias.model_name, "deepseek-v4-flash-0731");
        assert_eq!(alias.name, "DeepSeek V4.1 Flash");
    }

    #[test]
    fn off_peak_windows_apply_outside_weekday_peaks() {
        use chrono::TimeZone;
        let pricing = get_model("deepseek/deepseek-flash")
            .expect("known model")
            .pricing
            .expect("pricing");
        let off = pricing.off_peak.expect("off-peak tier");
        let utc = |y, mo, day, h, mi| chrono::Utc.with_ymd_and_hms(y, mo, day, h, mi, 0).unwrap();

        // 2026-09-23 is a Wednesday: both peak windows apply...
        assert!(off.is_peak(utc(2026, 9, 23, 2, 0)));
        assert!(off.is_peak(utc(2026, 9, 23, 6, 0)));
        // ...half-open bounds: the end minute is already off-peak...
        assert!(off.is_peak(utc(2026, 9, 23, 3, 59)));
        assert!(!off.is_peak(utc(2026, 9, 23, 4, 0)));
        // ...and the gap between the windows is off-peak too.
        assert!(!off.is_peak(utc(2026, 9, 23, 5, 0)));
        assert!(!off.is_peak(utc(2026, 9, 23, 12, 0)));

        // The same peak hour on Saturday bills off-peak.
        assert!(!off.is_peak(utc(2026, 9, 26, 2, 0)));

        // A request billed off-peak takes the discounted rates whole.
        let discounted = pricing.for_request(1_000, utc(2026, 9, 26, 2, 0));
        assert_eq!(discounted.input, off.input);
        assert_eq!(discounted.output, off.output);
        let peak = pricing.for_request(1_000, utc(2026, 9, 23, 2, 0));
        assert_eq!(peak.input, pricing.input);

        // A synthetic window wrapping midnight covers both sides of it.
        let wrap = OffPeakPricing {
            peak_windows: &[(23 * 60, 60)],
            peak_weekdays_only: false,
            ..off
        };
        assert!(wrap.is_peak(utc(2026, 9, 23, 23, 30)));
        assert!(wrap.is_peak(utc(2026, 9, 24, 0, 30)));
        assert!(!wrap.is_peak(utc(2026, 9, 24, 1, 0)));
    }

    #[test]
    fn test_model_id_parse_and_custom() {
        let known = ModelId::parse_known("openai/gpt-4o").expect("known id");
        assert_eq!(known.as_str(), "openai/gpt-4o");
        assert_eq!(known.owner(), Some("openai"));
        assert_eq!(known.model_name(), "gpt-4o");
        assert!(known.to_model_config().is_some());

        let custom = ModelId::from_str_or_custom("totally-made-up");
        assert!(matches!(custom, ModelId::Custom(_)));
        assert_eq!(custom.as_str(), "totally-made-up");
        assert!(custom.to_model_config().is_none());
    }
}
