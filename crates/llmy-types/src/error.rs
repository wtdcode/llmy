use std::fmt::Display;

use async_openai::error::{ApiError, OpenAIError};
use rmcp::service::{ClientInitializeError, ServerInitializeError, ServiceError};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct GeneralToolCall {
    pub tool_id: String,
    pub tool_name: String,
    pub tool_args: String,
}

impl Display for GeneralToolCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "ToolCall(name={}, args={:?}, id={})",
            self.tool_name, self.tool_args, self.tool_id
        ))
    }
}

/// Details of a billing-cap exhaustion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingExhausted {
    pub cap: Decimal,
    pub current: Decimal,
    /// Name of the scope whose cap was exceeded, if it had one.
    pub scope: Option<String>,
    /// Billing-tree node id whose cap was exceeded (0 = root).
    pub node: u64,
}

impl Display for BillingExhausted {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "scope {:?} (#{}) reached cap {}, current {}",
            self.scope, self.node, self.cap, self.current
        ))
    }
}

/// A failed request rendered for a debug record, as `{"error": {...}}`.
///
/// The typed error is decomposed rather than stringified, so a failure can be
/// grouped by `kind` and its interesting cases picked out in SQL — previously
/// this was a bare `Debug` rendering, and reading it back meant regexing prose.
/// `detail` keeps that rendering for the cases the fields do not cover.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorPayload {
    pub error: ErrorFields,
}

/// The decomposed fields inside [`ErrorPayload`].
#[derive(Debug, Clone, Serialize)]
pub struct ErrorFields {
    pub kind: &'static str,
    pub message: String,
    pub detail: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filtered: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub billing: Option<BillingExhausted>,
}

/// The link OpenAI puts in its cybersecurity-filter refusal.
///
/// That refusal carries nothing machine-readable — it arrives as a plain
/// `invalid_request_error` with `code: None` — so the message is all there is to
/// go on, and a URL is the one part of it that cannot collide with ordinary
/// prose or be reworded casually.
const CYBER_REFUSAL_URL: &str = "https://chatgpt.com/cyber";

/// The error, if this is OpenAI turning the prompt away on policy grounds
/// rather than because the request was malformed.
///
/// Only `ApiError` is read: a deserialize failure carries the model's own output
/// in its message, and must never be scanned for refusal wording.
fn policy_refusal(error: &OpenAIError) -> Option<&ApiError> {
    let OpenAIError::ApiError(api) = error else {
        return None;
    };
    api.message.contains(CYBER_REFUSAL_URL).then_some(api)
}

#[derive(Error, Debug)]
pub enum LLMYError {
    #[error("io error: {0}")]
    IO(#[from] std::io::Error),
    // No `#[from]`: the conversion is hand-written below so a policy refusal
    // lands in `Filtered` instead of being buried in here.
    #[error("openai error: {0}")]
    OpenAI(OpenAIError),
    #[error("json error: {0}")]
    STDJSON(#[from] serde_json::Error),
    #[error("billing error: {0}")]
    Billing(BillingExhausted),
    #[error("incorrect tool call for tool {0} with args {1} given schema {2:?}")]
    IncorrectToolCall(String, String, schemars::Schema),
    /// A well-formed tool call that the tool itself refused: the arguments
    /// parse, but their content fails the tool's own requirements. Raised
    /// from `validate` (nothing has run yet), the agent loop discards the
    /// whole model turn and asks again; raised from `invoke` (execution
    /// started, side effects may exist), it degrades to a soft tool result
    /// like an `IncorrectToolCall`. The reason is for logs only and is never
    /// fed back to the model.
    #[error("tool call rejected: {0}: {1}")]
    ToolCallRejected(GeneralToolCall, String),
    #[error("response filtered: {0}")]
    Filtered(String),
    #[error("no choice is returned")]
    EmptyChoice,
    #[error("reach output length limit")]
    OutputLength,
    #[error("mcp server init error: {0}")]
    McpServerInit(#[from] ServerInitializeError),
    #[error("mcp client init error: {0}")]
    McpClientInit(#[from] ClientInitializeError),
    #[error("mcp service error: {0}")]
    McpService(#[from] ServiceError),
    #[error("mcp task join error: {0}")]
    McpJoin(#[from] tokio::task::JoinError),
    #[error(transparent)]
    Other(#[from] color_eyre::Report),
}

impl From<OpenAIError> for LLMYError {
    /// A refusal is a verdict on the prompt, not a transport or protocol
    /// failure, so it becomes [`LLMYError::Filtered`] and callers can tell the
    /// two apart without string-matching an opaque error.
    fn from(error: OpenAIError) -> Self {
        match policy_refusal(&error) {
            Some(api) => Self::Filtered(api.message.clone()),
            None => Self::OpenAI(error),
        }
    }
}

impl LLMYError {
    /// A rejection raised from inside a tool's own code, where the wire
    /// `tool_id` is not known; the agent loop rebinds the error to the real
    /// call before logging or propagating it.
    pub fn tool_call_rejected(
        tool_name: impl Into<String>,
        tool_args: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::ToolCallRejected(
            GeneralToolCall {
                tool_id: String::new(),
                tool_name: tool_name.into(),
                tool_args: tool_args.into(),
            },
            reason.into(),
        )
    }

    pub fn billing_exhaustion(&self) -> Option<BillingExhausted> {
        match self {
            Self::Billing(exhausted) => Some(exhausted.clone()),
            Self::Other(e) => e.chain().find_map(|v| match v.downcast_ref::<LLMYError>() {
                Some(rhs) => rhs.billing_exhaustion(),
                None => None,
            }),
            _ => None,
        }
    }

    /// Machine-readable name of this variant, so recorded failures can be
    /// grouped and filtered without parsing prose out of a `Debug` rendering.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::IO(_) => "io",
            Self::OpenAI(_) => "openai",
            Self::STDJSON(_) => "json",
            Self::Billing(_) => "billing",
            Self::IncorrectToolCall(..) => "incorrect_tool_call",
            Self::ToolCallRejected(..) => "tool_call_rejected",
            Self::Filtered(_) => "filtered",
            Self::EmptyChoice => "empty_choice",
            Self::OutputLength => "output_length",
            Self::McpServerInit(_) => "mcp_server_init",
            Self::McpClientInit(_) => "mcp_client_init",
            Self::McpService(_) => "mcp_service",
            Self::McpJoin(_) => "mcp_join",
            Self::Other(_) => "other",
        }
    }

    /// Render this error for a debug record; see [`ErrorPayload`] for why the
    /// variant is decomposed instead of stringified.
    pub fn payload(&self) -> ErrorPayload {
        ErrorPayload {
            error: ErrorFields {
                kind: self.kind(),
                message: self.to_string(),
                detail: format!("{self:?}"),
                filtered: self.filtered().map(str::to_string),
                billing: self.billing_exhaustion(),
            },
        }
    }

    /// The refusal message, if this error is — or wraps — a content filter
    /// rejection: either the model refusing in band, or OpenAI turning the
    /// prompt away (see [`policy_refusal`]).
    ///
    /// Digs through `Other` the same way [`Self::billing_exhaustion`] does,
    /// because by the time a caller sees it the refusal is usually several
    /// `wrap_err` layers down.
    pub fn filtered(&self) -> Option<&str> {
        match self {
            Self::Filtered(message) => Some(message),
            Self::Other(e) => e.chain().find_map(|v| match v.downcast_ref::<LLMYError>() {
                Some(rhs) => rhs.filtered(),
                None => None,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exhausted(scope: Option<&str>) -> BillingExhausted {
        BillingExhausted {
            cap: Decimal::from(10u32),
            current: Decimal::from(13u32),
            scope: scope.map(str::to_string),
            node: 7,
        }
    }

    fn api_error(message: &str, code: Option<&str>) -> OpenAIError {
        OpenAIError::ApiError(ApiError {
            message: message.to_string(),
            r#type: Some("invalid_request_error".to_string()),
            param: None,
            code: code.map(str::to_string),
        })
    }

    #[test]
    fn a_cyber_refusal_becomes_filtered_not_an_opaque_openai_error() {
        // Verbatim from a recorded response. Note `type` is the generic
        // `invalid_request_error` and `code` is absent, so the prose is the only
        // thing that marks it as a refusal.
        let cyber = "This content was flagged for possible cybersecurity risk. If this seems \
                     wrong, try rephrasing your request. To get authorized for security work, \
                     join the Trusted Access for Cyber program: https://chatgpt.com/cyber";
        match LLMYError::from(api_error(cyber, None)) {
            LLMYError::Filtered(message) => assert_eq!(message, cyber),
            other => panic!("expected Filtered, got {other:?}"),
        }
    }

    #[test]
    fn an_ordinary_api_error_stays_an_openai_error() {
        for message in [
            "max_tokens is too large: 999999",
            "Invalid schema for function 'read_file': 'properties' is required",
            "You didn't provide an API key.",
            // Refusal-sounding prose without the link is not enough: only the
            // cyber filter's own URL is treated as the marker.
            "This content was flagged for review.",
        ] {
            match LLMYError::from(api_error(message, None)) {
                LLMYError::OpenAI(_) => {}
                other => panic!("{message:?} should stay OpenAI, got {other:?}"),
            }
        }
    }

    #[test]
    fn only_api_errors_are_read_for_refusals() {
        // A deserialize failure echoes the model's own output back into the
        // error, so it must never be scanned for these words.
        let poisoned = OpenAIError::JSONDeserialize(
            serde_json::from_str::<serde_json::Value>("{").unwrap_err(),
            "the assistant said the content was flagged".to_string(),
        );
        match LLMYError::from(poisoned) {
            LLMYError::OpenAI(_) => {}
            other => panic!("expected OpenAI, got {other:?}"),
        }
    }

    #[test]
    fn filtered_unwraps_the_variant_and_digs_through_other() {
        assert_eq!(LLMYError::Filtered("nope".into()).filtered(), Some("nope"));

        // Buried under a couple of `wrap_err` layers, which is how a caller
        // usually meets it.
        let wrapped = LLMYError::Other(
            color_eyre::Report::new(LLMYError::Filtered("nope".into()))
                .wrap_err("while reading the file")
                .wrap_err("while answering the user"),
        );
        assert_eq!(wrapped.filtered(), Some("nope"));

        // Anything else has none.
        assert!(LLMYError::EmptyChoice.filtered().is_none());
        assert!(
            LLMYError::from(api_error("max_tokens is too large", None))
                .filtered()
                .is_none()
        );
    }

    #[test]
    fn a_cyber_refusal_is_reachable_as_filtered_end_to_end() {
        // The whole path a caller depends on: an `OpenAIError` off the wire,
        // through the `From` classifier, wrapped by an intermediate layer, and
        // still recognisable at the top.
        let cyber = format!(
            "This content was flagged for possible cybersecurity risk. \
             Join the Trusted Access for Cyber program: {CYBER_REFUSAL_URL}"
        );
        let wrapped = LLMYError::Other(
            color_eyre::Report::new(LLMYError::from(api_error(&cyber, None)))
                .wrap_err("while stepping the agent"),
        );
        assert_eq!(wrapped.filtered(), Some(cyber.as_str()));
        assert!(wrapped.billing_exhaustion().is_none());
    }

    #[test]
    fn billing_error_renders_through_the_struct_display() {
        // The wording lives in `BillingExhausted::Display` now and the variant
        // just wraps it, so pin the joined result — that seam is easy to drift.
        assert_eq!(
            exhausted(Some("planner")).to_string(),
            r#"scope Some("planner") (#7) reached cap 10, current 13"#
        );
        assert_eq!(
            LLMYError::Billing(exhausted(Some("planner"))).to_string(),
            r#"billing error: scope Some("planner") (#7) reached cap 10, current 13"#
        );
        assert_eq!(
            LLMYError::Billing(exhausted(None)).to_string(),
            "billing error: scope None (#7) reached cap 10, current 13"
        );
    }

    #[test]
    fn billing_exhaustion_unwraps_the_variant_and_digs_through_other() {
        let direct = LLMYError::Billing(exhausted(Some("planner")));
        let found = direct.billing_exhaustion().expect("billing exhaustion");
        assert_eq!(found.node, 7);
        assert_eq!(found.scope.as_deref(), Some("planner"));

        // Wrapped in a report, it still surfaces from anywhere in the chain.
        let wrapped = LLMYError::Other(
            color_eyre::Report::new(LLMYError::Billing(exhausted(None))).wrap_err("while planning"),
        );
        assert_eq!(wrapped.billing_exhaustion().map(|e| e.node), Some(7));

        // Anything else has none.
        assert!(LLMYError::EmptyChoice.billing_exhaustion().is_none());
    }

    #[test]
    fn a_refusal_payload_records_its_kind_and_message_not_just_a_blob() {
        let payload =
            serde_json::to_value(LLMYError::Filtered("flagged for cyber".into()).payload())
                .expect("serialize payload");
        let error = &payload["error"];

        assert_eq!(error["kind"], "filtered");
        assert_eq!(error["message"], "response filtered: flagged for cyber");
        // The interesting part is pulled out, so it is greppable in SQL rather
        // than buried in the `Debug` text.
        assert_eq!(error["filtered"], "flagged for cyber");
        assert!(
            error["detail"]
                .as_str()
                .expect("detail is a string")
                .contains("Filtered")
        );
        assert!(error.get("billing").is_none());
    }

    #[test]
    fn a_cap_exhaustion_payload_records_the_structured_billing_details() {
        let payload =
            serde_json::to_value(LLMYError::Billing(exhausted(Some("planner"))).payload())
                .expect("serialize payload");
        let error = &payload["error"];

        assert_eq!(error["kind"], "billing");
        assert_eq!(error["billing"]["node"], 7);
        assert_eq!(error["billing"]["scope"], "planner");
        assert!(error.get("filtered").is_none());
    }

    #[test]
    fn an_unremarkable_error_payload_still_records_kind_and_message() {
        let payload =
            serde_json::to_value(LLMYError::EmptyChoice.payload()).expect("serialize payload");
        assert_eq!(payload["error"]["kind"], "empty_choice");
        assert_eq!(payload["error"]["message"], "no choice is returned");
        assert!(payload["error"].get("filtered").is_none());
        assert!(payload["error"].get("billing").is_none());
    }
}
