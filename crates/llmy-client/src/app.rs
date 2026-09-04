//! Application identity for outgoing requests: the `User-Agent` (plus any
//! app-specific headers) announced to the endpoint. The identity can be
//! llmy's own, mimic a known client application (Claude Code, Codex CLI),
//! or carry a caller-provided override. It is applied as the default header
//! set of the underlying HTTP client, so every request of every protocol
//! carries it.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use color_eyre::eyre::eyre;
use llmy_types::error::LLMYError;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, USER_AGENT};

/// Client version announced by the Claude Code preset.
const CLAUDE_CODE_VERSION: &str = "2.0.62";
/// SDK version announced in the Claude Code preset's stainless headers.
const CLAUDE_CODE_SDK_VERSION: &str = "0.70.1";
/// Client version announced by the Codex preset.
const CODEX_VERSION: &str = "0.50.0";

/// A known application whose request annotation can be mimicked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppIdentityPreset {
    /// llmy announcing itself honestly.
    Llmy,
    /// Claude Code's CLI headers.
    Claude,
    /// Codex CLI's headers.
    Codex,
}

impl FromStr for AppIdentityPreset {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "llmy" => Ok(Self::Llmy),
            "claude" | "claude-code" | "claude_code" => Ok(Self::Claude),
            "codex" => Ok(Self::Codex),
            other => Err(format!(
                "unknown app preset {other:?}; available: llmy, claude, codex"
            )),
        }
    }
}

impl fmt::Display for AppIdentityPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Llmy => "llmy",
            Self::Claude => "claude",
            Self::Codex => "codex",
        };
        f.write_str(name)
    }
}

/// The concrete header set announced to the endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppIdentity {
    pub user_agent: String,
    /// Additional headers beyond `User-Agent`, name -> value.
    pub headers: BTreeMap<String, String>,
}

impl AppIdentity {
    pub fn preset(preset: AppIdentityPreset) -> Self {
        match preset {
            AppIdentityPreset::Llmy => Self::llmy(),
            AppIdentityPreset::Claude => Self::claude_code(),
            AppIdentityPreset::Codex => Self::codex(),
        }
    }

    /// llmy's own identity.
    pub fn llmy() -> Self {
        Self {
            user_agent: format!("llmy/{}", env!("CARGO_PKG_VERSION")),
            headers: BTreeMap::new(),
        }
    }

    /// The header set Claude Code sends: its CLI `User-Agent`, the `x-app`
    /// marker, and the TypeScript SDK's stainless telemetry headers matched
    /// to the current host.
    pub fn claude_code() -> Self {
        let mut headers = BTreeMap::new();
        headers.insert("x-app".to_string(), "cli".to_string());
        headers.insert("x-stainless-lang".to_string(), "js".to_string());
        headers.insert("x-stainless-runtime".to_string(), "node".to_string());
        headers.insert(
            "x-stainless-runtime-version".to_string(),
            "v22.14.0".to_string(),
        );
        headers.insert(
            "x-stainless-package-version".to_string(),
            CLAUDE_CODE_SDK_VERSION.to_string(),
        );
        headers.insert(
            "x-stainless-os".to_string(),
            Self::stainless_os().to_string(),
        );
        headers.insert(
            "x-stainless-arch".to_string(),
            Self::stainless_arch().to_string(),
        );
        headers.insert("x-stainless-retry-count".to_string(), "0".to_string());
        Self {
            user_agent: format!("claude-cli/{CLAUDE_CODE_VERSION} (external, cli)"),
            headers,
        }
    }

    /// The header set Codex CLI sends: its Rust client `User-Agent` and the
    /// `originator` marker.
    pub fn codex() -> Self {
        let mut headers = BTreeMap::new();
        headers.insert("originator".to_string(), "codex_cli_rs".to_string());
        Self {
            user_agent: format!(
                "codex_cli_rs/{CODEX_VERSION} ({}; {})",
                Self::codex_os(),
                std::env::consts::ARCH
            ),
            headers,
        }
    }

    /// A caller-provided `User-Agent`, no extra headers.
    pub fn custom(user_agent: impl Into<String>) -> Self {
        Self {
            user_agent: user_agent.into(),
            headers: BTreeMap::new(),
        }
    }

    fn stainless_os() -> &'static str {
        match std::env::consts::OS {
            "linux" => "Linux",
            "macos" => "MacOS",
            "windows" => "Windows",
            _ => "Unknown",
        }
    }

    fn stainless_arch() -> &'static str {
        match std::env::consts::ARCH {
            "x86_64" => "x64",
            "aarch64" => "arm64",
            "x86" => "x32",
            _ => "unknown",
        }
    }

    fn codex_os() -> &'static str {
        match std::env::consts::OS {
            "linux" => "Linux",
            "macos" => "Mac OS",
            "windows" => "Windows",
            other => other,
        }
    }

    /// The identity as a header map, `User-Agent` included. Header values
    /// are sanitized to printable ASCII (anything else cannot go on the
    /// wire); a header whose name is invalid is skipped. Both cases warn
    /// instead of failing, so an identity can always be applied.
    pub fn header_map(&self) -> HeaderMap {
        let mut map = HeaderMap::new();
        if let Some(user_agent) = Self::sanitized_value(&self.user_agent) {
            map.insert(USER_AGENT, user_agent);
        }
        for (name, value) in &self.headers {
            let Ok(name) = HeaderName::from_str(name) else {
                tracing::warn!("skipping app identity header with invalid name {:?}", name);
                continue;
            };
            if let Some(value) = Self::sanitized_value(value) {
                map.insert(name, value);
            }
        }
        map
    }

    fn sanitized_value(value: &str) -> Option<HeaderValue> {
        let filtered: String = value
            .chars()
            .filter(|c| (' '..='~').contains(c))
            .collect::<String>()
            .trim()
            .to_string();
        if filtered != value {
            tracing::warn!(
                "app identity header value {:?} sanitized to {:?} (printable ASCII only)",
                value,
                filtered
            );
        }
        match HeaderValue::from_str(&filtered) {
            Ok(value) => Some(value),
            Err(error) => {
                tracing::warn!(
                    "skipping unrepresentable header value {:?}: {}",
                    value,
                    error
                );
                None
            }
        }
    }

    /// An HTTP client carrying this identity on every request.
    pub fn http_client(&self) -> Result<reqwest::Client, LLMYError> {
        reqwest::Client::builder()
            .default_headers(self.header_map())
            .build()
            .map_err(|e| eyre!("failed to build http client for app identity: {}", e).into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_parse_and_render() {
        assert_eq!(
            "claude".parse::<AppIdentityPreset>(),
            Ok(AppIdentityPreset::Claude)
        );
        assert_eq!(
            "Claude-Code".parse::<AppIdentityPreset>(),
            Ok(AppIdentityPreset::Claude)
        );
        assert_eq!(
            "codex".parse::<AppIdentityPreset>(),
            Ok(AppIdentityPreset::Codex)
        );
        assert_eq!(
            "llmy".parse::<AppIdentityPreset>(),
            Ok(AppIdentityPreset::Llmy)
        );
        assert!("gemini".parse::<AppIdentityPreset>().is_err());
        assert_eq!(AppIdentityPreset::Claude.to_string(), "claude");
    }

    #[test]
    fn claude_preset_carries_cli_marker_headers() {
        let identity = AppIdentity::preset(AppIdentityPreset::Claude);
        assert!(identity.user_agent.starts_with("claude-cli/"));
        assert!(identity.user_agent.ends_with("(external, cli)"));
        assert_eq!(
            identity.headers.get("x-app").map(String::as_str),
            Some("cli")
        );
        assert_eq!(
            identity.headers.get("x-stainless-lang").map(String::as_str),
            Some("js")
        );

        let map = identity.header_map();
        assert_eq!(
            map.get(USER_AGENT).and_then(|v| v.to_str().ok()),
            Some(identity.user_agent.as_str())
        );
        assert!(map.contains_key("x-stainless-os"));
    }

    #[test]
    fn codex_preset_carries_originator() {
        let identity = AppIdentity::preset(AppIdentityPreset::Codex);
        assert!(identity.user_agent.starts_with("codex_cli_rs/"));
        assert_eq!(
            identity.headers.get("originator").map(String::as_str),
            Some("codex_cli_rs")
        );
    }

    #[test]
    fn custom_identity_is_just_a_user_agent() {
        let identity = AppIdentity::custom("my-audit-bot/1.0");
        assert!(identity.headers.is_empty());
        let map = identity.header_map();
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn control_characters_are_sanitized_not_fatal() {
        let map = AppIdentity::custom("bad\nagent/1.0").header_map();
        assert_eq!(
            map.get(USER_AGENT).and_then(|v| v.to_str().ok()),
            Some("badagent/1.0")
        );
    }

    #[test]
    fn llmy_preset_announces_the_crate_version() {
        let identity = AppIdentity::llmy();
        assert_eq!(
            identity.user_agent,
            format!("llmy/{}", env!("CARGO_PKG_VERSION"))
        );
    }
}
