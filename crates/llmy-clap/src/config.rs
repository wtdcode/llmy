//! The TOML-file entry point for configuring LLMs: named profiles (each one
//! the same data model as the CLI's flag set) plus the fallback-chain knobs.
//! This complements the env/flag entry point — same structs, second door.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

use clap::Args;
use color_eyre::eyre::eyre;
use llmy_client::client::{DEFAULT_FALLBACK_COOLDOWN, LLM, LLMProfile};
use llmy_types::error::LLMYError;

use crate::OpenAISetup;

/// The llmy TOML config. Profile tables reuse the CLI's own field set —
/// kebab-case keys named like the `--…` flags without the leading dashes —
/// and are env-isolated: nothing inside a profile resolves from environment
/// variables (only `LLMY_CONFIG` itself, which locates this file, is read).
///
/// ```toml
/// billing-cap = 20.0
/// fallback = ["ali", "official"]
///
/// [profiles.ali]
/// model = "deepseek-v4-flash"
/// openai-url = "https://example.test/v1"
/// openai-key = "sk-..."
///
/// [profiles.official]
/// model = "deepseek-v4.1"
/// openai-url = "https://api.deepseek.example/v1"
/// openai-key = "sk-..."
/// ```
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct LLMYConfig {
    /// Global spend cap in USD for the whole fallback chain; defaults to 10
    /// like the flag entry point. Each profile's own `biling-cap` (aliases:
    /// `billing-cap`, `llm-billing-cap`) additionally caps just that
    /// profile's slice — a profile over its own cap is skipped, spilling
    /// traffic to the ones after it.
    #[serde(default)]
    pub billing_cap: Option<rust_decimal::Decimal>,
    /// Debug backend for the whole chain (`LLM_DEBUG` syntax: a folder path
    /// or a `sqlite3://…` target). Chain-level — a profile-level `llm-debug`
    /// is ignored with a warning.
    #[serde(default)]
    pub llm_debug: Option<String>,
    /// Default cap on concurrently in-flight requests for every profile
    /// (0 = unlimited). A profile's own `llm-concurrent` overrides it — an
    /// override relation, unlike the spend caps which stack.
    #[serde(default)]
    pub llm_concurrent: Option<usize>,
    /// Profile names in fallback order, primary first. Optional when there
    /// is exactly one profile; `--llm-profiles` overrides it.
    #[serde(default)]
    pub fallback: Option<Vec<String>>,
    /// Seconds a profile that exhausted its retries is demoted to a last
    /// resort before the chain tries it first again; unset means
    /// [`DEFAULT_FALLBACK_COOLDOWN`].
    #[serde(default)]
    pub fallback_cooldown_secs: Option<f64>,
    /// The LLM profiles by name.
    pub profiles: BTreeMap<String, OpenAISetup>,
}

impl LLMYConfig {
    /// Parse a config from TOML text. Unknown keys are refused (they are
    /// almost always typos), and so is an empty profile table.
    pub fn parse_toml(raw: &str) -> Result<Self, LLMYError> {
        let config: Self = toml::from_str(raw)
            .map_err(|error| LLMYError::Other(eyre!("invalid llmy config: {error}")))?;
        if config.profiles.is_empty() {
            return Err(LLMYError::Other(eyre!(
                "the llmy config defines no [profiles.<name>] tables"
            )));
        }
        Ok(config)
    }

    /// Read and parse the config file at `path`.
    pub fn load(path: &Path) -> Result<Self, LLMYError> {
        let raw = std::fs::read_to_string(path).map_err(|error| {
            LLMYError::Other(eyre!("cannot read llmy config {}: {error}", path.display()))
        })?;
        Self::parse_toml(&raw)
            .map_err(|error| LLMYError::Other(eyre!("{}: {error}", path.display())))
    }

    /// The profile names this run uses, in fallback order: the CLI override
    /// when given, else the file's `fallback` list, else the single profile.
    /// Several profiles with no order given is refused rather than silently
    /// ordered alphabetically; so are unknown names and duplicates.
    pub fn selection(&self, cli_override: Option<&[String]>) -> Result<Vec<String>, LLMYError> {
        let names: Vec<String> = match cli_override {
            Some(names) if !names.is_empty() => names.to_vec(),
            _ => match &self.fallback {
                Some(order) if !order.is_empty() => order.clone(),
                _ if self.profiles.len() == 1 => self.profiles.keys().cloned().collect(),
                _ => {
                    return Err(LLMYError::Other(eyre!(
                        "the config has {} profiles; order them with `fallback = [..]` \
                         or --llm-profiles",
                        self.profiles.len()
                    )));
                }
            },
        };
        let mut seen = BTreeSet::new();
        for name in &names {
            if !self.profiles.contains_key(name) {
                return Err(LLMYError::Other(eyre!(
                    "unknown profile '{}'; the config defines: {}",
                    name,
                    self.profiles.keys().cloned().collect::<Vec<_>>().join(", ")
                )));
            }
            if !seen.insert(name.clone()) {
                return Err(LLMYError::Other(eyre!(
                    "profile '{}' appears twice in the fallback order",
                    name
                )));
            }
        }
        Ok(names)
    }

    /// Turn one named profile table into the client-side [`LLMProfile`].
    fn build_profile(&self, name: &str) -> Result<LLMProfile, LLMYError> {
        let setup = self
            .profiles
            .get(name)
            .ok_or_else(|| LLMYError::Other(eyre!("unknown profile '{name}'")))?;
        if setup.llm_debug.is_some() {
            tracing::warn!(
                "profile '{}': llm-debug is chain-level; set it at the top of the config \
                 (ignoring the profile-level value)",
                name
            );
        }
        // Profiles are self-contained: the config layer resolves them
        // env-isolated, so no ambient `OPENAI_API_KEY` & co. leaks into
        // every profile.
        let model = setup
            .resolved_model(true)
            .map_err(|error| LLMYError::Other(eyre!("profile '{name}': {error}")))?
            .ok_or_else(|| LLMYError::Other(eyre!("profile '{name}' needs a `model`")))?
            .with_full_id(setup.use_full_model_id);
        let config = setup
            .to_config(true)
            .map_err(|error| LLMYError::Other(eyre!("profile '{name}': {error}")))?;
        let mut settings = setup.settings();
        // The chain-level concurrency default applies only where the profile
        // says nothing itself (override relation, unlike the stacking caps).
        if setup.llm_concurrent.is_none()
            && let Some(concurrent) = self.llm_concurrent
        {
            settings.llm_concurrent = concurrent;
        }
        Ok(LLMProfile {
            name: name.to_string(),
            config,
            model,
            settings,
            cap: setup.biling_cap,
        })
    }

    /// Build the fallback-chain LLM this config describes. `cli_override`
    /// (from `--llm-profiles`) picks and orders a subset of the profiles.
    pub async fn to_llm(&self, cli_override: Option<&[String]>) -> Result<LLM, LLMYError> {
        let names = self.selection(cli_override)?;
        let profiles = names
            .iter()
            .map(|name| self.build_profile(name))
            .collect::<Result<Vec<_>, _>>()?;
        LLM::new_with_fallback_async(
            profiles,
            self.billing_cap.unwrap_or(rust_decimal::dec!(10.0)),
            // The checked conversion refuses NaN/negative/absurd values, all
            // of which fall back to the default instead of panicking.
            self.fallback_cooldown_secs
                .and_then(|secs| Duration::try_from_secs_f64(secs).ok())
                .unwrap_or(DEFAULT_FALLBACK_COOLDOWN),
            Some(String::new()),
            self.llm_debug.clone(),
        )
        .await
    }
}

/// The CLI flags of the TOML entry point, meant to sit beside an
/// [`OpenAISetup`] in a binary: when `--llmy-config`/`LLMY_CONFIG` is given
/// the file wins, otherwise the flag/env setup serves as before.
#[derive(Args, Clone, Debug)]
pub struct LLMYConfigSetup {
    /// Path to the llmy TOML config file with LLM profiles.
    #[arg(long = "llmy-config", env = "LLMY_CONFIG")]
    pub llmy_config: Option<PathBuf>,

    /// Profile names to use in fallback order (comma-separated), overriding
    /// the config file's `fallback` list.
    #[arg(long = "llm-profiles", value_delimiter = ',', requires = "llmy_config")]
    pub llm_profiles: Option<Vec<String>>,
}

impl LLMYConfigSetup {
    /// The fallback-chain LLM the config file describes, or `None` when no
    /// file was given.
    pub async fn may_llm(&self) -> Result<Option<LLM>, LLMYError> {
        let Some(path) = &self.llmy_config else {
            return Ok(None);
        };
        let config = LLMYConfig::load(path)?;
        Ok(Some(config.to_llm(self.llm_profiles.as_deref()).await?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_profile_config() -> LLMYConfig {
        LLMYConfig::parse_toml(
            r#"
            billing-cap = 20.0
            fallback = ["ali", "official"]

            [profiles.ali]
            model = "ali-model,1,2"
            openai-url = "http://localhost:0/v1"
            openai-key = "sk-ali"
            billing-cap = 5.0
            llm-retry = 3

            [profiles.official]
            model = "official-model,2,4"
            anthropic-url = "http://localhost:0/v1"
            anthropic-key = "sk-official"
            "#,
        )
        .expect("config")
    }

    #[test]
    fn a_toml_profile_gets_the_cli_defaults() {
        let setup: OpenAISetup = toml::from_str(r#"model = "captest,1,2""#).expect("profile");
        let settings = setup.settings();
        assert_eq!(
            settings.llm_prompt_timeout,
            crate::DEFAULT_LLM_PROMPT_TIMEOUT
        );
        assert_eq!(settings.llm_retry, crate::DEFAULT_LLM_RETRY);
        assert_eq!(
            settings.tool_reject_retries,
            crate::DEFAULT_LLM_TOOL_REJECT_RETRIES
        );
        assert!(settings.unknown_tool_hard_reject);
        assert!(settings.auto_strip);
        assert!(settings.auto_cache_key);
        assert_eq!(
            settings.billing_log_tokens,
            crate::DEFAULT_LLM_BILLING_LOG_TOKENS
        );
        assert_eq!(
            settings.cache_key_ttl,
            llmy_client::cache_key::DEFAULT_TTL_SECS
        );
        assert_eq!(
            settings.cache_key_rpm,
            llmy_client::cache_key::DEFAULT_MAX_RPM
        );
        assert!(!settings.allow_implicit_convert);
        assert!(!settings.llm_stream);
    }

    #[test]
    fn config_profiles_are_env_isolated() {
        // SAFETY: the mutated names are read only by env-resolution code,
        // which the config layer must switch off for its profiles — and the
        // guard below serializes with any other env-touching test.
        let _env = crate::tests::env_lock();
        unsafe {
            std::env::set_var("OPENAI_API_MODEL", "env-model,1,1");
        }
        let config = LLMYConfig::parse_toml(
            r#"
            [profiles.only]
            openai-url = "http://localhost:0/v1"
            "#,
        )
        .expect("config");
        // Without isolation the env alias would supply the model; the config
        // layer must refuse the profile instead.
        let err = config
            .build_profile("only")
            .expect_err("the model must not come from env");
        assert!(err.to_string().contains("needs a `model`"), "{err}");
        unsafe {
            std::env::remove_var("OPENAI_API_MODEL");
        }
    }

    #[test]
    fn unknown_profile_keys_are_refused() {
        let err = LLMYConfig::parse_toml(
            r#"
            [profiles.a]
            model = "m,1,2"
            openai-urk = "typo"
            "#,
        )
        .expect_err("typo must be rejected")
        .to_string();
        assert!(err.contains("openai-urk"), "{err}");
    }

    #[test]
    fn selection_honors_fallback_order_override_and_validation() {
        let config = two_profile_config();
        assert_eq!(config.selection(None).expect("order"), ["ali", "official"]);

        // The CLI override picks and reorders a subset.
        let onlyone = ["official".to_string()];
        assert_eq!(
            config.selection(Some(&onlyone)).expect("order"),
            ["official"]
        );

        // Unknown names and duplicates are refused.
        let unknown = ["nope".to_string()];
        assert!(config.selection(Some(&unknown)).is_err());
        let dup = ["ali".to_string(), "ali".to_string()];
        assert!(config.selection(Some(&dup)).is_err());

        // Several profiles with no order at all is ambiguous.
        let mut orderless = config.clone();
        orderless.fallback = None;
        assert!(orderless.selection(None).is_err());

        // A single profile needs no order.
        let single = LLMYConfig::parse_toml(
            r#"
            [profiles.only]
            model = "m,1,2"
            "#,
        )
        .expect("config");
        assert_eq!(single.selection(None).expect("order"), ["only"]);
    }

    #[tokio::test]
    async fn to_llm_builds_the_chain_in_order_with_caps() {
        let config = two_profile_config();
        let llm = config.to_llm(None).await.expect("llm");

        let names: Vec<&str> = llm.targets.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["ali", "official"]);
        assert_eq!(llm.billing_snapshot().cap, rust_decimal::dec!(20.0));
        // The primary profile's own knobs made it into the target settings.
        assert_eq!(llm.targets[0].settings.llm_retry, 3);
        assert_eq!(llm.targets[0].settings.llm_retry, 3);
        // The second profile picked the anthropic protocol.
        assert_eq!(llm.targets[1].client.protocol(), "anthropic");
    }

    #[test]
    fn chain_level_concurrency_is_a_default_each_profile_can_override() {
        let config = LLMYConfig::parse_toml(
            r#"
            llm-concurrent = 4
            fallback = ["inherits", "overrides"]

            [profiles.inherits]
            model = "m,1,2"

            [profiles.overrides]
            model = "m,1,2"
            llm-concurrent = 0
            "#,
        )
        .expect("config");
        let inherits = config.build_profile("inherits").expect("profile");
        assert_eq!(inherits.settings.llm_concurrent, 4);
        let overrides = config.build_profile("overrides").expect("profile");
        assert_eq!(overrides.settings.llm_concurrent, 0);
    }

    #[test]
    fn a_profile_without_a_model_is_refused_by_name() {
        let config = LLMYConfig::parse_toml(
            r#"
            [profiles.nomodel]
            openai-url = "http://localhost:0/v1"
            "#,
        )
        .expect("config");
        let err = config.build_profile("nomodel").expect_err("no model");
        assert!(err.to_string().contains("nomodel"), "{err}");
    }
}
