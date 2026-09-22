//! The TOML-file entry point for configuring LLMs: named profiles (each one
//! the same data model as the CLI's flag set) plus the fallback-chain knobs.
//! This complements the env/flag entry point — same structs, second door.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

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
    ///
    /// `injected` is the name of the env/flag profile joining the chain (see
    /// [`crate::OpenAISetup`]): it is a valid name to reference, but never
    /// picked by the defaults — only the config's own profiles are.
    pub fn selection(
        &self,
        cli_override: Option<&[String]>,
        injected: Option<&str>,
    ) -> Result<Vec<String>, LLMYError> {
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
            if !self.profiles.contains_key(name) && Some(name.as_str()) != injected {
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
    /// (from `--llm-profiles`) picks and orders a subset of the profiles;
    /// `injected` is the env/flag setup joining the profile set (see
    /// [`crate::OpenAISetup`]) — referencable by its name, and winning over
    /// a config profile of the same name with a warning.
    pub async fn to_llm(
        &self,
        cli_override: Option<&[String]>,
        injected: Option<LLMProfile>,
    ) -> Result<LLM, LLMYError> {
        let names = self.selection(cli_override, injected.as_ref().map(|p| p.name.as_str()))?;
        let mut profiles = Vec::new();
        for name in &names {
            match injected.as_ref().filter(|profile| &profile.name == name) {
                Some(profile) => {
                    if self.profiles.contains_key(name) {
                        tracing::warn!(
                            "config profile '{}' is overridden by the env/flag setup \
                             injected under the same name",
                            name
                        );
                    }
                    profiles.push(profile.clone());
                }
                None => profiles.push(self.build_profile(name)?),
            }
        }
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
        assert_eq!(
            config.selection(None, None).expect("order"),
            ["ali", "official"]
        );

        // The CLI override picks and reorders a subset.
        let onlyone = ["official".to_string()];
        assert_eq!(
            config.selection(Some(&onlyone), None).expect("order"),
            ["official"]
        );

        // Unknown names and duplicates are refused.
        let unknown = ["nope".to_string()];
        assert!(config.selection(Some(&unknown), None).is_err());
        let dup = ["ali".to_string(), "ali".to_string()];
        assert!(config.selection(Some(&dup), None).is_err());

        // Several profiles with no order at all is ambiguous.
        let mut orderless = config.clone();
        orderless.fallback = None;
        assert!(orderless.selection(None, None).is_err());

        // A single profile needs no order.
        let single = LLMYConfig::parse_toml(
            r#"
            [profiles.only]
            model = "m,1,2"
            "#,
        )
        .expect("config");
        assert_eq!(single.selection(None, None).expect("order"), ["only"]);
    }

    #[tokio::test]
    async fn to_llm_builds_the_chain_in_order_with_caps() {
        let config = two_profile_config();
        let llm = config.to_llm(None, None).await.expect("llm");

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

    fn injected_profile(model: &str) -> LLMProfile {
        LLMProfile {
            name: llmy_client::client::DEFAULT_PROFILE_NAME.to_string(),
            config: llmy_client::client::SupportedConfig::new("http://localhost:0/v1", "sk-env"),
            model: model
                .parse::<llmy_client::model::OpenAIModel>()
                .expect("model"),
            settings: OpenAISetup::default().settings(),
            cap: None,
        }
    }

    fn target_names(llm: &LLM) -> Vec<&str> {
        llm.targets.iter().map(|t| t.name.as_str()).collect()
    }

    #[tokio::test]
    async fn the_env_setup_joins_the_chain_only_when_referenced() {
        let config = two_profile_config();

        // Not referenced: the chain is exactly the config's fallback list.
        let llm = config
            .to_llm(None, Some(injected_profile("env-model,1,1")))
            .await
            .expect("llm");
        assert_eq!(target_names(&llm), ["ali", "official"]);

        // Referenced by the CLI override: it joins in the given position.
        let order = ["official".to_string(), "default".to_string()];
        let llm = config
            .to_llm(Some(&order), Some(injected_profile("env-model,1,1")))
            .await
            .expect("llm");
        assert_eq!(target_names(&llm), ["official", "default"]);
        assert_eq!(llm.targets[1].model.model_name(), "env-model");

        // Without an injected profile the name is unknown.
        assert!(config.to_llm(Some(&order), None).await.is_err());
    }

    #[tokio::test]
    async fn an_injected_default_overrides_a_config_profile_of_that_name() {
        let config = LLMYConfig::parse_toml(
            r#"
            fallback = ["default"]

            [profiles.default]
            model = "config-model,1,1"
            openai-url = "http://localhost:0/v1"
            "#,
        )
        .expect("config");

        // The env/flag setup wins over the same-named config profile.
        let llm = config
            .to_llm(None, Some(injected_profile("env-model,1,1")))
            .await
            .expect("llm");
        assert_eq!(llm.targets[0].model.model_name(), "env-model");

        // Without it the config's own `default` serves untouched.
        let llm = config.to_llm(None, None).await.expect("llm");
        assert_eq!(llm.targets[0].model.model_name(), "config-model");
    }

    #[tokio::test]
    async fn the_llmy_config_flag_switches_the_setup_to_the_chain() {
        // Guards the env reads of the non-isolated `default_profile` path.
        let _env = crate::tests::env_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("llmy.toml");
        std::fs::write(
            &path,
            r#"
            fallback = ["a", "default"]

            [profiles.a]
            model = "a-model,1,1"
            openai-url = "http://localhost:0/v1"
            "#,
        )
        .expect("write config");

        // The opt-opt flavor, so real env vars cannot interfere.
        let setup = crate::OptOptOpenAISetup {
            model: Some("env-model,1,1".parse().expect("model")),
            openai_url: Some("http://localhost:0/v1".to_string()),
            llmy_config: Some(path),
            ..Default::default()
        };

        let llm = setup.may_llm().await.expect("llm").expect("chain");
        assert_eq!(target_names(&llm), ["a", "default"]);
        assert_eq!(llm.targets[1].model.model_name(), "env-model");
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
