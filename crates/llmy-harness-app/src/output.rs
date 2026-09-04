//! Structured output: the caller hands the harness a JSON Schema, and the
//! run only ends once a conforming result exists. Two submission paths feed
//! the same sink: the `submit_result` tool (schema advertised as the tool's
//! parameters, validation errors fed back as the tool result), and a
//! fallback that parses the model's final message when it stops without
//! submitting.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};

use color_eyre::eyre::eyre;
use llmy_agent::tool::ToolDyn;
use llmy_types::error::LLMYError;
use serde_json::Value;

/// Shared structured-output state: the compiled schema and the accepted
/// result, if any.
#[derive(Clone)]
pub struct StructuredOutputState {
    schema: Value,
    advertised: schemars::Schema,
    wrapped: bool,
    validator: Arc<jsonschema::Validator>,
    sink: Arc<StdMutex<Option<Value>>>,
}

impl fmt::Debug for StructuredOutputState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StructuredOutputState")
            .field("wrapped", &self.wrapped)
            .finish()
    }
}

impl StructuredOutputState {
    pub fn new(schema: Value) -> Result<Self, LLMYError> {
        let validator = jsonschema::validator_for(&schema)
            .map_err(|e| eyre!("output schema does not compile: {}", e))?;

        // Tool parameters must be a JSON object; a schema of any other shape
        // is advertised wrapped in {"result": ...} and unwrapped on submit.
        let is_object_schema = schema
            .get("type")
            .and_then(|t| t.as_str())
            .map(|t| t == "object")
            .unwrap_or_else(|| schema.get("properties").is_some());
        let (advertised_value, wrapped) = if is_object_schema {
            (schema.clone(), false)
        } else {
            (
                serde_json::json!({
                    "type": "object",
                    "properties": {"result": schema.clone()},
                    "required": ["result"],
                    "additionalProperties": false,
                }),
                true,
            )
        };
        let advertised = serde_json::from_value::<schemars::Schema>(advertised_value)
            .map_err(|e| eyre!("output schema is not a valid JSON Schema document: {}", e))?;

        Ok(Self {
            schema,
            advertised,
            wrapped,
            validator: Arc::new(validator),
            sink: Arc::new(StdMutex::new(None)),
        })
    }

    pub fn schema(&self) -> &Value {
        &self.schema
    }

    pub fn wrapped(&self) -> bool {
        self.wrapped
    }

    /// The accepted result, once one exists.
    pub fn accepted(&self) -> Option<Value> {
        match self.sink.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Validate `candidate` against the schema; on success it becomes the
    /// accepted result. Errors are model-facing explanations.
    pub fn try_accept_value(&self, candidate: Value) -> Result<(), Vec<String>> {
        let errors: Vec<String> = self
            .validator
            .iter_errors(&candidate)
            .map(|error| {
                format!(
                    "{} (at instance path {:?})",
                    error,
                    error.instance_path().to_string()
                )
            })
            .collect();
        if !errors.is_empty() {
            return Err(errors);
        }
        match self.sink.lock() {
            Ok(mut guard) => *guard = Some(candidate),
            Err(poisoned) => *poisoned.into_inner() = Some(candidate),
        }
        Ok(())
    }

    /// Try to interpret free text (the model's final message) as the result:
    /// the whole text as JSON, then any fenced code block, then the widest
    /// brace/bracket span.
    pub fn try_accept_text(&self, text: &str) -> Result<(), Vec<String>> {
        let mut last_errors = vec!["the final message contains no parseable JSON".to_string()];
        for candidate in Self::json_candidates(text) {
            match self.try_accept_value(candidate) {
                Ok(()) => return Ok(()),
                Err(errors) => last_errors = errors,
            }
        }
        Err(last_errors)
    }

    fn json_candidates(text: &str) -> Vec<Value> {
        let mut out = vec![];
        let trimmed = text.trim();
        if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            out.push(value);
        }

        // Fenced code blocks, with or without a language tag.
        let mut rest = trimmed;
        while let Some(start) = rest.find("```") {
            let after = &rest[start + 3..];
            let Some(end) = after.find("```") else { break };
            let block = &after[..end];
            let body = match block.find('\n') {
                Some(newline) if !block[..newline].trim().is_empty() => &block[newline + 1..],
                _ => block,
            };
            if let Ok(value) = serde_json::from_str::<Value>(body.trim()) {
                out.push(value);
            }
            rest = &after[end + 3..];
        }

        for (open, close) in [('{', '}'), ('[', ']')] {
            if let Some(first) = trimmed.find(open)
                && let Some(last) = trimmed.rfind(close)
                && first < last
                && let Ok(value) = serde_json::from_str::<Value>(&trimmed[first..=last])
            {
                out.push(value);
            }
        }
        out
    }
}

/// The `submit_result` tool. Implemented directly on [`ToolDyn`] because its
/// parameter schema is the caller-provided output schema, not a derived Rust
/// type.
#[derive(Clone)]
pub struct SubmitResultTool {
    state: StructuredOutputState,
}

impl fmt::Debug for SubmitResultTool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubmitResultTool").finish()
    }
}

impl SubmitResultTool {
    pub const NAME: &str = "submit_result";

    pub fn new(state: StructuredOutputState) -> Self {
        Self { state }
    }
}

impl ToolDyn for SubmitResultTool {
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> Option<String> {
        Some(
            "Submit the final result of this run. The arguments must conform to the advertised \
             JSON schema; validation errors come back as the tool result so you can fix and \
             resubmit. The run ends once a result is accepted."
                .to_string(),
        )
    }

    fn schema(&self) -> schemars::Schema {
        self.state.advertised.clone()
    }

    fn run(
        &self,
        arguments: Value,
    ) -> Pin<Box<dyn Future<Output = Result<String, LLMYError>> + Send + '_>> {
        Box::pin(async move {
            let candidate = if self.state.wrapped {
                match arguments.get("result") {
                    Some(inner) => inner.clone(),
                    None => {
                        return Ok(
                            "submit_result rejected: the arguments must be {\"result\": <value conforming to the schema>}"
                                .to_string(),
                        );
                    }
                }
            } else {
                arguments
            };
            match self.state.try_accept_value(candidate) {
                Ok(()) => Ok("Result accepted. The run will finish.".to_string()),
                Err(errors) => Ok(format!(
                    "submit_result rejected, the payload does not conform to the schema:\n- {}\nFix the payload and call submit_result again.",
                    errors.join("\n- ")
                )),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object_schema() -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "verdict": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["verdict", "score"],
            "additionalProperties": false,
        })
    }

    #[test]
    fn object_schemas_are_advertised_unwrapped() {
        let state = StructuredOutputState::new(object_schema()).expect("compile");
        assert!(!state.wrapped());
    }

    #[test]
    fn non_object_schemas_are_wrapped() {
        let state = StructuredOutputState::new(
            serde_json::json!({"type": "array", "items": {"type": "string"}}),
        )
        .expect("compile");
        assert!(state.wrapped());
    }

    #[test]
    fn validation_gates_the_sink() {
        let state = StructuredOutputState::new(object_schema()).expect("compile");
        let errors = state
            .try_accept_value(serde_json::json!({"verdict": "ok"}))
            .expect_err("missing score");
        assert!(errors.iter().any(|e| e.contains("score")));
        assert!(state.accepted().is_none());

        state
            .try_accept_value(serde_json::json!({"verdict": "ok", "score": 5}))
            .expect("valid");
        assert_eq!(state.accepted().expect("accepted")["score"], 5);
    }

    #[test]
    fn text_parsing_tries_fences_and_brace_spans() {
        let state = StructuredOutputState::new(object_schema()).expect("compile");
        state
            .try_accept_text(
                "Here is the result:\n```json\n{\"verdict\": \"ok\", \"score\": 3}\n```\nDone.",
            )
            .expect("fenced json");
        assert_eq!(state.accepted().expect("accepted")["score"], 3);

        let state = StructuredOutputState::new(object_schema()).expect("compile");
        state
            .try_accept_text("prefix {\"verdict\": \"ok\", \"score\": 4} suffix")
            .expect("brace span");
        assert_eq!(state.accepted().expect("accepted")["score"], 4);

        let state = StructuredOutputState::new(object_schema()).expect("compile");
        assert!(state.try_accept_text("no json here").is_err());
    }
}
