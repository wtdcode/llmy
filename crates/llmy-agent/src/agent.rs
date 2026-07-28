//! Outcome of one step of an agent loop.
//!
//! A "step" is a single round-trip with the model: the caller sends the
//! current conversation, the model responds, and the response either ends
//! the turn ([`StepResult::Stop`]) or asks for tools to be executed
//! ([`StepResult::Toolcalled`]). Higher-level loops (see `llmy-harness`)
//! drive these steps until a `Stop` is observed.

use llmy_types::error::GeneralToolCall;
use serde::{Deserialize, Serialize};

/// The result of running one agent step.
#[derive(Debug, Clone)]
pub enum StepResult {
    /// The model produced a final assistant message and did not request any
    /// further tool calls. The wrapped string is that assistant message.
    Stop(String),
    /// The model issued one or more tool calls. The wrapped value is the
    /// optional assistant text emitted alongside the tool calls (some
    /// providers include reasoning or commentary in this field, others
    /// leave it empty, hence the `Option`).
    Toolcalled(Option<String>),
}

impl StepResult {
    /// Returns the assistant text associated with this step, if any.
    ///
    /// For [`StepResult::Stop`] this is always the final message; for
    /// [`StepResult::Toolcalled`] it is the optional text the model attached
    /// to the tool-call response.
    pub fn assistant_message(&self) -> Option<&String> {
        match self {
            Self::Stop(v) => Some(v),
            Self::Toolcalled(v) => v.as_ref(),
        }
    }

    /// Returns `true` if the model requested tool calls in this step.
    pub fn did_tool_call(&self) -> bool {
        matches!(self, Self::Toolcalled(_))
    }

    /// Returns `true` if the model finished its turn with a final message.
    pub fn did_stop(&self) -> bool {
        matches!(self, Self::Stop(_))
    }
}

/// An observable event emitted while a step executes the model's tool calls.
///
/// A single step can run several tool calls. [`StepResult`] plus the messages
/// appended to the conversation are enough to *drive* the loop, but they hide
/// *what happened* to each individual tool — information a host application
/// often wants to observe (logging, UI, telemetry). The harness'
/// `step_with_events` returns these events alongside the [`StepResult`], one
/// per tool call in call order, so failures are visible even when a tool
/// aborts the step.
///
/// Every variant carries the originating [`GeneralToolCall`] so events can be
/// correlated back to the exact request the model issued. The type is
/// [`Serialize`]/[`Deserialize`] so events can be forwarded out of process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentEvent {
    /// A tool ran to completion; `output` is the text returned to the model.
    ToolCallCompleted {
        call: GeneralToolCall,
        output: String,
    },
    /// A tool ran but failed. This is a *hard* failure: the step aborts and
    /// returns `Err` — but only after every outcome has been recorded, so a
    /// single step can emit several of these. The step's `Err` is the first one
    /// in call order; for any others `error` (the rendered error) is all that
    /// survives.
    ToolCallFailed {
        call: GeneralToolCall,
        error: String,
    },
    /// The model's arguments did not conform to the tool's schema. The step
    /// asks the model to retry instead of aborting; `error` describes the
    /// mismatch.
    ToolCallInvalidArguments {
        call: GeneralToolCall,
        error: String,
    },
    /// The model requested a tool that is not registered in the toolbox.
    ToolCallNotFound { call: GeneralToolCall },
}

/// The events emitted during a single step, in tool-call order.
pub type AgentEvents = Vec<AgentEvent>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_event_round_trips_through_json() {
        let call = GeneralToolCall {
            tool_id: "id-1".to_string(),
            tool_name: "alpha_tool".to_string(),
            tool_args: "{}".to_string(),
        };
        let event = AgentEvent::ToolCallCompleted {
            call: call.clone(),
            output: "hello".to_string(),
        };

        let json = serde_json::to_string(&event).unwrap();
        let restored: AgentEvent = serde_json::from_str(&json).unwrap();

        match restored {
            AgentEvent::ToolCallCompleted { call: got, output } => {
                assert_eq!(got.tool_id, call.tool_id);
                assert_eq!(got.tool_name, call.tool_name);
                assert_eq!(output, "hello");
            }
            other => panic!("expected ToolCallCompleted, got {other:?}"),
        }
    }
}
