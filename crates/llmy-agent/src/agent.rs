//! Outcome of one step of an agent loop.
//!
//! A "step" is a single round-trip with the model: the caller sends the
//! current conversation, the model responds, and the response either ends
//! the turn ([`StepResult::Stop`]) or asks for tools to be executed
//! ([`StepResult::Toolcalled`]). Higher-level loops (see `llmy-harness`)
//! drive these steps until a `Stop` is observed.

use serde::{Deserialize, Serialize};

/// The result of running one agent step.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
