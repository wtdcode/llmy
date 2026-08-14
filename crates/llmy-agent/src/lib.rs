//! Building blocks for tool-using LLM agents.
//!
//! `llmy-agent` provides the small, opinionated primitives that higher-level
//! crates (such as `llmy-harness`) compose into a complete agent loop:
//!
//! * [`mod@tool`] — the [`Tool`] / [`ToolDyn`] traits and the
//!   [`tool::ToolBox`] registry, used to expose Rust functions to a model and
//!   to dispatch tool calls back to them.
//! * [`agent`] — the [`StepResult`] type that summarises a single turn of an
//!   agent loop (the model either stopped with a final assistant message or
//!   issued tool calls that need to be executed before the next step).
//!
//! # Defining a tool
//!
//! Implementing [`Tool`] by hand is straightforward, but the
//! [`llmy_agent_derive::tool`] attribute macro (re-exported here as
//! [`tool`](macro@tool), and also reachable through the umbrella crate as
//! `llmy::agent::tool`) generates the boilerplate for you:
//!
//! ```ignore
//! use llmy_agent::{LLMYError, tool};
//! use schemars::JsonSchema;
//! use serde::Deserialize;
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct AddArgs { a: i64, b: i64 }
//!
//! #[derive(Clone, Debug)]
//! #[tool(
//!     description = "Add two integers",
//!     arguments = AddArgs,
//!     invoke = run,
//! )]
//! struct AddTool;
//!
//! impl AddTool {
//!     async fn run(&self, args: AddArgs) -> Result<String, LLMYError> {
//!         Ok((args.a + args.b).to_string())
//!     }
//! }
//! ```
//!
//! Tools can then be collected into a [`tool::ToolBox`], rendered as OpenAI
//! tool descriptors with [`tool::ToolBox::openai_objects`], and dispatched
//! against incoming `tool_calls` with the various `invoke_*` methods.
//!
//! # Re-exports
//!
//! For convenience the most commonly used items are re-exported at the crate
//! root: [`StepResult`], [`Tool`], [`ToolDyn`], [`LLMYError`] and the
//! [`tool`](macro@tool) attribute macro.

pub mod agent;
/// MCP server support — serve a [`tool::ToolBox`] over stdio or HTTP.
pub mod mcp;
pub mod tool;

pub use agent::{AgentEvent, AgentEvents, StepResult};
pub use llmy_agent_derive::tool;
pub use llmy_types::error::LLMYError;
pub use tool::{Tool, ToolDyn};

/// Re-export of the [`rmcp`] crate for MCP protocol types.
pub mod rcmp {
    pub use rmcp::*;
}
