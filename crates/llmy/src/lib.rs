//! High-level umbrella crate for building LLM-driven agents with tools, memory, and tokenization.
//!
//! This crate re-exports the main `llmy-*` crates behind a single top-level API so downstream
//! users can build an agent without having to depend on each sub-crate individually.
//!
//! # Building An Agent
//!
//! The smallest useful agent needs three pieces:
//!
//! 1. A system prompt.
//! 2. A [`agent::tool::ToolBox`] containing zero or more tools.
//! 3. A [`harness::Agent`] to hold conversation state and orchestrate tool calls.
//!
//! ```no_run
//! use llmy::agent::tool::ToolBox;
//! use llmy::agent::tools::files::ReadFileTool;
//! use llmy::harness::Agent;
//!
//! let mut tools = ToolBox::new();
//! tools.add_tool(ReadFileTool::new(std::env::current_dir().unwrap()));
//!
//! let agent = Agent::new(
//!     "You are a helpful assistant.".to_string(),
//!     tools,
//!     "docs-example".to_string(),
//! );
//!
//! let _ = agent;
//! ```
//!
//! Once the agent exists, you typically:
//!
//! 1. Create an [`client::client::LLM`] from CLI-style configuration in [`clap`] or directly from
//!    [`client`] primitives.
//! 2. Push user input with [`harness::Agent::step_with_user`].
//! 3. Continue stepping while the agent is still issuing tool calls.
//!
//! # Memory-Enabled Agents
//!
//! If you want the agent to search and update structured memory, construct an
//! [`agent::tools::memory::AgentMemoryContext`] and then build the agent with
//! [`harness::Agent::with_memory`].
//!
//! ```no_run
//! use llmy::agent::tool::ToolBox;
//! use llmy::agent::tools::memory::{
//!     AgentMemory,
//!     AgentMemoryContext,
//!     embed::{SimilarityModel, SimilarityModelConfig},
//! };
//! use llmy::harness::{Agent, memory::AgentMemorySystemPromptCriteria};
//!
//! async fn build_agent() -> Result<Agent, llmy::LLMYError> {
//!     let memory = AgentMemoryContext::new(
//!         AgentMemory::default(),
//!         SimilarityModel::new(SimilarityModelConfig::default()).await?,
//!     );
//!
//!     Ok(Agent::with_memory(
//!         "You are a helpful assistant.".to_string(),
//!         ToolBox::new(),
//!         "docs-memory-example".to_string(),
//!         &memory,
//!         &AgentMemorySystemPromptCriteria::default(),
//!     )
//!     .await)
//! }
//! ```
//!
//! # Module Guide
//!
//! - [`clap`] contains CLI-oriented configuration helpers that can build an LLM client from flags
//!   and environment variables.
//! - [`client`] contains the lower-level LLM client, billing, settings, debug, and model modules.
//! - [`agent`] contains the core tool traits and the aggregated tool modules used by agents.
//! - [`ebmed`] re-exports the embedding helpers used by memory search and similarity matching.
//! - [`harness`] contains the concrete in-memory agent implementation.
//! - [`tokenizer`] contains model metadata and token counting helpers.
//! - [`openai`] re-exports `async-openai` for callers that need direct access to request and
//!   response types.

/// Command-line and environment-driven LLM configuration helpers.
pub mod clap {
    pub use llmy_clap::*;
}

/// Lower-level client, model, billing, debug, and settings modules used to talk to LLM backends.
pub mod client {
    pub use llmy_client::*;
}

/// Core agent traits plus the bundled tool modules used by `llmy` agents.
pub mod agent {
    /// Tool implementations that can be installed into an [`super::agent::tool::ToolBox`].
    pub mod tools {
        pub use llmy_agent_tools::*;
    }
    pub use llmy_agent::*;
}

/// Embedding and similarity helpers used by memory search, token counting, and input truncation.
pub mod ebmed {
    pub use llmy_agent_tools::memory::embed::*;
}

/// Concrete in-memory agent harness with context management, compaction, and optional memory.
pub mod harness {
    pub use llmy_harness::*;
}

/// Tokenizer helpers and model metadata for approximate token counting and context sizing.
pub mod tokenizer {
    pub use llmy_tokenizer::*;
}

/// Common llmy error type shared across agent, tool, and client layers.
pub use llmy_types::error::LLMYError;

/// Raw `async-openai` re-export for callers that need direct protocol-level types.
pub mod openai {
    pub use async_openai::*;
}
