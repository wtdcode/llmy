//! The llmy harness application: a single-run, Claude-Code-style agent
//! harness with SQLite-persisted state, truncated-but-recoverable tool
//! results, background bash and monitors, a JSON scratchpad, structured
//! output gating, and a knowledge-graph memory shared across runs.

pub mod files;
pub mod kg;
pub mod output;
pub mod prompts;
pub mod runner;
pub mod scratchpad;
pub mod state;
pub mod tasks;

pub use prompts::SystemPromptTemplate;
pub use runner::{HarnessOptions, HarnessOutcome, HarnessRunStatus, HarnessRunner, MemoryOptions};
pub use state::{HarnessStateDB, ToolResultPolicy};
pub use tasks::TaskConfig;
