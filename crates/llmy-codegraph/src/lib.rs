//! Tree-sitter based code indexing for smart contracts: call graphs and
//! state graphs over Solidity, Rust (Anchor / CosmWasm) and Move (Aptos /
//! Sui), plus agent-facing query tools.

pub mod builder;
pub mod extract;
pub mod model;
pub mod move_lang;
pub mod rust_lang;
pub mod solidity;
pub mod store;
pub mod tools;

pub use builder::{BuildResult, CodeGraphBuilder};
pub use model::CodeGraph;
pub use store::CodeGraphStore;
pub use tools::CodegraphContext;
