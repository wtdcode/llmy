//! Language-neutral code graph model shared by every extractor: containers
//! (contracts / modules), callables (functions / modifiers / entries), state
//! items (storage variables / accounts / resources / objects) and the edges
//! between them. Call edges resolved purely from syntax keep their ambiguity
//! explicit instead of pretending precision the parser does not have.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use color_eyre::eyre::eyre;
use llmy_types::error::LLMYError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    Solidity,
    Rust,
    MoveAptos,
    MoveSui,
}

impl Language {
    pub fn render(&self) -> &'static str {
        match self {
            Self::Solidity => "Solidity",
            Self::Rust => "Rust",
            Self::MoveAptos => "Move (Aptos)",
            Self::MoveSui => "Move (Sui)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModuleKind {
    Contract,
    Interface,
    Library,
    /// A Rust or Move module.
    Module,
}

impl ModuleKind {
    pub fn render(&self) -> &'static str {
        match self {
            Self::Contract => "contract",
            Self::Interface => "interface",
            Self::Library => "library",
            Self::Module => "module",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CallableKind {
    Function,
    Constructor,
    Modifier,
    Fallback,
    Receive,
    /// A Move `entry` function or an Anchor instruction handler — the
    /// externally reachable surface.
    Entry,
}

impl CallableKind {
    pub fn render(&self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Constructor => "constructor",
            Self::Modifier => "modifier",
            Self::Fallback => "fallback",
            Self::Receive => "receive",
            Self::Entry => "entry",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateKind {
    /// A Solidity storage variable.
    StateVariable,
    /// An Anchor `#[account]` data struct.
    AnchorAccount,
    /// A CosmWasm `Item` storage slot.
    CwItem,
    /// A CosmWasm `Map` storage collection.
    CwMap,
    /// An Aptos global resource (struct with `key`).
    MoveResource,
    /// A Sui object (struct with `key`).
    SuiObject,
}

impl StateKind {
    pub fn render(&self) -> &'static str {
        match self {
            Self::StateVariable => "state variable",
            Self::AnchorAccount => "anchor account",
            Self::CwItem => "cw item",
            Self::CwMap => "cw map",
            Self::MoveResource => "move resource",
            Self::SuiObject => "sui object",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessKind {
    Read,
    Write,
}

impl AccessKind {
    pub fn render(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
        }
    }
}

/// 1-based inclusive line span inside a source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineSpan {
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    pub id: i64,
    pub language: Language,
    pub kind: ModuleKind,
    pub name: String,
    /// Path relative to the indexed root.
    pub file: PathBuf,
    pub span: LineSpan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Callable {
    pub id: i64,
    pub module_id: i64,
    pub kind: CallableKind,
    pub name: String,
    /// Signature head as written in the source (cut before the body).
    pub signature: String,
    pub file: PathBuf,
    pub span: LineSpan,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateItem {
    pub id: i64,
    pub module_id: i64,
    pub kind: StateKind,
    pub name: String,
    /// Declared type, verbatim.
    pub type_text: String,
    pub file: PathBuf,
    pub span: LineSpan,
}

/// A syntactically resolved callee. Name-based resolution over tree-sitter
/// cannot always pick a single target, so ambiguity is first-class.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalleeRef {
    Resolved(i64),
    Ambiguous(Vec<i64>),
    /// Not defined inside the indexed project (external library, builtin, or
    /// an interface without an in-project implementation).
    External(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallEdge {
    pub caller_id: i64,
    pub callee: CalleeRef,
    /// The call site as written (e.g. `token.transfer`), for display.
    pub callee_text: String,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEdge {
    pub callable_id: i64,
    pub state_id: i64,
    pub access: AccessKind,
    pub line: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParentRef {
    Resolved(i64),
    External(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InheritEdge {
    pub module_id: i64,
    pub parent: ParentRef,
}

/// The complete index over one project root.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CodeGraph {
    pub modules: BTreeMap<i64, Module>,
    pub callables: BTreeMap<i64, Callable>,
    pub states: BTreeMap<i64, StateItem>,
    pub call_edges: Vec<CallEdge>,
    pub state_edges: Vec<StateEdge>,
    pub inherit_edges: Vec<InheritEdge>,
}

impl CodeGraph {
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn modules_by_name(&self, name: &str) -> Vec<&Module> {
        self.modules.values().filter(|m| m.name == name).collect()
    }

    pub fn callables_of_module(&self, module_id: i64) -> Vec<&Callable> {
        self.callables
            .values()
            .filter(|c| c.module_id == module_id)
            .collect()
    }

    pub fn states_of_module(&self, module_id: i64) -> Vec<&StateItem> {
        self.states
            .values()
            .filter(|s| s.module_id == module_id)
            .collect()
    }

    /// Find callables by name, optionally scoped to a module name.
    pub fn find_callables(&self, module: Option<&str>, name: &str) -> Vec<&Callable> {
        self.callables
            .values()
            .filter(|c| c.name == name)
            .filter(|c| match module {
                Some(module) => self
                    .modules
                    .get(&c.module_id)
                    .map(|m| m.name == module)
                    .unwrap_or(false),
                None => true,
            })
            .collect()
    }

    /// Find state items by name, optionally scoped to a module name.
    pub fn find_states(&self, module: Option<&str>, name: &str) -> Vec<&StateItem> {
        self.states
            .values()
            .filter(|s| s.name == name)
            .filter(|s| match module {
                Some(module) => self
                    .modules
                    .get(&s.module_id)
                    .map(|m| m.name == module)
                    .unwrap_or(false),
                None => true,
            })
            .collect()
    }

    pub fn outgoing_calls(&self, caller_id: i64) -> Vec<&CallEdge> {
        self.call_edges
            .iter()
            .filter(|e| e.caller_id == caller_id)
            .collect()
    }

    pub fn incoming_calls(&self, callee_id: i64) -> Vec<&CallEdge> {
        self.call_edges
            .iter()
            .filter(|e| match &e.callee {
                CalleeRef::Resolved(id) => *id == callee_id,
                CalleeRef::Ambiguous(ids) => ids.contains(&callee_id),
                CalleeRef::External(_) => false,
            })
            .collect()
    }

    pub fn state_accesses_of(&self, callable_id: i64) -> Vec<&StateEdge> {
        self.state_edges
            .iter()
            .filter(|e| e.callable_id == callable_id)
            .collect()
    }

    pub fn accessors_of_state(&self, state_id: i64) -> Vec<&StateEdge> {
        self.state_edges
            .iter()
            .filter(|e| e.state_id == state_id)
            .collect()
    }

    /// Transitive ancestors of a module through inheritance edges,
    /// cycle-safe.
    pub fn ancestors_of(&self, module_id: i64) -> BTreeSet<i64> {
        let mut out = BTreeSet::new();
        let mut frontier = vec![module_id];
        while let Some(current) = frontier.pop() {
            for edge in self.inherit_edges.iter().filter(|e| e.module_id == current) {
                if let ParentRef::Resolved(parent) = edge.parent
                    && out.insert(parent)
                {
                    frontier.push(parent);
                }
            }
        }
        out
    }

    /// Transitive descendants of a module through inheritance edges,
    /// cycle-safe.
    pub fn descendants_of(&self, module_id: i64) -> BTreeSet<i64> {
        let mut out = BTreeSet::new();
        let mut frontier = vec![module_id];
        while let Some(current) = frontier.pop() {
            for edge in self.inherit_edges.iter() {
                if edge.parent == ParentRef::Resolved(current) && out.insert(edge.module_id) {
                    frontier.push(edge.module_id);
                }
            }
        }
        out
    }

    /// State items visible to a module: its own plus everything declared by
    /// its ancestors.
    pub fn visible_states(&self, module_id: i64) -> Vec<&StateItem> {
        let mut scope = self.ancestors_of(module_id);
        scope.insert(module_id);
        self.states
            .values()
            .filter(|s| scope.contains(&s.module_id))
            .collect()
    }

    pub fn render_callee(&self, callee: &CalleeRef) -> String {
        match callee {
            CalleeRef::Resolved(id) => self.render_callable_ref(*id),
            CalleeRef::Ambiguous(ids) => {
                let rendered = ids
                    .iter()
                    .map(|id| self.render_callable_ref(*id))
                    .collect::<Vec<_>>()
                    .join(" | ");
                format!("ambiguous({rendered})")
            }
            CalleeRef::External(name) => format!("external({name})"),
        }
    }

    pub fn render_callable_ref(&self, callable_id: i64) -> String {
        match self.callables.get(&callable_id) {
            Some(callable) => {
                let module = self
                    .modules
                    .get(&callable.module_id)
                    .map(|m| m.name.as_str())
                    .unwrap_or("?");
                format!("{}.{}", module, callable.name)
            }
            None => format!("#{callable_id}"),
        }
    }

    /// Read the source lines of a span from disk, relative to `root`.
    pub async fn read_span(
        &self,
        root: &Path,
        file: &Path,
        span: LineSpan,
    ) -> Result<String, LLMYError> {
        let content = tokio::fs::read_to_string(root.join(file)).await?;
        let start = span.start_line.saturating_sub(1);
        let count = span.end_line.saturating_sub(start);
        let selected = content
            .lines()
            .skip(start)
            .take(count)
            .collect::<Vec<_>>()
            .join("\n");
        if selected.is_empty() {
            return Err(eyre!(
                "span {}..{} of {} is empty or out of range",
                span.start_line,
                span.end_line,
                file.display()
            )
            .into());
        }
        Ok(selected)
    }

    pub fn counts(&self) -> String {
        format!(
            "{} modules, {} callables, {} state items, {} call edges, {} state edges, {} inheritance edges",
            self.modules.len(),
            self.callables.len(),
            self.states.len(),
            self.call_edges.len(),
            self.state_edges.len(),
            self.inherit_edges.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph_with_inheritance() -> CodeGraph {
        let mut graph = CodeGraph::default();
        for (id, name) in [(1, "Base"), (2, "Mid"), (3, "Leaf")] {
            graph.modules.insert(
                id,
                Module {
                    id,
                    language: Language::Solidity,
                    kind: ModuleKind::Contract,
                    name: name.to_string(),
                    file: PathBuf::from("a.sol"),
                    span: LineSpan {
                        start_line: 1,
                        end_line: 10,
                    },
                },
            );
        }
        graph.inherit_edges.push(InheritEdge {
            module_id: 2,
            parent: ParentRef::Resolved(1),
        });
        graph.inherit_edges.push(InheritEdge {
            module_id: 3,
            parent: ParentRef::Resolved(2),
        });
        graph
    }

    #[test]
    fn ancestors_and_descendants_are_transitive() {
        let graph = graph_with_inheritance();
        assert_eq!(graph.ancestors_of(3), BTreeSet::from([1, 2]));
        assert_eq!(graph.descendants_of(1), BTreeSet::from([2, 3]));
        assert!(graph.ancestors_of(1).is_empty());
    }

    #[test]
    fn visible_states_include_inherited_ones() {
        let mut graph = graph_with_inheritance();
        graph.states.insert(
            1,
            StateItem {
                id: 1,
                module_id: 1,
                kind: StateKind::StateVariable,
                name: "owner".to_string(),
                type_text: "address".to_string(),
                file: PathBuf::from("a.sol"),
                span: LineSpan {
                    start_line: 2,
                    end_line: 2,
                },
            },
        );
        let visible = graph.visible_states(3);
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].name, "owner");
        assert!(graph.visible_states(1).len() == 1);
    }

    #[test]
    fn incoming_calls_match_resolved_and_ambiguous() {
        let mut graph = graph_with_inheritance();
        for (id, name) in [(10, "a"), (11, "b"), (12, "c")] {
            graph.callables.insert(
                id,
                Callable {
                    id,
                    module_id: 1,
                    kind: CallableKind::Function,
                    name: name.to_string(),
                    signature: format!("function {name}()"),
                    file: PathBuf::from("a.sol"),
                    span: LineSpan {
                        start_line: 3,
                        end_line: 5,
                    },
                },
            );
        }
        graph.call_edges.push(CallEdge {
            caller_id: 10,
            callee: CalleeRef::Resolved(11),
            callee_text: "b".to_string(),
            line: 4,
        });
        graph.call_edges.push(CallEdge {
            caller_id: 12,
            callee: CalleeRef::Ambiguous(vec![10, 11]),
            callee_text: "ab".to_string(),
            line: 4,
        });

        assert_eq!(graph.incoming_calls(11).len(), 2);
        assert_eq!(graph.incoming_calls(10).len(), 1);
        assert_eq!(graph.outgoing_calls(10).len(), 1);
    }
}
