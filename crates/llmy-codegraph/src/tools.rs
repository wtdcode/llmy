//! Agent-facing query tools over a built [`CodeGraph`]: overview, per-module
//! and per-callable source reading, call-edge lookups and state
//! cross-references.

use std::path::PathBuf;
use std::sync::Arc;

use itertools::Itertools;
use llmy_agent::tool::ToolBox;
use llmy_types::error::LLMYError;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::model::{AccessKind, Callable, CodeGraph, Module};

const MAX_SOURCE_MATCHES: usize = 3;

/// Shared context for every codegraph tool.
#[derive(Debug, Clone)]
pub struct CodegraphContext {
    graph: Arc<CodeGraph>,
    root: PathBuf,
}

impl CodegraphContext {
    pub fn new(graph: CodeGraph, root: PathBuf) -> Self {
        Self {
            graph: Arc::new(graph),
            root,
        }
    }

    pub fn graph(&self) -> &CodeGraph {
        &self.graph
    }

    /// A short project map for the system prompt: counts plus the module
    /// list with per-module callable/state tallies.
    pub fn render_overview(&self) -> String {
        if self.graph.is_empty() {
            return "The code graph is empty (no supported source files were found).".to_string();
        }
        let mut lines = vec![format!("Indexed: {}.", self.graph.counts())];
        for module in self.graph.modules.values() {
            let callables = self.graph.callables_of_module(module.id);
            let states = self.graph.states_of_module(module.id);
            lines.push(format!(
                "- {} {} ({}, {}): {} callables, {} state items",
                module.kind.render(),
                module.name,
                module.language.render(),
                module.file.display(),
                callables.len(),
                states.len()
            ));
        }
        lines.join("\n")
    }

    pub fn tool_box(&self) -> ToolBox {
        let mut tools = ToolBox::new();
        tools.add_tool(CodegraphOverviewTool::new(self.clone()));
        tools.add_tool(ListCallablesTool::new(self.clone()));
        tools.add_tool(LookupCallableTool::new(self.clone()));
        tools.add_tool(LookupStateTool::new(self.clone()));
        tools.add_tool(ReadCallableSourceTool::new(self.clone()));
        tools.add_tool(ReadModuleSourceTool::new(self.clone()));
        tools
    }

    fn modules_named(&self, name: &str) -> Vec<&Module> {
        self.graph.modules_by_name(name)
    }

    fn render_callable_line(&self, callable: &Callable) -> String {
        let module = self
            .graph
            .modules
            .get(&callable.module_id)
            .map(|m| m.name.as_str())
            .unwrap_or("?");
        format!(
            "{}.{} [{}] ({}:{}..{})",
            module,
            callable.name,
            callable.kind.render(),
            callable.file.display(),
            callable.span.start_line,
            callable.span.end_line
        )
    }

    fn render_callable_details(&self, callable: &Callable) -> String {
        let mut sections = vec![
            self.render_callable_line(callable),
            format!("signature: {}", callable.signature),
        ];

        let outgoing = self.graph.outgoing_calls(callable.id);
        if outgoing.is_empty() {
            sections.push("outgoing calls: none".to_string());
        } else {
            let rendered = outgoing
                .iter()
                .map(|edge| {
                    format!(
                        "  - line {}: {} -> {}",
                        edge.line,
                        edge.callee_text,
                        self.graph.render_callee(&edge.callee)
                    )
                })
                .join("\n");
            sections.push(format!("outgoing calls:\n{rendered}"));
        }

        let incoming = self.graph.incoming_calls(callable.id);
        if incoming.is_empty() {
            sections.push("incoming calls: none (not called from indexed code)".to_string());
        } else {
            let rendered = incoming
                .iter()
                .map(|edge| {
                    format!(
                        "  - {} at line {}",
                        self.graph.render_callable_ref(edge.caller_id),
                        edge.line
                    )
                })
                .join("\n");
            sections.push(format!("incoming calls:\n{rendered}"));
        }

        let accesses = self.graph.state_accesses_of(callable.id);
        if accesses.is_empty() {
            sections.push("state accesses: none detected".to_string());
        } else {
            let rendered = accesses
                .iter()
                .map(|edge| {
                    let state = self
                        .graph
                        .states
                        .get(&edge.state_id)
                        .map(|s| s.name.as_str())
                        .unwrap_or("?");
                    format!(
                        "  - {} {} (line {})",
                        edge.access.render(),
                        state,
                        edge.line
                    )
                })
                .join("\n");
            sections.push(format!("state accesses:\n{rendered}"));
        }

        sections.join("\n")
    }
}

/// Arguments accepted by [`CodegraphOverviewTool`].
#[derive(Deserialize, JsonSchema)]
pub struct CodegraphOverviewArgs {}

/// Renders the module map.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = CodegraphOverviewArgs,
    invoke = overview,
    name = "codegraph_overview",
    description = "Show the code graph overview: indexed counts and every module/contract with its file, callable count and state item count.",
)]
pub struct CodegraphOverviewTool {
    context: CodegraphContext,
}

impl CodegraphOverviewTool {
    pub fn new(context: CodegraphContext) -> Self {
        Self { context }
    }

    async fn overview(&self, _args: CodegraphOverviewArgs) -> Result<String, LLMYError> {
        Ok(self.context.render_overview())
    }
}

/// Arguments accepted by [`ListCallablesTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ListCallablesArgs {
    /// Module / contract name to list.
    pub module: String,
}

/// Lists a module's callables and state items.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ListCallablesArgs,
    invoke = list,
    name = "list_callables",
    description = "List every callable (with kind and signature) and state item of one module/contract.",
)]
pub struct ListCallablesTool {
    context: CodegraphContext,
}

impl ListCallablesTool {
    pub fn new(context: CodegraphContext) -> Self {
        Self { context }
    }

    async fn list(&self, args: ListCallablesArgs) -> Result<String, LLMYError> {
        let modules = self.context.modules_named(&args.module);
        if modules.is_empty() {
            return Ok(format!(
                "No module named {:?} in the code graph; use codegraph_overview for the module list.",
                args.module
            ));
        }
        let mut sections = vec![];
        for module in modules {
            let mut lines = vec![format!(
                "{} {} ({}, {})",
                module.kind.render(),
                module.name,
                module.language.render(),
                module.file.display()
            )];
            let states = self.context.graph.states_of_module(module.id);
            if !states.is_empty() {
                lines.push("state items:".to_string());
                for state in states {
                    lines.push(format!(
                        "  - {} [{}]: {} (line {})",
                        state.name,
                        state.kind.render(),
                        state.type_text,
                        state.span.start_line
                    ));
                }
            }
            lines.push("callables:".to_string());
            for callable in self.context.graph.callables_of_module(module.id) {
                lines.push(format!(
                    "  - [{}] {} (lines {}..{})",
                    callable.kind.render(),
                    callable.signature,
                    callable.span.start_line,
                    callable.span.end_line
                ));
            }
            sections.push(lines.join("\n"));
        }
        Ok(sections.join("\n\n"))
    }
}

/// Arguments accepted by [`LookupCallableTool`].
#[derive(Deserialize, JsonSchema)]
pub struct LookupCallableArgs {
    /// Function / callable name to look up.
    pub name: String,
    /// Optional module/contract name to disambiguate.
    #[serde(default)]
    pub module: Option<String>,
}

/// Call edges and state accesses of a callable.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = LookupCallableArgs,
    invoke = lookup,
    name = "lookup_callable",
    description = "Look up a function/callable by name (optionally scoped to a module): its signature, outgoing calls, incoming callers, and the state items it reads/writes. Edges marked ambiguous/external are syntactic guesses — confirm in source.",
)]
pub struct LookupCallableTool {
    context: CodegraphContext,
}

impl LookupCallableTool {
    pub fn new(context: CodegraphContext) -> Self {
        Self { context }
    }

    async fn lookup(&self, args: LookupCallableArgs) -> Result<String, LLMYError> {
        let matches = self
            .context
            .graph
            .find_callables(args.module.as_deref(), &args.name);
        if matches.is_empty() {
            return Ok(format!(
                "No callable named {:?}{} in the code graph.",
                args.name,
                args.module
                    .as_deref()
                    .map(|m| format!(" in module {m:?}"))
                    .unwrap_or_default()
            ));
        }
        Ok(matches
            .iter()
            .map(|callable| self.context.render_callable_details(callable))
            .join("\n\n---\n\n"))
    }
}

/// Arguments accepted by [`LookupStateTool`].
#[derive(Deserialize, JsonSchema)]
pub struct LookupStateArgs {
    /// State item name (state variable, account struct, resource/object).
    pub name: String,
    /// Optional module/contract name to disambiguate.
    #[serde(default)]
    pub module: Option<String>,
}

/// Readers and writers of a state item.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = LookupStateArgs,
    invoke = lookup,
    name = "lookup_state",
    description = "Look up a state item (Solidity state variable, Anchor account, CosmWasm Item/Map, Move resource/object) by name: where it is declared and every callable that reads or writes it.",
)]
pub struct LookupStateTool {
    context: CodegraphContext,
}

impl LookupStateTool {
    pub fn new(context: CodegraphContext) -> Self {
        Self { context }
    }

    async fn lookup(&self, args: LookupStateArgs) -> Result<String, LLMYError> {
        let matches = self
            .context
            .graph
            .find_states(args.module.as_deref(), &args.name);
        if matches.is_empty() {
            return Ok(format!(
                "No state item named {:?} in the code graph.",
                args.name
            ));
        }
        let mut sections = vec![];
        for state in matches {
            let module = self
                .context
                .graph
                .modules
                .get(&state.module_id)
                .map(|m| m.name.as_str())
                .unwrap_or("?");
            let mut lines = vec![format!(
                "{} [{}] declared in {} ({}:{})\ntype: {}",
                state.name,
                state.kind.render(),
                module,
                state.file.display(),
                state.span.start_line,
                state.type_text
            )];
            let accessors = self.context.graph.accessors_of_state(state.id);
            let mut writers = vec![];
            let mut readers = vec![];
            for edge in accessors {
                let rendered = format!(
                    "  - {} (line {})",
                    self.context.graph.render_callable_ref(edge.callable_id),
                    edge.line
                );
                match edge.access {
                    AccessKind::Write => writers.push(rendered),
                    AccessKind::Read => readers.push(rendered),
                }
            }
            lines.push(if writers.is_empty() {
                "writers: none detected".to_string()
            } else {
                format!("writers:\n{}", writers.join("\n"))
            });
            lines.push(if readers.is_empty() {
                "readers: none detected".to_string()
            } else {
                format!("readers:\n{}", readers.join("\n"))
            });
            sections.push(lines.join("\n"));
        }
        Ok(sections.join("\n\n---\n\n"))
    }
}

/// Arguments accepted by [`ReadCallableSourceTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ReadCallableSourceArgs {
    /// Function / callable name.
    pub name: String,
    /// Optional module/contract name to disambiguate.
    #[serde(default)]
    pub module: Option<String>,
}

/// Reads a callable's source by name.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ReadCallableSourceArgs,
    invoke = read,
    name = "read_callable_source",
    description = "Read the full source of a function/callable by name (optionally scoped to a module/contract).",
)]
pub struct ReadCallableSourceTool {
    context: CodegraphContext,
}

impl ReadCallableSourceTool {
    pub fn new(context: CodegraphContext) -> Self {
        Self { context }
    }

    async fn read(&self, args: ReadCallableSourceArgs) -> Result<String, LLMYError> {
        let matches = self
            .context
            .graph
            .find_callables(args.module.as_deref(), &args.name);
        if matches.is_empty() {
            return Ok(format!(
                "No callable named {:?} in the code graph.",
                args.name
            ));
        }
        let mut sections = vec![];
        for callable in matches.iter().take(MAX_SOURCE_MATCHES) {
            let source = self
                .context
                .graph
                .read_span(&self.context.root, &callable.file, callable.span)
                .await?;
            sections.push(format!(
                "{}\n```\n{}\n```",
                self.context.render_callable_line(callable),
                source
            ));
        }
        if matches.len() > MAX_SOURCE_MATCHES {
            sections.push(format!(
                "[{} more matches omitted; scope with `module`]",
                matches.len() - MAX_SOURCE_MATCHES
            ));
        }
        Ok(sections.join("\n\n"))
    }
}

/// Arguments accepted by [`ReadModuleSourceTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ReadModuleSourceArgs {
    /// Module / contract name.
    pub module: String,
}

/// Reads a whole module/contract source by name.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ReadModuleSourceArgs,
    invoke = read,
    name = "read_module_source",
    description = "Read the full source of a module/contract by name.",
)]
pub struct ReadModuleSourceTool {
    context: CodegraphContext,
}

impl ReadModuleSourceTool {
    pub fn new(context: CodegraphContext) -> Self {
        Self { context }
    }

    async fn read(&self, args: ReadModuleSourceArgs) -> Result<String, LLMYError> {
        let matches = self.context.modules_named(&args.module);
        if matches.is_empty() {
            return Ok(format!(
                "No module named {:?} in the code graph; use codegraph_overview for the module list.",
                args.module
            ));
        }
        let mut sections = vec![];
        for module in matches.iter().take(MAX_SOURCE_MATCHES) {
            let source = self
                .context
                .graph
                .read_span(&self.context.root, &module.file, module.span)
                .await?;
            sections.push(format!(
                "{} {} ({}:{}..{})\n```\n{}\n```",
                module.kind.render(),
                module.name,
                module.file.display(),
                module.span.start_line,
                module.span.end_line,
                source
            ));
        }
        Ok(sections.join("\n\n"))
    }
}
