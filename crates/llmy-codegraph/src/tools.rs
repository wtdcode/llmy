//! Agent-facing query tools over a built [`CodeGraph`]: overview, per-module
//! and per-callable source reading, call-edge lookups and state
//! cross-references.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use itertools::Itertools;
use llmy_agent::tool::ToolBox;
use llmy_types::error::LLMYError;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::model::{
    AccessKind, Callable, CodeGraph, Declaration, DeclarationKind, LineSpan, Module, ParentRef,
};

const MAX_SOURCE_MATCHES: usize = 3;
/// Characters of a state item's declaration `read_contract_metadata` shows.
const MAX_DECLARATION_CHARS: usize = 240;

/// Which tools [`CodegraphContext::tool_box_with`] offers and how much one
/// module read returns.
#[derive(Debug, Clone, Copy)]
pub struct CodegraphToolOptions {
    /// Offer `read_module_source`.
    pub module_source: bool,
    /// Bytes one `read_module_source` reply returns per module, cut at a line
    /// boundary; the reply says where to continue.
    pub max_read_bytes: usize,
}

impl Default for CodegraphToolOptions {
    fn default() -> Self {
        Self {
            module_source: true,
            max_read_bytes: 8192,
        }
    }
}

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
            let declarations = self.graph.declarations_of_module(module.id);
            lines.push(format!(
                "- {} {} ({}, {}): {} callables, {} state items, {} declarations",
                module.kind.render(),
                module.name,
                module.language.render(),
                module.file.display(),
                callables.len(),
                states.len(),
                declarations.len()
            ));
        }
        lines.join("\n")
    }

    /// Every codegraph tool, `read_module_source` included.
    pub fn tool_box(&self) -> ToolBox {
        self.tool_box_with(CodegraphToolOptions::default())
    }

    pub fn tool_box_with(&self, options: CodegraphToolOptions) -> ToolBox {
        let mut tools = ToolBox::new();
        tools.add_tool(CodegraphOverviewTool::new(self.clone()));
        tools.add_tool(ReadContractMetadataTool::new(self.clone()));
        tools.add_tool(LookupCallableTool::new(self.clone()));
        tools.add_tool(LookupStateTool::new(self.clone()));
        tools.add_tool(ReadCallableSourceTool::new(self.clone()));
        tools.add_tool(ReadTypeDefinitionTool::new(self.clone()));
        if options.module_source {
            tools.add_tool(ReadModuleSourceTool::new(
                self.clone(),
                options.max_read_bytes,
            ));
        }
        tools
    }

    fn modules_named(&self, name: &str) -> Vec<&Module> {
        self.graph.modules_by_name(name)
    }

    fn render_lines(span: LineSpan) -> String {
        if span.start_line == span.end_line {
            format!("line {}", span.start_line)
        } else {
            format!("lines {}..{}", span.start_line, span.end_line)
        }
    }

    fn render_parent(&self, parent: &ParentRef) -> String {
        match parent {
            ParentRef::Resolved(id) => match self.graph.modules.get(id) {
                Some(m) => format!("{} ({})", m.name, m.file.display()),
                None => format!("#{id}"),
            },
            ParentRef::External(name) => format!("{name} (not in the code graph)"),
        }
    }

    /// Ancestors beyond the direct parents, nearest first: each parent's own
    /// parents in declaration order, breadth first, each once.
    fn further_ancestors(&self, module_id: i64) -> Vec<String> {
        let direct: Vec<&ParentRef> = self.graph.parents_of(module_id);
        let mut seen: Vec<ParentRef> = direct.iter().map(|p| (*p).clone()).collect();
        let mut frontier: Vec<i64> = direct
            .iter()
            .filter_map(|p| match p {
                ParentRef::Resolved(id) => Some(*id),
                ParentRef::External(_) => None,
            })
            .collect();
        let mut out = vec![];
        while !frontier.is_empty() {
            let mut next = vec![];
            for id in frontier {
                for parent in self.graph.parents_of(id) {
                    if seen.contains(parent) || *parent == ParentRef::Resolved(module_id) {
                        continue;
                    }
                    seen.push(parent.clone());
                    out.push(self.render_parent(parent));
                    if let ParentRef::Resolved(parent_id) = parent {
                        next.push(*parent_id);
                    }
                }
            }
            frontier = next;
        }
        out
    }

    fn render_declaration_line(declaration: &Declaration) -> String {
        format!(
            "  - {} {} ({})",
            declaration.kind.render(),
            declaration.name,
            Self::render_lines(declaration.span)
        )
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
    description = "Show the code graph overview: indexed counts and every module/contract with its file and its callable, state item and declaration counts.",
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

/// Arguments accepted by [`ReadContractMetadataTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ReadContractMetadataArgs {
    /// Module / contract name.
    pub module: String,
}

/// The shape of one module: what it inherits, the libraries it attaches, the
/// types, events and errors it declares, its state and its callables.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ReadContractMetadataArgs,
    invoke = read,
    name = "read_contract_metadata",
    description = "Describe one module/contract by name: its direct parents and further ancestors, `using … for` directives, the structs, enums, events, errors and user-defined value types it declares, every state item with its declaration, every callable with its signature and lines, and the file-level declarations of its file. Read a listed type with `read_type_definition`, a callable with `read_callable_source`.",
)]
pub struct ReadContractMetadataTool {
    context: CodegraphContext,
}

impl ReadContractMetadataTool {
    pub fn new(context: CodegraphContext) -> Self {
        Self { context }
    }

    async fn read(&self, args: ReadContractMetadataArgs) -> Result<String, LLMYError> {
        let graph = &self.context.graph;
        let modules = self.context.modules_named(&args.module);
        if modules.is_empty() {
            return Ok(format!(
                "No module named {:?} in the code graph; use codegraph_overview for the module list.",
                args.module
            ));
        }
        let mut sections = vec![];
        for module in modules {
            let source = tokio::fs::read_to_string(self.context.root.join(&module.file))
                .await
                .unwrap_or_default();
            let source_lines: Vec<&str> = source.lines().collect();
            let mut lines = vec![format!(
                "{} {} ({}, {}:{}..{})",
                module.kind.render(),
                module.name,
                module.language.render(),
                module.file.display(),
                module.span.start_line,
                module.span.end_line
            )];
            let parents = graph.parents_of(module.id);
            if !parents.is_empty() {
                lines.push(format!(
                    "inherits: {}",
                    parents
                        .iter()
                        .map(|p| self.context.render_parent(p))
                        .join(", ")
                ));
                let further = self.context.further_ancestors(module.id);
                if !further.is_empty() {
                    lines.push(format!("further ancestors: {}", further.join(", ")));
                }
            }
            let declarations = graph.declarations_of_module(module.id);
            let (using, declared): (Vec<&Declaration>, Vec<&Declaration>) = declarations
                .into_iter()
                .partition(|d| d.kind == DeclarationKind::UsingFor);
            if !using.is_empty() {
                lines.push("using:".to_string());
                for directive in using {
                    lines.push(format!(
                        "  - {} ({})",
                        directive.name,
                        CodegraphContext::render_lines(directive.span)
                    ));
                }
            }
            if !declared.is_empty() {
                lines.push("declarations:".to_string());
                for declaration in declared {
                    lines.push(CodegraphContext::render_declaration_line(declaration));
                }
            }
            let states = graph.states_of_module(module.id);
            if !states.is_empty() {
                lines.push("state items:".to_string());
                for state in states {
                    // The declaration as written carries visibility,
                    // constant/immutable and the initial value.
                    let written = source_lines
                        .get(
                            state.span.start_line.saturating_sub(1)
                                ..state.span.end_line.min(source_lines.len()),
                        )
                        .map(|l| l.join(" ").split_whitespace().join(" "))
                        .filter(|t| !t.is_empty())
                        .unwrap_or_else(|| state.type_text.clone());
                    let written = match written.char_indices().nth(MAX_DECLARATION_CHARS) {
                        Some((cut, _)) => format!("{}…", &written[..cut]),
                        None => written,
                    };
                    lines.push(format!(
                        "  - {} [{}] ({}): {}",
                        state.name,
                        state.kind.render(),
                        CodegraphContext::render_lines(state.span),
                        written
                    ));
                }
            }
            lines.push("callables:".to_string());
            for callable in graph.callables_of_module(module.id) {
                lines.push(format!(
                    "  - [{}] {} ({})",
                    callable.kind.render(),
                    callable.signature,
                    CodegraphContext::render_lines(callable.span)
                ));
            }
            let file_level = graph.file_declarations(&module.file);
            if !file_level.is_empty() {
                lines.push(format!(
                    "file-level declarations in {}:",
                    module.file.display()
                ));
                for declaration in file_level {
                    lines.push(CodegraphContext::render_declaration_line(declaration));
                }
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

/// Arguments accepted by [`ReadTypeDefinitionTool`].
#[derive(Deserialize, JsonSchema)]
pub struct ReadTypeDefinitionArgs {
    /// Name of the struct, enum, event, error, user-defined value type or
    /// file-level constant.
    pub name: String,
    /// Optional name of the module/contract that declares it (file-level
    /// declarations have none).
    #[serde(default)]
    pub module: Option<String>,
    /// Optional file that declares it, relative to the project root: the way
    /// to pick one of several file-level declarations sharing a name.
    #[serde(default)]
    pub file: Option<String>,
}

/// Reads a struct, enum, event, error, user-defined value type or file-level
/// constant by name.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ReadTypeDefinitionArgs,
    invoke = read,
    name = "read_type_definition",
    description = "Read the definition of a struct, enum, event, custom error, user-defined value type or file-level constant by name, optionally scoped to the module/contract or the file that declares it. `read_contract_metadata` lists what a contract and its file declare.",
)]
pub struct ReadTypeDefinitionTool {
    context: CodegraphContext,
}

impl ReadTypeDefinitionTool {
    pub fn new(context: CodegraphContext) -> Self {
        Self { context }
    }

    async fn read(&self, args: ReadTypeDefinitionArgs) -> Result<String, LLMYError> {
        let graph = &self.context.graph;
        let owner = |declaration: &Declaration| match declaration.module_id {
            Some(id) => graph
                .modules
                .get(&id)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| format!("#{id}")),
            None => "file level".to_string(),
        };
        let where_declared = |declarations: &[&Declaration]| {
            declarations
                .iter()
                .map(|d| format!("{} ({})", owner(d), d.file.display()))
                .join(", ")
        };
        let file = args
            .file
            .as_deref()
            .map(|f| Path::new(f.trim_start_matches("./")).to_path_buf());
        let matches: Vec<&Declaration> = graph
            .find_declarations(args.module.as_deref(), &args.name)
            .into_iter()
            .filter(|d| file.as_ref().is_none_or(|f| &d.file == f))
            .collect();
        if matches.is_empty() {
            let elsewhere = graph.find_declarations(None, &args.name);
            let scoped = args.module.is_some() || args.file.is_some();
            return Ok(if scoped && !elsewhere.is_empty() {
                format!(
                    "No declaration named {:?} in that module or file; it is declared in: {}.",
                    args.name,
                    where_declared(&elsewhere)
                )
            } else {
                format!(
                    "No struct, enum, event, error, user-defined value type or file-level constant named {:?} in the code graph.",
                    args.name
                )
            });
        }
        let mut sections = vec![];
        for declaration in matches.iter().take(MAX_SOURCE_MATCHES) {
            let source = graph
                .read_span(&self.context.root, &declaration.file, declaration.span)
                .await?;
            sections.push(format!(
                "{} {} in {} ({}:{}..{})\n```\n{}\n```",
                declaration.kind.render(),
                declaration.name,
                owner(declaration),
                declaration.file.display(),
                declaration.span.start_line,
                declaration.span.end_line,
                source
            ));
        }
        if matches.len() > MAX_SOURCE_MATCHES {
            sections.push(format!(
                "[{} more matches omitted: {}; scope with `module` or `file`]",
                matches.len() - MAX_SOURCE_MATCHES,
                where_declared(&matches[MAX_SOURCE_MATCHES..])
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
    /// Optional file line number to start from, inside the module. Defaults
    /// to the module's first line.
    #[serde(default)]
    pub start_line: Option<usize>,
    /// Optional number of lines to read from `start_line`.
    #[serde(default)]
    pub line_count: Option<usize>,
}

/// Reads a module/contract source by name, whole or a line range, capped in
/// bytes.
#[derive(Debug, Clone)]
#[llmy_agent::tool(
    arguments = ReadModuleSourceArgs,
    invoke = read,
    name = "read_module_source",
    description = "Read the source of a module/contract by name, whole or from `start_line` (a file line number) for `line_count` lines. One reply is capped in size and says where to continue; prefer `read_contract_metadata`, `read_callable_source` and `read_type_definition` for the parts you need.",
)]
pub struct ReadModuleSourceTool {
    context: CodegraphContext,
    max_read_bytes: usize,
}

impl ReadModuleSourceTool {
    pub fn new(context: CodegraphContext, max_read_bytes: usize) -> Self {
        Self {
            context,
            max_read_bytes,
        }
    }

    async fn read(&self, args: ReadModuleSourceArgs) -> Result<String, LLMYError> {
        if args.start_line == Some(0) || args.line_count == Some(0) {
            return Ok("start_line and line_count must be at least 1".to_string());
        }
        let matches = self.context.modules_named(&args.module);
        if matches.is_empty() {
            return Ok(format!(
                "No module named {:?} in the code graph; use codegraph_overview for the module list.",
                args.module
            ));
        }
        let mut sections = vec![];
        for module in matches.iter().take(MAX_SOURCE_MATCHES) {
            let span = module.span;
            let header = format!(
                "{} {} ({}:{}..{})",
                module.kind.render(),
                module.name,
                module.file.display(),
                span.start_line,
                span.end_line
            );
            let from = args
                .start_line
                .unwrap_or(span.start_line)
                .max(span.start_line);
            if from > span.end_line {
                sections.push(format!(
                    "{header}\nstart_line {from} is past the module's last line {}",
                    span.end_line
                ));
                continue;
            }
            let to = args
                .line_count
                .map(|n| from.saturating_add(n - 1))
                .unwrap_or(span.end_line)
                .min(span.end_line);
            let text = self
                .context
                .graph
                .read_span(
                    &self.context.root,
                    &module.file,
                    LineSpan {
                        start_line: from,
                        end_line: to,
                    },
                )
                .await
                .unwrap_or_default();
            // Whole lines up to the cap; a single longer line is cut at a
            // char boundary.
            let mut shown = String::new();
            let mut last = from - 1;
            for line in text.lines() {
                let first = last < from;
                if !first && shown.len() + 1 + line.len() > self.max_read_bytes {
                    break;
                }
                if first && line.len() > self.max_read_bytes {
                    let mut cut = self.max_read_bytes;
                    while cut > 0 && !line.is_char_boundary(cut) {
                        cut -= 1;
                    }
                    shown.push_str(&line[..cut]);
                    last += 1;
                    break;
                }
                if !first {
                    shown.push('\n');
                }
                shown.push_str(line);
                last += 1;
            }
            let mut section = format!("{header}, lines {from}..{last}\n```\n{shown}\n```");
            if last < to {
                section.push_str(&format!(
                    "\n[{} more lines up to line {to}; continue with start_line={}]",
                    to - last,
                    last + 1
                ));
            }
            sections.push(section);
        }
        if matches.len() > MAX_SOURCE_MATCHES {
            sections.push(format!(
                "[{} more modules named {:?} omitted]",
                matches.len() - MAX_SOURCE_MATCHES,
                args.module
            ));
        }
        Ok(sections.join("\n\n"))
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::model::{InheritEdge, Language, ModuleKind};

    /// Leaf is Mid; Mid is Base, Context (outside the graph); Base is Context.
    fn layered() -> CodegraphContext {
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
                        end_line: 2,
                    },
                },
            );
        }
        for (module_id, parent) in [
            (3, ParentRef::Resolved(2)),
            (2, ParentRef::Resolved(1)),
            (2, ParentRef::External("Context".to_string())),
            (1, ParentRef::External("Context".to_string())),
        ] {
            graph.inherit_edges.push(InheritEdge { module_id, parent });
        }
        CodegraphContext::new(graph, PathBuf::from("/nonexistent"))
    }

    #[tokio::test]
    async fn metadata_lists_direct_parents_then_further_ancestors_once() {
        let metadata = layered()
            .tool_box()
            .invoke(
                "read_contract_metadata".to_string(),
                r#"{"module": "Leaf"}"#.to_string(),
            )
            .await
            .expect("tool exists")
            .expect("tool ran");
        assert!(metadata.contains("inherits: Mid (a.sol)\n"), "{metadata}");
        assert!(
            metadata.contains("further ancestors: Base (a.sol), Context (not in the code graph)\n"),
            "{metadata}"
        );
    }
}
