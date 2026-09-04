//! Builds a [`CodeGraph`] from a project root: file discovery (gitignore
//! aware), Move dialect detection per package, extraction, and by-name
//! resolution of call sites and state references. Resolution is syntactic:
//! same-module candidates win, a qualifier that names a module scopes the
//! search, everything else stays ambiguous or external — explicitly.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use color_eyre::eyre::eyre;
use ignore::WalkBuilder;
use llmy_types::error::LLMYError;

use crate::extract::{FileExtraction, SourceFile};
use crate::model::{
    CallEdge, Callable, CalleeRef, CodeGraph, InheritEdge, Language, Module, ParentRef, StateEdge,
    StateItem,
};
use crate::move_lang::{MoveAptosExtractor, MoveSuiExtractor};
use crate::rust_lang::RustExtractor;
use crate::solidity::SolidityExtractor;

const SOLIDITY_BUILTINS: [&str; 22] = [
    "require",
    "assert",
    "revert",
    "keccak256",
    "sha256",
    "ripemd160",
    "ecrecover",
    "addmod",
    "mulmod",
    "selfdestruct",
    "blockhash",
    "gasleft",
    "payable",
    "type",
    "address",
    "uint",
    "uint256",
    "int",
    "bytes",
    "bytes32",
    "string",
    "bool",
];
const RUST_BUILTINS: [&str; 6] = ["Ok", "Err", "Some", "None", "vec", "panic"];
const MOVE_BUILTINS: [&str; 4] = ["assert", "abort", "freeze", "vector"];

/// The result of one indexing pass.
#[derive(Debug, Clone)]
pub struct BuildResult {
    pub graph: CodeGraph,
    /// Files that were parsed, with their per-file parse error counts.
    pub files: Vec<(PathBuf, Language, usize)>,
    /// Fingerprint of the indexed inputs, for cache staleness checks.
    pub fingerprint: String,
}

impl BuildResult {
    pub fn total_parse_errors(&self) -> usize {
        self.files.iter().map(|(_, _, errors)| errors).sum()
    }
}

pub struct CodeGraphBuilder {
    root: PathBuf,
}

impl CodeGraphBuilder {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub async fn build(&self) -> Result<BuildResult, LLMYError> {
        let sources = self.discover().await?;
        let mut extractions = vec![];
        let mut files = vec![];
        let mut fingerprint_entries = vec![];

        for (source, language) in sources {
            let extraction = match language {
                Language::Solidity => SolidityExtractor::extract(&source)?,
                Language::Rust => RustExtractor::extract(&source)?,
                Language::MoveAptos => MoveAptosExtractor::extract(&source)?,
                Language::MoveSui => MoveSuiExtractor::extract(&source)?,
            };
            if extraction.parse_errors > 0 {
                tracing::warn!(
                    "{} produced {} parse errors in {}",
                    language.render(),
                    extraction.parse_errors,
                    source.relative.display()
                );
            }
            fingerprint_entries.push(format!(
                "{}:{}",
                source.relative.display(),
                source.content.len()
            ));
            files.push((source.relative.clone(), language, extraction.parse_errors));
            extractions.push(extraction);
        }

        fingerprint_entries.sort();
        let graph = GraphAssembler::assemble(extractions);
        Ok(BuildResult {
            graph,
            files,
            fingerprint: fingerprint_entries.join("\n"),
        })
    }

    /// Current fingerprint of the root without extracting anything — cheap
    /// staleness probe for the cache.
    pub async fn fingerprint(&self) -> Result<String, LLMYError> {
        let sources = self.discover().await?;
        let mut entries: Vec<String> = sources
            .iter()
            .map(|(source, _)| format!("{}:{}", source.relative.display(), source.content.len()))
            .collect();
        entries.sort();
        Ok(entries.join("\n"))
    }

    async fn discover(&self) -> Result<Vec<(SourceFile, Language)>, LLMYError> {
        let root = self
            .root
            .canonicalize()
            .map_err(|e| eyre!("cannot canonicalize {}: {}", self.root.display(), e))?;

        let mut move_dialects: BTreeMap<PathBuf, Language> = BTreeMap::new();
        let mut out = vec![];
        for entry in WalkBuilder::new(&root).build() {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    tracing::debug!("codegraph walk error: {}", error);
                    continue;
                }
            };
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                continue;
            }
            let path = entry.path();
            let language = match path.extension().and_then(|e| e.to_str()) {
                Some("sol") => Language::Solidity,
                Some("rs") => Language::Rust,
                Some("move") => self.move_dialect(path, &mut move_dialects).await,
                _ => continue,
            };
            let content = match tokio::fs::read_to_string(path).await {
                Ok(content) => content,
                Err(error) => {
                    tracing::debug!("skipping unreadable {}: {}", path.display(), error);
                    continue;
                }
            };
            let relative = path.strip_prefix(&root).unwrap_or(path).to_path_buf();
            out.push((SourceFile { relative, content }, language));
        }
        out.sort_by(|a, b| a.0.relative.cmp(&b.0.relative));
        Ok(out)
    }

    /// Dialect of a `.move` file: the nearest `Move.toml` decides (a Sui
    /// framework dependency or a `sui` edition marks Sui), cached per
    /// directory. Without a manifest the Aptos grammar is assumed — the
    /// caller sees parse error counts either way.
    async fn move_dialect(&self, file: &Path, cache: &mut BTreeMap<PathBuf, Language>) -> Language {
        let mut dir = file.parent();
        while let Some(current) = dir {
            if let Some(cached) = cache.get(current) {
                return *cached;
            }
            let manifest = current.join("Move.toml");
            if manifest.is_file() {
                let dialect = match tokio::fs::read_to_string(&manifest).await {
                    Ok(content) => {
                        let lowered = content.to_lowercase();
                        if lowered.contains("sui") {
                            Language::MoveSui
                        } else {
                            Language::MoveAptos
                        }
                    }
                    Err(_) => Language::MoveAptos,
                };
                cache.insert(current.to_path_buf(), dialect);
                return dialect;
            }
            dir = current.parent();
        }
        Language::MoveAptos
    }
}

/// Turns per-file extractions into one resolved [`CodeGraph`].
struct GraphAssembler {
    graph: CodeGraph,
    /// callable name -> ids, for call resolution.
    callables_by_name: BTreeMap<String, Vec<i64>>,
    /// module name -> ids.
    modules_by_name: BTreeMap<String, Vec<i64>>,
}

impl GraphAssembler {
    fn assemble(extractions: Vec<FileExtraction>) -> CodeGraph {
        let mut assembler = Self {
            graph: CodeGraph::default(),
            callables_by_name: BTreeMap::new(),
            modules_by_name: BTreeMap::new(),
        };
        // Pass one: nodes with globally assigned ids. Raw call sites and
        // state refs are kept alongside for pass two.
        let mut pending_calls: Vec<(i64, crate::extract::RawCallSite, Language)> = vec![];
        let mut pending_states: Vec<(i64, i64, crate::extract::RawStateRef, Language)> = vec![];
        let mut pending_parents: Vec<(i64, String)> = vec![];

        let mut next_module = 1i64;
        let mut next_callable = 1i64;
        let mut next_state = 1i64;

        for extraction in extractions {
            for raw_module in extraction.modules {
                let module_id = next_module;
                next_module += 1;
                assembler.graph.modules.insert(
                    module_id,
                    Module {
                        id: module_id,
                        language: extraction.language,
                        kind: raw_module.kind,
                        name: raw_module.name.clone(),
                        file: extraction.file.clone(),
                        span: raw_module.span,
                    },
                );
                assembler
                    .modules_by_name
                    .entry(raw_module.name.clone())
                    .or_default()
                    .push(module_id);
                for parent in raw_module.parents {
                    pending_parents.push((module_id, parent));
                }
                for raw_state in raw_module.states {
                    let state_id = next_state;
                    next_state += 1;
                    assembler.graph.states.insert(
                        state_id,
                        StateItem {
                            id: state_id,
                            module_id,
                            kind: raw_state.kind,
                            name: raw_state.name,
                            type_text: raw_state.type_text,
                            file: extraction.file.clone(),
                            span: raw_state.span,
                        },
                    );
                }
                for raw_callable in raw_module.callables {
                    let callable_id = next_callable;
                    next_callable += 1;
                    assembler.graph.callables.insert(
                        callable_id,
                        Callable {
                            id: callable_id,
                            module_id,
                            kind: raw_callable.kind,
                            name: raw_callable.name.clone(),
                            signature: raw_callable.signature,
                            file: extraction.file.clone(),
                            span: raw_callable.span,
                        },
                    );
                    assembler
                        .callables_by_name
                        .entry(raw_callable.name)
                        .or_default()
                        .push(callable_id);
                    for call in raw_callable.calls {
                        pending_calls.push((callable_id, call, extraction.language));
                    }
                    for state_ref in raw_callable.state_refs {
                        pending_states.push((
                            callable_id,
                            module_id,
                            state_ref,
                            extraction.language,
                        ));
                    }
                }
            }
        }

        for (module_id, parent_name) in pending_parents {
            let parent = match assembler.modules_by_name.get(&parent_name) {
                Some(ids) if ids.len() == 1 => ParentRef::Resolved(ids[0]),
                Some(ids) if !ids.is_empty() => ParentRef::Resolved(ids[0]),
                _ => ParentRef::External(parent_name),
            };
            assembler
                .graph
                .inherit_edges
                .push(InheritEdge { module_id, parent });
        }

        assembler.resolve_calls(pending_calls);
        assembler.resolve_states(pending_states);
        assembler.graph
    }

    fn builtin(language: Language, name: &str) -> bool {
        match language {
            Language::Solidity => SOLIDITY_BUILTINS.contains(&name),
            Language::Rust => RUST_BUILTINS.contains(&name),
            Language::MoveAptos | Language::MoveSui => MOVE_BUILTINS.contains(&name),
        }
    }

    fn resolve_calls(&mut self, pending: Vec<(i64, crate::extract::RawCallSite, Language)>) {
        let mut seen: BTreeSet<(i64, String, usize)> = BTreeSet::new();
        for (caller_id, site, language) in pending {
            if Self::builtin(language, &site.name) {
                continue;
            }
            if !seen.insert((caller_id, site.text.clone(), site.line)) {
                continue;
            }
            let caller_module = self
                .graph
                .callables
                .get(&caller_id)
                .map(|c| c.module_id)
                .unwrap_or(0);

            let mut candidates: Vec<i64> = self
                .callables_by_name
                .get(&site.name)
                .cloned()
                .unwrap_or_default();

            // A qualifier that names a known module scopes the candidates to
            // that module (`token.transfer` with a `Token` variable does not
            // match this — only real module names do).
            if let Some(qualifier) = &site.qualifier {
                let qualified_modules: BTreeSet<i64> = self
                    .modules_by_name
                    .iter()
                    .filter(|(name, _)| {
                        name.as_str() == qualifier || name.eq_ignore_ascii_case(qualifier)
                    })
                    .flat_map(|(_, ids)| ids.iter().copied())
                    .collect();
                if !qualified_modules.is_empty() {
                    let scoped: Vec<i64> = candidates
                        .iter()
                        .copied()
                        .filter(|id| {
                            self.graph
                                .callables
                                .get(id)
                                .map(|c| qualified_modules.contains(&c.module_id))
                                .unwrap_or(false)
                        })
                        .collect();
                    if !scoped.is_empty() {
                        candidates = scoped;
                    }
                }
            }

            // Same-module (or inherited) candidates shadow project-wide ones.
            if candidates.len() > 1 {
                let mut visible = self.graph.ancestors_of(caller_module);
                visible.insert(caller_module);
                let local: Vec<i64> = candidates
                    .iter()
                    .copied()
                    .filter(|id| {
                        self.graph
                            .callables
                            .get(id)
                            .map(|c| visible.contains(&c.module_id))
                            .unwrap_or(false)
                    })
                    .collect();
                if !local.is_empty() {
                    candidates = local;
                }
            }

            let callee = match candidates.len() {
                0 => {
                    // Unqualified method-style noise in Rust (mostly stdlib
                    // methods) is not worth an external edge.
                    if language == Language::Rust && site.qualifier.is_some() {
                        continue;
                    }
                    CalleeRef::External(site.text.clone())
                }
                1 => CalleeRef::Resolved(candidates[0]),
                _ => CalleeRef::Ambiguous(candidates),
            };
            self.graph.call_edges.push(CallEdge {
                caller_id,
                callee,
                callee_text: site.text,
                line: site.line,
            });
        }
    }

    fn resolve_states(&mut self, pending: Vec<(i64, i64, crate::extract::RawStateRef, Language)>) {
        let mut seen: BTreeSet<(i64, i64, bool)> = BTreeSet::new();
        for (callable_id, module_id, state_ref, language) in pending {
            // Move objects/resources may live in another module of the
            // project; Solidity/Rust state is scoped to the module (plus
            // inherited contracts for Solidity).
            let project_wide = matches!(language, Language::MoveAptos | Language::MoveSui);
            let matched: Vec<i64> = if project_wide {
                self.graph
                    .states
                    .values()
                    .filter(|s| s.name == state_ref.name)
                    .map(|s| s.id)
                    .collect()
            } else {
                self.graph
                    .visible_states(module_id)
                    .into_iter()
                    .filter(|s| s.name == state_ref.name)
                    .map(|s| s.id)
                    .collect()
            };
            for state_id in matched {
                let access = if state_ref.write {
                    crate::model::AccessKind::Write
                } else {
                    crate::model::AccessKind::Read
                };
                if seen.insert((callable_id, state_id, state_ref.write)) {
                    self.graph.state_edges.push(StateEdge {
                        callable_id,
                        state_id,
                        access,
                        line: state_ref.line,
                    });
                }
            }
        }
    }
}
