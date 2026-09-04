//! Shared extraction machinery: the raw per-file representation every
//! language extractor produces, the tree-sitter language registry (including
//! the vendored Move grammars), and node helpers used by all extractors.

use std::path::PathBuf;

use color_eyre::eyre::eyre;
use llmy_types::error::LLMYError;
use tree_sitter::{Language as TsLanguage, Node, Parser, Tree};

use crate::model::{CallableKind, Language, LineSpan, ModuleKind, StateKind};

unsafe extern "C" {
    fn tree_sitter_move_on_aptos() -> *const std::ffi::c_void;
    fn tree_sitter_move() -> *const std::ffi::c_void;
}

/// A source file handed to an extractor, path kept relative to the root.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub relative: PathBuf,
    pub content: String,
}

/// A call site as written in the source, before resolution.
#[derive(Debug, Clone)]
pub struct RawCallSite {
    /// Full call target text (e.g. `token.transfer`, `coin::withdraw`).
    pub text: String,
    /// The final name segment used for resolution.
    pub name: String,
    /// The qualifier before the final segment, if any (`token`, `coin`).
    pub qualifier: Option<String>,
    pub line: usize,
}

/// A by-name reference from a callable body to a state item, before
/// resolution.
#[derive(Debug, Clone)]
pub struct RawStateRef {
    pub name: String,
    pub write: bool,
    pub line: usize,
}

#[derive(Debug, Clone)]
pub struct RawCallable {
    pub name: String,
    pub kind: CallableKind,
    pub signature: String,
    pub span: LineSpan,
    pub calls: Vec<RawCallSite>,
    pub state_refs: Vec<RawStateRef>,
}

#[derive(Debug, Clone)]
pub struct RawState {
    pub name: String,
    pub kind: StateKind,
    pub type_text: String,
    pub span: LineSpan,
}

#[derive(Debug, Clone)]
pub struct RawModule {
    pub name: String,
    pub kind: ModuleKind,
    pub span: LineSpan,
    /// Direct parents by name (Solidity inheritance).
    pub parents: Vec<String>,
    pub callables: Vec<RawCallable>,
    pub states: Vec<RawState>,
}

/// Everything extracted from one file.
#[derive(Debug, Clone)]
pub struct FileExtraction {
    pub file: PathBuf,
    pub language: Language,
    pub modules: Vec<RawModule>,
    /// Number of ERROR nodes tree-sitter produced — used for dialect picking
    /// and reported as a quality signal.
    pub parse_errors: usize,
}

/// Parser front-end over the four supported grammars.
pub struct GrammarSet;

impl GrammarSet {
    pub fn language_of(language: Language) -> TsLanguage {
        match language {
            Language::Solidity => tree_sitter_solidity::LANGUAGE.into(),
            Language::Rust => tree_sitter_rust::LANGUAGE.into(),
            Language::MoveAptos => {
                let ptr = unsafe { tree_sitter_move_on_aptos() };
                unsafe { TsLanguage::from_raw(ptr.cast()) }
            }
            Language::MoveSui => {
                let ptr = unsafe { tree_sitter_move() };
                unsafe { TsLanguage::from_raw(ptr.cast()) }
            }
        }
    }

    pub fn parse(language: Language, content: &str) -> Result<Tree, LLMYError> {
        let mut parser = Parser::new();
        parser
            .set_language(&Self::language_of(language))
            .map_err(|e| eyre!("failed to load {} grammar: {}", language.render(), e))?;
        parser
            .parse(content, None)
            .ok_or_else(|| eyre!("{} parse returned no tree", language.render()).into())
    }

    pub fn count_errors(node: Node<'_>) -> usize {
        let mut count = 0;
        let mut cursor = node.walk();
        let mut stack = vec![node];
        while let Some(current) = stack.pop() {
            if current.is_error() || current.is_missing() {
                count += 1;
            }
            for child in current.children(&mut cursor) {
                stack.push(child);
            }
        }
        count
    }

    /// Render a parse tree for grammar debugging (kinds, lines and short
    /// text), used by tests when adapting to a grammar's node kinds.
    pub fn dump(node: Node<'_>, source: &str, depth: usize) -> String {
        let mut out = String::new();
        if node.is_named() {
            let text = node.text_of(source);
            let short: String = text
                .chars()
                .take(40)
                .collect::<String>()
                .replace('\n', "\\n");
            out.push_str(&format!(
                "{}{} [{}..{}] {:?}\n",
                "  ".repeat(depth),
                node.kind(),
                node.start_position().row + 1,
                node.end_position().row + 1,
                short
            ));
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            out.push_str(&Self::dump(child, source, depth + 1));
        }
        out
    }
}

/// Convenience methods every extractor needs on tree-sitter nodes.
pub trait NodeUtil<'tree> {
    fn text_of(&self, source: &str) -> String;
    fn line_span(&self) -> LineSpan;
    fn field_text(&self, field: &str, source: &str) -> Option<String>;
    fn child_of_kind(&self, kind: &str) -> Option<Node<'tree>>;
    fn children_of_kind(&self, kind: &str) -> Vec<Node<'tree>>;
    /// Every descendant (including self) whose kind is in `kinds`, in
    /// document order. Does not descend *into* matched nodes when
    /// `enter_matches` is false.
    fn descendants_of_kinds(&self, kinds: &[&str], enter_matches: bool) -> Vec<Node<'tree>>;
    /// The first identifier-like descendant's text.
    fn first_identifier(&self, source: &str) -> Option<String>;
    /// The declaration's text cut at the start of its body child — the
    /// signature head as written, whitespace-normalized.
    fn signature_head(&self, body_kinds: &[&str], source: &str) -> String;
}

impl<'tree> NodeUtil<'tree> for Node<'tree> {
    fn text_of(&self, source: &str) -> String {
        source
            .get(self.start_byte()..self.end_byte())
            .unwrap_or_default()
            .to_string()
    }

    fn line_span(&self) -> LineSpan {
        LineSpan {
            start_line: self.start_position().row + 1,
            end_line: self.end_position().row + 1,
        }
    }

    fn field_text(&self, field: &str, source: &str) -> Option<String> {
        self.child_by_field_name(field.as_bytes())
            .map(|node| node.text_of(source))
    }

    fn child_of_kind(&self, kind: &str) -> Option<Node<'tree>> {
        let mut cursor = self.walk();
        self.children(&mut cursor).find(|c| c.kind() == kind)
    }

    fn children_of_kind(&self, kind: &str) -> Vec<Node<'tree>> {
        let mut cursor = self.walk();
        self.children(&mut cursor)
            .filter(|c| c.kind() == kind)
            .collect()
    }

    fn descendants_of_kinds(&self, kinds: &[&str], enter_matches: bool) -> Vec<Node<'tree>> {
        let mut out = vec![];
        let mut stack = vec![*self];
        while let Some(current) = stack.pop() {
            let matched = kinds.contains(&current.kind());
            if matched && current.id() != self.id() {
                out.push(current);
                if !enter_matches {
                    continue;
                }
            }
            let mut cursor = current.walk();
            let mut children: Vec<_> = current.children(&mut cursor).collect();
            children.reverse();
            stack.extend(children);
        }
        // Depth-first with a stack visits in reverse; restore document order.
        out.sort_by_key(|node| node.start_byte());
        out
    }

    fn first_identifier(&self, source: &str) -> Option<String> {
        if self.kind().contains("identifier") {
            return Some(self.text_of(source));
        }
        let mut stack = vec![*self];
        let mut found: Vec<(usize, String)> = vec![];
        while let Some(current) = stack.pop() {
            if current.kind().contains("identifier") {
                found.push((current.start_byte(), current.text_of(source)));
                continue;
            }
            let mut cursor = current.walk();
            for child in current.children(&mut cursor) {
                stack.push(child);
            }
        }
        found.sort();
        found.into_iter().next().map(|(_, text)| text)
    }

    fn signature_head(&self, body_kinds: &[&str], source: &str) -> String {
        let end = {
            let mut cursor = self.walk();
            self.children(&mut cursor)
                .find(|c| body_kinds.contains(&c.kind()))
                .map(|body| body.start_byte())
                .unwrap_or(self.end_byte())
        };
        let head = source.get(self.start_byte()..end).unwrap_or_default();
        head.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}
