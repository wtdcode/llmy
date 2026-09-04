//! Solidity extraction over tree-sitter-solidity: contracts / interfaces /
//! libraries, callables (functions, constructor, modifiers,
//! fallback/receive), state variables, inheritance, call sites and
//! state-variable reads/writes. Writes through storage-pointer aliases are a
//! known precision loss of the pure-syntax approach.

use std::collections::BTreeSet;

use llmy_types::error::LLMYError;
use tree_sitter::Node;

use crate::extract::{
    FileExtraction, GrammarSet, NodeUtil, RawCallSite, RawCallable, RawModule, RawState,
    RawStateRef, SourceFile,
};
use crate::model::{CallableKind, Language, ModuleKind, StateKind};

const CONTAINER_KINDS: [&str; 3] = [
    "contract_declaration",
    "interface_declaration",
    "library_declaration",
];
const CALLABLE_KINDS: [&str; 4] = [
    "function_definition",
    "constructor_definition",
    "modifier_definition",
    "fallback_receive_definition",
];
const WRITE_KINDS: [&str; 3] = [
    "assignment_expression",
    "augmented_assignment_expression",
    "update_expression",
];

pub struct SolidityExtractor;

impl SolidityExtractor {
    pub fn extract(file: &SourceFile) -> Result<FileExtraction, LLMYError> {
        let tree = GrammarSet::parse(Language::Solidity, &file.content)?;
        let root = tree.root_node();
        let parse_errors = GrammarSet::count_errors(root);

        let modules = root
            .descendants_of_kinds(&CONTAINER_KINDS, false)
            .into_iter()
            .map(|node| Self::extract_container(node, &file.content))
            .collect();

        Ok(FileExtraction {
            file: file.relative.clone(),
            language: Language::Solidity,
            modules,
            parse_errors,
        })
    }

    fn extract_container(node: Node<'_>, source: &str) -> RawModule {
        let kind = match node.kind() {
            "interface_declaration" => ModuleKind::Interface,
            "library_declaration" => ModuleKind::Library,
            _ => ModuleKind::Contract,
        };
        let name = node
            .child_of_kind("identifier")
            .map(|n| n.text_of(source))
            .unwrap_or_else(|| "<anonymous>".to_string());
        let parents = node
            .children_of_kind("inheritance_specifier")
            .into_iter()
            .filter_map(|spec| spec.first_identifier(source))
            .collect();

        let mut states = vec![];
        let mut callables = vec![];
        if let Some(body) = node.child_of_kind("contract_body") {
            for declaration in body.children_of_kind("state_variable_declaration") {
                let Some(state_name) = declaration
                    .child_of_kind("identifier")
                    .map(|n| n.text_of(source))
                else {
                    continue;
                };
                let type_text = declaration
                    .child_of_kind("type_name")
                    .map(|n| n.text_of(source))
                    .unwrap_or_default();
                states.push(RawState {
                    name: state_name,
                    kind: StateKind::StateVariable,
                    type_text,
                    span: declaration.line_span(),
                });
            }

            let mut cursor = body.walk();
            for child in body.children(&mut cursor) {
                if CALLABLE_KINDS.contains(&child.kind()) {
                    callables.push(Self::extract_callable(child, source));
                }
            }
        }

        RawModule {
            name,
            kind,
            span: node.line_span(),
            parents,
            callables,
            states,
        }
    }

    fn extract_callable(node: Node<'_>, source: &str) -> RawCallable {
        let (name, kind) = match node.kind() {
            "constructor_definition" => ("constructor".to_string(), CallableKind::Constructor),
            "modifier_definition" => (
                node.child_of_kind("identifier")
                    .map(|n| n.text_of(source))
                    .unwrap_or_else(|| "<modifier>".to_string()),
                CallableKind::Modifier,
            ),
            "fallback_receive_definition" => {
                let head = node.text_of(source);
                if head.trim_start().starts_with("receive") {
                    ("receive".to_string(), CallableKind::Receive)
                } else {
                    ("fallback".to_string(), CallableKind::Fallback)
                }
            }
            _ => (
                node.child_of_kind("identifier")
                    .map(|n| n.text_of(source))
                    .unwrap_or_else(|| "<function>".to_string()),
                CallableKind::Function,
            ),
        };

        let signature = node.signature_head(&["function_body"], source);
        let mut calls = vec![];

        // Modifier invocations on the declaration are call edges to the
        // modifier (or a base constructor).
        for invocation in node.children_of_kind("modifier_invocation") {
            if let Some(modifier_name) = invocation.first_identifier(source) {
                calls.push(RawCallSite {
                    text: modifier_name.clone(),
                    name: modifier_name,
                    qualifier: None,
                    line: invocation.line_span().start_line,
                });
            }
        }

        let mut state_refs = vec![];
        if let Some(body) = node.child_of_kind("function_body") {
            for call in body.descendants_of_kinds(&["call_expression"], true) {
                if let Some(site) = Self::call_target(call, source) {
                    calls.push(site);
                }
            }

            let locals = Self::collect_locals(node, body, source);
            state_refs = Self::collect_state_refs(body, source, &locals);
        }

        RawCallable {
            name,
            kind,
            signature,
            span: node.line_span(),
            calls,
            state_refs,
        }
    }

    /// The `(qualifier, name)` of a call expression's target: the callee
    /// side of `call_expression > expression > (identifier |
    /// member_expression)`.
    fn call_target(call: Node<'_>, source: &str) -> Option<RawCallSite> {
        let target = Self::unwrap_expression(call.named_child(0)?);
        let line = call.line_span().start_line;
        match target.kind() {
            "identifier" => {
                let name = target.text_of(source);
                Some(RawCallSite {
                    text: name.clone(),
                    name,
                    qualifier: None,
                    line,
                })
            }
            "member_expression" => {
                let mut cursor = target.walk();
                let identifiers: Vec<Node<'_>> = target
                    .children(&mut cursor)
                    .filter(|c| c.is_named())
                    .collect();
                let property = identifiers.last()?;
                if property.kind() != "identifier" {
                    return None;
                }
                let name = property.text_of(source);
                let qualifier = identifiers
                    .first()
                    .filter(|object| object.id() != property.id())
                    .map(|object| object.text_of(source));
                Some(RawCallSite {
                    text: target.text_of(source),
                    name,
                    qualifier,
                    line,
                })
            }
            _ => None,
        }
    }

    fn unwrap_expression(node: Node<'_>) -> Node<'_> {
        let mut current = node;
        while current.kind() == "expression" {
            match current.named_child(0) {
                Some(inner) => current = inner,
                None => break,
            }
        }
        current
    }

    /// Names that shadow state inside this callable: parameters, return
    /// parameters and local variable declarations.
    fn collect_locals(declaration: Node<'_>, body: Node<'_>, source: &str) -> BTreeSet<String> {
        let mut locals = BTreeSet::new();
        for parameter in declaration.descendants_of_kinds(&["parameter"], false) {
            if let Some(identifier) = parameter.child_of_kind("identifier") {
                locals.insert(identifier.text_of(source));
            }
        }
        for declaration in body.descendants_of_kinds(&["variable_declaration"], false) {
            if let Some(identifier) = declaration.child_of_kind("identifier") {
                locals.insert(identifier.text_of(source));
            }
        }
        locals
    }

    /// Identifier references in the body, classified read/write. Property
    /// identifiers of member accesses (`msg.sender`'s `sender`) are skipped;
    /// resolution against actual state variable names happens later.
    fn collect_state_refs(
        body: Node<'_>,
        source: &str,
        locals: &BTreeSet<String>,
    ) -> Vec<RawStateRef> {
        let mut refs = vec![];

        // Writes: the base identifier of every assignment LHS / update
        // operand.
        for write in body.descendants_of_kinds(&WRITE_KINDS, true) {
            let Some(lhs) = write.named_child(0) else {
                continue;
            };
            if let Some(name) = Self::base_identifier(Self::unwrap_expression(lhs), source)
                && !locals.contains(&name)
            {
                refs.push(RawStateRef {
                    name,
                    write: true,
                    line: write.line_span().start_line,
                });
            }
        }

        // Reads: every remaining identifier that is not a member property.
        for identifier in body.descendants_of_kinds(&["identifier"], false) {
            let text = identifier.text_of(source);
            if locals.contains(&text) {
                continue;
            }
            if let Some(parent) = identifier.parent()
                && parent.kind() == "member_expression"
                && parent
                    .named_child(0)
                    .map(|first| first.id() != identifier.id())
                    .unwrap_or(false)
            {
                continue;
            }
            refs.push(RawStateRef {
                name: text,
                write: false,
                line: identifier.line_span().start_line,
            });
        }

        refs
    }

    /// The storage base of an lvalue: `balances[msg.sender].total` ->
    /// `balances`.
    fn base_identifier(node: Node<'_>, source: &str) -> Option<String> {
        let mut current = node;
        loop {
            match current.kind() {
                "identifier" => return Some(current.text_of(source)),
                "expression"
                | "array_access"
                | "member_expression"
                | "tuple_expression"
                | "parenthesized_expression" => {
                    current = current.named_child(0)?;
                }
                _ => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn extract(source: &str) -> FileExtraction {
        SolidityExtractor::extract(&SourceFile {
            relative: PathBuf::from("test.sol"),
            content: source.to_string(),
        })
        .expect("extract")
    }

    const SAMPLE: &str = r#"
pragma solidity ^0.8.0;
contract Counter is Ownable {
    uint256 public count;
    mapping(address => uint256) balances;
    modifier onlyPositive() { require(count > 0); _; }
    constructor(uint256 start) { count = start; }
    function increment() public onlyPositive {
        count += 1;
        balances[msg.sender] = count;
        helper();
        token.transfer(msg.sender, 1);
    }
    function helper() internal view returns (uint256) {
        uint256 local = count;
        return local;
    }
}
interface IToken {
    function transfer(address to, uint256 amount) external returns (bool);
}
"#;

    #[test]
    fn containers_states_and_inheritance_are_extracted() {
        let extraction = extract(SAMPLE);
        assert_eq!(extraction.parse_errors, 0);
        assert_eq!(extraction.modules.len(), 2);

        let counter = &extraction.modules[0];
        assert_eq!(counter.name, "Counter");
        assert_eq!(counter.kind, ModuleKind::Contract);
        assert_eq!(counter.parents, vec!["Ownable".to_string()]);
        let state_names: Vec<_> = counter.states.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(state_names, vec!["count", "balances"]);
        assert_eq!(counter.states[1].type_text, "mapping(address => uint256)");

        let token = &extraction.modules[1];
        assert_eq!(token.kind, ModuleKind::Interface);
        assert_eq!(token.callables.len(), 1);
        assert!(token.callables[0].signature.contains("function transfer"));
    }

    #[test]
    fn callables_carry_calls_and_modifier_invocations() {
        let extraction = extract(SAMPLE);
        let counter = &extraction.modules[0];
        let names: Vec<_> = counter.callables.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["onlyPositive", "constructor", "increment", "helper"]
        );

        let increment = &counter.callables[2];
        assert_eq!(increment.kind, CallableKind::Function);
        let call_names: Vec<_> = increment.calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(call_names, vec!["onlyPositive", "helper", "transfer"]);
        let transfer = &increment.calls[2];
        assert_eq!(transfer.qualifier.as_deref(), Some("token"));
        assert_eq!(transfer.text, "token.transfer");
    }

    #[test]
    fn state_refs_classify_reads_and_writes_and_skip_locals() {
        let extraction = extract(SAMPLE);
        let counter = &extraction.modules[0];

        let increment = &counter.callables[2];
        let writes: Vec<_> = increment
            .state_refs
            .iter()
            .filter(|r| r.write)
            .map(|r| r.name.as_str())
            .collect();
        assert!(writes.contains(&"count"), "{writes:?}");
        assert!(writes.contains(&"balances"), "{writes:?}");

        let helper = &counter.callables[3];
        assert!(
            helper
                .state_refs
                .iter()
                .any(|r| r.name == "count" && !r.write)
        );
        // `local` is shadowed by the declaration and never a state ref.
        assert!(!helper.state_refs.iter().any(|r| r.name == "local"));
        // `msg.sender`'s property never leaks in as a read.
        assert!(!increment.state_refs.iter().any(|r| r.name == "sender"));
    }
}
