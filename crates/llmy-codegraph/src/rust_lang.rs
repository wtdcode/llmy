//! Rust contract extraction over tree-sitter-rust. One module per file.
//! State is recognized for the two supported contract ecosystems:
//! Anchor (`#[account]` structs, expanded through `Context<T>` /
//! `#[derive(Accounts)]` containers) and CosmWasm (`Item` / `Map` storage
//! declarations with their `save`/`load`-style accesses). Plain Rust still
//! gets modules, functions and call edges.

use std::collections::BTreeMap;

use llmy_types::error::LLMYError;
use tree_sitter::Node;

use crate::extract::{
    FileExtraction, GrammarSet, NodeUtil, RawCallSite, RawCallable, RawModule, RawState,
    RawStateRef, SourceFile,
};
use crate::model::{CallableKind, Language, ModuleKind, StateKind};

const CW_WRITE_METHODS: [&str; 4] = ["save", "update", "remove", "replace"];
const CW_READ_METHODS: [&str; 7] = [
    "load", "may_load", "has", "query", "range", "keys", "prefix",
];

/// One field of an Anchor `#[derive(Accounts)]` container: the inner account
/// type plus whether the field is `#[account(mut)]`.
#[derive(Debug, Clone)]
struct AccountsField {
    type_name: String,
    mutable: bool,
}

pub struct RustExtractor;

impl RustExtractor {
    pub fn extract(file: &SourceFile) -> Result<FileExtraction, LLMYError> {
        let tree = GrammarSet::parse(Language::Rust, &file.content)?;
        let root = tree.root_node();
        let parse_errors = GrammarSet::count_errors(root);
        let source = &file.content;

        let module_name = file
            .relative
            .with_extension("")
            .display()
            .to_string()
            .replace('\\', "/");

        let mut states = vec![];
        // Accounts containers: name -> fields, used to expand Context<T>.
        let mut containers: BTreeMap<String, Vec<AccountsField>> = BTreeMap::new();

        for item in root.descendants_of_kinds(&["struct_item", "const_item", "static_item"], false)
        {
            match item.kind() {
                "struct_item" => {
                    let Some(name) = item
                        .child_of_kind("type_identifier")
                        .map(|n| n.text_of(source))
                    else {
                        continue;
                    };
                    let attributes = Self::preceding_attributes(item, source);
                    if attributes.iter().any(|a| a == "account") {
                        states.push(RawState {
                            name,
                            kind: StateKind::AnchorAccount,
                            type_text: "#[account] struct".to_string(),
                            span: item.line_span(),
                        });
                    } else if attributes.iter().any(|a| a.contains("Accounts")) {
                        containers.insert(name, Self::container_fields(item, source));
                    }
                }
                _ => {
                    let Some(name) = item.child_of_kind("identifier").map(|n| n.text_of(source))
                    else {
                        continue;
                    };
                    let Some(type_node) = item.child_of_kind("generic_type") else {
                        continue;
                    };
                    let head = type_node
                        .child_of_kind("type_identifier")
                        .map(|n| n.text_of(source))
                        .unwrap_or_default();
                    let kind = match head.as_str() {
                        "Item" => StateKind::CwItem,
                        "Map" => StateKind::CwMap,
                        _ => continue,
                    };
                    states.push(RawState {
                        name,
                        kind,
                        type_text: type_node.text_of(source),
                        span: item.line_span(),
                    });
                }
            }
        }

        // Modules under a #[program] attribute mark Anchor instruction
        // handlers.
        let program_mods: Vec<Node<'_>> = root
            .descendants_of_kinds(&["mod_item"], true)
            .into_iter()
            .filter(|m| {
                Self::preceding_attributes(*m, source)
                    .iter()
                    .any(|a| a == "program")
            })
            .collect();

        let cw_state_names: Vec<String> = states
            .iter()
            .filter(|s| matches!(s.kind, StateKind::CwItem | StateKind::CwMap))
            .map(|s| s.name.clone())
            .collect();

        let mut callables = vec![];
        for function in root.descendants_of_kinds(&["function_item"], false) {
            callables.push(Self::extract_function(
                function,
                source,
                &program_mods,
                &containers,
                &cw_state_names,
            ));
        }

        Ok(FileExtraction {
            file: file.relative.clone(),
            language: Language::Rust,
            modules: vec![RawModule {
                name: module_name,
                kind: ModuleKind::Module,
                span: root.line_span(),
                parents: vec![],
                callables,
                states,
            }],
            parse_errors,
        })
    }

    /// The attribute names written directly above an item (`#[account]` ->
    /// "account", `#[derive(Accounts)]` -> "derive(Accounts)").
    fn preceding_attributes(item: Node<'_>, source: &str) -> Vec<String> {
        let mut out = vec![];
        let mut current = item;
        while let Some(previous) = current.prev_named_sibling() {
            if previous.kind() != "attribute_item" {
                break;
            }
            if let Some(attribute) = previous.child_of_kind("attribute") {
                out.push(attribute.text_of(source));
            }
            current = previous;
        }
        out
    }

    fn container_fields(item: Node<'_>, source: &str) -> Vec<AccountsField> {
        let Some(fields) = item.child_of_kind("field_declaration_list") else {
            return vec![];
        };
        let mut out = vec![];
        for field in fields.children_of_kind("field_declaration") {
            let mutable = Self::preceding_attributes(field, source)
                .iter()
                .any(|a| a.starts_with("account") && a.contains("mut"));
            // The inner account type is the last type_identifier of the
            // field's type (e.g. `Account<'info, Counter>` -> `Counter`).
            let type_names = field.descendants_of_kinds(&["type_identifier"], true);
            if let Some(inner) = type_names.last() {
                out.push(AccountsField {
                    type_name: inner.text_of(source),
                    mutable,
                });
            }
        }
        out
    }

    fn extract_function(
        function: Node<'_>,
        source: &str,
        program_mods: &[Node<'_>],
        containers: &BTreeMap<String, Vec<AccountsField>>,
        cw_state_names: &[String],
    ) -> RawCallable {
        let name = function
            .child_of_kind("identifier")
            .map(|n| n.text_of(source))
            .unwrap_or_else(|| "<function>".to_string());
        let signature = function.signature_head(&["block"], source);

        let attributes = Self::preceding_attributes(function, source);
        let in_program_mod = program_mods.iter().any(|m| {
            m.start_byte() <= function.start_byte() && function.end_byte() <= m.end_byte()
        });
        let kind = if in_program_mod || attributes.iter().any(|a| a.contains("entry_point")) {
            CallableKind::Entry
        } else {
            CallableKind::Function
        };

        let mut calls = vec![];
        let mut state_refs = vec![];

        // Context<T> parameters expand through the Accounts container into
        // per-account state references.
        if let Some(parameters) = function.child_of_kind("parameters") {
            for generic in parameters.descendants_of_kinds(&["generic_type"], true) {
                let head = generic
                    .child_of_kind("type_identifier")
                    .map(|n| n.text_of(source))
                    .unwrap_or_default();
                if head != "Context" {
                    continue;
                }
                let Some(arguments) = generic.child_of_kind("type_arguments") else {
                    continue;
                };
                let Some(container_name) = arguments
                    .descendants_of_kinds(&["type_identifier"], false)
                    .first()
                    .map(|n| n.text_of(source))
                else {
                    continue;
                };
                for field in containers.get(&container_name).into_iter().flatten() {
                    state_refs.push(RawStateRef {
                        name: field.type_name.clone(),
                        write: field.mutable,
                        line: function.line_span().start_line,
                    });
                }
            }
        }

        if let Some(body) = function.child_of_kind("block") {
            for call in body.descendants_of_kinds(&["call_expression"], true) {
                let Some(target) = call.named_child(0) else {
                    continue;
                };
                let line = call.line_span().start_line;
                match target.kind() {
                    "identifier" => {
                        let call_name = target.text_of(source);
                        calls.push(RawCallSite {
                            text: call_name.clone(),
                            name: call_name,
                            qualifier: None,
                            line,
                        });
                    }
                    "scoped_identifier" => {
                        let mut cursor = target.walk();
                        let parts: Vec<String> = target
                            .children(&mut cursor)
                            .filter(|c| c.is_named())
                            .map(|c| c.text_of(source))
                            .collect();
                        let Some(call_name) = parts.last().cloned() else {
                            continue;
                        };
                        let qualifier = (parts.len() > 1).then(|| parts[0].clone());
                        calls.push(RawCallSite {
                            text: target.text_of(source),
                            name: call_name,
                            qualifier,
                            line,
                        });
                    }
                    "field_expression" => {
                        let Some(method) = target
                            .child_of_kind("field_identifier")
                            .map(|n| n.text_of(source))
                        else {
                            continue;
                        };
                        let receiver = target
                            .named_child(0)
                            .map(|n| n.text_of(source))
                            .unwrap_or_default();

                        // CosmWasm storage access: STATE.save(...) etc.
                        if cw_state_names.contains(&receiver) {
                            if CW_WRITE_METHODS.contains(&method.as_str()) {
                                state_refs.push(RawStateRef {
                                    name: receiver.clone(),
                                    write: true,
                                    line,
                                });
                                continue;
                            }
                            if CW_READ_METHODS.contains(&method.as_str()) {
                                state_refs.push(RawStateRef {
                                    name: receiver.clone(),
                                    write: false,
                                    line,
                                });
                                continue;
                            }
                        }

                        calls.push(RawCallSite {
                            text: target.text_of(source),
                            name: method,
                            qualifier: Some(receiver),
                            line,
                        });
                    }
                    _ => {}
                }
            }
        }

        RawCallable {
            name,
            kind,
            signature,
            span: function.line_span(),
            calls,
            state_refs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn extract(source: &str) -> FileExtraction {
        RustExtractor::extract(&SourceFile {
            relative: PathBuf::from("src/lib.rs"),
            content: source.to_string(),
        })
        .expect("extract")
    }

    const COSMWASM: &str = r#"
use cw_storage_plus::{Item, Map};
const STATE: Item<State> = Item::new("state");
const BALANCES: Map<&Addr, u128> = Map::new("balances");

#[entry_point]
pub fn execute(deps: DepsMut) -> Result<Response, ContractError> {
    let mut state = STATE.load(deps.storage)?;
    state.count += 1;
    STATE.save(deps.storage, &state)?;
    helper(&state);
    Ok(Response::new())
}

fn helper(state: &State) -> u64 { state.count }
"#;

    const ANCHOR: &str = r#"
#[program]
pub mod counter {
    use super::*;
    pub fn increment(ctx: Context<Increment>) -> Result<()> {
        ctx.accounts.counter.count += 1;
        Ok(())
    }
}

#[account]
pub struct Counter { pub count: u64 }

#[derive(Accounts)]
pub struct Increment<'info> {
    #[account(mut)]
    pub counter: Account<'info, Counter>,
    pub user: Signer<'info>,
}
"#;

    #[test]
    fn cosmwasm_state_and_accesses_are_extracted() {
        let extraction = extract(COSMWASM);
        assert_eq!(extraction.parse_errors, 0);
        let module = &extraction.modules[0];
        assert_eq!(module.name, "src/lib");

        let kinds: Vec<_> = module
            .states
            .iter()
            .map(|s| (s.name.as_str(), s.kind))
            .collect();
        assert_eq!(
            kinds,
            vec![("STATE", StateKind::CwItem), ("BALANCES", StateKind::CwMap)]
        );

        let execute = &module.callables[0];
        assert_eq!(execute.kind, CallableKind::Entry);
        let refs: Vec<_> = execute
            .state_refs
            .iter()
            .map(|r| (r.name.as_str(), r.write))
            .collect();
        assert!(refs.contains(&("STATE", false)), "{refs:?}");
        assert!(refs.contains(&("STATE", true)), "{refs:?}");
        assert!(execute.calls.iter().any(|c| c.name == "helper"));
        // Storage method calls became state refs, not call edges.
        assert!(!execute.calls.iter().any(|c| c.name == "save"));
    }

    #[test]
    fn anchor_accounts_expand_through_context() {
        let extraction = extract(ANCHOR);
        assert_eq!(extraction.parse_errors, 0);
        let module = &extraction.modules[0];

        assert_eq!(module.states.len(), 1);
        assert_eq!(module.states[0].name, "Counter");
        assert_eq!(module.states[0].kind, StateKind::AnchorAccount);

        let increment = &module.callables[0];
        assert_eq!(increment.name, "increment");
        assert_eq!(increment.kind, CallableKind::Entry);
        let refs: Vec<_> = increment
            .state_refs
            .iter()
            .map(|r| (r.name.as_str(), r.write))
            .collect();
        assert!(refs.contains(&("Counter", true)), "{refs:?}");
        // Signer is also expanded but resolves to no state item later.
        assert!(refs.contains(&("Signer", false)), "{refs:?}");
    }
}
