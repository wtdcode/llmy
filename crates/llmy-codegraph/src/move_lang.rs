//! Move extraction, one extractor per dialect: Aptos Move (vendored
//! aptos-labs grammar; global storage accessed via `move_to` /
//! `borrow_global*`) and Sui Move (vendored tzakian grammar; objects with
//! `key` flowing through function parameters). The dialects differ enough —
//! syntax and storage model both — that sharing one extractor would hide
//! more than it saves.

use llmy_types::error::LLMYError;
use tree_sitter::Node;

use crate::extract::{
    FileExtraction, GrammarSet, NodeUtil, RawCallSite, RawCallable, RawModule, RawState,
    RawStateRef, SourceFile,
};
use crate::model::{CallableKind, Language, ModuleKind, StateKind};

pub struct MoveAptosExtractor;

impl MoveAptosExtractor {
    pub fn extract(file: &SourceFile) -> Result<FileExtraction, LLMYError> {
        let tree = GrammarSet::parse(Language::MoveAptos, &file.content)?;
        let root = tree.root_node();
        let parse_errors = GrammarSet::count_errors(root);
        let source = &file.content;

        let modules = root
            .descendants_of_kinds(&["module_declaration"], false)
            .into_iter()
            .map(|node| Self::extract_module(node, source))
            .collect();

        Ok(FileExtraction {
            file: file.relative.clone(),
            language: Language::MoveAptos,
            modules,
            parse_errors,
        })
    }

    fn extract_module(node: Node<'_>, source: &str) -> RawModule {
        let name = node
            .child_of_kind("module_identity")
            .and_then(|identity| {
                identity
                    .children_of_kind("identifier")
                    .last()
                    .map(|n| n.text_of(source))
            })
            .unwrap_or_else(|| "<module>".to_string());

        let mut states = vec![];
        let mut callables = vec![];
        if let Some(body) = node.child_of_kind("module_body") {
            for declaration in body.children_of_kind("struct_declaration") {
                let Some(struct_name) = declaration
                    .child_of_kind("identifier")
                    .map(|n| n.text_of(source))
                else {
                    continue;
                };
                let abilities: Vec<String> = declaration
                    .descendants_of_kinds(&["ability"], false)
                    .into_iter()
                    .map(|a| a.text_of(source))
                    .collect();
                if !abilities.iter().any(|a| a == "key") {
                    continue;
                }
                states.push(RawState {
                    name: struct_name,
                    kind: StateKind::MoveResource,
                    type_text: format!("struct has {}", abilities.join(", ")),
                    span: declaration.line_span(),
                });
            }

            for function in body.children_of_kind("function_declaration") {
                callables.push(Self::extract_function(function, source));
            }
        }

        RawModule {
            name,
            kind: ModuleKind::Module,
            span: node.line_span(),
            parents: vec![],
            callables,
            states,
        }
    }

    fn extract_function(function: Node<'_>, source: &str) -> RawCallable {
        let name = function
            .child_of_kind("identifier")
            .map(|n| n.text_of(source))
            .unwrap_or_else(|| "<function>".to_string());
        let kind = if function.child_of_kind("entry_modifier").is_some() {
            CallableKind::Entry
        } else {
            CallableKind::Function
        };
        let signature = function.signature_head(&["block"], source);

        let mut calls = vec![];
        let mut state_refs = vec![];
        if let Some(body) = function.child_of_kind("block") {
            for call in body.descendants_of_kinds(&["call_expression"], true) {
                let Some(chain) = call.child_of_kind("name_access_chain") else {
                    continue;
                };
                let parts: Vec<String> = chain
                    .children_of_kind("identifier")
                    .into_iter()
                    .map(|n| n.text_of(source))
                    .collect();
                let Some(call_name) = parts.last().cloned() else {
                    continue;
                };
                let line = call.line_span().start_line;

                // Global storage intrinsics become state references on the
                // resource named in the type argument.
                let storage_write = match call_name.as_str() {
                    "move_to" | "move_from" | "borrow_global_mut" => Some(true),
                    "borrow_global" | "exists" => Some(false),
                    _ => None,
                };
                if let Some(write) = storage_write {
                    if let Some(resource) = call
                        .child_of_kind("type_arguments")
                        .and_then(|args| args.first_identifier(source))
                    {
                        state_refs.push(RawStateRef {
                            name: resource,
                            write,
                            line,
                        });
                    } else if call_name == "move_to" {
                        // `move_to(account, Counter { .. })` — the resource is
                        // the packed struct in the second argument.
                        if let Some(packed) = call.child_of_kind("arg_list").and_then(|args| {
                            args.descendants_of_kinds(&["pack_expression"], false)
                                .first()
                                .and_then(|p| p.first_identifier(source))
                        }) {
                            state_refs.push(RawStateRef {
                                name: packed,
                                write: true,
                                line,
                            });
                        }
                    }
                    continue;
                }

                let qualifier = (parts.len() > 1).then(|| parts[0].clone());
                calls.push(RawCallSite {
                    text: chain.text_of(source),
                    name: call_name,
                    qualifier,
                    line,
                });
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

pub struct MoveSuiExtractor;

impl MoveSuiExtractor {
    pub fn extract(file: &SourceFile) -> Result<FileExtraction, LLMYError> {
        let tree = GrammarSet::parse(Language::MoveSui, &file.content)?;
        let root = tree.root_node();
        let parse_errors = GrammarSet::count_errors(root);
        let source = &file.content;

        let modules = root
            .descendants_of_kinds(&["module_definition"], false)
            .into_iter()
            .map(|node| Self::extract_module(node, source))
            .collect();

        Ok(FileExtraction {
            file: file.relative.clone(),
            language: Language::MoveSui,
            modules,
            parse_errors,
        })
    }

    fn extract_module(node: Node<'_>, source: &str) -> RawModule {
        let name = node
            .child_of_kind("module_identity")
            .and_then(|identity| {
                identity
                    .children_of_kind("module_identifier")
                    .last()
                    .map(|n| n.text_of(source))
            })
            .unwrap_or_else(|| "<module>".to_string());

        let mut states = vec![];
        let mut callables = vec![];
        if let Some(body) = node.child_of_kind("module_body") {
            for declaration in body.children_of_kind("struct_definition") {
                let Some(struct_name) = declaration
                    .child_of_kind("struct_identifier")
                    .map(|n| n.text_of(source))
                else {
                    continue;
                };
                let abilities: Vec<String> = declaration
                    .descendants_of_kinds(&["ability"], false)
                    .into_iter()
                    .map(|a| a.text_of(source))
                    .collect();
                if !abilities.iter().any(|a| a == "key") {
                    continue;
                }
                states.push(RawState {
                    name: struct_name,
                    kind: StateKind::SuiObject,
                    type_text: format!("struct has {}", abilities.join(", ")),
                    span: declaration.line_span(),
                });
            }

            for function in body.descendants_of_kinds(
                &[
                    "function_definition",
                    "native_function_definition",
                    "macro_function_definition",
                ],
                false,
            ) {
                callables.push(Self::extract_function(function, source));
            }
        }

        RawModule {
            name,
            kind: ModuleKind::Module,
            span: node.line_span(),
            parents: vec![],
            callables,
            states,
        }
    }

    fn extract_function(function: Node<'_>, source: &str) -> RawCallable {
        let name = function
            .child_of_kind("function_identifier")
            .map(|n| n.text_of(source))
            .unwrap_or_else(|| "<function>".to_string());
        let is_entry = function
            .children_of_kind("modifier")
            .into_iter()
            .any(|m| m.text_of(source) == "entry");
        let kind = if is_entry {
            CallableKind::Entry
        } else {
            CallableKind::Function
        };
        let signature = function.signature_head(&["block"], source);

        // Objects flow through parameters: `&mut T` (or by value) writes,
        // `&T` reads. Non-object type names simply resolve to nothing later.
        let mut state_refs = vec![];
        if let Some(parameters) = function.child_of_kind("function_parameters") {
            for parameter in parameters.children_of_kind("function_parameter") {
                let mut cursor = parameter.walk();
                let Some(type_node) = parameter
                    .children(&mut cursor)
                    .filter(|c| c.is_named())
                    .nth(1)
                else {
                    continue;
                };
                let line = parameter.line_span().start_line;
                match type_node.kind() {
                    "ref_type" => {
                        let mutable = type_node.child_of_kind("mut_ref").is_some();
                        if let Some(type_name) = type_node
                            .child_of_kind("apply_type")
                            .and_then(|t| Self::last_identifier(t, source))
                        {
                            state_refs.push(RawStateRef {
                                name: type_name,
                                write: mutable,
                                line,
                            });
                        }
                    }
                    "apply_type" => {
                        if let Some(type_name) = Self::last_identifier(type_node, source) {
                            state_refs.push(RawStateRef {
                                name: type_name,
                                write: true,
                                line,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut calls = vec![];
        if let Some(body) = function.child_of_kind("block") {
            for call in body.descendants_of_kinds(&["call_expression"], true) {
                let Some(access) = call
                    .child_of_kind("name_expression")
                    .and_then(|n| n.child_of_kind("module_access"))
                else {
                    continue;
                };
                let mut cursor = access.walk();
                let parts: Vec<(String, String)> = access
                    .children(&mut cursor)
                    .filter(|c| c.is_named())
                    .map(|c| (c.kind().to_string(), c.text_of(source)))
                    .collect();
                let Some((_, call_name)) = parts
                    .iter()
                    .rev()
                    .find(|(kind, _)| kind == "identifier")
                    .cloned()
                else {
                    continue;
                };
                let qualifier = parts
                    .iter()
                    .find(|(kind, _)| kind == "module_identifier")
                    .map(|(_, text)| text.clone());
                calls.push(RawCallSite {
                    text: access.text_of(source),
                    name: call_name,
                    qualifier,
                    line: call.line_span().start_line,
                });
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

    fn last_identifier(node: Node<'_>, source: &str) -> Option<String> {
        node.descendants_of_kinds(&["identifier"], false)
            .last()
            .map(|n| n.text_of(source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const APTOS: &str = r#"
module counter_addr::counter {
    use std::signer;

    struct Counter has key {
        value: u64,
    }

    public entry fun initialize(account: &signer) {
        move_to(account, Counter { value: 0 });
    }

    public entry fun increment(account: &signer) acquires Counter {
        let counter = borrow_global_mut<Counter>(signer::address_of(account));
        counter.value = counter.value + 1;
        helper(counter.value);
    }

    fun helper(v: u64): u64 {
        let snapshot = borrow_global<Counter>(@counter_addr);
        v + snapshot.value
    }
}
"#;

    const SUI: &str = r#"
module counter::counter {
    use sui::transfer;

    public struct Counter has key {
        id: UID,
        value: u64,
    }

    public entry fun create(ctx: &mut TxContext) {
        let counter = Counter { id: object::new(ctx), value: 0 };
        transfer::share_object(counter);
    }

    public entry fun increment(counter: &mut Counter) {
        counter.value = counter.value + 1;
        helper(counter);
    }

    fun helper(counter: &Counter): u64 {
        counter.value
    }
}
"#;

    #[test]
    fn aptos_resources_calls_and_storage_ops_are_extracted() {
        let extraction = MoveAptosExtractor::extract(&SourceFile {
            relative: PathBuf::from("sources/counter.move"),
            content: APTOS.to_string(),
        })
        .expect("extract");
        assert_eq!(extraction.parse_errors, 0);

        let module = &extraction.modules[0];
        assert_eq!(module.name, "counter");
        assert_eq!(module.states.len(), 1);
        assert_eq!(module.states[0].name, "Counter");
        assert_eq!(module.states[0].kind, StateKind::MoveResource);

        let names: Vec<_> = module.callables.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["initialize", "increment", "helper"]);
        assert_eq!(module.callables[0].kind, CallableKind::Entry);
        assert_eq!(module.callables[2].kind, CallableKind::Function);

        // move_to without type args still resolves via the packed struct.
        let initialize = &module.callables[0];
        assert!(
            initialize
                .state_refs
                .iter()
                .any(|r| r.name == "Counter" && r.write)
        );

        let increment = &module.callables[1];
        assert!(
            increment
                .state_refs
                .iter()
                .any(|r| r.name == "Counter" && r.write)
        );
        assert!(increment.calls.iter().any(|c| c.name == "helper"));
        assert!(
            increment
                .calls
                .iter()
                .any(|c| c.name == "address_of" && c.qualifier.as_deref() == Some("signer"))
        );
        // Storage intrinsics never appear as call edges.
        assert!(
            !increment
                .calls
                .iter()
                .any(|c| c.name == "borrow_global_mut")
        );

        let helper = &module.callables[2];
        assert!(
            helper
                .state_refs
                .iter()
                .any(|r| r.name == "Counter" && !r.write)
        );
    }

    #[test]
    fn sui_objects_flow_through_parameters() {
        let extraction = MoveSuiExtractor::extract(&SourceFile {
            relative: PathBuf::from("sources/counter.move"),
            content: SUI.to_string(),
        })
        .expect("extract");
        assert_eq!(extraction.parse_errors, 0);

        let module = &extraction.modules[0];
        assert_eq!(module.name, "counter");
        assert_eq!(module.states.len(), 1);
        assert_eq!(module.states[0].kind, StateKind::SuiObject);

        let names: Vec<_> = module.callables.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["create", "increment", "helper"]);
        assert_eq!(module.callables[1].kind, CallableKind::Entry);

        let increment = &module.callables[1];
        assert!(
            increment
                .state_refs
                .iter()
                .any(|r| r.name == "Counter" && r.write)
        );
        assert!(increment.calls.iter().any(|c| c.name == "helper"));

        let helper = &module.callables[2];
        assert!(
            helper
                .state_refs
                .iter()
                .any(|r| r.name == "Counter" && !r.write)
        );

        let create = &module.callables[0];
        assert!(
            create
                .calls
                .iter()
                .any(|c| c.name == "share_object" && c.qualifier.as_deref() == Some("transfer"))
        );
    }
}
