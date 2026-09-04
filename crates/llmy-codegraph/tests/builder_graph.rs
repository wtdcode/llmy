//! End-to-end graph building over the fixture projects: discovery, dialect
//! detection, resolution, caching and the agent tools' rendering.

use std::path::PathBuf;

use llmy_codegraph::model::{AccessKind, CallableKind, CalleeRef, Language, StateKind};
use llmy_codegraph::{CodeGraphBuilder, CodeGraphStore, CodegraphContext};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[tokio::test]
async fn solidity_vault_graph_resolves_inheritance_calls_and_state() {
    let result = CodeGraphBuilder::new(fixture("solidity-vault"))
        .build()
        .await
        .expect("build");
    let graph = &result.graph;
    assert_eq!(result.total_parse_errors(), 0);

    // Vault inherits Ownable and IVault, both resolved in-project.
    let vault = graph.modules_by_name("Vault")[0];
    let parents: Vec<_> = graph
        .inherit_edges
        .iter()
        .filter(|e| e.module_id == vault.id)
        .collect();
    assert_eq!(parents.len(), 2);

    // withdraw -> nonReentrant modifier edge, resolved.
    let withdraw = graph.find_callables(Some("Vault"), "withdraw")[0];
    let outgoing = graph.outgoing_calls(withdraw.id);
    assert!(
        outgoing.iter().any(|e| {
            matches!(&e.callee, CalleeRef::Resolved(id)
                if graph.callables.get(id).map(|c| c.name == "nonReentrant").unwrap_or(false))
        }),
        "{outgoing:?}"
    );

    // withdraw writes balances and totalDeposits.
    let accesses = graph.state_accesses_of(withdraw.id);
    let writes: Vec<_> = accesses
        .iter()
        .filter(|e| e.access == AccessKind::Write)
        .filter_map(|e| graph.states.get(&e.state_id))
        .map(|s| s.name.as_str())
        .collect();
    assert!(writes.contains(&"balances"), "{writes:?}");
    assert!(writes.contains(&"totalDeposits"), "{writes:?}");

    // The inherited `owner` state variable is visible from Vault's sweep
    // (via onlyOwner it is not; but transferOwnership writes it in Ownable).
    let transfer_ownership = graph.find_callables(Some("Ownable"), "transferOwnership")[0];
    let ownable_writes: Vec<_> = graph
        .state_accesses_of(transfer_ownership.id)
        .into_iter()
        .filter(|e| e.access == AccessKind::Write)
        .filter_map(|e| graph.states.get(&e.state_id))
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(ownable_writes, vec!["owner"]);

    // lookup_state style query: who writes totalDeposits.
    let total = graph.find_states(Some("Vault"), "totalDeposits")[0];
    let writers: Vec<_> = graph
        .accessors_of_state(total.id)
        .into_iter()
        .filter(|e| e.access == AccessKind::Write)
        .map(|e| graph.render_callable_ref(e.callable_id))
        .collect();
    assert!(
        writers.contains(&"Vault.deposit".to_string()),
        "{writers:?}"
    );
    assert!(
        writers.contains(&"Vault.withdraw".to_string()),
        "{writers:?}"
    );
}

#[tokio::test]
async fn move_dialects_are_detected_from_manifests() {
    let aptos = CodeGraphBuilder::new(fixture("aptos-counter"))
        .build()
        .await
        .expect("build aptos");
    assert_eq!(aptos.total_parse_errors(), 0);
    assert!(
        aptos
            .files
            .iter()
            .all(|(_, language, _)| *language == Language::MoveAptos)
    );
    let counter_state = &aptos.graph.find_states(None, "Counter");
    assert_eq!(counter_state.len(), 1);
    assert_eq!(counter_state[0].kind, StateKind::MoveResource);

    let increment = aptos.graph.find_callables(None, "increment")[0];
    assert_eq!(increment.kind, CallableKind::Entry);
    assert!(
        aptos
            .graph
            .state_accesses_of(increment.id)
            .iter()
            .any(|e| e.access == AccessKind::Write)
    );
    // increment -> next resolved in-module.
    assert!(aptos.graph.outgoing_calls(increment.id).iter().any(|e| {
        matches!(&e.callee, CalleeRef::Resolved(id)
            if aptos.graph.callables.get(id).map(|c| c.name == "next").unwrap_or(false))
    }));

    let sui = CodeGraphBuilder::new(fixture("sui-counter"))
        .build()
        .await
        .expect("build sui");
    assert_eq!(sui.total_parse_errors(), 0);
    assert!(
        sui.files
            .iter()
            .all(|(_, language, _)| *language == Language::MoveSui)
    );
    let object = &sui.graph.find_states(None, "Counter");
    assert_eq!(object[0].kind, StateKind::SuiObject);

    let sui_increment = sui.graph.find_callables(None, "increment")[0];
    assert!(
        sui.graph
            .state_accesses_of(sui_increment.id)
            .iter()
            .any(|e| e.access == AccessKind::Write)
    );
    let current = sui.graph.find_callables(None, "current")[0];
    assert!(
        sui.graph
            .state_accesses_of(current.id)
            .iter()
            .all(|e| e.access == AccessKind::Read)
    );
}

#[tokio::test]
async fn rust_ecosystems_produce_entries_and_state_edges() {
    let cw = CodeGraphBuilder::new(fixture("cosmwasm-counter"))
        .build()
        .await
        .expect("build cosmwasm");
    assert_eq!(cw.total_parse_errors(), 0);
    let execute = cw.graph.find_callables(None, "execute")[0];
    assert_eq!(execute.kind, CallableKind::Entry);
    let state_names: Vec<_> = cw
        .graph
        .state_accesses_of(execute.id)
        .into_iter()
        .filter_map(|e| cw.graph.states.get(&e.state_id))
        .map(|s| s.name.as_str())
        .collect();
    assert!(state_names.contains(&"STATE"), "{state_names:?}");
    assert!(state_names.contains(&"SCORES"), "{state_names:?}");
    assert!(cw.graph.outgoing_calls(execute.id).iter().any(|e| {
        matches!(&e.callee, CalleeRef::Resolved(id)
            if cw.graph.callables.get(id).map(|c| c.name == "ensure_owner").unwrap_or(false))
    }));

    let anchor = CodeGraphBuilder::new(fixture("anchor-counter"))
        .build()
        .await
        .expect("build anchor");
    assert_eq!(anchor.total_parse_errors(), 0);
    let increment = anchor.graph.find_callables(None, "increment")[0];
    assert_eq!(increment.kind, CallableKind::Entry);
    let counter_writes: Vec<_> = anchor
        .graph
        .state_accesses_of(increment.id)
        .into_iter()
        .filter(|e| e.access == AccessKind::Write)
        .filter_map(|e| anchor.graph.states.get(&e.state_id))
        .map(|s| s.name.as_str())
        .collect();
    assert_eq!(counter_writes, vec!["Counter"]);
}

#[tokio::test]
async fn snapshot_cache_roundtrips_and_detects_staleness() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db_path = dir.path().join("codegraph.sqlite3");
    let store = CodeGraphStore::open(&db_path.display().to_string())
        .await
        .expect("open store");

    let builder = CodeGraphBuilder::new(fixture("solidity-vault"));
    let result = builder.build().await.expect("build");
    store.save("solidity-vault", &result).await.expect("save");

    let loaded = store
        .load_fresh("solidity-vault", &result.fingerprint)
        .await
        .expect("load")
        .expect("fresh hit");
    assert_eq!(loaded.counts(), result.graph.counts());

    let stale = store
        .load_fresh("solidity-vault", "different fingerprint")
        .await
        .expect("load");
    assert!(stale.is_none());
}

#[tokio::test]
async fn tools_render_overview_and_xrefs() {
    let result = CodeGraphBuilder::new(fixture("solidity-vault"))
        .build()
        .await
        .expect("build");
    let context = CodegraphContext::new(result.graph, fixture("solidity-vault"));

    let overview = context.render_overview();
    assert!(overview.contains("contract Vault"), "{overview}");
    assert!(overview.contains("interface IVault"), "{overview}");

    let tools = context.tool_box();
    let lookup = tools
        .invoke(
            "lookup_callable".to_string(),
            r#"{"name": "withdraw", "module": "Vault"}"#.to_string(),
        )
        .await
        .expect("tool exists")
        .expect("tool ran");
    assert!(lookup.contains("outgoing calls:"), "{lookup}");
    assert!(lookup.contains("nonReentrant"), "{lookup}");
    assert!(lookup.contains("write balances"), "{lookup}");

    let source = tools
        .invoke(
            "read_callable_source".to_string(),
            r#"{"name": "withdraw"}"#.to_string(),
        )
        .await
        .expect("tool exists")
        .expect("tool ran");
    assert!(source.contains("function withdraw"), "{source}");

    let module_source = tools
        .invoke(
            "read_module_source".to_string(),
            r#"{"module": "Ownable"}"#.to_string(),
        )
        .await
        .expect("tool exists")
        .expect("tool ran");
    assert!(
        module_source.contains("transferOwnership"),
        "{module_source}"
    );

    let state = tools
        .invoke(
            "lookup_state".to_string(),
            r#"{"name": "balances"}"#.to_string(),
        )
        .await
        .expect("tool exists")
        .expect("tool ran");
    assert!(state.contains("writers:"), "{state}");
    assert!(state.contains("Vault.deposit"), "{state}");
}
