//! Grammar exploration helper: dumps parse trees of small samples so the
//! extractors can be written against the real node kinds. Run explicitly:
//! `cargo test -p llmy-codegraph --test dump_trees -- --ignored --nocapture`

use llmy_codegraph::extract::GrammarSet;
use llmy_codegraph::model::Language;

fn dump(language: Language, source: &str) {
    let tree = GrammarSet::parse(language, source).expect("parse");
    println!("===== {} =====", language.render());
    println!("{}", GrammarSet::dump(tree.root_node(), source, 0));
}

#[test]
#[ignore]
fn dump_solidity() {
    dump(
        Language::Solidity,
        r#"
pragma solidity ^0.8.0;
contract Counter is Ownable {
    uint256 public count;
    mapping(address => uint256) balances;
    modifier onlyPositive() { require(count > 0); _; }
    constructor(uint256 start) { count = start; }
    function increment() public onlyPositive {
        count += 1;
        balances[msg.sender] = count;
        emit Incremented(count);
        helper();
        token.transfer(msg.sender, 1);
    }
    function helper() internal view returns (uint256) {
        uint256 local = count;
        return local;
    }
}
"#,
    );
}

#[test]
#[ignore]
fn dump_rust() {
    dump(
        Language::Rust,
        r#"
use cw_storage_plus::{Item, Map};
const STATE: Item<State> = Item::new("state");
const BALANCES: Map<&Addr, u128> = Map::new("balances");

#[entry_point]
pub fn execute(deps: DepsMut, env: Env, info: MessageInfo, msg: ExecuteMsg) -> Result<Response, ContractError> {
    let mut state = STATE.load(deps.storage)?;
    state.count += 1;
    STATE.save(deps.storage, &state)?;
    helper(&state);
    Ok(Response::new())
}

fn helper(state: &State) -> u64 { state.count }

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
"#,
    );
}

#[test]
#[ignore]
fn dump_move_aptos() {
    dump(
        Language::MoveAptos,
        r#"
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
"#,
    );
}

#[test]
#[ignore]
fn dump_move_sui() {
    dump(
        Language::MoveSui,
        r#"
module counter::counter {
    use sui::object::{Self, UID};
    use sui::transfer;
    use sui::tx_context::TxContext;

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
"#,
    );
}
