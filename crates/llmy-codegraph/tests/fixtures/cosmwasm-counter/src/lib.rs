use cosmwasm_std::{DepsMut, Env, MessageInfo, Response, StdResult};
use cw_storage_plus::{Item, Map};

pub struct State {
    pub count: u64,
    pub owner: String,
}

pub const STATE: Item<State> = Item::new("state");
pub const SCORES: Map<&str, u64> = Map::new("scores");

#[entry_point]
pub fn execute(deps: DepsMut, _env: Env, info: MessageInfo) -> StdResult<Response> {
    let mut state = STATE.load(deps.storage)?;
    ensure_owner(&state, &info)?;
    state.count += 1;
    STATE.save(deps.storage, &state)?;
    SCORES.save(deps.storage, info.sender.as_str(), &state.count)?;
    Ok(Response::new())
}

fn ensure_owner(state: &State, info: &MessageInfo) -> StdResult<()> {
    if state.owner != info.sender.as_str() {
        panic!("not owner");
    }
    Ok(())
}
