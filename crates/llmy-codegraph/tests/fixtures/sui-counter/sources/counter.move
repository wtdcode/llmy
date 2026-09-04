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
        counter.value = next(counter.value);
    }

    public fun current(counter: &Counter): u64 {
        counter.value
    }

    fun next(v: u64): u64 {
        v + 1
    }
}
