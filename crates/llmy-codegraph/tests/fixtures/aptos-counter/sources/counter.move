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
        counter.value = next(counter.value);
    }

    public fun current(addr: address): u64 acquires Counter {
        borrow_global<Counter>(addr).value
    }

    fun next(v: u64): u64 {
        v + 1
    }
}
