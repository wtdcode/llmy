// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IToken {
    function transfer(address to, uint256 amount) external returns (bool);
    function balanceOf(address who) external view returns (uint256);
}

interface IPool {
    function settle(address who, uint256 amount) external payable;
}

abstract contract Base {
    uint256 public balance;
    mapping(address => uint256) public owed;

    function _transfer(address to, uint256 amount) internal virtual;

    function _withdraw(address to, uint256 amount) internal {
        _transfer(to, amount);
        balance = balance - amount;
    }
}

contract Impl is Base {
    function _transfer(address to, uint256 amount) internal override {
        (bool ok, ) = to.call{value: amount}("");
        require(ok, "transfer failed");
    }

    function exit(uint256 amount) external {
        _withdraw(msg.sender, amount);
    }
}

contract Vault {
    struct Position { uint256 size; uint256 debt; }

    address public owner;
    address public token;
    IPool public pool;
    uint256 public total;
    mapping(address => Position) internal positions;

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor(address token_, IPool pool_) {
        owner = msg.sender;
        token = token_;
        pool = pool_;
    }

    function deposit(uint256 amount) external payable {
        Position storage p = positions[msg.sender];
        p.size += amount;
        total += amount;
        pool.settle{value: msg.value}(msg.sender, amount);
        IToken(token).transfer(address(this), amount);
    }

    function held() external view returns (uint256) {
        return IToken(token).balanceOf(address(this));
    }

    function sweep(address to) external onlyOwner {
        uint256 amount = total;
        total = 0;
        assembly {
            let ok := call(gas(), to, amount, 0, 0, 0, 0)
            if iszero(ok) { revert(0, 0) }
        }
    }
}
