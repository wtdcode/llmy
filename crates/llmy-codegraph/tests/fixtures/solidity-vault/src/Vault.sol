// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Ownable} from "./Base.sol";
import {IVault} from "./IVault.sol";

contract Vault is Ownable, IVault {
    mapping(address => uint256) public balances;
    uint256 public totalDeposits;
    bool private locked;

    modifier nonReentrant() {
        require(!locked, "reentrant");
        locked = true;
        _;
        locked = false;
    }

    function deposit() external payable {
        balances[msg.sender] += msg.value;
        totalDeposits += msg.value;
    }

    function withdraw(uint256 amount) external nonReentrant {
        require(balances[msg.sender] >= amount, "insufficient");
        (bool ok, ) = msg.sender.call{value: amount}("");
        require(ok, "transfer failed");
        balances[msg.sender] -= amount;
        totalDeposits -= amount;
    }

    function sweep(address token) external onlyOwner {
        uint256 amount = totalDeposits;
        emit Swept(token, amount);
    }

    event Swept(address token, uint256 amount);
}
