// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract OmniShieldAudit {
    
    struct Log {
        uint256 timestamp;
        string actionType; // "AUDIO_MUTE" or "VIDEO_BLUR"
        string dataHash;   // SHA-256 hash of the redacted segment (simulated)
    }

    mapping(address => Log[]) public audits;
    event NewAuditEntry(address indexed user, uint256 timestamp, string actionType);

    function addLog(string memory _actionType, string memory _dataHash) public {
        audits[msg.sender].push(Log(block.timestamp, _actionType, _dataHash));
        emit NewAuditEntry(msg.sender, block.timestamp, _actionType);
    }

    function getLogCount() public view returns (uint256) {
        return audits[msg.sender].length;
    }
}