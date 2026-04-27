// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract OmniShieldAudit {

    // ── Access control ─────────────────────────────────────────────────────────
    address public owner;

    modifier onlyOwner() {
        require(msg.sender == owner, "Unauthorized");
        _;
    }

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    constructor() {
        owner = msg.sender;
    }

    /// @notice Transfer contract ownership to a new wallet.
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Zero address");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    // ── Data model ─────────────────────────────────────────────────────────────
    struct AuditRecord {
        bytes32 documentHash;
        uint256 redactionCount;
        uint256 timestamp;
        address recordOwner;
        string  docType;
    }

    // documentHash => record
    mapping(bytes32 => AuditRecord) private _records;

    // wallet => ordered list of document hashes anchored by that wallet
    mapping(address => bytes32[]) private _ownerRecords;

    // user-supplied owner => document hashes (used when recordOwner != msg.sender)
    mapping(address => bytes32[]) private _userRecords;

    event AuditRecorded(
        bytes32 indexed documentHash,
        address indexed recordOwner,
        uint256 redactionCount,
        uint256 timestamp,
        string  docType
    );


    // ── Write ──────────────────────────────────────────────────────────────────

    /// @notice Anchor a redaction audit record on-chain.
    /// @dev Restricted to the contract owner (the deploying wallet).
    ///      Pass a non-zero recordOwner to attribute the record to a specific
    ///      user wallet (e.g. a MetaMask address) even though the transaction
    ///      is sent from the Ganache default account.
    function addRecord(
        bytes32 documentHash,
        uint256 redactionCount,
        string calldata docType,
        address recordOwner
    ) external onlyOwner {
        require(
            _records[documentHash].timestamp == 0,
            "Record already exists"
        );

        address effectiveOwner = recordOwner != address(0) ? recordOwner : msg.sender;

        AuditRecord storage r = _records[documentHash];
        r.documentHash   = documentHash;
        r.redactionCount = redactionCount;
        r.timestamp      = block.timestamp;
        r.recordOwner    = effectiveOwner;
        r.docType        = docType;

        _ownerRecords[msg.sender].push(documentHash);

        // Index by user-supplied owner so getRecordsByUser() works
        if (effectiveOwner != msg.sender) {
            _userRecords[effectiveOwner].push(documentHash);
        }

        emit AuditRecorded(
            documentHash, effectiveOwner, redactionCount, block.timestamp, docType
        );
    }

    // ── Read ───────────────────────────────────────────────────────────────────

    /// @notice Check whether a document hash has been recorded.
    /// @return exists         True if the hash is in the ledger.
    /// @return redactionCount Number of redactions recorded for this document.
    /// @return timestamp      Block timestamp when the record was written.
    function verify(bytes32 documentHash)
        external
        view
        returns (
            bool    exists,
            uint256 redactionCount,
            uint256 timestamp
        )
    {
        AuditRecord storage r = _records[documentHash];
        if (r.timestamp == 0) {
            return (false, 0, 0);
        }
        return (true, r.redactionCount, r.timestamp);
    }

    /// @notice Return all document hashes anchored by a given wallet (msg.sender index).
    function getRecordsByOwner(address _owner)
        external
        view
        returns (bytes32[] memory)
    {
        return _ownerRecords[_owner];
    }

    /// @notice Return all document hashes where recordOwner == user (user-supplied index).
    function getRecordsByUser(address user)
        external
        view
        returns (bytes32[] memory)
    {
        return _userRecords[user];
    }
}
