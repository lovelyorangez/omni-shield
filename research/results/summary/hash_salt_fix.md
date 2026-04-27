# Hash Linkability Fix — Implemented

## Problem
SHA256 is deterministic — same document = same on-chain hash.
260/424 audit records were duplicates, enabling linkability attacks.

## Fix implemented
Salt formula: anchor_hash = SHA256(sha256_hex + owner_address + timestamp)

Changes made:
- blockchain_manager.py: anchor_record() and anchor_record_async()
  now accept salt=None, default = "{owner_address}:{unix_timestamp}"
  On-chain value = SHA256(sha256_hex + salt), not raw sha256_hex

- backend_server.py: All 6 redaction endpoints now compute
  anchor_salt and anchor_hash before blockchain call.
  Both stored in audit JSON alongside original sha256_hex.

- /verify endpoint: Two-step lookup:
  1. Find audit JSON by sha256_hex (user-verifiable)
  2. Extract anchor_hash (salted) and verify that on-chain
  Falls back to direct hash for old pre-fix records.

## Proof of fix
Same document processed twice:
  sha256_hex  : a980d05ebd5c42f4... (identical — same file)
  anchor_hash1: 3f49faf395f4eb0e... (unique)
  anchor_hash2: a90374e74d4bc5a8... (unique)

## Audit JSON structure (after fix)
{
  "sha256_hex":   "a980d05e...",  // original — user verifiable
  "anchor_hash":  "3f49faf3...",  // on-chain — unlinkable
  "anchor_salt":  "0xOwner:1712345678",  // stored for verify
}

## Paper text update for §6
"To prevent linkability attacks — where an adversary 
monitoring the blockchain could detect repeated processing
of the same document — each anchor hash is salted with the
owner address and a Unix timestamp before anchoring:
anchor_hash = SHA256(sha256_hex || owner_address || timestamp).
The original SHA256 hash is preserved in the audit record
for user-side verification."
