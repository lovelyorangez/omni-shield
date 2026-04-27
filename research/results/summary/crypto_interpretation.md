# Cryptographic Audit Trail Benchmarks

## Key numbers
| Operation          | Result      | Note                          |
|--------------------|-------------|-------------------------------|
| Blockchain verify  | 1.2ms       | O(1) mapping, near-instant    |
| ZK proof verify    | 22.1ms      | Groth16/BN128                 |
| ZK proof generate  | 20.3s       | Async background thread       |
| Gas per anchor     | 264,333     | addRecord() Solidity function |
| Cost — mainnet     | $13.22 USD  | Not viable at scale           |
| Cost — Polygon     | $0.006 USD  | Production deployment target  |
| Duplicate hashes   | 260 / 424   | Linkability vulnerability     |

## Deployment recommendation
Use Polygon L2 in production. One env var change from Ganache.
$0.006/anchor is viable for enterprise compliance workflows.

## Known vulnerability
SHA256 without salt = linkability attack.
Fix: hash = SHA256(document_bytes + timestamp + owner_address)
Status: documented as future work, not yet implemented.

## ZK proof notes
412 documents have valid Groth16/BN128 proofs.
1 unverified record predates ZK integration.
Proof generation runs async — does not block user response.
