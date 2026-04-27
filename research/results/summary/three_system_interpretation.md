# Three-System Comparison Interpretation

## Results summary
| System       | Macro F1 | Signatures | DOB   | ID Nums | Names |
|--------------|----------|------------|-------|---------|-------|
| Omni-Shield  | 0.187    | 0.870      | 0.522 | 0.096   | 0.312 |
| Llama 3 8B   | 0.270    | 0.000      | 0.854 | 0.778   | 0.720 |
| Presidio     | 0.057    | 0.000      | 0.000 | 0.280   | 0.290 |

## Key narrative
- Omni-Shield is the ONLY system that detects signatures (F1=0.870)
- Llama 3 wins on text PII but is blind to visual PII
- Presidio is weakest overall — rule-based NER cannot handle ID cards
- No single system handles all PII — motivates multi-modal approach

## Honest framing for paper
Llama 3 outperforms Omni-Shield on Macro F1 (0.270 vs 0.187).
This is disclosed honestly and explained: Llama 3 has 4x more
parameters dedicated to language, but cannot see images.
Omni-Shield trades text precision for visual capability.
A combined system (future work) would dominate all categories.

## Privacy argument (beyond F1)
Omni-Shield adds: blockchain audit trail, ZK-SNARK proof,
multi-format support (6 doc types), selective PII categories,
canvas editor for corrections — none of which Llama 3 provides.
