# Comparison Results — Omni-Shield vs Presidio

## Overall
| System       | Macro F1 | Notes                        |
|--------------|----------|------------------------------|
| Omni-Shield  | 0.1867   | Visual + text hybrid         |
| Presidio     | 0.0566   | Text/regex only              |

## Where Omni-Shield wins decisively
- Signatures: 0.870 vs 0.000 — Presidio has no concept of signatures
- Date of Birth: 0.522 vs 0.000 — Presidio misses DOB on ID card layouts
- Addresses: 0.067 vs 0.000 — Presidio needs NER context, not label+value pairs

## Where Presidio is competitive
- ID Numbers: 0.28 vs 0.10 — Presidio's regex patterns (SSN, passport)
  match structured number formats reliably. Omni-Shield's VLM has
  low recall on ID numbers due to small font + format variation.
- Names: 0.29 vs 0.31 — roughly equal. Both struggle with recall.

## Paper narrative
Presidio's strength is structured text patterns (regex-based ID numbers).
Omni-Shield's strength is visual PII (signatures, faces, layout-aware DOB).
The comparison validates the multi-modal approach — a text-only system
cannot redact ID cards reliably because most PII is layout-dependent.
