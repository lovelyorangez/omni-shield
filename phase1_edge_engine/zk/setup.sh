#!/bin/bash
# One-time ZK trusted setup for OMNI-SHIELD redaction circuit.
# Re-run this if redaction.zok is modified.
set -e
export PATH="$PATH:$HOME/.zokrates/bin"
cd "$(dirname "$0")"

echo "[ZK setup] Compiling redaction.zok..."
zokrates compile -i redaction.zok -o redaction

echo "[ZK setup] Generating proving and verification keys..."
zokrates setup -i redaction

echo "[ZK setup] Exporting Solidity verifier..."
zokrates export-verifier -o Verifier.sol

echo "[ZK setup] Done."
ls -lh proving.key verification.key Verifier.sol
