"""
zk_prover.py — ZK-SNARK proof generation and verification for OMNI-SHIELD.

Circuit: zk/redaction.zok
  Public inputs:  doc_hash (u32[8]), redaction_count (u32)
  Private inputs: items_hash (u32[8])
  Output:         commitment (u32[8])  = SHA-256(doc_hash || (items_hash with [0]^=count))

The commitment in proof.json["inputs"][9:17] always matches:
  import hashlib, struct
  bound = items_hash[:]; bound[0] ^= redaction_count
  hashlib.sha256(struct.pack('>8I', *doc_hash) + struct.pack('>8I', *bound)).digest()

Usage:
  proof = generate_proof(document_bytes, redacted_items, redaction_count)
  ok    = verify_proof(proof)
"""

import hashlib
import json
import logging
import os
import struct
import subprocess
import threading
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

ZK_DIR     = Path(__file__).parent / "zk"
ZOKRATES   = os.environ.get("ZOKRATES_BIN", "zokrates")
_ZK_LOCK   = threading.Lock()   # one proof at a time (ZoKrates uses fixed filenames)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256_u32(data: bytes) -> list[int]:
    """Return SHA-256 of data as a list of 8 big-endian u32 values."""
    digest = hashlib.sha256(data).digest()
    return list(struct.unpack(">8I", digest))


def _items_hash_u32(redacted_items: list[str]) -> list[int]:
    """
    Canonical hash of the redacted item set.
    Sort + join with '|' so order doesn't matter.
    """
    combined = "|".join(sorted(set(str(v) for v in redacted_items if v))).encode()
    return _sha256_u32(combined)


def compute_commitment(doc_hash_u32: list[int],
                       items_hash_u32: list[int],
                       redaction_count: int) -> list[int]:
    """
    Python-side commitment that matches the circuit output exactly:
      SHA-256(pack(doc_hash) || pack(bound_items))
    where bound_items = items_hash with [0] ^= redaction_count.
    """
    bound = items_hash_u32[:]
    bound[0] ^= redaction_count
    doc_bytes   = struct.pack(">8I", *doc_hash_u32)
    bound_bytes = struct.pack(">8I", *bound)
    return _sha256_u32(doc_bytes + bound_bytes)


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=ZK_DIR, **kwargs
    )


# ── Public API ────────────────────────────────────────────────────────────────

def is_available() -> bool:
    """Return True if ZoKrates and the proving key are both present."""
    if not (ZK_DIR / "proving.key").exists():
        return False
    try:
        r = _run([ZOKRATES, "--version"])
        return r.returncode == 0
    except FileNotFoundError:
        return False


def generate_proof(
    document_bytes: bytes,
    redacted_items: list[str],
    redaction_count: int,
) -> dict:
    """
    Generate a Groth16 ZK-SNARK proving the document was redacted.

    Public  (in proof):  doc_hash, redaction_count, commitment
    Private (never seen): items_hash

    Returns a dict suitable for JSON serialisation and storage in the audit JSON.
    Raises RuntimeError if ZoKrates is unavailable or the circuit setup is missing.
    """
    if not is_available():
        raise RuntimeError(
            "ZoKrates proving key not found. "
            "Run phase1_edge_engine/zk/setup.sh first."
        )

    doc_hash_u32   = _sha256_u32(document_bytes)
    items_hash_u32 = _items_hash_u32(redacted_items)

    # Verify count > 0 before calling into ZoKrates (circuit will assert this too)
    if redaction_count < 1:
        raise ValueError("redaction_count must be >= 1")

    # Compute expected commitment so we can cross-check after proof generation
    expected_commitment = compute_commitment(doc_hash_u32, items_hash_u32, redaction_count)

    args = (
        [str(x) for x in doc_hash_u32]
        + [str(redaction_count)]
        + [str(x) for x in items_hash_u32]
    )

    with _ZK_LOCK:
        # compute-witness
        r = _run([ZOKRATES, "compute-witness", "-i", "redaction", "-a"] + args)
        if r.returncode != 0:
            raise RuntimeError(f"ZoKrates compute-witness failed:\n{r.stderr}")

        # generate-proof
        r = _run([ZOKRATES, "generate-proof", "-i", "redaction"])
        if r.returncode != 0:
            raise RuntimeError(f"ZoKrates generate-proof failed:\n{r.stderr}")

        proof_data = json.loads((ZK_DIR / "proof.json").read_text())

    # Extract commitment from proof inputs (indices 9-16 = output u32[8])
    raw_inputs   = proof_data["inputs"]
    commitment   = [int(x, 16) for x in raw_inputs[9:17]]

    if commitment != expected_commitment:
        log.warning(
            "[ZK] commitment mismatch — "
            f"circuit={commitment} python={expected_commitment}"
        )

    return {
        "proof":            proof_data["proof"],
        "inputs":           proof_data["inputs"],
        "commitment":       commitment,
        "doc_hash_u32":     doc_hash_u32,
        "redaction_count":  redaction_count,
        "scheme":           proof_data.get("scheme", "g16"),
        "curve":            proof_data.get("curve", "bn128"),
    }


def verify_proof(proof_dict: dict) -> bool:
    """
    Verify a stored proof dict using the on-disk verification key.
    Returns True iff ZoKrates reports PASSED.
    """
    vk = ZK_DIR / "verification.key"
    if not vk.exists():
        log.warning("[ZK] verification.key not found — cannot verify")
        return False

    tmp = ZK_DIR / "_verify_tmp.json"
    with _ZK_LOCK:
        try:
            tmp.write_text(json.dumps({
                "scheme": proof_dict.get("scheme", "g16"),
                "curve":  proof_dict.get("curve",  "bn128"),
                "proof":  proof_dict["proof"],
                "inputs": proof_dict["inputs"],
            }))
            r = _run([ZOKRATES, "verify", "-j", "_verify_tmp.json"])
            return "PASSED" in r.stdout
        finally:
            tmp.unlink(missing_ok=True)
