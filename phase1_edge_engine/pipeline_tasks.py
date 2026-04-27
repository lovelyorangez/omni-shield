"""
RQ task functions for the OMNI-SHIELD image pipeline.

Each function runs inside the rq worker process (separate from the FastAPI
process). With SimpleWorker (--worker-class rq.SimpleWorker), jobs run in the
SAME process without forking, so models loaded here stay in memory between
jobs — reducing warm-job latency from ~71s to ~12s.

Models are pre-loaded at first job invocation and reused thereafter via the
module-level globals below.
"""

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

log = logging.getLogger("omni_shield.tasks")

# ── Module-level model cache ──────────────────────────────────────────────────
# Populated on first job; reused by every subsequent job in the same worker
# process (only effective with SimpleWorker / no-fork mode).
_models_loaded = False


def _ensure_models_loaded() -> None:
    """Pre-load VLM and text models into the module-level caches."""
    global _models_loaded
    if _models_loaded:
        return
    from omni_shield_agents import CFG, load_vlm, load_text_model
    log.info("Worker: pre-loading VLM model …")
    load_vlm(CFG)
    log.info("Worker: pre-loading text model …")
    load_text_model(CFG)
    _models_loaded = True
    log.info("Worker: models ready — subsequent jobs will skip load")


def run_image_pipeline(
    tmp_path: str,
    owner_address: str,
    pii_filter_raw: str,
    custom_terms_raw: str,
    custom_patterns_raw: str,
    job_id: str,
) -> dict:
    """
    Run the full 9-agent VLM image pipeline inside an rq worker process.
    Returns a JSON-serialisable result dict.
    """
    from omni_shield_agents import OmniShieldOrchestrator, CFG

    # Ensure models are loaded (no-op after first call with SimpleWorker)
    _ensure_models_loaded()

    try:
        pii_filter      = json.loads(pii_filter_raw)      if pii_filter_raw      else {}
        custom_terms    = json.loads(custom_terms_raw)    if custom_terms_raw    else []
        custom_patterns = json.loads(custom_patterns_raw) if custom_patterns_raw else []

        orch = OmniShieldOrchestrator(CFG)
        state = orch.run(
            tmp_path,
            pii_filter=pii_filter,
            custom_terms=custom_terms,
            custom_patterns=custom_patterns,
        )

        # ── Preserve original before tmp cleanup ─────────────────────────────
        suffix            = os.path.splitext(tmp_path)[1] or ".jpg"
        original_filename = f"original_{job_id}{suffix}"
        original_dest     = os.path.join("redacted_output", original_filename)
        shutil.copy2(tmp_path, original_dest)

        redacted_filename = (
            os.path.basename(state.output_path) if state.output_path else ""
        )
        redaction_count = len(state.audit.get("redactions", []))

        # ── Compute identity fields from original bytes (still on disk) ───────
        sha256_hex  = hashlib.sha256(Path(tmp_path).read_bytes()).hexdigest()
        anchor_salt = f"{owner_address}:{int(time.time())}"
        anchor_hash = hashlib.sha256((sha256_hex + anchor_salt).encode()).hexdigest()

        # Stamp identity fields onto the in-memory audit dict so the return
        # dict (and the on-disk patch below) share the same values.
        state.audit.update({
            "sha256_hex":      sha256_hex,
            "file_hash":       sha256_hex,
            "anchor_hash":     anchor_hash,
            "anchor_salt":     anchor_salt,
            "owner_address":   owner_address,
            "filename":        Path(tmp_path).name,
            "original_file":   original_filename,
            "redacted_file":   redacted_filename,
            "timestamp":       int(time.time()),
            "redaction_count": redaction_count,
            "pii_filter":      pii_filter,
        })

        # ── Patch on-disk audit JSON ──────────────────────────────────────────
        if state.output_path:
            audit_path = Path(state.output_path).with_suffix(".audit.json")
            if audit_path.exists():
                try:
                    audit_data = json.loads(audit_path.read_text())
                    audit_data.update(state.audit)
                    audit_path.write_text(json.dumps(audit_data, indent=2))
                except Exception as e:
                    log.warning(f"Worker: could not patch audit JSON: {e}")

        # ── Blockchain anchor (synchronous, non-fatal) ───────────────────────
        try:
            from blockchain_manager import anchor_record as _anc_sync
            returned_hash, returned_salt = _anc_sync(
                sha256_hex, redaction_count,
                state.doc_type or "image", owner_address, salt=anchor_salt,
            )
            log.info(f"[BLOCKCHAIN] Image anchored: {sha256_hex[:16]}…")
            # Patch on-disk audit with confirmed anchor values
            if state.output_path:
                _ap = Path(state.output_path).with_suffix(".audit.json")
                if _ap.exists():
                    try:
                        _ad = json.loads(_ap.read_text())
                        _ad["anchor_hash"] = returned_hash
                        _ad["anchor_salt"] = returned_salt
                        _ap.write_text(json.dumps(_ad, indent=2))
                    except Exception as _pe:
                        log.warning(f"Worker: could not patch anchor in audit JSON: {_pe}")
        except Exception as e:
            log.warning(f"[BLOCKCHAIN] anchor failed: {e}")

        return {
            "job_id":          job_id,
            "status":          "done",
            "original_file":   original_filename,
            "redacted_file":   redacted_filename,
            "output_path":     state.output_path,
            "audit":           state.audit,
            "coverage_pct":    state.coverage_pct,
            "doc_type":        state.doc_type,
            "agent_timings":   state.agent_timings,
            "redaction_count": redaction_count,
            "file_hash":       state.audit.get("sha256_hex", ""),
            "anchor_hash":     state.audit.get("anchor_hash", ""),
            "redactions":      state.audit.get("redactions", []),
        }
    finally:
        # Clean up temp upload regardless of success/failure
        try:
            os.remove(tmp_path)
        except OSError:
            pass
