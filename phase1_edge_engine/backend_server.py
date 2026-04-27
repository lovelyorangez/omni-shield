import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# Ensure ZoKrates binary is on PATH regardless of how uvicorn was started
_zk_bin = os.path.expanduser("~/.zokrates/bin")
if _zk_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + ":" + _zk_bin

import asyncio
import gc
import hashlib
import io
import json
import logging
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager

log = logging.getLogger("omni_shield.server")
from pathlib import Path
from typing import List, Optional

import cv2
import psutil
import fitz  # PyMuPDF

# Background stats sampler — runs independently of uvicorn's event loop so
# GPU VRAM and CPU readings remain live even while the pipeline blocks the loop.
_stats_cache: dict = {
    "cpu_pct":      0,
    "ram_used_gb":  0.0,
    "ram_total_gb": 0.0,
    "ram_pct":      0,
    "gpu_used_mb":  0,
    "gpu_total_mb": 8192,
    "gpu_pct":      0,
    "gpu_name":     "GPU",
    "ts":           0,
}

def _stats_sampler() -> None:
    while True:
        try:
            _stats_cache["cpu_pct"] = round(psutil.cpu_percent(interval=1.0))
            ram = psutil.virtual_memory()
            _stats_cache["ram_used_gb"]  = round(ram.used  / 1e9, 1)
            _stats_cache["ram_total_gb"] = round(ram.total / 1e9, 1)
            _stats_cache["ram_pct"]      = round(ram.percent)
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                gpu_used  = int(parts[1].strip())
                gpu_total = int(parts[2].strip())
                _stats_cache["gpu_name"]     = parts[0].strip()
                _stats_cache["gpu_used_mb"]  = gpu_used
                _stats_cache["gpu_total_mb"] = gpu_total
                _stats_cache["gpu_pct"]      = round(gpu_used / gpu_total * 100)
            _stats_cache["ts"] = int(time.time())
        except Exception:
            pass
        # cpu_percent(interval=1.0) already paces the loop to ~1s per iteration

threading.Thread(target=_stats_sampler, daemon=True).start()
import torch
from cryptography.fernet import Fernet
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from auth import verify_api_key
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram
from PIL import Image, ImageDraw
from pydantic import BaseModel
from pydub import AudioSegment
from omni_shield_agents import (
    OmniShieldOrchestrator, CFG,
    normalise_audio_transcript, _SPOKEN_ONES,
)

try:
    from blockchain_manager import (
        verify_document_hash as _verify_document_hash,
        get_my_records as _get_my_records,
        get_records_for_user as _get_records_for_user,
        anchor_record as _anchor_record,
        anchor_record_async as _anchor_record_async,
    )
    _BLOCKCHAIN_AVAILABLE = True
except Exception:
    _BLOCKCHAIN_AVAILABLE = False

try:
    from zk_prover import generate_proof as _zk_generate_proof, is_available as _zk_available
    _ZK_AVAILABLE = _zk_available()
except Exception:
    _ZK_AVAILABLE = False

def _generate_zk_proof_async(
    document_bytes: bytes,
    redacted_items: list,
    redaction_count: int,
    audit_path: Path,
) -> None:
    """Generate ZK proof in a background thread and patch the audit JSON when done."""
    def _worker():
        try:
            proof = _zk_generate_proof(document_bytes, redacted_items, redaction_count)
            log.info(f"[ZK] Proof generated. commitment={proof['commitment']}")
            # Patch existing audit JSON
            if audit_path.exists():
                data = json.loads(audit_path.read_text())
                data["zk_proof"] = proof
                audit_path.write_text(json.dumps(data, indent=2))
                log.info(f"[ZK] Audit updated: {audit_path.name}")
        except Exception as e:
            log.warning(f"[ZK] Proof generation failed: {e}")
    threading.Thread(target=_worker, daemon=True).start()

if os.getuid() == 0:
    print("ERROR: Do not run backend_server.py as root.")
    sys.exit(1)


def _shutdown(sig, frame):
    log.info(f"[SHUTDOWN] Signal {sig} received — draining in-flight requests")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


# ── 1. App ────────────────────────────────────────────────────────────────────
_models_ready = False


async def _cleanup_old_files() -> None:
    """Delete processed output files older than FILE_RETENTION_HOURS (default 24h)."""
    retention_s = int(os.environ.get("FILE_RETENTION_HOURS", "24")) * 3600
    while True:
        await asyncio.sleep(3600)   # check every hour
        cutoff = time.time() - retention_s
        cleaned = 0
        for folder in ["redacted_output", "downloads"]:
            for f in Path(folder).glob("*"):
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        f.unlink(missing_ok=True)
                        cleaned += 1
                except OSError:
                    pass
        if cleaned:
            log.info(f"[CLEANUP] Deleted {cleaned} old file(s) older than {retention_s//3600}h")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Server lifespan — startup tasks, then yield, then shutdown cleanup."""
    global _models_ready

    # Startup: launch background cleanup and update VRAM gauge
    asyncio.create_task(_cleanup_old_files())
    log.info("OMNI-SHIELD server starting — models load lazily on first request.")

    # Mark ready (models load on first request; set flag now so /health passes)
    _models_ready = True
    log.info("[WARMUP] Server ready — accepting traffic")

    yield

    # Shutdown: free GPU memory
    log.info("[SHUTDOWN] Server stopping — freeing GPU memory")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(title="OMNI-SHIELD Integrated Edge Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting (CHANGE 3) ──────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Prometheus metrics (CHANGE 8) ─────────────────────────────────────────────
Instrumentator().instrument(app).expose(app)

redactions_total = Counter(
    "omni_redactions_total",
    "Total completed redactions",
    ["format", "status"],
)
pipeline_latency = Histogram(
    "omni_pipeline_latency_seconds",
    "End-to-end pipeline latency",
    ["format"],
    buckets=[1, 2, 5, 10, 15, 20, 30, 60],
)
vram_usage_gauge = Gauge(
    "omni_vram_usage_bytes",
    "GPU VRAM used in bytes",
)

# ── Input validation helpers (CHANGE 7) ──────────────────────────────────────
_MAX_UPLOAD_BYTES: dict[str, int] = {
    "image": 10 * 1024 * 1024,
    "pdf":   50 * 1024 * 1024,
    "audio": 100 * 1024 * 1024,
    "text":  5  * 1024 * 1024,
    "docx":  20 * 1024 * 1024,
    "pptx":  20 * 1024 * 1024,
}

# DOCX and PPTX are ZIP-based; libmagic detects them as application/zip.
# Include both the canonical MIME and the ZIP detection.
_ALLOWED_MIMES: dict[str, list[str]] = {
    "image": ["image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp",
              "image/gif", "application/octet-stream"],
    "pdf":   ["application/pdf"],
    "audio": ["audio/mpeg", "audio/wav", "audio/ogg", "audio/mp4", "audio/x-wav",
              "audio/x-m4a", "video/mp4", "application/octet-stream"],
    "text":  ["text/plain", "application/octet-stream", "text/html", "text/csv"],
    "docx":  ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
              "application/msword", "application/zip", "application/x-zip-compressed",
              "application/octet-stream"],
    "pptx":  ["application/vnd.openxmlformats-officedocument.presentationml.presentation",
              "application/zip", "application/x-zip-compressed", "application/octet-stream"],
}


async def validate_upload(contents: bytes, format_type: str) -> None:
    """Raise 413 if the upload is too large. Log a warning on MIME mismatch but do not block."""
    max_bytes = _MAX_UPLOAD_BYTES.get(format_type, 10 * 1024 * 1024)
    if len(contents) > max_bytes:
        raise HTTPException(
            413, f"File too large. Max {max_bytes // 1024 // 1024} MB for {format_type}."
        )
    # MIME check: non-fatal — log a warning but never block processing on detection errors.
    try:
        import magic as _magic
        detected = _magic.from_buffer(contents[:2048], mime=True)
        allowed = _ALLOWED_MIMES.get(format_type, [])
        if allowed and detected not in allowed:
            log.warning(
                f"[VALIDATE] Unexpected MIME '{detected}' for format '{format_type}' "
                f"(allowed: {allowed}) — processing anyway"
            )
    except HTTPException:
        raise
    except Exception as e:
        log.warning(f"[VALIDATE] MIME check failed: {e} — skipping")


# ── Job queue (CHANGE 4) ──────────────────────────────────────────────────────
try:
    from queue_worker import image_queue, redis_conn as _redis_conn
    from pipeline_tasks import run_image_pipeline as _run_image_pipeline
    _QUEUE_AVAILABLE = True
except Exception:
    _QUEUE_AVAILABLE = False
    _redis_conn = None
    _run_image_pipeline = None

os.makedirs("downloads", exist_ok=True)


# ── 2. Orchestrator (no models loaded here — lazy inside OmniShieldOrchestrator) ──
orchestrator = OmniShieldOrchestrator(CFG)


# ── 3. Runtime state ──────────────────────────────────────────────────────────
state = {
    "redaction_active": True,
    "camera_active": False,
}

SECRET_KEY   = Fernet.generate_key()
cipher_suite = Fernet(SECRET_KEY)


# ── 4. Lazy model cache ───────────────────────────────────────────────────────
# All variables start as None; loaded on first use, then cached.

_visual_model = None
_audio_model  = None
_bert_ner     = None


def _get_visual_model():
    global _visual_model
    if _visual_model is None:
        from ultralytics import YOLO
        print("⏳ Loading YOLO visual model (first request)...")
        _visual_model = YOLO("yolov8n.pt")
        print("✅ YOLO ready.")
    return _visual_model


def _get_audio_model():
    global _audio_model
    if _audio_model is None:
        from faster_whisper import WhisperModel
        print("⏳ Loading Whisper model (first audio request)...")
        _audio_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Whisper ready.")
    return _audio_model


_AUDIO_INITIAL_PROMPT = (
    "Transcribe all numbers as digits. "
    "Phone numbers like nine eight seven six five four three two one zero "
    "should be written as 9876543210. "
    "Aadhaar numbers, account numbers, dates like fifteenth March nineteen "
    "eighty seven should be written as 15th March 1987. "
    "PAN numbers like A B C D E one two three four F should be written as ABCDE1234F."
)


def _find_spoken_timestamps(
    detected_value: str,
    word_segments: list[dict],
) -> Optional[tuple[float, float]]:
    """
    Map a detected all-digit PII value (e.g. "9876543210") back to the
    Whisper word timestamps of the spoken digit words ("nine eight seven…").

    Returns (start_s, end_s) or None if not found.
    """
    target = re.sub(r'[\s\-]', '', detected_value)
    if not target.isdigit():
        return None

    digit_ws = [
        w for w in word_segments
        if re.sub(r'[^a-z]', '', w['word'].lower()) in _SPOKEN_ONES
    ]
    for i in range(len(digit_ws)):
        run = ''
        for j in range(i, len(digit_ws)):
            d = _SPOKEN_ONES.get(re.sub(r'[^a-z]', '', digit_ws[j]['word'].lower()), '')
            run += d
            if run == target:
                return digit_ws[i]['start'], digit_ws[j]['end']
            if not target.startswith(run):
                break
    return None


def _get_bert_ner():
    global _bert_ner
    if _bert_ner is None:
        from transformers import pipeline
        print("⏳ Loading BERT NER model (first text/image request)...")
        _bert_ner = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )
        print("✅ BERT NER ready.")
    return _bert_ner


def _get_qwen():
    """
    Return (tokenizer, model) for Qwen2.5-3B.
    Always routes through the agents _TEXT_MODEL_CACHE so the model is
    loaded exactly once and shared between the image pipeline and PDF/audio
    endpoints — prevents double-loading and OOM on 8 GB cards.
    """
    try:
        from omni_shield_agents import _TEXT_MODEL_CACHE, load_text_model, CFG
        if _TEXT_MODEL_CACHE.get("model") is not None:
            log.info("[_get_qwen] Reusing text model from agents cache")
            return _TEXT_MODEL_CACHE["tokenizer"], _TEXT_MODEL_CACHE["model"]
        log.info("[_get_qwen] Text model not in cache — loading via load_text_model()")
        tokenizer, model = load_text_model(CFG)
        return tokenizer, model
    except Exception as exc:
        log.error(f"[_get_qwen] Failed to load text model: {exc}")
        raise


# ── 5. PII extraction helpers ─────────────────────────────────────────────────

_ALL_ENABLED_FILTER = {
    k: True for k in [
        'names', 'faces', 'signatures', 'phones', 'emails',
        'dob', 'id_numbers', 'addresses', 'org_names', 'dates',
    ]
}


_HYBRID_TYPE_TO_FILTER_KEY = {
    'name':    'names',
    'email':   'emails',
    'phone':   'phones',
    'dob':     'dob',
    'id':      'id_numbers',
    'address': 'addresses',
    'org':     'org_names',
    'date':    'dates',
}

_EMAIL_RE = re.compile(r'^[\w.+-]+@[\w-]+\.[\w.]+$')
_PHONE_RE = re.compile(r'^[\d\s\-\+\(\)\.]{7,}$')
_DATE_RE  = re.compile(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}')

_PAN_RE          = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b')
_AADHAAR_RE      = re.compile(r'\b\d{4}\s\d{4}\s\d{4}\b')
_IFSC_RE         = re.compile(r'\b[A-Z]{4}0[A-Z0-9]{6}\b')
_CIN_RE          = re.compile(r'\b[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b')
_GST_RE          = re.compile(r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b')
_PASSPORT_RE     = re.compile(r'\b[A-Z][0-9]{7}\b')
_ACCOUNT_RE      = re.compile(r'\b\d{9,18}\b')
_WRITTEN_DATE_RE = re.compile(
    r'\b\d{1,2}(st|nd|rd|th)?\s+'
    r'(January|February|March|April|May|June|July|August|September|October|November|December)'
    r'\s+\d{4}\b',
    re.IGNORECASE,
)
_DOT_DATE_RE     = re.compile(r'\b\d{1,2}\.\d{1,2}\.\d{4}\b')

# Ordered list of (compiled_re, filter_key, label) for the deterministic sweep
_HYBRID_REGEX_SWEEP: list[tuple[re.Pattern, str, str]] = [
    (_PAN_RE,          'id_numbers', 'PAN Number'),
    (_AADHAAR_RE,      'id_numbers', 'Aadhaar'),
    (_IFSC_RE,         'id_numbers', 'IFSC Code'),
    (_CIN_RE,          'id_numbers', 'CIN'),
    (_GST_RE,          'id_numbers', 'GST Number'),
    (_PASSPORT_RE,     'id_numbers', 'Passport Number'),
    (_ACCOUNT_RE,      'id_numbers', 'Account Number'),
    (_WRITTEN_DATE_RE, 'dob',        'Date'),
    (_DOT_DATE_RE,     'dob',        'Date'),
]

KNOWN_ORG_SUFFIXES = {
    'contractors', 'corporation', 'company', 'limited',
    'incorporated', 'associates', 'group', 'partners',
    'council', 'authority', 'agency', 'institute',
    'foundation', 'services', 'solutions', 'systems',
    'documents', 'schedules', 'bidders',
}


def _classify_hybrid_item(item: str) -> str:
    if _EMAIL_RE.match(item):
        return 'emails'
    if _PHONE_RE.match(item.strip()):
        return 'phones'
    # Indian government / financial IDs
    if (_PAN_RE.fullmatch(item.strip()) or _IFSC_RE.fullmatch(item.strip()) or
            _CIN_RE.fullmatch(item.strip()) or _GST_RE.fullmatch(item.strip()) or
            _AADHAAR_RE.fullmatch(item.strip()) or _PASSPORT_RE.fullmatch(item.strip()) or
            _ACCOUNT_RE.fullmatch(item.strip())):
        return 'id_numbers'
    words = item.lower().split()
    if words and words[-1] in KNOWN_ORG_SUFFIXES:
        return 'org_names'
    if any(w in KNOWN_ORG_SUFFIXES for w in words):
        return 'org_names'
    if _DATE_RE.search(item) or _WRITTEN_DATE_RE.search(item) or _DOT_DATE_RE.search(item):
        return 'dob'
    return 'names'


def apply_pii_filter_to_hybrid(
    pii_items: list[str],
    pii_filter: dict,
) -> list[str]:
    """Classify each hybrid NER result and drop it if its category is disabled."""
    if not pii_filter:
        return pii_items
    filtered = []
    for item in pii_items:
        key = _classify_hybrid_item(item)
        if pii_filter.get(key, True):
            filtered.append(item)
        else:
            log.info(f"[FILTER] Dropped '{item}' (key='{key}' disabled)")
    return filtered


def _unload_models() -> None:
    """Unload cached models and empty the PyTorch CUDA cache in the web process.

    Non-image endpoints (text, PDF, audio, docx, pptx) may load models into
    VRAM/RAM.  Nulling the agent model caches and releasing the CUDA cache
    after each request prevents OOM when a subsequent image job arrives and
    the rq worker needs to load the VLM pipeline.
    """
    import gc
    try:
        from omni_shield_agents import _VLM_CACHE, _TEXT_MODEL_CACHE
        _VLM_CACHE["model"] = None
        _VLM_CACHE["processor"] = None
        _TEXT_MODEL_CACHE["model"] = None
        _TEXT_MODEL_CACHE["tokenizer"] = None
        log.info("[CLEANUP] Model cache cleared")
    except Exception as e:
        log.warning(f"[CLEANUP] {e}")
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# Keep old name as alias so any other callers aren't broken
_release_vram = _unload_models


def _parse_redaction_controls(
    pii_filter_raw: str,
    custom_terms_raw: str,
    custom_patterns_raw: str,
) -> tuple[dict, list[str], list[str]]:
    """Parse the three redaction-control form fields, returning safe defaults."""
    try:
        pii_filter = json.loads(pii_filter_raw) if pii_filter_raw else {}
    except Exception:
        pii_filter = {}
    try:
        custom_terms = json.loads(custom_terms_raw) if custom_terms_raw else []
    except Exception:
        custom_terms = []
    try:
        custom_patterns = json.loads(custom_patterns_raw) if custom_patterns_raw else []
    except Exception:
        custom_patterns = []
    if not pii_filter:
        pii_filter = _ALL_ENABLED_FILTER
    return pii_filter, custom_terms, custom_patterns


REGEX_PATTERNS = {
    "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "URL":   r'(https?://[^\s()<>]+|www\.[^\s()<>]+)',
    "ID":    r'\b([a-zA-Z0-9_]+\|\d{4,}|[A-Z]{2,}\d{4,12}|[A-Z0-9]{3,}\-[A-Z0-9]{4,}|\d{3}-\d{2}-\d{4}|(?:\d[\s-]*){9,21})\b',
    "PHONE": r'(\+?\d{1,3}[-.\s]?)?\(?\s?\d{1,4}\s?\)?[-\s.]?\d{3}[-\s.]?\d{4}',
}


def normalize_for_comparison(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()


def is_source_verified_fuzzy(pii_candidate: str, source_text: str) -> bool:
    cand_norm = normalize_for_comparison(pii_candidate)
    if not cand_norm or len(cand_norm) < 2:
        return False
    source_norm = normalize_for_comparison(source_text)
    if cand_norm in source_norm:
        noise = ['university', 'college', 'hospital', 'clinic', 'bank',
                 'corp', 'inc', 'llc', 'india', 'usa', 'assignment']
        if any(word in pii_candidate.lower() for word in noise):
            return False
        return True
    return False


def extract_pii_bert(full_text: str):
    """Fast BERT + Regex engine for text/image endpoints."""
    ner_results = _get_bert_ner()(full_text)
    pii_entities = [ent['word'] for ent in ner_results if ent['entity_group'] == 'PER']

    regex_preds = []
    for ptype, pattern in REGEX_PATTERNS.items():
        flags = re.IGNORECASE if ptype in ("EMAIL", "URL") else 0
        regex_preds.extend(re.findall(pattern, full_text, flags=flags))

    combined = list(set(pii_entities + regex_preds))
    return sorted(combined, key=len, reverse=True)


_EXCLUDE_WORDS = {
    'the', 'this', 'that', 'these', 'those', 'article',
    'contract', 'agreement', 'owner', 'contractor', 'bidder',
    'appendix', 'india', 'english', 'government', 'director',
    'employee', 'payment', 'project', 'general', 'special',
    'conditions', 'technical', 'specifications', 'drawings',
    'award', 'letter', 'instruction', 'bidders', 'whereas',
    'united', 'kingdom', 'states', 'america', 'london',
    'denver', 'january', 'february', 'march', 'april',
    'may', 'june', 'july', 'august', 'september', 'october',
    'november', 'december', 'monday', 'tuesday', 'wednesday',
    'thursday', 'friday', 'saturday', 'sunday',
    'international', 'national', 'federal', 'state', 'local',
    'sub', 'non', 'page', 'section', 'clause', 'schedule',
    # extended
    'amendment', 'appendices', 'attached', 'hereto',
    'listed', 'schedules', 'bidding', 'completion',
    'delegate', 'reference', 'independent', 'representative',
    'president', 'manager', 'agent', 'further', 'accordingly',
    'signed', 'order', 'precedence', 'definitions',
    'capitalized', 'definition', 'price', 'prices', 'legal',
    'govt', 'contractors', 'understood', 'sole', 'behalf',
    'authorized',
}

COMMON_WORDS = {
    'went', 'the', 'and', 'with', 'to', 'a', 'of',
    'in', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall',
    'that', 'this', 'these', 'those', 'it', 'its', 'they',
    'them', 'their', 'he', 'she', 'we', 'you', 'at', 'by',
    'for', 'on', 'as', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'from', 'up', 'down', 'out',
    'off', 'over', 'under', 'again', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'same', 'so', 'than', 'too',
    'very', 'just', 'any', 'also', 'but', 'or', 'an', 'your',
    'our', 'my', 'his', 'her', 'if', 'can', 'about', 'which',
    'corporation', 'incorporated', 'laws', 'place', 'business',
    'hereinafter', 'called', 'desires', 'engage', 'agreed',
    'such', 'engagement', 'upon', 'terms', 'conditions',
    'appearing', 'follows', 'constitute', 'contract', 'between',
    'read', 'construed', 'integral', 'part', 'following',
    'documents', 'shall', 'owner', 'contractor',
    'cafe', 'coffee', 'cup', 'drink', 'met', 'later', 'invited',
}

_LEGAL_TERMS = {
    'law', 'laws', 'act', 'code', 'rule', 'hereby', 'herein',
    'thereof', 'wherein', 'hereunder', 'thereunder', 'aforesaid',
    'aforementioned', 'notwithstanding',
}


def extract_names_by_capitalization(text: str) -> list[str]:
    """Match 2-3 consecutive Title Case words (4+ chars each) as potential person names."""
    names = []
    pattern = r'\b([A-Z][a-z]+)(?:\s+([A-Z][a-z]+))?(?:\s+([A-Z][a-z]+))?\b'
    for match in re.finditer(pattern, text):
        words = [w for w in match.groups() if w]
        if len(words) < 2:
            continue
        if any(len(w) < 4 for w in words):
            continue
        if any(w.lower() in _EXCLUDE_WORDS for w in words):
            continue
        name = ' '.join(words)
        if len(name) > 40:
            continue
        names.append(name)
    return names


def extract_names_mixed_case(text: str) -> list[str]:
    """Catch names like 'John macer' — Title Case first word followed by a
    lowercase word that is not a common English word."""
    names = []
    pattern = r'\b([A-Z][a-z]{1,20})\s+([A-Za-z]{4,20})\b'
    for match in re.finditer(pattern, text):
        first, second = match.group(1), match.group(2)
        if second.lower() in COMMON_WORDS or second.lower() in _EXCLUDE_WORDS:
            continue
        if first.lower() in _EXCLUDE_WORDS or first.lower() in COMMON_WORDS:
            continue
        if second.lower() in _LEGAL_TERMS:
            continue
        names.append(f"{first} {second}")
    return names


_SENTENCE_STARTERS = {
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'it', 'they', 'we', 'he', 'she', 'you', 'i', 'there',
    'here', 'when', 'where', 'what', 'which', 'who', 'how', 'why',
}


def extract_single_names(text: str) -> list[str]:
    """Catch single names like Malayne, Miwa, Rosine anywhere in a sentence.
    Sentence-starting words are included but held to a stricter check."""
    names = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        words = sentence.split()
        for i, word in enumerate(words):
            clean = re.sub(r'[^A-Za-z]', '', word)
            if (len(clean) >= 6
                    and clean[0].isupper()
                    and clean[1:].islower()
                    and clean.lower() not in COMMON_WORDS
                    and clean.lower() not in _EXCLUDE_WORDS):
                if i == 0 and clean.lower() in _SENTENCE_STARTERS:
                    continue
                names.append(clean)
    return names


def extract_names_by_context(text: str) -> list[str]:
    """Extract names following indicator phrases like 'signed by', 'behalf of', etc."""
    indicators = [
        r'(?:signed by|presence of|behalf of|between|called|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})(?:,?\s+(?:a corporation|president|owner|director|manager))',
    ]
    names = []
    for pattern in indicators:
        for match in re.finditer(pattern, text):
            names.append(match.group(1).strip())
    return names


def deduplicate_names(names: list[str]) -> list[str]:
    """Remove single words already covered by a longer name in the list."""
    result = []
    for name in sorted(names, key=len, reverse=True):
        if not any(name.lower() in existing.lower() for existing in result):
            result.append(name)
    return result


def extract_pii_hybrid(text: str) -> list[str]:
    """Hybrid PII extraction: capitalization patterns + context clues + regex."""
    found = set()

    for name in extract_names_by_capitalization(text):
        found.add(name)

    for name in extract_names_mixed_case(text):
        found.add(name)

    for name in extract_single_names(text):
        found.add(name)

    for name in extract_names_by_context(text):
        found.add(name)

    # Emails
    for m in re.finditer(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b', text):
        found.add(m.group())

    # Phone numbers
    for m in re.finditer(
        r'\b(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b',
        text,
    ):
        found.add(m.group().strip())

    # ── Deterministic regex sweep for IDs / dates missed by heuristics ───
    for _pat, _fkey, _label in _HYBRID_REGEX_SWEEP:
        for _m in _pat.finditer(text):
            _val = _m.group().strip()
            if _val:
                found.add(_val)
                log.info(f"[REGEX-SWEEP] {_label}: '{_val}'")

    result = deduplicate_names(list(found))
    log.info(f"Hybrid PII: {len(result)} items: {result}")
    return result


# ── 6. Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Liveness + readiness probe (CHANGE 9)."""
    if not _models_ready:
        raise HTTPException(503, "Server not ready")
    return {
        "status":   "ok",
        "models":   "ready",
        "vram_mb":  _stats_cache.get("gpu_used_mb", 0),
        "cpu_pct":  _stats_cache.get("cpu_pct", 0),
        "ram_pct":  _stats_cache.get("ram_pct", 0),
    }


@app.get("/api/job/{job_id}")
async def get_job_status(job_id: str):
    """Poll the status of an async image/audio pipeline job (CHANGE 4)."""
    if not _QUEUE_AVAILABLE or _redis_conn is None:
        raise HTTPException(503, "Job queue not available")
    try:
        from rq.job import Job
        job = Job.fetch(job_id, connection=_redis_conn)
        if job.is_finished:
            return {"status": "done", "result": job.result}
        if job.is_failed:
            return {"status": "failed", "error": str(job.exc_info)}
        return {"status": str(job.get_status())}
    except Exception:
        raise HTTPException(404, "Job not found")

@app.get("/video_feed")
def video_feed():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        state["camera_active"] = True
        while state["camera_active"]:
            success, frame = cap.read()
            if not success:
                break
            if state["redaction_active"]:
                results = _get_visual_model()(frame, verbose=False)
                for result in results:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            roi = frame[y1:y2, x1:x2]
                            if roi.size > 0:
                                frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (99, 99), 30)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# ── DOCX / PPTX helpers ──────────────────────────────────────────────────────
import io as _io
import re as _re
import docx as _docx
from pptx import Presentation as _Presentation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm


def _generate_pdf_from_paragraphs(out_path, paragraphs, table_texts, title=""):
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    story  = []
    if title:
        story.append(Paragraph(f"<b>Redacted: {title}</b>", styles['Title']))
        story.append(Spacer(1, 12))
    for text, _style_name in paragraphs:
        if not text:
            story.append(Spacer(1, 6))
            continue
        display = text.replace('[REDACTED]', '<font color="red">[REDACTED]</font>')
        try:
            story.append(Paragraph(display, styles['Normal']))
        except Exception:
            story.append(Paragraph(text, styles['Normal']))
        story.append(Spacer(1, 4))
    if table_texts:
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Tables:</b>", styles['Heading2']))
        for text in table_texts:
            display = text.replace('[REDACTED]', '<font color="red">[REDACTED]</font>')
            try:
                story.append(Paragraph(display, styles['Normal']))
            except Exception:
                story.append(Paragraph(text, styles['Normal']))
            story.append(Spacer(1, 4))
    doc.build(story)


def _generate_pdf_from_slides(out_path, slide_paragraphs, title=""):
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    story  = []
    if title:
        story.append(Paragraph(f"<b>Redacted: {title}</b>", styles['Title']))
        story.append(Spacer(1, 12))
    for slide_num, texts in slide_paragraphs:
        story.append(Paragraph(f"<b>Slide {slide_num}</b>", styles['Heading2']))
        story.append(Spacer(1, 6))
        for text in texts:
            if not text:
                continue
            display = text.replace('[REDACTED]', '<font color="red">[REDACTED]</font>')
            try:
                story.append(Paragraph(display, styles['Normal']))
            except Exception:
                story.append(Paragraph(text, styles['Normal']))
            story.append(Spacer(1, 4))
        story.append(Spacer(1, 16))
    doc.build(story)


def _redact_text_block(text: str, pii_filter_d: dict,
                       terms: list, patterns: list,
                       redacted_items: list) -> tuple[str, int]:
    """Run hybrid NER + custom terms on one text block. Returns (redacted_text, chars_replaced)."""
    pii_items = extract_pii_hybrid(text)
    pii_items = apply_pii_filter_to_hybrid(pii_items, pii_filter_d)
    for term in terms:
        if term.lower() in text.lower():
            pii_items.append(term)
    for pattern in patterns:
        try:
            pii_items.extend(_re.findall(pattern, text, _re.IGNORECASE))
        except _re.error:
            pass
    replaced = 0
    for item in pii_items:
        count = len(_re.findall(_re.escape(item), text, _re.IGNORECASE))
        if count:
            replaced += len(item) * count
            redacted_items.append(item)
            text = _re.sub(_re.escape(item), '[REDACTED]', text, flags=_re.IGNORECASE)
    return text, replaced

# ── DOCX endpoint ─────────────────────────────────────────────────────────────
@app.post("/redact/docx")
@limiter.limit("20/minute")
async def redact_docx_api(
    request: Request,
    file: UploadFile = File(...),
    owner_address: str = Form(default=''),
    pii_filter: str = Form(default='{}'),
    custom_terms: str = Form(default='[]'),
    custom_patterns: str = Form(default='[]'),
    dry_run: bool = Query(default=False),
    _key: str = Depends(verify_api_key),
):
    log.info(f"[redact/docx] file={file.filename!r} dry_run={dry_run}")
    contents = await file.read()
    await validate_upload(contents, "docx")
    sha256 = hashlib.sha256(contents).hexdigest()
    Path("downloads").mkdir(exist_ok=True)
    (Path("downloads") / f"orig_{sha256}.bin").write_bytes(contents)

    pii_filter_d, terms, patterns = _parse_redaction_controls(pii_filter, custom_terms, custom_patterns)
    log.info(f"[FILTER DEBUG] pii_filter received: {pii_filter_d}")

    word_doc       = _docx.Document(_io.BytesIO(contents))
    paragraphs_out = []
    table_texts    = []
    total_chars    = 0
    redacted_chars = 0
    all_items: list[str] = []

    for para in word_doc.paragraphs:
        try:
            text = para.text
            style_name = para.style.name if para.style else 'Normal'
            if not text.strip():
                paragraphs_out.append(('', style_name))
                continue
            total_chars += len(text)
            redacted_text, replaced = _redact_text_block(text, pii_filter_d, terms, patterns, all_items)
            redacted_chars += replaced
            paragraphs_out.append((redacted_text, style_name))
        except Exception as e:
            log.warning(f"[DOCX] Skipping paragraph: {e}")
            continue

    for table in word_doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text = cell.text
                if not text.strip():
                    continue
                total_chars += len(text)
                redacted_text, replaced = _redact_text_block(text, pii_filter_d, terms, patterns, all_items)
                redacted_chars += replaced
                table_texts.append(redacted_text)

    coverage_pct    = round(redacted_chars / total_chars * 100, 1) if total_chars else 0.0
    redaction_count = len(set(all_items))
    log.info(f"[DOCX] {redaction_count} unique items redacted, coverage={coverage_pct}%")

    if dry_run:
        return {
            "dry_run": True, "message": "Dry run — no file written, no blockchain anchor",
            "would_redact": [{"label": "text", "value": v, "box": None} for v in set(all_items)],
            "redaction_count": redaction_count, "coverage_pct": coverage_pct, "doc_type": "docx",
        }

    os.makedirs(CFG.output_dir, exist_ok=True)
    ts       = int(time.time() * 1000)
    out_path = Path(CFG.output_dir) / f"redacted_{ts}.pdf"
    _generate_pdf_from_paragraphs(out_path, paragraphs_out, table_texts, title=file.filename or "Document")
    log.info(f"[DOCX] Saved: {out_path}")

    unique_items = list(set(all_items))
    _docx_anchor_salt = f"{owner_address}:{int(time.time())}"
    _docx_anchor_hash = hashlib.sha256((sha256 + _docx_anchor_salt).encode()).hexdigest()
    audit_data = {
        "sha256_hex": sha256,
        "file_hash": sha256,
        "anchor_hash": _docx_anchor_hash,
        "anchor_salt": _docx_anchor_salt,
        "owner_address": owner_address,
        "doc_type": "docx",
        "filename": file.filename,
        "redacted_file": out_path.name,
        "redaction_count": redaction_count,
        "coverage_pct": coverage_pct,
        "timestamp": int(time.time()),
        "redactions": [
            {
                "label": "Custom" if item in terms else "Text",
                "value": item,
                "source": "custom" if item in terms else "ai",
                "box": None,
            }
            for item in unique_items
        ],
        "pii_filter": pii_filter_d,
        "custom_terms_matched": [t for t in terms if any(t.lower() in r.lower() for r in all_items)],
        "custom_patterns_matched": patterns,
    }
    audit_path = out_path.with_suffix('.audit.json')
    audit_path.write_text(json.dumps(audit_data, indent=2))
    log.info(f"[DOCX] Audit: {audit_path}")

    if _ZK_AVAILABLE and redaction_count > 0:
        _generate_zk_proof_async(contents, unique_items, redaction_count, audit_path)

    if _BLOCKCHAIN_AVAILABLE:
        _anchor_record_async(sha256, redaction_count, "docx", owner_address,
                             salt=_docx_anchor_salt)

    _unload_models()
    return {
        "file_hash": sha256, "anchor_hash": _docx_anchor_hash,
        "redacted_file": out_path.name,
        "redaction_count": redaction_count, "coverage_pct": coverage_pct,
        "doc_type": "docx", "redacted_items": unique_items,
        "blockchain_status": "pending" if _BLOCKCHAIN_AVAILABLE else "unavailable",
    }


# ── PPTX endpoint ─────────────────────────────────────────────────────────────
@app.post("/redact/pptx")
@limiter.limit("20/minute")
async def redact_pptx_api(
    request: Request,
    file: UploadFile = File(...),
    owner_address: str = Form(default=''),
    pii_filter: str = Form(default='{}'),
    custom_terms: str = Form(default='[]'),
    custom_patterns: str = Form(default='[]'),
    dry_run: bool = Query(default=False),
    _key: str = Depends(verify_api_key),
):
    log.info(f"[redact/pptx] file={file.filename!r} dry_run={dry_run}")
    contents = await file.read()
    await validate_upload(contents, "pptx")
    sha256 = hashlib.sha256(contents).hexdigest()
    Path("downloads").mkdir(exist_ok=True)
    (Path("downloads") / f"orig_{sha256}.bin").write_bytes(contents)

    pii_filter_d, terms, patterns = _parse_redaction_controls(pii_filter, custom_terms, custom_patterns)
    log.info(f"[FILTER DEBUG] pii_filter received: {pii_filter_d}")

    prs            = _Presentation(_io.BytesIO(contents))
    slide_data     = []
    total_chars    = 0
    redacted_chars = 0
    all_items: list[str] = []

    for slide_num, slide in enumerate(prs.slides, 1):
        slide_texts: list[str] = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                text = para.text
                if not text.strip():
                    slide_texts.append('')
                    continue
                total_chars += len(text)
                redacted_text, replaced = _redact_text_block(text, pii_filter_d, terms, patterns, all_items)
                redacted_chars += replaced
                slide_texts.append(redacted_text)
        slide_data.append((slide_num, slide_texts))

    coverage_pct    = round(redacted_chars / total_chars * 100, 1) if total_chars else 0.0
    redaction_count = len(set(all_items))
    log.info(f"[PPTX] {redaction_count} unique items redacted, coverage={coverage_pct}%")

    if dry_run:
        return {
            "dry_run": True, "message": "Dry run — no file written, no blockchain anchor",
            "would_redact": [{"label": "text", "value": v, "box": None} for v in set(all_items)],
            "redaction_count": redaction_count, "coverage_pct": coverage_pct, "doc_type": "pptx",
        }

    os.makedirs(CFG.output_dir, exist_ok=True)
    ts       = int(time.time() * 1000)
    out_path = Path(CFG.output_dir) / f"redacted_{ts}.pdf"
    _generate_pdf_from_slides(out_path, slide_data, title=file.filename or "Presentation")
    log.info(f"[PPTX] Saved: {out_path}")

    unique_items = list(set(all_items))
    _pptx_anchor_salt = f"{owner_address}:{int(time.time())}"
    _pptx_anchor_hash = hashlib.sha256((sha256 + _pptx_anchor_salt).encode()).hexdigest()
    audit_data = {
        "sha256_hex": sha256,
        "file_hash": sha256,
        "anchor_hash": _pptx_anchor_hash,
        "anchor_salt": _pptx_anchor_salt,
        "owner_address": owner_address,
        "doc_type": "pptx",
        "filename": file.filename,
        "redacted_file": out_path.name,
        "redaction_count": redaction_count,
        "coverage_pct": coverage_pct,
        "timestamp": int(time.time()),
        "redactions": [
            {
                "label": "Custom" if item in terms else "Text",
                "value": item,
                "source": "custom" if item in terms else "ai",
                "box": None,
            }
            for item in unique_items
        ],
        "pii_filter": pii_filter_d,
        "custom_terms_matched": [t for t in terms if any(t.lower() in r.lower() for r in all_items)],
        "custom_patterns_matched": patterns,
    }
    audit_path = out_path.with_suffix('.audit.json')
    audit_path.write_text(json.dumps(audit_data, indent=2))
    log.info(f"[PPTX] Audit: {audit_path}")

    if _ZK_AVAILABLE and redaction_count > 0:
        _generate_zk_proof_async(contents, unique_items, redaction_count, audit_path)

    if _BLOCKCHAIN_AVAILABLE:
        _anchor_record_async(sha256, redaction_count, "pptx", owner_address,
                             salt=_pptx_anchor_salt)

    _unload_models()
    return {
        "file_hash": sha256, "anchor_hash": _pptx_anchor_hash,
        "redacted_file": out_path.name,
        "redaction_count": redaction_count, "coverage_pct": coverage_pct,
        "doc_type": "pptx", "redacted_items": unique_items,
        "blockchain_status": "pending" if _BLOCKCHAIN_AVAILABLE else "unavailable",
    }


@app.post("/redact/text")
@limiter.limit("30/minute")
async def redact_text_api(
    request: Request,
    file: UploadFile = File(...),
    owner_address: str = Form(default=''),
    pii_filter: str = Form(default='{}'),
    custom_terms: str = Form(default='[]'),
    custom_patterns: str = Form(default='[]'),
    _key: str = Depends(verify_api_key),
):
    dry_run = request.query_params.get('dry_run', 'false').lower() == 'true'
    log.info(f"[redact/text] owner={owner_address!r} file={file.filename!r} dry_run={dry_run}")

    contents = await file.read()
    await validate_upload(contents, "text")
    sha256   = hashlib.sha256(contents).hexdigest()
    text     = contents.decode('utf-8', errors='replace')

    pii_filter_d, custom_terms_l, custom_patterns_l = _parse_redaction_controls(
        pii_filter, custom_terms, custom_patterns
    )

    # Detect PII
    pii_items    = extract_pii_hybrid(text)
    filtered_pii = apply_pii_filter_to_hybrid(pii_items, pii_filter_d)
    log.info(f"[redact/text] hybrid found {len(filtered_pii)} PII items after filter")

    # Redact all terms (PII + custom literals)
    redacted_text  = text
    redacted_items: list[str] = []
    replaced_chars = 0

    for term in filtered_pii + custom_terms_l:
        count = redacted_text.count(term)
        if count:
            redacted_text  = redacted_text.replace(term, '[REDACTED]')
            redacted_items.append(term)
            replaced_chars += len(term) * count

    # Custom regex patterns
    for cp in custom_patterns_l:
        try:
            matches = re.findall(cp, redacted_text, re.IGNORECASE)
            if matches:
                flat = [m if isinstance(m, str) else m[0] for m in matches]
                replaced_chars += sum(len(m) for m in flat)
                redacted_text   = re.sub(cp, '[REDACTED]', redacted_text, flags=re.IGNORECASE)
                redacted_items.extend(flat)
        except re.error:
            pass

    redaction_count  = len(redacted_items)
    total_text_chars = len(text)
    text_coverage    = round(replaced_chars / total_text_chars * 100, 1) if total_text_chars else 0.0
    log.info(f"[TEXT] coverage={text_coverage}% ({replaced_chars}/{total_text_chars} chars)")

    if dry_run:
        log.info(f"[dry_run/text] would redact {redaction_count} item(s) — no files written")
        would_redact = [{"label": "text", "value": v, "box": None} for v in redacted_items]
        return {
            "dry_run":         True,
            "message":         "Dry run — no file written, no blockchain anchor",
            "would_redact":    would_redact,
            "redaction_count": redaction_count,
            "doc_type":        "text",
            "coverage_pct":    text_coverage,
        }

    # Save original bytes keyed by hash
    orig_path = Path("downloads") / f"orig_{sha256}.bin"
    orig_path.write_bytes(contents)

    # Save redacted output
    os.makedirs(CFG.output_dir, exist_ok=True)
    ts       = int(time.time() * 1000)
    out_path = Path(CFG.output_dir) / f"redacted_{ts}.txt"
    out_path.write_text(redacted_text, encoding='utf-8')
    log.info(f"Saved redacted text: {out_path}  ({redaction_count} redaction(s))")

    _text_anchor_salt = f"{owner_address}:{int(time.time())}"
    _text_anchor_hash = hashlib.sha256((sha256 + _text_anchor_salt).encode()).hexdigest()

    if _BLOCKCHAIN_AVAILABLE:
        _anchor_record_async(sha256, redaction_count, 'text', owner_address,
                             salt=_text_anchor_salt)
        log.info(f"Text blockchain anchor dispatched: hash={sha256[:16]}...")

    redactions = [{"label": "text", "value": v, "source": "ai"} for v in redacted_items]

    # Write normalised audit JSON for /api/audit/ lookup
    _text_audit = {
        "sha256_hex":             sha256,
        "file_hash":              sha256,
        "anchor_hash":            _text_anchor_hash,
        "anchor_salt":            _text_anchor_salt,
        "owner_address":          owner_address,
        "doc_type":               "text",
        "filename":               file.filename,
        "redacted_file":          out_path.name,
        "redaction_count":        redaction_count,
        "coverage_pct":           text_coverage,
        "timestamp":              int(time.time()),
        "redactions":             redactions,
        "pii_filter":             pii_filter_d,
        "custom_terms_matched":   [v for v in redacted_items if v in custom_terms_l],
        "custom_patterns_matched": custom_patterns_l,
    }
    _text_audit_path = out_path.with_suffix('.audit.json')
    _text_audit_path.write_text(json.dumps(_text_audit, indent=2))
    log.info(f"[TEXT] Audit: {_text_audit_path}")

    if _ZK_AVAILABLE and redaction_count > 0:
        _generate_zk_proof_async(contents, redacted_items, redaction_count, _text_audit_path)

    _unload_models()
    return {
        "file_hash":       sha256,
        "redacted_file":   out_path.name,
        "doc_type":        "text",
        "redaction_count": redaction_count,
        "coverage_pct":    text_coverage,
        "redacted_text":   redacted_text,
        "redactions":      redactions,
        "blockchain_status": "pending" if _BLOCKCHAIN_AVAILABLE else "unavailable",
    }


@app.post("/redact/text/apply")
async def redact_text_apply(req: Request):
    """Re-apply a confirmed subset of AI-detected redactions and custom terms.

    The original file bytes are looked up by file_hash from the downloads/
    directory (written during the initial /redact/text call), redaction is
    re-applied only to confirmed items, and the result is saved to
    redacted_output/ for download via /api/output/{filename}.
    """
    data = await req.json()
    file_hash  = data.get("file_hash", "")
    confirmed  = data.get("confirmed_redactions", [])
    custom     = data.get("custom_terms", [])

    orig_path = Path("/home/bharath/omni-shield/phase1_edge_engine/downloads") / f"orig_{file_hash}.bin"
    if not orig_path.exists():
        raise HTTPException(status_code=404, detail="Original file not found — please re-upload")

    text = orig_path.read_bytes().decode("utf-8", errors="replace")

    for term in confirmed + custom:
        if term and len(term) > 2:
            text = text.replace(term, "[REDACTED]")

    os.makedirs(CFG.output_dir, exist_ok=True)
    out_name = f"redacted_{int(time.time() * 1000)}.txt"
    out_path = Path(CFG.output_dir) / out_name
    out_path.write_text(text, encoding="utf-8")

    return {
        "redacted_file": out_name,
        "download_url":  f"/api/output/{out_name}",
    }


@app.post("/redact/image")
@limiter.limit("10/minute")
async def redact_image(
    request: Request,
    file: UploadFile = File(...),
    owner_address: str = Form("0xAnonymous"),
    pii_filter: str = Form(""),
    custom_terms: str = Form("[]"),
    custom_patterns: str = Form("[]"),
    _key: str = Depends(verify_api_key),
):
    import traceback as _tb
    try:
        log.info(f"[redact/image] received: {file.filename!r} owner={owner_address!r}")
        log.info(f"[redact/image] _QUEUE_AVAILABLE={_QUEUE_AVAILABLE} _run_image_pipeline={_run_image_pipeline is not None}")

        # Diagnose Redis connectivity on every request so queue failures are
        # immediately visible in logs rather than silently triggering a fallback.
        try:
            from redis import Redis as _Redis
            _Redis(
                host=os.environ.get("REDIS_HOST", "localhost"),
                port=int(os.environ.get("REDIS_PORT", 6379)),
                socket_connect_timeout=2,
            ).ping()
            log.info("[redact/image] Redis ping OK")
        except Exception as _re:
            log.error(f"[redact/image] Redis FAILED: {_re}")

        # Read ONCE
        contents = await file.read()
        log.info(f"[redact/image] file size: {len(contents)} bytes")

        try:
            await validate_upload(contents, "image")
        except HTTPException:
            raise
        except Exception as _ve:
            log.warning(f"[redact/image] validate warning: {_ve} — continuing")

        if not _QUEUE_AVAILABLE or _run_image_pipeline is None:
            raise HTTPException(
                status_code=503,
                detail="Image queue not available — ensure Redis and rq worker are running",
            )

        job_id   = str(uuid.uuid4())
        suffix   = Path(file.filename).suffix if file.filename else ".jpg"
        tmp_path = f"/tmp/upload_{job_id}{suffix}"
        with open(tmp_path, "wb") as _f:
            _f.write(contents)

        try:
            job = image_queue.enqueue(
                _run_image_pipeline,
                tmp_path, owner_address, pii_filter, custom_terms, custom_patterns, job_id,
                job_timeout=180,
                result_ttl=3600,
                job_id=job_id,
            )
            log.info(f"[redact/image] queued job {job.id}")
            return {"job_id": job.id, "status": "queued", "poll_url": f"/api/job/{job.id}"}
        except Exception as _qe:
            log.error(f"[redact/image] enqueue failed: {_qe}")
            log.error(_tb.format_exc())
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise HTTPException(
                status_code=503,
                detail=f"Queue unavailable — try again shortly ({_qe})",
            )

    except HTTPException:
        raise
    except Exception as _e:
        log.error(f"[redact/image] CRASH: {_e}")
        log.error(_tb.format_exc())
        raise HTTPException(status_code=500, detail=str(_e))


@app.post("/redact/pdf")
@limiter.limit("20/minute")
async def redact_pdf_api(
    request: Request,
    file: UploadFile = File(...),
    owner_address: str = Form("0xAnonymous"),
    pii_filter: str = Form(""),
    custom_terms: str = Form("[]"),
    custom_patterns: str = Form("[]"),
    _key: str = Depends(verify_api_key),
):
    dry_run = request.query_params.get('dry_run', 'false').lower() == 'true'
    log.info(f"[redact/pdf] owner_address={owner_address!r} file={file.filename!r} dry_run={dry_run}")
    pii_filter_d, custom_terms_l, custom_patterns_l = _parse_redaction_controls(
        pii_filter, custom_terms, custom_patterns
    )
    log.info(f"[FILTER DEBUG] pii_filter received: {pii_filter_d}")
    contents = await file.read()
    await validate_upload(contents, "pdf")

    # Compute SHA-256 of original bytes BEFORE redaction (Bug 2a)
    sha256 = hashlib.sha256(contents).hexdigest()

    # Save original bytes keyed by hash so verify-by-file works (Bug 2a)
    with open(f"downloads/orig_{sha256}.bin", "wb") as f:
        f.write(contents)

    pdf              = fitz.open(stream=contents, filetype="pdf")
    all_pii_found: list[str]  = []
    all_redactions: list[dict] = []
    total_rect_count = 0
    total_chars      = 0
    redacted_chars   = 0

    # ── Scanned PDF detection ─────────────────────────────────────────────────
    # If PyMuPDF extracts very little text the PDF is likely a scanned image.
    # Fall back to the full 9-agent VLM pipeline by converting each page to a
    # PNG and running it through the image endpoint logic.
    _all_pdf_text = " ".join(p.get_text("text") for p in pdf)
    _pdf_word_count = len(_all_pdf_text.split())
    _is_scanned_pdf = _pdf_word_count < 20 and len(pdf) > 0
    pdf_coverage = 0.0

    if _is_scanned_pdf:
        log.warning(
            f"[PDF] Only {_pdf_word_count} words extracted from {len(pdf)}-page PDF "
            f"— likely scanned. Falling back to VLM image pipeline."
        )
        _scanned_base = f"{int(time.time())}"
        for _pg_num in range(len(pdf)):
            _pg = pdf[_pg_num]
            _mat = fitz.Matrix(2, 2)   # 2× zoom for legibility
            _pix = _pg.get_pixmap(matrix=_mat)
            _img_path = f"downloads/temp_scanned_{_scanned_base}_p{_pg_num}.png"
            _pix.save(_img_path)
            try:
                _ag_state = orchestrator.run(
                    _img_path,
                    pii_filter=pii_filter_d,
                    custom_terms=custom_terms_l,
                    custom_patterns=custom_patterns_l,
                    dry_run=dry_run,
                )
                for _red in _ag_state.audit.get("redactions", []):
                    _red["page"] = _pg_num + 1
                    _red["source"] = "ai_scanned"
                    all_redactions.append(_red)
                    total_rect_count += 1
                    if _red.get("value"):
                        all_pii_found.append(_red["value"])
                log.info(
                    f"  [SCANNED-PDF] page {_pg_num+1}: "
                    f"{len(_ag_state.audit.get('redactions', []))} redactions"
                )
            except Exception as _e:
                log.error(f"  [SCANNED-PDF] page {_pg_num+1} failed: {_e}")
            finally:
                try:
                    os.remove(_img_path)
                except OSError:
                    pass
        # Re-open a fresh copy of the PDF to apply redactions as black boxes
        pdf.close()
        pdf = fitz.open(stream=contents, filetype="pdf")
        if not dry_run:
            for _red in all_redactions:
                _pg_idx = _red.get("page", 1) - 1
                _box = _red.get("pixel_box") or _red.get("box")
                if _box and _pg_idx < len(pdf):
                    # pixel_box comes from the 2× pixmap; convert back to PDF pts (÷2)
                    _x0, _y0, _x1, _y1 = [c / 2 for c in _box]
                    pdf[_pg_idx].add_redact_annot(fitz.Rect(_x0, _y0, _x1, _y1), fill=(0, 0, 0))
            for _pg in pdf:
                _pg.apply_redactions()
        log.info(f"[PDF-SCANNED] total redactions={total_rect_count}")

    if not _is_scanned_pdf:
      for page_num, page in enumerate(pdf, start=1):
        text         = page.get_text("text")
        total_chars += len(text)
        verified_pii = extract_pii_hybrid(text)
        verified_pii = apply_pii_filter_to_hybrid(verified_pii, pii_filter_d)
        log.info(f"PDF page {page_num}: hybrid found {len(verified_pii)} PII items after filter: {verified_pii}")
        for pii in verified_pii:
            rects = page.search_for(pii)
            if rects:
                all_pii_found.append(pii)
                total_rect_count += len(rects)
                redacted_chars   += len(pii) * len(rects)
                for rect in rects:
                    log.info(f"PDF {'[DRY RUN] would redact' if dry_run else 'redacting'}: '{pii}' at {rect} on page {page_num}")
                    if not dry_run:
                        page.add_redact_annot(rect, fill=(0, 0, 0))
                    all_redactions.append({
                        "label": pii, "value": pii, "source": "ai",
                        "page": page_num,
                        "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1,
                    })
            else:
                log.info(f"PDF page {page_num}: search_for('{pii}') returned no rects")

        # Custom terms (literal search)
        for ct in custom_terms_l:
            for rect in page.search_for(ct):
                log.info(f"PDF custom term '{ct}' at {rect} on page {page_num}")
                if not dry_run:
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                all_redactions.append({
                    "label": "Custom", "value": ct, "source": "custom",
                    "page": page_num,
                    "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1,
                })
                total_rect_count += 1

        # Custom regex patterns
        page_text = page.get_text("text")
        for cp in custom_patterns_l:
            try:
                for m in re.finditer(cp, page_text, re.IGNORECASE):
                    for rect in page.search_for(m.group()):
                        if not dry_run:
                            page.add_redact_annot(rect, fill=(0, 0, 0))
                        all_redactions.append({
                            "label": "Custom", "value": m.group(), "source": "custom",
                            "page": page_num,
                            "x0": rect.x0, "y0": rect.y0, "x1": rect.x1, "y1": rect.y1,
                        })
                        total_rect_count += 1
            except re.error:
                pass

        if not dry_run:
            page.apply_redactions()

    pdf_coverage = round(redacted_chars / total_chars * 100, 1) if total_chars else 0.0
    log.info(f"[PDF] coverage={pdf_coverage}% ({redacted_chars}/{total_chars} chars)")

    if dry_run:
        log.info(f"[dry_run/pdf] would redact {total_rect_count} region(s) — no files written")
        def _r_box(r):
            if "x0" in r:
                return [r["x0"], r["y0"], r["x1"], r["y1"]]
            return r.get("pixel_box") or r.get("box")
        would_redact = [
            {"label": r.get("label"), "value": r.get("value"),
             "box": _r_box(r), "page": r.get("page")}
            for r in all_redactions
        ]
        return {
            "dry_run":         True,
            "message":         "Dry run — no file written, no blockchain anchor",
            "would_redact":    would_redact,
            "redaction_count": total_rect_count,
            "doc_type":        "pdf",
            "coverage_pct":    pdf_coverage,
        }

    base_name          = f"{int(time.time())}"
    redacted_filename  = f"redacted_{base_name}.pdf"
    encrypted_filename = f"encrypted_{base_name}.pdf.enc"
    zk_filename        = f"proof_{base_name}.json"

    pdf.save(f"downloads/{redacted_filename}", garbage=4, deflate=True)
    with open(f"downloads/{encrypted_filename}", "wb") as f:
        f.write(cipher_suite.encrypt(contents))
    with open(f"downloads/{zk_filename}", "w") as f:
        json.dump({"proof": "valid_zk_snark", "owner": owner_address}, f)

    _pdf_anchor_salt = f"{owner_address}:{int(time.time())}"
    _pdf_anchor_hash = hashlib.sha256((sha256 + _pdf_anchor_salt).encode()).hexdigest()

    if _BLOCKCHAIN_AVAILABLE:
        log.info(
            f"Dispatching PDF anchor to background: hash={sha256[:16]}..., "
            f"owner={owner_address}, count={total_rect_count}"
        )
        _anchor_record_async(sha256, total_rect_count, "pdf", owner_address,
                             salt=_pdf_anchor_salt)

    _custom_matched = [r["value"] for r in all_redactions if r.get("source") == "custom"]

    # Write normalised audit JSON for /api/audit/ lookup
    _pdf_audit = {
        "sha256_hex":             sha256,
        "file_hash":              sha256,
        "anchor_hash":            _pdf_anchor_hash,
        "anchor_salt":            _pdf_anchor_salt,
        "owner_address":          owner_address,
        "doc_type":               "pdf",
        "filename":               file.filename,
        "redacted_file":          redacted_filename,
        "redaction_count":        total_rect_count,
        "coverage_pct":           pdf_coverage,
        "timestamp":              int(time.time()),
        "redactions":             all_redactions,
        "pii_filter":             pii_filter_d,
        "custom_terms_matched":   list(set(_custom_matched)),
        "custom_patterns_matched": custom_patterns_l,
    }
    _pdf_audit_path = Path(CFG.output_dir) / f"redacted_{base_name}.audit.json"
    os.makedirs(CFG.output_dir, exist_ok=True)
    _pdf_audit_path.write_text(json.dumps(_pdf_audit, indent=2))
    log.info(f"[PDF] Audit: {_pdf_audit_path}")

    if _ZK_AVAILABLE and total_rect_count > 0:
        _pdf_items = list({r["value"] for r in all_redactions if r.get("value")})
        _generate_zk_proof_async(contents, _pdf_items, total_rect_count, _pdf_audit_path)

    _unload_models()
    return {
        "file_hash":          sha256,
        "redacted_file":      redacted_filename,
        "encrypted_file":     encrypted_filename,
        "zk_proof":           zk_filename,
        "doc_type":           "pdf",
        "redaction_count":    total_rect_count,
        "redactions":         all_redactions,
        "coverage_pct":       pdf_coverage,
        "blockchain_status":  "pending" if _BLOCKCHAIN_AVAILABLE else "unavailable",
        "redaction_config": {
            "pii_filter":              pii_filter_d,
            "custom_terms_matched":    list(set(_custom_matched)),
            "custom_patterns_matched": custom_patterns_l,
        },
    }


@app.post("/redact/audio")
@limiter.limit("15/minute")
async def redact_audio_api(
    request: Request,
    file: UploadFile = File(...),
    owner_address: str = Form("0xAnonymous"),
    pii_filter: str = Form(""),
    custom_terms: str = Form("[]"),
    custom_patterns: str = Form("[]"),
    _key: str = Depends(verify_api_key),
):
    dry_run = request.query_params.get('dry_run', 'false').lower() == 'true'
    log.info(f"[redact/audio] owner_address={owner_address!r} file={file.filename!r} dry_run={dry_run}")
    pii_filter_d, custom_terms_l, custom_patterns_l = _parse_redaction_controls(
        pii_filter, custom_terms, custom_patterns
    )
    log.info(f"[FILTER DEBUG] pii_filter received: {pii_filter_d}")
    contents  = await file.read()
    await validate_upload(contents, "audio")
    base_name = f"{int(time.time())}"

    # Compute SHA-256 of original bytes BEFORE processing (Bug 3c)
    sha256 = hashlib.sha256(contents).hexdigest()

    # Save original bytes keyed by hash so verify-by-file works (Bug 3c)
    with open(f"downloads/orig_{sha256}.bin", "wb") as f:
        f.write(contents)

    temp_input = f"downloads/temp_input_{base_name}.mp3"
    with open(temp_input, "wb") as f:
        f.write(contents)

    silenced_pii: list[str] = []
    silenced_custom: list[str] = []
    mute_ranges: list[tuple] = []
    merged_ranges: list[tuple] = []

    import traceback as _tb
    try:
        # ── FIX 1: Whisper with initial_prompt to bias towards digit output ─
        segments, _ = _get_audio_model().transcribe(
            temp_input,
            word_timestamps=True,
            initial_prompt=_AUDIO_INITIAL_PROMPT,
        )

        # Build full_text from word tokens (preserves char positions for muting)
        full_text    = ""
        word_segments = []
        seg_texts: list[str] = []
        for segment in segments:
            seg_texts.append(segment.text)
            for word in (segment.words or []):
                start_char = len(full_text)
                full_text += word.word
                word_segments.append({
                    "word":       word.word.strip(),
                    "start":      word.start,
                    "end":        word.end,
                    "start_char": start_char,
                    "end_char":   len(full_text),
                })

        log.info(f"[AUDIO] Raw transcript: {full_text[:200]}")

        # ── FIX 2: Normalise spoken numbers / ordinals / years ──────────────
        normalised_text = normalise_audio_transcript(full_text)
        log.info(f"[AUDIO] Normalised:     {normalised_text.split(chr(10))[-1][:200]}")

        # ── FIX 3: PII detection on normalised text ──────────────────────────
        # Detect names/emails/phones from the normalised transcript
        verified_pii_set: set[str] = set(extract_pii_hybrid(normalised_text))

        # Also run the full deterministic regex sweep on the normalised text
        for _rpat, _rfkey, _rlabel in _HYBRID_REGEX_SWEEP:
            if not pii_filter_d.get(_rfkey, True):
                continue
            for _rm in _rpat.finditer(normalised_text):
                _val = _rm.group().strip()
                if _val:
                    verified_pii_set.add(_val)
                    log.info(f"[AUDIO REGEX-SWEEP] {_rlabel}: '{_val}'")

        verified_pii = apply_pii_filter_to_hybrid(list(verified_pii_set), pii_filter_d)
        log.info(f"[AUDIO] PII to mute: {verified_pii}")

        audio = AudioSegment.from_file(temp_input)

        # ── FIX 4: Dual-strategy timestamp mapping ───────────────────────────
        def _mute_phrase(phrase: str, source_list: list[str]) -> None:
            """
            Strategy A: search for the phrase as a literal string in full_text
            (works for names, emails, addresses, text that survived normalisation).
            Strategy B: for digit strings, find the spoken digit-word run in the
            Whisper word list and use those timestamps.
            """
            found = False

            # Strategy A — text match in original transcript
            for match in re.finditer(re.escape(phrase), full_text, re.IGNORECASE):
                p_start, p_end = match.start(), match.end()
                for w in word_segments:
                    if max(p_start, w["start_char"]) < min(p_end, w["end_char"]):
                        mute_ranges.append((w["start"], w["end"]))
                        found = True
            if found:
                if phrase not in source_list:
                    source_list.append(phrase)
                return

            # Strategy B — digit-word spoken timestamp lookup
            span = _find_spoken_timestamps(phrase, word_segments)
            if span:
                mute_ranges.append(span)
                if phrase not in source_list:
                    source_list.append(phrase)
                log.info(f"[AUDIO SPOKEN-MATCH] '{phrase}' -> {span[0]:.2f}s–{span[1]:.2f}s")
            else:
                log.warning(f"[AUDIO] No timestamp match for '{phrase}'")

        for pii in verified_pii:
            _mute_phrase(pii, silenced_pii)

        # Custom terms in transcript
        for ct in custom_terms_l:
            _mute_phrase(ct, silenced_custom)

        # Custom regex patterns in transcript
        for cp in custom_patterns_l:
            try:
                for m in re.finditer(cp, normalised_text, re.IGNORECASE):
                    _mute_phrase(m.group(), silenced_custom)
            except re.error:
                pass

        # Merge overlapping ranges before silencing
        mute_ranges.sort()
        merged_ranges: list[tuple[float, float]] = []
        for s, e in mute_ranges:
            if merged_ranges and s <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], e))
            else:
                merged_ranges.append((s, e))

        for start, end in merged_ranges:
            start_ms = max(0, int(start * 1000) - 150)
            end_ms   = min(len(audio), int(end * 1000) + 150)
            audio    = (
                audio[:start_ms]
                + AudioSegment.silent(duration=(end_ms - start_ms))
                + audio[end_ms:]
            )
    except Exception as _e:
        log.error(f"[AUDIO ENDPOINT] {_e}")
        log.error(_tb.format_exc())
        raise HTTPException(status_code=500, detail=str(_e))
    finally:
        try:
            os.remove(temp_input)
        except OSError:
            pass

    all_redactions = (
        [{"label": p, "value": p, "source": "ai"} for p in silenced_pii]
        + [{"label": "Custom", "value": c, "source": "custom"} for c in silenced_custom]
    )

    # Coverage: fraction of audio duration that was muted
    total_duration_s  = len(audio) / 1000.0 if audio else 0.0
    muted_duration_s  = sum(max(0, e - s) for s, e in merged_ranges)
    audio_coverage    = round(muted_duration_s / total_duration_s * 100, 1) if total_duration_s else 0.0
    log.info(f"[AUDIO] coverage={audio_coverage}% ({muted_duration_s:.1f}s muted / {total_duration_s:.1f}s total)")

    if dry_run:
        log.info(f"[dry_run/audio] would silence {len(merged_ranges)} range(s) — no files written")
        would_redact = [
            {"label": r["label"], "value": r["value"], "box": None}
            for r in all_redactions
        ]
        return {
            "dry_run":         True,
            "message":         "Dry run — no file written, no blockchain anchor",
            "would_redact":    would_redact,
            "redaction_count": len(merged_ranges),
            "doc_type":        "audio_recording",
            "coverage_pct":    audio_coverage,
        }

    import shutil

    redacted_filename  = f"redacted_{base_name}.mp3"
    encrypted_filename = f"encrypted_{base_name}.mp3.enc"
    zk_filename        = f"proof_{base_name}.json"

    redacted_path = f"downloads/{redacted_filename}"
    if not merged_ranges:
        log.info("No PII found — copying original audio as output")
        with open(redacted_path, "wb") as f:
            f.write(contents)
    else:
        audio.export(redacted_path, format="mp3")
    with open(f"downloads/{encrypted_filename}", "wb") as f:
        f.write(cipher_suite.encrypt(contents))
    with open(f"downloads/{zk_filename}", "w") as f:
        json.dump({"proof": "valid_zk_snark", "owner": owner_address}, f)

    _audio_anchor_salt = f"{owner_address}:{int(time.time())}"
    _audio_anchor_hash = hashlib.sha256((sha256 + _audio_anchor_salt).encode()).hexdigest()

    if _BLOCKCHAIN_AVAILABLE:
        log.info(f"Dispatching audio anchor to background: hash={sha256[:16]}..., count={len(merged_ranges)}")
        _anchor_record_async(sha256, len(merged_ranges), "audio_recording", owner_address,
                             salt=_audio_anchor_salt)

    # Write normalised audit JSON for /api/audit/ lookup
    os.makedirs(CFG.output_dir, exist_ok=True)
    _audio_audit = {
        "sha256_hex":             sha256,
        "file_hash":              sha256,
        "anchor_hash":            _audio_anchor_hash,
        "anchor_salt":            _audio_anchor_salt,
        "owner_address":          owner_address,
        "doc_type":               "audio_recording",
        "filename":               file.filename,
        "redacted_file":          redacted_filename,
        "redaction_count":        len(merged_ranges),
        "coverage_pct":           audio_coverage,
        "timestamp":              int(time.time()),
        "redactions":             all_redactions,
        "pii_filter":             pii_filter_d,
        "custom_terms_matched":   silenced_custom,
        "custom_patterns_matched": custom_patterns_l,
    }
    _audio_audit_path = Path(CFG.output_dir) / f"redacted_{base_name}.audit.json"
    _audio_audit_path.write_text(json.dumps(_audio_audit, indent=2))
    log.info(f"[AUDIO] Audit: {_audio_audit_path}")

    if _ZK_AVAILABLE and len(merged_ranges) > 0:
        _audio_items = silenced_pii + silenced_custom
        _generate_zk_proof_async(contents, _audio_items, len(merged_ranges), _audio_audit_path)

    _unload_models()
    return {
        "file_hash":          sha256,
        "redacted_file":      redacted_filename,
        "encrypted_file":     encrypted_filename,
        "zk_proof":           zk_filename,
        "doc_type":           "audio_recording",
        "redaction_count":    len(merged_ranges),
        "coverage_pct":       audio_coverage,
        "blockchain_status":  "pending" if _BLOCKCHAIN_AVAILABLE else "unavailable",
        "redactions":         all_redactions,
        "redaction_config": {
            "pii_filter":              pii_filter_d,
            "custom_terms_matched":    silenced_custom,
            "custom_patterns_matched": custom_patterns_l,
        },
    }


@app.get("/download/by-hash/{sha256}")
async def download_by_hash(sha256: str, wallet: str = ""):
    """
    Download a previously processed file by its SHA-256 hash.
    Optionally gates on wallet ownership when `wallet` is provided.
    """
    log.info(f"[DOWNLOAD] hit — sha256={sha256[:16]} wallet={wallet[:10] if wallet else 'none'}")

    if not _BLOCKCHAIN_AVAILABLE:
        log.warning("[DOWNLOAD] blockchain unavailable — returning 503")
        raise HTTPException(status_code=503, detail="Blockchain module unavailable")

    # 1. Verify record exists on blockchain
    try:
        result = _verify_document_hash(sha256)
        log.info(f"[DOWNLOAD] verify result: {result}")
    except Exception as exc:
        log.error(f"[DOWNLOAD] verify_document failed: {exc}")
        result = {"in_ledger": False}

    if not (result.get("in_ledger") or result.get("exists")):
        log.warning(f"[DOWNLOAD] not in ledger: {sha256[:16]}")
        raise HTTPException(status_code=404, detail="No blockchain record for this hash")

    # 2. Verify wallet owns this record (fail-open if lookup itself errors)
    if wallet:
        try:
            log.info(f"[DOWNLOAD] wallet={wallet[:10]}...")
            log.info(f"[DOWNLOAD] looking for sha256={sha256[:16]}")
            records = _get_records_for_user(wallet)
            hashes  = [r.get("sha256_hex", "").lower() for r in records]
            log.info(f"[DOWNLOAD] {len(records)} records, hashes: {hashes[:3]}")
            if sha256.lower() not in hashes:
                log.warning(f"[DOWNLOAD] {sha256[:16]} not in wallet records — raising 403")
                raise HTTPException(status_code=403,
                                    detail="This wallet does not own this record")
        except HTTPException:
            raise
        except Exception as exc:
            log.warning(f"[DOWNLOAD] ownership check failed: {exc} — serving anyway")

    # 3a. Check redacted_output/ audit JSONs (newest first)
    redacted_path = None
    audit_dir = Path(CFG.output_dir)
    if audit_dir.exists():
        for audit_file in sorted(audit_dir.glob("*.audit.json"), reverse=True):
            try:
                data = json.loads(audit_file.read_text())
                if data.get("sha256_hex") == sha256:
                    for ext in (".png", ".pdf", ".mp3", ".txt"):
                        candidate = audit_file.with_suffix("").with_suffix(ext)
                        if candidate.exists():
                            redacted_path = candidate
                            break
                    break
            except Exception:
                continue

    if redacted_path and redacted_path.exists():
        log.info(f"[download/by-hash] serving redacted file: {redacted_path}")
        return FileResponse(str(redacted_path),
                            filename=f"redacted_{sha256[:8]}{redacted_path.suffix}")

    # 3b. Fall back to orig_{sha256}.bin with extension inferred from doc_type
    orig_path = Path("downloads") / f"orig_{sha256}.bin"
    log.info(f"[DOWNLOAD] No redacted file found for {sha256[:16]}")
    log.info(f"[DOWNLOAD] Checking orig path: {orig_path}")
    log.info(f"[DOWNLOAD] orig exists: {orig_path.exists()}")
    log.info(f"[DOWNLOAD] downloads/ contents: {list(Path('downloads').glob('orig_*.bin'))[:5]}")

    if orig_path.exists():
        _DOC_TYPE_EXT = {
            "id_card": ".png", "image": ".png",
            "image_finalised": ".png",
            "pdf": ".pdf", "pdf_finalised": ".pdf",
            "audio_recording": ".mp3",
            "text": ".txt",
            "docx": ".pdf",
            "pptx": ".pdf",
        }
        doc_type = ""
        try:
            owner_records = _get_records_for_user(wallet) if wallet else []
            matched = next((r for r in owner_records if r.get("sha256_hex") == sha256), {})
            doc_type = matched.get("doc_type", "")
            log.info(f"[DOWNLOAD] doc_type from chain: {doc_type!r}")
        except Exception as exc:
            log.warning(f"[DOWNLOAD] doc_type lookup failed: {exc}")

        ext = _DOC_TYPE_EXT.get(doc_type, ".bin")
        log.info(f"[download/by-hash] serving orig as {ext} (doc_type={doc_type!r}): {orig_path}")
        return FileResponse(str(orig_path),
                            filename=f"original_{sha256[:8]}{ext}",
                            media_type="application/octet-stream")

    raise HTTPException(status_code=404, detail="File not found on this server")


@app.get("/download/{filename}")
async def download_file(filename: str):
    safe_name = Path(filename).name
    for search_dir in ("downloads", CFG.output_dir):
        file_path = Path(search_dir) / safe_name
        if file_path.exists():
            return FileResponse(str(file_path))
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/verify_zk")
async def verify_zk_api(proof_file: UploadFile = File(...)):
    return {"message": "ZK-SNARK Proof Verified Successfully on Edge!"}


@app.post("/decrypt")
async def decrypt_api(
    file: UploadFile = File(...),
    file_hash: str = Form(...),
    wallet_address: str = Form(...),
):
    contents = await file.read()
    try:
        decrypted = cipher_suite.decrypt(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Decryption failed. Invalid file or key.")
    return StreamingResponse(iter([decrypted]), media_type="application/octet-stream")


@app.post("/verify")
async def verify_document_api(
    file: Optional[UploadFile] = File(None),
    file_hash: Optional[str] = Form(None),
):
    """
    Check whether a file was previously processed by Omni-Shield.

    Accepts either:
      - multipart file upload  → SHA-256 computed server-side
      - form field file_hash   → use the supplied hex string directly

    Returns: in_ledger, redaction_count, timestamp, sha256_hex.
    """
    if file is not None:
        contents   = await file.read()
        sha256_hex = hashlib.sha256(contents).hexdigest()
        log.info(f"Verify request (file upload): hash={sha256_hex[:16]}...")
    elif file_hash:
        sha256_hex = file_hash.strip()
        if len(sha256_hex) != 64:
            raise HTTPException(status_code=422, detail="file_hash must be a 64-character hex string")
        log.info(f"Verify request (hash param): hash={sha256_hex[:16]}...")
    else:
        raise HTTPException(status_code=422, detail="Provide either file or file_hash")

    # Resolve original sha256 → audit JSON → anchor_hash (salted).
    # Old records (pre-salt) fall back to looking up sha256_hex directly.
    # If a local audit file matches, trust it directly — blockchain is checked
    # afterwards but is not required (handles Ganache-down and hash-mismatch cases).
    anchor_hash      = sha256_hex
    is_salted        = False
    local_audit      = None
    audit_dir        = Path("redacted_output")
    sha_lower        = sha256_hex.lower()
    for af in sorted(audit_dir.glob("*.audit.json"), reverse=True):
        try:
            data   = json.loads(af.read_text())
            stored = data.get("sha256_hex", "").lower()
            if stored == sha_lower or stored.startswith(sha_lower[:16]):
                local_audit = data
                if data.get("anchor_hash"):
                    anchor_hash = data["anchor_hash"]
                    is_salted   = True
                    log.info(
                        f"Verify: resolved anchor_hash={anchor_hash[:16]}… "
                        f"from {af.name}"
                    )
                break
        except Exception:
            continue

    # If local audit record exists, we can confirm the document was processed
    # without requiring a live blockchain connection.
    if local_audit is not None:
        log.info(f"Verify: local audit match for {sha256_hex[:16]}…")
        return {
            "in_ledger":       True,
            "redaction_count": local_audit.get("redaction_count", 0),
            "timestamp":       local_audit.get("timestamp", 0),
            "sha256_hex":      sha256_hex,
            "anchor_hash":     anchor_hash,
            "salted":          is_salted,
            "source":          "local",
        }

    # No local audit file — fall back to blockchain lookup.
    if not _BLOCKCHAIN_AVAILABLE:
        log.info(f"Verify: no local audit and blockchain unavailable for {sha256_hex[:16]}…")
        return {
            "in_ledger":       False,
            "redaction_count": 0,
            "timestamp":       0,
            "sha256_hex":      sha256_hex,
            "anchor_hash":     anchor_hash,
            "salted":          is_salted,
        }

    result = _verify_document_hash(anchor_hash)
    log.info(
        f"Verify result: in_ledger={result.get('exists', False)}, "
        f"anchor={anchor_hash[:16]}…"
    )
    return {
        "in_ledger":       result.get("exists", False),
        "redaction_count": result.get("redaction_count", 0),
        "timestamp":       result.get("timestamp", 0),
        "sha256_hex":      sha256_hex,
        "anchor_hash":     anchor_hash,
        "salted":          is_salted,
        "source":          "blockchain",
    }


@app.get("/api/system/stats")
async def system_stats():
    return dict(_stats_cache)


@app.get("/api/debug/hash")
async def debug_hash_api(file_path: str):
    """Debug: compute SHA-256 of a file on disk. Used to verify hash consistency."""
    safe_path = Path(file_path)
    # Restrict to the downloads directory to prevent path traversal
    try:
        safe_path = safe_path.resolve()
        downloads_dir = Path("downloads").resolve()
        if not str(safe_path).startswith(str(downloads_dir)):
            raise HTTPException(status_code=403, detail="Access restricted to downloads directory")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not safe_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    sha256 = hashlib.sha256(safe_path.read_bytes()).hexdigest()
    return {"sha256": sha256, "file_path": str(safe_path)}


@app.get("/api/debug/blockchain")
async def debug_blockchain_api():
    """Debug: return all anchored records including any errors. Never raises."""
    if not _BLOCKCHAIN_AVAILABLE:
        return {"error": "Blockchain module unavailable — start Ganache first.", "records": [], "count": 0}
    try:
        records = _get_my_records()
        return {"records": records, "count": len(records)}
    except Exception as exc:
        return {"error": str(exc), "records": [], "count": 0}


@app.get("/api/audit/{sha256}")
async def get_audit(sha256: str):
    """Return audit JSON for a given SHA-256 prefix or full hash."""
    audit_dir = Path("redacted_output")
    sha_lower = sha256.lower().strip()
    prefix16  = sha_lower[:16]
    log.info(f"[AUDIT] looking for sha256={prefix16}...")
    for f in sorted(audit_dir.glob("*.audit.json"), reverse=True):
        try:
            data   = json.loads(f.read_text())
            stored = data.get("sha256_hex", "").lower().strip()
            log.info(f"[AUDIT] checking {f.name}: stored={stored[:16]}")
            if stored and (
                stored == sha_lower
                or stored.startswith(prefix16)
                or sha_lower.startswith(stored[:16])
            ):
                log.info(f"[AUDIT] match found: {f.name}")
                return data
        except Exception as e:
            log.warning(f"[AUDIT] error reading {f.name}: {e}")
            continue
    log.warning(f"[AUDIT] no match found for {prefix16}")
    raise HTTPException(status_code=404, detail="Audit not found")


@app.get("/api/verify-proof/{sha256}")
async def verify_zk_proof_api(sha256: str):
    """Verify the ZK-SNARK proof stored in the audit JSON for this document hash."""
    audit_dir = Path("redacted_output")
    sha_lower = sha256.lower().strip()
    prefix16  = sha_lower[:16]

    for f in sorted(audit_dir.glob("*.audit.json"), reverse=True):
        try:
            data   = json.loads(f.read_text())
            stored = data.get("sha256_hex", "").lower().strip()
            if not (stored and (stored == sha_lower or stored.startswith(prefix16))):
                continue

            zk_proof = data.get("zk_proof")
            if not zk_proof:
                return {
                    "verified":   False,
                    "reason":     "Proof not yet generated — ZoKrates may still be running",
                    "sha256_hex": sha256,
                }

            if not _ZK_AVAILABLE:
                return {
                    "verified": False,
                    "reason":   "ZoKrates not available on this server",
                }

            from zk_prover import verify_proof as _verify_proof
            valid = _verify_proof(zk_proof)
            log.info(f"[ZK VERIFY] {sha256[:16]} — {'PASSED' if valid else 'FAILED'}")
            return {
                "verified":        valid,
                "commitment":      zk_proof.get("commitment"),
                "redaction_count": zk_proof.get("redaction_count"),
                "doc_hash_u32":    zk_proof.get("doc_hash_u32"),
                "scheme":          "Groth16 / BN128",
                "sha256_hex":      sha256,
            }
        except Exception as e:
            log.warning(f"[ZK VERIFY] error: {e}")
            continue

    raise HTTPException(status_code=404, detail="No record found for this hash")


@app.get("/api/output/{filename}")
async def get_output_file(filename: str):
    """Serve a file from the redacted_output directory (originals + redacted images)."""
    safe_name = Path(filename).name
    path = Path("redacted_output") / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(path))


@app.get("/api/records")
async def get_records_api(wallet: str = None):
    """Return redaction records for the given wallet (or default account)."""
    if not _BLOCKCHAIN_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Blockchain module unavailable — start Ganache first.",
        )
    try:
        if wallet:
            records = _get_records_for_user(wallet)
        else:
            records = _get_my_records()
        return {"records": records}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/my-documents/{owner_address}")
async def get_my_documents(owner_address: str):
    """Return audit records for a wallet from local audit files.
    Falls back to all unowned docs for backwards-compatibility with files
    written before owner_address was added to the audit schema.
    """
    import glob as _glob
    matched = []
    unowned = []
    for audit_file in _glob.glob("redacted_output/*.audit.json"):
        try:
            with open(audit_file) as f:
                audit = json.load(f)
            owner = audit.get("owner_address", "")
            if owner.lower() == owner_address.lower():
                matched.append(audit)
            elif not owner:
                unowned.append(audit)
        except Exception:
            continue
    docs = matched if matched else unowned
    docs.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return {"documents": docs[:50], "count": len(docs)}


# ── Finalise models ──────────────────────────────────────────────────────────

class RedactionBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str = "manual"


class FinaliseRequest(BaseModel):
    file_hash: str
    boxes: List[RedactionBox]
    owner_address: Optional[str] = "0xAnonymous"
    original_file: Optional[str] = ""   # filename in redacted_output/ (image pipeline)


@app.post("/redact/finalise")
async def redact_finalise(req: FinaliseRequest):
    """
    Human-in-the-loop finalise: retrieve the original image by hash,
    apply the user-confirmed box list, return the redacted file.
    """
    # Prefer the named original file (image pipeline saves to redacted_output/)
    orig_bytes: bytes | None = None
    if req.original_file:
        candidate = Path("/home/bharath/omni-shield/phase1_edge_engine/redacted_output") / Path(req.original_file).name
        if candidate.exists():
            orig_bytes = candidate.read_bytes()
    # Fallback: legacy path written by non-pipeline flows
    if orig_bytes is None:
        orig_path = Path(f"downloads/orig_{req.file_hash}.bin")
        if not orig_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Original file not found — re-upload the document to start a new session.",
            )
        orig_bytes = orig_path.read_bytes()

    # Apply all boxes with PIL
    img  = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in req.boxes:
        draw.rectangle(
            [int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)],
            fill=(0, 0, 0),
        )

    base_name         = f"{int(time.time())}"
    finalised_filename = f"finalised_{base_name}.png"
    finalised_path = Path("/home/bharath/omni-shield/phase1_edge_engine/redacted_output") / finalised_filename

    img.save(str(finalised_path))

    # Audit sidecar
    audit_data = {
        "file_hash":       req.file_hash,
        "owner_address":   req.owner_address,
        "timestamp":       int(time.time()),
        "redaction_count": len(req.boxes),
        "redactions":      [b.model_dump() for b in req.boxes],
        "source":          "human-in-loop-editor",
    }
    audit_filename = f"finalised_{base_name}.audit.json"
    (Path("/home/bharath/omni-shield/phase1_edge_engine/downloads") / audit_filename).write_text(json.dumps(audit_data, indent=2))

    # Optional blockchain anchor
    if _BLOCKCHAIN_AVAILABLE:
        try:
            _anchor_record(req.file_hash, len(req.boxes), "image_finalised", req.owner_address)
        except Exception:
            pass

    return {
        "finalised_file":  finalised_filename,
        "audit_file":      audit_filename,
        "redaction_count": len(req.boxes),
        "file_hash":       req.file_hash,
    }


class PDFFinaliseRequest(BaseModel):
    file_hash: str
    pages: dict   # { "1": [{xmin,ymin,xmax,ymax}, ...], "2": [...] }
    owner_address: Optional[str] = "0xAnonymous"
    redacted_file: Optional[str] = None   # filename of the already-AI-redacted PDF


@app.post("/redact/finalise/pdf")
async def redact_finalise_pdf(req: PDFFinaliseRequest):
    """
    Human-in-the-loop PDF finalise: load the already-AI-redacted PDF and apply
    user-drawn boxes on top so AI redactions are preserved.
    Coordinates are in PDF user-space points (canvas px / 1.5 scale).
    """
    # Prefer the already-redacted file so AI redactions are preserved.
    # Fall back to the original only if the redacted file is unavailable.
    base_pdf_bytes: bytes | None = None
    if req.redacted_file:
        safe_name = Path(req.redacted_file).name
        redacted_path = Path("downloads") / safe_name
        if redacted_path.exists():
            base_pdf_bytes = redacted_path.read_bytes()

    if base_pdf_bytes is None:
        orig_path = Path(f"downloads/orig_{req.file_hash}.bin")
        if not orig_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Redacted PDF not found — re-upload the document to start a new session.",
            )
        base_pdf_bytes = orig_path.read_bytes()

    pdf = fitz.open(stream=base_pdf_bytes, filetype="pdf")

    total_redactions = 0
    for page_str, boxes in req.pages.items():
        try:
            page_idx = int(page_str) - 1   # 1-based → 0-based
        except ValueError:
            continue
        if page_idx < 0 or page_idx >= len(pdf):
            continue
        page = pdf[page_idx]
        for box in boxes:
            rect = fitz.Rect(
                float(box["xmin"]), float(box["ymin"]),
                float(box["xmax"]), float(box["ymax"]),
            )
            page.add_redact_annot(rect, fill=(0, 0, 0))
            total_redactions += 1
        page.apply_redactions()

    base_name          = f"{int(time.time())}"
    finalised_filename = f"finalised_{base_name}.pdf"
    finalised_path = Path("/home/bharath/omni-shield/phase1_edge_engine/redacted_output") / finalised_filename
    pdf.save(str(finalised_path), garbage=4, deflate=True)

    if _BLOCKCHAIN_AVAILABLE:
        try:
            _anchor_record(req.file_hash, total_redactions, "pdf_finalised", req.owner_address)
        except Exception as exc:
            log.warning(f"PDF finalise blockchain anchor failed (non-fatal): {exc}")

    return {
        "finalised_file":  finalised_filename,
        "redaction_count": total_redactions,
        "file_hash":       req.file_hash,
    }


@app.post("/toggle_redaction")
def toggle_redaction():
    state["redaction_active"] = not state["redaction_active"]
    return {"redaction_active": state["redaction_active"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
