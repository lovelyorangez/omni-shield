#!/usr/bin/env python3
"""
Omni-Shield Multi-Agent Pipeline — v3 (Agentic Pivot)
======================================================

Agent graph (all local, no data leaves the machine):

  Input
    └─► RouterAgent          — classifies doc type + subtype
          ├─► LayoutAgent    — VLM: extracts label-value pairs with boxes
          ├─► OCRAgent       — EasyOCR: word-level pixel bounding boxes
          └─► TextPIIAgent   — fine-tuned 3B: text-domain PII extraction
                └─► ContextAgent   ← KEY: "ID Number: 99 999 999"
                                          reads the label, classifies the value
                      └─► BBRefinerAgent  — snaps coarse VLM boxes to OCR boxes
                              └─► CriticAgent   — removes false positives
                                      └─► RedactionAgent + audit JSON

Problem this solves
-------------------
Previous pipeline asked the model "is this text PII?" without any surrounding
context. A lone number like "99 999 999" is ambiguous — it could be a serial
number, a price, or a national ID. The ContextAgent looks at the *adjacent
field label* first: if the label says "ID Number" the value is PII; if it says
"Invoice Total", it is not. This eliminates the bulk of false positives and
missed detections on structured documents.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import sys
import time
import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# Must be set before torch initialises CUDA to take effect
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import importlib.util

import torch
from PIL import Image, ImageDraw

# ── Heavy ML libraries — LAZY imports only ────────────────────────────────────
# Do NOT import transformers / bitsandbytes / easyocr / ultralytics here.
# Importing bitsandbytes at module level initialises its CUDA kernels and
# consumes ~2–3 GB of VRAM even before any model is loaded.  The web process
# imports this module too; if these were top-level the web process would eat
# all VRAM at startup and leave none for the rq worker.
#
# Instead we use importlib.util.find_spec() to check availability (cheap, no
# CUDA init) and do the real import inside the functions that need them.

_EASYOCR_AVAILABLE = importlib.util.find_spec("easyocr")     is not None
_YOLO_AVAILABLE    = importlib.util.find_spec("ultralytics")  is not None

try:
    from blockchain_manager import anchor_audit as _anchor_audit
    _BLOCKCHAIN_AVAILABLE = True
except ImportError:
    _BLOCKCHAIN_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info
    _QWEN_VL_UTILS = True
except ImportError:
    _QWEN_VL_UTILS = False

try:
    import cv2
    import numpy as np
    from scipy.ndimage import rotate as scipy_rotate
    _DESKEW_AVAILABLE = True
except ImportError:
    _DESKEW_AVAILABLE = False

try:
    import pdfplumber
    import fitz as _fitz          # PyMuPDF
    from pdf2image import convert_from_path as _pdf2images
    _PDF_AVAILABLE = True
except ImportError:
    _PDF_AVAILABLE = False

try:
    from faster_whisper import WhisperModel as _WhisperModel
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False

try:
    from pydub import AudioSegment as _AudioSegment
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("omni_shield")

# ── Module-level model cache (singleton across requests) ──────────────────────
_VLM_CACHE: dict = {"model": None, "processor": None}
_TEXT_MODEL_CACHE: dict = {"model": None, "tokenizer": None}

# Note: the old transformers.modeling_utils.caching_allocator_warmup monkey-patch
# was removed — it required a module-level transformers import which triggers
# bitsandbytes CUDA init.  The same suppression now happens lazily inside
# load_vlm() / load_text_model() after the lazy 'from transformers import'.
os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:128",
)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class OmniConfig:
    # Models
    vlm_model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    text_model_id: str = "Qwen/Qwen2.5-3B-Instruct"   # swap with fine-tuned path

    # Memory strategy
    # "dual" — load both models simultaneously (~3.2 GB weights, safe on 8 GB)
    # "lazy" — load one at a time, unload between agents (slower but safer)
    memory_mode: str = "lazy"

    # Image resolution cap (lower = less VRAM during VLM inference)
    max_pixels: int = 200_000

    # Generation
    max_new_tokens: int = 512

    # BB Refiner: minimum IoU overlap to snap a VLM box to OCR words
    iou_snap_threshold: float = 0.05

    # Output
    output_dir: str = "redacted_output"


CFG = OmniConfig()

# PII label keywords — if a field label contains any of these, value is PII
_PII_LABEL_KEYWORDS = {
    "name", "full name", "first name", "last name", "surname", "given name",
    "dob", "date of birth", "birth date", "birthday",
    "id", "id number", "identification", "national id", "passport", "licence",
    "license", "driver", "ssn", "social security", "tax", "nric", "nid",
    "idn", "4d idn", "dl", "dln", "dd", "document discriminator", "4d", "lic", "licence no", "license no",
    "phone", "mobile", "telephone", "cell",
    "email", "e-mail", "mail",
    "address", "street", "postcode", "zip", "pin code",
    "account", "bank", "iban", "sort code", "card",
    "signature", "fingerprint", "biometric",
}

# Non-PII label keywords — if a field label contains any of these, drop it
_NON_PII_LABEL_KEYWORDS = {
    "invoice", "total", "amount", "price", "cost", "sku", "ref", "reference",
    "order", "serial", "version", "batch", "quantity", "qty", "item",
    "department", "division", "company", "organisation", "organization",
    "title", "role", "position", "job",
    "exp", "iss", "4a", "4b", "expiry", "expiration", "issued", "issue date",
}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE (passed between agents like a message bus)
# ══════════════════════════════════════════════════════════════════════════════
@dataclasses.dataclass
class LabelValuePair:
    """A field label and its associated value, both with spatial grounding."""
    label: str
    value: str
    label_box: Optional[list[int]] = None        # [ymin, xmin, ymax, xmax] 0-1000
    value_box: Optional[list[int]] = None        # [ymin, xmin, ymax, xmax] 0-1000
    is_pii: Optional[bool] = None                # set by ContextAgent
    refined_pixel_box: Optional[list[int]] = None  # [xmin, ymin, xmax, ymax] px


@dataclasses.dataclass
class OCRWord:
    """A single word extracted by OCR with its pixel bounding box."""
    text: str
    box: list[int]   # [xmin, ymin, xmax, ymax] in pixels
    confidence: float


@dataclasses.dataclass
class AgentState:
    """Mutable shared state threaded through the entire agent pipeline."""
    image_path: str
    image_size: tuple[int, int] = dataclasses.field(default_factory=lambda: (0, 0))

    # Classification (RouterAgent)
    doc_type: str = "unknown"      # "id_card" | "form" | "essay" | "invoice" | ...
    input_format: str = "unknown"  # "image" | "pdf" | "text"

    # Extraction
    label_value_pairs: list[LabelValuePair] = dataclasses.field(default_factory=list)
    ocr_words: list[OCRWord] = dataclasses.field(default_factory=list)
    raw_text_pii: list[str] = dataclasses.field(default_factory=list)

    # After ContextAgent: only confirmed PII
    confirmed_pii: list[LabelValuePair] = dataclasses.field(default_factory=list)

    # After BBRefinerAgent + CriticAgent: final redaction targets
    final_redactions: list[LabelValuePair] = dataclasses.field(default_factory=list)

    # Selective redaction controls (set by OmniShieldOrchestrator from request params)
    pii_filter: dict = dataclasses.field(default_factory=dict)
    custom_terms: list[str] = dataclasses.field(default_factory=list)
    custom_patterns: list[str] = dataclasses.field(default_factory=list)

    # Output
    output_path: Optional[str] = None
    audit: dict = dataclasses.field(default_factory=dict)

    # Per-agent wall-clock timings in seconds {"RouterAgent": 0.03, ...}
    agent_timings: dict = dataclasses.field(default_factory=dict)

    # Fraction of image area that was redacted (0.0–100.0)
    coverage_pct: float = 0.0

    # Dry-run: run full pipeline but skip file writes and blockchain
    dry_run: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# MODEL MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def _quant_4bit():
    from transformers import BitsAndBytesConfig  # lazy — triggers bitsandbytes CUDA init
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )


_VRAM_EVICT_THRESHOLD_GB = 7.0  # only evict cached models above this pressure


def _evict_all_models(label: str = "") -> None:
    """Unconditionally clear both model caches and flush the CUDA allocator.

    Call this after any non-image pipeline run in the web process so the rq
    worker can claim GPU memory.  (The worker keeps its own in-process cache
    warm between jobs — eviction there would defeat the caching optimisation.)
    """
    _VLM_CACHE["model"]              = None
    _VLM_CACHE["processor"]          = None
    _TEXT_MODEL_CACHE["model"]       = None
    _TEXT_MODEL_CACHE["tokenizer"]   = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1024**3
        log.info(f"[VRAM | {label}] evicted all caches → {alloc:.2f} GB remaining")


def _free_vram(label: str = "") -> None:
    """Release CUDA allocator cache. Evicts cached models only if VRAM > threshold."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated() / 1024**3
    log.info(f"[VRAM{' | ' + label if label else ''}] {alloc:.2f} GB allocated")

    if alloc > _VRAM_EVICT_THRESHOLD_GB:
        log.warning(
            f"[VRAM] Pressure {alloc:.2f} GB > {_VRAM_EVICT_THRESHOLD_GB} GB threshold — "
            "evicting model cache to reclaim VRAM"
        )
        _VLM_CACHE["model"] = None
        _VLM_CACHE["processor"] = None
        _TEXT_MODEL_CACHE["model"] = None
        _TEXT_MODEL_CACHE["tokenizer"] = None
        gc.collect()
        torch.cuda.empty_cache()
        log.info(f"[VRAM] Post-eviction: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")


def load_vlm(cfg: OmniConfig):
    if _VLM_CACHE["model"] is not None:
        log.info("VLM cache hit — reusing loaded model")
        return _VLM_CACHE["processor"], _VLM_CACHE["model"]

    log.info(f"Loading VLM: {cfg.vlm_model_id}")
    _free_vram("before VLM load")
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration  # lazy — bitsandbytes inits here
    import transformers.modeling_utils
    transformers.modeling_utils.caching_allocator_warmup = lambda *a, **kw: None  # suppress VRAM spike
    processor = AutoProcessor.from_pretrained(cfg.vlm_model_id)
    try:
        log.info("  Attempting GPU (4-bit NF4)")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.vlm_model_id,
            device_map="auto",
            quantization_config=_quant_4bit(),
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
    except torch.cuda.OutOfMemoryError:
        log.warning("  GPU OOM — falling back to CPU (float32, slower)")
        torch.cuda.empty_cache()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.vlm_model_id,
            device_map="cpu",
            low_cpu_mem_usage=True,
            dtype=torch.float32,
        )
    adapter_path = "./fine_tuned_vlm/final_adapter"
    _skip_adapter = os.environ.get("OMNI_SKIP_ADAPTER", "0") == "1"
    if _skip_adapter:
        log.info("[BASE MODEL] Adapter skipped — running base Qwen2-VL-2B-Instruct (OMNI_SKIP_ADAPTER=1)")
    elif Path(adapter_path).exists():
        from peft import PeftModel
        log.info(f"Loading fine-tuned adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        log.info("Fine-tuned adapter merged successfully")
    else:
        log.warning("No fine-tuned adapter found, using base model")

    model.eval()
    _VLM_CACHE["model"] = model
    _VLM_CACHE["processor"] = processor
    _free_vram("after VLM load")
    return processor, model


def load_text_model(cfg: OmniConfig):
    if _TEXT_MODEL_CACHE["model"] is not None:
        log.info("Text model cache hit — reusing loaded model")
        return _TEXT_MODEL_CACHE["tokenizer"], _TEXT_MODEL_CACHE["model"]

    log.info(f"Loading text model: {cfg.text_model_id}")
    _free_vram("before text model load")
    from transformers import AutoTokenizer, AutoModelForCausalLM  # lazy
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_id)
    try:
        log.info("  Attempting GPU (4-bit NF4)")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.text_model_id,
            device_map="auto",
            quantization_config=_quant_4bit(),
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
        )
    except torch.cuda.OutOfMemoryError:
        log.warning("  GPU OOM — falling back to CPU (float32, slower)")
        torch.cuda.empty_cache()
        model = AutoModelForCausalLM.from_pretrained(
            cfg.text_model_id,
            device_map="cpu",
            low_cpu_mem_usage=True,
            dtype=torch.float32,
        )
    model.eval()
    _TEXT_MODEL_CACHE["model"] = model
    _TEXT_MODEL_CACHE["tokenizer"] = tokenizer
    _free_vram("after text model load")
    return tokenizer, model


# ══════════════════════════════════════════════════════════════════════════════
# BASE AGENT
# ══════════════════════════════════════════════════════════════════════════════
class BaseAgent(ABC):
    name: str = "BaseAgent"

    def __call__(self, state: AgentState) -> AgentState:
        t0 = time.perf_counter()
        log.info(f"[{self.name}] starting ...")
        state = self.run(state)
        log.info(f"[{self.name}] done in {time.perf_counter() - t0:.2f}s")
        return state

    @abstractmethod
    def run(self, state: AgentState) -> AgentState:
        ...


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1: ROUTER
# ══════════════════════════════════════════════════════════════════════════════
# DESKEW HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _detect_skew_angle(gray: "np.ndarray") -> float:
    """
    Estimate document skew angle using Probabilistic Hough Line Transform.
    Returns degrees to rotate counter-clockwise to straighten the image.
    Angle is clamped to [-45, 45] — beyond that, it's likely a layout
    rotation rather than photographic skew.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10
    )
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only use near-horizontal lines (within 45° of horizontal)
            if abs(angle) < 45:
                angles.append(angle)

    if not angles:
        return 0.0

    median_angle = float(np.median(angles))
    return median_angle


def _deskew_image(image_path: str) -> tuple[str, float]:
    """
    Detect and correct skew in an image. If skew > 0.5°, saves a deskewed
    copy next to the original with suffix _deskewed and returns its path.
    Returns (path_to_use, angle_corrected).
    """
    if not _DESKEW_AVAILABLE:
        log.warning("  Deskew skipped — cv2/scipy not available")
        return image_path, 0.0

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return image_path, 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    angle = _detect_skew_angle(gray)

    if abs(angle) < 0.5:
        log.info(f"  Detected skew: {angle:.2f}° — within tolerance, no correction needed")
        return image_path, angle

    log.info(f"  Detected skew: {angle:.2f}° — applying correction")

    # scipy_rotate rotates counter-clockwise for positive angles;
    # we want to *undo* the skew, so rotate by -angle
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    corrected = scipy_rotate(img_rgb, -angle, reshape=False, cval=255)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    p = Path(image_path)
    out_path = str(p.parent / f"{p.stem}_deskewed{p.suffix}")
    Image.fromarray(corrected).save(out_path)
    log.info(f"  Deskewed image saved: {out_path}")
    return out_path, angle


# ══════════════════════════════════════════════════════════════════════════════
class RouterAgent(BaseAgent):
    """
    Classifies the input into input_format and doc_type using filename
    heuristics. No model needed — pure filename + extension logic.
    """
    name = "RouterAgent"

    _SUBTYPE_HINTS = {
        "id": "id_card", "passport": "passport", "licence": "id_card",
        "license": "id_card", "driver": "id_card", "form": "form",
        "application": "form", "essay": "essay", "invoice": "invoice",
        "receipt": "invoice", "statement": "invoice",
    }

    def run(self, state: AgentState) -> AgentState:
        p = Path(state.image_path)
        suffix = p.suffix.lower()

        if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}:
            state.input_format = "image"
        elif suffix == ".pdf":
            state.input_format = "pdf"
        elif suffix in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
            state.input_format = "audio"
        elif suffix in {".txt", ".md"}:
            state.input_format = "text"
        else:
            state.input_format = "image"

        # Audio files always have doc_type "audio_recording" (Bug 3b)
        if state.input_format == "audio":
            state.doc_type = "audio_recording"
            log.info(f"  format={state.input_format}  doc_type={state.doc_type}  size={state.image_size}")
            return state

        name_lower = p.stem.lower()
        state.doc_type = "unknown"
        for kw, dtype in self._SUBTYPE_HINTS.items():
            if kw in name_lower:
                state.doc_type = dtype
                break

        if state.input_format == "image":
            # Deskew before anything else touches the image
            deskewed_path, skew_angle = _deskew_image(state.image_path)
            if deskewed_path != state.image_path:
                state.image_path = deskewed_path
                state.audit["skew_angle_corrected"] = round(skew_angle, 2)
            else:
                state.audit["skew_angle_detected"] = round(skew_angle, 2)

            img = Image.open(state.image_path)
            state.image_size = img.size
            img.close()

        log.info(f"  format={state.input_format}  doc_type={state.doc_type}  size={state.image_size}")
        return state


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2: LAYOUT AGENT (VLM)
# ══════════════════════════════════════════════════════════════════════════════
_LAYOUT_PROMPT = """List every labelled field in this document image.

For each field output exactly one line:
LABEL: <label text> | VALUE: <value text> | VALUE_BOX: (x1,y1),(x2,y2)

Rules:
- Coordinates are normalised 0-1000
- VALUE_BOX covers the value text only, not the label
- One line per field — no headers, no blank lines, nothing else
- If no labelled fields are visible, output: NONE
"""

_DOC_TYPE_PROMPT = (
    "What type of document is this? "
    "Answer with exactly one word chosen from: "
    "id_card, passport, form, invoice, essay, unknown."
)

_DOC_TYPE_VALID = {"id_card", "passport", "form", "invoice", "essay", "unknown"}


class LayoutAgent(BaseAgent):
    """
    Uses the VLM to spatially parse a document into label-value pairs.

    Instead of asking "find PII", we ask "find all labelled fields".
    The label is what lets the downstream ContextAgent decide whether
    the value is sensitive — this is the core architectural change.

    Example VLM output:
        LABEL: ID Number | VALUE: 99 999 999 | VALUE_BOX: (340,120),(360,310)
        LABEL: Full Name | VALUE: Jane Smith | VALUE_BOX: (200,120),(220,400)
    """
    name = "LayoutAgent"

    # Format 1 (expected): LABEL: X | VALUE: Y | VALUE_BOX: (y1,x1),(y2,x2)
    # Format 2 (bracket):  LABEL: X | VALUE: Y | VALUE_BOX: [y1,x1],[y2,x2]
    # Format 3 (4-tuple):  LABEL: X | VALUE: Y | VALUE_BOX: (y1,x1,y2,x2)
    # Format 4 (truncated):last line may be missing closing ) — \)? handles it
    _PATTERN = re.compile(
        r'LABEL:\s*(.+?)\s*\|\s*VALUE:\s*(.+?)\s*\|\s*VALUE_BOX:\s*'
        r'[(\[]\s*(\d+)\s*,\s*(\d+)\s*[)\]]?\s*,\s*[(\[]\s*(\d+)\s*,\s*(\d+)\s*[)\]?]?',
        re.IGNORECASE,
    )
    # Fallback: single 4-tuple (y1,x1,y2,x2) — some 4-bit outputs use this
    _PATTERN_4TUPLE = re.compile(
        r'LABEL:\s*(.+?)\s*\|\s*VALUE:\s*(.+?)\s*\|\s*VALUE_BOX:\s*'
        r'[(\[]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[)\]?]?',
        re.IGNORECASE,
    )

    def __init__(self, processor, model, cfg: OmniConfig):
        self.processor = processor
        self.model = model
        self.cfg = cfg

    def _vlm_classify_doc_type(self, image_path: str) -> str:
        """
        Single-token VLM call to classify document type.
        Runs only when RouterAgent couldn't determine the type from the filename.
        Reuses the already-loaded processor/model — no extra load cost.
        """
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path, "max_pixels": self.cfg.max_pixels},
                {"type": "text", "text": _DOC_TYPE_PROMPT},
            ],
        }]
        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            gen_ids = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip().lower()
        del inputs, gen_ids, trimmed

        # Accept the first token that matches a valid doc type
        for word in re.split(r'[\s,\.]+', raw):
            if word in _DOC_TYPE_VALID and word != "unknown":
                return word
        return "unknown"

    def run(self, state: AgentState) -> AgentState:
        if state.input_format != "image" or not _QWEN_VL_UTILS:
            log.info("  LayoutAgent skipped")
            return state

        # Bug 6: wrap entire body so a hang or OOM returns empty pairs instead of blocking
        try:
            return self._run_inner(state)
        except Exception as exc:
            log.warning(f"  [LayoutAgent] failed — returning empty pairs ({exc})")
            state.label_value_pairs = []
            return state

    def _run_inner(self, state: AgentState) -> AgentState:
        # Skip the expensive VLM call when OCR found very few words — the image
        # is almost certainly not a document (e.g. a meme or natural photo).
        if len(state.ocr_words) < 5:
            log.warning(
                f"  [LayoutAgent] Only {len(state.ocr_words)} OCR word(s) — "
                f"skipping VLM (likely not a document)"
            )
            state.label_value_pairs = []
            return state

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": state.image_path, "max_pixels": self.cfg.max_pixels},
                {"type": "text", "text": _LAYOUT_PROMPT},
            ],
        }]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_prompt], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(self.model.device)

        _free_vram("pre-layout-gen")
        with torch.inference_mode():
            # Bug 6: cap to 256 tokens — layout labels are short, 512 causes hangs
            # on complex ID cards.
            gen_ids = self.model.generate(
                **inputs, max_new_tokens=256, do_sample=False
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        del inputs, gen_ids, trimmed
        _free_vram("post-layout-gen")

        log.info(f"  [RAW LAYOUT OUTPUT]\n{raw}")

        # Strip lines that are literal prompt-template echoes such as:
        #   "[Label] | [Value] | [Value_Box]: (0, 100), (100, 100)"
        # These appear when the VLM outputs its own format instructions
        # instead of real field values and would otherwise waste parse time.
        _raw_lines = [
            line for line in raw.splitlines()
            if "[Label]" not in line and "[Value]" not in line and line.strip()
        ]

        pairs = []
        for line in _raw_lines:
            m = self._PATTERN.search(line)
            if not m:
                m = self._PATTERN_4TUPLE.search(line)
            if not m:
                log.debug(f"  [LAYOUT SKIP] no match: {line!r}")
                continue
            label, value = m.group(1).strip(), m.group(2).strip()
            ymin, xmin, ymax, xmax = map(int, m.groups()[2:])
            # Clamp to [0, 1000]; allow small overshoot (model sometimes emits 1001)
            ymin, xmin = max(0, min(ymin, 999)), max(0, min(xmin, 999))
            ymax, xmax = max(1, min(ymax, 1000)), max(1, min(xmax, 1000))
            if ymin >= ymax or xmin >= xmax or not label or not value:
                log.debug(f"  [LAYOUT SKIP] bad coords ({ymin},{xmin})-({ymax},{xmax}): {label!r}")
                continue
            pairs.append(LabelValuePair(label=label, value=value, value_box=[ymin, xmin, ymax, xmax]))

        state.label_value_pairs = pairs
        log.info(f"  Found {len(pairs)} label-value pair(s)")

        # Second-pass doc_type classification: only when filename heuristic failed.
        if state.doc_type == "unknown":
            doc_type = self._vlm_classify_doc_type(state.image_path)
            if doc_type != "unknown":
                log.info(f"  [VLM doc_type] '{state.doc_type}' -> '{doc_type}'")
                state.doc_type = doc_type
            else:
                log.info("  [VLM doc_type] could not determine type, staying 'unknown'")

        return state


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3: OCR AGENT
# ══════════════════════════════════════════════════════════════════════════════
class OCRAgent(BaseAgent):
    """
    Runs EasyOCR (CPU) to get word-level pixel bounding boxes.
    These boxes are later used by BBRefinerAgent to sharpen VLM boxes,
    and by ContextAgent to map TextPII strings back to pixel locations.
    """
    name = "OCRAgent"

    def __init__(self, languages: list[str] = None):
        self.languages = languages or ["en"]
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            log.info("  Initialising EasyOCR (CPU) ...")
            import easyocr  # lazy — avoid CUDA init in web process
            self._reader = easyocr.Reader(self.languages, gpu=False)
        return self._reader

    def run(self, state: AgentState) -> AgentState:
        if state.input_format != "image" or not _EASYOCR_AVAILABLE:
            log.info("  OCRAgent skipped")
            return state

        results = self.reader.readtext(state.image_path)
        words = []
        for (corners, text, conf) in results:
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            words.append(OCRWord(
                text=text.strip(),
                box=[int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                confidence=float(conf),
            ))

        state.ocr_words = words
        log.info(f"  OCR extracted {len(words)} word(s)")
        return state


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4: TEXT PII AGENT
# ══════════════════════════════════════════════════════════════════════════════

# Deterministic regex patterns that always run after the LLM pass.
# Tuple: (pattern, filter_key, label)
_REGEX_PII_PATTERNS: list[tuple[str, str, str]] = [
    # Indian government / financial IDs
    (r'\b[A-Z]{5}[0-9]{4}[A-Z]\b',                                      'id_numbers', 'PAN'),
    (r'\b\d{4}\s\d{4}\s\d{4}\b',                                        'id_numbers', 'Aadhaar'),
    (r'\b\d{12}\b',                                                      'id_numbers', 'Aadhaar (unspaced)'),
    (r'\b[A-Z]{4}0[A-Z0-9]{6}\b',                                       'id_numbers', 'IFSC'),
    (r'\b[UL]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6}\b',                       'id_numbers', 'CIN'),
    (r'\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b',                    'id_numbers', 'GST'),
    (r'\b[A-Z]{2}\d{7}\b',                                               'id_numbers', 'Voter ID'),
    (r'\b[A-Z]{4}\d{7}\b',                                               'id_numbers', 'Licence No'),
    # International passports / IDs
    (r'\b[A-Z][0-9]{7}\b',                                               'id_numbers', 'Passport'),
    (r'\b[A-Z]{2}\d{6}[A-Z]\b',                                         'id_numbers', 'UK Passport'),
    (r'\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b',                  'id_numbers', 'NI Number'),
    (r'\b\d{3}-\d{2}-\d{4}\b',                                          'id_numbers', 'SSN'),
    # Financial
    (r'\b\d{9,18}\b',                                                    'id_numbers', 'Account Number'),
    (r'\b[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\s?[A-Z0-9]{4}\b',    'id_numbers', 'Card Number'),
    # Phone numbers
    (r'(\+91[\s-]?|0)?[6-9]\d{4}[\s-]?\d{5}\b',                        'phones',     'Phone'),
    (r'\+\d{1,3}[\s-]?\d{4,5}[\s-]?\d{4,6}\b',                         'phones',     'International Phone'),
    # Email
    (r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b',            'emails',     'Email'),
    # Dates
    (r'\b\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4}\b',                       'dob',        'Date'),
    (r'\b\d{1,2}(st|nd|rd|th)?\s+'
     r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
     r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|'
     r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
     r'\s+\d{4}\b',                                                      'dob',        'Date (written)'),
]

_TEXT_PII_PROMPT = """Extract all personally identifiable information from the text below.

Output ONLY the actual values found in the text, one per line. No labels, no categories, no explanations, no formatting. Only exact strings that appear in the text.

Find:
- Person names
- Phone numbers
- Email addresses
- ID numbers (Aadhaar, PAN, passport, driving licence, employee ID, account numbers)
- Dates of birth
- Physical addresses
- Bank account and IFSC codes
- Any other personal identifiers

If nothing is found, output: NONE

TEXT:
{text}

VALUES FOUND:"""

# ── Audio transcript normalisation ───────────────────────────────────────────

_SPOKEN_ONES: dict[str, str] = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'oh': '0', 'o': '0', 'nought': '0',
}

_SPOKEN_ORDINALS: dict[str, str] = {
    'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
    'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
    'ninth': '9th', 'tenth': '10th', 'eleventh': '11th', 'twelfth': '12th',
    'thirteenth': '13th', 'fourteenth': '14th', 'fifteenth': '15th',
    'sixteenth': '16th', 'seventeenth': '17th', 'eighteenth': '18th',
    'nineteenth': '19th', 'twentieth': '20th', 'thirtieth': '30th',
    'twenty-first': '21st', 'twenty first': '21st',
    'twenty-second': '22nd', 'twenty second': '22nd',
    'twenty-third': '23rd', 'twenty third': '23rd',
    'twenty-fourth': '24th', 'twenty fourth': '24th',
    'twenty-fifth': '25th', 'twenty fifth': '25th',
    'twenty-sixth': '26th', 'twenty-seventh': '27th',
    'twenty-eighth': '28th', 'twenty-ninth': '29th',
    'thirty-first': '31st',
}

_SPOKEN_DECADES: dict[str, str] = {
    'eighteen': '18', 'nineteen': '19', 'twenty': '20',
}

_SPOKEN_TENS: dict[str, str] = {
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
    'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
    'eighty': '80', 'ninety': '90',
}


def normalise_audio_transcript(text: str) -> str:
    """
    Convert spoken numbers / ordinals / years to digit form so that regex
    patterns can match them after Whisper transcription.

    Transformations applied:
      1. Ordinal words → "15th", "3rd" etc.
      2. Runs of 4+ digit-words → concatenated digit string ("nine eight seven six" → "9876")
      3. Spoken years → "nineteen eighty seven" → "1987"
      4. Spaced single letters/digits → PAN-style collapse ("A B C D E 1 2 3 4 F" → "ABCDE1234F")

    Returns original text + '\\n' + normalised text so LLM still sees the
    spoken form while the regex sweep sees the digit form.
    """
    normalised = text

    # Step 1: replace ordinal words with numeric ordinals
    for word, num in _SPOKEN_ORDINALS.items():
        normalised = re.sub(rf'\b{re.escape(word)}\b', num, normalised, flags=re.IGNORECASE)

    # Step 2: collapse runs of 4+ consecutive digit-words into digit strings
    tokens = normalised.split()
    result_tokens: list[str] = []
    digit_run: list[str] = []
    for token in tokens:
        clean = re.sub(r'[^a-z]', '', token.lower())
        if clean in _SPOKEN_ONES:
            digit_run.append(_SPOKEN_ONES[clean])
        else:
            if len(digit_run) >= 4:
                result_tokens.append(''.join(digit_run))
            elif digit_run:
                result_tokens.extend(digit_run)
            digit_run = []
            result_tokens.append(token)
    if len(digit_run) >= 4:
        result_tokens.append(''.join(digit_run))
    elif digit_run:
        result_tokens.extend(digit_run)
    normalised = ' '.join(result_tokens)

    # Step 3: spoken years — "nineteen eighty seven" → "1987"
    def _replace_year(m: re.Match) -> str:
        decade = _SPOKEN_DECADES.get(m.group(1).lower(), m.group(1))
        rest   = _SPOKEN_TENS.get(m.group(2).lower(), m.group(2))
        return decade + rest

    normalised = re.sub(
        r'\b(eighteen|nineteen|twenty)\s+'
        r'(eighty|ninety|seventy|sixty|fifty|forty|thirty|twenty|ten|'
        r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|'
        r'eighteen|nineteen)\b',
        _replace_year, normalised, flags=re.IGNORECASE,
    )

    # Step 4: spaced PAN-style sequences — "A B C D E 1 2 3 4 F" → "ABCDE1234F"
    normalised = re.sub(
        r'\b([A-Z] ){4,}[A-Z0-9]\b',
        lambda m: m.group(0).replace(' ', ''),
        normalised,
    )

    return text + '\n' + normalised


# Map single digit char back to a canonical spoken word (for spoken-timestamp lookup)
_DIGIT_TO_SPOKEN: dict[str, str] = {v: k for k, v in _SPOKEN_ONES.items() if k not in ('oh', 'o', 'nought')}


class TextPIIAgent(BaseAgent):
    """
    Runs the fine-tuned Qwen2.5-3B over OCR-reconstructed text.
    Produces a flat list of raw PII strings (no boxes yet).
    These are later absorbed by ContextAgent and mapped to boxes by BBRefiner.
    """
    name = "TextPIIAgent"

    def __init__(self, tokenizer, model, cfg: OmniConfig):
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = cfg

    def run(self, state: AgentState) -> AgentState:
        if not state.ocr_words:
            log.info("  No OCR words — TextPIIAgent skipped")
            return state

        # Sort top-to-bottom, left-to-right for reading order
        words_sorted = sorted(state.ocr_words, key=lambda w: (w.box[1] // 20, w.box[0]))
        text = " ".join(w.text for w in words_sorted)

        prompt = _TEXT_PII_PROMPT

        messages = [
            {"role": "system", "content": "You are a precise PII extractor."},
            {"role": "user", "content": prompt.format(text=text)},
        ]
        batch = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        # apply_chat_template returns a BatchEncoding (has attention_mask) or a plain
        # tensor depending on the tokenizer version.  Always extract both so the model
        # gets an explicit attention_mask — without it, pad==eos causes silent truncation.
        if hasattr(batch, "input_ids"):
            encoded    = batch.input_ids.to(self.model.device)
            attn_mask  = batch.attention_mask.to(self.model.device)
        else:
            encoded   = batch.to(self.model.device)
            attn_mask = torch.ones_like(encoded)

        with torch.inference_mode():
            gen_ids = self.model.generate(
                encoded,
                attention_mask=attn_mask,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
            )

        output = self.tokenizer.decode(gen_ids[0][encoded.shape[1]:], skip_special_tokens=True)
        del encoded, gen_ids
        _free_vram("post-textpii-gen")

        log.info(f"  [RAW TEXT PII]\n{output}")

        pii_items = []
        for line in output.splitlines():
            line = line.strip().lstrip("-*•·").strip()
            # Strip "**Label**: " or "Label: " prefixes the model sometimes adds
            # e.g. "**Name**: Alice Johnson" → "Alice Johnson"
            line = re.sub(r'^\*{0,2}[A-Za-z /()]{1,30}\*{0,2}:\s*', '', line).strip()
            if line and line.upper() not in {"NONE", "N/A", ""}:
                pii_items.append(line)

        # ── Deduplicate preserving order ──────────────────────────────────────
        seen: set[str] = set()
        unique_items: list[str] = []
        for item in pii_items:
            key = item.strip().lower()
            if key not in seen and len(key) > 2:
                seen.add(key)
                unique_items.append(item)
        pii_items = unique_items[:20]  # cap at 20 items to prevent garbage floods

        # ── Filter field-label noise the model hallucinates as PII values ─────
        _LABEL_NOISE = {
            "4d. licence no", "licence no", "issue date",
            "expiry date", "date of issue", "date of expiry",
            "date of birth", "address", "postcode", "name",
            "gender", "nationality", "place of birth",
            "signature", "none", "n/a",
        }
        pii_items = [
            item for item in pii_items
            if item.strip().lower() not in _LABEL_NOISE
            and len(item.strip()) > 3
        ]

        # ── Deterministic regex sweep (always runs, supplements LLM output) ──
        _full_text = "\n".join(w.text for w in state.ocr_words) if state.ocr_words else ""
        _pii_set = {item.strip() for item in pii_items}
        _active_filter = state.pii_filter or {}
        for _pat, _fkey, _label in _REGEX_PII_PATTERNS:
            if not _active_filter.get(_fkey, True):
                continue
            for _m in re.finditer(_pat, _full_text):
                _val = _m.group().strip()
                if _val and _val not in _pii_set:
                    pii_items.append(_val)
                    _pii_set.add(_val)
                    log.info(f"  [REGEX-SWEEP] {_label}: '{_val}'")

        state.raw_text_pii = pii_items
        log.info(f"  Raw PII items: {pii_items}")
        return state


# ── PII filter helpers ────────────────────────────────────────────────────────
_LABEL_TO_FILTER_KEY = {
    'name':        'names',
    'date of birth': 'dob',
    'dob':         'dob',
    'birth':       'dob',
    'id no':       'id_numbers',
    'id no.':      'id_numbers',
    'id number':   'id_numbers',
    'passport':    'id_numbers',
    'license':     'id_numbers',
    'phone':       'phones',
    'mobile':      'phones',
    'tel':         'phones',
    'email':       'emails',
    'address':     'addresses',
    'org':         'org_names',
    'company':     'org_names',
    'organisation': 'org_names',
    'organization': 'org_names',
    'date':        'dates',
    'face':        'faces',
    'signature':   'signatures',
}


def apply_pii_filter(confirmed: list, pii_filter: dict) -> list:
    """Drop confirmed PII items whose category is disabled in pii_filter."""
    if not pii_filter:
        return confirmed
    filtered = []
    for item in confirmed:
        label_lower = item.label.lower()
        filter_key = None
        for k, v in _LABEL_TO_FILTER_KEY.items():
            if k in label_lower:
                filter_key = v
                break
        # No mapping found → keep by default (safe)
        if filter_key is None or pii_filter.get(filter_key, True):
            filtered.append(item)
        else:
            log.info(
                f"  [FILTER] Dropped '{item.label}' -> '{item.value}' "
                f"(category '{filter_key}' disabled by user)"
            )
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5: CONTEXT AGENT  ← KEY INNOVATION
# ══════════════════════════════════════════════════════════════════════════════
class ContextAgent(BaseAgent):
    """
    Resolves ambiguous values by reading their adjacent field label.

    "99 999 999" is ambiguous in isolation. But if the label next to it
    reads "ID Number", it is definitively PII. If the label reads "Invoice
    Total", it is not. This eliminates the largest class of false positives
    on structured documents (ID cards, forms, applications).

    Two-stage approach (fast heuristic first, LLM only for ambiguous cases):

    Stage 1 — Keyword heuristic (no LLM, ~0ms):
        Checks label against _PII_LABEL_KEYWORDS and _NON_PII_LABEL_KEYWORDS.
        Resolves ~80% of cases instantly.

    Stage 2 — LLM binary classifier (used only for unknown labels):
        Tight yes/no prompt to the 3B text model.
        max_new_tokens=5 so it costs almost nothing.

    Also absorbs raw_text_pii from TextPIIAgent: items that weren't
    captured as labelled fields are wrapped as labelless pairs so they
    still get redacted.
    """
    name = "ContextAgent"

    def __init__(self, tokenizer, model, cfg: OmniConfig):
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = cfg

    def _heuristic(self, label: str) -> Optional[bool]:
        """True = is PII, False = not PII, None = ambiguous."""
        label_lower = label.lower()
        for kw in _NON_PII_LABEL_KEYWORDS:
            if kw in label_lower:
                return False
        for kw in _PII_LABEL_KEYWORDS:
            if kw in label_lower:
                return True
        return None

    def _llm_classify(self, label: str, value: str) -> bool:
        """Ask the 3B model: is this value PII given its label? Returns True/False."""
        prompt = (
            f"Field label: '{label}'\n"
            f"Field value: '{value}'\n\n"
            "Is this field value Personal Identifiable Information (PII)?\n"
            "PII includes: full name, ID number, passport, date of birth, phone, email, address.\n"
            "Answer with exactly one word: YES or NO"
        )
        messages = [
            {"role": "system", "content": "You are a PII classifier. Answer only YES or NO."},
            {"role": "user", "content": prompt},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        encoded = encoded.to(self.model.device)

        with torch.inference_mode():
            gen_ids = self.model.generate(encoded, max_new_tokens=5, do_sample=False)

        answer = self.tokenizer.decode(
            gen_ids[0][encoded.shape[1]:], skip_special_tokens=True
        ).strip().upper()
        del encoded, gen_ids
        return "NO" not in answer   # Default YES on ambiguous output

    def run(self, state: AgentState) -> AgentState:
        confirmed = []
        seen: set[tuple[str, str]] = set()  # (label.lower(), value.lower()) dedup key

        # Process label-value pairs from LayoutAgent
        for pair in state.label_value_pairs:
            heuristic = self._heuristic(pair.label)
            if heuristic is True:
                pair.is_pii = True
                log.info(f"  [HEU PII]  '{pair.label}' -> '{pair.value}'")
            elif heuristic is False:
                pair.is_pii = False
                log.info(f"  [HEU DROP] '{pair.label}' -> '{pair.value}' (non-PII label)")
            else:
                pair.is_pii = self._llm_classify(pair.label, pair.value)
                tag = "LLM PII " if pair.is_pii else "LLM DROP"
                log.info(f"  [{tag}] '{pair.label}' -> '{pair.value}'")

            if pair.is_pii:
                key = (pair.label.lower(), pair.value.lower())
                if key in seen:
                    log.info(f"  [DEDUP]    '{pair.label}' -> '{pair.value}' (duplicate skipped)")
                    continue
                seen.add(key)
                confirmed.append(pair)

        # Absorb TextPII items not already found by LayoutAgent
        found_values = {p.value.lower() for p in confirmed}
        for raw_item in state.raw_text_pii:
            if raw_item.lower() not in found_values:
                pair = LabelValuePair(label="", value=raw_item, is_pii=True)
                confirmed.append(pair)
                log.info(f"  [TXT PII]  (no label) -> '{raw_item}'")

        if state.pii_filter:
            confirmed = apply_pii_filter(confirmed, state.pii_filter)

        state.confirmed_pii = confirmed
        log.info(f"  Confirmed {len(confirmed)} PII item(s) after context filtering")
        return state


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 6: BOUNDING BOX REFINER
# ══════════════════════════════════════════════════════════════════════════════
class BBRefinerAgent(BaseAgent):
    """
    Sharpens coarse VLM boxes by snapping them to precise OCR word boxes.

    VLM outputs normalized [0-1000] boxes that often over/undershoot the
    actual text because the model reasons about position roughly.
    EasyOCR gives character-accurate pixel boxes.

    For each VLM box:
      1. Convert to pixel coordinates.
      2. Find all OCR words with IoU >= iou_snap_threshold.
      3. Replace VLM box with the pixel-union of matching OCR words.
      4. If no overlap found, keep VLM box as-is.

    For labelless TextPII items (no VLM box):
      - Text-match against OCR word list to find the pixel box.
    """
    name = "BBRefinerAgent"

    def __init__(self, cfg: OmniConfig):
        self.cfg = cfg

    @staticmethod
    def _norm_to_pixel(box_norm: list[int], w: int, h: int) -> list[int]:
        ymin, xmin, ymax, xmax = box_norm
        return [
            int((xmin / 1000.0) * w), int((ymin / 1000.0) * h),
            int((xmax / 1000.0) * w), int((ymax / 1000.0) * h),
        ]

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    @staticmethod
    def _union_box(boxes: list[list[int]]) -> list[int]:
        return [
            min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes),
        ]

    def _find_by_name_words(self, text: str, words: list[OCRWord]) -> Optional[list[int]]:
        """Split a name value on commas/spaces and union boxes of individually matched tokens.

        "SAMPLE, ANDREW JASON" → search for "SAMPLE", "ANDREW", "JASON" separately
        and return the bounding union. Only used when _find_by_text fails and the
        item's label indicates a name field.
        """
        tokens = [t.strip() for t in re.split(r'[,\s]+', text) if t.strip()]
        if len(tokens) <= 1:
            return None
        boxes = [b for b in (self._find_by_text(tok, words) for tok in tokens) if b]
        return self._union_box(boxes) if boxes else None

    def _find_by_text(self, text: str, words: list[OCRWord]) -> Optional[list[int]]:
        target_words = text.lower().split()
        if not target_words:
            return None
        word_texts = [w.text.lower() for w in words]
        n = len(target_words)
        for i in range(len(word_texts) - n + 1):
            if word_texts[i:i + n] == target_words:
                return self._union_box([words[j].box for j in range(i, i + n)])
        # Exact whole-word match before substring fallback
        text_lower = text.lower()
        for word in words:
            if word.text.lower() == text_lower:
                return word.box
        # Fallback: substring match on single words
        for word in words:
            if text_lower in word.text.lower():
                return word.box
        # Date fallback: OCR often merges slashes with digits ("01/0711973" for "01/07/1973").
        # Try matching just the 4-digit year component — it survives most OCR corruption.
        if '/' in text:
            for _part in re.split(r'[/\-]', text):
                if len(_part) == 4 and _part.isdigit():
                    for word in words:
                        if _part in word.text:
                            return word.box
        return None

    def run(self, state: AgentState) -> AgentState:
        w, h = state.image_size
        if not w or not h:
            log.info("  No image size — BBRefinerAgent skipped")
            state.final_redactions = state.confirmed_pii
            return state

        _MIN_BOX_PX = 8  # discard degenerate boxes smaller than this in either dimension

        _NAME_LABELS = {"name", "full name", "first name", "last name", "surname", "given name"}

        for pair in state.confirmed_pii:
            if pair.value_box:
                pixel_box = self._norm_to_pixel(pair.value_box, w, h)

                # 1. Try pixel-accurate text match first — always preferred over VLM coords
                found = self._find_by_text(pair.value, state.ocr_words)
                if found:
                    pair.refined_pixel_box = found
                    log.info(f"  [TEXT-FIRST] '{pair.value}' -> {found}")
                elif pair.label and pair.label.lower() in _NAME_LABELS:
                    # 1b. Name fields: split on comma/space and union per-word boxes
                    found = self._find_by_name_words(pair.value, state.ocr_words)
                    if found:
                        pair.refined_pixel_box = found
                        log.info(f"  [NAME-SPLIT] '{pair.value}' -> {found}")

                if not pair.refined_pixel_box:
                    # 2. Fall back to IoU snapping against OCR word bounding boxes
                    overlapping = [
                        ocr_w for ocr_w in state.ocr_words
                        if self._iou(pixel_box, ocr_w.box) >= self.cfg.iou_snap_threshold
                    ]
                    if overlapping:
                        pair.refined_pixel_box = self._union_box([ow.box for ow in overlapping])
                        log.info(f"  [SNAP] '{pair.value}' VLM={pixel_box} -> OCR={pair.refined_pixel_box}")
                    else:
                        # 3. Last resort: use the raw VLM box
                        bw, bh = pixel_box[2] - pixel_box[0], pixel_box[3] - pixel_box[1]
                        if bw >= _MIN_BOX_PX and bh >= _MIN_BOX_PX:
                            pair.refined_pixel_box = pixel_box
                            log.info(f"  [KEEP VLM] '{pair.value}' {pixel_box}")
                        else:
                            log.warning(
                                f"  [DEGENERATE BOX] '{pair.value}' {pixel_box} "
                                f"({bw}x{bh}px < {_MIN_BOX_PX}px min) -- cleared"
                            )
                            pair.refined_pixel_box = None
            else:
                found = self._find_by_text(pair.value, state.ocr_words)
                if found:
                    pair.refined_pixel_box = found
                    log.info(f"  [TEXT MATCH] '{pair.value}' -> {found}")
                elif pair.label and pair.label.lower() in _NAME_LABELS:
                    # Name-word split fallback for labelless-box name items
                    found = self._find_by_name_words(pair.value, state.ocr_words)
                    if found:
                        pair.refined_pixel_box = found
                        log.info(f"  [NAME-SPLIT] '{pair.value}' -> {found}")
                    else:
                        log.warning(f"  [NO BOX] '{pair.value}' -- no OCR match")
                else:
                    log.warning(f"  [NO BOX] '{pair.value}' -- no OCR match")

        state.final_redactions = [p for p in state.confirmed_pii if p.refined_pixel_box]
        log.info(f"  {len(state.final_redactions)}/{len(state.confirmed_pii)} item(s) have pixel boxes")
        return state


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 7: CRITIC AGENT
# ══════════════════════════════════════════════════════════════════════════════
# CriticAgent constants — compiled once at module load, not per run()

# Override DROP when the reason text itself confirms the item IS PII
# (observed on CPU float32 where the model inverts KEEP/DROP vocabulary).
_PII_CONFIRM_RE = re.compile(
    r'is (?:personally identifiable|pii|sensitive|private)', re.IGNORECASE
)

# Safe-keep: never drop items whose label falls into a definitively-PII category.
# Covers identity, biographic, contact, and financial identifiers.
_CRITIC_SAFE_KEEP_LABELS_RE = re.compile(
    r'\b(?:'
    r'name|full\s+name|first\s+name|last\s+name|surname|given\s+name'
    r'|dob|date\s+of\s+birth|birth(?:day|date)?|birth'
    r'|address|street|zip(?:\s+code)?|postcode'
    r'|phone|mobile|email'
    r'|id(?:\s+number)?|ssn|passport|licence|license'
    r'|account|bank'
    r')\b',
    re.IGNORECASE,
)

# Safe-keep: never drop items whose value looks like a date.
# Accepts slash or hyphen separator: DD/MM/YYYY, MM-DD-YY, etc.
_DATE_VALUE_RE = re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}')

# ══════════════════════════════════════════════════════════════════════════════
_CRITIC_PROMPT = """You are a conservative PII safety reviewer for a document redaction system. Your job is to review a list of detected PII items and ONLY drop items that are clearly NOT personal information.

KEEP (do not drop) any of these even if uncertain:
- Names of real-sounding people
- Phone numbers (any format)
- Email addresses
- ID numbers of any kind
- Dates that could be date of birth
- Addresses and address components
- Bank account numbers, IFSC codes
- Government ID numbers

ONLY DROP items that are unambiguously not PII:
- Generic field labels ("Name:", "Address:", "Date:")
- Common English words with no identifying context
- Company names of large public organisations (not the individual's employer)
- Generic place names with no personal connection
- Document section headings
- Legal boilerplate text (standard clauses)
- The word "NONE", "N/A", "nil", "null"

BIAS TOWARD KEEPING. When uncertain, KEEP the item.
In a redaction system, missing PII is dangerous. Over-redacting is safe.

DO NOT drop items just because:
- "It could be a pseudonym" — KEEP IT
- "Without more context" — KEEP IT
- "It might not be PII" — KEEP IT
- "Common word" — if it's a name, KEEP IT

For each item output exactly:
KEEP -- <one line reason>
or
DROP -- <one line reason why it is clearly NOT PII>

Items to review:
{items}
"""


class CriticAgent(BaseAgent):
    """
    Final validation: reviews all detections and drops clear false positives.
    Skipped if fewer than 3 detections (overhead not worth it).
    """
    name = "CriticAgent"

    def __init__(self, tokenizer, model, cfg: OmniConfig):
        self.tokenizer = tokenizer
        self.model = model
        self.cfg = cfg

    def run(self, state: AgentState) -> AgentState:
        if len(state.final_redactions) < 3:
            log.info("  Fewer than 3 detections -- CriticAgent skipped")
            return state

        items_text = "\n".join(
            f"[{i+1}] label='{p.label}' value='{p.value}'"
            for i, p in enumerate(state.final_redactions)
        )
        messages = [
            {"role": "system", "content": "You are a conservative PII safety reviewer. When in doubt, KEEP the item."},
            {"role": "user", "content": _CRITIC_PROMPT.format(items=items_text)},
        ]
        encoded = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        encoded = encoded.to(self.model.device)

        with torch.inference_mode():
            gen_ids = self.model.generate(
                encoded, max_new_tokens=self.cfg.max_new_tokens, do_sample=False
            )

        review = self.tokenizer.decode(gen_ids[0][encoded.shape[1]:], skip_special_tokens=True)
        del encoded, gen_ids
        _free_vram("post-critic-gen")

        log.info(f"  [CRITIC OUTPUT]\n{review}")

        # Parse per-line decisions. A DROP whose reason text still says the item
        # IS PII means the model inverted the vocabulary (observed on CPU float32);
        # override those back to KEEP.
        drop_indices: set[int] = set()
        for m in re.finditer(
            r'\[(\d+)\]\s+(KEEP|DROP)\s*--\s*(.+)', review, re.IGNORECASE
        ):
            idx = int(m.group(1)) - 1
            decision = m.group(2).upper()
            reason = m.group(3)
            if decision == "DROP" and not _PII_CONFIRM_RE.search(reason):
                drop_indices.add(idx)

        # Safe-keep check: applied before any drop is committed.
        # Force-keep if the label is a definitively-PII category OR the value
        # looks like a date, regardless of what the LLM decided.
        kept, dropped = [], 0
        for i, pair in enumerate(state.final_redactions):
            if i in drop_indices:
                if _CRITIC_SAFE_KEEP_LABELS_RE.search(pair.label or "") or \
                        _DATE_VALUE_RE.match(pair.value or ""):
                    log.info(f"  [SAFE-KEEP] '{pair.label}' -> '{pair.value}' (PII label override)")
                    kept.append(pair)
                else:
                    log.info(f"  [DROP] '{pair.label}' -> '{pair.value}'")
                    dropped += 1
            else:
                kept.append(pair)

        log.info(f"  Critic: kept {len(kept)}, dropped {dropped}")
        state.final_redactions = kept
        return state


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 8: REDACTION AGENT
# ══════════════════════════════════════════════════════════════════════════════
class RedactionAgent(BaseAgent):
    """
    Draws black rectangles and writes redacted PNG + JSON audit sidecar.
    The audit JSON feeds directly into the blockchain layer (Node 4).
    """
    name = "RedactionAgent"

    def __init__(self, cfg: OmniConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self, state: AgentState) -> AgentState:
        _pii_filter = state.pii_filter or {}
        _skip_faces = not _pii_filter.get('faces', True)
        _skip_sigs  = not _pii_filter.get('signatures', True)

        # Face detection pass using YOLOv8 (runs before text redactions)
        _yolo_model_path = Path(__file__).parent / "yolov8n.pt"
        if not _skip_faces and _YOLO_AVAILABLE and _yolo_model_path.exists():
            from ultralytics import YOLO as _YOLO  # lazy — avoid CUDA init in web process
            yolo = _YOLO(str(_yolo_model_path))
            results = yolo(state.image_path, verbose=False)
            n_raw = sum(len(r.boxes) for r in results)
            log.info(f"  [FACE] YOLO raw detections: {n_raw}")
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == 0 and conf > 0.15:  # class 0 = person; lowered from 0.3 for ID-card photos
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                        state.final_redactions.append(LabelValuePair(
                            label="Face",
                            value="",
                            refined_pixel_box=[x1, y1, x2, y2],
                        ))
                        log.info(f"  [FACE] conf={conf:.2f} box=[{x1},{y1},{x2},{y2}]")
                    else:
                        log.info(f"  [FACE SKIP] cls={cls_id} conf={conf:.2f} (threshold 0.15)")
        elif not _skip_faces and not _yolo_model_path.exists():
            log.warning(f"  [FACE] Model not found: {_yolo_model_path} — face detection skipped")
        elif _skip_faces:
            log.info("  [FILTER] Face detection skipped (faces disabled by user)")

        # ── Image dimensions for sweep helpers ────────────────────────────────
        _w_img, _h_img = state.image_size if state.image_size != (0, 0) else (0, 0)

        def _covered(box: list[int]) -> bool:
            """True if box is >50% overlapped by any existing redaction (avoids duplicates)."""
            for _eb in [p.refined_pixel_box for p in state.final_redactions if p.refined_pixel_box]:
                _ix1, _iy1 = max(box[0], _eb[0]), max(box[1], _eb[1])
                _ix2, _iy2 = min(box[2], _eb[2]), min(box[3], _eb[3])
                _inter = max(0, _ix2 - _ix1) * max(0, _iy2 - _iy1)
                _area = (box[2] - box[0]) * (box[3] - box[1])
                if _area > 0 and _inter / _area > 0.5:
                    return True
            return False

        # ── Signature zone detection ──────────────────────────────────────────
        if _skip_sigs:
            log.info("  [FILTER] Signature detection skipped (signatures disabled by user)")
        # Pass 1: OCR word containing "signature" — redact everything to the right
        for _sw in ([] if _skip_sigs else state.ocr_words):
            if "signature" in _sw.text.lower() and _w_img:
                _sig_box = [0, _sw.box[1], _w_img, _sw.box[3] + 60]
                if not _covered(_sig_box):
                    state.final_redactions.append(LabelValuePair(
                        label="Signature", value="", refined_pixel_box=_sig_box
                    ))
                    log.info(f"  [SIGNATURE-OCR] '{_sw.text}' -> {_sig_box}")

        # Pass 2: OpenCV contours in bottom 30% with aspect ratio 2:1–8:1,
        #          then NMS to at most 2 non-overlapping boxes.
        if not _skip_sigs and _DESKEW_AVAILABLE and state.image_path:
            _cv_img = cv2.imread(state.image_path)
            if _cv_img is not None:
                _img_h, _img_w = _cv_img.shape[:2]
                _bottom_y = int(_img_h * 0.70)
                _roi = _cv_img[_bottom_y:, :]
                _gray = cv2.cvtColor(_roi, cv2.COLOR_BGR2GRAY)
                _edges = cv2.Canny(_gray, 50, 150)
                _cnts, _ = cv2.findContours(_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Collect all aspect-valid candidates
                _sig_candidates: list[list[int]] = []
                for _cnt in _cnts:
                    _x, _y, _cw, _ch = cv2.boundingRect(_cnt)
                    if _ch == 0:
                        continue
                    _aspect = _cw / _ch
                    if 2.0 <= _aspect <= 8.0 and (_cw * _ch) > 2000:
                        _sig_candidates.append([_x, _bottom_y + _y, _x + _cw, _bottom_y + _y + _ch])

                # NMS: sort largest-area first, keep if IoU < 0.3 with all kept boxes, max 2
                _sig_candidates.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
                _kept: list[list[int]] = []
                for _sb in _sig_candidates:
                    _sb_area = ((_sb[2]-_sb[0]) * (_sb[3]-_sb[1]))
                    _keep = True
                    for _kb in _kept:
                        _ix1 = max(_sb[0], _kb[0]); _iy1 = max(_sb[1], _kb[1])
                        _ix2 = min(_sb[2], _kb[2]); _iy2 = min(_sb[3], _kb[3])
                        _inter = max(0, _ix2-_ix1) * max(0, _iy2-_iy1)
                        _kb_area = (_kb[2]-_kb[0]) * (_kb[3]-_kb[1])
                        _union = _sb_area + _kb_area - _inter
                        if _union > 0 and _inter / _union > 0.3:
                            _keep = False
                            break
                    if _keep:
                        _kept.append(_sb)
                    if len(_kept) >= 2:
                        break

                # Merge the (at most 2) kept boxes if they overlap at all
                if len(_kept) == 2:
                    _a, _b = _kept[0], _kept[1]
                    _any_overlap = (min(_a[2], _b[2]) > max(_a[0], _b[0]) and
                                   min(_a[3], _b[3]) > max(_a[1], _b[1]))
                    if _any_overlap:
                        _kept = [[min(_a[0], _b[0]), min(_a[1], _b[1]),
                                  max(_a[2], _b[2]), max(_a[3], _b[3])]]
                        log.info(f"  [SIGNATURE-MERGE] -> {_kept[0]}")

                for _sig_box in _kept:
                    if not _covered(_sig_box):
                        state.final_redactions.append(LabelValuePair(
                            label="Signature", value="", refined_pixel_box=_sig_box
                        ))
                        log.info(f"  [SIGNATURE-CONTOUR] -> {_sig_box}")

        # ── Post-OCR sweep: catch missed ID numbers and long digit strings ────
        # Exact-match set for Sweep 2 (label tokens that ARE the trigger word alone)
        _ID_TRIGGER_EXACT = {"idn", "dl", "dln", "dd", "document discriminator",
                             "id number", "id no", "id"}
        # Substrings that flag a token as an ID label (catches "4d IDN:", "IDN:", etc.)
        _ID_LABEL_SUBSTRS = ("idn", "dln", "document discriminator")

        # Sweep 1a: single OCR token matching ID number pattern "XX[. ]XXX[. ]XXX"
        # EasyOCR may read "99 999 999" as "99.999 999" (dot instead of space) — both handled.
        for _ow in state.ocr_words:
            if re.search(r'\d{2}[.\s]\d{3}[.\s]\d{3}', _ow.text):
                if not _covered(_ow.box):
                    state.final_redactions.append(LabelValuePair(
                        label="IDNumber", value=_ow.text, refined_pixel_box=_ow.box
                    ))
                    log.info(f"  [OCR-SWEEP ID] '{_ow.text}' -> {_ow.box}")

        # Sweep 1b: 3-word window for cases where digits are separate OCR tokens
        for _i in range(len(state.ocr_words) - 2):
            _w0, _w1, _w2 = state.ocr_words[_i], state.ocr_words[_i + 1], state.ocr_words[_i + 2]
            _joined = f"{_w0.text} {_w1.text} {_w2.text}"
            if re.fullmatch(r'\d{2}\s\d{3}\s\d{3}', _joined):
                _ub = [
                    min(_w0.box[0], _w1.box[0], _w2.box[0]),
                    min(_w0.box[1], _w1.box[1], _w2.box[1]),
                    max(_w0.box[2], _w1.box[2], _w2.box[2]),
                    max(_w0.box[3], _w1.box[3], _w2.box[3]),
                ]
                if not _covered(_ub):
                    state.final_redactions.append(LabelValuePair(
                        label="IDNumber", value=_joined, refined_pixel_box=_ub
                    ))
                    log.info(f"  [OCR-SWEEP ID-3W] '{_joined}' -> {_ub}")

        # Sweep 2: ID-label token (exact or substring) followed within 3 tokens by a value.
        # Handles "4d IDN:" (substring match) as well as bare "IDN" (exact match).
        for _i, _lw in enumerate(state.ocr_words):
            _lt = _lw.text.lower()
            _is_id_label = (_lt in _ID_TRIGGER_EXACT or
                            any(sub in _lt for sub in _ID_LABEL_SUBSTRS))
            if _is_id_label:
                for _j in range(_i + 1, min(_i + 4, len(state.ocr_words))):
                    _vw = state.ocr_words[_j]
                    _vt = _vw.text.lower()
                    if re.search(r'\w', _vw.text) and not any(sub in _vt for sub in _ID_LABEL_SUBSTRS):
                        # Require at least 6 digits — rejects OCR garbage like "Andnlu Sompl"
                        if sum(c.isdigit() for c in _vw.text) < 6:
                            break
                        if not _covered(_vw.box):
                            state.final_redactions.append(LabelValuePair(
                                label="IDNumber", value=_vw.text, refined_pixel_box=_vw.box
                            ))
                            log.info(f"  [OCR-SWEEP LABEL] '{_lw.text}' -> '{_vw.text}' {_vw.box}")
                        break

        # Sweep 3: any token containing 10+ consecutive digits (document discriminator / SSN).
        # Uses re.search (not fullmatch) to catch "DD:138489882313*" style tokens.
        # The display value is cleaned to just the digit run; the box covers the full token.
        for _dw in state.ocr_words:
            _dm = re.search(r'\d{10,}', _dw.text)
            if _dm and not _covered(_dw.box):
                state.final_redactions.append(LabelValuePair(
                    label="LongDigit", value=_dm.group(), refined_pixel_box=_dw.box
                ))
                log.info(f"  [OCR-SWEEP LONG-DIGIT] '{_dm.group()}' (raw: '{_dw.text}') -> {_dw.box}")

        # ── Sweep 4: deterministic regex sweep over full OCR text ────────────
        # Catches PAN / IFSC / CIN / GST / Aadhaar / dates that agents missed.
        if state.ocr_words:
            _ocr_full = " ".join(w.text for w in state.ocr_words)
            _active_filter = state.pii_filter or {}
            for _rpat, _rfkey, _rlabel in _REGEX_PII_PATTERNS:
                if not _active_filter.get(_rfkey, True):
                    continue
                for _rm in re.finditer(_rpat, _ocr_full):
                    _rval = _rm.group().strip()
                    # Find the OCR word(s) that contain this match and redact them
                    for _rw in state.ocr_words:
                        if _rval in _rw.text or _rw.text in _rval:
                            if not _covered(_rw.box):
                                state.final_redactions.append(LabelValuePair(
                                    label=_rlabel, value=_rval, refined_pixel_box=_rw.box
                                ))
                                log.info(f"  [REGEX-SWEEP4] {_rlabel}: '{_rval}' -> {_rw.box}")

        if not state.final_redactions and not state.custom_terms and not state.custom_patterns:
            log.warning("  No confirmed redactions -- image unchanged")
            return state

        # ── Custom term matching (OCR-based) ─────────────────────────────────
        if state.ocr_words and (state.custom_terms or state.custom_patterns):
            for _ow in state.ocr_words:
                _ow_lower = _ow.text.lower()
                for _ct in state.custom_terms:
                    if _ct.lower() in _ow_lower and not _covered(_ow.box):
                        state.final_redactions.append(LabelValuePair(
                            label="Custom", value=_ct, refined_pixel_box=_ow.box
                        ))
                        log.info(f"  [CUSTOM-TERM] '{_ct}' matched '{_ow.text}' -> {_ow.box}")
                for _cp in state.custom_patterns:
                    try:
                        if re.search(_cp, _ow.text, re.IGNORECASE) and not _covered(_ow.box):
                            state.final_redactions.append(LabelValuePair(
                                label="Custom", value=_cp, refined_pixel_box=_ow.box
                            ))
                            log.info(f"  [CUSTOM-PATTERN] '{_cp}' matched '{_ow.text}' -> {_ow.box}")
                    except re.error:
                        pass

        if not state.final_redactions:
            log.warning("  No confirmed redactions -- image unchanged")
            return state

        # Build the redaction records list (always — dry_run needs it too)
        records = []
        for pair in state.final_redactions:
            box = pair.refined_pixel_box
            if not box:
                continue
            src = "custom" if pair.label == "Custom" else "ai"
            records.append({"label": pair.label, "value": pair.value, "pixel_box": box, "source": src})
            prefix = f"[{pair.label}] " if pair.label else ""
            log.info(f"  [REDACT] {prefix}'{pair.value}'  box={box}")

        # Coverage: fraction of image area covered by redaction boxes
        iw, ih = state.image_size if state.image_size else (0, 0)
        total_px = iw * ih
        if total_px > 0:
            redacted_px = sum(
                max(0, b[2] - b[0]) * max(0, b[3] - b[1])
                for r in records for b in [r["pixel_box"]] if r.get("pixel_box")
            )
            state.coverage_pct = round(redacted_px / total_px * 100, 1)
        log.info(f"  Coverage: {state.coverage_pct}% ({total_px}px total)")

        if state.dry_run:
            log.info(f"  [DRY RUN] Would redact {len(records)} item(s) — skipping file write and audit sidecar")
            state.audit = {
                "doc_type": state.doc_type,
                "redaction_count": len(records),
                "redactions": records,
                "coverage_pct": state.coverage_pct,
                "dry_run": True,
            }
            return state

        # Real run: draw rectangles and save output image
        img = Image.open(state.image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for pair in state.final_redactions:
            box = pair.refined_pixel_box
            if not box:
                continue
            xmin, ymin, xmax, ymax = box
            draw.rectangle([xmin, ymin, xmax, ymax], fill="black")

        stem = Path(state.image_path).stem
        out_path = Path(self.cfg.output_dir) / f"redacted_{stem}.png"
        img.save(out_path)
        log.info(f"Saved: {out_path}  ({len(records)} redaction(s))")

        # Audit sidecar for blockchain layer (Node 4)
        audit = {
            "source": str(Path(state.image_path).resolve()),
            "output": str(out_path.resolve()),
            "doc_type": state.doc_type,
            "model_vlm": CFG.vlm_model_id,
            "model_text": CFG.text_model_id,
            "redaction_count": len(records),
            "redactions": records,
            "coverage_pct": state.coverage_pct,
        }
        audit_path = out_path.with_suffix(".audit.json")
        audit["audit_path"] = str(audit_path)
        audit_path.write_text(json.dumps(audit, indent=2))
        log.info(f"Audit: {audit_path}")
        # Anchoring is done by backend_server with the correct owner_address.
        # Do NOT call _anchor_audit() here — it runs without owner_address
        # and would store the record under 0x0000 before the backend can anchor it.

        state.output_path = str(out_path)
        state.audit = audit
        return state


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
# PDF AGENT
# ══════════════════════════════════════════════════════════════════════════════
class PDFAgent(BaseAgent):
    """
    Redacts PII from PDF files.

    Text-based PDFs (embedded text detected by pdfplumber):
      - Extract text + character coordinates per page via pdfplumber.
      - Build synthetic OCR words from pdfplumber character spans so the
        existing TextPIIAgent / ContextAgent / BBRefiner pipeline can run.
      - Draw black rectangles over confirmed PII using PyMuPDF (fitz).
      - Save a redacted PDF.

    Scanned PDFs (no embedded text, or text coverage < threshold):
      - Convert each page to a PNG at 200 DPI using pdf2image.
      - Run each PNG through the full image pipeline (LayoutAgent → …
        → RedactionAgent) and collect the redacted PNGs.
      - Stitch the redacted PNGs back into a single PDF with PyMuPDF.
    """
    name = "PDFAgent"

    # Minimum fraction of pages that must have embedded text to be
    # considered "text-based" rather than scanned.
    _TEXT_COVERAGE_THRESHOLD = 0.5
    # DPI for rasterising scanned pages.
    _SCAN_DPI = 200

    def __init__(self, cfg: OmniConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _words_from_page(page) -> list[OCRWord]:
        """Convert pdfplumber word-level spans to OCRWord objects.

        pdfplumber's extract_words() returns dicts with keys
        x0, top, x1, bottom, text — all in PDF points (72 pt = 1 inch).
        We store them as [xmin, ymin, xmax, ymax] in points; PyMuPDF
        uses the same coordinate system so no scaling is needed.
        """
        words = []
        for w in page.extract_words():
            words.append(OCRWord(
                text=w["text"],
                box=[int(w["x0"]), int(w["top"]), int(w["x1"]), int(w["bottom"])],
                confidence=1.0,
            ))
        return words

    @staticmethod
    def _page_has_text(page) -> bool:
        text = (page.extract_text() or "").strip()
        return len(text) > 20

    def _redact_text_pdf(self, pdf_path: str, tokenizer, text_model) -> str:
        """Extract text, find PII, draw redaction boxes, return output path."""
        out_path = Path(self.cfg.output_dir) / f"redacted_{Path(pdf_path).stem}.pdf"
        doc = _fitz.open(pdf_path)
        all_audit_records: list[dict] = []

        with pdfplumber.open(pdf_path) as plumb:
            for page_idx, (fitz_page, plumb_page) in enumerate(
                zip(doc, plumb.pages)
            ):
                ocr_words = self._words_from_page(plumb_page)
                if not ocr_words:
                    continue

                # Reuse the text + context pipeline on synthetic OCR words
                state = AgentState(image_path=pdf_path)
                state.input_format = "pdf"
                state.ocr_words = ocr_words

                state = TextPIIAgent(tokenizer, text_model, self.cfg)(state)
                state = ContextAgent(tokenizer, text_model, self.cfg)(state)

                # BBRefiner: text-match only (no image_size needed for PDFs)
                refiner = BBRefinerAgent(self.cfg)
                state.image_size = (int(plumb_page.width), int(plumb_page.height))
                state = refiner(state)

                # Draw redaction boxes on the fitz page
                for pair in state.final_redactions:
                    box = pair.refined_pixel_box
                    if not box:
                        continue
                    rect = _fitz.Rect(box[0], box[1], box[2], box[3])
                    fitz_page.draw_rect(rect, color=(0, 0, 0), fill=(0, 0, 0))
                    all_audit_records.append({
                        "page": page_idx + 1,
                        "label": pair.label,
                        "value": pair.value,
                        "box_pt": box,
                    })
                    log.info(f"  [PDF-REDACT p{page_idx+1}] [{pair.label}] '{pair.value}' {box}")

        doc.save(str(out_path), deflate=True)
        doc.close()
        log.info(f"Saved redacted PDF: {out_path}  ({len(all_audit_records)} redaction(s))")
        return str(out_path), all_audit_records

    def _redact_scanned_pdf(self, pdf_path: str, orchestrator) -> str:
        """Rasterise pages, run image pipeline on each, stitch back to PDF."""
        tmp_dir = Path(self.cfg.output_dir) / "_pdf_pages"
        tmp_dir.mkdir(exist_ok=True)

        pages_pil = _pdf2images(pdf_path, dpi=self._SCAN_DPI)
        redacted_pngs: list[str] = []
        all_audit_records: list[dict] = []

        for page_idx, pil_img in enumerate(pages_pil):
            png_path = str(tmp_dir / f"page_{page_idx:04d}.png")
            pil_img.save(png_path)

            # Run the full image pipeline on this page
            page_state = orchestrator._run_image_pipeline(png_path)
            if page_state.output_path and Path(page_state.output_path).exists():
                redacted_pngs.append(page_state.output_path)
            else:
                redacted_pngs.append(png_path)  # unchanged if no redactions

            for pair in page_state.final_redactions:
                all_audit_records.append({
                    "page": page_idx + 1,
                    "label": pair.label,
                    "value": pair.value,
                    "box_px": pair.refined_pixel_box,
                })

        # Stitch PNGs into a PDF with PyMuPDF
        out_path = Path(self.cfg.output_dir) / f"redacted_{Path(pdf_path).stem}.pdf"
        out_doc = _fitz.open()
        for png in redacted_pngs:
            img_doc = _fitz.open(png)
            pdf_bytes = img_doc.convert_to_pdf()
            img_doc.close()
            img_as_pdf = _fitz.open("pdf", pdf_bytes)
            out_doc.insert_pdf(img_as_pdf)
        out_doc.save(str(out_path))
        out_doc.close()

        log.info(f"Saved redacted PDF: {out_path}  ({len(all_audit_records)} redaction(s))")
        return str(out_path), all_audit_records

    def run(self, state: AgentState, tokenizer, text_model, orchestrator) -> AgentState:
        pdf_path = state.image_path
        log.info(f"[PDFAgent] processing: {pdf_path}")

        # Detect whether the PDF has embedded text
        with pdfplumber.open(pdf_path) as plumb:
            pages_with_text = sum(1 for p in plumb.pages if self._page_has_text(p))
            total_pages = len(plumb.pages)

        text_fraction = pages_with_text / max(total_pages, 1)
        log.info(f"  {pages_with_text}/{total_pages} page(s) have embedded text "
                 f"(fraction={text_fraction:.2f})")

        if text_fraction >= self._TEXT_COVERAGE_THRESHOLD:
            out_path, records = self._redact_text_pdf(pdf_path, tokenizer, text_model)
        else:
            out_path, records = self._redact_scanned_pdf(pdf_path, orchestrator)

        # Write audit sidecar
        audit = {
            "source": str(Path(pdf_path).resolve()),
            "output": str(Path(out_path).resolve()),
            "doc_type": state.doc_type,
            "redaction_count": len(records),
            "redactions": records,
        }
        audit_path = Path(out_path).with_suffix(".audit.json")
        audit["audit_path"] = str(audit_path)
        audit_path.write_text(json.dumps(audit, indent=2))
        log.info(f"Audit: {audit_path}")
        # Anchoring is done by backend_server with the correct owner_address.

        state.output_path = out_path
        state.audit = audit
        return state


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO AGENT
# ══════════════════════════════════════════════════════════════════════════════
class AudioAgent(BaseAgent):
    """
    Redacts PII from audio files.

    Pipeline:
      1. Transcribe with faster-whisper (word-level timestamps).
      2. Reconstruct full transcript; run TextPIIAgent to identify PII spans.
      3. For each PII phrase, binary-search the word timestamps to find
         [start - 200 ms, end + 200 ms] silence window.
      4. Replace those segments with 0-amplitude audio using pydub.
      5. Export as the original format; write audit JSON.
    """
    name = "AudioAgent"

    _BUFFER_MS = 200          # silence buffer around each PII span
    _WHISPER_MODEL = "base"   # trade-off: "tiny" is fastest, "small" better accuracy

    def __init__(self, cfg: OmniConfig):
        self.cfg = cfg
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _find_span_timestamps(
        pii_phrase: str,
        word_timestamps: list[dict],   # [{"word", "start", "end"}, ...]
    ) -> Optional[tuple[float, float]]:
        """Find start/end seconds for a PII phrase in the word-timestamp list.

        Tries a sliding-window match over the transcript words.  Matching is
        case-insensitive; only alphanumeric characters are compared so leading
        spaces (" Alice") and trailing punctuation ("Johnson.") in Whisper word
        tokens don't prevent a match.
        """
        def _alnum(s: str) -> str:
            """Keep only letters and digits, lowercase."""
            return re.sub(r'[^a-z0-9]', '', s.lower())

        pii_tokens = [_alnum(t) for t in pii_phrase.split() if _alnum(t)]
        n = len(pii_tokens)
        if n == 0:
            return None

        ts_alnum = [_alnum(w["word"]) for w in word_timestamps]

        # Exact sliding-window match on normalised tokens
        for i in range(len(ts_alnum) - n + 1):
            if ts_alnum[i:i + n] == pii_tokens:
                return word_timestamps[i]["start"], word_timestamps[i + n - 1]["end"]

        # Partial match: accept window where every non-empty pii_token is a
        # substring of the corresponding ts_token (handles OCR hallucinations
        # like "55ml" matching "555").
        for i in range(len(ts_alnum) - n + 1):
            window = ts_alnum[i:i + n]
            if all(pt and (pt in wt or wt in pt) for pt, wt in zip(pii_tokens, window) if pt):
                return word_timestamps[i]["start"], word_timestamps[i + n - 1]["end"]

        # Single-token fallback: substring in any word
        if n == 1:
            for w in word_timestamps:
                if pii_tokens[0] and (pii_tokens[0] in _alnum(w["word"])):
                    return w["start"], w["end"]

        # Digit-word fallback: for numeric PII ("9876543210") find the run of
        # spoken digit-words whose combined digit values match.
        target_digits = re.sub(r'[\s\-]', '', pii_phrase)
        if target_digits.isdigit():
            digit_ws = [
                w for w in word_timestamps
                if re.sub(r'[^a-z]', '', w["word"].lower()) in _SPOKEN_ONES
            ]
            for i in range(len(digit_ws)):
                run = ''
                for j in range(i, len(digit_ws)):
                    d = _SPOKEN_ONES.get(re.sub(r'[^a-z]', '', digit_ws[j]["word"].lower()), '')
                    run += d
                    if run == target_digits:
                        return digit_ws[i]["start"], digit_ws[j]["end"]
                    if not target_digits.startswith(run):
                        break

        return None

    def run(self, state: AgentState, tokenizer, text_model) -> AgentState:
        if not _WHISPER_AVAILABLE:
            log.error("[AudioAgent] faster-whisper not installed — skipping")
            return state
        if not _PYDUB_AVAILABLE:
            log.error("[AudioAgent] pydub not installed — skipping")
            return state

        audio_path = state.image_path
        suffix = Path(audio_path).suffix.lstrip(".").lower() or "mp3"
        log.info(f"[AudioAgent] processing: {audio_path}")

        # ── Step 1: Transcribe ──────────────────────────────────────────────
        # Always use CPU for Whisper: the text model already occupies the GPU
        # and cuDNN may not be available for faster-whisper's CUDA kernels.
        whisper = _WhisperModel(self._WHISPER_MODEL, device="cpu", compute_type="int8")

        _initial_prompt = (
            "Transcribe all numbers as digits. "
            "Phone numbers like nine eight seven six five four three two one zero "
            "should be written as 9876543210. "
            "Aadhaar numbers, account numbers, dates like fifteenth March nineteen "
            "eighty seven should be written as 15th March 1987. "
            "PAN numbers like A B C D E one two three four F should be written as ABCDE1234F."
        )
        segments, _ = whisper.transcribe(
            audio_path,
            word_timestamps=True,
            beam_size=5,
            initial_prompt=_initial_prompt,
        )
        word_timestamps: list[dict] = []
        seg_texts: list[str] = []
        for seg in segments:
            seg_texts.append(seg.text)
            for w in (seg.words or []):
                word_timestamps.append({"word": w.word, "start": w.start, "end": w.end})

        # Use segment-level text (Whisper formats this as natural sentences) rather
        # than re-joining individual word tokens, which would produce double-spaces
        # and break email/phone formatting.
        full_transcript = " ".join(t.strip() for t in seg_texts if t.strip())
        log.info(f"  Transcript ({len(word_timestamps)} words): {full_transcript[:120]}…")
        del whisper
        _free_vram("post-whisper")

        # ── Step 2: Normalise spoken digits / ordinals / years ──────────────
        # Appends a normalised copy of the transcript (ordinals → "15th",
        # digit-runs → "9876543210", years → "1987") so that regex patterns
        # in TextPIIAgent catch phone numbers, Aadhaar, dates etc.
        normalised_transcript = normalise_audio_transcript(full_transcript)
        log.info(f"  Raw transcript:        {full_transcript[:120]}")
        log.info(f"  Normalised transcript: {normalised_transcript.split(chr(10))[-1][:120]}")

        # ── Step 3: Find PII spans via TextPIIAgent ─────────────────────────
        # Store the full transcript as a single OCRWord so TextPIIAgent receives
        # one coherent sentence instead of fragmented per-word tokens.
        state.ocr_words = [
            OCRWord(text=normalised_transcript, box=[0, 0, 1, 1], confidence=1.0)
        ]
        state = TextPIIAgent(tokenizer, text_model, self.cfg)(state)

        pii_phrases = state.raw_text_pii
        log.info(f"  PII phrases found: {pii_phrases}")

        # ── Step 3: Map phrases → timestamp windows ─────────────────────────
        silence_windows: list[tuple[float, float]] = []   # (start_s, end_s)
        for phrase in pii_phrases:
            span = self._find_span_timestamps(phrase, word_timestamps)
            if span:
                start_s = max(0.0, span[0] - self._BUFFER_MS / 1000)
                end_s   = span[1] + self._BUFFER_MS / 1000
                silence_windows.append((start_s, end_s))
                log.info(f"  [AUDIO-PII] '{phrase}' -> [{start_s:.2f}s, {end_s:.2f}s]")
            else:
                log.warning(f"  [AUDIO-PII] '{phrase}' -- no timestamp match")

        if not silence_windows:
            log.warning("  No audio PII spans located — output unchanged")
            state.output_path = audio_path
            return state

        # ── Step 4: Silence PII segments ────────────────────────────────────
        audio = _AudioSegment.from_file(audio_path)
        silence = _AudioSegment.silent(duration=1)  # 1 ms template, will be extended

        # Merge overlapping windows before silencing
        silence_windows.sort()
        merged: list[tuple[float, float]] = []
        for start, end in silence_windows:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        redacted_audio = audio
        offset_ms = 0  # track cumulative slice offset after replacements
        for start_s, end_s in merged:
            start_ms = int(start_s * 1000) + offset_ms
            end_ms   = int(end_s   * 1000) + offset_ms
            duration_ms = end_ms - start_ms
            if duration_ms <= 0:
                continue
            silence_seg = _AudioSegment.silent(duration=duration_ms,
                                               frame_rate=audio.frame_rate)
            redacted_audio = (redacted_audio[:start_ms]
                              + silence_seg
                              + redacted_audio[end_ms:])

        # ── Step 5: Export ───────────────────────────────────────────────────
        stem = Path(audio_path).stem
        out_path = Path(self.cfg.output_dir) / f"redacted_{stem}.{suffix}"
        export_fmt = {"m4a": "mp4", "ogg": "ogg"}.get(suffix, suffix)
        redacted_audio.export(str(out_path), format=export_fmt)
        log.info(f"Saved redacted audio: {out_path}")

        # ── Step 6: Audit JSON ───────────────────────────────────────────────
        audit = {
            "source": str(Path(audio_path).resolve()),
            "output": str(out_path.resolve()),
            "doc_type": state.doc_type,
            "redaction_count": len(merged),
            "transcript_word_count": len(word_timestamps),
            "pii_phrases": pii_phrases,
            "silenced_segments": [
                {"start_s": round(s, 3), "end_s": round(e, 3)}
                for s, e in merged
            ],
        }
        audit_path = out_path.with_suffix(".audit.json")
        audit["audit_path"] = str(audit_path)
        audit_path.write_text(json.dumps(audit, indent=2))
        log.info(f"Audit: {audit_path}")
        # Anchoring is done by backend_server with the correct owner_address.

        state.output_path = str(out_path)
        state.audit = audit
        return state


# ══════════════════════════════════════════════════════════════════════════════
class OmniShieldOrchestrator:
    """
    Coordinates all 8 agents. Supports two memory modes:

    "lazy" (default, safer for 8 GB)
        VLM loaded -> LayoutAgent -> VLM unloaded
        Text model loaded -> TextPIIAgent + ContextAgent -> Text model unloaded
        BBRefinerAgent (no model)
        Text model loaded -> CriticAgent -> Text model unloaded
        RedactionAgent (no model)

    "dual" (faster, ~30s saved on reloads)
        Both models loaded simultaneously (~3.2 GB peak weights).
        Leaves ~4.3 GB for KV-cache + OS. Tight but workable.
    """

    def __init__(self, cfg: OmniConfig = CFG):
        self.cfg = cfg

    def run(
        self,
        image_path: str,
        pii_filter: dict | None = None,
        custom_terms: list[str] | None = None,
        custom_patterns: list[str] | None = None,
        dry_run: bool = False,
    ) -> AgentState:
        if not Path(image_path).exists():
            log.error(f"File not found: {image_path}")
            return AgentState(image_path=image_path)

        state = AgentState(image_path=image_path)
        if pii_filter:
            state.pii_filter = pii_filter
        if custom_terms:
            state.custom_terms = custom_terms
        if custom_patterns:
            state.custom_patterns = custom_patterns
        state.dry_run = dry_run

        t0 = time.perf_counter()

        log.info("=" * 60)
        log.info(f"Omni-Shield  |  {image_path}")
        log.info("=" * 60)

        # Stage 0: Route (no model)
        _t = time.perf_counter(); state = RouterAgent()(state); state.agent_timings["RouterAgent"] = round(time.perf_counter() - _t, 2)

        if state.input_format == "pdf":
            if not _PDF_AVAILABLE:
                log.error("PDF support requires: pip install pdfplumber pymupdf pdf2image")
                return state
            tokenizer, text_model = load_text_model(self.cfg)
            state = PDFAgent(self.cfg).run(state, tokenizer, text_model, self)
            del tokenizer, text_model
            # Unconditionally release all GPU model caches so the rq worker can
            # claim VRAM.  PDFAgent may also have loaded VLM for scanned pages.
            _evict_all_models("after PDFAgent")
            # Safety net: explicitly null caches in case _evict_all_models
            # skipped due to a CUDA availability check.
            _VLM_CACHE["model"] = None
            _VLM_CACHE["processor"] = None
            _TEXT_MODEL_CACHE["model"] = None
            _TEXT_MODEL_CACHE["tokenizer"] = None
            gc.collect()

        elif state.input_format == "audio":
            if not (_WHISPER_AVAILABLE and _PYDUB_AVAILABLE):
                log.error("Audio support requires: pip install faster-whisper pydub")
                return state
            tokenizer, text_model = load_text_model(self.cfg)
            state = AudioAgent(self.cfg).run(state, tokenizer, text_model)
            del tokenizer, text_model
            _evict_all_models("after AudioAgent")
            # Safety net: explicitly null caches.
            _VLM_CACHE["model"] = None
            _VLM_CACHE["processor"] = None
            _TEXT_MODEL_CACHE["model"] = None
            _TEXT_MODEL_CACHE["tokenizer"] = None
            gc.collect()

        else:
            # Image pipeline — runs in the rq worker, not the web process.
            # Do NOT evict here; the worker keeps the cache warm between jobs.
            state = self._run_image_pipeline_on_state(state)

        log.info(f"Pipeline complete in {time.perf_counter() - t0:.1f}s -> {state.output_path}")
        log.info("=" * 60)
        return state

    def _run_image_pipeline(self, image_path: str) -> AgentState:
        """Run the full image pipeline on a single PNG/JPEG. Used by PDFAgent
        when processing scanned PDF pages."""
        state = AgentState(image_path=image_path)
        state = RouterAgent()(state)
        return self._run_image_pipeline_on_state(state)

    def _run_image_pipeline_on_state(self, state: AgentState) -> AgentState:
        """Run OCR → model agents → redaction on an already-routed AgentState."""
        _t = time.perf_counter(); state = OCRAgent()(state); state.agent_timings["OCRAgent"] = round(time.perf_counter() - _t, 2)
        if self.cfg.memory_mode == "dual":
            state = self._run_dual(state)
        else:
            state = self._run_lazy(state)
        _t = time.perf_counter(); state = RedactionAgent(self.cfg)(state); state.agent_timings["RedactionAgent"] = round(time.perf_counter() - _t, 2)
        return state

    def _run_dual(self, state: AgentState) -> AgentState:
        processor, vlm = load_vlm(self.cfg)
        tokenizer, text_model = load_text_model(self.cfg)

        _t = time.perf_counter(); state = LayoutAgent(processor, vlm, self.cfg)(state); state.agent_timings["LayoutAgent"] = round(time.perf_counter() - _t, 2)
        del processor, vlm
        gc.collect()
        torch.cuda.synchronize()
        _free_vram("after LayoutAgent (dual)")

        _t = time.perf_counter(); state = TextPIIAgent(tokenizer, text_model, self.cfg)(state); state.agent_timings["TextPIIAgent"] = round(time.perf_counter() - _t, 2)
        _t = time.perf_counter(); state = ContextAgent(tokenizer, text_model, self.cfg)(state); state.agent_timings["ContextAgent"] = round(time.perf_counter() - _t, 2)
        _t = time.perf_counter(); state = BBRefinerAgent(self.cfg)(state); state.agent_timings["BBRefinerAgent"] = round(time.perf_counter() - _t, 2)
        _t = time.perf_counter(); state = CriticAgent(tokenizer, text_model, self.cfg)(state); state.agent_timings["CriticAgent"] = round(time.perf_counter() - _t, 2)
        del tokenizer, text_model
        gc.collect()
        torch.cuda.synchronize()
        _free_vram("after text agents (dual)")
        return state

    def _run_lazy(self, state: AgentState) -> AgentState:
        # VLM pass
        processor, vlm = load_vlm(self.cfg)
        _t = time.perf_counter(); state = LayoutAgent(processor, vlm, self.cfg)(state); state.agent_timings["LayoutAgent"] = round(time.perf_counter() - _t, 2)
        del processor, vlm
        gc.collect()
        torch.cuda.synchronize()
        _free_vram("after LayoutAgent (lazy)")

        # Single text-model pass — BBRefiner and Critic folded in here to
        # avoid a second load (CPU float32 ~12 GB would OOM on second load).
        tokenizer, text_model = load_text_model(self.cfg)
        _t = time.perf_counter(); state = TextPIIAgent(tokenizer, text_model, self.cfg)(state); state.agent_timings["TextPIIAgent"] = round(time.perf_counter() - _t, 2)
        _t = time.perf_counter(); state = ContextAgent(tokenizer, text_model, self.cfg)(state); state.agent_timings["ContextAgent"] = round(time.perf_counter() - _t, 2)
        _t = time.perf_counter(); state = BBRefinerAgent(self.cfg)(state); state.agent_timings["BBRefinerAgent"] = round(time.perf_counter() - _t, 2)
        _t = time.perf_counter(); state = CriticAgent(tokenizer, text_model, self.cfg)(state); state.agent_timings["CriticAgent"] = round(time.perf_counter() - _t, 2)
        del tokenizer, text_model
        gc.collect()
        torch.cuda.synchronize()
        _free_vram("after text agents (lazy)")
        return state


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "new.jpeg"
    state = OmniShieldOrchestrator(CFG).run(image_path)

    print("\nSummary")
    print("-------")
    print(f"Doc type   : {state.doc_type}")
    print(f"Redactions : {len(state.final_redactions)}")
    print(f"Output     : {state.output_path}")
