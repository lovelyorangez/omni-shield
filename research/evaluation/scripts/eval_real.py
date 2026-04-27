"""
Omni-Shield Real-Document Evaluation
Evaluates the live backend against annotated real specimen ID cards.

Key differences from eval.py:
  - iou_thresh=0.0  — text value matching only (annotations may have bbox=null)
  - Levenshtein distance < 3 as a fuzzy-match fallback for OCR noise
  - Default paths point to real_samples/
  - Results carry evaluation_type="real_specimen_text_only"

USAGE:
  python eval_real.py \\
    --test-dir  ../../datasets/real_samples/images \\
    --annotations ../../datasets/real_samples/annotations \\
    --output    ../../results/raw/eval_real.json

Requires:
  - Backend running: uvicorn backend_server:app --port 8000
  - Annotations in place: python annotate_real_claude.py
  - pip install python-Levenshtein
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import requests

try:
    from Levenshtein import distance as _levenshtein_distance
    _LEVENSHTEIN_AVAILABLE = True
except ImportError:
    _LEVENSHTEIN_AVAILABLE = False

# ── shared constants (mirrored from eval.py) ─────────────────────────────────

API_URL    = "http://localhost:8000"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

PII_CATEGORIES = [
    "names", "faces", "signatures", "phones",
    "emails", "dob", "id_numbers", "addresses",
    "org_names", "dates",
]

# Text-only matching: every match goes through substring containment, not IoU.
# Set to 0.0 so bbox_iou is never the deciding factor.
IOU_THRESH = 0.0

LABEL_TO_CATEGORY = {
    "name": "names", "surname": "names", "given name": "names",
    "given names": "names", "full name": "names", "first name": "names",
    "last name": "names", "family name": "names", "forename": "names",
    "holder": "names", "cardholder": "names",
    "nom": "names", "prenom": "names", "nachname": "names", "vorname": "names",
    "face": "faces", "photo": "faces", "photograph": "faces",
    "picture": "faces", "image": "faces",
    "signature": "signatures", "sign": "signatures",
    "phone": "phones", "mobile": "phones", "tel": "phones",
    "telephone": "phones", "contact": "phones",
    "email": "emails", "e-mail": "emails",
    "date of birth": "dob", "dob": "dob", "birth date": "dob",
    "birth": "dob", "born": "dob", "birthdate": "dob",
    "geburtsdatum": "dob", "date naissance": "dob",
    "id no": "id_numbers", "id no.": "id_numbers", "id number": "id_numbers",
    "license no": "id_numbers", "licence no": "id_numbers",
    "licence no.": "id_numbers", "licence number": "id_numbers",
    "license number": "id_numbers", "passport no": "id_numbers",
    "passport number": "id_numbers", "document no": "id_numbers",
    "document number": "id_numbers", "card no": "id_numbers",
    "card number": "id_numbers", "number": "id_numbers",
    "dl no": "id_numbers", "driver license": "id_numbers",
    "driving licence": "id_numbers", "national id": "id_numbers",
    "pan": "id_numbers", "aadhar": "id_numbers", "aadhaar": "id_numbers",
    "ssn": "id_numbers", "medicare": "id_numbers",
    "address": "addresses", "addr": "addresses",
    "home address": "addresses", "residential address": "addresses",
    "adresse": "addresses",
    "org": "org_names", "organisation": "org_names",
    "organization": "org_names", "issuer": "org_names",
    "date": "dates", "expiry": "dates", "expiry date": "dates",
    "expiration": "dates", "issue date": "dates", "valid until": "dates",
}


def normalize_cat(label: str, value: str = "") -> str:
    import re
    s = label.lower().strip()
    if s in LABEL_TO_CATEGORY:
        return LABEL_TO_CATEGORY[s]
    for k, v in LABEL_TO_CATEGORY.items():
        if k in s:
            return v
    if value:
        v = value.strip()
        if re.match(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", v): return "dob"
        if re.match(r"\d{1,2}\s+\w+\s+\d{4}", v):             return "dob"
        if re.match(r"[\w.+-]+@[\w-]+\.\w+", v):               return "emails"
        if re.match(r"[\d\s\+\-\(\)]{7,}$", v):                return "phones"
        if re.match(r"[A-Z]{1,3}\d{5,}", v):                   return "id_numbers"
    return s


def _text_match(pv: str, gv: str) -> bool:
    """
    Return True if two value strings should be considered a match.

    Matching tiers (in order):
      1. Exact match (case-insensitive)
      2. Substring containment — one value inside the other
      3. Levenshtein distance ≤ 2 on the shorter of the two strings
         (catches single-character OCR errors like '0' vs 'O', missing
         space, etc.)  Only applied when both strings are ≥ 4 chars to
         avoid false positives on very short values.
    """
    if not pv or not gv:
        return False
    if pv == gv:
        return True
    if pv in gv or gv in pv:
        return True
    if _LEVENSHTEIN_AVAILABLE and len(pv) >= 4 and len(gv) >= 4:
        if _levenshtein_distance(pv, gv) <= 2:
            return True
    return False


def match_text_only(preds: list[dict], gts: list[dict]) -> dict:
    """
    Text-value matching — no IoU (iou_thresh=0.0 throughout).

    A prediction matches a GT item when:
      - categories agree, AND
      - for text fields: _text_match() returns True
        (exact / substring / Levenshtein ≤ 2)
      - for visual fields (face, signature): category match alone suffices
        because the value is null
    """
    counts  = {c: {"tp": 0, "fp": 0, "fn": 0} for c in PII_CATEGORIES}
    gt_used: set[int] = set()

    for pred in preds:
        cat = normalize_cat(pred.get("label", ""), pred.get("value", "") or "")
        if cat not in PII_CATEGORIES:
            counts.setdefault(cat, {"tp": 0, "fp": 0, "fn": 0})
            counts[cat]["fp"] += 1
            continue

        pv      = (pred.get("value") or "").lower().strip()
        matched = False

        for i, gt in enumerate(gts):
            if i in gt_used or gt.get("category") != cat:
                continue
            gv = (gt.get("value") or "").lower().strip()

            if cat in ("faces", "signatures"):
                # Visual fields: presence of the category is sufficient
                gt_used.add(i)
                matched = True
                break

            if _text_match(pv, gv):
                gt_used.add(i)
                matched = True
                break

        counts[cat]["tp" if matched else "fp"] += 1

    for i, gt in enumerate(gts):
        if i not in gt_used and gt.get("category") in PII_CATEGORIES:
            counts[gt["category"]]["fn"] += 1

    return counts


def metrics(counts: dict) -> dict:
    out = {}
    tp_tot = fp_tot = fn_tot = 0
    for cat, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        tp_tot += tp; fp_tot += fp; fn_tot += fn
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        out[cat] = {
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1":        round(f, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }
    f1s = [v["f1"] for v in out.values()]
    out["macro_f1"] = round(float(np.mean(f1s)), 4)
    mp = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) else 0.0
    mr = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) else 0.0
    out["micro_f1"] = round(2 * mp * mr / (mp + mr + 1e-9), 4)
    return out


def call_api(doc_path: Path) -> dict | None:
    with open(doc_path, "rb") as fh:
        r = requests.post(
            f"{API_URL}/redact/image",
            files={"file": (doc_path.name, fh)},
            data={
                "owner_address":   "0xREALEVAL",
                "pii_filter":      "{}",
                "custom_terms":    "[]",
                "custom_patterns": "[]",
            },
            timeout=180,
        )
    r.raise_for_status()
    return r.json()


def run(test_dir, ann_dir, output_path, limit=None):
    test_dir = Path(test_dir)
    ann_dir  = Path(ann_dir)
    out      = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Verify backend is reachable
    try:
        requests.get(API_URL, timeout=3)
    except Exception:
        print(f"ERROR: backend not reachable at {API_URL}")
        print("Start: uvicorn backend_server:app --port 8000")
        sys.exit(1)

    # Load annotations
    gt_map = {}
    for f in ann_dir.glob("*.json"):
        if f.name == "schema.json":
            continue
        d = json.loads(f.read_text())
        gt_map[d.get("doc_id", f.stem)] = d

    # Collect image files
    files = sorted(
        f for f in test_dir.iterdir()
        if f.suffix.lower() in IMAGE_EXTS
    )
    if limit:
        files = files[:limit]

    lev_status = "enabled" if _LEVENSHTEIN_AVAILABLE else "disabled (pip install python-Levenshtein)"
    print(f"Real-document evaluation: {len(files)} images")
    print(f"Matching: text-value only (iou_thresh={IOU_THRESH})")
    print(f"Levenshtein fuzzy match: {lev_status}")
    print(f"Annotations: {len(gt_map)} loaded from {ann_dir}")
    print()

    agg = {c: {"tp": 0, "fp": 0, "fn": 0} for c in PII_CATEGORIES}
    results: list[dict] = []
    latencies: list[float] = []

    # Per-country aggregates
    country_agg: dict[str, dict] = {}

    for doc in files:
        ann = gt_map.get(doc.stem)
        if not ann:
            print(f"  [SKIP] no annotation: {doc.stem}")
            continue

        gts     = ann.get("pii_annotations", [])
        country = ann.get("country", "xx")

        print(f"  {doc.name} [{country}] ...", end=" ", flush=True)
        t0 = time.time()

        try:
            resp = call_api(doc)
            lat  = round(time.time() - t0, 2)
            latencies.append(lat)

            preds = resp.get("redactions", [])
            if not preds:
                preds = [
                    {"label": "Text", "value": v, "box": None}
                    for v in resp.get("redacted_items", [])
                ]

            c = match_text_only(preds, gts)
            for cat in PII_CATEGORIES:
                for k in ("tp", "fp", "fn"):
                    agg[cat][k] += c[cat][k]
                    country_agg.setdefault(country, {}).setdefault(
                        cat, {"tp": 0, "fp": 0, "fn": 0}
                    )[k] += c[cat][k]

            m = metrics(c)
            print(
                f"F1={m['macro_f1']:.3f}  "
                f"preds={len(preds)}  gt={len(gts)}  ({lat}s)"
            )

            results.append({
                "doc_id":        doc.stem,
                "country":       country,
                "latency_s":     lat,
                "agent_timings": resp.get("agent_timings", {}),
                "metrics":       m,
                "pred_count":    len(preds),
                "gt_count":      len(gts),
                "predictions":   preds,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"doc_id": doc.stem, "country": country,
                             "error": str(e)})

    # Overall metrics
    overall = metrics(agg)
    per_country = {cc: metrics(cats) for cc, cats in country_agg.items()}
    mean_lat = round(float(np.mean(latencies)), 2) if latencies else 0.0

    summary = {
        "evaluation_type": "real_specimen_text_only",
        "note":            "IoU=0 — bbox matching disabled, text value + Levenshtein(≤2) only",
        "dataset":         "real_specimens",
        "match_method":    "text_value_only (iou_thresh=0.0)",
        "levenshtein":     _LEVENSHTEIN_AVAILABLE,
        "n_docs":          len(results),
        "mean_latency_s":  mean_lat,
        "overall_metrics": overall,
        "per_country":     per_country,
        "per_doc":         results,
    }

    out.write_text(json.dumps(summary, indent=2))

    print(f"\nResults → {out}")
    print(f"Macro F1 : {overall['macro_f1']}")
    print(f"Micro F1 : {overall['micro_f1']}")
    print(f"Mean lat : {mean_lat}s/image")
    print()
    print("Per-category:")
    for cat in PII_CATEGORIES:
        m = overall.get(cat, {})
        if isinstance(m, dict) and (m.get("tp", 0) + m.get("fp", 0) + m.get("fn", 0)) > 0:
            print(
                f"  {cat:15s}  "
                f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})"
            )
    if per_country:
        print()
        print("Per-country macro F1:")
        for cc, cm in sorted(per_country.items()):
            print(f"  {cc}: {cm.get('macro_f1', 0):.3f}")

    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Omni-Shield real-document evaluation (text-value matching)")
    p.add_argument("--test-dir",
        default="../../datasets/real_samples/images",
        help="Directory with downloaded specimen images")
    p.add_argument("--annotations",
        default="../../datasets/real_samples/annotations",
        help="Directory with annotation JSON files")
    p.add_argument("--output",
        default="../../results/raw/eval_real.json",
        help="Where to write results JSON")
    p.add_argument("--limit", type=int, default=None,
        help="Process only first N images")
    args = p.parse_args()
    run(args.test_dir, args.annotations, args.output, args.limit)
