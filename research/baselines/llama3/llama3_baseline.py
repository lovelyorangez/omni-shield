"""
Baseline: Llama 3 8B via Ollama
Prompts the LLM to identify PII from OCR-extracted text.
Represents the "naive LLM" approach on the same hardware.

REQUIRES:
  ollama running: ollama serve
  model pulled:   ollama pull llama3
  easyocr in venv (already installed in phase1_edge_engine/venv)

USAGE:
  python llama3_baseline.py \
    --test-dir ../../datasets/synthetic_test \
    --annotations ../../datasets/annotations \
    --output ../../baselines/results/llama3_results.json
"""

import argparse, json, time, re, sys
from pathlib import Path
import requests as http_req

sys.path.insert(0, str(
    Path(__file__).parent.parent.parent / "evaluation" / "scripts"))
from eval import match, metrics, PII_CATEGORIES

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

# ---------------------------------------------------------------------------
# OCR — initialised lazily and cached on the function object
# ---------------------------------------------------------------------------

def extract_text(image_path: Path) -> tuple[str, list]:
    """Return (joined text, raw easyocr results) for an image."""
    import easyocr
    if not hasattr(extract_text, "_reader"):
        print("  Loading EasyOCR...", flush=True)
        extract_text._reader = easyocr.Reader(
            ['en'], gpu=True, verbose=False)
    results = extract_text._reader.readtext(str(image_path))
    return " ".join([r[1] for r in results]), results

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a PII detection system.
Given OCR text from an ID card, identify all personally identifiable information.

Respond ONLY with a JSON array. Each item must have:
  "label": one of [Name, Date of Birth, ID Number,
                   Address, Phone, Email, Signature, Face]
  "value": the exact text found (empty string if not text)

Example:
[
  {"label": "Name", "value": "John Smith"},
  {"label": "Date of Birth", "value": "15/03/1987"},
  {"label": "ID Number", "value": "AB1234567"}
]

If no PII found, return [].
Return ONLY the JSON array, nothing else."""


def call_llama(ocr_text: str) -> list[dict]:
    prompt = f"{SYSTEM_PROMPT}\n\nOCR text:\n{ocr_text}"
    try:
        resp = http_req.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 512,
            }
        }, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # Llama sometimes wraps JSON in markdown fences
        m = re.search(r'\[.*?\]', raw, re.DOTALL)
        if m:
            return json.loads(m.group())
        return []
    except Exception as e:
        print(f"    Ollama error: {e}", flush=True)
        return []

# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

LABEL_TO_CAT = {
    "name": "names", "full name": "names",
    "surname": "names", "given name": "names",
    "date of birth": "dob", "dob": "dob",
    "birth date": "dob",
    "id number": "id_numbers", "id no": "id_numbers",
    "passport": "id_numbers", "license": "id_numbers",
    "address": "addresses", "home address": "addresses",
    "phone": "phones", "mobile": "phones",
    "email": "emails",
    "signature": "signatures",
    "face": "faces", "photo": "faces",
}


def normalize_label(label: str) -> str:
    s = label.lower().strip()
    for k, v in LABEL_TO_CAT.items():
        if k in s:
            return v
    return s

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(test_dir, ann_dir, output_path, limit=None):
    test_dir    = Path(test_dir)
    ann_dir     = Path(ann_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Verify Ollama is reachable before doing any OCR work
    try:
        http_req.get("http://localhost:11434", timeout=3)
    except Exception:
        print("ERROR: Ollama not running. Start with: ollama serve")
        sys.exit(1)

    # Load ground-truth annotations
    gt_map = {}
    for f in ann_dir.glob("*.json"):
        if f.name == "schema.json":
            continue
        d = json.loads(f.read_text())
        gt_map[d.get("doc_id", f.stem)] = d

    # Collect image files
    files = sorted([
        f for f in test_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])
    if limit:
        files = files[:limit]

    print(f"Running Llama 3 baseline on {len(files)} images...")
    print(f"Model : {MODEL} via Ollama")
    print(f"GT docs in annotations: {len(gt_map)}")

    agg = {c: {"tp": 0, "fp": 0, "fn": 0} for c in PII_CATEGORIES}
    results = []
    latencies = []

    for doc in files:
        ann = gt_map.get(doc.stem)
        if not ann:
            continue
        gts = ann.get("pii_annotations", [])

        print(f"  {doc.name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            # Step 1: OCR
            ocr_text, ocr_raw = extract_text(doc)

            # Step 2: Llama 3 PII detection
            llama_preds = call_llama(ocr_text)

            # Step 3: Normalise labels to our category taxonomy
            preds = []
            for item in llama_preds:
                cat = normalize_label(item.get("label", ""))
                preds.append({
                    "label":    item.get("label", ""),
                    "value":    item.get("value", ""),
                    "category": cat,
                    "box":      None,
                })

            lat = round(time.time() - t0, 2)
            latencies.append(lat)

            # Step 4: Match against GT — text value only, no bbox
            c = match(preds, gts, iou_thresh=0.0)
            for cat in PII_CATEGORIES:
                for k in ("tp", "fp", "fn"):
                    agg[cat][k] += c[cat][k]

            m = metrics(c)
            print(f"F1={m['macro_f1']:.3f}  "
                  f"preds={len(preds)}  gt={len(gts)}  ({lat}s)")

            results.append({
                "doc_id":             doc.stem,
                "latency_s":          lat,
                "ocr_word_count":     len(ocr_text.split()),
                "llama_predictions":  llama_preds,
                "metrics":            m,
                "pred_count":         len(preds),
                "gt_count":           len(gts),
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"doc_id": doc.stem, "error": str(e)})

    import numpy as np
    overall = metrics(agg)
    mean_lat = round(float(np.mean(latencies)), 2) if latencies else 0

    summary = {
        "system":          "Llama 3 8B (Ollama)",
        "model":           MODEL,
        "approach":        "OCR (EasyOCR) → LLM prompt → PII extraction",
        "n_docs":          len(results),
        "mean_latency_s":  mean_lat,
        "overall_metrics": overall,
        "per_doc":         results,
    }

    output_path.write_text(json.dumps(summary, indent=2))

    print(f"\nResults → {output_path}")
    print(f"Macro F1     : {overall['macro_f1']}")
    print(f"Micro F1     : {overall['micro_f1']}")
    print(f"Mean latency : {mean_lat}s/image")
    print()
    print("Per-category:")
    for cat in PII_CATEGORIES:
        m = overall.get(cat, {})
        if isinstance(m, dict) and (
                m.get("tp", 0) + m.get("fp", 0) + m.get("fn", 0)) > 0:
            print(f"  {cat:15s}  "
                  f"P={m['precision']:.3f}  "
                  f"R={m['recall']:.3f}  "
                  f"F1={m['f1']:.3f}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Llama 3 8B PII detection baseline")
    p.add_argument("--test-dir",
        default="../../datasets/synthetic_test",
        help="Directory containing .jpg test images")
    p.add_argument("--annotations",
        default="../../datasets/annotations",
        help="Directory containing GT annotation JSON files")
    p.add_argument("--output",
        default="../../baselines/results/llama3_results.json",
        help="Where to write results JSON")
    p.add_argument("--limit", type=int, default=None,
        help="Cap number of images (useful for smoke tests)")
    args = p.parse_args()
    run(args.test_dir, args.annotations, args.output, args.limit)
