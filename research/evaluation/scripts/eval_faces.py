"""
Face Detection Evaluation
Evaluates YOLOv8 face detection separately from text PII.

Faces come from a separate YOLO model inside RedactionAgent and are NOT
included in the text-pipeline predictions used by eval.py.  This script
calls the live /redact/image endpoint with all categories disabled except
faces, reads the dedicated face_boxes field from the response, and matches
predicted boxes against ground-truth face bboxes using IoU.

USAGE:
  python eval_faces.py \
    --test-dir  ../../datasets/synthetic_test \
    --annotations ../../datasets/annotations \
    --output    ../../results/raw/eval_faces.json

  # Smoke test (5 images):
  python eval_faces.py --limit 5 --output /tmp/eval_faces_smoke.json
"""

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import requests

API_URL    = "http://localhost:8000"
IOU_THRESH = 0.3

# Only enable face detection — disable all text-PII categories so the
# pipeline finishes faster and doesn't touch the GPU models unnecessarily.
FACE_ONLY_FILTER = json.dumps({
    "faces":      True,
    "names":      False,
    "signatures": False,
    "phones":     False,
    "emails":     False,
    "dob":        False,
    "id_numbers": False,
    "addresses":  False,
    "org_names":  False,
    "dates":      False,
})


# ── IoU ──────────────────────────────────────────────────────────────────────

def bbox_iou(a, b) -> float:
    """Intersection-over-union for two [x1,y1,x2,y2] boxes."""
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (a[2] - a[0]) * (a[3] - a[1])
    aB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aA + aB - inter)


# ── main ─────────────────────────────────────────────────────────────────────

def eval_faces(test_dir, ann_dir, output_path, limit=None):
    test_dir = Path(test_dir)
    ann_dir  = Path(ann_dir)
    out      = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Verify backend is reachable before doing any work
    try:
        requests.get(f"{API_URL}/health", timeout=3)
    except Exception:
        try:
            requests.get(API_URL, timeout=3)
        except Exception:
            print(f"ERROR: backend not reachable at {API_URL}")
            print("Start it with: uvicorn backend_server:app --port 8000")
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

    print(f"Face evaluation on {len(files)} images  (IoU≥{IOU_THRESH})")
    print(f"Backend: {API_URL}")
    print(f"GT docs : {len(gt_map)}")

    tp = fp = fn = 0
    latencies = []
    results   = []

    for doc in files:
        ann = gt_map.get(doc.stem)
        if not ann:
            continue

        # Ground-truth face boxes for this document
        gt_faces = [
            a["bbox"] for a in ann.get("pii_annotations", [])
            if a.get("category") == "faces" and a.get("present", True)
        ]

        print(f"  {doc.name} ...", end=" ", flush=True)
        t0 = time.time()

        try:
            with open(doc, "rb") as fh:
                resp = requests.post(
                    f"{API_URL}/redact/image",
                    files={"file": (doc.name, fh, "image/jpeg")},
                    data={
                        "owner_address":   "0xFACEEVAL",
                        "pii_filter":      FACE_ONLY_FILTER,
                        "custom_terms":    "[]",
                        "custom_patterns": "[]",
                    },
                    timeout=180,
                )
            resp.raise_for_status()
            lat = round(time.time() - t0, 2)
            latencies.append(lat)
            data = resp.json()

            # face_boxes is the dedicated field added to the /redact/image response.
            # Fall back to scanning redactions if the backend is an older version.
            pred_faces = data.get("face_boxes") or [
                r.get("pixel_box") or r.get("box")
                for r in data.get("redactions", [])
                if r.get("label", "").lower() in ("face", "photo")
                and (r.get("pixel_box") or r.get("box"))
            ]

            # Match each predicted box to the best available GT box
            doc_tp = doc_fp = 0
            gt_matched: set[int] = set()

            for pred_box in pred_faces:
                if not pred_box:
                    continue
                best_iou = 0.0
                best_i   = None
                for i, gt_box in enumerate(gt_faces):
                    if i in gt_matched:
                        continue
                    iou = bbox_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_i   = i
                if best_i is not None and best_iou >= IOU_THRESH:
                    doc_tp += 1
                    gt_matched.add(best_i)
                else:
                    doc_fp += 1

            doc_fn = len(gt_faces) - len(gt_matched)

            tp += doc_tp
            fp += doc_fp
            fn += doc_fn

            p_  = doc_tp / (doc_tp + doc_fp) if (doc_tp + doc_fp) else 0.0
            r_  = doc_tp / (doc_tp + doc_fn) if (doc_tp + doc_fn) else 0.0
            f1  = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0

            print(
                f"pred={len(pred_faces)} gt={len(gt_faces)} "
                f"tp={doc_tp} fp={doc_fp} fn={doc_fn} "
                f"F1={f1:.3f} ({lat}s)"
            )

            results.append({
                "doc_id":      doc.stem,
                "pred_faces":  len(pred_faces),
                "gt_faces":    len(gt_faces),
                "tp":  doc_tp,
                "fp":  doc_fp,
                "fn":  doc_fn,
                "precision":   round(p_, 4),
                "recall":      round(r_, 4),
                "f1":          round(f1, 4),
                "latency_s":   lat,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"doc_id": doc.stem, "error": str(e)})

    # Overall metrics
    prec   = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1     = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    mean_lat = round(float(np.mean(latencies)), 2) if latencies else 0.0

    summary = {
        "n_docs":          len(results),
        "mean_latency_s":  mean_lat,
        "iou_threshold":   IOU_THRESH,
        "face_metrics": {
            "precision": round(prec, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "per_doc": results,
    }

    out.write_text(json.dumps(summary, indent=2))

    print(f"\nResults → {out}")
    print(f"Face  P={prec:.3f}  R={recall:.3f}  F1={f1:.3f}")
    print(f"      TP={tp}  FP={fp}  FN={fn}")
    print(f"Mean latency: {mean_lat}s/image")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate YOLOv8 face detection against GT annotations")
    p.add_argument("--test-dir",
        default="../../datasets/synthetic_test",
        help="Directory containing .jpg test images")
    p.add_argument("--annotations",
        default="../../datasets/annotations",
        help="Directory containing GT annotation JSON files")
    p.add_argument("--output",
        default="../../results/raw/eval_faces.json",
        help="Where to write results JSON")
    p.add_argument("--limit", type=int, default=None,
        help="Process only first N images (smoke test)")
    args = p.parse_args()
    eval_faces(args.test_dir, args.annotations, args.output, args.limit)
