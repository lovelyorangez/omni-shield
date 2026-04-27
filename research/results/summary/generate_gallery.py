"""
Failure/success gallery generator for Omni-Shield eval results.

Reads eval_full.json, selects the 10 lowest-F1 and 5 highest-F1 docs,
copies the originals and annotates GT bboxes in green.

NOTE: eval_full.json stores only aggregate metrics per document — not raw
predictions.  Predicted bboxes are therefore unavailable from this file.
The script draws GT bboxes only.  To get predicted bboxes, re-run eval.py
with a modified call_api() that also saves the raw response to a sidecar
JSON (see comment at end of this file).

Usage:
    cd ~/omni-shield/research/results/summary
    python generate_gallery.py [--eval ../../raw/eval_full.json]
                               [--images ../../../datasets/synthetic_test]
                               [--annotations ../../../datasets/annotations]
"""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── colour palette ─────────────────────────────────────────────────────────────
CAT_COLOURS = {
    "names":      (0,   200,   0),   # green
    "faces":      (0,   180, 255),   # cyan
    "signatures": (255, 140,   0),   # orange
    "dob":        (180,   0, 255),   # purple
    "id_numbers": (255,   0,   0),   # red
    "addresses":  (255, 215,   0),   # gold
    "org_names":  (0,   255, 200),   # teal
    "dates":      (200, 200, 200),   # grey
}
DEFAULT_COLOUR = (0, 255, 0)
LABEL_BG = (0, 0, 0, 180)   # semi-transparent black for label background


def _load_font(size=11):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_gt_boxes(img: Image.Image, annotations: list) -> Image.Image:
    """Draw GT bboxes on a copy of img.  Returns the annotated copy."""
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font = _load_font(11)

    for ann in annotations:
        cat   = ann.get("category", "")
        label = ann.get("label", cat)
        bbox  = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox
        colour = CAT_COLOURS.get(cat, DEFAULT_COLOUR)
        # Filled semi-transparent rect
        d.rectangle([x1, y1, x2, y2], fill=(*colour, 50), outline=(*colour, 230), width=2)
        # Label chip
        tw, th = d.textlength(label, font=font), 13
        d.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1], fill=LABEL_BG)
        d.text((x1 + 2, y1 - th - 1), label, fill=(255, 255, 255, 255), font=font)

    out = Image.alpha_composite(out, overlay)
    return out.convert("RGB")


def draw_pred_boxes(img: Image.Image, predictions: list) -> Image.Image:
    """Draw predicted bboxes (red) on a copy of img."""
    out = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    font = _load_font(11)
    colour = (255, 60, 60)

    for pred in predictions:
        bbox = pred.get("pixel_box") or pred.get("box")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox
        label = pred.get("label", "")
        d.rectangle([x1, y1, x2, y2], fill=(*colour, 50),
                    outline=(*colour, 230), width=2)
        tw = d.textlength(label, font=font)
        d.rectangle([x1, y1 - 15, x1 + tw + 4, y1], fill=(0, 0, 0, 180))
        d.text((x1 + 2, y1 - 14), label, fill=(255, 200, 200, 255), font=font)

    out = Image.alpha_composite(out, overlay)
    return out.convert("RGB")


def make_compare(original: Image.Image, gt_img: Image.Image,
                 pred_img: Image.Image, doc_id: str, f1: float) -> Image.Image:
    """Side-by-side: original | GT boxes (green) | predicted boxes (red)."""
    W, H = original.size
    gap = 8
    banner = 28
    canvas = Image.new("RGB", (W * 3 + gap * 2, H + banner), (30, 30, 30))

    head = ImageDraw.Draw(canvas)
    font = _load_font(13)
    head.text((4, 6), f"{doc_id}   macro_F1={f1:.3f}", fill=(220, 220, 220), font=font)

    def paste(im, col):
        canvas.paste(im, (col * (W + gap), banner))

    paste(original, 0)
    paste(gt_img,   1)
    paste(pred_img, 2)

    ld = ImageDraw.Draw(canvas)
    for i, lbl in enumerate(["Original", "GT boxes (green)", "Predictions (red)"]):
        ld.text((i * (W + gap) + 4, 4), lbl, fill=(180, 220, 255), font=font)

    return canvas


def process_case(doc_id: str, f1: float, pred_count: int, gt_count: int,
                 predictions: list, img_dir: Path, ann_dir: Path,
                 out_dir: Path) -> dict:
    img_path = img_dir / f"{doc_id}.jpg"
    ann_path = ann_dir / f"{doc_id}.json"

    if not img_path.exists():
        print(f"  [WARN] image not found: {img_path}")
        return {}
    if not ann_path.exists():
        print(f"  [WARN] annotation not found: {ann_path}")
        return {}

    original = Image.open(img_path).convert("RGB")
    ann_data = json.loads(ann_path.read_text())
    annotations = ann_data.get("pii_annotations", [])

    # Original copy
    shutil.copy(img_path, out_dir / f"{doc_id}.jpg")

    # GT annotated (green)
    gt_img = draw_gt_boxes(original, annotations)
    gt_img.save(out_dir / f"{doc_id}_gt.jpg", quality=90)

    # Predicted boxes (red)
    pred_img = draw_pred_boxes(original, predictions)
    pred_img.save(out_dir / f"{doc_id}_pred.jpg", quality=90)

    # Side-by-side comparison
    compare = make_compare(original, gt_img, pred_img, doc_id, f1)
    compare.save(out_dir / f"{doc_id}_compare.jpg", quality=90)

    return {
        "doc_id":     doc_id,
        "macro_f1":   f1,
        "pred_count": pred_count,
        "gt_count":   gt_count,
        "files": {
            "original": f"{doc_id}.jpg",
            "gt":       f"{doc_id}_gt.jpg",
            "pred":     f"{doc_id}_pred.jpg",
            "compare":  f"{doc_id}_compare.jpg",
        },
    }


def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",        default=str(here / "../raw/eval_full.json"))
    parser.add_argument("--images",      default=str(here / "../../datasets/synthetic_test"))
    parser.add_argument("--annotations", default=str(here / "../../datasets/annotations"))
    parser.add_argument("--n_failures",  type=int, default=10)
    parser.add_argument("--n_successes", type=int, default=5)
    args = parser.parse_args()

    eval_path = Path(args.eval)
    img_dir   = Path(args.images)
    ann_dir   = Path(args.annotations)
    fail_dir  = here / "gallery" / "failures"
    succ_dir  = here / "gallery" / "successes"
    fail_dir.mkdir(parents=True, exist_ok=True)
    succ_dir.mkdir(parents=True, exist_ok=True)

    data     = json.loads(eval_path.read_text())
    per_doc  = data.get("per_doc", [])

    # Build lookup: doc_id → predictions list
    pred_map = {p["doc_id"]: p.get("predictions", [])
                for p in per_doc if "metrics" in p}

    # Only scored docs (skip error entries)
    scored = [
        (p["doc_id"], p["metrics"]["macro_f1"], p.get("pred_count", 0), p.get("gt_count", 0))
        for p in per_doc if "metrics" in p
    ]
    scored.sort(key=lambda x: x[1])

    failures  = scored[:args.n_failures]
    successes = scored[-args.n_successes:]

    gallery_index = {"failures": [], "successes": []}

    print(f"Processing {len(failures)} failure cases ...")
    for doc_id, f1, pred_c, gt_c in failures:
        print(f"  {doc_id}  F1={f1:.3f}")
        entry = process_case(doc_id, f1, pred_c, gt_c,
                             pred_map.get(doc_id, []), img_dir, ann_dir, fail_dir)
        if entry:
            gallery_index["failures"].append(entry)

    print(f"\nProcessing {len(successes)} success cases ...")
    for doc_id, f1, pred_c, gt_c in successes:
        print(f"  {doc_id}  F1={f1:.3f}")
        entry = process_case(doc_id, f1, pred_c, gt_c,
                             pred_map.get(doc_id, []), img_dir, ann_dir, succ_dir)
        if entry:
            gallery_index["successes"].append(entry)

    idx_path = here / "gallery_index.json"
    idx_path.write_text(json.dumps(gallery_index, indent=2))
    print(f"\nGallery index → {idx_path}")
    print(f"Failures  → {fail_dir}  ({len(gallery_index['failures'])} cases)")
    print(f"Successes → {succ_dir}  ({len(gallery_index['successes'])} cases)")


if __name__ == "__main__":
    main()
