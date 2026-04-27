"""
Confusion Matrix for PII Category Detection
Reads eval_full.json + per-doc annotations and builds a confusion
matrix showing which PII categories get mis-detected as which others.

Rows  = predicted category
Cols  = true (ground-truth) category
Diag  = correct detections (TP)
Off-d = mis-classifications
Last  = FP (prediction matched no GT box)
"FN"  = GT items that were never matched (shown in summary only)

USAGE (from this directory):
  python confusion_matrix.py

Outputs:
  confusion_matrix.json  — raw counts
  confusion_matrix.png   — heatmap figure
  confusion_matrix.tex   — LaTeX table
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── categories ──────────────────────────────────────────────────────────────

PII_CATEGORIES = [
    "names", "dob", "id_numbers", "addresses",
    "signatures", "faces", "phones", "emails",
    "org_names", "dates",
]

# Maps prediction label strings → our taxonomy.
# Predictions carry no pre-computed category field — label is all we have.
LABEL_TO_CAT = {
    # names
    "name":             "names",
    "names":            "names",
    "surname":          "names",
    "given name":       "names",
    "given name(s)":    "names",
    "given surname":    "names",
    "full name":        "names",
    "first name":       "names",
    "last name":        "names",
    "vorname":          "names",        # German "given name"
    "father's name":    "names",
    "mother's name":    "names",
    # dob
    "date of birth":    "dob",
    "dob":              "dob",
    "birth date":       "dob",
    "born":             "dob",
    # id numbers
    "id no":            "id_numbers",
    "id no.":           "id_numbers",
    "idnumber":         "id_numbers",
    "id number":        "id_numbers",
    "licence no":       "id_numbers",
    "licence no.":      "id_numbers",
    "license no":       "id_numbers",
    "license no.":      "id_numbers",
    "driver licence number":  "id_numbers",
    "driver's licence number": "id_numbers",
    "passport":         "id_numbers",
    "passport no":      "id_numbers",
    "card no":          "id_numbers",
    "longdigit":        "id_numbers",   # catch-all digit string from LayoutAgent
    "unique identification authority of india": "id_numbers",
    # addresses
    "address":          "addresses",
    "home address":     "addresses",
    "postcode":         "addresses",
    "postal code":      "addresses",
    # signatures
    "signature":        "signatures",
    "sign":             "signatures",
    # faces
    "face":             "faces",
    "photo":            "faces",
    "photograph":       "faces",
    # phones
    "phone":            "phones",
    "mobile":           "phones",
    "telephone":        "phones",
    # emails
    "email":            "emails",
    "e-mail":           "emails",
    # org names
    "org":              "org_names",
    "organisation":     "org_names",
    "organization":     "org_names",
    # dates (general)
    "date":             "dates",
    "expiry":           "dates",
    "expiry date":      "dates",
    "issue date":       "dates",
    "valid until":      "dates",
}


def normalize_cat(label: str, value: str = "") -> str:
    """Map a prediction label string to a PII category."""
    s = label.lower().strip()
    if s in LABEL_TO_CAT:
        return LABEL_TO_CAT[s]
    for k, v in LABEL_TO_CAT.items():
        if k in s:
            return v
    # Value-pattern fallback
    if value:
        v = value.strip()
        if re.match(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}", v):
            return "dob"
        if re.match(r"\d{1,2}\s+\w+\s+\d{4}", v):
            return "dob"
        if re.match(r"[\w.+-]+@[\w-]+\.\w+", v):
            return "emails"
        if re.match(r"[A-Z]{1,3}\d{5,}", v):
            return "id_numbers"
        if re.match(r"[A-Z][a-z]+ [A-Z]", v):
            return "names"
    return "other"


# ── IoU ──────────────────────────────────────────────────────────────────────

def bbox_iou(a, b) -> float:
    """Intersection-over-union for two [x1,y1,x2,y2] boxes."""
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (a[2] - a[0]) * (a[3] - a[1])
    aB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aA + aB - inter)


def containment(inner, outer) -> float:
    """Fraction of inner box area that overlaps outer box."""
    xA, yA = max(inner[0], outer[0]), max(inner[1], outer[1])
    xB, yB = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area  = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / area if area > 0 else 0.0


# ── matrix builder ───────────────────────────────────────────────────────────

def build_matrix(eval_path: Path, ann_dir: Path) -> dict:
    """
    Returns confusion[pred_cat][true_cat] = count.
    Special keys: "unmatched" (FP column) and "fn" (FN row).
    """
    data = json.loads(eval_path.read_text())

    # Load GT annotations indexed by doc_id
    gt_map = {}
    for f in ann_dir.glob("*.json"):
        if f.name == "schema.json":
            continue
        ann = json.loads(f.read_text())
        gt_map[ann.get("doc_id", f.stem)] = ann

    confusion = defaultdict(lambda: defaultdict(int))
    total_matched = total_fp = total_fn = 0

    for doc in data["per_doc"]:
        if "error" in doc:
            continue
        ann = gt_map.get(doc["doc_id"])
        if not ann:
            continue

        gts   = ann.get("pii_annotations", [])
        preds = doc.get("predictions", [])

        gt_matched   = set()
        pred_matched = set()

        for pi, pred in enumerate(preds):
            pred_cat = normalize_cat(
                pred.get("label", ""),
                pred.get("value", "") or "",
            )
            pb = pred.get("pixel_box")

            best_iou  = -1.0
            best_gi   = None
            best_gt_cat = None

            for gi, gt in enumerate(gts):
                if gi in gt_matched:
                    continue
                gt_cat = gt.get("category", "")
                gb = gt.get("bbox")

                if pb and gb:
                    iou = bbox_iou(pb, gb)

                    # Signatures often span the full card width —
                    # accept if the GT box is ≥50 % contained in pred
                    if pred_cat == "signatures" and iou < 0.1:
                        if containment(gb, pb) >= 0.5:
                            iou = 0.5

                    if iou > best_iou:
                        best_iou    = iou
                        best_gi     = gi
                        best_gt_cat = gt_cat

            # IoU threshold: signatures are more lenient
            thresh = 0.1 if pred_cat == "signatures" else 0.3

            if best_gi is not None and best_iou >= thresh:
                confusion[pred_cat][best_gt_cat] += 1
                gt_matched.add(best_gi)
                pred_matched.add(pi)
                total_matched += 1
            else:
                confusion[pred_cat]["unmatched"] += 1
                total_fp += 1

        # FN — GT items never matched
        for gi, gt in enumerate(gts):
            if gi not in gt_matched:
                gt_cat = gt.get("category", "")
                confusion["fn"][gt_cat] += 1
                total_fn += 1

    print(f"  Matched (TP-ish): {total_matched}")
    print(f"  Unmatched preds (FP): {total_fp}")
    print(f"  Missed GT items (FN): {total_fn}")
    return confusion


# ── output: JSON ─────────────────────────────────────────────────────────────

def save_json(confusion: dict, out_path: Path):
    out = {k: dict(v) for k, v in confusion.items()}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"JSON  → {out_path}")


# ── output: LaTeX ─────────────────────────────────────────────────────────────

SHORT = {
    "names":      "Names",
    "dob":        "DOB",
    "id_numbers": "ID No.",
    "addresses":  "Addr.",
    "signatures": "Sig.",
    "faces":      "Faces",
    "phones":     "Phone",
    "emails":     "Email",
    "org_names":  "Org",
    "dates":      "Dates",
    "other":      "Other",
}


def save_latex(confusion: dict, out_path: Path):
    # Only include categories that have any activity
    active = [c for c in PII_CATEGORIES
              if any(confusion[c].values()) or
                 any(confusion[p].get(c, 0) for p in PII_CATEGORIES)]

    lines = [
        r"\begin{table}[h]\centering",
        r"\caption{PII Category Confusion Matrix (predicted $\times$ true)}",
        r"\label{tab:confusion}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "r" * len(active) + "r}",
        r"\toprule",
    ]

    # Header row
    hdr = r"Pred $\backslash$ True"
    for c in active:
        hdr += " & " + SHORT.get(c, c)
    hdr += r" & FP \\"
    lines.append(hdr)
    lines.append(r"\midrule")

    for pred in active:
        row = SHORT.get(pred, pred)
        for true in active:
            val = confusion[pred].get(true, 0)
            if val == 0:
                row += " & --"
            elif pred == true:
                row += r" & \textbf{" + str(val) + "}"
            else:
                row += " & " + str(val)
        fp = confusion[pred].get("unmatched", 0)
        row += f" & {fp}" + r" \\"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines))
    print(f"LaTeX → {out_path}")


# ── output: PNG ──────────────────────────────────────────────────────────────

def save_png(confusion: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Only rows/cols with data
    active = [c for c in PII_CATEGORIES
              if any(confusion[c].values()) or
                 any(confusion[p].get(c, 0) for p in PII_CATEGORIES)]
    n = len(active)

    matrix = np.zeros((n, n), dtype=int)
    for i, pred in enumerate(active):
        for j, true in enumerate(active):
            matrix[i][j] = confusion[pred].get(true, 0)

    labels = [SHORT.get(c, c) for c in active]

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto",
                   vmin=0, vmax=max(matrix.max(), 1))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("True category", fontsize=11, labelpad=8)
    ax.set_ylabel("Predicted category", fontsize=11, labelpad=8)
    ax.set_title(
        "PII Category Confusion Matrix\n(Omni-Shield, 150 ID cards)",
        fontsize=12, pad=14,
    )

    # Cell annotations
    threshold = matrix.max() * 0.6
    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            if val > 0:
                color = "white" if val > threshold else "black"
                weight = "bold" if i == j else "normal"
                ax.text(j, i, str(val),
                        ha="center", va="center",
                        fontsize=9, color=color, fontweight=weight)

    cbar = plt.colorbar(im, ax=ax, label="Count", shrink=0.85)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PNG   → {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    HERE     = Path(__file__).parent
    EVAL     = HERE / "../raw/eval_full.json"
    ANN_DIR  = HERE / "../../datasets/annotations"

    print("Building confusion matrix from eval_full.json ...")
    confusion = build_matrix(EVAL.resolve(), ANN_DIR.resolve())

    save_json( confusion, HERE / "confusion_matrix.json")
    save_latex(confusion, HERE / "confusion_matrix.tex")
    save_png(  confusion, HERE / "confusion_matrix.png")

    # ── readable summary ──────────────────────────────────────────────────────
    print()
    print(f"{'Predicted':<18}  {'True':<18}  {'Count':>5}  Note")
    print("-" * 58)
    for pred in sorted(confusion):
        if pred == "fn":
            continue
        for true, count in sorted(confusion[pred].items(),
                                   key=lambda x: -x[1]):
            if count == 0:
                continue
            if true == "unmatched":
                note = "(FP — no GT match)"
            elif pred == true:
                note = "✓ correct"
            else:
                note = "✗ mis-classified"
            print(f"  {pred:<16}  {true:<18}  {count:>5}  {note}")

    print()
    print("FN (missed GT items):")
    for true, count in sorted(confusion["fn"].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {'(no prediction)':<16}  {true:<18}  {count:>5}  (FN)")
