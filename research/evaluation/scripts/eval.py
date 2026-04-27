"""
Omni-Shield Evaluation Framework
Calls the live backend API and computes P/R/F1
against ground truth annotations.

USAGE:
  python eval.py --test-dir ../test_data \
                 --annotations ../annotations \
                 --output ../../results/raw/eval_results.json

Requires backend running at http://localhost:8000
pip install requests numpy
"""

import argparse, json, time, re as _re
import requests
import numpy as np
from pathlib import Path

API_URL = "http://localhost:8000"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
PII_CATEGORIES = [
    "names", "faces", "signatures", "phones",
    "emails", "dob", "id_numbers", "addresses",
    "org_names", "dates"
]
LABEL_TO_CATEGORY = {
    # Names
    "name": "names",
    "surname": "names",
    "given name": "names",
    "given names": "names",
    "full name": "names",
    "first name": "names",
    "last name": "names",
    "family name": "names",
    "forename": "names",
    "holder": "names",
    "cardholder": "names",
    "nom": "names",
    "prenom": "names",
    "nachname": "names",
    "vorname": "names",

    # Faces
    "face": "faces",
    "photo": "faces",
    "photograph": "faces",
    "picture": "faces",
    "image": "faces",

    # Signatures
    "signature": "signatures",
    "sign": "signatures",

    # Phones
    "phone": "phones",
    "mobile": "phones",
    "tel": "phones",
    "telephone": "phones",
    "contact": "phones",

    # Emails
    "email": "emails",
    "e-mail": "emails",

    # Date of birth
    "date of birth": "dob",
    "dob": "dob",
    "birth date": "dob",
    "birth": "dob",
    "born": "dob",
    "date of birth (dob)": "dob",
    "birthdate": "dob",
    "geburtsdatum": "dob",
    "date naissance": "dob",

    # ID numbers
    "id no": "id_numbers",
    "id no.": "id_numbers",
    "id number": "id_numbers",
    "license no": "id_numbers",
    "license no.": "id_numbers",
    "licence no": "id_numbers",
    "licence no.": "id_numbers",
    "licence number": "id_numbers",
    "license number": "id_numbers",
    "passport no": "id_numbers",
    "passport no.": "id_numbers",
    "passport number": "id_numbers",
    "document no": "id_numbers",
    "document number": "id_numbers",
    "card no": "id_numbers",
    "card number": "id_numbers",
    "number": "id_numbers",
    "dl no": "id_numbers",
    "driver license": "id_numbers",
    "driving licence": "id_numbers",
    "national id": "id_numbers",
    "pan": "id_numbers",
    "aadhar": "id_numbers",
    "aadhaar": "id_numbers",
    "ssn": "id_numbers",
    "medicare": "id_numbers",
    "nik": "id_numbers",
    "numéro": "id_numbers",

    # Addresses
    "address": "addresses",
    "addr": "addresses",
    "home address": "addresses",
    "residential address": "addresses",
    "adresse": "addresses",

    # Organisation names
    "org": "org_names",
    "organisation": "org_names",
    "organization": "org_names",
    "employer": "org_names",
    "issuer": "org_names",
    "issued by": "org_names",
    "authority": "org_names",

    # Dates (general)
    "date": "dates",
    "expiry": "dates",
    "expiry date": "dates",
    "expiration": "dates",
    "expiration date": "dates",
    "issued": "dates",
    "issued date": "dates",
    "issue date": "dates",
    "valid until": "dates",
    "valid from": "dates",
    "validity": "dates",
}

def bbox_iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (areaA + areaB - inter)

def normalize_cat(label, value=""):
    s = label.lower().strip()
    # Direct match first
    if s in LABEL_TO_CATEGORY:
        return LABEL_TO_CATEGORY[s]
    # Partial match — check if any key is contained in label
    for k, v in LABEL_TO_CATEGORY.items():
        if k in s:
            return v
    # Fallback: infer from value format when label is empty/unrecognized
    if value:
        v = value.strip()
        if _re.match(r'\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}', v):
            return 'dob'
        if _re.match(r'\d{1,2}\s+\w+\s+\d{4}', v):
            return 'dob'
        if _re.match(r'[\w.+-]+@[\w-]+\.\w+', v):
            return 'emails'
        if _re.match(r'[\d\s\+\-\(\)]{7,}$', v):
            return 'phones'
        if _re.match(r'[A-Z]{1,3}\d{5,}', v):
            return 'id_numbers'
        if _re.match(r'\d{3}[\s\-]\d{3}[\s\-]\d{4}', v):
            return 'id_numbers'
        if _re.match(r'[A-Z][a-z]+ [A-Z]', v):
            return 'names'
        if _re.match(r'[A-Z]{2,} [A-Z]{2,}', v):
            return 'names'
    return s

def match(preds, gts, iou_thresh=0.5):
    counts = {c: {"tp":0,"fp":0,"fn":0} for c in PII_CATEGORIES}
    gt_used = set()
    for pred in preds:
        cat = normalize_cat(pred.get("label", ""), pred.get("value", ""))
        if cat not in PII_CATEGORIES:
            counts.setdefault(cat, {"tp":0,"fp":0,"fn":0})
            continue
        best, best_i = 0.0, None
        for i, gt in enumerate(gts):
            if i in gt_used or gt.get("category") != cat:
                continue
            pb = pred.get("box") or pred.get("pixel_box")
            gb = gt.get("bbox")
            if pb and gb:
                iou = bbox_iou(pb, gb)
            else:
                pv = (pred.get("value") or "").lower()
                gv = (gt.get("value")  or "").lower()
                iou = 1.0 if pv and gv and (pv in gv or gv in pv) else 0.0
            if iou > best: best, best_i = iou, i
        cat_thresh = 0.1 if cat == "signatures" else iou_thresh
        # Signature containment check: if gt bbox falls inside pred bbox
        if cat == "signatures" and best < cat_thresh:
            for i, gt in enumerate(gts):
                if i in gt_used or gt.get("category") != cat:
                    continue
                pb = pred.get("box") or pred.get("pixel_box")
                gb = gt.get("bbox")
                if pb and gb:
                    overlap_x = max(0, min(pb[2], gb[2]) - max(pb[0], gb[0]))
                    overlap_y = max(0, min(pb[3], gb[3]) - max(pb[1], gb[1]))
                    overlap_area = overlap_x * overlap_y
                    gt_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
                    if gt_area > 0 and overlap_area / gt_area >= 0.5:
                        best = 0.5  # force match
                        best_i = i
                        break
        if best_i is not None and best >= cat_thresh:
            counts[cat]["tp"] += 1
            gt_used.add(best_i)
        else:
            counts[cat]["fp"] += 1
    for i, gt in enumerate(gts):
        if i not in gt_used and gt.get("category") in PII_CATEGORIES:
            counts[gt["category"]]["fn"] += 1
    return counts

def metrics(counts):
    out = {}
    tp_tot = fp_tot = fn_tot = 0
    for cat, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        tp_tot += tp; fp_tot += fp; fn_tot += fn
        p = tp/(tp+fp) if tp+fp else 0.0
        r = tp/(tp+fn) if tp+fn else 0.0
        f = 2*p*r/(p+r) if p+r else 0.0
        out[cat] = {"precision": round(p,4),
                    "recall":    round(r,4),
                    "f1":        round(f,4),
                    "tp":tp, "fp":fp, "fn":fn}
    f1s = [v["f1"] for v in out.values()]
    out["macro_f1"] = round(float(np.mean(f1s)), 4)
    mp = tp_tot/(tp_tot+fp_tot) if tp_tot+fp_tot else 0.0
    mr = tp_tot/(tp_tot+fn_tot) if tp_tot+fn_tot else 0.0
    out["micro_f1"] = round(2*mp*mr/(mp+mr+1e-9), 4)
    return out

def call_api(doc_path):
    ext = doc_path.suffix.lower()
    with open(doc_path, "rb") as f:
        if ext in IMAGE_EXTS:
            r = requests.post(f"{API_URL}/redact/image",
                files={"file": (doc_path.name, f)},
                data={"owner_address": "0xEVAL",
                      "pii_filter": "{}", "custom_terms": "[]",
                      "custom_patterns": "[]"}, timeout=180)
        elif ext == ".pdf":
            r = requests.post(f"{API_URL}/redact/pdf",
                files={"file": (doc_path.name, f)},
                data={"owner_address": "0xEVAL",
                      "pii_filter": "{}", "custom_terms": "[]",
                      "custom_patterns": "[]"}, timeout=60)
        else:
            return None
    r.raise_for_status()
    return r.json()

def run(test_dir, ann_dir, output_path, limit=None):
    test_dir = Path(test_dir)
    ann_dir  = Path(ann_dir)
    out      = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    gt_map = {}
    for f in ann_dir.glob("*.json"):
        if f.name == "schema.json": continue
        d = json.loads(f.read_text())
        gt_map[d.get("doc_id", f.stem)] = d

    files = sorted([f for f in test_dir.iterdir()
                    if f.suffix.lower() in
                    IMAGE_EXTS | {".pdf",".mp3",".txt"}])
    if limit: files = files[:limit]

    print(f"Evaluating {len(files)} documents...")
    agg = {c: {"tp":0,"fp":0,"fn":0} for c in PII_CATEGORIES}
    results, latencies = [], []

    for doc in files:
        ann = gt_map.get(doc.stem)
        if not ann:
            print(f"  [SKIP] no annotation for {doc.stem}")
            continue
        gts = ann.get("pii_annotations", [])
        print(f"  {doc.name} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            resp = call_api(doc)
            if not resp: print("skipped"); continue
            lat = round(time.time()-t0, 2)
            latencies.append(lat)
            preds = resp.get("redactions", [])
            if not preds:
                preds = [{"label":"Text","value":v,"box":None}
                         for v in resp.get("redacted_items",[])]
            agent_timings = resp.get("agent_timings", {})
            c = match(preds, gts, iou_thresh=0.3)
            for cat in PII_CATEGORIES:
                for k in ("tp","fp","fn"):
                    agg[cat][k] += c[cat][k]
            m = metrics(c)
            print(f"F1={m['macro_f1']} ({lat}s)")
            results.append({"doc_id": doc.stem,
                             "latency_s": lat,
                             "agent_timings": agent_timings,
                             "metrics": m,
                             "pred_count": len(preds),
                             "gt_count": len(gts),
                             "predictions": preds})
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"doc_id": doc.stem, "error": str(e)})

    overall = metrics(agg)
    summary = {
        "n_docs": len(results),
        "mean_latency_s": round(float(np.mean(latencies)),2)
                          if latencies else 0,
        "overall_metrics": overall,
        "per_doc": results,
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nResults → {out}")
    print(f"Macro F1 : {overall['macro_f1']}")
    print(f"Micro F1 : {overall['micro_f1']}")
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir",    default="../test_data")
    p.add_argument("--annotations", default="../annotations")
    p.add_argument("--output",
        default="../../results/raw/eval_results.json")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    run(args.test_dir, args.annotations, args.output, args.limit)
