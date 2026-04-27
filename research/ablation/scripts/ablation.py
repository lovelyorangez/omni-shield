"""
Omni-Shield Agent Ablation Study
Disables agents one at a time via pii_filter and dry_run
to measure each agent's contribution to F1.

USAGE:
  python ablation.py --test-dir ../../evaluation/test_data \
                     --annotations ../../evaluation/annotations \
                     --output ../../results/raw/ablation.json

Requires backend running at http://localhost:8000
"""

import argparse, json, requests, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                        "evaluation" / "scripts"))
from eval import match, metrics, PII_CATEGORIES, IMAGE_EXTS

API_URL = "http://localhost:8000"

ALL_ON  = {c: True  for c in PII_CATEGORIES}
ALL_OFF = {c: False for c in PII_CATEGORIES}

CONFIGS = [
    {"name": "full_pipeline",
     "desc": "All agents (baseline)",
     "filter": ALL_ON},
    {"name": "no_faces_signatures",
     "desc": "Faces + signatures disabled",
     "filter": {**ALL_ON, "faces": False, "signatures": False}},
    {"name": "names_only",
     "desc": "Names only",
     "filter": {**ALL_OFF, "names": True}},
    {"name": "no_dob_ids",
     "desc": "DOB + ID numbers disabled",
     "filter": {**ALL_ON, "dob": False, "id_numbers": False}},
    {"name": "text_pii_only",
     "desc": "Phones + emails + addresses only",
     "filter": {**ALL_OFF, "phones": True,
                "emails": True, "addresses": True}},
]

def run_config(cfg, files, gt_map):
    agg = {c: {"tp":0,"fp":0,"fn":0} for c in PII_CATEGORIES}
    for doc in files:
        ann = gt_map.get(doc.stem)
        if not ann: continue
        gts = ann.get("pii_annotations", [])
        try:
            with open(doc, "rb") as f:
                r = requests.post(
                    f"{API_URL}/redact/image",
                    files={"file": (doc.name, f)},
                    data={"owner_address": "0xABLATION",
                          "pii_filter": json.dumps(cfg["filter"]),
                          "custom_terms": "[]",
                          "custom_patterns": "[]"},
                    params={"dry_run": "true"},
                    timeout=180)
            preds = r.json().get("redactions",
                    r.json().get("would_redact", []))
            c = match(preds, gts)
            for cat in PII_CATEGORIES:
                for k in ("tp","fp","fn"):
                    agg[cat][k] += c[cat][k]
        except Exception as e:
            print(f"    {doc.name}: {e}")
    return metrics(agg)

def run(test_dir, ann_dir, output_path):
    test_dir = Path(test_dir)
    ann_dir  = Path(ann_dir)
    out      = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    gt_map = {}
    for f in ann_dir.glob("*.json"):
        if f.name == "schema.json": continue
        d = json.loads(f.read_text())
        gt_map[d.get("doc_id", f.stem)] = d

    files = [f for f in sorted(test_dir.iterdir())
             if f.suffix.lower() in IMAGE_EXTS]

    all_results = []
    for cfg in CONFIGS:
        print(f"\n{cfg['name']}: {cfg['desc']}")
        m = run_config(cfg, files, gt_map)
        print(f"  Macro F1={m['macro_f1']}  Micro F1={m['micro_f1']}")
        all_results.append({"config": cfg["name"],
                             "description": cfg["desc"],
                             "metrics": m})

    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nAblation → {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir",
        default="../../evaluation/test_data")
    p.add_argument("--annotations",
        default="../../evaluation/annotations")
    p.add_argument("--output",
        default="../../results/raw/ablation.json")
    args = p.parse_args()
    run(args.test_dir, args.annotations, args.output)
