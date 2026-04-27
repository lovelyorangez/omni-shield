"""
Baseline: Microsoft Presidio
Run on same test documents for comparison table.

INSTALL (in a separate venv to avoid conflicts):
  cd ~/omni-shield/research
  python -m venv venv_research
  source venv_research/bin/activate
  pip install presidio-analyzer spacy requests numpy easyocr
  python -m spacy download en_core_web_lg

USAGE:
  python presidio_baseline.py \
    --test-dir ../../evaluation/test_data \
    --annotations ../../evaluation/annotations \
    --output ../../baselines/results/presidio_results.json
"""

import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                        "evaluation" / "scripts"))
from eval import match, metrics, PII_CATEGORIES

ENTITY_TO_CAT = {
    "PERSON": "names", "EMAIL_ADDRESS": "emails",
    "PHONE_NUMBER": "phones", "DATE_TIME": "dob",
    "LOCATION": "addresses", "ORGANIZATION": "org_names",
    "NRP": "org_names", "IN_PAN": "id_numbers",
    "US_SSN": "id_numbers", "UK_NHS": "id_numbers",
    "CREDIT_CARD": "id_numbers", "IBAN_CODE": "id_numbers",
}

def run(test_dir, ann_dir, output_path):
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError:
        print("Install: pip install presidio-analyzer spacy")
        print("Then:    python -m spacy download en_core_web_lg")
        return

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en",
                    "model_name": "en_core_web_lg"}]})
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(),
        supported_languages=["en"])

    try:
        import easyocr
        ocr_reader = easyocr.Reader(['en'], gpu=True)
    except ImportError:
        print("Install: pip install easyocr")
        return

    test_dir = Path(test_dir)
    ann_dir  = Path(ann_dir)
    out      = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    gt_map = {}
    for f in ann_dir.glob("*.json"):
        if f.name == "schema.json": continue
        d = json.loads(f.read_text())
        gt_map[d.get("doc_id", f.stem)] = d

    agg = {c: {"tp":0,"fp":0,"fn":0} for c in PII_CATEGORIES}
    results = []

    for doc in sorted(test_dir.glob("*.jpg")):
        ann = gt_map.get(doc.stem)
        if not ann: continue
        gts  = ann.get("pii_annotations", [])
        ocr_results = ocr_reader.readtext(str(doc))
        text = " ".join([r[1] for r in ocr_results])
        hits = analyzer.analyze(text=text, language="en")
        preds = []
        for h in hits:
            cat = ENTITY_TO_CAT.get(h.entity_type)
            if cat:
                preds.append({"label": h.entity_type,
                              "value": text[h.start:h.end],
                              "category": cat, "box": None})
        c = match(preds, gts, iou_thresh=0.0)
        for cat in PII_CATEGORIES:
            for k in ("tp","fp","fn"):
                agg[cat][k] += c[cat][k]
        m = metrics(c)
        print(f"  {doc.name}: F1={m['macro_f1']} (OCR words: {len(ocr_results)})")
        results.append({"doc_id": doc.stem, "metrics": m,
                         "pred_count": len(preds),
                         "gt_count": len(gts)})

    overall = metrics(agg)
    summary = {"system": "Microsoft Presidio",
               "model": "en_core_web_lg",
               "n_docs": len(results),
               "overall_metrics": overall,
               "per_doc": results}
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nPresidio baseline → {out}")
    print(f"Macro F1: {overall['macro_f1']}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir",
        default="../../evaluation/test_data")
    p.add_argument("--annotations",
        default="../../evaluation/annotations")
    p.add_argument("--output",
        default="../../baselines/results/presidio_results.json")
    args = p.parse_args()
    run(args.test_dir, args.annotations, args.output)
