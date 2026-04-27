"""
Pre vs Post Fine-tuning Comparison
===================================
Runs eval on a subset of the synthetic test cards in two modes:
  1. Fine-tuned adapter   (OMNI_SKIP_ADAPTER=0, default)
  2. Base model, no adapter (OMNI_SKIP_ADAPTER=1)

Quantifies the knowledge-distillation contribution of QLoRA fine-tuning.

USAGE:
  python eval_base_vs_finetuned.py \\
    --test-dir  ../../datasets/synthetic_test \\
    --annotations ../../datasets/annotations \\
    --output    ../../results/raw/eval_base_vs_ft.json \\
    --limit 30

NOTE:
  The script runs eval against the live backend twice.
  Between runs it prompts you to restart the backend with/without the flag:

    # Fine-tuned (run 1):
    cd ~/omni-shield/phase1_edge_engine && source venv/bin/activate
    uvicorn backend_server:app --port 8000

    # Base model (run 2):
    OMNI_SKIP_ADAPTER=1 uvicorn backend_server:app --port 8000

  Both runs use the same --limit cards (first N in sorted order) so
  the comparison is on an identical sample.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# Allow importing eval.run from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from eval import run as _run_eval

API_URL = "http://localhost:8000"
PII_CATEGORIES = [
    "names", "faces", "signatures", "phones",
    "emails", "dob", "id_numbers", "addresses",
    "org_names", "dates",
]


def _wait_for_backend(timeout: int = 120) -> bool:
    """Poll until backend responds or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(API_URL, timeout=3)
            return True
        except Exception:
            time.sleep(3)
    return False


def _run_phase(label: str, test_dir, ann_dir, limit, tmp_path: Path) -> dict:
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")

    print("Waiting for backend…", end=" ", flush=True)
    if not _wait_for_backend():
        print("TIMEOUT — backend did not respond within 120s")
        sys.exit(1)
    print("ready.\n")

    summary = _run_eval(test_dir, ann_dir, str(tmp_path), limit)

    overall = summary["overall_metrics"]
    per_cat = {
        cat: overall[cat]
        for cat in PII_CATEGORIES
        if cat in overall and isinstance(overall[cat], dict)
    }
    return {
        "macro_f1":      overall["macro_f1"],
        "micro_f1":      overall["micro_f1"],
        "mean_latency_s": summary["mean_latency_s"],
        "n_docs":        summary["n_docs"],
        "per_category":  per_cat,
    }


def main(test_dir, ann_dir, output, limit):
    print("=" * 55)
    print("  Pre vs Post Fine-tuning Comparison")
    print(f"  Cards per run : {limit}")
    print(f"  Backend       : {API_URL}")
    print("=" * 55)

    # ── Run 1: Fine-tuned adapter ──────────────────────────────────────────────
    print("""
RUN 1  Fine-tuned adapter  (OMNI_SKIP_ADAPTER=0)
─────────────────────────────────────────────────
Start (or confirm) the backend WITHOUT the skip flag:

  cd ~/omni-shield/phase1_edge_engine
  source venv/bin/activate
  uvicorn backend_server:app --port 8000

Wait for "Application startup complete." then press Enter.
""")
    input("Press Enter when fine-tuned backend is ready › ")

    ft = _run_phase(
        "RUN 1 — Fine-tuned Qwen2-VL-2B + QLoRA adapter",
        test_dir, ann_dir, limit,
        Path("/tmp/eval_finetuned_tmp.json"),
    )
    print(f"\n  Fine-tuned Macro F1 : {ft['macro_f1']:.4f}")
    print(f"  Fine-tuned Micro F1 : {ft['micro_f1']:.4f}")

    # ── Run 2: Base model ──────────────────────────────────────────────────────
    print("""
RUN 2  Base model  (OMNI_SKIP_ADAPTER=1)
──────────────────────────────────────────
STOP the current backend (Ctrl+C in its terminal), then start:

  OMNI_SKIP_ADAPTER=1 uvicorn backend_server:app --port 8000

You should see the log line:
  "[BASE MODEL] Adapter skipped — running base Qwen2-VL-2B-Instruct"

Wait for "Application startup complete." then press Enter.
""")
    input("Press Enter when base model backend is ready › ")

    base = _run_phase(
        "RUN 2 — Base Qwen2-VL-2B-Instruct (no adapter)",
        test_dir, ann_dir, limit,
        Path("/tmp/eval_base_tmp.json"),
    )
    print(f"\n  Base model Macro F1 : {base['macro_f1']:.4f}")
    print(f"  Base model Micro F1 : {base['micro_f1']:.4f}")

    # ── Comparison ─────────────────────────────────────────────────────────────
    ft_f1   = ft["macro_f1"]
    base_f1 = base["macro_f1"]
    abs_gain = round(ft_f1 - base_f1, 4)
    rel_gain = round((ft_f1 - base_f1) / max(base_f1, 1e-6) * 100, 1)

    comparison = {
        "base_macro_f1":      base_f1,
        "finetuned_macro_f1": ft_f1,
        "absolute_gain":      abs_gain,
        "relative_gain_pct":  rel_gain,
        "n_cards":            limit,
        "conclusion": (
            f"Fine-tuning improves Macro F1 by "
            f"{abs_gain:+.4f} ({rel_gain:+.1f}% relative) "
            f"over base Qwen2-VL-2B-Instruct on {limit} synthetic cards"
        ),
    }

    print(f"\n{'='*55}")
    print("  Per-category comparison:")
    print(f"  {'Category':15s} {'Base F1':>8s} {'FT F1':>8s} {'Gain':>8s}")
    print("  " + "─" * 44)
    for cat in PII_CATEGORIES:
        bf = base["per_category"].get(cat, {}).get("f1", 0.0)
        ff = ft["per_category"].get(cat, {}).get("f1", 0.0)
        gain = ff - bf
        marker = " ↑" if gain > 0.02 else (" ↓" if gain < -0.02 else "  ")
        print(f"  {cat:15s} {bf:>8.3f} {ff:>8.3f} {gain:>+8.3f}{marker}")
    print("  " + "─" * 44)
    print(f"  {'Macro F1':15s} {base_f1:>8.4f} {ft_f1:>8.4f} {abs_gain:>+8.4f}")
    print(f"\n  {comparison['conclusion']}")
    print()

    # ── Save JSON ──────────────────────────────────────────────────────────────
    results = {
        "evaluation_type": "base_vs_finetuned_comparison",
        "finetuned":   ft,
        "base":        base,
        "comparison":  comparison,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Results → {out}")

    # ── LaTeX table ────────────────────────────────────────────────────────────
    rows = []
    for cat in PII_CATEGORIES:
        bf = base["per_category"].get(cat, {}).get("f1", 0.0)
        ff = ft["per_category"].get(cat, {}).get("f1", 0.0)
        gain = ff - bf
        ft_cell = f"\\textbf{{{ff:.3f}}}" if gain > 0.02 else f"{ff:.3f}"
        rows.append(
            f"  {cat.replace('_', ' ').title():20s} & {bf:.3f} & {ft_cell} & {gain:+.3f} \\\\"
        )

    latex = (
        "% Knowledge distillation contribution — generated by eval_base_vs_finetuned.py\n"
        "\\begin{table}[h]\\centering\n"
        "\\caption{Effect of QLoRA Fine-tuning on PII Detection (Macro F1)}\n"
        "\\label{tab:finetune}\n"
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Category & Base Model & Fine-tuned & Gain \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\midrule\n"
        f"  \\textbf{{Macro F1}} & {base_f1:.4f} & \\textbf{{{ft_f1:.4f}}} & {abs_gain:+.4f} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )

    latex_path = out.parent.parent / "tables" / "finetune_comparison.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(latex)
    print(f"LaTeX  → {latex_path}")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Pre vs post fine-tuning comparison for Omni-Shield VLM")
    p.add_argument("--test-dir",
        default="../../datasets/synthetic_test",
        help="Directory of test images")
    p.add_argument("--annotations",
        default="../../datasets/annotations",
        help="Directory of ground truth annotation JSON files")
    p.add_argument("--output",
        default="../../results/raw/eval_base_vs_ft.json",
        help="Where to write the comparison JSON")
    p.add_argument("--limit", type=int, default=30,
        help="Cards per run — 30 gives stable F1 estimates (default: 30)")
    args = p.parse_args()
    main(args.test_dir, args.annotations, args.output, args.limit)
