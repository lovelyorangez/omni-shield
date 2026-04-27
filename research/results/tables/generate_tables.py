"""
Generate LaTeX and CSV comparison tables for the paper.

USAGE:
  python generate_tables.py \
    --omni    ../raw/eval_results.json \
    --presidio ../../baselines/results/presidio_results.json \
    --llama3  ../../baselines/results/llama3_results.json \
    --ablation ../raw/ablation.json \
    --output  ./
"""

import argparse, json, csv
from pathlib import Path

PII_CATEGORIES = [
    "names","faces","signatures","phones","emails",
    "dob","id_numbers","addresses","org_names","dates"
]
LABELS = {
    "names":"Names","faces":"Faces",
    "signatures":"Signatures","phones":"Phone Numbers",
    "emails":"Email Addresses","dob":"Date of Birth",
    "id_numbers":"ID Numbers","addresses":"Addresses",
    "org_names":"Org Names","dates":"Dates (General)"
}

def load(p):
    f = Path(p)
    return json.loads(f.read_text()) if f.exists() else None

def latex_comparison(systems, out_dir):
    cols = "l" + "ccc"*len(systems)
    rows = [r"\begin{table}[h]\centering",
            r"\caption{PII Detection — Comparison with Baselines}",
            r"\label{tab:comparison}",
            f"\\begin{{tabular}}{{{cols}}}",
            r"\toprule"]
    sh = " & ".join(
        r"\multicolumn{3}{c}{"+n+"}" for n in systems)
    rows.append("Category & " + sh + r" \\")
    rows.append(" & " + " & ".join(["P & R & F1"]*len(systems)) + r" \\")
    rows.append(r"\midrule")
    for cat in PII_CATEGORIES:
        row = [LABELS[cat]]
        for data in systems.values():
            m = (data or {}).get("overall_metrics",{}).get(cat,{})
            row.append(f"{m.get('precision',0):.2f} & "
                       f"{m.get('recall',0):.2f} & "
                       f"{m.get('f1',0):.2f}")
        rows.append(" & ".join(row) + r" \\")
    rows.append(r"\midrule")
    mrow = [r"\textbf{Macro F1}"]
    for data in systems.values():
        f1 = (data or {}).get("overall_metrics",{}).get("macro_f1",0)
        mrow.append(f"-- & -- & \\textbf{{{f1:.4f}}}")
    rows.append(" & ".join(mrow) + r" \\")
    rows += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    p = out_dir/"comparison_table.tex"
    p.write_text("\n".join(rows))
    print(f"LaTeX → {p}")

def csv_comparison(systems, out_dir):
    p = out_dir/"comparison_table.csv"
    with open(p,"w",newline="") as f:
        w = csv.writer(f)
        h = ["Category"]
        for n in systems: h += [f"{n} P",f"{n} R",f"{n} F1"]
        w.writerow(h)
        for cat in PII_CATEGORIES:
            row = [LABELS[cat]]
            for data in systems.values():
                m = (data or {}).get("overall_metrics",{}).get(cat,{})
                row += [round(m.get("precision",0),4),
                        round(m.get("recall",0),4),
                        round(m.get("f1",0),4)]
            w.writerow(row)
        mrow = ["Macro F1"]
        for data in systems.values():
            f1 = (data or {}).get("overall_metrics",{}).get("macro_f1",0)
            mrow += ["","",round(f1,4)]
        w.writerow(mrow)
    print(f"CSV → {p}")

def latex_ablation(ablation, out_dir):
    if not ablation: return
    rows = [r"\begin{table}[h]\centering",
            r"\caption{Agent Ablation Study — Macro F1}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcc}",r"\toprule",
            r"Configuration & Macro F1 & Micro F1 \\",r"\midrule"]
    for cfg in ablation:
        desc = cfg.get("description","?")
        m    = cfg.get("metrics",{})
        f1   = m.get("macro_f1",0)
        mf1  = m.get("micro_f1",0)
        if "full" in cfg.get("config","").lower():
            rows.append(
                rf"\textbf{{{desc}}} & \textbf{{{f1:.4f}}} & \textbf{{{mf1:.4f}}} \\")
        else:
            rows.append(f"{desc} & {f1:.4f} & {mf1:.4f} \\\\")
    rows += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    p = out_dir/"ablation_table.tex"
    p.write_text("\n".join(rows))
    print(f"Ablation LaTeX → {p}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--omni",     default="../raw/eval_results.json")
    p.add_argument("--presidio", default="../../baselines/results/presidio_results.json")
    p.add_argument("--llama3",   default="../../baselines/results/llama3_results.json")
    p.add_argument("--ablation", default="../raw/ablation.json")
    p.add_argument("--output",   default="./")
    args = p.parse_args()
    out = Path(args.output); out.mkdir(exist_ok=True)
    systems = {"Omni-Shield": load(args.omni),
               "Presidio":    load(args.presidio),
               "Llama 3 8B":  load(args.llama3)}
    systems = {k:v for k,v in systems.items() if v}
    if systems:
        latex_comparison(systems, out)
        csv_comparison(systems, out)
    latex_ablation(load(args.ablation), out)
    print("Done.")
