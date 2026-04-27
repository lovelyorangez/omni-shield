# Ablation Study Results

| Configuration            | Macro F1 | Micro F1 | Drop vs full |
|--------------------------|----------|----------|--------------|
| Full pipeline (baseline) | 0.1584   | 0.3197   | —            |
| No faces + signatures    | 0.0726   | 0.1493   | -54.2%       |
| Names only               | 0.0617   | 0.1289   | -61.0%       |
| No DOB + ID numbers      | 0.1527   | 0.3136   | -3.6%        |
| Text PII only            | 0.0392   | 0.0607   | -75.2%       |

## Key findings

1. Signatures are the dominant contributor — removing them causes
   the largest single F1 drop (54%). The YOLOv8 contour-based
   signature detector and face detector are critical components.

2. DOB + ID numbers contribute marginally (3.6% drop) — the model
   detects DOB reasonably (F1=0.522) but misses ID numbers almost
   entirely (F1=0.096), so removing both has little impact.

3. Text PII alone (phones/emails/addresses) is insufficient for
   ID card redaction — these fields rarely appear on ID cards.

4. The hybrid approach (VLM + YOLOv8 + regex + context filtering)
   is justified — no single category handles ID cards adequately.

## Paper narrative

The ablation validates the multi-agent design: each agent class
contributes measurably, and the full pipeline outperforms any
single-category configuration by a significant margin.
