# Real Document Evaluation — Results

## Setup
- 19/20 real specimen images from Wikimedia Commons
- 1 error: real_xx_09.jpg (Slovak passport — VRAM)
- Annotated by Claude Code vision
- Text-value matching only (bbox=null)

## Results vs synthetic
| Category   | Synthetic F1 | Real F1 | Change  |
|------------|-------------|---------|---------|
| Faces      | 0.000*      | 0.905   | +0.905  |
| Signatures | 0.870       | 0.684   | -0.186  |
| DOB        | 0.522       | 0.583   | +0.061  |
| ID Numbers | 0.096       | 0.333   | +0.237  |
| Names      | 0.312       | 0.182   | -0.130  |
| Addresses  | 0.067       | 0.000   | —       |
| Macro F1   | 0.187       | 0.269   | +0.082  |

*Synthetic faces=0 was evaluation limitation, not model failure

## Key findings
1. Better on real than synthetic (0.269 > 0.187) — generalises well
2. Face detection confirmed working (F1=0.905, recall=1.0)
3. ID numbers: perfect precision (P=1.0) on real docs
4. Names: biggest transfer gap (diacritics, ALL CAPS, multi-word)
5. Signatures transfer with 21% relative drop

## Per-country (real docs)
ch/mt: 0.300 · gr: 0.283 · fr/lv: 0.267 · nl: 0.233
hr/lu/si: 0.200 · it: 0.152 · is: 0.100
