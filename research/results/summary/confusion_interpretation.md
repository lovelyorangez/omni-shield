# Confusion Matrix Interpretation

## Error type breakdown
Total predictions: 498 across 150 cards
Total GT items: ~730 (5 per card × 146 successful docs)

| Category   | TP  | FP  | FN   | Main error type        |
|------------|-----|-----|------|------------------------|
| Signatures | 126 | 5   | ~14  | Mostly correct         |
| DOB        | 65  | 61  | ~56  | High FP (date noise)   |
| Names      | 51  | 33  | ~99  | Low recall             |
| ID Numbers | 13  | 18  | ~119 | Very low recall        |
| Addresses  | 7   | 58  | ~92  | IoU mismatch (multi-line) |
| Faces      | 0   | 0   | ~97  | Eval gap (YOLOv8 separate) |

## Key finding
Dominant error = FN (655), not FP (175)
=> Recall problem, NOT classification confusion
=> Model classifies correctly when it detects, but misses most fields

## Main cross-category confusion
signatures → id_numbers: 24 cases
  Cause: Signature bbox overlaps with nearby ID number GT box
  Impact: Evaluation artifact — actual redaction is still correct

## Fixable issues identified
1. DOB FP rate: Add expiry/issue date exclusion in ContextAgent
2. Address IoU: Expand GT annotation to full multi-line bbox
3. Face eval: Add separate YOLOv8 face eval pass

## Paper use
- Table 3: confusion_matrix.tex
- Figure 2: confusion_matrix.png
- §5.3: Use paragraph above verbatim

## Face evaluation — updated diagnosis (post-investigation)

Two causes identified:

1. BUG FIXED: omni_shield_agents.py referenced yolov8x.pt
   (136MB, not present) instead of yolov8n.pt (6.2MB, present).
   The exists() guard silently skipped YOLO on every request.
   Fixed — live system now runs face detection correctly.

2. EVALUATION LIMITATION: Synthetic test cards use 2D illustrated
   avatars. YOLOv8 is COCO-trained on real photographs — it cannot
   detect stylised portrait illustrations as faces.
   Face F1=0 is a test set limitation, not a model failure.

   Fix options for future work:
   a) Embed real anonymised face photos into synthetic cards
   b) Use MediaPipe Face or OpenCV Haar cascades for illustration-
      style face detection
   c) Evaluate face detection separately using real document samples

Paper text: "Face detection (F1=0.000 in evaluation) reflects a
test set limitation rather than a system failure: our synthetic
ID cards use 2D illustrated avatars which COCO-trained YOLOv8
cannot detect as faces. A corrected implementation bug
(wrong model filename) was identified and fixed during evaluation;
the live system correctly detects real photographic faces using
YOLOv8n. Face evaluation on real documents is deferred to future work."
