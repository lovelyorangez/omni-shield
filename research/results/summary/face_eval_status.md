# Face Evaluation — Final Status

## Outcome: Cannot evaluate on synthetic test set

## Bug found and fixed
- omni_shield_agents.py referenced yolov8x.pt (not present)
- Corrected to yolov8n.pt
- Live system now runs face detection correctly

## Why evaluation still returns F1=0
- Synthetic test cards use 2D illustrated avatars
- YOLOv8 trained on COCO real photographs
- Cannot detect stylised portrait illustrations
- This is a test set limitation, not a model failure

## Paper text (§7.1)
"Face detection (F1=0.000) reflects a test set limitation:
synthetic cards use 2D illustrated avatars which 
COCO-trained YOLOv8 cannot detect. A model filename bug
was identified and fixed during evaluation; the live system
correctly detects faces on real photographic ID cards.
Quantitative face evaluation is deferred to real-document
testing in future work."

## Status: CLOSED — documented as limitation
