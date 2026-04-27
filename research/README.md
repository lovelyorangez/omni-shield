# Omni-Shield — Research Folder

Completely isolated from the live system.
Nothing here touches phase1_edge_engine/ or omni-shield-app/.

## Structure
research/
├── evaluation/          # Evaluation framework
│   ├── scripts/         # eval.py, metrics.py
│   ├── test_data/       # Held-out test documents
│   ├── annotations/     # Ground truth PII labels (JSON)
│   └── metrics/         # Output metrics
├── baselines/           # Comparison systems
│   ├── presidio/        # Microsoft Presidio
│   ├── spacy/           # spaCy NER
│   └── results/         # Baseline results
├── ablation/            # Agent ablation studies
│   ├── scripts/         # ablation_runner.py
│   └── results/         # F1 per configuration
├── datasets/            # Test datasets
│   ├── synthetic_test/  # Generated test ID cards
│   ├── real_samples/    # Anonymised real samples
│   └── annotations/     # Ground truth labels
├── results/             # Final numbers for paper
│   ├── tables/          # CSV/LaTeX tables
│   ├── plots/           # matplotlib charts
│   └── raw/             # Raw JSON outputs
├── figures/             # Paper diagrams
├── literature/          # Related work notes
└── paper/               # LaTeX source

## Ground rules
1. Never import from phase1_edge_engine/
2. Call the live backend via HTTP (http://localhost:8000)
3. All scripts are standalone with explicit inputs/outputs
4. Backend must be running during evaluation:
   cd ~/omni-shield/phase1_edge_engine
   source venv/bin/activate
   uvicorn backend_server:app --reload --port 8000

## Paper target
PETS 2026 — Privacy Enhancing Technologies Symposium
arXiv preprint first.
