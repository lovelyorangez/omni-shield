# OMNI-SHIELD

AI-powered multimodal privacy protection system that detects and redacts PII from images, PDFs, and audio using a multi-agent pipeline, ZK-SNARK proofs, and MetaMask wallet-gated decryption.

## Architecture

```
omni-shield/
├── omni-shield-app/               # React + Electron desktop frontend
│   └── src/App.jsx                # Main UI: upload/redact, ZK verify, decrypt
├── phase1_edge_engine/            # Python ML backend
│   ├── backend_server.py          # FastAPI server (core API)
│   ├── omni_shield_agents.py      # Multi-agent pipeline (v3, primary inference)
│   ├── blockchain_manager.py      # Smart contract interaction
│   ├── security_engine.py         # ZK-SNARK + encryption layer
│   └── datasets/                  # Training corpora (legal NER, Kaggle PII)
├── docker-compose.yml             # Multi-container orchestration
├── Dockerfile.backend             # CUDA 12.4-based backend
└── Dockerfile.frontend            # Nginx-served frontend
```

## Multi-Agent Pipeline (`omni_shield_agents.py`)

The core inference pipeline is a sequential 8-agent graph. All processing is local — no data leaves the machine.

```
Input
  └─► RouterAgent          — classifies format (image/pdf/text) + doc subtype
        ├─► LayoutAgent    — VLM (Qwen2-VL-2B): extracts label-value pairs with boxes
        ├─► OCRAgent       — EasyOCR (CPU): word-level pixel bounding boxes
        └─► TextPIIAgent   — fine-tuned Qwen2.5-3B: flat PII string extraction
              └─► ContextAgent      — resolves ambiguity by reading the field label
                    └─► BBRefinerAgent  — snaps coarse VLM boxes to precise OCR boxes
                            └─► CriticAgent   — drops false positives via LLM review
                                    └─► RedactionAgent  — draws black boxes + audit JSON
```

### Agent Responsibilities

| Agent | Model | Role |
|---|---|---|
| `RouterAgent` | none | Filename/extension heuristic → `input_format` + `doc_type` |
| `LayoutAgent` | Qwen2-VL-2B (4-bit) | Structured layout parsing: emits `LABEL / VALUE / VALUE_BOX` triples |
| `OCRAgent` | EasyOCR (CPU) | Word-level pixel boxes; feeds BBRefiner and ContextAgent |
| `TextPIIAgent` | Qwen2.5-3B (4-bit) | Text-domain PII extraction from OCR-reconstructed text |
| `ContextAgent` | Qwen2.5-3B (4-bit) | **Key innovation**: classifies values using adjacent field label; heuristic-first, LLM fallback for ambiguous cases |
| `BBRefinerAgent` | none | Converts normalized VLM boxes → precise pixel boxes via IoU-snapping to OCR words |
| `CriticAgent` | Qwen2.5-3B (4-bit) | Final audit pass; drops clear false positives (skipped if < 3 detections) |
| `RedactionAgent` | none | Draws redactions, saves PNG + `.audit.json` sidecar for blockchain layer |

### Why ContextAgent Matters

The previous single-model VLM approach asked "is this text PII?" without surrounding context. A lone value like `99 999 999` is ambiguous — serial number, price, or national ID. `ContextAgent` reads the adjacent field label first:
- Label contains `"ID Number"` → value **is** PII
- Label contains `"Invoice Total"` → value **is not** PII

This eliminates the bulk of false positives and missed detections on structured documents. Stage 1 uses keyword heuristics (~80% of cases, ~0ms); Stage 2 falls back to the 3B LLM (`max_new_tokens=5`) for ambiguous labels.

### Shared State (`AgentState`)

Agents communicate through a single `AgentState` dataclass threaded through the pipeline:
- `label_value_pairs` — populated by LayoutAgent, consumed by ContextAgent
- `ocr_words` — populated by OCRAgent, consumed by BBRefiner + ContextAgent
- `raw_text_pii` — populated by TextPIIAgent, absorbed by ContextAgent
- `confirmed_pii` — set by ContextAgent, refined by BBRefiner
- `final_redactions` — set by BBRefiner, filtered by Critic, consumed by RedactionAgent

### Memory Modes (`OmniConfig.memory_mode`)

| Mode | Behavior | VRAM |
|---|---|---|
| `"lazy"` (default) | Load/unload each model between agent stages | Safer on 8 GB |
| `"dual"` | Both VLM + text model loaded simultaneously | ~3.2 GB weights, ~30s faster |

## Stack

**Frontend:** React 19 + Vite + Electron 39 + Tailwind CSS + ethers.js 6
**Backend:** FastAPI + PyTorch (4-bit BnB quantization) + Transformers
**Inference models:** Qwen2-VL-2B-Instruct (VLM), Qwen2.5-3B-Instruct (text, swappable with fine-tuned path)
**CV/Audio:** EasyOCR, YOLOv8, OpenCV, PyMuPDF, Faster-Whisper, PyAudio
**Crypto:** Fernet encryption, ZoKrates ZK-SNARK circuits (`circuit.zok`)
**Blockchain:** Solidity (`AuditLog.sol`), MetaMask wallet auth
**Infra:** Docker Compose + NVIDIA GPU runtime

## Key API Endpoints (`backend_server.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/redact/{type}` | POST | Run agent pipeline on image/pdf/audio → encrypted file + ZK proof |
| `/decrypt` | POST | Decrypt via MetaMask wallet verification |
| `/verify_zk` | POST | Audit ZK-SNARK proof |
| `/download/{file}` | GET | Secure file download |

## Models

- `qwen_3b_pii_qlora/` — Qwen2.5-3B QLora adapter (swap in via `OmniConfig.text_model_id`)
- `local_pii_rejector/` — Fine-tuned PII classification model
- `local_legal_ner_model/` — Legal entity recognition (trained on CUAD v1)
- `yolov8x.pt` — Person/face detection

## Dev Setup

### Ganache (required before starting the backend)

> **Ganache must be running before starting `backend_server.py` or calling
> any `blockchain_manager` function.**  The backend will print a clear error
> and exit if it cannot reach Ganache at `http://127.0.0.1:7545`.

```bash
npx ganache --port 7545 --deterministic
```

The `--deterministic` flag seeds Ganache with a fixed mnemonic so the same
10 wallet addresses are generated every run.  This keeps the `ownerRecords`
mapping consistent across restarts — records anchored in one session are
queryable in the next as long as the same contract address is in `ledger.json`.

### Backend
```bash
cd phase1_edge_engine
source venv/bin/activate       # or venv_nightly for cutting-edge torch
uvicorn backend_server:app --reload --port 8000
```

Run the agent pipeline directly:
```bash
python omni_shield_agents.py <image_path>
```

### Frontend
```bash
cd omni-shield-app
npm install
npm run dev          # Vite dev server
npm run electron     # Electron desktop app
```

### Docker (full stack)
```bash
docker compose up --build
```
Requires NVIDIA Docker runtime for GPU inference.

## GPU / Memory Notes

- Target: 8 GB VRAM with 4-bit NF4 quantization (BitsAndBytes, bfloat16 compute)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128` set by default
- `max_pixels=200_000` caps VLM input resolution to reduce KV-cache pressure
- If OOM: lower `OmniConfig.max_pixels`, reduce `max_new_tokens`, or use `memory_mode="lazy"`
- `venv_nightly/` exists for testing nightly PyTorch builds

## Training Scripts

Located in `phase1_edge_engine/`:
- `train_pii_rejector*.py` — versioned PII detector training iterations
- `train_legal_ner.py` — legal NER fine-tuning
- `train_kaggle_pii.py` — Kaggle PII dataset fine-tuning
- `cross_domain_eval.py` / `error_analysis_diagnostics.py` — evaluation

## Blockchain / ZK

- `circuit.zok` — ZoKrates circuit definition for ZK-SNARK generation
- `blockchain_manager.py` — deploys/calls `AuditLog.sol` via Ganache at `http://127.0.0.1:7545`
- Contract address is cached in `ledger.json` after first deploy; subsequent runs reuse it
- `RedactionAgent` writes `.audit.json` sidecars consumed by the blockchain layer
- Wallet auth flow: MetaMask signs a challenge → backend verifies → decrypts file

## File Conventions

- Redacted outputs: `phase1_edge_engine/redacted_output/` (agent pipeline) or `processed_files/` (legacy)
- Audit JSON sidecars: `redacted_output/redacted_<stem>.audit.json`
- Active learning logs: `active_learning_logs.json`
- Blockchain ledger: `ledger.json`
