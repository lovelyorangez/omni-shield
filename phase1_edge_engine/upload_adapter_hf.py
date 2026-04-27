"""
Upload QLoRA adapter weights to HuggingFace Hub.
Run once before paper submission.

USAGE:
  pip install huggingface_hub
  huggingface-cli login   # enter your HF token
  python upload_adapter_hf.py \\
    --repo-name YOUR_HF_USERNAME/omni-shield-qwen2vl-pii-adapter \\
    --adapter-path ./fine_tuned_vlm/final_adapter

This publishes the adapter so anyone can reproduce
the fine-tuning results by running:

  from peft import PeftModel
  from transformers import Qwen2VLForConditionalGeneration
  model = Qwen2VLForConditionalGeneration.from_pretrained(
      "Qwen/Qwen2-VL-2B-Instruct")
  model = PeftModel.from_pretrained(model,
      "YOUR_HF_USERNAME/omni-shield-qwen2vl-pii-adapter")
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

MODEL_CARD_TEMPLATE = """---
base_model: Qwen/Qwen2-VL-2B-Instruct
library_name: peft
tags:
  - pii-detection
  - document-redaction
  - qlora
  - vision-language-model
  - privacy
license: apache-2.0
---

# Omni-Shield QLoRA Adapter for PII Detection

Fine-tuned adapter for **Qwen2-VL-2B-Instruct** trained on synthetic
identity documents for PII detection and redaction.

## Training details
- Base model: Qwen/Qwen2-VL-2B-Instruct
- Method: QLoRA (rank-16 LoRA adapters, 4-bit NF4 quantization)
- Training data: 2,000 synthetic ID cards (6 countries: AU, CA, DE, IN, UK, US)
- Teacher labels: Claude API (knowledge distillation via annotate_with_claude.py)
- Hardware: NVIDIA RTX 5060 8GB VRAM
- Training time: ~47 minutes, 3 epochs
- Optimizer: adamw_bnb_8bit

## Supported document types
ID cards and driving licences from: Australia, Canada, Germany, India, UK, USA

## PII categories detected
`names`, `faces`, `signatures`, `phones`, `emails`, `dob`,
`id_numbers`, `addresses`, `org_names`, `dates`

## Usage

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
import torch

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(
    base_model,
    "{repo_name}"
)
model = model.merge_and_unload()
model.eval()
```

## Key finding: label consistency
Fine-tuning teaches the model to emit structured `LABEL | VALUE` pairs
consistently. The base model detects values correctly but frequently omits
field labels, causing downstream pipeline failures. Fine-tuning fixes this
without degrading value detection accuracy.

Example (German ID card):
| Field     | Base model output         | Fine-tuned output         |
|-----------|---------------------------|---------------------------|
| Name      | `Name \\| Leonie Dörr`    | `Name \\| Leonie Dörr`    |
| DOB       | `Date of Birth \\| 01.08.1991` | `Date of Birth \\| 01.08.1991` |
| ID Number | `[empty] \\| M898585166`  | `ID No. \\| M898585166`   |
| Signature | `Signature \\| [none]`    | `Signature \\| [none]`    |

## Citation
```bibtex
@article{{omni-shield-2025,
  title={{Omni-Shield: Local-First Multi-Modal PII Redaction
         with Cryptographic Audit Trail}},
  author={{[Anonymous for review]}},
  year={{2025}}
}}
```

## Paper
Submitted to PETS 2026. arXiv preprint forthcoming.
"""


def upload(repo_name: str, adapter_path: str) -> None:
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}")
        print("Train the adapter first with train_vlm.py")
        return

    adapter_files = list(adapter_path.iterdir())
    print(f"Adapter directory: {adapter_path}")
    print(f"Files to upload: {[f.name for f in adapter_files]}")

    api = HfApi()

    print(f"\nCreating repository: {repo_name}")
    try:
        create_repo(repo_name, repo_type="model", exist_ok=True, private=False)
        print(f"Repository ready: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Repo note: {e}")

    print(f"\nUploading adapter weights...")
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_name,
        repo_type="model",
        commit_message="Upload QLoRA adapter weights for Omni-Shield PII detection",
        ignore_patterns=["*.pyc", "__pycache__", "README.md"],
    )
    print("Weights uploaded.")

    print("Uploading model card...")
    card_text = MODEL_CARD_TEMPLATE.replace("{repo_name}", repo_name)
    card_path = adapter_path / "README.md"
    card_path.write_text(card_text)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="model",
        commit_message="Add model card",
    )

    print(f"\nDone! Adapter published at:")
    print(f"  https://huggingface.co/{repo_name}")
    print()
    print("Add to paper:")
    print(f'  "Adapter weights available at \\url{{https://huggingface.co/{repo_name}}}"')


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Upload QLoRA adapter to HuggingFace Hub for paper reproducibility")
    p.add_argument("--repo-name", required=True,
        help="HF repo ID, e.g. yourusername/omni-shield-qwen2vl-pii-adapter")
    p.add_argument("--adapter-path", default="./fine_tuned_vlm/final_adapter",
        help="Path to adapter directory (default: ./fine_tuned_vlm/final_adapter)")
    args = p.parse_args()
    upload(args.repo_name, args.adapter_path)
