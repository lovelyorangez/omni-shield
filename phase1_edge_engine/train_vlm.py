#!/usr/bin/env python3
"""
Omni-Shield VLM Fine-Tuning Script
====================================
Stage 3 of the knowledge distillation pipeline.

Fine-tunes Qwen2-VL-2B-Instruct on the annotated ID card dataset
produced by annotate_with_claude.py using QLoRA (4-bit base weights
with trainable low-rank adapter layers).

WHAT THIS SCRIPT DOES
─────────────────────
Knowledge distillation works by training a small "student" model
(Qwen2-VL-2B) to produce the same outputs as a large "teacher" model
(Claude claude-sonnet-4). The student sees the same images the teacher saw
and is penalised when its output doesn't match the teacher's annotation.
Over thousands of training steps the student learns to recognise PII
in identity documents the way the teacher does — including faces,
signatures, ID numbers, and field abbreviations.

WHY QLORA
──────────
Full fine-tuning of a 2B VLM would require ~16 GB VRAM. QLoRA
reduces this to ~5-6 GB by:
1. Loading the base model in 4-bit NF4 (same as inference)
2. Freezing all base model weights
3. Adding tiny trainable "adapter" matrices (rank-16) to the
   attention layers — these are the only weights that change
4. Training the adapters in bfloat16 while the base stays in 4-bit

After training, the adapter is saved as a small ~50-200 MB file.
At inference time, load_vlm() in omni_shield_agents.py can be
updated to merge the adapter into the base model automatically.

HARDWARE REQUIREMENTS
──────────────────────
Minimum: RTX 5060 8GB (your setup) — this script is tuned for exactly this
Batch size 1 with gradient accumulation 8 = effective batch 8
Training 1000 cards for 3 epochs takes approximately 4-6 hours.

Usage:
    python train_vlm.py \
        --train_file ./dataset/train.jsonl \
        --output_dir ./fine_tuned_vlm \
        --epochs 3 \
        --max_samples 1000
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)

try:
    from qwen_vl_utils import process_vision_info
    _QWEN_VL_UTILS = True
except ImportError:
    _QWEN_VL_UTILS = False
    print("WARNING: qwen_vl_utils not found. Install with: pip install qwen-vl-utils")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# All hyperparameters in one place — tuned for RTX 5060 8GB
# ══════════════════════════════════════════════════════════════════════════════
MODEL_ID         = "Qwen/Qwen2-VL-2B-Instruct"
LORA_RANK        = 16     # Higher rank = more capacity but more memory
LORA_ALPHA       = 32     # Alpha = 2 * rank is a good default
LORA_DROPOUT     = 0.05
# Target modules: the attention projection matrices are where
# VLMs learn to "look" at different parts of the image/text.
# We train all four (Q, K, V, output projection) for best results.
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE    = 2e-4   # QLoRA canonical learning rate
WARMUP_RATIO     = 0.05   # Warm up for 5% of steps to stabilise early training
WEIGHT_DECAY     = 0.01
MAX_SEQ_LENGTH   = 512    # Maximum token length per sample
MAX_PIXELS       = 200_000  # Same as inference — keeps VRAM predictable


# ══════════════════════════════════════════════════════════════════════════════
# DATASET PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
def load_dataset_from_jsonl(jsonl_path: str, max_samples: int | None = None) -> Dataset:
    """
    Load the annotated JSONL file produced by annotate_with_claude.py
    and convert it to a HuggingFace Dataset.

    Each sample in the JSONL has a "conversations" field with two turns:
    - Turn 0 (human): the instruction prompt
    - Turn 1 (gpt): the target JSON annotation (what we want the model to produce)

    We convert this into the format Qwen2-VL's processor expects:
    a list of message dicts with image + text content.
    """
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError:
                continue

    if max_samples:
        samples = samples[:max_samples]

    # Shuffle for training stability
    import random
    random.shuffle(samples)

    print(f"Loaded {len(samples)} training samples from {jsonl_path}")

    # Count by country for balance reporting
    from collections import Counter
    countries = Counter(s.get("country", "unknown") for s in samples)
    for country, count in sorted(countries.items()):
        print(f"  {country}: {count} samples")

    return Dataset.from_list(samples)


# ══════════════════════════════════════════════════════════════════════════════
# COLLATOR
# The collator is called by the Trainer for each batch. It takes raw
# samples from the dataset and converts them into model-ready tensors.
# ══════════════════════════════════════════════════════════════════════════════
class VLMCollator:
    """
    Converts raw training samples into batched model inputs.

    The key challenge for VLM training is that each sample has both
    an image and text, and the loss should be computed ONLY on the
    model's output tokens (the annotation JSON), not on the input
    tokens (the instruction prompt + image). We achieve this by
    setting input_ids positions corresponding to the instruction to -100
    (the standard HuggingFace "ignore this token in the loss" sentinel).
    """

    def __init__(self, processor, max_length: int = 2048):
        self.processor   = processor
        self.max_length  = max_length

    def __call__(self, batch: list[dict]) -> dict:
        input_ids_list   = []
        attention_mask_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        labels_list      = []

        for sample in batch:
            image_path    = sample["image"]
            conversations = sample["conversations"]

            # Build the message structure Qwen2-VL expects
            instruction = conversations[0]["value"]  # human turn
            target      = conversations[1]["value"]  # gpt turn (what we want to learn)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type":       "image",
                            "image":      image_path,
                            "max_pixels": MAX_PIXELS,
                        },
                        {
                            "type": "text",
                            "text": instruction,
                        },
                    ],
                },
                {
                    "role":    "assistant",
                    "content": target,
                },
            ]

            # Apply chat template to get the full token sequence
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Get image features
            if _QWEN_VL_UTILS:
                image_inputs, _ = process_vision_info(messages)
            else:
                image_inputs = [Image.open(image_path).convert("RGB")]

            # Tokenize
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids     = inputs.input_ids[0]
            attention_mask = inputs.attention_mask[0]

            # Create labels: copy input_ids but mask the instruction portion.
            # We find where the assistant's response starts by looking for the
            # assistant role token boundary in the token sequence.
            labels = input_ids.clone()

            # Find the assistant token boundary.
            # Qwen2-VL uses "<|im_start|>assistant\n" to mark the assistant turn.
            # Everything before this (the instruction + image tokens) should be -100.
            assistant_marker = self.processor.tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )
            marker_len = len(assistant_marker)

            # Find the marker in the token sequence
            mask_end = 0
            for i in range(len(input_ids) - marker_len):
                if input_ids[i:i+marker_len].tolist() == assistant_marker:
                    mask_end = i + marker_len
                    break

            # Mask instruction tokens — only compute loss on assistant output
            labels[:mask_end] = -100

            # Also mask padding tokens
            labels[attention_mask == 0] = -100

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

            if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
                # pixel_values is already [num_patches, feature_dim] — no batch dim.
                # Append the full tensor; we'll cat across samples later.
                pixel_values_list.append(inputs.pixel_values)
            if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
                # image_grid_thw is [1, 3] per image; [0] gives the [3] row.
                image_grid_thw_list.append(inputs.image_grid_thw[0])

        result = {
            "input_ids":      torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels":         torch.stack(labels_list),
        }
        if pixel_values_list:
            # Qwen2-VL vision encoder expects packed format: [total_patches, feature_dim].
            # torch.stack would add a batch dim → [B, patches, dim], making the attention
            # split see seq_len=1 instead of the actual patch count. cat along dim=0
            # gives the correct [total_patches, feature_dim] packed layout.
            result["pixel_values"]   = torch.cat(pixel_values_list, dim=0)
        if image_grid_thw_list:
            # image_grid_thw elements are [3] tensors (t, h, w); stack → [B, 3]
            # which is the shape the model uses to recompute cu_seqlens.
            result["image_grid_thw"] = torch.stack(image_grid_thw_list)

        return result


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_model_for_training():
    """
    Load Qwen2-VL-2B in 4-bit NF4 and wrap it with QLoRA adapters.

    The prepare_model_for_kbit_training() call is essential — it handles
    the tricky interaction between bitsandbytes 4-bit layers and the
    gradient checkpointing that keeps VRAM usage manageable during backprop.
    Without this call, gradients would flow incorrectly through quantized layers.
    """
    print(f"Loading base model: {MODEL_ID}")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # Essential for 4-bit training — prepares quantized layers for gradient flow.
    # use_gradient_checkpointing=False: Qwen2-VL's vision encoder uses dynamic
    # sequence splitting (torch.split on variable-length seqs) that corrupts
    # length metadata during checkpoint replay, causing a RuntimeError on the
    # first forward pass. We disable GC entirely and compensate with MAX_SEQ_LENGTH=1024.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    # Belt-and-suspenders: explicitly zero out GC flags on the visual encoder
    # in case any upstream call re-enables them.
    model.model.visual.gradient_checkpointing = False
    for layer in model.model.visual.blocks:
        layer.gradient_checkpointing = False

    # Add LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        # bias="none" is the standard choice — training biases adds
        # little benefit but doubles the adapter parameter count
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train(
    train_file: str,
    output_dir: str,
    epochs: int = 3,
    max_samples: int | None = None,
    eval_split: float = 0.1,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load dataset and split into train/eval
    full_dataset = load_dataset_from_jsonl(train_file, max_samples)

    split = full_dataset.train_test_split(test_size=eval_split, seed=42)
    train_dataset = split["train"]
    eval_dataset  = split["test"]

    print(f"\nTrain: {len(train_dataset)} samples")
    print(f"Eval:  {len(eval_dataset)} samples")

    # Load model
    model, processor = load_model_for_training()
    collator = VLMCollator(processor, max_length=MAX_SEQ_LENGTH)

    # Training arguments — tuned for RTX 5060 8GB
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,

        # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
        # = 1 * 4 = 4. Reduced from 8 to lower peak activation memory per step.
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        per_device_eval_batch_size=1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,  # Keep only the 3 best checkpoints

        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",   # Cosine decay is standard for fine-tuning

        # Mixed precision — bfloat16 is more stable than float16 for this model
        bf16=True,
        fp16=False,
        optim="adamw_bnb_8bit",   # 8-bit Adam halves optimizer state memory vs fp32 Adam

        # Gradient checkpointing trades compute for VRAM — essential on 8GB
        gradient_checkpointing=False,

        logging_dir=str(output_path / "logs"),
        logging_steps=10,
        report_to="none",   # Disable wandb/tensorboard by default

        # Load the best checkpoint at the end based on eval loss
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        dataloader_num_workers=0,   # 0 for stability with PIL images
        remove_unused_columns=False,  # Keep image_path and other non-tensor columns
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    print(f"\nStarting training for {epochs} epoch(s) ...")
    print(f"Estimated time: {len(train_dataset) * epochs // 60} minutes on RTX 5060")
    print(f"Output: {output_dir}\n")

    trainer.train()

    # Save the final LoRA adapter
    adapter_path = output_path / "final_adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))

    print(f"\nTraining complete.")
    print(f"LoRA adapter saved to: {adapter_path}")
    print(f"\nTo use the fine-tuned model in Omni-Shield, update OmniConfig:")
    print(f"  vlm_model_id = '{MODEL_ID}'")
    print(f"  vlm_adapter_path = '{adapter_path}'")
    print(f"\nThen update load_vlm() in omni_shield_agents.py to merge the adapter:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{adapter_path}')")
    print(f"  model = model.merge_and_unload()   # bake adapter into weights")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION HELPER
# Run after training to measure how well the fine-tuned model performs
# compared to the base model on a held-out set.
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(adapter_path: str, test_jsonl: str, num_samples: int = 20):
    """
    Quick qualitative evaluation: run the fine-tuned model on a sample
    of test images and print the predicted vs. expected annotations.
    """
    from peft import PeftModel

    print(f"Loading fine-tuned model from {adapter_path} ...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(adapter_path)

    samples = []
    with open(test_jsonl) as f:
        for line in f:
            try:
                samples.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    samples = samples[:num_samples]

    correct_labels = 0
    total_expected = 0
    total_predicted = 0

    for i, sample in enumerate(samples):
        image_path = sample["image"]
        expected   = json.loads(sample["conversations"][1]["value"])

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path, "max_pixels": MAX_PIXELS},
                {"type": "text",  "text": sample["conversations"][0]["value"]},
            ],
        }]

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if _QWEN_VL_UTILS:
            image_inputs, _ = process_vision_info(messages)
        else:
            image_inputs = [Image.open(image_path)]

        inputs = processor(
            text=[text_prompt], images=image_inputs,
            return_tensors="pt", padding=True,
        ).to(model.device)

        with torch.inference_mode():
            gen_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

        output = processor.batch_decode(
            [gen_ids[0][inputs.input_ids.shape[1]:]],
            skip_special_tokens=True
        )[0]

        try:
            predicted = json.loads(output)
        except json.JSONDecodeError:
            predicted = []

        expected_labels  = {item["label"].lower() for item in expected}
        predicted_labels = {item["label"].lower() for item in predicted}
        overlap = expected_labels & predicted_labels

        correct_labels  += len(overlap)
        total_expected  += len(expected)
        total_predicted += len(predicted)

        print(f"\nSample {i+1}: {sample.get('country', '?')} {sample.get('card_type', '?')}")
        print(f"  Expected:  {sorted(expected_labels)}")
        print(f"  Predicted: {sorted(predicted_labels)}")
        print(f"  Match:     {sorted(overlap)}")

    precision = correct_labels / total_predicted if total_predicted else 0
    recall    = correct_labels / total_expected  if total_expected  else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n{'='*50}")
    print(f"Evaluation results over {num_samples} samples:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2-VL on annotated ID card dataset using QLoRA"
    )
    subparsers = parser.add_subparsers(dest="command")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Run fine-tuning")
    train_parser.add_argument("--train_file",  required=True,
        help="Path to train.jsonl from annotate_with_claude.py")
    train_parser.add_argument("--output_dir",  required=True,
        help="Directory to save checkpoints and final adapter")
    train_parser.add_argument("--epochs",      type=int, default=3)
    train_parser.add_argument("--max_samples", type=int, default=None,
        help="Limit training to N samples (useful for quick tests)")
    train_parser.add_argument("--eval_split",  type=float, default=0.1,
        help="Fraction of data to hold out for evaluation (default 10%)")

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Evaluate a fine-tuned adapter")
    eval_parser.add_argument("--adapter_path", required=True,
        help="Path to the saved LoRA adapter directory")
    eval_parser.add_argument("--test_file",    required=True,
        help="JSONL file to evaluate on")
    eval_parser.add_argument("--num_samples",  type=int, default=20,
        help="Number of samples to evaluate")

    args = parser.parse_args()

    if args.command == "train":
        train(
            train_file=args.train_file,
            output_dir=args.output_dir,
            epochs=args.epochs,
            max_samples=args.max_samples,
            eval_split=args.eval_split,
        )
    elif args.command == "eval":
        evaluate_model(
            adapter_path=args.adapter_path,
            test_jsonl=args.test_file,
            num_samples=args.num_samples,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
