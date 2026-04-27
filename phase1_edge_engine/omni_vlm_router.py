#!/usr/bin/env python3
"""
Omni-Shield VLM Router: Optimized for 8GB VRAM
- Vision tower offloaded to CPU
- Image resolution limited
- 4‑bit quantisation with fallback to 8‑bit
"""

import os
import gc
import torch
import re
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import transformers.modeling_utils

# ==========================================
# MEMORY OPTIMISATION SETTINGS
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
# Disable the unnecessary CUDA warmup that eats VRAM
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

print(f"🚀 Booting Multimodal Engine ({MODEL_ID})...")

# Clean GPU memory before loading
gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

# ==========================================
# QUANTISATION CONFIG (start with 4‑bit)
# ==========================================
quant_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Fallback to 8‑bit if 4‑bit fails (use as second attempt)
quant_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  # Offload some layers to CPU
    bnb_8bit_compute_dtype=torch.bfloat16,
)

# ==========================================
# CUSTOM DEVICE MAP: Force vision tower to CPU
# ==========================================
# We define a device map that sends the vision tower modules to CPU
# and keeps the language model (quantised) on GPU.
# This pattern is described in the Hugging Face docs for multi‑modal models.
device_map = {
    "model.visual": "cpu",   # vision tower
    "model.language_model": "auto",  # language model on GPU (auto will use GPU)
    "lm_head": "auto",
}

print("🧠 Loading model with vision tower on CPU, language model quantised...")

# Try 4‑bit first
try:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        quantization_config=quant_config_4bit,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    print("✅ Model loaded with 4‑bit quantisation (vision tower on CPU).")
except Exception as e:
    print(f"⚠️ 4‑bit loading failed: {e}")
    print("🔄 Trying 8‑bit fallback...")
    gc.collect()
    torch.cuda.empty_cache()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        quantization_config=quant_config_8bit,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    print("✅ Model loaded with 8‑bit quantisation (vision tower on CPU).")

model.eval()

# ==========================================
# PII EXTRACTION & REDACTION FUNCTIONS
# ==========================================

def extract_pii_from_image(image_path: str) -> str:
    """
    Send image to VLM and get raw output with bounding boxes.
    Image resolution is limited to save VRAM.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "max_pixels": 200000,   # Reduced further from 313600
                },
                {
                    "type": "text",
                    "text": """Extract all Personal Identifiable Information (PII) from this image.
For each piece of PII, provide its exact text and its bounding box coordinates.
Format each finding exactly like this on a new line:
[extracted text] <box>(ymin, xmin), (ymax, xmax)</box>
The coordinates should be normalized to 0-1000 range.
If no PII, output: []""",
                },
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Clear cache before generation
    gc.collect()
    torch.cuda.empty_cache()

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    # Decode only the new tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n🤖 [RAW VLM OUTPUT]:")
    print(output_text)
    return output_text


def parse_vlm_output(response_text: str) -> list:
    """
    Parse the model's response to extract (text, bounding_box) pairs.
    Expected format: "[text] <box>(y1,x1),(y2,x2)</box>"
    Returns list of dicts: {"text": str, "box_2d": [ymin, xmin, ymax, xmax]}
    """
    pii_data = []
    pattern = r'\[(.*?)\]\s*<box>\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)</box>'
    for line in response_text.split('\n'):
        match = re.search(pattern, line)
        if match:
            text = match.group(1).strip()
            ymin, xmin, ymax, xmax = map(int, match.groups()[1:])
            pii_data.append({"text": text, "box_2d": [ymin, xmin, ymax, xmax]})
    return pii_data


def redact_image_vlm(image_path: str, pii_data: list):
    """
    Redact the image by drawing black rectangles over the PII bounding boxes.
    """
    if not pii_data:
        print("⚠️ No PII data to redact.")
        return

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    redaction_count = 0
    for item in pii_data:
        text = item.get("text", "Unknown")
        box = item.get("box_2d")
        if len(box) == 4:
            ymin, xmin, ymax, xmax = box
            # Convert normalized (0‑1000) coordinates to absolute pixels
            x1 = (xmin / 1000.0) * width
            y1 = (ymin / 1000.0) * height
            x2 = (xmax / 1000.0) * width
            y2 = (ymax / 1000.0) * height
            draw.rectangle([x1, y1, x2, y2], fill="black")
            redaction_count += 1
            print(f"  [X] Redacted: {text}")

    output_path = f"redacted_vlm_{os.path.basename(image_path)}"
    img.save(output_path)
    print(f"✅ Saved redacted image: {output_path} ({redaction_count} items censored)")


def run_omni_vlm(image_path: str):
    """
    Main entry point: extract PII + boxes, then redact the image.
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    print("-" * 50)
    raw_output = extract_pii_from_image(image_path)
    pii_data = parse_vlm_output(raw_output)
    redact_image_vlm(image_path, pii_data)
    print("-" * 50)


if __name__ == "__main__":
    # Example usage – replace with your image path
    run_omni_vlm("new.jpeg")