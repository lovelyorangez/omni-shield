import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch
import re
import numpy as np
import ast
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./qwen_3b_pii_qlora"

print(f"\n🚀 Loading RECALL-MAXIMIZED ENSEMBLE Engine...")

is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto", 
    quantization_config=quantization_config,
    torch_dtype=compute_dtype
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# RECALL-FOCUSED PROMPT
SYSTEM_PROMPT = """You are an exhaustive PII Extraction AI. 
EXTRACT EVERY SINGLE PIECE of Personal Identifiable Information from the text. 

REQUIRED ENTITIES:
- All Names (Authors, Peers, Teachers)
- All Email Addresses
- All URLs (LinkedIn, YouTube, Personal Blogs)
- All Numeric IDs (Aadhar, Student IDs, Account Numbers)
- All Phone Numbers

OUTPUT: Return ONLY a JSON array of strings. If unsure, INCLUDE IT."""

# Regex Safety Net (Optimized to fix the ID False Positive explosion)
REGEX_PATTERNS = {
    "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "URL": r'(https?://[^\s()<>]+|www\.[^\s()<>]+|youtube\.com/[^\s()<>]+|linkedin\.com/[^\s()<>]+)',
    "ID": r'\b([a-zA-Z0-9_]+\|\d{4,}|[A-Z]{2,}\d{4,12}|[A-Z0-9]{3,}\-[A-Z0-9]{4,})\b',
    "PHONE": r'(\+?\d{1,3}[-.\s]?)?\(?\s?\d{1,4}\s?\)?[-\s.]?\d{3}[-\s.]?\d{4}'
}

def classify_pii_type(text):
    text = str(text).strip()
    if re.search(REGEX_PATTERNS["EMAIL"], text, re.IGNORECASE): return "EMAIL"
    if re.search(REGEX_PATTERNS["URL"], text, re.IGNORECASE): return "URL"
    if re.search(REGEX_PATTERNS["ID"], text): return "ID"
    if re.search(REGEX_PATTERNS["PHONE"], text): return "PHONE"
    return "NAME"

def normalize_for_comparison(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

def is_source_verified_fuzzy(pii_candidate: str, source_text: str) -> bool:
    cand_norm = normalize_for_comparison(pii_candidate)
    if not cand_norm or len(cand_norm) < 2: return False
    source_norm = normalize_for_comparison(source_text)
    if cand_norm in source_norm:
        noise = ['university', 'college', 'india', 'china', 'usa', 'assignment', 'project', 'report']
        if any(word in pii_candidate.lower() for word in noise): return False
        return True
    return False

def get_pii_from_chunk(chunk_text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"TEXT: {chunk_text}"}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    try:
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except: return []

# 1. Load Validation Data
with open("datasets/kaggle_train.json", "r", encoding="utf-8") as f:
    kaggle_raw = json.load(f)

docs_with_pii = [doc for doc in kaggle_raw if any(label != "O" for label in doc["labels"])]
import random
random.seed(42)
random.shuffle(docs_with_pii)
validation_docs = docs_with_pii[int(len(docs_with_pii) * 0.7):]

category_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
error_log = []

print(f"📊 Running SLIDING WINDOW + REGEX RESCUE on {len(validation_docs)} documents...")

for idx, doc in enumerate(validation_docs):
    full_text = " ".join(doc["tokens"])
    
    # GROUND TRUTH
    gt_entities = []
    current_entity = []
    for token, label in zip(doc["tokens"], doc["labels"]):
        if label != "O": current_entity.append(token)
        elif current_entity:
            gt_entities.append(" ".join(current_entity))
            current_entity = []
    if current_entity: gt_entities.append(" ".join(current_entity))
    gt_categorized = [{"text": e, "type": classify_pii_type(e)} for e in gt_entities]

    # PASS 1: SLIDING WINDOW LLM INFERENCE
    chunk_size = 3500
    overlap = 500
    all_llm_preds = []
    for start in range(0, len(full_text), chunk_size - overlap):
        chunk = full_text[start:start + chunk_size]
        all_llm_preds.extend(get_pii_from_chunk(chunk))
    
    # PASS 2: REGEX RESCUE (Safety Net for missed technical PII)
    regex_preds = []
    for ptype, pattern in REGEX_PATTERNS.items():
        matches = re.findall(pattern, full_text, flags=re.IGNORECASE if ptype in ["EMAIL", "URL"] else 0)
        regex_preds.extend(matches)
        
    # COMBINE & VERIFY
    combined_preds = list(set(all_llm_preds + regex_preds))
    pred_clean = [p for p in combined_preds if is_source_verified_fuzzy(p, full_text)]
    
    # METRIC CALCULATION
    matched_gt_indices = set()
    for p in pred_clean:
        p_type = classify_pii_type(p)
        found_match = False
        p_norm = normalize_for_comparison(p)
        for g_idx, g in enumerate(gt_categorized):
            if g_idx in matched_gt_indices: continue
            g_norm = normalize_for_comparison(g["text"])
            if p_norm in g_norm or g_norm in p_norm:
                category_stats[g["type"]]["tp"] += 1
                matched_gt_indices.add(g_idx)
                found_match = True
                break
        if not found_match: category_stats[p_type]["fp"] += 1

    for g_idx, g in enumerate(gt_categorized):
        if g_idx not in matched_gt_indices:
            category_stats[g["type"]]["fn"] += 1
            error_log.append({"type": "FN", "entity": g["text"], "category": g["type"]})

    if (idx + 1) % 25 == 0: print(f"  Processed {idx + 1}/{len(validation_docs)}...")

# Final Report
print("\n" + "="*70)
print(f"{'CATEGORY':<12} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10} | {'GT'}")
print("-" * 70)
total_tp, total_fp, total_fn = 0, 0, 0
for cat in ["NAME", "EMAIL", "URL", "ID", "PHONE", "UNKNOWN"]:
    stats = category_stats[cat]
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    if tp == 0 and fp == 0 and fn == 0: continue
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"{cat:<12} | {p:>9.2%} | {r:>9.2%} | {f1:>9.2%} | {tp+fn:>4}")
    total_tp += tp; total_fp += fp; total_fn += fn

print("-" * 70)
p_ov = total_tp / (total_tp + total_fp); r_ov = total_tp / (total_tp + total_fn)
print(f"{'OVERALL':<12} | {p_ov:>9.2%} | {r_ov:>9.2%} | {2*p_ov*r_ov/(p_ov+r_ov):>9.2%} | {total_tp+total_fn}")
print("="*70)

print("\n🚨 Top 5 Missing (Recall Killers):")
for e in error_log[:5]: print(f" - [{e['category']}]: {e['entity']}")