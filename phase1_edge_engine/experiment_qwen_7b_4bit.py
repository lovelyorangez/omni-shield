import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import ast

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

torch.cuda.empty_cache()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cuda", 
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16
)

def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

def calculate_overlap_diagnostic(ground_truth, predicted):
    matched_truth, matched_preds = set(), set()
    for t in ground_truth:
        for p in predicted:
            if p in matched_preds: continue
            if t in p or p in t:
                matched_truth.add(t)
                matched_preds.add(p)
                break
    return len(matched_truth), list(predicted - matched_preds), list(ground_truth - matched_truth)

SYSTEM_PROMPT = """You are an advanced PII Extraction AI. Your strict task is to extract ALL Personal Identifiable Information (PII) from the text. Maximize RECALL. If you see a human name, extract it!

TARGET CATEGORIES TO EXTRACT:
1. Person Names (Extract FULL names AND isolated first names).
2. Email Addresses.
3. Phone Numbers.
4. Personal URLs / Links.
5. Identification Numbers (e.g., account numbers, IDs like "IV-8322").

Return your extracted entities as a valid JSON array of strings. DO NOT provide any explanations. If there are no matches, return exactly: []

Example 1:
Text: My name is Junior Wallace. I go to Gitam University.
Output: ["Junior Wallace"]

Example 2:
Text: Student David and Cristiane worked together. David submitted ID V69230 and IV-8322.
Output: ["David", "Cristiane", "V69230", "IV-8322"]

Example 3:
Text: Contact me at hbrown@yahoo.com or my site https://alvarado.com/categoriesindex.html.
Output: ["hbrown@yahoo.com", "https://alvarado.com/categoriesindex.html"]"""

import random
random.seed(42)

print("\n⏳ Loading Local Dataset for Evaluation (Kaggle 70/30 Split)...")
with open("datasets/kaggle_train.json", "r", encoding="utf-8") as f:
    kaggle_raw = json.load(f)

# Filter for documents that actually contain PII
docs_with_pii = [doc for doc in kaggle_raw if any(label != "O" for label in doc["labels"])]

# Create a true 70/30 random split of the dataset
random.shuffle(docs_with_pii)
split_index = int(len(docs_with_pii) * 0.7)
validation_docs = docs_with_pii[split_index:]

dataset = []

for doc in validation_docs:  
    text = " ".join(doc["tokens"])
    entities, current_entity = [], []
    for token, label in zip(doc["tokens"], doc["labels"]):
        if label != "O": current_entity.append(token)
        elif current_entity:
            entities.append(" ".join(current_entity))
            current_entity = []
    if current_entity: entities.append(" ".join(current_entity))
    dataset.append({"text": text, "ground_truth": entities})

total_tp, total_fp, total_fn = 0, 0, 0
latencies = []

print(f"\n📊 Running 7B 4-Bit LLM Evaluation on {len(dataset)} validation documents...")
print("-" * 75)

for idx, doc in enumerate(dataset):
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TEXT: {doc['text'][:5000]}"} 
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    predicted_entities = []
    try:
        list_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if list_match:
            try:
                predicted_entities = json.loads(list_match.group(0))
            except json.JSONDecodeError:
                predicted_entities = ast.literal_eval(list_match.group(0))
        
        if not isinstance(predicted_entities, list): 
            predicted_entities = []
        predicted_entities = [str(e) for e in predicted_entities]
    except Exception:
        predicted_entities = []

    ground_truth_cleaned = set(clean_text(e) for e in doc["ground_truth"] if len(clean_text(e)) > 2)
    predicted_clean = set(clean_text(e) for e in predicted_entities if len(clean_text(e)) > 2)

    tp, fps, fns = calculate_overlap_diagnostic(ground_truth_cleaned, predicted_clean)
    
    total_tp += tp
    total_fp += len(fps)
    total_fn += len(fns)
    
    latency = time.time() - start_time
    latencies.append(latency)
    
    print(f"Doc {idx+1:02d} | Found: {predicted_entities} | Truth: {doc['ground_truth']} | Time: {latency:.2f}s")

avg_latency = np.mean(latencies) if latencies else 0.0
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\n==================================================")
print(f"🚀 7B 4-BIT LLM OPTIMIZED RESULTS (Validation Split)")
print(f"==================================================")
print(f"Total True Positives: {total_tp}")
print(f"Total False Positives: {total_fp}")
print(f"Total False Negatives: {total_fn}")
print(f"🎯 Precision:  {precision:.2%}")
print(f"🔍 Recall:     {recall:.2%}")
print(f"⭐ F1-Score:   {f1:.2%}")
print(f"⏱️ Avg Latency: {avg_latency:.2f} seconds / document")
print(f"==================================================")