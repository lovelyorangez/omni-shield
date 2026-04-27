import json
import time
import torch
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. LOAD THE 3.1B PARAMETER MODEL (Qwen2.5-3B-Instruct)
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

print(f"🚀 Initializing 3B Parameter Research Experiment...")
print(f"⏳ Loading {MODEL_ID} (High-Precision Inference Mode)...")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
)

if not torch.cuda.is_available():
    model.to("cpu")

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

# 2. DIRECT EXTRACTION PROMPT
SYSTEM_PROMPT = """Extract the Personal Identifiable Information (PII) of the student/author from the text. 
Target entities: Student Names, Emails, Phone Numbers, and Personal URLs.
Ignore: Historical figures, cities, countries, and universities.

You must respond ONLY with a valid JSON list of strings. Do not add markdown formatting, explanations, or conversational text.
If no PII is found, output: []

Example of expected output:
["Jane Doe", "jane.doe@email.com", "555-0198"]"""

print("\n⏳ Loading Kaggle Dataset...")
with open("datasets/kaggle_train.json", "r", encoding="utf-8") as f:
    kaggle_raw = json.load(f)

dataset = []
for doc in kaggle_raw[:30]:  
    text = " ".join(doc["tokens"])
    entities, current_entity = [], []
    for token, label in zip(doc["tokens"], doc["labels"]):
        if label != "O": current_entity.append(token)
        elif current_entity:
            entities.append(" ".join(current_entity))
            current_entity = []
    if current_entity: entities.append(" ".join(current_entity))
    dataset.append({"text": text, "ground_truth": entities})

print(f"\n📊 Running 3B LLM Evaluation on 30 documents...")
print("-" * 75)

total_tp, total_fp, total_fn = 0, 0, 0
latencies = []

for idx, doc in enumerate(dataset):
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TEXT: {doc['text'][:3500]}"} 
    ]
    
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        # CRITICAL FIX: Removed repetition_penalty (it breaks JSON syntax like quotes and brackets)
        # CRITICAL FIX: Set do_sample=False for strict greedy decoding to stop hallucinations
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=150, 
            do_sample=False 
        )
        
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    predicted_entities = []
    try:
        # Robustly extract content inside brackets (non-greedy .*?)
        list_match = re.search(r'\[.*?\]', response, re.DOTALL)
        if list_match:
            predicted_entities = json.loads(list_match.group(0).replace("'", '"'))
        
        if not isinstance(predicted_entities, list): 
            predicted_entities = []
        predicted_entities = [str(e) for e in predicted_entities]
    except Exception:
        predicted_entities = []

    # Clean for metric comparison
    ground_truth_cleaned = set(clean_text(e) for e in doc["ground_truth"] if len(clean_text(e)) > 2)
    predicted_clean = set(clean_text(e) for e in predicted_entities if len(clean_text(e)) > 2)

    tp, fps, fns = calculate_overlap_diagnostic(ground_truth_cleaned, predicted_clean)
    
    total_tp += tp
    total_fp += len(fps)
    total_fn += len(fns)
    
    latency = time.time() - start_time
    latencies.append(latency)
    
    print(f"Doc {idx+1:02d} | Found: {predicted_entities} | Truth: {doc['ground_truth']} | Time: {latency:.2f}s")

avg_latency = np.mean(latencies)
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\n==================================================")
print(f"🚀 3B LLM GREEDY DECODING RESULTS (Qwen2.5-3B)")
print(f"==================================================")
print(f"Total True Positives: {total_tp}")
print(f"Total False Positives: {total_fp}")
print(f"Total False Negatives: {total_fn}")
print(f"🎯 Precision:  {precision:.2%}")
print(f"🔍 Recall:     {recall:.2%}")
print(f"⭐ F1-Score:   {f1:.2%}")
print(f"⏱️ Avg Latency: {avg_latency:.2f} seconds / document")
print(f"==================================================")