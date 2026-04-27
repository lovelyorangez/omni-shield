import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch
import re
import numpy as np
import random
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./qwen_3b_pii_qlora"

print(f"\n🚀 Loading Base Model ({MODEL_ID}) & LoRA Adapters...")

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
    low_cpu_mem_usage=True,
    torch_dtype=compute_dtype
)

# Load the fine-tuned PEFT adapters
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

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

def is_likely_pii(text: str) -> bool:
    text_lower = text.lower().strip()
    job_patterns = [
        r'\b(manager|director|engineer|specialist|coordinator|assistant|supervisor|lead|head|officer|president|ceo|cto|founder|hr|human resources|marketing|sales|finance|operations|technology|product|design|peer|student|professor|teacher)\b',
        r'\b(learning launch|design thinking|mind mapping|storytelling|visualization|assignment|course|project)\b',
    ]
    if any(re.search(p, text_lower) for p in job_patterns): return False
    org_keywords = ['university', 'college', 'institute', 'school', 'academy', 'inc', 'ltd', 'corporation', 'company']
    if any(word in text_lower for word in org_keywords): return False
    locations = ['india', 'china', 'usa', 'uk', 'germany', 'france', 'italy', 'spain', 'canada', 'mexico', 'brazil', 'colombia', 'argentina', 'chile', 'peru', 'venezuela', 'ecuador', 'bolivia', 'paraguay', 'uruguay', 'guyana', 'suriname', 'french guiana']
    if text_lower in locations: return False
    if len(text.split()) == 1 and text_lower in ['workshop', 'training', 'tool', 'project', 'report', 'essay', 'assignment', 'reflection', 'video', 'link', 'site', 'page', 'home', 'about', 'contact', 'faq', 'blog', 'article', 'post']: return False
    return True

SYSTEM_PROMPT = """You are an advanced PII Extraction AI. Your strict task is to extract ALL Personal Identifiable Information (PII) from the text. Maximize RECALL – if there’s any doubt, include it. We will filter later.

TARGET CATEGORIES TO EXTRACT (EVERYTHING BELOW):
1. Person Names – Full names AND isolated first names of the author, peers, or anyone mentioned.
2. Email Addresses.
3. Phone Numbers.
4. URLs / Links – ANY web link (http, https, www, YouTube, Facebook, etc.).
5. Identification Numbers – account numbers, IDs like "IV-8322", "Z.S. 30407059", "V69230", "06EYD876".

Return your extracted entities as a valid JSON array of strings. DO NOT provide explanations. If no matches, return exactly: []"""

print("\n⏳ Loading Local Dataset for Evaluation (Kaggle 70/30 Split)...")
with open("datasets/kaggle_train.json", "r", encoding="utf-8") as f:
    kaggle_raw = json.load(f)

docs_with_pii = [doc for doc in kaggle_raw if any(label != "O" for label in doc["labels"])]

# Consistent shuffle and 70/30 split to match training
random.seed(42)
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

print(f"\n📊 Running Fine-Tuned 3B LLM Evaluation on {len(dataset)} validation documents...")
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
            max_new_tokens=512, 
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
        
        if not isinstance(predicted_entities, list): predicted_entities = []
        predicted_entities = [str(e) for e in predicted_entities]
    except Exception:
        predicted_entities = []

    ground_truth_cleaned = set(clean_text(e) for e in doc["ground_truth"] if len(clean_text(e)) > 2)
    predicted_clean = set(clean_text(e) for e in predicted_entities if len(clean_text(e)) > 2)

    # Apply Post-Processing Filter
    filtered_predicted_clean = set(e for e in predicted_clean if is_likely_pii(e))

    tp, fps, fns = calculate_overlap_diagnostic(ground_truth_cleaned, filtered_predicted_clean)
    
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
print(f"🚀 FINE-TUNED 3B LLM RESULTS (Validation Split)")
print(f"==================================================")
print(f"Total True Positives: {total_tp}")
print(f"Total False Positives: {total_fp}")
print(f"Total False Negatives: {total_fn}")
print(f"🎯 Precision:  {precision:.2%}")
print(f"🔍 Recall:     {recall:.2%}")
print(f"⭐ F1-Score:   {f1:.2%}")
print(f"⏱️ Avg Latency: {avg_latency:.2f} seconds / document")
print(f"==================================================")