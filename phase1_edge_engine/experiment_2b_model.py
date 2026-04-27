import json
import time
import torch
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. LOAD THE 1.5B PARAMETER MODEL
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"🚀 Initializing 1.5B Parameter Experiment...")
print(f"⏳ Downloading/Loading {MODEL_ID}...")

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

# 2. THE ABSTRACT SYSTEM PROMPT (No realistic examples to copy!)
SYSTEM_PROMPT = """You are an elite Data Privacy Extraction Engine. Read the following student essay and extract ALL Personal Identifiable Information (PII) belonging ONLY to the author/student.

CRITICAL RULES:
1. Extract exact strings of the student's name, email, phone numbers, ID numbers, and URLs.
2. DO NOT hallucinate. DO NOT invent information. Extract ONLY what is physically written in the text.
3. IGNORE historical figures (e.g., John Stuart Mill, Adam Smith, Buzan).
4. IGNORE locations, universities, or companies.

OUTPUT FORMAT:
Output ONLY a valid JSON list of strings. 
Format: ["<extracted_string_1>", "<extracted_string_2>"]
If no PII is found, output exactly: []"""

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

print(f"\n📊 Running 1.5B LLM Evaluation on 30 documents...")
print("-" * 75)

total_tp, total_fp, total_fn = 0, 0, 0
latencies = []
diagnostics = []

for idx, doc in enumerate(dataset):
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"ESSAY TEXT:\n{doc['text'][:3000]}"} 
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=50,
            do_sample=False, # STRICT GREEDY DECODING (Stops hallucination)
        )
        
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    predicted_entities = []
    try:
        list_str = re.search(r'\[.*?\]', response, re.DOTALL).group(0)
        # Safely evaluate string representation of list
        import ast
        predicted_entities = ast.literal_eval(list_str)
        if not isinstance(predicted_entities, list): predicted_entities = []
        # Force convert everything to string just in case it extracts a raw number
        predicted_entities = [str(e) for e in predicted_entities]
    except Exception:
        predicted_entities = []

    ground_truth_cleaned = set(clean_text(e) for e in doc["ground_truth"] if len(clean_text(e)) > 2)
    predicted_clean = set(clean_text(e) for e in predicted_entities if len(clean_text(e)) > 2)

    tp, fps, fns = calculate_overlap_diagnostic(ground_truth_cleaned, predicted_clean)
    
    total_tp += tp
    total_fp += len(fps)
    total_fn += len(fns)
    
    for fp in fps: diagnostics.append((fp, "FALSE POSITIVE", doc['ground_truth']))
    for fn in fns: diagnostics.append((fn, "FALSE NEGATIVE", response))
    
    latency = time.time() - start_time
    latencies.append(latency)
    
    print(f"Doc {idx+1:02d} | Found: {predicted_entities} | Truth: {doc['ground_truth']} | Time: {latency:.2f}s")

avg_latency = np.mean(latencies)
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\n==================================================")
print(f"🚀 1.5B LLM EXPERIMENT - RESULTS")
print(f"==================================================")
print(f"Total True Positives: {total_tp}")
print(f"Total False Positives: {total_fp}")
print(f"Total False Negatives: {total_fn}")
print(f"🎯 Precision:  {precision:.2%}")
print(f"🔍 Recall:     {recall:.2%}")
print(f"⭐ F1-Score:   {f1:.2%}")
print(f"⏱️ Avg Latency: {avg_latency:.2f} seconds / document")
print(f"==================================================")