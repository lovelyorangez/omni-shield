import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

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

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

# --- HELPER FUNCTIONS ---

def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

def calculate_overlap_diagnostic(ground_truth, predicted):
    matched_truth, matched_preds = set(), set()
    for t in ground_truth:
        for p in predicted:
            if p in matched_preds: continue
            # Check for partial overlaps (e.g., "John Doe" vs "John")
            if t in p or p in t:
                matched_truth.add(t)
                matched_preds.add(p)
                break
    return len(matched_truth), list(predicted - matched_preds), list(ground_truth - matched_truth)

def is_likely_pii(text: str) -> bool:
    # Keeps your existing robust filtering logic
    text_lower = text.lower().strip()
    job_patterns = [r'\b(manager|director|engineer|specialist|coordinator|assistant|supervisor|lead|head|officer|president|ceo|cto|founder|hr|human resources|marketing|sales|finance|operations|technology|product|design|peer|student|professor|teacher)\b']
    if any(re.search(p, text_lower) for p in job_patterns): return False
    org_keywords = ['university', 'college', 'institute', 'school', 'academy', 'inc', 'ltd', 'corporation', 'company']
    if any(word in text_lower for word in org_keywords): return False
    if len(text.split()) == 1 and text_lower in ['workshop', 'training', 'tool', 'project', 'report', 'essay', 'assignment']: return False
    return True

def extract_json_from_llm(response_text):
    """Robustly extracts JSON arrays even if the LLM adds markdown or conversational text."""
    try:
        # First attempt: Clean markdown
        clean_res = response_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_res)
        if isinstance(parsed, list):
            return [str(e) for e in parsed]
    except json.JSONDecodeError:
        pass
    
    try:
        # Second attempt: Regex fallback for the array
        m = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if m:
            parsed = json.loads(m.group(0))
            if isinstance(parsed, list):
                return [str(e) for e in parsed]
    except:
        pass
    
    return []

SYSTEM_PROMPT = """You are an advanced PII Extraction AI. Your strict task is to extract ALL Personal Identifiable Information (PII) from the text. Maximize RECALL – if there’s any doubt, include it. We will filter later.

TARGET CATEGORIES TO EXTRACT:
1. Person Names.
2. Email Addresses.
3. Phone Numbers.
4. URLs / Links.
5. Identification Numbers (SSNs, Account numbers, Medical IDs).

Return your extracted entities as a valid JSON array of strings. DO NOT provide explanations. If no matches, return exactly: []"""

# --- ROBUST DATASET LOADING ---

def load_evaluation_dataset(domain="medical", limit=20):
    print(f"\n⏳ Fetching verified {domain.upper()} dataset from Hugging Face...")
    dataset_records = []
    
    try:
        # Using Gretel's synthetic PII dataset which has rich medical/financial contexts
        raw_data = load_dataset("gretelai/synthetic_pii", split="train")
        
        # Filter for the specific domain we want to test
        domain_data = raw_data.filter(lambda x: x.get("domain", "") == domain or x.get("context", "") == domain)
        
        # Take the first 'limit' rows
        for row in domain_data.select(range(min(limit, len(domain_data)))):
            text = row.get("text", "")
            spans = json.loads(row.get("spans", "[]"))
            
            # Extract ground truth entities from the spans
            ground_truth = []
            for span in spans:
                # Filter out generic tags we don't care about (like generic dates)
                if span.get("label") not in ["DATE", "TIME"]: 
                    ground_truth.append(span.get("value"))
            
            if text and ground_truth:
                dataset_records.append({"text": text, "ground_truth": list(set(ground_truth))})
                
        print(f"✅ Loaded {len(dataset_records)} verified records for {domain}.")
        return dataset_records
        
    except Exception as e:
        print(f"❌ Failed to load {domain} dataset: {e}")
        return []

# --- EVALUATION LOOP ---

# Testing the cross-domain generalizability you need for the paper
DOMAINS_TO_TEST = ["medical", "finance"]

for domain in DOMAINS_TO_TEST:
    dataset = load_evaluation_dataset(domain, limit=20) # Increase limit for final paper numbers
    if not dataset: continue
    
    print(f"\n📊 Evaluating Fine-Tuned 3B on: {domain.upper()} Data...")
    print("-" * 75)
    total_tp, total_fp, total_fn, latencies = 0, 0, 0, []

    for idx, doc in enumerate(dataset):
        start_time = time.time()
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}, 
            {"role": "user", "content": f"TEXT: {str(doc['text'])[:4000]}"}
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
        
        # Decode the output, ignoring the prompt portion
        res = tokenizer.batch_decode([g[len(i):] for i, g in zip(model_inputs.input_ids, generated_ids)], skip_special_tokens=True)[0].strip()
        
        predicted = extract_json_from_llm(res)

        # Clean and filter
        gt_clean = set(clean_text(e) for e in doc["ground_truth"] if len(clean_text(e)) > 2)
        pred_clean = set(clean_text(e) for e in predicted if is_likely_pii(e) and len(clean_text(e)) > 2)

        tp, fps, fns = calculate_overlap_diagnostic(gt_clean, pred_clean)
        
        total_tp += tp
        total_fp += len(fps)
        total_fn += len(fns)
        latencies.append(time.time() - start_time)
        
        print(f"Doc {idx+1:02d} | Latency: {latencies[-1]:.2f}s")
        if fps: print(f"   [!] False Positives: {fps}")
        if fns: print(f"   [?] False Negatives: {fns}")

    # Final Math for this Domain
    avg_lat = np.mean(latencies) if latencies else 0.0
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"🚀 RESULTS: {domain.upper()}")
    print(f"Precision: {p:.2%} | Recall: {r:.2%} | F1: {f1:.2%}")
    print(f"Avg Latency: {avg_lat:.2f}s")
    print(f"{'='*50}\n")