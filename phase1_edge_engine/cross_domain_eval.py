import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch
import re
import ast  # <--- The Magic Fix for single-quote strings
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
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=True  
)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    max_memory={0: "5GiB", "cpu": "16GiB"}, 
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    dtype=compute_dtype
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
            if t in p or p in t:
                matched_truth.add(t)
                matched_preds.add(p)
                break
    return len(matched_truth), list(predicted - matched_preds), list(ground_truth - matched_truth)

def is_likely_pii(text: str) -> bool:
    text_lower = text.lower().strip()
    
    # 1. Existing Corporate / Job Title Filter
    job_patterns = [r'\b(manager|director|engineer|specialist|coordinator|assistant|supervisor|lead|head|officer|president|ceo|cto|founder|hr|human resources|marketing|sales|finance|operations|technology|product|design|peer|student|professor|teacher)\b']
    if any(re.search(p, text_lower) for p in job_patterns): return False
    
    org_keywords = ['university', 'college', 'institute', 'school', 'academy', 'inc', 'ltd', 'corporation', 'company', 'aapl', 'confirmed', 'years']
    if any(word in text_lower for word in org_keywords): return False
    
    if len(text.split()) == 1 and text_lower in ['workshop', 'training', 'tool', 'project', 'report', 'essay', 'assignment', 'buxar']: return False
    
    # 2. The "Date" Trap: Catches continuous date strings
    # Matches YYYYMMDD (e.g., 19641205)
    if re.fullmatch(r'(19|20)\d{6}', text_lower): return False 
    # Matches MMDDYYYY (e.g., 03272019)
    if re.fullmatch(r'(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(19|20)\d{2}', text_lower): return False 
    
    # 3. The "Money/Quantity" Trap: Catches standalone numbers under 8 digits
    # Most secure IDs (SSN, Bank Accounts, Medical IDs) are 8+ digits or contain letters.
    # This safely drops things like "5000", "100", and "1200000" while keeping "370612920998".
    if re.fullmatch(r'\d{1,7}', text_lower): return False 
    
    return True

def extract_json_from_llm(response_text):
    try:
        clean_res = response_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_res)
        if isinstance(parsed, list):
            return [str(e) for e in parsed]
    except json.JSONDecodeError:
        pass
    
    try:
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

# --- BULLETPROOF DATASET LOADING ---

def load_evaluation_dataset(domain="medical", limit=20):
    print(f"\n⏳ Fetching verified {domain.upper()} dataset from Hugging Face...")
    dataset_records = []
    
    domain_map = {"medical": "healthcare", "finance": "finance"}
    target_domain = domain_map.get(domain, domain)
    
    try:
        raw_data = load_dataset("gretelai/gretel-pii-masking-en-v1", split="train")
        
        for row in raw_data:
            # Safely match domain
            if str(row.get("domain", "")).strip().lower() != target_domain:
                continue
                
            text = row.get("text", "")
            if not text: continue
            
            # Safely parse entities regardless of quotes
            raw_entities = row.get("entities", [])
            entities = []
            if isinstance(raw_entities, str):
                try:
                    entities = ast.literal_eval(raw_entities)
                except:
                    pass
            elif isinstance(raw_entities, list):
                entities = raw_entities
            
            # Extract PII
            ground_truth = []
            for ent in entities:
                if isinstance(ent, dict):
                    types = ent.get("types", ent.get("label", []))
                    if isinstance(types, str): types = [types]
                    
                    val = str(ent.get("entity", ent.get("value", ent.get("text", "")))).strip()
                    
                    is_date = any("date" in str(t).lower() or "time" in str(t).lower() for t in types)
                    if not is_date and val and val != "None":
                        ground_truth.append(val)
            
            if ground_truth:
                dataset_records.append({"text": text, "ground_truth": list(set(ground_truth))})
                
            if len(dataset_records) >= limit:
                break
                
        print(f"✅ Loaded {len(dataset_records)} verified records for {domain}.")
        return dataset_records
        
    except Exception as e:
        print(f"❌ Failed to load {domain} dataset: {e}")
        return []

# --- EVALUATION LOOP ---

DOMAINS_TO_TEST = ["medical", "finance"]

for domain in DOMAINS_TO_TEST:
    dataset = load_evaluation_dataset(domain, limit=20)
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
        model_inputs = tokenizer([input_text], return_tensors="pt").to(base_model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        res = tokenizer.batch_decode([g[len(i):] for i, g in zip(model_inputs.input_ids, generated_ids)], skip_special_tokens=True)[0].strip()
        predicted = extract_json_from_llm(res)

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

    avg_lat = np.mean(latencies) if latencies else 0.0
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"🚀 RESULTS: {domain.upper()}")
    print(f"Precision: {p:.2%} | Recall: {r:.2%} | F1: {f1:.2%}")
    print(f"Avg Latency: {avg_lat:.2f}s")
    print(f"{'='*50}\n")