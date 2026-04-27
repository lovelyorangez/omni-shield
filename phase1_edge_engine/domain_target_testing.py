import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch
import re
import numpy as np
import random
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./qwen_3b_pii_qlora"

is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print(f"\n🚀 Loading Domain-Specific Validation Engine...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto", 
    quantization_config=quantization_config,
    torch_dtype=compute_dtype
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

SYSTEM_PROMPT = """You are an exhaustive PII Extraction AI. 
EXTRACT EVERY SINGLE PIECE of Personal Identifiable Information from the text. 

REQUIRED ENTITIES:
- All Names (Authors, Patients, Clients, Employees)
- All Email Addresses
- All URLs
- All Numeric IDs (MRN, SSN, Account Numbers, Passports)
- All Phone Numbers

OUTPUT: Return ONLY a JSON array of strings. If unsure, INCLUDE IT."""

REGEX_PATTERNS = {
    "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "URL": r'(https?://[^\s()<>]+|www\.[^\s()<>]+)',
    "ID": r'\b([a-zA-Z0-9_]+\|\d{4,}|[A-Z]{2,}\d{4,12}|[A-Z0-9]{3,}\-[A-Z0-9]{4,}|\d{3}-\d{2}-\d{4}|\d{9,21})\b',
    "PHONE": r'(\+?\d{1,3}[-.\s]?)?\(?\s?\d{1,4}\s?\)?[-\s.]?\d{3}[-\s.]?\d{4}'
}

def normalize_for_comparison(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', str(text)).lower()

def is_source_verified_fuzzy(pii_candidate: str, source_text: str) -> bool:
    cand_norm = normalize_for_comparison(pii_candidate)
    if not cand_norm or len(cand_norm) < 2: return False
    source_norm = normalize_for_comparison(source_text)
    if cand_norm in source_norm:
        noise = ['university', 'college', 'hospital', 'clinic', 'bank', 'corp', 'inc', 'llc']
        if any(word in pii_candidate.lower() for word in noise): return False
        return True
    return False

def get_pii_from_chunk(chunk_text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"TEXT: {chunk_text[:3500]}"}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    try:
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except: return []

def load_medical_dataset(limit=20):
    print("⏳ Generating Synthetic Medical PII Validation Set...")
    patients = ["Gregory House", "Allison Cameron", "James Wilson", "Lisa Cuddy", "Eric Foreman", "Robert Chase"]
    mrns = ["901299331", "8819234", "112233445", "998877665", "554433221"]
    emails = ["ghouse@princeton-plainsboro.edu", "acameron@hospital.org", "jwilson@med.net"]
    phones = ["1-800-555-0199", "(609) 555-0100", "+1 212-555-0198"]
    
    formatted = []
    for i in range(limit):
        p = patients[i % len(patients)]
        m = mrns[i % len(mrns)]
        e = emails[i % len(emails)]
        ph = phones[i % len(phones)]
        
        if i % 2 == 0:
            text = f"Patient Profile: {p}. Medical Record Number (MRN): {m}. Contact at {e}."
            gt = [p, m, e]
        else:
            text = f"Discharge summary for patient {p}. Insurance ID: BCBS-{m}. Please follow up at {ph}."
            gt = [p, f"BCBS-{m}", ph]
            
        formatted.append({"domain": "MEDICAL", "text": text, "gt": gt})
    return formatted

def load_legal_dataset(limit=20):
    print("⏳ Loading Legal Dataset (CUAD_v1)...")
    try:
        with open("datasets/CUAD_v1.json", "r", encoding="utf-8") as f:
            cuad_data = json.load(f)
        formatted = []
        for doc in cuad_data.get("data", []):
            for paragraph in doc.get("paragraphs", []):
                text = paragraph.get("context", "")
                gt = []
                for qa in paragraph.get("qas", []):
                    if "Parties" in qa.get("question", ""):
                        for ans in qa.get("answers", []):
                            gt.append(ans["text"])
                if text and gt:
                    formatted.append({"domain": "LEGAL", "text": text, "gt": gt})
                    if len(formatted) >= limit: return formatted
        return formatted
    except Exception as e:
        print(f"⚠️ Legal dataset failed. Ensure datasets/CUAD_v1.json exists. {e}")
        return []

def load_financial_dataset(limit=20):
    print("⏳ Generating Financial PII Validation Set...")
    names = ["Bruce Wayne", "Selina Kyle", "Arthur Pendelton", "Maria Garcia", "Chen Wei", "Aisha Khan", "John Doe", "Jane Smith"]
    accts = ["00192837465", "8829100293", "CH9300000123456789012", "US99300029911", "992-102-3948"]
    emails = ["b.wayne@wayne.com", "selina@kyle-finance.net", "arthur.p@bank.org", "m.garcia@invest.co"]
    phones = ["212-555-0188", "+1 800-555-0199", "415.555.2031", "(312) 555-9912"]
    
    formatted = []
    for i in range(limit):
        n = names[i % len(names)]
        a = accts[i % len(accts)]
        e = emails[i % len(emails)]
        p = phones[i % len(phones)]
        
        if i % 2 == 0:
            text = f"Wire transfer authorized by client {n}. Source Account: {a}. Confirm via {e} or call {p}."
            gt = [n, a, e, p]
        else:
            text = f"Audit report for portfolio manager {n}. Employee Tax ID / Routing Number: {a}. Direct inquiries to {e}."
            gt = [n, a, e]
            
        formatted.append({"domain": "FINANCIAL", "text": text, "gt": gt})
    return formatted

# --- COMPILE DOMAIN DATA ---
DOMAIN_DATA = load_medical_dataset(20) + load_legal_dataset(20) + load_financial_dataset(20)

print("\n--------------------------------------------------")
domain_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

for idx, doc in enumerate(DOMAIN_DATA):
    full_text = doc["text"]
    gt_entities = doc["gt"]
    
    # PASS 1: LLM
    llm_preds = get_pii_from_chunk(full_text)
    
    # PASS 2: REGEX RESCUE
    regex_preds = []
    for ptype, pattern in REGEX_PATTERNS.items():
        matches = re.findall(pattern, full_text, flags=re.IGNORECASE if ptype in ["EMAIL", "URL"] else 0)
        regex_preds.extend(matches)
        
    # COMBINE & VERIFY
    combined_preds = list(set(llm_preds + regex_preds))
    pred_clean = [p for p in combined_preds if is_source_verified_fuzzy(p, full_text)]
    
    # METRICS
    matched_gt_indices = set()
    domain = doc["domain"]
    
    for p in pred_clean:
        found_match = False
        p_norm = normalize_for_comparison(p)
        for g_idx, g in enumerate(gt_entities):
            if g_idx in matched_gt_indices: continue
            g_norm = normalize_for_comparison(g)
            if p_norm in g_norm or g_norm in p_norm:
                domain_metrics[domain]["tp"] += 1
                matched_gt_indices.add(g_idx)
                found_match = True
                break
        if not found_match: 
            domain_metrics[domain]["fp"] += 1

    for g_idx, g in enumerate(gt_entities):
        if g_idx not in matched_gt_indices:
            domain_metrics[domain]["fn"] += 1
            
    if (idx + 1) % 10 == 0: print(f"  Processed {idx + 1}/{len(DOMAIN_DATA)} documents...")

print(f"\n{'DOMAIN':<12} | {'PRECISION':<10} | {'RECALL':<10} | {'F1-SCORE':<10}")
print("-" * 50)

total_tp, total_fp, total_fn = 0, 0, 0
for domain, stats in domain_metrics.items():
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"{domain:<12} | {p:>9.2%} | {r:>9.2%} | {f1:>9.2%}")
    total_tp += tp; total_fp += fp; total_fn += fn

print("-" * 50)
p_ov = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
r_ov = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1_ov = 2 * p_ov * r_ov / (p_ov + r_ov) if (p_ov + r_ov) > 0 else 0
print(f"{'OVERALL':<12} | {p_ov:>9.2%} | {r_ov:>9.2%} | {f1_ov:>9.2%}")
print("="*50)