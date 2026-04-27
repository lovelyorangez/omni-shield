import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import magic
import easyocr
import pdfplumber
import json
import re
import ast
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ==========================================
# NODE 0: INITIALIZE ENGINES (CPU & GPU Split)
# ==========================================
print("⚙️ Booting CPU Vision Node (EasyOCR)...")
# Force CPU for OCR to save VRAM for the LLM
reader = easyocr.Reader(['en'], gpu=False) 

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "./qwen_3b_pii_qlora"

print(f"🚀 Booting GPU Extraction Node ({MODEL_ID})...")
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

SYSTEM_PROMPT = """You are an advanced PII Extraction AI. Extract ALL Personal Identifiable Information (PII) from the text. 
TARGET CATEGORIES: Person Names, Emails, Phone Numbers, URLs, Identification Numbers.
Return extracted entities as a valid JSON array of strings. DO NOT provide explanations. If no matches, return: []"""

# ==========================================
# NODE 1 & 2: ROUTING & PREPROCESSING
# ==========================================
def inspect_and_preprocess(filepath):
    """Determines file type and extracts clean text (and bounding boxes for images)."""
    mime = magic.from_file(filepath, mime=True)
    print(f"\n📂 File detected as: {mime}")
    
    if mime.startswith('image/'):
        print("👁️ Routing to Vision Node (OCR)...")
        results = reader.readtext(filepath)
        text = ' '.join([item[1] for item in results])
        boxes = [{"box": item[0], "text": item[1]} for item in results]
        return "image", text, boxes
        
    elif mime == 'application/pdf':
        print("📄 Routing to Document Node (PDF Plumber)...")
        text = ""
        with pdfplumber.open(filepath) as pdf:
            text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
        return "pdf", text, None
        
    else:
        print("📝 Routing to Text Node...")
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        return "text", text, None

# ==========================================
# NODE 3: GPU EXTRACTION (The Brain)
# ==========================================
def extract_json_from_llm(response_text):
    try:
        clean_res = response_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_res)
        if isinstance(parsed, list): return [str(e) for e in parsed]
    except: pass
    try:
        m = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if m: return [str(e) for e in json.loads(m.group(0))]
    except: pass
    return []

def extract_pii(text):
    """Feeds cleaned text to Qwen-3B and returns the JSON list of PII."""
    print("🧠 Engaging Qwen-3B Neural Extraction...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}, 
        {"role": "user", "content": f"TEXT: {text[:4000]}"}
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_text], return_tensors="pt").to(base_model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, max_new_tokens=512, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    res = tokenizer.batch_decode([g[len(i):] for i, g in zip(model_inputs.input_ids, generated_ids)], skip_special_tokens=True)[0].strip()
    return extract_json_from_llm(res)

# ==========================================
# NODE 4: REDACTION & RECONSTRUCTION
# ==========================================
def redact_image(filepath, boxes, pii_list):
    """Draws black rectangles over identified PII on the original image."""
    print("⬛ Applying Visual Redaction...")
    img = Image.open(filepath)
    draw = ImageDraw.Draw(img)
    
    redaction_count = 0
    # Clean the PII list for better matching
    clean_pii = [p.lower().strip() for p in pii_list]
    
    for item in boxes:
        box = item["box"]
        ocr_text = item["text"].lower().strip()
        
        # If the OCR text contains any of the PII strings (or vice versa), redact the box
        if any(pii in ocr_text or ocr_text in pii for pii in clean_pii if len(pii) > 2):
            # EasyOCR box format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            p1, p2, p3, p4 = box
            draw.polygon([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1]], fill="black")
            redaction_count += 1
            
    output_path = f"redacted_{os.path.basename(filepath)}"
    img.save(output_path)
    print(f"✅ Saved redacted image to: {output_path} ({redaction_count} items censored)")

def redact_text(text, pii_list):
    print("⬛ Applying Text Redaction...")
    redacted_text = text
    for pii in pii_list:
        if len(pii) > 2:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(pii), re.IGNORECASE)
            redacted_text = pattern.sub("[REDACTED]", redacted_text)
    return redacted_text

# ==========================================
# ORCHESTRATOR EXECUTION
# ==========================================
def run_omni_shield(filepath):
    print("-" * 50)
    file_type, text, boxes = inspect_and_preprocess(filepath)
    
    if not text.strip():
        print("⚠️ No text could be extracted from the file.")
        return
        
    pii_list = extract_pii(text)
    print(f"🎯 Target PII Identified: {pii_list}")
    
    if file_type == "image":
        redact_image(filepath, boxes, pii_list)
    else:
        output_txt = redact_text(text, pii_list)
        out_path = f"redacted_{os.path.basename(filepath)}.txt"
        with open(out_path, "w") as f:
            f.write(output_txt)
        print(f"✅ Saved redacted document to: {out_path}")
    print("-" * 50)

if __name__ == "__main__":
    run_omni_shield("new.jpeg")
    