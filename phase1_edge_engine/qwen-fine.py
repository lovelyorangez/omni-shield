import os
import json
import torch
import random
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "./qwen_3b_pii_qlora"

print("\n🚀 Initializing QLoRA Fine-Tuning for Qwen2.5-3B...")

is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if is_bf16_supported else torch.float16

# 1. Load Tokenizer & Quantized Model
print(f"⏳ Loading {MODEL_ID} in 4-bit...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token # Qwen doesn't have a default pad token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto", 
    quantization_config=quantization_config,
    low_cpu_mem_usage=True,
    torch_dtype=compute_dtype
)

# 2. Prepare for QLoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# Target modules for Qwen architecture
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 3. Prepare Kaggle Dataset (70% Train Split)
print("\n⏳ Parsing and formatting Kaggle dataset...")
SYSTEM_PROMPT = """You are an advanced PII Extraction AI. Your strict task is to extract ALL Personal Identifiable Information (PII) from the text. Maximize RECALL – if there’s any doubt, include it. We will filter later.

TARGET CATEGORIES TO EXTRACT (EVERYTHING BELOW):
1. Person Names – Full names AND isolated first names of the author, peers, or anyone mentioned.
2. Email Addresses.
3. Phone Numbers.
4. URLs / Links – ANY web link (http, https, www, YouTube, Facebook, etc.).
5. Identification Numbers – account numbers, IDs like "IV-8322", "Z.S. 30407059", "V69230", "06EYD876".

Return your extracted entities as a valid JSON array of strings. DO NOT provide explanations. If no matches, return exactly: []"""

with open("datasets/kaggle_balanced.json", "r", encoding="utf-8") as f:
    kaggle_raw = json.load(f)

# Filter for documents that actually contain PII to make training dense and efficient
docs_with_pii = [doc for doc in kaggle_raw if any(label != "O" for label in doc["labels"])]

# Consistent shuffle and 70/30 split
random.seed(42)
random.shuffle(docs_with_pii)
split_index = int(len(docs_with_pii) * 0.7)
train_docs = docs_with_pii[:split_index]

formatted_data = []
for doc in train_docs:
    text = " ".join(doc["tokens"])
    
    # Extract Ground Truth
    entities, current_entity = [], []
    for token, label in zip(doc["tokens"], doc["labels"]):
        if label != "O": current_entity.append(token)
        elif current_entity:
            entities.append(" ".join(current_entity))
            current_entity = []
    if current_entity: entities.append(" ".join(current_entity))
    
    # Create Chat Template structure for SFTTrainer
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TEXT: {text[:3500]}"},
        {"role": "assistant", "content": json.dumps(entities)}
    ]
    
    # Apply Qwen's specific chat formatting
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    formatted_data.append({"text": formatted_text})

train_dataset = Dataset.from_list(formatted_data)
print(f"✅ Generated {len(train_dataset)} training examples.")

# 4. Configure Training Arguments
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",              # Moved to SFTConfig in newer trl versions
    per_device_train_batch_size=1,          # Reduced to fit 8GB VRAM
    gradient_accumulation_steps=8,          # Effective batch size = 8
    learning_rate=2e-4,                     # Standard QLoRA LR
    logging_steps=10,
    max_steps=200,                          # Number of training steps (adjust based on needs)
    save_steps=50,
    optim="paged_adamw_8bit",               # Memory efficient optimizer
    fp16=not is_bf16_supported,
    bf16=is_bf16_supported,
    max_grad_norm=0.3,
    warmup_steps=10,                        # Replaced deprecated warmup_ratio
    lr_scheduler_type="cosine",
    report_to="none"
)

# 5. Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    args=training_args,
)

# 6. Start Training
print("\n🔥 Starting QLoRA Fine-Tuning (This will take a while)...")
trainer.train()

# 7. Save Adapter Weights
print(f"\n💾 Saving fine-tuned LoRA adapters to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ QLoRA Training Complete!")