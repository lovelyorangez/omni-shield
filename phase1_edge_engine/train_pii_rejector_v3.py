import os
import json
import torch
import random
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. Setup Model
MODEL_NAME = "distilbert-base-uncased"
print(f"⏳ Loading Tokenizer & Model ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Labels: 0 = GENERIC (False Positive), 1 = PII (True Positive)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2,
    id2label={0: "GENERIC", 1: "PII"},
    label2id={"GENERIC": 0, "PII": 1}
)

# 2. Extract BALANCED Dataset from Kaggle JSON
print("⏳ Extracting Essay-Specific Context Sentences from Kaggle...")
with open("datasets/kaggle_train.json", "r", encoding="utf-8") as f:
    kaggle_data = json.load(f)

positives = []
negatives = []

# The exact FPs from our diagnostic dump
HARD_NEGATIVES = [
    "Paris", "Canada", "Spain", "Columbia", "Brazil", "Capão", "Barcelona", # Locations
    "George", "Geoff", "John Stuart Mill", "Adam Smith", "Buzan", "Fehr", "Ythier", "Stefano", # Cited Authors
    "Delicatesens", "Storytelling", "Sathyabama", "Credo", "Les Éditions", # Essay Artifacts
    "2021", "2020", "2015", "1999", "student", "teacher", "course", "design thinking"
]

for doc in kaggle_data:
    text = " ".join(doc["tokens"])
    
    # 1. Extract True PII (Label 1)
    current_entity = []
    for token, label in zip(doc["tokens"], doc["labels"]):
        if label != "O":
            current_entity.append(token)
        elif current_entity:
            entity_str = " ".join(current_entity)
            idx = text.find(entity_str)
            if idx != -1:
                context = text[max(0, idx-60): min(len(text), idx+len(entity_str)+60)]
                positives.append(f"{entity_str} [SEP] {context}")
            current_entity = []
            
    # 2. Extract Hard Negatives (Label 0)
    lower_text = text.lower()
    for hn in HARD_NEGATIVES:
        if hn.lower() in lower_text:
            idx = lower_text.find(hn.lower())
            context = text[max(0, idx-60): min(len(text), idx+len(hn)+60)]
            negatives.append(f"{hn} [SEP] {context}")

# Balance the dataset (Match the sizes)
min_size = min(len(positives), len(negatives), 500)
random.shuffle(positives)
random.shuffle(negatives)

positives = positives[:min_size]
negatives = negatives[:min_size]

train_texts = positives + negatives
train_labels = [1] * len(positives) + [0] * len(negatives)

# Shuffle combined dataset
dataset_list = list(zip(train_texts, train_labels))
random.shuffle(dataset_list)
train_texts, train_labels = zip(*dataset_list)

print(f"📊 Training on {len(positives)} PII examples and {len(negatives)} GENERIC examples.")

hf_dataset = Dataset.from_dict({"text": list(train_texts), "label": list(train_labels)})
dataset_split = hf_dataset.train_test_split(test_size=0.2, seed=42)

# 3. Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

print("⏳ Tokenizing Data...")
tokenized_datasets = dataset_split.map(tokenize_function, batched=True)

# 4. Train
output_dir = "./local_pii_rejector"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer
)

print("🚀 Training Essay-Specific Binary Rejector...")
trainer.train()

print(f"💾 Saving to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ Done!")