import os
import json
import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 1. Setup Model
MODEL_NAME = "distilbert-base-uncased"
print(f"⏳ Loading Tokenizer & Model ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2,
    id2label={0: "GENERIC", 1: "PII"},
    label2id={"GENERIC": 0, "PII": 1}
)

# 2. Extract Data from Kaggle JSON
print("⏳ Mining Hard Negatives & Hard Positives from Kaggle Dataset...")
with open("datasets/kaggle_train.json", "r", encoding="utf-8") as f:
    kaggle_data = json.load(f)

positives = []
negatives = []

# The Ultimate Hitlist (Everything the model falsely thinks is PII)
HARD_NEGATIVES = [
    "Washington", "YouTube", "Swiffer", "Garda", "Bachhpan", "A Smith",
    "Paris", "Canada", "Spain", "Columbia", "Brazil", "Capão", "Barcelona", "India", "Ludhiana",
    "George", "Geoff", "John Stuart Mill", "Adam Smith", "Buzan", "Fehr", "Ythier", "Stefano Lovato", 
    "Alexander Shmakov", "Gerashchenko", "Angela Meyer", "Sakir Ahmad", "Kazantseva", "Kolm", "Andreoni",
    "Delicatesens", "Storytelling", "Sathyabama", "Credo", "Les Éditions", "COVID-19", "Raspberry Pi", 
    "Marias Gamesa", "A & R", "MDI", "Coursera", "Elsevier",
    "2021", "2020", "2015", "1999", "student", "teacher", "course", "design thinking"
]

# The Tricky Initials & Fragments (Force the model to learn these ARE PII)
HARD_POSITIVES = [
    "Sin", "Sam", "De", "P", "L", "Nat", "T", "S", "El", "Sa", "Estra", "Far", "Kara", "Luis Rama", "Milton Des"
]

for doc in kaggle_data:
    text = " ".join(doc["tokens"])
    lower_text = text.lower()
    
    # 1. Extract True PII (Label 1)
    current_entity = []
    for token, label in zip(doc["tokens"], doc["labels"]):
        if label != "O":
            current_entity.append(token)
        elif current_entity:
            entity_str = " ".join(current_entity)
            idx = text.find(entity_str)
            if idx != -1:
                context = text[max(0, idx-80): min(len(text), idx+len(entity_str)+80)]
                positives.append(f"{entity_str} [SEP] {context}")
            current_entity = []
            
    # 2. Inject Hard Positives (Label 1)
    for hp in HARD_POSITIVES:
        if hp in text:
            idx = text.find(hp)
            context = text[max(0, idx-80): min(len(text), idx+len(hp)+80)]
            positives.append(f"{hp} [SEP] {context}")
            
    # 3. Hunt for Hard Negatives (Label 0)
    for hn in HARD_NEGATIVES:
        if hn.lower() in lower_text:
            idx = lower_text.find(hn.lower())
            context = text[max(0, idx-80): min(len(text), idx+len(hn)+80)]
            negatives.append(f"{hn} [SEP] {context}")

# Balance the dataset
min_size = min(len(positives), len(negatives), 800)
random.shuffle(positives)
random.shuffle(negatives)

positives = positives[:min_size]
negatives = negatives[:min_size]

train_texts = positives + negatives
train_labels = [1] * len(positives) + [0] * len(negatives)

dataset_list = list(zip(train_texts, train_labels))
random.shuffle(dataset_list)
train_texts, train_labels = zip(*dataset_list)

print(f"📊 Training on {len(positives)} TRUE PII and {len(negatives)} HARD NEGATIVES.")

hf_dataset = Dataset.from_dict({"text": list(train_texts), "label": list(train_labels)})
dataset_split = hf_dataset.train_test_split(test_size=0.2, seed=42)

# 3. Tokenize
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset_split.map(tokenize_function, batched=True)

# 4. Train
output_dir = "./local_pii_rejector"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-5,
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

print("🚀 Forcing Rejector to learn the False Positives and Rescue Initials...")
trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ Rejector successfully re-educated!")