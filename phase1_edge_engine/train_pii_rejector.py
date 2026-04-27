import os
import json
import numpy as np
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
import evaluate

# 1. Setup Lightweight Model & Tokenizer
# We use DistilBERT for maximum speed during the Stage 2 filtering phase.
MODEL_NAME = "distilbert-base-cased"
print(f"⏳ Loading Lightweight Base Model ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define Binary Labels
LABEL_LIST = ["O", "PII"]
id2label = {0: "O", 1: "PII"}
label2id = {"O": 0, "PII": 1}

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# 2. Binary Data Parsing
# Map all Kaggle tags to a simple "PII" label.
KAGGLE_TO_BINARY = {
    "O": "O",
    "B-NAME_STUDENT": "PII", "I-NAME_STUDENT": "PII",
    "B-STREET_ADDRESS": "PII", "I-STREET_ADDRESS": "PII",
    "B-EMAIL": "PII", "I-EMAIL": "PII",
    "B-USERNAME": "PII", "I-USERNAME": "PII",
    "B-ID_NUM": "PII", "I-ID_NUM": "PII",
    "B-PHONE_NUM": "PII", "I-PHONE_NUM": "PII",
    "B-URL_PERSONAL": "PII", "I-URL_PERSONAL": "PII"
}

print("⏳ Processing Kaggle Dataset for Binary Classification...")
with open("datasets/kaggle_train.json", "r", encoding="utf-8") as f:
    kaggle_data = json.load(f)

formatted_data = {"tokens": [], "ner_tags": []}

for doc in kaggle_data:
    tokens = doc["tokens"]
    labels = [label2id[KAGGLE_TO_BINARY.get(l, "O")] for l in doc["labels"]]
    formatted_data["tokens"].append(tokens)
    formatted_data["ner_tags"].append(labels)

hf_dataset = Dataset.from_dict(formatted_data)
dataset_split = hf_dataset.train_test_split(test_size=0.2, seed=42)

# 3. Tokenization & Alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=512
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("⏳ Tokenizing data...")
tokenized_datasets = dataset_split.map(tokenize_and_align_labels, batched=True)

# 4. Metrics Setup
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"]
    }

# 5. Custom Trainer for Strict FP Penalty
class BinaryWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        device = logits.device
        # Weight O at 1.0, and PII at 0.5 to keep the rejector highly skeptical
        class_weights = torch.tensor([1.0, 0.5], device=device)
        
        loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# 6. Training Configuration
output_dir = "./local_pii_rejector"

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True, 
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch", 
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_steps=10,
    report_to="none"
)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer = BinaryWeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer
)

# 7. Start Training
print("🚀 Starting Fine-Tuning Phase on Binary Rejector...")
trainer.train()

# 8. Save Model
print(f"💾 Saving fine-tuned Stage 2 Rejector to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ Training Complete!")