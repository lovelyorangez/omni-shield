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

# 1. Setup Model & Tokenizer
# FIX: Switched to bert-base to fit in 8GB VRAM and allow dual-model inference later
MODEL_NAME = "dslim/bert-base-NER"
print(f"⏳ Loading Tokenizer and Base Model ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# 2. Data Parsing & Label Alignment
KAGGLE_TO_CONLL = {
    "O": "O",
    "B-NAME_STUDENT": "B-PER", "I-NAME_STUDENT": "I-PER",
    "B-STREET_ADDRESS": "B-LOC", "I-STREET_ADDRESS": "I-LOC",
    "B-EMAIL": "B-MISC", "I-EMAIL": "I-MISC",
    "B-USERNAME": "B-MISC", "I-USERNAME": "I-MISC",
    "B-ID_NUM": "B-MISC", "I-ID_NUM": "I-MISC",
    "B-PHONE_NUM": "B-MISC", "I-PHONE_NUM": "I-MISC",
    "B-URL_PERSONAL": "B-MISC", "I-URL_PERSONAL": "I-MISC"
}

print("⏳ Processing Kaggle Dataset...")
with open("datasets/kaggle_train.json", "r", encoding="utf-8") as f:
    kaggle_data = json.load(f)

formatted_data = {"tokens": [], "ner_tags": []}
MODEL_LABEL2ID = model.config.label2id

for doc in kaggle_data:
    tokens = doc["tokens"]
    labels = [MODEL_LABEL2ID[KAGGLE_TO_CONLL.get(l, "O")] for l in doc["labels"]]
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

print("⏳ Tokenizing data (max_length=512)...")
tokenized_datasets = dataset_split.map(tokenize_and_align_labels, batched=True)

# 4. Metrics setup
seqeval = evaluate.load("seqeval")
label_list = list(model.config.id2label.values())

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 5. CUSTOM WEIGHTED TRAINER
class KaggleWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        device = logits.device
        class_weights = torch.tensor([1.0, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], device=device)
        
        loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# 6. Training Configuration
output_dir = "./local_kaggle_pii_model"

try:
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,     
        fp16=True,                         
        gradient_checkpointing=True,       # FIX: Drops VRAM usage by selectively clearing memory
        num_train_epochs=4,
        weight_decay=0.01,
        eval_strategy="epoch", 
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_steps=10,
        report_to="none" 
    )
except TypeError:
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,     
        fp16=True,                         
        gradient_checkpointing=True,       # FIX: Drops VRAM usage by selectively clearing memory
        num_train_epochs=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        logging_steps=10,
        report_to="none" 
    )

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

trainer_kwargs = {
    "model": model,
    "args": training_args,
    "train_dataset": tokenized_datasets["train"],
    "eval_dataset": tokenized_datasets["test"],
    "data_collator": data_collator,
    "compute_metrics": compute_metrics,
}

try:
    trainer = KaggleWeightedTrainer(**trainer_kwargs, processing_class=tokenizer)
except TypeError:
    trainer = KaggleWeightedTrainer(**trainer_kwargs, tokenizer=tokenizer)

# 7. Start Training
print("🚀 Starting Fine-Tuning Phase on Kaggle PII...")
trainer.train()

# 8. Save the customized model
print(f"💾 Saving fine-tuned Kaggle Domain model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ Training Complete!")