import os
import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
import evaluate

# 1. Load the generated dataset
print("⏳ Loading dataset...")
dataset = load_dataset('json', data_files='datasets/legal_ner_finetune.jsonl', split='train')

# Split into 90% train, 10% validation
dataset = dataset.train_test_split(test_size=0.1)

# 2. Define CoNLL-03 Labels
LABEL_LIST = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
id2label = {i: label for i, label in enumerate(LABEL_LIST)}
label2id = {label: i for i, label in enumerate(LABEL_LIST)}

# 3. Setup Domain-Specific Legal Model & Tokenizer
# We are using a model pre-trained on legal documents (contracts, laws, court cases)
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
print(f"⏳ Loading Tokenizer and Base Model ({MODEL_NAME})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Because this is a BASE language model (not already fine-tuned for NER), 
# we initialize a fresh classification head dynamically set to our 9 label types.
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=id2label,
    label2id=label2id
)

# 4. Tokenization & Alignment
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=256
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # Ignore special tokens (CLS, SEP)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) # Only label the first token of a given word
            else:
                label_ids.append(-100) # Ignore subword continuations
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

print("⏳ Tokenizing data...")
tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# 5. Metrics setup
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
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# 6. CUSTOM WEIGHTED TRAINER
class WeightedNERTrainer(Trainer):
    """
    Subclassing Trainer to inject a custom CrossEntropyLoss.
    We severely penalize False Positives by forcing the model to favor the 'O' class.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Weights: 'O' gets 1.0 (High Importance). All PII gets 0.70 (Lower Confidence Required).
        # This mathematically restricts the model from hallucinating PII unless it is VERY sure.
        device = logits.device
        class_weights = torch.tensor([1.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7], device=device)
        
        # Apply weights, ignoring the -100 padding index
        loss_fct = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
        
        # Flatten logits and labels to compute loss
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

# 7. Training Configuration
output_dir = "./local_legal_ner_model"

try:
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5, # Slightly higher LR for training a fresh head
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5, # 5 Epochs
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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
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

# Version-Agnostic initialization
try:
    trainer = WeightedNERTrainer(**trainer_kwargs, processing_class=tokenizer)
except TypeError:
    trainer = WeightedNERTrainer(**trainer_kwargs, tokenizer=tokenizer)

# 8. Start Training
print("🚀 Starting Fine-Tuning Phase on LEGAL-BERT (With Custom Loss Penalty)...")
trainer.train()

# 9. Save the customized model
print(f"💾 Saving fine-tuned Legal-Domain model to {output_dir}...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ Training Complete! Your edge engine will now automatically load this specialized model.")