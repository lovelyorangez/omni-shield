import json
import os
import spacy
import re
import random

# Ensure you have run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spacy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

CUAD_PATH = "datasets/CUAD_v1.json"
OUTPUT_PATH = "datasets/legal_ner_finetune.jsonl"

# CoNLL-03 standard NER tag mapping
TAG2ID = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8
}

# The HARD NEGATIVES we want to force the model to ignore (labeled as 'O')
HARD_NEGATIVES = {
    "agreement", "contract", "company", "corporation", "inc", "llc", "ltd", "party", "parties",
    "section", "article", "state", "court", "law", "act", "notice", "herein", "witness", "whereof",
    "date", "term", "conditions", "business", "services", "client", "provider", "purchaser", "seller",
    "lessee", "lessor", "amendment", "exhibit", "schedule", "appendix", "thereto", "hereby", "undersigned",
    "vendor", "buyer", "owner", "tenant", "landlord", "employee", "employer", "contractor", "subcontractor"
}

def create_dataset(limit=500):
    if not os.path.exists(CUAD_PATH):
        print(f"❌ CUAD dataset not found at {CUAD_PATH}")
        return

    print("⏳ Parsing CUAD for True Legal Entities and Hard Negatives...")
    with open(CUAD_PATH, "r", encoding="utf-8") as f:
        cuad_data = json.load(f)

    training_data = []
    processed_count = 0

    for doc in cuad_data.get("data", []):
        for paragraph in doc.get("paragraphs", []):
            if processed_count >= limit:
                break
                
            text = paragraph.get("context", "")
            if not text: continue

            # 1. Extract true entities from CUAD QA annotations
            true_entities = [] # list of (start_char, end_char, label)
            for qa in paragraph.get("qas", []):
                q_text = qa.get("question", "")
                
                # We mainly care about Parties (ORG/PER) and Dates
                label = "ORG" if "Parties" in q_text else None
                
                if label:
                    for answer in qa.get("answers", []):
                        start = answer.get("answer_start")
                        text_ans = answer.get("text")
                        if start is not None and text_ans:
                            true_entities.append((start, start + len(text_ans), label))

            # 2. Tokenize with SpaCy
            doc_spacy = nlp(text)
            tokens = []
            ner_tags = []

            for token in doc_spacy:
                # Skip pure whitespace tokens
                if not token.text.strip():
                    continue

                tok_start = token.idx
                tok_end = token.idx + len(token.text)
                
                # Determine if token falls inside a True Entity span
                tag = "O"
                for (ent_start, ent_end, label) in true_entities:
                    if tok_start >= ent_start and tok_end <= ent_end:
                        # If it's the very first token of the entity, use B-, else I-
                        if tok_start == ent_start:
                            tag = f"B-{label}"
                        else:
                            tag = f"I-{label}"
                        break
                
                # 3. ENFORCE HARD NEGATIVES
                # If the token was NOT part of a confirmed True Entity, but is a generic legal word,
                # we explicitly ensure it is 'O'. (It defaults to O anyway, but we log it mentally).
                if tag == "O" and token.text.lower() in HARD_NEGATIVES:
                    tag = "O" # Explicitly teaching the model this is NOT an entity

                tokens.append(token.text)
                ner_tags.append(TAG2ID[tag])

            # Only add to dataset if it has actual tokens
            if tokens:
                training_data.append({
                    "tokens": tokens,
                    "ner_tags": ner_tags
                })
                processed_count += 1
                
        if processed_count >= limit:
            break

    # Save as JSONL
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Successfully created {len(training_data)} training examples.")
    print(f"📁 Saved to {OUTPUT_PATH}")
    print("\nSample Output:")
    sample = training_data[0]
    for t, tag in zip(sample["tokens"][:20], sample["ner_tags"][:20]):
        # Reverse lookup tag ID to string for display
        tag_str = [k for k, v in TAG2ID.items() if v == tag][0]
        print(f"{t:15} \t {tag_str}")

if __name__ == "__main__":
    # Create 1000 highly-curated legal paragraphs for fine-tuning
    create_dataset(limit=1000)