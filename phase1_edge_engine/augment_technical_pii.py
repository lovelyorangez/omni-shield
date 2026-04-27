import json
import random

def augment_pii_dataset(input_file, output_file):
    """
    Injects synthetic URLs and IDs into documents to fix the 
    low recall observed in error_analysis_diagnostics.py.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    synthetic_urls = [
        "https://linkedin.com/in/student-profile-99",
        "http://github.com/project-repo-alpha",
        "https://portfolio.me/user123",
        "www.personal-blog.io/about"
    ]
    
    synthetic_ids = [
        "ID-992384-X", "ACC-88271-99", "USR_8821_ABC", "V-992031"
    ]

    augmented_count = 0
    for doc in data:
        # Target documents that only have names to add variety
        if "B-NAME_STUDENT" in doc["labels"] and random.random() < 0.3:
            # Append a synthetic URL or ID to the tokens
            new_pii = random.choice(synthetic_urls if random.random() > 0.5 else synthetic_ids)
            doc["tokens"].append("\nContact:")
            doc["labels"].append("O")
            doc["tokens"].append(new_pii)
            
            # Label appropriately for re-training
            label = "B-URL_PERSONAL" if "http" in new_pii or "www" in new_pii else "B-ID_NUM"
            doc["labels"].append(label)
            augmented_count += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    
    print(f"✅ Augmented {augmented_count} documents with technical PII.")

if __name__ == "__main__":
    augment_pii_dataset("datasets/kaggle_train.json", "datasets/kaggle_balanced.json")