import json
import time
import os
import re
import numpy as np

try:
    from backend_server import redact_core_engine
except ImportError:
    print("❌ Could not import backend_server. Ensure you are in the phase1_edge_engine directory.")
    exit()

def clean_text(text):
    return re.sub(r'[^\w\s]', '', str(text).lower()).strip()

def calculate_overlap_diagnostic(ground_truth, predicted):
    matched_truth = set()
    matched_preds = set()
    
    for t in ground_truth:
        for p in predicted:
            if p in matched_preds: continue
            if t in p or p in t:
                matched_truth.add(t)
                matched_preds.add(p)
                break
                
    false_negatives = ground_truth - matched_truth
    false_positives = predicted - matched_preds
    
    return len(matched_truth), list(false_positives), list(false_negatives)

def run_systematic_evaluation(dataset_name, dataset, text_key, entity_key):
    print(f"\n📊 Running Systematic Evaluation on {dataset_name} ({len(dataset)} documents)...")
    print("-------------------------------------------------------------------------")
    
    total_tp, total_fp, total_fn = 0, 0, 0
    latencies = []
    
    all_telemetry = []
    
    for idx, doc in enumerate(dataset):
        start_time = time.time()
        text = doc[text_key]
        
        raw_ground_truth = {clean_text(e): e for e in doc[entity_key] if len(clean_text(e)) > 2}
        ground_truth_cleaned = set(raw_ground_truth.keys())

        # Call with telemetry enabled
        _, predicted_entities, doc_telemetry = redact_core_engine(text, grade="HIGH", return_telemetry=True)
        predicted_clean = set(clean_text(e) for e in predicted_entities if len(clean_text(e)) > 2)

        tp, fps, fns = calculate_overlap_diagnostic(ground_truth_cleaned, predicted_clean)
        
        total_tp += tp
        total_fp += len(fps)
        total_fn += len(fns)
        
        # Attach Ground Truth Status to Telemetry
        for t in doc_telemetry:
            t_clean = clean_text(t["word"])
            t["is_ground_truth"] = any(t_clean in gt or gt in t_clean for gt in ground_truth_cleaned)
            t["is_fp"] = t_clean in fps
            all_telemetry.append(t)
        
        latencies.append(time.time() - start_time)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} documents...")

    avg_latency = np.mean(latencies) * 1000
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n==================================================")
    print(f"🏆 {dataset_name} - RESEARCH METRICS")
    print(f"==================================================")
    print(f"Total True Positives: {total_tp}")
    print(f"Total False Positives: {total_fp}")
    print(f"Total False Negatives: {total_fn}")
    print(f"🎯 Precision:  {precision:.2%}")
    print(f"🔍 Recall:     {recall:.2%}")
    print(f"⭐ F1-Score:   {f1:.2%}")
    print(f"⏱️ Avg Latency: {avg_latency:.2f} ms / document")
    print(f"==================================================")
    
    print("\n🚨 SYSTEMATIC CANDIDATE DUMP (Focusing on Errors) 🚨")
    print(f"{'WORD':<25} | {'TYPE':<5} | {'CONF':<5} | {'VOTES':<5} | {'STATUS':<15} | {'TRUTH?':<6} | {'REASON / LOG'}")
    print("-" * 110)
    
    # Sort to show False Positives and False Negatives first
    error_telemetry = [t for t in all_telemetry if (t["is_fp"] and t["status"] == "REDACTED") or (t["is_ground_truth"] and t["status"] != "REDACTED")]
    
    for t in error_telemetry[:50]: # Show top 50 errors
        word = t['word'][:24]
        is_truth = "YES" if t['is_ground_truth'] else "NO"
        print(f"{word:<25} | {t['type']:<5} | {t['conf']:<5.2f} | {t['votes']:<5} | {t['status']:<15} | {is_truth:<6} | {t['reason']}")

if __name__ == "__main__":
    print("\n🚀 Initializing Omni-Shield Systematic Tuning Suite...")
    
    print("\n⏳ Loading Kaggle PII Dataset from datasets/kaggle_train.json...")
    try:
        with open("datasets/kaggle_train.json", "r", encoding="utf-8") as f:
            kaggle_raw = json.load(f)
            
        kaggle_data = []
        for doc in kaggle_raw[:50]: 
            text = " ".join(doc["tokens"])
            entities, current_entity = [], []
            for token, label in zip(doc["tokens"], doc["labels"]):
                if label != "O": current_entity.append(token)
                elif current_entity:
                    entities.append(" ".join(current_entity))
                    current_entity = []
            if current_entity: entities.append(" ".join(current_entity))
            kaggle_data.append({"text": text, "ground_truth_pii": entities})
            
        run_systematic_evaluation("Kaggle PII Detection", kaggle_data, "text", "ground_truth_pii")
    except Exception as e:
        print(f"❌ Failed to load Kaggle Dataset: {e}")