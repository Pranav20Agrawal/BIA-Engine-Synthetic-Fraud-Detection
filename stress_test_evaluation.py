# stress_test_evaluation.py (Corrected)

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import textstat
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import warnings
from datetime import datetime

# Suppress the deprecation warning for a cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# We need to import all the classes and functions from our final training script
from final_training import BIADataset, collate_fn_v2, BIA_Model_v2

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CHALLENGE_DATASET_PATH = 'challenge_dataset_balanced.csv'
MODEL_PATH = 'bia_model_master.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8

# --- Helper Functions ---
def calculate_formality(text):
    if not isinstance(text, str) or len(text.strip()) == 0: return 0
    try:
        reading_ease = 100 - textstat.flesch_reading_ease(text)
        avg_sentence_length = textstat.avg_sentence_length(text)
        return (reading_ease * 0.4) + (avg_sentence_length * 0.6)
    except: return 0

# --- Main Evaluation Script ---
if __name__ == "__main__":
    print(f"--- Stress-Testing Final Model on OOD Data: {CHALLENGE_DATASET_PATH} ---")
    
    try:
        df = pd.read_csv(CHALLENGE_DATASET_PATH)
    except FileNotFoundError:
        print(f"ERROR: Challenge dataset '{CHALLENGE_DATASET_PATH}' not found. Please create it first.")
        exit()

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    df['formality_score'] = df['text'].apply(calculate_formality)
    df['timestamp'] = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    df = df.sort_values(by=['user_id', 'timestamp'])
    time_diffs = df.groupby('user_id')['timestamp'].diff()
    df['time_between_posts_hours'] = time_diffs.dt.total_seconds() / 3600
    df['time_between_posts_hours'] = df['time_between_posts_hours'].fillna(0)
    df['tokens'] = df['text'].apply(lambda text: tokenizer.encode(str(text), truncation=True, max_length=512))
    
    sequences = df.groupby('user_id').agg({
        'tokens': list, 'time_between_posts_hours': list,
        'formality_score': list, 'label': lambda x: x.iloc[0]
    }).reset_index()
    sequences['numerical_features'] = sequences.apply(lambda r: list(zip(r['time_between_posts_hours'], r['formality_score'])), axis=1)

    challenge_dataset = BIADataset(sequences)
    challenge_dataloader = DataLoader(challenge_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_v2)
    
    model = BIA_Model_v2(num_numerical_features=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(challenge_dataloader, desc="Evaluating Challenge Set"):
            input_ids, attn_mask, num_feat, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['numerical_features'].to(DEVICE), batch['labels'].to(DEVICE)
            outputs = model(input_ids, attn_mask, num_feat)
            preds = torch.sigmoid(outputs).squeeze() > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n--- Final Stress-Test Performance Report ---")
    # FIX: Add the `labels` parameter to handle cases where one class is not present in the results.
    print(classification_report(all_labels, all_preds, target_names=['Human (0)', 'Synthetic (1)'], labels=[0, 1], zero_division=0))
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Human', 'Synthetic'], yticklabels=['Human', 'Synthetic'])
    plt.title('Stress-Test Confusion Matrix (OOD Data)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()