# create_master_dataset.py

import pandas as pd
import sqlite3
import textstat
from transformers import AutoTokenizer
import logging
from datetime import datetime

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- INPUT FILES ---
FINANCE_HUMAN_DB = 'reddit_bia_database.db'
GENERAL_HUMAN_DB = 'reddit_ood_human_database.db'
SYNTHETIC_DATA_CSV = 'synthetic_data_final.csv' # Using the original training synthetic data

# --- OUTPUT FILE ---
FINAL_OUTPUT_CSV = 'bia_preprocessed_master.csv'
MODEL_NAME = 'distilbert-base-uncased'

# --- Helper Function ---
def calculate_formality(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0
    try:
        # Note: textstat.avg_sentence_length is deprecated, but we use it for consistency with training
        reading_ease = 100 - textstat.flesch_reading_ease(text)
        avg_sentence_length = textstat.avg_sentence_length(text)
        return (reading_ease * 0.4) + (avg_sentence_length * 0.6)
    except (ValueError, ZeroDivisionError):
        return 0

# --- Main Script ---
def create_master_dataset():
    logging.info("--- Creating Master Training Dataset ---")

    # 1. Load All Data Sources
    logging.info(f"Loading financial human data from '{FINANCE_HUMAN_DB}'...")
    conn_fin = sqlite3.connect(FINANCE_HUMAN_DB)
    df_human_fin = pd.read_sql_query("SELECT user_id, text, created_utc FROM posts", conn_fin)
    conn_fin.close()

    logging.info(f"Loading general human data from '{GENERAL_HUMAN_DB}'...")
    conn_gen = sqlite3.connect(GENERAL_HUMAN_DB)
    df_human_gen = pd.read_sql_query("SELECT user_id, text, created_utc FROM posts", conn_gen)
    conn_gen.close()
    
    logging.info(f"Loading synthetic data from '{SYNTHETIC_DATA_CSV}'...")
    df_synthetic = pd.read_csv(SYNTHETIC_DATA_CSV)

    # 2. Standardize and Combine
    df_human_fin['label'] = 0
    df_human_gen['label'] = 0
    df_human = pd.concat([df_human_fin, df_human_gen], ignore_index=True)
    df_human['timestamp'] = pd.to_datetime(df_human['created_utc'], unit='s', errors='coerce')
    df_human = df_human.drop(columns=['created_utc'])

    # Add placeholder timestamp to synthetic data for sorting
    df_synthetic['timestamp'] = pd.to_datetime(datetime.now())

    df_combined = pd.concat([df_human, df_synthetic], ignore_index=True)
    logging.info(f"Combined dataset has {len(df_combined)} total posts.")
    logging.info("\nLabel distribution (by post):")
    print(df_combined['label'].value_counts())

    # 3. Run Full Preprocessing Pipeline
    logging.info("\nStarting full preprocessing pipeline...")
    
    # a. Clean empty text entries
    df_combined.dropna(subset=['text'], inplace=True)
    df_combined = df_combined[df_combined['text'].str.strip() != '']


    # b. Feature Engineering: Formality Score
    logging.info("Calculating formality scores...")
    df_combined['formality_score'] = df_combined['text'].apply(calculate_formality)

    # c. Feature Engineering: Time Between Posts
    logging.info("Calculating time between posts...")
    df_combined = df_combined.sort_values(by=['user_id', 'timestamp'])
    time_diffs = df_combined.groupby('user_id')['timestamp'].diff()
    df_combined['time_between_posts_hours'] = time_diffs.dt.total_seconds() / 3600
    df_combined['time_between_posts_hours'] = df_combined['time_between_posts_hours'].fillna(0)

    # d. Tokenization
    logging.info(f"Loading tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logging.info("Applying tokenizer to all text entries...")
    df_combined['tokens'] = df_combined['text'].apply(
        lambda text: tokenizer.encode(str(text), truncation=True, max_length=512)
    )
    
    # 4. Save Final Preprocessed Dataset
    logging.info(f"Preprocessing complete. Saving master dataset to '{FINAL_OUTPUT_CSV}'...")
    df_combined.to_csv(FINAL_OUTPUT_CSV, index=False)
    logging.info("--- ✅ Master Dataset Ready for Retraining! ---")

if __name__ == "__main__":
    create_master_dataset()