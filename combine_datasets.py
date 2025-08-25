# combine_datasets.py

import pandas as pd
import sqlite3
import textstat
from transformers import AutoTokenizer
from datetime import datetime
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
HUMAN_DATA_DB = 'reddit_bia_database.db'
SYNTHETIC_DATA_CSV = 'synthetic_data_final.csv'
FINAL_OUTPUT_CSV = 'bia_preprocessed_large.csv'
MODEL_NAME = 'distilbert-base-uncased'

# --- Feature Calculation Functions ---
def calculate_formality(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0
    try:
        reading_ease = 100 - textstat.flesch_reading_ease(text)
        avg_sentence_length = textstat.avg_sentence_length(text)
        return (reading_ease * 0.4) + (avg_sentence_length * 0.6)
    except (ValueError, ZeroDivisionError):
        return 0

# --- Main Script ---
def create_final_dataset():
    # 1. Load Human Data from SQLite DB
    logging.info(f"Loading human data from '{HUMAN_DATA_DB}'...")
    try:
        conn = sqlite3.connect(HUMAN_DATA_DB)
        # We only need user_id and text, we'll generate other fields for consistency
        df_human = pd.read_sql_query("SELECT user_id, text FROM posts", conn)
        conn.close()
        df_human['label'] = 0
        df_human['source'] = 'reddit_scraped'
        # Add a placeholder timestamp
        df_human['timestamp'] = datetime.now().strftime('%Y-%m-%d')
        logging.info(f"Loaded {len(df_human)} human posts.")
    except Exception as e:
        logging.error(f"Failed to load human data. Make sure '{HUMAN_DATA_DB}' exists. Error: {e}")
        return

    # 2. Load Synthetic Data from CSV
    logging.info(f"Loading synthetic data from '{SYNTHETIC_DATA_CSV}'...")
    try:
        df_synthetic = pd.read_csv(SYNTHETIC_DATA_CSV)
        # Add placeholder columns to match
        df_synthetic['timestamp'] = datetime.now().strftime('%Y-%m-%d')
        df_synthetic['source'] = 'claude_generated'
        logging.info(f"Loaded {len(df_synthetic)} synthetic posts.")
    except Exception as e:
        logging.error(f"Failed to load synthetic data. Make sure '{SYNTHETIC_DATA_CSV}' exists. Error: {e}")
        return

    # 3. Combine Datasets
    logging.info("Combining human and synthetic datasets...")
    df_combined = pd.concat([df_human, df_synthetic], ignore_index=True)
    logging.info(f"Total posts in combined dataset: {len(df_combined)}")
    logging.info("\nLabel distribution in new dataset:")
    print(df_combined['label'].value_counts())

    # 4. Run Full Preprocessing Pipeline
    logging.info("\nStarting full preprocessing pipeline...")

    # a. Feature Engineering: Formality Score
    logging.info("Calculating formality scores...")
    df_combined['formality_score'] = df_combined['text'].apply(calculate_formality)

    # b. Feature Engineering: Time Between Posts
    logging.info("Calculating time between posts...")
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'], errors='coerce')
    df_combined = df_combined.sort_values(by=['user_id', 'timestamp'])
    time_diffs = df_combined.groupby('user_id')['timestamp'].diff()
    df_combined['time_between_posts_hours'] = time_diffs.dt.total_seconds() / 3600
    df_combined['time_between_posts_hours'] = df_combined['time_between_posts_hours'].fillna(0)

    # c. Tokenization
    logging.info(f"Loading tokenizer for '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logging.info("Applying tokenizer to all text entries...")
    df_combined['tokens'] = df_combined['text'].apply(
        lambda text: tokenizer.encode(str(text), truncation=True, max_length=512)
    )
    
    # 5. Save Final Preprocessed Dataset
    logging.info(f"Preprocessing complete. Saving final dataset to '{FINAL_OUTPUT_CSV}'...")
    df_combined.to_csv(FINAL_OUTPUT_CSV, index=False)
    logging.info("--- ✅ Final Dataset Ready for Training! ---")

if __name__ == "__main__":
    create_final_dataset()