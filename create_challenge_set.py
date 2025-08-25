# create_challenge_set.py

import pandas as pd
import sqlite3
import textstat
from transformers import AutoTokenizer
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
HUMAN_DATA_DB = 'reddit_ood_human_database.db'
SYNTHETIC_CHALLENGE_CSV = 'challenge_dataset.csv'
FINAL_OUTPUT_CSV = 'challenge_dataset_balanced.csv'
MODEL_NAME = 'distilbert-base-uncased'

def calculate_formality(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0
    try:
        reading_ease = 100 - textstat.flesch_reading_ease(text)
        avg_sentence_length = textstat.avg_sentence_length(text)
        return (reading_ease * 0.4) + (avg_sentence_length * 0.6)
    except (ValueError, ZeroDivisionError):
        return 0

logging.info("--- Creating Balanced Challenge Dataset ---")

# 1. Load and Process New Human Data
logging.info(f"Loading OOD human data from '{HUMAN_DATA_DB}'...")
conn = sqlite3.connect(HUMAN_DATA_DB)
df_human = pd.read_sql_query("SELECT user_id, text, created_utc FROM posts", conn)
conn.close()

# Convert timestamp and add label
df_human['timestamp'] = pd.to_datetime(df_human['created_utc'], unit='s')
df_human['label'] = 0
df_human = df_human.drop(columns=['created_utc'])
logging.info(f"Loaded {len(df_human['user_id'].unique())} unique human users.")

# Preprocess human data
df_human['formality_score'] = df_human['text'].apply(calculate_formality)
df_human = df_human.sort_values(by=['user_id', 'timestamp'])
time_diffs = df_human.groupby('user_id')['timestamp'].diff()
df_human['time_between_posts_hours'] = time_diffs.dt.total_seconds() / 3600
df_human['time_between_posts_hours'] = df_human['time_between_posts_hours'].fillna(0)

logging.info(f"Tokenizing human data with '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
df_human['tokens'] = df_human['text'].apply(
    lambda text: tokenizer.encode(str(text), truncation=True, max_length=512)
)

# 2. Load Synthetic Challenge Data
logging.info(f"Loading synthetic data from '{SYNTHETIC_CHALLENGE_CSV}'...")
df_synthetic = pd.read_csv(SYNTHETIC_CHALLENGE_CSV)

# 3. Combine and Save
logging.info("Combining datasets...")
# For a fair test, let's ensure we have a similar number of users
human_users = df_human['user_id'].unique()
synthetic_users = df_synthetic['user_id'].unique()

# We might have more than 50 human users, let's sample 50 to match the synthetic set
if len(human_users) > len(synthetic_users):
    sampled_human_users = pd.Series(human_users).sample(n=len(synthetic_users), random_state=42).tolist()
    df_human_final = df_human[df_human['user_id'].isin(sampled_human_users)]
else:
    df_human_final = df_human

# Combine the final human data with all synthetic data
df_balanced_challenge = pd.concat([df_human_final, df_synthetic], ignore_index=True)

df_balanced_challenge.to_csv(FINAL_OUTPUT_CSV, index=False)
logging.info(f"--- ✅ Balanced challenge set saved to '{FINAL_OUTPUT_CSV}' ---")
logging.info("Label distribution in new challenge set:")
print(df_balanced_challenge.groupby('user_id')['label'].first().value_counts())