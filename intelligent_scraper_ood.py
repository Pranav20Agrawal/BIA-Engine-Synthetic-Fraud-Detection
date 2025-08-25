# intelligent_scraper_ood.py

import praw
import pandas as pd
import time
import sqlite3
import logging
from datetime import datetime
import textstat
import numpy as np

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# CHANGE 1: New database file for our test data
DB_PATH = 'reddit_ood_human_database.db'
POST_LIMIT_PER_SUB = 100 
MIN_POSTS_PER_USER = 5
# CHANGE 2: Let's just get 50 users for a quick but effective test
TARGET_USER_COUNT = 50

# CHANGE 3: New, non-financial subreddits
SUBREDDITS = [
    'travel', 
    'gadgets', 
    'movies', 
    'gaming', 
    'Cooking',
    'photography',
    'history',
    'DIY'
]

# --- PRAW.INI Authentication ---
# This script assumes you have a valid praw.ini file in this directory
# with a section named [BIA_SCRAPER]

# --- Database Setup ---
def init_database():
    """Initializes the SQLite database and creates the necessary table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS posts (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        subreddit TEXT,
        text TEXT,
        created_utc INTEGER,
        lexical_diversity REAL,
        formality_score REAL,
        caps_ratio REAL
    )
    ''')
    conn.commit()
    return conn

# --- Feature Calculation ---
def calculate_features(text):
    """Calculates advanced features for a given piece of text."""
    if not text or not isinstance(text, str) or len(text.split()) == 0:
        return 0, 0, 0
    
    # Lexical Diversity: Ratio of unique words to total words
    words = text.split()
    lexical_diversity = len(set(words)) / len(words)
    
    # Formality Score (our previous heuristic)
    try:
        reading_ease = 100 - textstat.flesch_reading_ease(text)
        avg_sentence_length = textstat.avg_sentence_length(text)
        formality_score = (reading_ease * 0.4) + (avg_sentence_length * 0.6)
    except:
        formality_score = 0

    # Capitalization Ratio
    caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
    
    return lexical_diversity, formality_score, caps_ratio

# --- Main Scraper Logic ---
def run_scraper():
    conn = init_database()
    cursor = conn.cursor()

    try:
        logging.info("Authenticating with Reddit via praw.ini...")
        reddit = praw.Reddit("BIA_SCRAPER")
        logging.info(f"Authentication successful for user: {reddit.user.me()}")
    except Exception as e:
        logging.error(f"Authentication failed. Ensure praw.ini is correct. Error: {e}")
        return

    # 1. Discover a large pool of active users
    logging.info(f"Discovering active users from {len(SUBREDDITS)} subreddits...")
    active_users = set()
    for sub_name in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub_name)
            for submission in subreddit.hot(limit=POST_LIMIT_PER_SUB):
                if submission.author:
                    active_users.add(submission.author.name)
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:20]: # Top 20 commenters
                    if comment.author:
                        active_users.add(comment.author.name)
            logging.info(f"Found {len(active_users)} unique users so far (from r/{sub_name})...")
            time.sleep(1) # Be polite to the API
        except Exception as e:
            logging.warning(f"Could not process r/{sub_name}: {e}")

    # 2. Collect post history for the discovered users
    logging.info(f"\nFound a pool of {len(active_users)} total users. Fetching histories...")
    collected_user_count = 0
    for i, username in enumerate(list(active_users)):
        if collected_user_count >= TARGET_USER_COUNT:
            logging.info("Target user count reached. Stopping collection.")
            break
        
        try:
            redditor = reddit.redditor(username)
            user_posts = []
            # Combine submissions and comments into one list
            content = list(redditor.submissions.new(limit=50)) + list(redditor.comments.new(limit=50))
            
            if len(content) < MIN_POSTS_PER_USER:
                continue # Skip users with not enough history

            logging.info(f"({i+1}/{len(active_users)}) Collecting {len(content)} posts for user '{username}'...")
            for item in content:
                text = item.selftext if hasattr(item, 'selftext') else item.body
                lex_div, form_score, caps_rat = calculate_features(text)
                
                cursor.execute(
                    "INSERT OR IGNORE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (item.id, username, item.subreddit.display_name, text, int(item.created_utc), lex_div, form_score, caps_rat)
                )
            
            conn.commit()
            collected_user_count += 1
            time.sleep(1) # API politeness

        except Exception as e:
            logging.warning(f"Could not fetch history for '{username}': {e}")

    logging.info(f"\nScraping complete. Stored data for {collected_user_count} users in '{DB_PATH}'.")
    conn.close()

if __name__ == "__main__":
    run_scraper()