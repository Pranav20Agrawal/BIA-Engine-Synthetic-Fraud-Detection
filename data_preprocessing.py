# data_preprocessing.py

# Import the pandas library, which is the standard for working with tabular data in Python.
# We use the alias 'pd' by convention.
import pandas as pd
import textstat
from transformers import AutoTokenizer

# --- 1. LOAD THE DATASET ---
# We use the read_csv function from pandas to load our dataset into a DataFrame.
# A DataFrame is the primary data structure in pandas, similar to a table or spreadsheet.
print("Loading bia_dataset.csv...")
df = pd.read_csv('bia_dataset.csv')


# --- 2. EXPLORATORY DATA ANALYSIS (EDA) ---
# It's crucial to understand our data before we start working with it.

print("\n--- Initial Data Exploration ---")

# Display the first 5 rows of the DataFrame to get a quick look at the data.
print("\n[INFO] First 5 rows of the dataset:")
print(df.head())

# Use the .info() method to get a concise summary of the DataFrame.
# This is great for checking column data types and looking for any missing values.
print("\n[INFO] Dataset summary:")
df.info()

# Check the distribution of our labels (0 for human, 1 for synthetic).
# This is very important to see if our dataset is balanced or imbalanced.
print("\n[INFO] Label distribution:")
print(df['label'].value_counts())


# --- 3. FEATURE ENGINEERING ---
# This is where we create our custom behavioral features from the raw data.

print("\n--- Feature Engineering ---")

# Convert the 'timestamp' column from a string to a proper datetime object.
# This allows us to perform calculations with the dates, which is essential for our next step.
# The 'errors='coerce'' argument will turn any unparseable dates into 'NaT' (Not a Time).
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Sort the DataFrame first by user_id and then by timestamp.
# This is critical to ensure that when we calculate the time difference,
# we are comparing consecutive posts from the SAME user in the correct order.
df = df.sort_values(by=['user_id', 'timestamp'])

# Calculate the time difference between consecutive posts for each user.
# - df.groupby('user_id'): This groups the DataFrame so that calculations are performed independently for each user.
# - ['timestamp'].diff(): This calculates the difference between the current row's timestamp and the previous one within its group.
# The first post for each user will have 'NaT' (Not a Time) because there's no previous post to compare to.
time_diffs = df.groupby('user_id')['timestamp'].diff()

# Create a new column 'time_between_posts_hours'.
# We convert the time difference (which is a Timedelta object) into total hours.
# .dt.total_seconds() gets the difference in seconds, and we divide by 3600 to get hours.
df['time_between_posts_hours'] = time_diffs.dt.total_seconds() / 3600

# Fill the missing values (the first post of each user) with 0.
# For our model, we can treat the first post as having a 0-hour difference from a non-existent previous post.
df['time_between_posts_hours'] = df['time_between_posts_hours'].fillna(0)


# --- 4. DISPLAY RESULTS ---
# Let's look at our DataFrame again to see the new feature we created.

print("\n[SUCCESS] Feature 'time_between_posts_hours' created.")
print("\n[INFO] First 10 rows with the new feature:")
# We'll display more rows this time to see how the calculation works across different users.
print(df.head(10))

print("\n--- Engineering Formality Score ---")

# --- 5. CALCULATE FORMALITY SCORE ---
# We'll create a simple heuristic for formality based on text statistics.
# textstat makes it easy to get metrics like average sentence length and syllable count.

def calculate_formality(text):
    """
    Calculates a heuristic-based formality score for a given text.
    The score is a combination of average sentence length and reading difficulty.
    A higher score suggests higher formality.
    """
    try:
        # We'll use the Flesch Reading Ease score. A lower score means the text is harder to read (more formal).
        # We invert it (100 - score) so that a higher value corresponds to higher formality.
        reading_ease = 100 - textstat.flesch_reading_ease(text)

        # We'll also use average sentence length. Longer sentences are often more formal.
        avg_sentence_length = textstat.avg_sentence_length(text)

        # Combine these two metrics. The weights (0.4 and 0.6) are chosen to balance the two measures.
        # This is a simple formula, but effective for detecting changes in style.
        formality_score = (reading_ease * 0.4) + (avg_sentence_length * 0.6)
        
        return formality_score
    except (ValueError, ZeroDivisionError):
        # Handle cases where the text is too short or malformed for textstat to analyze.
        return 0

# Apply our new function to the 'text' column to create the 'formality_score' column.
# The .apply() method runs our function on every single row in the 'text' column.
df['formality_score'] = df['text'].apply(calculate_formality)

# --- 6. DISPLAY FINAL RESULTS ---
# Let's view the DataFrame with both of our new engineered features.

print("\n[SUCCESS] Feature 'formality_score' created.")
print("\n[INFO] Dataframe with all engineered features:")

# To keep the output clean, we'll select and display only the most important columns.
# This helps us focus on the user, the label, and our new behavioral features.
print(df[['user_id', 'label', 'time_between_posts_hours', 'formality_score']].head(15))

print("\n--- Tokenizing Text for BERT ---")

# --- 7. INITIALIZE THE TOKENIZER ---
# We specify the model we plan to use ('distilbert-base-uncased').
# This downloads the corresponding vocabulary and rules for that specific model.
# It's critical that the tokenizer and the model are an exact match.
MODEL_NAME = 'distilbert-base-uncased'
print(f"Loading tokenizer for model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# --- 8. TOKENIZE THE TEXT DATA ---
# We create a new column called 'tokens'.
# For each piece of text in our 'text' column, we will apply the tokenizer.
# The tokenizer.encode() function handles the entire process:
#   1. Breaking text into tokens (words/sub-words).
#   2. Converting tokens into their corresponding integer IDs from the model's vocabulary.
#   - truncation=True: If a post is longer than the model's max length, it will be cut short.
#   - max_length=512: The standard maximum sequence length for most BERT-style models.
print("Applying tokenizer to all text entries...")
df['tokens'] = df['text'].apply(
    lambda text: tokenizer.encode(text, truncation=True, max_length=512)
)


# --- 9. DISPLAY FINAL PREPROCESSED DATA ---
print("\n[SUCCESS] Tokenization complete.")
print("\n[INFO] Final preprocessed dataframe sample:")

# Display the key columns, including our new 'tokens' column.
# You'll see that the raw text has now been converted into lists of numbers!
print(df[['user_id', 'label', 'tokens', 'time_between_posts_hours', 'formality_score']].head())

# Save the preprocessed data to a new CSV file for the next stage.
# This way, we don't have to run this entire script every time.
print("\nSaving preprocessed data to 'bia_preprocessed_dataset.csv'...")
df.to_csv('bia_preprocessed_dataset.csv', index=False)
print("[SUCCESS] File saved.")