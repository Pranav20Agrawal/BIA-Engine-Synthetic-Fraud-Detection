# model_training_v2.py (Sequence-Aware Model with LSTM)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import ast
import numpy as np
from torch.nn.utils.rnn import pad_sequence

print("--- BIA Model v2.0 Training Script (Sequence-Aware) ---")

# Define a constant for sequence length
MAX_SEQ_LENGTH = 5 # Model will look at the last 5 posts

# --- 1. LOAD AND STRUCTURE THE DATA (UNCHANGED) ---
print("Loading preprocessed data...")
df = pd.read_csv('bia_preprocessed_dataset.csv')
df['tokens'] = df['tokens'].apply(ast.literal_eval)

print("Grouping posts into user sequences...")
sequences = df.groupby('user_id').agg({
    'tokens': list,
    'time_between_posts_hours': list,
    'formality_score': list,
    'label': lambda x: x.iloc[0]
}).reset_index()

sequences['numerical_features'] = sequences.apply(
    lambda row: list(zip(row['time_between_posts_hours'], row['formality_score'])),
    axis=1
)

# --- 2. SPLIT DATA (UNCHANGED) ---
print("\nSplitting data into training and validation sets...")
train_df, val_df = train_test_split(
    sequences, test_size=0.2, random_state=42, stratify=sequences['label']
)

# --- 3. PYTORCH DATASET CLASS (UNCHANGED) ---
class BIADataset(Dataset):
    def __init__(self, dataframe):
        self.sequences = dataframe

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences.iloc[idx]
        return {
            'tokens': [torch.tensor(t) for t in item['tokens']],
            'numerical_features': torch.tensor(item['numerical_features'], dtype=torch.float32),
            'label': torch.tensor(item['label'], dtype=torch.float32)
        }

# --- 4. UPDATED COLLATE FUNCTION for SEQUENCES ---
# This function is now more complex. It pads both tokens within each post AND posts within each sequence.
def collate_fn_v2(batch):
    labels = torch.stack([item['label'] for item in batch])
    
    # --- Sequence Padding/Truncation ---
    # Ensure every user sequence has exactly MAX_SEQ_LENGTH posts.
    all_user_tokens = []
    all_user_numericals = []

    for item in batch:
        user_tokens = item['tokens']
        user_numericals = item['numerical_features']

        # Pad or truncate the number of posts
        if len(user_tokens) > MAX_SEQ_LENGTH: # Truncate (take the most recent posts)
            user_tokens = user_tokens[-MAX_SEQ_LENGTH:]
            user_numericals = user_numericals[-MAX_SEQ_LENGTH:]
        elif len(user_tokens) < MAX_SEQ_LENGTH: # Pad with empty posts
            num_to_pad = MAX_SEQ_LENGTH - len(user_tokens)
            # Pad tokens with a tensor containing only the [PAD] token
            user_tokens += [torch.tensor([0])] * num_to_pad 
            # Pad numericals with a tensor of zeros
            user_numericals = torch.cat([user_numericals, torch.zeros(num_to_pad, 2)], dim=0)
        
        all_user_tokens.append(user_tokens)
        all_user_numericals.append(user_numericals)

    # --- Token Padding ---
    # Now pad the tokens within each post so they all have the same length in the batch.
    padded_posts = []
    for user_tokens in all_user_tokens:
        padded_user_posts = pad_sequence(user_tokens, batch_first=True, padding_value=0)
        padded_posts.append(padded_user_posts)

    # We now have a list of tensors of different shapes. We need to pad them again to the max token length in the whole batch.
    max_token_len = max(p.shape[1] for p in padded_posts)
    
    final_padded_tokens = torch.zeros(len(batch), MAX_SEQ_LENGTH, max_token_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), MAX_SEQ_LENGTH, max_token_len, dtype=torch.float32)

    for i, p in enumerate(padded_posts):
        final_padded_tokens[i, :, :p.shape[1]] = p
        attention_mask[i, :, :p.shape[1]] = (p != 0).float()
        
    return {
        'input_ids': final_padded_tokens,
        'attention_mask': attention_mask,
        'numerical_features': torch.stack(all_user_numericals),
        'labels': labels
    }

# --- 5. DEFINE THE BIA ENGINE v2.0 ARCHITECTURE ---
class BIA_Model_v2(nn.Module):
    def __init__(self, num_numerical_features, bert_model_name='distilbert-base-uncased'):
        super(BIA_Model_v2, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # --- NEW: LSTM Layer ---
        # This layer will process the sequence of post embeddings from BERT.
        # input_size is the size of a BERT embedding (768).
        # hidden_size is the size of the LSTM's memory cell.
        # batch_first=True tells the LSTM to expect tensors with the batch dimension first.
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, 
                              hidden_size=256, 
                              num_layers=1, 
                              batch_first=True)
        
        self.numerical_processor = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
        )
        
        # The classifier now takes input from the LSTM and the aggregated numerical features.
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256), # LSTM output (256) + Numerical output (128)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, numerical_features):
        batch_size, num_posts, seq_len = input_ids.shape
        
        # Reshape for BERT: (batch_size * num_posts, seq_len)
        input_ids = input_ids.view(-1, seq_len)
        attention_mask = attention_mask.view(-1, seq_len)
        
        # Get BERT embeddings for all posts in the batch at once.
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        post_embeddings = bert_output.last_hidden_state[:, 0]
        
        # Reshape back to (batch_size, num_posts, embedding_size)
        post_embeddings = post_embeddings.view(batch_size, num_posts, -1)
        
        # Process sequence of post embeddings with LSTM
        lstm_output, (h_n, c_n) = self.lstm(post_embeddings)
        
        # We take the final hidden state from the LSTM as the sequence representation.
        sequence_embedding = h_n.squeeze(0)
        
        # Process numerical features (we'll take the mean across all posts)
        numerical_embedding = self.numerical_processor(numerical_features.mean(dim=1))
        
        # Combine sequence and numerical embeddings
        combined_embedding = torch.cat((sequence_embedding, numerical_embedding), dim=1)
        
        # Final classification
        logits = self.classifier(combined_embedding)
        return logits

# --- 6. SET UP TRAINING (UNCHANGED, but with new model and collate_fn) ---
print("\nSetting up for training v2.0...")
NUM_EPOCHS = 20 # More complex model might need more epochs
BATCH_SIZE = 4
LEARNING_RATE = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_dataset = BIADataset(train_df)
val_dataset = BIADataset(val_df)

# Use the new v2 collate function
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_v2)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_v2)

# Use the new v2 model
model = BIA_Model_v2(num_numerical_features=2).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_function = nn.BCEWithLogitsLoss()

# --- 7. THE TRAINING LOOP (UNCHANGED) ---
print("\n--- Starting Training v2.0 ---")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numerical_features = batch['numerical_features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, numerical_features)
        loss = loss_function(outputs.squeeze(), labels)
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, numerical_features)
            loss = loss_function(outputs.squeeze(), labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_dataloader)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# --- 8. SAVE THE TRAINED MODEL ---
MODEL_SAVE_PATH = 'bia_model_v2.pth'
print(f"\nTraining complete. Saving model to {MODEL_SAVE_PATH}...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("--- Model v2.0 Saved Successfully ---")