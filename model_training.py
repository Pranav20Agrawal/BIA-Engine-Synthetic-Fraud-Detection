# model_training.py (Final Version)

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

print("--- BIA Model Training Script ---")

# --- 1. LOAD AND STRUCTURE THE DATA INTO SEQUENCES ---
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

# --- 2. SPLIT DATA INTO TRAINING AND VALIDATION SETS ---
print("\nSplitting data into training and validation sets...")
train_df, val_df = train_test_split(
    sequences, test_size=0.2, random_state=42, stratify=sequences['label']
)

# --- 3. CREATE THE PYTORCH DATASET CLASS ---
class BIADataset(Dataset):
    def __init__(self, dataframe):
        self.sequences = dataframe

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = self.sequences.iloc[idx]
        return {
            'user_id': item['user_id'],
            'tokens': [torch.tensor(t) for t in item['tokens']], # Convert to tensors here
            'numerical_features': torch.tensor(item['numerical_features'], dtype=torch.float32),
            'label': torch.tensor(item['label'], dtype=torch.float32)
        }

# --- 4. DEFINE THE CUSTOM COLLATE FUNCTION ---
# This function is our "custom packer" for creating batches.
def collate_fn(batch):
    # Separate the different parts of the data
    user_ids = [item['user_id'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    numerical_features = torch.stack([item['numerical_features'][0] for item in batch]) # Assuming 1 post for now for simplicity
    
    # --- Padding Logic for Tokens ---
    # We'll just take the first post's tokens for this simplified example
    # A more advanced version would handle multiple posts per user
    token_lists = [item['tokens'][0] for item in batch]
    
    # pad_sequence is a PyTorch utility that pads a list of sequences to the same length.
    # batch_first=True means the output shape will be (batch_size, sequence_length).
    # padding_value=0 is the ID for the [PAD] token in most tokenizers.
    padded_tokens = pad_sequence(token_lists, batch_first=True, padding_value=0)
    
    # Create an attention mask. This tells BERT to ignore the padded parts of the sequence.
    # It's a tensor of 1s (for real tokens) and 0s (for padding).
    attention_mask = (padded_tokens != 0).type(torch.float32)

    return {
        'user_id': user_ids,
        'input_ids': padded_tokens,
        'attention_mask': attention_mask,
        'numerical_features': numerical_features,
        'labels': labels
    }


# --- 5. DEFINE THE CUSTOM MODEL ARCHITECTURE ---
class BIA_Model(nn.Module):
    def __init__(self, num_numerical_features, bert_model_name='distilbert-base-uncased'):
        super(BIA_Model, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.numerical_processor = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, numerical_features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = bert_output.last_hidden_state[:, 0]
        numerical_embedding = self.numerical_processor(numerical_features)
        combined_embedding = torch.cat((text_embedding, numerical_embedding), dim=1)
        logits = self.classifier(combined_embedding)
        return logits

# --- 6. SET UP TRAINING ---
print("\nSetting up for training...")
# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
MODEL_NAME = 'distilbert-base-uncased'

# Check if a GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate Datasets
train_dataset = BIADataset(train_df)
val_dataset = BIADataset(val_df)

# Instantiate DataLoaders
# This is our automated assembly line that uses the collate_fn to create perfect batches.
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Instantiate Model
model = BIA_Model(num_numerical_features=2).to(device)

# Instantiate Optimizer and Loss Function
# AdamW is an optimizer well-suited for transformer models.
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
# BCEWithLogitsLoss is perfect for binary classification. It's numerically stable.
loss_function = nn.BCEWithLogitsLoss()

# --- 7. THE TRAINING LOOP ---
print("\n--- Starting Training ---")
for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train() # Put the model in training mode
    total_train_loss = 0
    for batch in train_dataloader:
        # Move the batch data to the selected device (GPU/CPU)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        numerical_features = batch['numerical_features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad() # Reset gradients from the previous step

        # Get model predictions (the "student's answer")
        outputs = model(input_ids, attention_mask, numerical_features)
        
        # Calculate the error/loss
        loss = loss_function(outputs.squeeze(), labels)
        total_train_loss += loss.item()

        # Backpropagation: calculate how to adjust the model
        loss.backward()
        
        # Update the model's weights
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    # --- Validation Phase ---
    model.eval() # Put the model in evaluation mode
    total_val_loss = 0
    with torch.no_grad(): # We don't need to calculate gradients during validation
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
# We save the model's 'state dictionary', which contains all its learned weights and parameters.
MODEL_SAVE_PATH = 'bia_model.pth'
print(f"\nTraining complete. Saving model to {MODEL_SAVE_PATH}...")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("--- Model Saved Successfully ---")