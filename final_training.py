# final_training.py (with Progress Bars)

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
import ast
from torch.nn.utils.rnn import pad_sequence
import logging
from tqdm import tqdm # New import for the progress bar

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DATASET_PATH = 'bia_preprocessed_large.csv' # Make sure this points to our new file
MODEL_SAVE_PATH = 'bia_model_final.pth'
MAX_SEQ_LENGTH = 5
NUM_EPOCHS = 10  # Increased to 10 for the final, large dataset
BATCH_SIZE = 16  # We can try a larger batch size with the GPU
LEARNING_RATE = 2e-5

# --- Re-define Dataset Class and Collate Function ---
class BIADataset(Dataset):
    def __init__(self, dataframe): self.sequences = dataframe
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx):
        item = self.sequences.iloc[idx]
        return {
            'tokens': [torch.tensor(t, dtype=torch.long) for t in item['tokens']],
            'numerical_features': torch.tensor(item['numerical_features'], dtype=torch.float32),
            'label': torch.tensor(item['label'], dtype=torch.float32)
        }

def collate_fn_v2(batch):
    labels = torch.stack([item['label'] for item in batch])
    all_user_tokens, all_user_numericals = [], []
    for item in batch:
        user_tokens, user_numericals = item['tokens'], item['numerical_features']
        if len(user_tokens) > MAX_SEQ_LENGTH:
            user_tokens, user_numericals = user_tokens[-MAX_SEQ_LENGTH:], user_numericals[-MAX_SEQ_LENGTH:]
        elif len(user_tokens) < MAX_SEQ_LENGTH:
            pad_len = MAX_SEQ_LENGTH - len(user_tokens)
            user_tokens += [torch.tensor([0], dtype=torch.long)] * pad_len
            user_numericals = torch.cat([user_numericals, torch.zeros(pad_len, 2)], dim=0)
        all_user_tokens.append(user_tokens)
        all_user_numericals.append(user_numericals)
    
    padded_posts = [pad_sequence(ut, batch_first=True, padding_value=0) for ut in all_user_tokens]
    max_token_len = max(p.shape[1] for p in padded_posts if p.nelement() > 0)
    
    final_tokens = torch.zeros(len(batch), MAX_SEQ_LENGTH, max_token_len, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), MAX_SEQ_LENGTH, max_token_len, dtype=torch.float32)

    for i, p in enumerate(padded_posts):
        if p.nelement() > 0:
            final_tokens[i, :, :p.shape[1]] = p
            attn_mask[i, :, :p.shape[1]] = (p != 0).float()
        
    return {
        'input_ids': final_tokens, 'attention_mask': attn_mask,
        'numerical_features': torch.stack(all_user_numericals), 'labels': labels
    }

# --- Re-define Model Architecture ---
class BIA_Model_v2(nn.Module):
    def __init__(self, num_numerical_features=2):
        super(BIA_Model_v2, self).__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.lstm = nn.LSTM(self.bert.config.hidden_size, 256, 1, batch_first=True)
        self.numerical_processor = nn.Sequential(nn.Linear(num_numerical_features, 128), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 1)
        )
    def forward(self, input_ids, attention_mask, numerical_features):
        b, n, s = input_ids.shape
        bert_out = self.bert(input_ids=input_ids.view(-1, s), attention_mask=attention_mask.view(-1, s)).last_hidden_state[:, 0]
        bert_out = bert_out.view(b, n, -1)
        _, (h_n, _) = self.lstm(bert_out)
        seq_out = h_n.squeeze(0)
        num_out = self.numerical_processor(numerical_features.mean(dim=1))
        combined = torch.cat((seq_out, num_out), dim=1)
        return self.classifier(combined)

# --- Main Training Script ---
if __name__ == "__main__":
    logging.info("--- BIA Model Final Training Run ---")
    
    df = pd.read_csv(DATASET_PATH)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    sequences = df.groupby('user_id').agg({
        'tokens': list, 'time_between_posts_hours': list,
        'formality_score': list, 'label': lambda x: x.iloc[0]
    }).reset_index()
    sequences['numerical_features'] = sequences.apply(lambda r: list(zip(r['time_between_posts_hours'], r['formality_score'])), axis=1)

    train_df, val_df = train_test_split(sequences, test_size=0.2, random_state=42, stratify=sequences['label'])
    logging.info(f"Training sequences: {len(train_df)}, Validation sequences: {len(val_df)}")

    label_counts = train_df['label'].value_counts()
    count_human = label_counts.get(0, 1)
    count_synthetic = label_counts.get(1, 1)
    pos_weight = count_human / count_synthetic
    pos_weight_tensor = torch.tensor([pos_weight])
    logging.info(f"Class imbalance detected. Human: {count_human}, Synthetic: {count_synthetic}")
    logging.info(f"Calculated positive weight for loss function: {pos_weight:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    train_dataset = BIADataset(train_df)
    val_dataset = BIADataset(val_df)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_v2)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_v2)

    model = BIA_Model_v2(num_numerical_features=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))

    logging.info(f"--- Starting Final Training for {NUM_EPOCHS} epochs ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        # FIX: Wrapped the dataloader with tqdm for a live progress bar
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]"):
            input_ids, attn_mask, num_feat, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['numerical_features'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attn_mask, num_feat)
            loss = loss_function(outputs.squeeze(), labels)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            # FIX: Added a progress bar for the validation loop too
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]"):
                input_ids, attn_mask, num_feat, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['numerical_features'].to(device), batch['labels'].to(device)
                outputs = model(input_ids, attn_mask, num_feat)
                loss = loss_function(outputs.squeeze(), labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    logging.info(f"\nTraining complete. Saving final model to {MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info("--- ✅ Final Model Saved Successfully ---")