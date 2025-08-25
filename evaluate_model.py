# evaluate_model.py

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModel
import ast
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# --- Re-define all necessary classes and functions from training scripts ---

# Re-define Dataset Class
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

# Re-define v1 Collate Function
def collate_fn_v1(batch):
    labels = torch.stack([item['label'] for item in batch])
    numerical_features = torch.stack([item['numerical_features'][0] for item in batch])
    token_lists = [item['tokens'][0] for item in batch]
    padded_tokens = pad_sequence(token_lists, batch_first=True, padding_value=0)
    attention_mask = (padded_tokens != 0).type(torch.float32)
    return {
        'input_ids': padded_tokens, 'attention_mask': attention_mask,
        'numerical_features': numerical_features, 'labels': labels
    }

# Re-define v2 Collate Function
MAX_SEQ_LENGTH = 5
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
    max_token_len = max(p.shape[1] for p in padded_posts)
    final_tokens = torch.zeros(len(batch), MAX_SEQ_LENGTH, max_token_len, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), MAX_SEQ_LENGTH, max_token_len, dtype=torch.float32)
    for i, p in enumerate(padded_posts):
        final_tokens[i, :, :p.shape[1]] = p
        attn_mask[i, :, :p.shape[1]] = (p != 0).float()
        
    return {
        'input_ids': final_tokens, 'attention_mask': attn_mask,
        'numerical_features': torch.stack(all_user_numericals), 'labels': labels
    }

# Re-define v1 Model Architecture
class BIA_Model_v1(nn.Module):
    def __init__(self, num_numerical_features):
        super(BIA_Model_v1, self).__init__()
        self.bert = AutoModel.from_pretrained('distilbert-base-uncased')
        self.numerical_processor = nn.Sequential(nn.Linear(num_numerical_features, 128), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 128, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, 1)
        )
    def forward(self, input_ids, attention_mask, numerical_features):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        num_out = self.numerical_processor(numerical_features)
        combined = torch.cat((bert_out, num_out), dim=1)
        return self.classifier(combined)

# Re-define v2 Model Architecture
class BIA_Model_v2(nn.Module):
    def __init__(self, num_numerical_features):
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

# --- EVALUATION FUNCTION ---
def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, numerical_features)
            # Convert logits to probabilities and then to binary predictions (0 or 1)
            preds = torch.sigmoid(outputs).squeeze() > 0.5 
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return all_labels, all_preds

# --- MAIN SCRIPT ---
# Load data and create datasets/dataloaders
df = pd.read_csv('bia_preprocessed_dataset.csv')
df['tokens'] = df['tokens'].apply(ast.literal_eval)
sequences = df.groupby('user_id').agg(list).reset_index()
sequences['label'] = sequences['label'].apply(lambda x: x[0])
sequences['numerical_features'] = sequences.apply(lambda r: list(zip(r['time_between_posts_hours'], r['formality_score'])), axis=1)
_, val_df = train_test_split(sequences, test_size=0.2, random_state=42, stratify=sequences['label'])

val_dataset = BIADataset(val_df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Evaluate Model v1.0 ---
print("\n--- Evaluating Model v1.0 (Single Post) ---")
model_v1 = BIA_Model_v1(num_numerical_features=2)
model_v1.load_state_dict(torch.load('bia_model.pth', map_location=device))
val_dataloader_v1 = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn_v1)
labels_v1, preds_v1 = evaluate(model_v1, val_dataloader_v1, device)

print(classification_report(labels_v1, preds_v1, target_names=['Human (0)', 'Synthetic (1)']))
cm_v1 = confusion_matrix(labels_v1, preds_v1)
plt.figure(figsize=(6,5))
sns.heatmap(cm_v1, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'Synthetic'], yticklabels=['Human', 'Synthetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Model v1.0')
plt.show()

# --- Evaluate Model v2.0 ---
print("\n--- Evaluating Model v2.0 (Sequence-Aware) ---")
model_v2 = BIA_Model_v2(num_numerical_features=2)
model_v2.load_state_dict(torch.load('bia_model_v2.pth', map_location=device))
val_dataloader_v2 = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn_v2)
labels_v2, preds_v2 = evaluate(model_v2, val_dataloader_v2, device)

print(classification_report(labels_v2, preds_v2, target_names=['Human (0)', 'Synthetic (1)']))
cm_v2 = confusion_matrix(labels_v2, preds_v2)
plt.figure(figsize=(6,5))
sns.heatmap(cm_v2, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'Synthetic'], yticklabels=['Human', 'Synthetic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Model v2.0')
plt.show()