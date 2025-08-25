# final_evaluation.py (Corrected Import)

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from torch.utils.data import DataLoader # FIX: Import DataLoader

# We need to import all the classes and functions from our final training script
# so this script knows what they are.
from final_training import BIADataset, collate_fn_v2, BIA_Model_v2

# --- MAIN EVALUATION SCRIPT ---
if __name__ == "__main__":
    print("--- Running Final Evaluation on BIA Model v2.0 (Large Dataset) ---")
    
    # Configuration
    DATASET_PATH = 'bia_preprocessed_large.csv'
    MODEL_PATH = 'bia_model_final.pth'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    
    # Load and prepare the data
    df = pd.read_csv(DATASET_PATH)
    df['tokens'] = df['tokens'].apply(ast.literal_eval)
    sequences = df.groupby('user_id').agg({
        'tokens': list, 'time_between_posts_hours': list,
        'formality_score': list, 'label': lambda x: x.iloc[0]
    }).reset_index()
    sequences['numerical_features'] = sequences.apply(lambda r: list(zip(r['time_between_posts_hours'], r['formality_score'])), axis=1)
    
    # Use the same train/test split to get the correct validation set
    _, val_df = train_test_split(sequences, test_size=0.2, random_state=42, stratify=sequences['label'])
    
    val_dataset = BIADataset(val_df)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_v2)
    
    # Load the trained model
    model = BIA_Model_v2(num_numerical_features=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Get predictions
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attn_mask, num_feat, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['numerical_features'].to(DEVICE), batch['labels'].to(DEVICE)
            outputs = model(input_ids, attn_mask, num_feat)
            preds = torch.sigmoid(outputs).squeeze() > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print the final report
    print("\n--- Final Performance Report ---")
    print(classification_report(all_labels, all_preds, target_names=['Human (0)', 'Synthetic (1)']))

    # Display the final confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'Synthetic'], yticklabels=['Human', 'Synthetic'])
    plt.title('Final Model Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()