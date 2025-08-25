# explain_prediction.py (Fixed for Non-Jupyter Environment)

import pandas as pd
import torch
import torch.nn as nn
import shap
import ast
import numpy as np
from transformers import AutoModel, AutoTokenizer
import textstat
import matplotlib.pyplot as plt
import seaborn as sns

print("--- BIA Model Explainer (XAI using SHAP) ---")

# --- 1. DEFINE HELPER FUNCTION AND MODEL ARCHITECTURE ---
def calculate_formality(text):
    try:
        reading_ease = 100 - textstat.flesch_reading_ease(text)
        avg_sentence_length = textstat.avg_sentence_length(text)
        formality_score = (reading_ease * 0.4) + (avg_sentence_length * 0.6)
        return formality_score
    except (ValueError, ZeroDivisionError):
        return 0

class BIA_Model_v1(nn.Module):
    def __init__(self, num_numerical_features=2):
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

# --- 2. LOAD EVERYTHING ---
print("Loading trained model and tokenizer...")
device = torch.device("cpu")
model = BIA_Model_v1()
model.load_state_dict(torch.load('bia_model.pth', map_location=device))
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# --- 3. PREPARE A SINGLE SAMPLE TO EXPLAIN ---
sample_text = "Okay, follow-up question for you pros. I've been analyzing the Greeks on some weekly options for NVDA. The delta is high and the theta decay looks manageable for a bull call spread. Do you think I should leverage up to maximize my exposure before the next earnings call? YOLO."
print(f"\nExplaining prediction for text: '{sample_text}'")

formality_score = calculate_formality(sample_text)
time_between = 72.0
numerical_features = torch.tensor([[time_between, formality_score]], dtype=torch.float32).to(device)

# --- 4. CREATE THE SHAP PREDICTION FUNCTION ---
def shap_prediction_function(text_list):
    inputs = tokenizer(list(text_list), return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    batch_size = input_ids.shape[0]
    expanded_numerical_features = numerical_features.expand(batch_size, -1)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask, expanded_numerical_features)
    
    probabilities = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    return np.atleast_1d(probabilities)

# --- 5. INITIALIZE SHAP EXPLAINER AND GENERATE VALUES ---
print("\nInitializing SHAP Explainer...")
explainer = shap.Explainer(shap_prediction_function, tokenizer)

print("Generating SHAP values... (this may take a moment)")
shap_values = explainer([sample_text])

# --- 6. GET PREDICTION ---
original_prediction = shap_prediction_function([sample_text])[0]
print(f"\nModel prediction: {original_prediction:.4f}")
print(f"Predicted class: {'Authentic' if original_prediction < 0.5 else 'Bot-like'}")

# --- 7. EXTRACT AND DISPLAY IMPORTANT WORDS ---
print("\n--- SHAP Analysis Results ---")

# Get the tokens and their SHAP values
tokens = shap_values.data[0]
shap_vals = shap_values.values[0]

# Create a list of (token, shap_value) pairs
token_importance = list(zip(tokens, shap_vals))

# Sort by absolute SHAP value (most important first)
token_importance_sorted = sorted(token_importance, key=lambda x: abs(x[1]), reverse=True)

print("\nTop 15 Most Influential Tokens:")
print("-" * 50)
for i, (token, importance) in enumerate(token_importance_sorted[:15]):
    direction = "→ BOT-like" if importance > 0 else "→ AUTHENTIC"
    print(f"{i+1:2d}. '{token}' | SHAP: {importance:+.4f} {direction}")

# --- 8. CREATE CUSTOM VISUALIZATIONS ---
print("\n[SUCCESS] Creating custom SHAP visualizations...")

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot 1: Bar plot of top important tokens
top_tokens = token_importance_sorted[:10]
tokens_plot = [item[0] for item in top_tokens]
values_plot = [item[1] for item in top_tokens]
colors = ['red' if val > 0 else 'green' for val in values_plot]

bars = ax1.barh(range(len(tokens_plot)), values_plot, color=colors, alpha=0.7)
ax1.set_yticks(range(len(tokens_plot)))
ax1.set_yticklabels(tokens_plot)
ax1.set_xlabel('SHAP Value')
ax1.set_title('Top 10 Most Influential Tokens\n(Red: Pushes toward Bot-like, Green: Pushes toward Authentic)')
ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax1.invert_yaxis()

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, values_plot)):
    ax1.text(val + (0.001 if val > 0 else -0.001), i, f'{val:.3f}', 
             ha='left' if val > 0 else 'right', va='center', fontsize=9)

# Plot 2: Sequential token importance
ax2.plot(range(len(shap_vals)), shap_vals, marker='o', alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Token Position')
ax2.set_ylabel('SHAP Value')
ax2.set_title('SHAP Values Across All Tokens in Sequence')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shap_explanation_custom.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 9. CREATE A TEXT HIGHLIGHTING VISUALIZATION ---
print("\nCreating text highlighting visualization...")

# Normalize SHAP values for color intensity
max_abs_shap = max(abs(val) for val in shap_vals)
normalized_shap = [val / max_abs_shap for val in shap_vals]

# Create HTML-style output for better readability
print("\n--- Text with SHAP Highlighting ---")
print("(Intensity shows importance: UPPERCASE = high impact)")
print("-" * 60)

highlighted_text = ""
for token, norm_shap in zip(tokens, normalized_shap):
    if abs(norm_shap) > 0.3:  # High importance
        if norm_shap > 0:
            highlighted_text += f"**{token.upper()}**[BOT+{norm_shap:.2f}] "
        else:
            highlighted_text += f"**{token.upper()}**[AUTH{norm_shap:.2f}] "
    elif abs(norm_shap) > 0.1:  # Medium importance
        if norm_shap > 0:
            highlighted_text += f"{token.upper()}[+{norm_shap:.2f}] "
        else:
            highlighted_text += f"{token.upper()}[{norm_shap:.2f}] "
    else:  # Low importance
        highlighted_text += f"{token} "

print(highlighted_text)

# --- 10. SUMMARY STATISTICS ---
print("\n--- Summary Statistics ---")
print(f"Total tokens analyzed: {len(tokens)}")
print(f"Tokens pushing toward Bot-like: {sum(1 for val in shap_vals if val > 0)}")
print(f"Tokens pushing toward Authentic: {sum(1 for val in shap_vals if val < 0)}")
print(f"Average SHAP value: {np.mean(shap_vals):.4f}")
print(f"Sum of positive SHAP values: {sum(val for val in shap_vals if val > 0):.4f}")
print(f"Sum of negative SHAP values: {sum(val for val in shap_vals if val < 0):.4f}")

print("\n--- Explanation Complete ---")
print("Visualizations saved as 'shap_explanation_custom.png'")
print("Check the console output above for detailed token analysis!")