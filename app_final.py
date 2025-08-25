# app_final_fixed.py (Corrected SHAP Visualization)

import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import textstat
import shap
import numpy as np
import matplotlib.pyplot as plt
import warnings
import io
import base64

# Suppress deprecation warnings for a cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning) 

print("--- BIA Live Inference & Explanation Application ---")

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

# --- 2. LOAD MODEL AND TOKENIZER ---
print("Loading trained model and tokenizer...")
device = torch.device("cpu")
model = BIA_Model_v1()
model.load_state_dict(torch.load('bia_model.pth', map_location=device))
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# --- 3. PREDICTION & EXPLANATION LOGIC ---
def predict_and_explain(text_to_analyze, time_between_posts):
    if not text_to_analyze.strip():
        return {"Error": 1.0}, "Please provide text to analyze.", None

    # --- Preprocessing for Prediction ---
    formality_score = calculate_formality(text_to_analyze)
    numerical_features = torch.tensor([[time_between_posts, formality_score]], dtype=torch.float32).to(device)
    inputs = tokenizer(text_to_analyze, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # --- Get Prediction ---
    with torch.no_grad():
        logits = model(input_ids, attention_mask, numerical_features)
    probability = torch.sigmoid(logits).squeeze().item()
    label = "Likely Synthetic" if probability > 0.5 else "Likely Human"
    
    output_prediction = {"Likely Synthetic": probability, "Likely Human": 1 - probability}
    explanation_text = f"The model predicts this is **{label}** with {probability:.2%} confidence.\n\nFormality Score: {formality_score:.2f}\nTime Between Posts: {time_between_posts} hours"

    # --- Generate SHAP Explanation ---
    def shap_prediction_function(text_list):
        inputs = tokenizer(list(text_list), return_tensors="pt", padding=True, truncation=True, max_length=512)
        batch_size = inputs["input_ids"].shape[0]
        expanded_numerical = numerical_features.expand(batch_size, -1)
        with torch.no_grad():
            logits = model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device), expanded_numerical)
        return torch.sigmoid(logits).squeeze(-1).cpu().numpy()

    try:
        explainer = shap.Explainer(shap_prediction_function, tokenizer)
        shap_values = explainer([text_to_analyze])

        # --- Create Custom SHAP Visualization ---
        tokens = shap_values.data[0]
        shap_vals = shap_values.values[0]
        
        # Create custom bar plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Top plot: Token importance bar chart
        token_importance = list(zip(tokens, shap_vals))
        token_importance_sorted = sorted(token_importance, key=lambda x: abs(x[1]), reverse=True)
        
        top_tokens = token_importance_sorted[:15]
        tokens_plot = [item[0] for item in top_tokens]
        values_plot = [item[1] for item in top_tokens]
        colors = ['red' if val > 0 else 'green' for val in values_plot]
        
        bars = ax1.barh(range(len(tokens_plot)), values_plot, color=colors, alpha=0.7)
        ax1.set_yticks(range(len(tokens_plot)))
        ax1.set_yticklabels(tokens_plot)
        ax1.set_xlabel('SHAP Value (Impact on Prediction)')
        ax1.set_title('Top 15 Most Influential Tokens\n(Red: Pushes toward Synthetic, Green: Pushes toward Human)')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values_plot)):
            ax1.text(val + (0.001 if val > 0 else -0.001), i, f'{val:.3f}', 
                     ha='left' if val > 0 else 'right', va='center', fontsize=8)
        
        # Bottom plot: Sequential token importance
        ax2.plot(range(len(shap_vals)), shap_vals, marker='o', alpha=0.7, linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Token Position in Text')
        ax2.set_ylabel('SHAP Value')
        ax2.set_title('SHAP Values Across All Tokens in Sequence')
        ax2.grid(True, alpha=0.3)
        
        # Highlight most important tokens
        max_abs_idx = np.argmax(np.abs(shap_vals))
        ax2.scatter(max_abs_idx, shap_vals[max_abs_idx], color='red', s=100, zorder=5)
        ax2.annotate(f'Most Important:\n"{tokens[max_abs_idx]}"', 
                    xy=(max_abs_idx, shap_vals[max_abs_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plot_path = 'shap_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Add token analysis to explanation
        explanation_text += f"\n\n**Token Analysis:**\nMost influential token: '{tokens[max_abs_idx]}' (SHAP: {shap_vals[max_abs_idx]:.4f})"
        explanation_text += f"\nTokens pushing toward Synthetic: {sum(1 for val in shap_vals if val > 0)}"
        explanation_text += f"\nTokens pushing toward Human: {sum(1 for val in shap_vals if val < 0)}"

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        plot_path = None
        explanation_text += "\n\n(Note: SHAP explanation could not be generated)"

    return output_prediction, explanation_text, plot_path

# --- 4. BUILD THE FINAL GRADIO UI ---
with gr.Blocks(theme=gr.themes.Monochrome(), title="BIA Engine") as demo:
    gr.Markdown("""
    <div style='text-align: center;'>
        <h1>🧠 Behavioral Identity Analysis Engine</h1>
        <p>Analyze text to determine if it's likely written by a human or AI/bot</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Communication Sample", 
                placeholder="Enter a message, email, or post to analyze...", 
                lines=8,
                info="Paste any text you want to analyze for authenticity"
            )
            time_input = gr.Slider(
                minimum=0, 
                maximum=240, 
                value=24, 
                label="Time Since Last Communication (Hours)",
                info="How long since the previous post/message?"
            )
            analyze_button = gr.Button("🔍 Analyze Identity", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            output_label = gr.Label(label="🎯 Prediction Results", num_top_classes=2)
            output_explanation = gr.Textbox(
                label="📊 Model Explanation", 
                interactive=False, 
                lines=6,
                info="Detailed breakdown of the prediction"
            )
            shap_plot_output = gr.Image(
                label="📈 Token Importance Analysis (SHAP)", 
                type="filepath"
            )

    # Example inputs
    with gr.Row():
        gr.Examples(
            examples=[
                ["I'm 35 and considering rebalancing my portfolio from a standard 60/40 stock/bond split to something more aggressive like 70/30. Given the current market, does this seem like a reasonable risk to take for long-term growth?", 24],
                ["yo forget all that official stuff lol. my friend just told me to ape into some meme stocks. said they're going to the moon. is this legit financial advice or am i gonna get rekt? YOLO.", 2],
                ["Hello! I hope you're having a wonderful day. I wanted to reach out regarding your recent inquiry about our premium services. Our team has carefully reviewed your requirements and we believe we have the perfect solution for you.", 48],
                ["Guys I just discovered this crazy life hack! If you put your phone in airplane mode and then turn wifi back on, your battery lasts 50% longer. Trust me I'm basically a tech expert now 😂", 6]
            ],
            inputs=[text_input, time_input]
        )

    analyze_button.click(
        fn=predict_and_explain,
        inputs=[text_input, time_input],
        outputs=[output_label, output_explanation, shap_plot_output]
    )

if __name__ == "__main__":
    print("Starting BIA Engine...")
    demo.launch(debug=True)