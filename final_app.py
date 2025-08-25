# final_app.py (The Definitive Demo)

import gradio as gr
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import textstat
from torch.nn.utils.rnn import pad_sequence

# --- Re-define the v2 Model Architecture ---
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

# --- Load Model and Helper Functions ---
DEVICE = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = BIA_Model_v2(num_numerical_features=2)
model.load_state_dict(torch.load('bia_model_final.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

def calculate_formality(text):
    try:
        reading_ease = 100 - textstat.flesch_reading_ease(text)
        return (reading_ease * 0.4) + (textstat.avg_sentence_length(text) * 0.6)
    except: return 0

# --- The Main Prediction Function ---
def predict_behavior(text_sequence_str):
    # Split the input text into a list of posts, separated by a newline
    posts = [p.strip() for p in text_sequence_str.split('\n') if p.strip()]
    if not posts: return {"Error": 1.0}, "Please enter at least one text post."

    # Preprocess the sequence
    tokens = [torch.tensor(tokenizer.encode(p, truncation=True, max_length=512), dtype=torch.long) for p in posts]
    
    # For a live demo, we create features "on the fly"
    # A real implementation would pull from a user's history
    formality_scores = [calculate_formality(p) for p in posts]
    time_diffs = [0.0] + [24.0] * (len(posts) - 1) # Assume 24h between posts for demo
    numerical_features = torch.tensor(list(zip(time_diffs, formality_scores)), dtype=torch.float32)

    # Pad the sequence to match the model's expected input format
    if len(tokens) < 5:
        tokens += [torch.tensor([0], dtype=torch.long)] * (5 - len(tokens))
        numerical_features = torch.cat([numerical_features, torch.zeros(5 - len(numerical_features), 2)], dim=0)
    tokens = tokens[:5]
    numerical_features = numerical_features[:5]

    # Pad tokens to the same length
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    attention_mask = (padded_tokens != 0).float()
    
    # Add a batch dimension for the model
    input_ids = padded_tokens.unsqueeze(0).to(DEVICE)
    attention_mask = attention_mask.unsqueeze(0).to(DEVICE)
    numerical_features = numerical_features.unsqueeze(0).to(DEVICE)
    
    # Get the prediction
    with torch.no_grad():
        logits = model(input_ids, attention_mask, numerical_features)
    probability = torch.sigmoid(logits).squeeze().item()
    
    label = "Likely Synthetic" if probability > 0.5 else "Likely Human"
    return {"Likely Synthetic": probability, "Likely Human": 1 - probability}

# --- Build the Gradio UI ---
with gr.Blocks(theme=gr.themes.Monochrome(), title="BIA Engine - Final") as demo:
    gr.Markdown("<div style='text-align: center;'><h1>🧠 BIA Engine (Final Model)</h1><p>Enter multiple posts from a single user, separated by a new line, to analyze their behavioral consistency.</p></div>")
    with gr.Row():
        text_input = gr.Textbox(label="Communication Sequence", placeholder="Paste the first post here...\nThen paste the second post on a new line...", lines=15)
        output_label = gr.Label(label="Prediction", num_top_classes=2)
    analyze_button = gr.Button("Analyze Full Sequence", variant="primary")
    analyze_button.click(fn=predict_behavior, inputs=text_input, outputs=output_label)

if __name__ == "__main__":
    demo.launch()