import streamlit as st
import torch
import json
import pandas as pd
from pathlib import Path

LABEL_MAP = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}

@st.cache_resource
def load_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_resource
def load_model(model_path, vocab_size, device):
    from torch import nn
    class SentimentRNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, model_type='lstm', dropout=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            if model_type == 'rnn':
                self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
            elif model_type == 'lstm':
                self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            elif model_type == 'gru':
                self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
            else:
                raise ValueError('Unsupported model_type')
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, text):
            embedded = self.embedding(text)
            output, hidden = self.rnn(embedded)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            hidden = hidden[-1]
            dropped = self.dropout(hidden)
            return self.fc(dropped)

    model = SentimentRNN(vocab_size, 128, 128, 3, model_type='lstm')
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.to(device)
    model.eval()
    return model

@st.cache_data
def clean_text(text):
    import re
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^A-Za-z0-9$%&()*+,\-/.:;?@#\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

@st.cache_data
def encode_text(text, vocab, max_len=50):
    tokens = text.split()
    encoded = [vocab.get(token, vocab.get('<UNK>', 1)) for token in tokens][:max_len]
    padding = [vocab.get('<PAD>', 0)] * (max_len - len(encoded))
    return encoded + padding

@st.cache_data
def predict(model, text, vocab, device):
    encoded = encode_text(clean_text(text), vocab)
    input_tensor = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    label_id = int(probs.argmax())
    return LABEL_MAP[label_id], {LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))}

st.title('Financial News Sentiment Prediction')
st.write('Enter financial tweet text to predict Bearish, Bullish, or Neutral sentiment.')

model_path = Path('best_lstm_baseline.pt')
vocab_path = Path('vocab.json')

if not model_path.exists() or not vocab_path.exists():
    st.warning('Model or vocabulary file not found. Please train the baseline model and save `best_lstm_baseline.pt` and `vocab.json` in this folder.')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = load_vocab(vocab_path)
    model = load_model(str(model_path), len(vocab), device)

    user_text = st.text_area('Tweet / Headline', value='Stocks dip as inflation fears rise around tech earnings')
    if st.button('Predict') and user_text.strip():
        label, probs = predict(model, user_text, vocab, device)
        st.metric('Predicted sentiment', label)
        st.bar_chart(pd.DataFrame([probs]).T.rename(columns={0: 'probability'}))

        st.write('### Probabilities')
        for sentiment, score in probs.items():
            st.write(f'- **{sentiment}**: {score:.3f}')
