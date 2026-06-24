import streamlit as st
import torch
import torch.nn as nn
import joblib
import os
import numpy as np

st.set_page_config(page_title="Spam Detection System", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #eceff4; }
    h1, h2, h3 { color: #88c0d0; }
    .spam-box { background-color: #bf616a; color: white; padding: 15px; border-radius: 8px; text-align: center; font-size: 1.5em; font-weight: bold;}
    .ham-box { background-color: #a3be8c; color: #2e3440; padding: 15px; border-radius: 8px; text-align: center; font-size: 1.5em; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

class SpamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

@st.cache_resource
def load_assets():
    model_path = os.path.join("src", "model.pth")
    vectorizer_path = os.path.join("src", "vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None
    vectorizer = joblib.load(vectorizer_path)
    model = SpamClassifier(input_dim=len(vectorizer.vocabulary_))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, vectorizer

st.title("🛡️ SMS / Email Spam Detection System")
st.markdown("Paste a text message below to classify it as Spam or Ham (Not Spam).")

model, vectorizer = load_assets()

if model is None:
    st.warning("⚠️ **Model or Vectorizer not found!** Please run `python src/train.py` first.")
else:
    user_input = st.text_area("Enter message text:", height=150, placeholder="e.g. Congratulations! You've won a $1000 gift card...")
    
    if st.button("Analyze Text"):
        if user_input.strip() == "":
            st.error("Please enter some text to analyze.")
        else:
            # Preprocess
            input_vec = vectorizer.transform([user_input]).toarray()
            input_tensor = torch.FloatTensor(input_vec)
            
            # Predict
            with torch.no_grad():
                prob = model(input_tensor).item()
                
            st.markdown("---")
            st.subheader("Prediction")
            
            if prob > 0.5:
                st.markdown('<div class="spam-box">🚫 SPAM DETECTED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ham-box">✅ HAM (SAFE)</div>', unsafe_allow_html=True)
                
            st.progress(prob)
            st.caption(f"Spam Probability: {prob * 100:.2f}%")
