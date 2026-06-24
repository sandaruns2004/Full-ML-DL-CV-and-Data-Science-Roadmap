import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶", layout="centered")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #eceff4; }
    h1, h2, h3 { color: #88c0d0; }
    .cat-box { background-color: #ebcb8b; color: #2e3440; padding: 15px; border-radius: 8px; text-align: center; font-size: 1.5em; font-weight: bold;}
    .dog-box { background-color: #a3be8c; color: #2e3440; padding: 15px; border-radius: 8px; text-align: center; font-size: 1.5em; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return self.sigmoid(x)

@st.cache_resource
def load_model():
    model_path = os.path.join("src", "model.pth")
    if not os.path.exists(model_path):
        return None
    model = CatDogCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

st.title("🐶🐱 Cat vs Dog Image Classifier")
st.markdown("Upload an image of a cat or a dog, and the Convolutional Neural Network will classify it.")

model = load_model()

if model is None:
    st.warning("⚠️ **Model not found!** Please run `python src/train.py` first.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        
        st.markdown("### 🤖 Prediction")
        with torch.no_grad():
            output = model(img_tensor).item()
            
        if output > 0.5:
            st.markdown('<div class="dog-box">🐶 It\'s a Dog!</div>', unsafe_allow_html=True)
            st.caption(f"Confidence: {output*100:.2f}%")
        else:
            st.markdown('<div class="cat-box">🐱 It\'s a Cat!</div>', unsafe_allow_html=True)
            st.caption(f"Confidence: {(1-output)*100:.2f}%")
