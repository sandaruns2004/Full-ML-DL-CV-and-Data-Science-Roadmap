import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import json

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #eceff4; }
    h1, h2, h3 { color: #88c0d0; }
    .prediction-box { background-color: #3b4252; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #88c0d0; }
    .healthy { color: #a3be8c; font-weight: bold; font-size: 2em; }
    .diseased { color: #bf616a; font-weight: bold; font-size: 2em; }
</style>
""", unsafe_allow_html=True)

class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_assets():
    model_path = os.path.join("src", "model.pth")
    classes_path = os.path.join("src", "classes.json")
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        return None, None
        
    with open(classes_path, 'r') as f:
        classes = json.load(f)
        
    model = PlantDiseaseCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, classes

st.title("🌿 Plant Disease Detection System")
st.markdown("Upload a photo of a plant leaf to diagnose potential diseases using our Convolutional Neural Network.")

model, classes = load_assets()

if model is None:
    st.warning("⚠️ **Model or classes not found!** Please run `python src/train.py` first.")
else:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        
        st.markdown("### 🔬 Diagnostic Results")
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        prediction_label = classes[predicted_idx.item()]
        
        css_class = "healthy" if "Healthy" in prediction_label else "diseased"
        formatted_label = prediction_label.replace("_", " ")
        
        st.markdown(f"""
        <div class="prediction-box">
            <h3>Diagnosis:</h3>
            <div class="{css_class}">{formatted_label}</div>
            <p>Confidence: {confidence.item()*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
