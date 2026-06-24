import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas
import os
from PIL import Image, ImageOps
import numpy as np

# Page Configuration
st.set_page_config(page_title="Bank Check Digit Reader", page_icon="🏦", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    h1, h2, h3 {
        color: #88c0d0;
    }
    .prediction-box {
        padding: 20px;
        background-color: #2e3440;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #88c0d0;
    }
</style>
""", unsafe_allow_html=True)

# Define the Model Architecture (Must match train.py)
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

@st.cache_resource
def load_model():
    model_path = os.path.join("src", "model.pth")
    if not os.path.exists(model_path):
        return None
    model = DigitClassifier()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# App Header
st.title("🏦 Bank Check Digit Reader")
st.markdown("Draw a digit (0-9) below to simulate reading a number from a bank check.")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### ✍️ Draw Here")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("### 🤖 Neural Network Prediction")
    model = load_model()
    
    if model is None:
        st.warning("⚠️ **Model not found!** Please run `python src/train.py` first to train and save the neural network model.")
    else:
        if canvas_result.image_data is not None:
            # The canvas gives an RGBA numpy array. Convert it to grayscale PIL image.
            img_array = canvas_result.image_data
            if img_array.any():
                img = Image.fromarray(img_array.astype('uint8'), 'RGBA')
                img = img.convert('L') # Convert to grayscale
                
                # Resize to 28x28
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                
                # Preprocess for the model
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                tensor_img = transform(img).unsqueeze(0)
                
                # Predict
                with torch.no_grad():
                    output = model(tensor_img)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    prediction = torch.argmax(probabilities).item()
                    confidence = probabilities[prediction].item() * 100
                
                # Display Prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Digit: <span style="color:#a3be8c; font-size: 3em;">{prediction}</span></h2>
                    <p>Confidence: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display probability bar chart
                st.markdown("#### Probability Distribution")
                st.bar_chart(probabilities.numpy())
            else:
                st.info("Waiting for drawing...")
        else:
            st.info("Waiting for drawing...")
