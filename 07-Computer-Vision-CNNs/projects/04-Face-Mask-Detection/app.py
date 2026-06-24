import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import os

st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #eceff4; }
    h1, h2, h3 { color: #88c0d0; }
</style>
""", unsafe_allow_html=True)

# Define CNN architecture for Mask Classification (assumes binary: 0=No Mask, 1=Mask)
class MaskCNN(nn.Module):
    def __init__(self):
        super(MaskCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 32 * 32)
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

@st.cache_resource
def load_models():
    # Load OpenCV Face Detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load PyTorch Mask Classifier
    model_path = os.path.join("src", "mask_detector.pth")
    model = MaskCNN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        has_weights = True
    else:
        has_weights = False
    model.eval()
    
    return face_cascade, model, has_weights

st.title("😷 Face Mask Detection System")
st.markdown("Upload an image with people. The app uses OpenCV to localize faces and a Convolutional Neural Network to classify if they are wearing masks.")

face_cascade, mask_model, has_weights = load_models()

if not has_weights:
    st.info("ℹ️ **Note:** `mask_detector.pth` not found in `src/`. The CNN will use uninitialized weights, so mask predictions will be random. Train a model and save it to enable accurate classification.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    img_arr = np.array(image)
    
    # Convert to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    st.markdown(f"### Detected {len(faces)} face(s)")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = image.crop((x, y, x+w, y+h))
        face_tensor = transform(face_roi).unsqueeze(0)
        
        # Predict Mask
        with torch.no_grad():
            prob = mask_model(face_tensor).item()
            
        is_mask = prob > 0.5
        label = "Mask" if is_mask else "No Mask"
        color = (0, 255, 0) if is_mask else (255, 0, 0) # Green for Mask, Red for No Mask
        
        # Draw bounding box on original image array
        cv2.rectangle(img_arr, (x, y), (x+w, y+h), color, 4)
        cv2.putText(img_arr, f"{label} ({prob*100:.1f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    # Display Result
    st.image(img_arr, caption="Detection Result", use_column_width=True)
