import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

st.set_page_config(page_title="Universal Image Classifier", page_icon="🖼️", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #eceff4; }
    h1, h2, h3 { color: #88c0d0; }
    .prediction-box { background-color: #3b4252; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #a3be8c; }
    .label { color: #a3be8c; font-weight: bold; font-size: 2.5em; text-transform: capitalize; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Load MobileNetV2 with pre-trained weights
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    model.eval()
    
    # Get ImageNet 1000 Categories
    categories = weights.meta["categories"]
    
    # Get standard ImageNet transforms
    preprocess = weights.transforms()
    
    return model, categories, preprocess

st.title("🖼️ Universal Image Classifier")
st.markdown("Upload any image, and our pre-trained State-of-the-Art Deep Learning model (MobileNetV2) will classify it into one of 1000 categories.")

try:
    model, categories, preprocess = load_model()
    
    uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
        with col2:
            st.markdown("### 🤖 Model Prediction")
            
            # Preprocess and prepare batch
            batch = preprocess(image).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                prediction = model(batch).squeeze(0)
                probabilities = torch.nn.functional.softmax(prediction, dim=0)
                
            # Get Top 3 Predictions
            top3_prob, top3_catid = torch.topk(probabilities, 3)
            
            best_label = categories[top3_catid[0]]
            best_prob = top3_prob[0].item() * 100
            
            st.markdown(f"""
            <div class="prediction-box">
                <div class="label">{best_label}</div>
                <p>Confidence: {best_prob:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### Top 3 Matches:")
            for i in range(3):
                st.write(f"{i+1}. **{categories[top3_catid[i]].capitalize()}** ({top3_prob[i].item() * 100:.2f}%)")
                
except Exception as e:
    st.error(f"Failed to load the model. Ensure you have internet connection to download weights. Error: {str(e)}")
