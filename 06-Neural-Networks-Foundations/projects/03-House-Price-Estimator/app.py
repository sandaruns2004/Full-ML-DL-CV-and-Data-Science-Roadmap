import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import joblib
import os
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="House Price Estimator", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0e1117; color: #eceff4; }
    h1, h2, h3 { color: #88c0d0; }
    .price-box {
        padding: 20px;
        background-color: #2e3440;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #a3be8c;
    }
</style>
""", unsafe_allow_html=True)

class HousePriceEstimator(nn.Module):
    def __init__(self, input_dim):
        super(HousePriceEstimator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

@st.cache_resource
def load_assets():
    model_path = os.path.join("src", "model.pth")
    scaler_path = os.path.join("src", "scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    model = HousePriceEstimator(input_dim=8)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler

st.title("🏠 California House Price Estimator")
st.markdown("Adjust the features below to estimate the median house value using a Deep Neural Network.")

model, scaler = load_assets()

if model is None:
    st.warning("⚠️ **Model or Scaler not found!** Please run `python src/train.py` first.")
else:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🏡 Property Features")
        med_inc = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.5)
        house_age = st.slider("House Age (Years)", 1.0, 50.0, 28.0)
        avg_rooms = st.slider("Avg Rooms per Household", 1.0, 10.0, 5.0)
        avg_bedrms = st.slider("Avg Bedrooms per Household", 0.5, 5.0, 1.0)
        population = st.slider("Population", 10, 5000, 1000)
        avg_occup = st.slider("Avg Occupancy", 1.0, 10.0, 3.0)
        latitude = st.slider("Latitude", 32.0, 42.0, 35.0)
        longitude = st.slider("Longitude", -124.0, -114.0, -119.0)
        
    with col2:
        st.subheader("💰 Price Estimation")
        
        # Prepare data
        input_data = np.array([[med_inc, house_age, avg_rooms, avg_bedrms, population, avg_occup, latitude, longitude]])
        input_scaled = scaler.transform(input_data)
        input_tensor = torch.FloatTensor(input_scaled)
        
        # Predict
        with torch.no_grad():
            price_pred = model(input_tensor).item()
            
        estimated_price = price_pred * 100000 # Since target is in 100,000s
        
        st.markdown(f"""
        <div class="price-box">
            <h2>Estimated Median Value:</h2>
            <h1 style="color:#a3be8c;">${estimated_price:,.2f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("The prediction is based on the California Housing dataset. The Neural Network maps the 8 input features to a continuous price output.")
