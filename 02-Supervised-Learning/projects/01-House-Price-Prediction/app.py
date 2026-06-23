import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 House Price Prediction Dashboard")
st.write("Welcome to the interactive House Price Predictor. Enter the property features below to get an estimated market value.")

st.info("Note: This is a skeleton dashboard. Connect your trained model in the backend to generate real predictions.")

# Sidebar for inputs
st.sidebar.header("Property Features")
sqft = st.sidebar.slider("Square Footage", 500, 10000, 2000)
bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1.0, 5.0, 2.0, step=0.5)
year_built = st.sidebar.slider("Year Built", 1900, 2024, 2000)

# Dummy prediction logic
if st.sidebar.button("Estimate Price"):
    st.subheader("Estimation Results")
    
    # Placeholder calculation
    base_price = 50000
    estimated_price = base_price + (sqft * 150) + (bedrooms * 10000) + (bathrooms * 15000)
    
    st.success(f"**Estimated Price:** ${estimated_price:,.2f}")
    
    # Display feature summary
    st.write("### Input Summary")
    df_summary = pd.DataFrame({
        "Feature": ["Square Footage", "Bedrooms", "Bathrooms", "Year Built"],
        "Value": [sqft, bedrooms, bathrooms, year_built]
    })
    st.table(df_summary)

st.markdown("---")
st.markdown("Developed as part of the Full ML Roadmap - Supervised Learning Projects.")
