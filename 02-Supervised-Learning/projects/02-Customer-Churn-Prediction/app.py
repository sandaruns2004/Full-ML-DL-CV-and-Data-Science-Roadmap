import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🏃", layout="wide")

st.title("🏃 Customer Churn Risk Analyzer")
st.write("Analyze customer profiles to predict the likelihood of them canceling their service.")

st.info("Note: This is a skeleton dashboard. Connect your trained classifier to generate real probabilities.")

# Sidebar for inputs
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 10.0, 200.0, 50.0)
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tech_support = st.sidebar.radio("Has Tech Support?", ["Yes", "No"])

# Dummy prediction logic
if st.sidebar.button("Analyze Risk"):
    st.subheader("Churn Risk Assessment")
    
    # Placeholder logic
    risk_score = 0.8 if contract_type == "Month-to-month" and tech_support == "No" else 0.15
    
    if risk_score > 0.5:
        st.error(f"**High Risk of Churn:** {risk_score*100:.1f}% probability")
        st.write("Recommendation: Offer a discount or targeted retention campaign.")
    else:
        st.success(f"**Low Risk of Churn:** {risk_score*100:.1f}% probability")
        st.write("Recommendation: Customer is stable. Monitor periodically.")
    
st.markdown("---")
st.markdown("Developed as part of the Full ML Roadmap - Supervised Learning Projects.")
