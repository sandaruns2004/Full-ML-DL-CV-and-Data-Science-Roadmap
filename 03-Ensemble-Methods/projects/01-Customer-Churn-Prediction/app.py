import streamlit as st

# Set page config
st.set_page_config(page_title="Customer Churn Ensemble Predictor", page_icon="🏃", layout="wide")

st.title("🏃 Customer Churn Ensemble Analyzer")
st.write("Analyze customer profiles using a robust ensemble model to predict churn probability.")

st.info("Note: This is a skeleton dashboard. Connect your trained XGBoost/Random Forest ensemble to generate real probabilities.")

# Sidebar for inputs
st.sidebar.header("Customer Profile")
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 10.0, 200.0, 50.0)
contract_type = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
tech_support = st.sidebar.radio("Has Tech Support?", ["Yes", "No"])

# Dummy prediction logic
if st.sidebar.button("Analyze Risk"):
    st.subheader("Ensemble Churn Risk Assessment")
    
    # Placeholder logic
    risk_score = 0.85 if contract_type == "Month-to-month" and tech_support == "No" else 0.10
    
    if risk_score > 0.5:
        st.error(f"**High Risk of Churn:** {risk_score*100:.1f}% probability")
        st.write("Recommendation: Intervene immediately with targeted retention offers.")
    else:
        st.success(f"**Low Risk of Churn:** {risk_score*100:.1f}% probability")
        st.write("Recommendation: Customer is stable.")
    
st.markdown("---")
st.markdown("Developed as part of the Full ML Roadmap - Ensemble Methods Projects.")
