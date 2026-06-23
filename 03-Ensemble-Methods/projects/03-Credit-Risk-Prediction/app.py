import streamlit as st

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

st.title("💳 Credit Risk Ensemble Predictor")
st.write("Evaluate applicant profiles using tree-based ensembles to predict default risk.")

st.info("Note: This is a skeleton dashboard. Connect your trained ensemble to generate real risk scores.")

st.sidebar.header("Applicant Profile")
income = st.sidebar.number_input("Annual Income ($)", 10000, 500000, 60000)
loan_amount = st.sidebar.number_input("Loan Amount ($)", 1000, 100000, 15000)
credit_history_len = st.sidebar.slider("Credit History (Years)", 1, 30, 5)

if st.sidebar.button("Calculate Risk"):
    st.subheader("Risk Assessment")
    
    # Placeholder logic
    risk = "High" if income < 30000 and loan_amount > 20000 else "Low"
    
    if risk == "High":
        st.error("🚨 **High Risk of Default**")
    else:
        st.success("✅ **Low Risk of Default**")

st.markdown("---")
st.markdown("Developed as part of the Full ML Roadmap - Ensemble Methods Projects.")
