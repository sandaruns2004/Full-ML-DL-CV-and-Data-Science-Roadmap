import streamlit as st

# Set page config
st.set_page_config(page_title="Loan Default Predictor", page_icon="💸", layout="wide")

st.title("💸 Loan Default Risk Scorer")
st.write("Evaluate applicant profiles to predict the risk of loan default.")

st.info("Note: This is a skeleton dashboard. Connect your trained ensemble model to generate real risk scores.")

st.sidebar.header("Applicant Profile")
income = st.sidebar.number_input("Annual Income ($)", 10000, 500000, 60000)
loan_amount = st.sidebar.number_input("Loan Amount ($)", 1000, 100000, 15000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
dti = st.sidebar.slider("Debt-to-Income Ratio (%)", 1, 60, 20)

if st.sidebar.button("Calculate Risk"):
    st.subheader("Risk Assessment")
    
    # Placeholder logic
    risk = "High" if credit_score < 600 or dti > 40 else "Low"
    
    if risk == "High":
        st.error("🚨 **High Risk of Default**")
        st.write("Recommendation: Reject application or require a co-signer.")
    else:
        st.success("✅ **Low Risk of Default**")
        st.write("Recommendation: Approve application.")

st.markdown("---")
st.markdown("Developed as part of the Full ML Roadmap - Supervised Learning Projects.")
