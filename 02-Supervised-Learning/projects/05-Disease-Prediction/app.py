import streamlit as st

# Set page config
st.set_page_config(page_title="Disease Predictor", page_icon="🩺", layout="wide")

st.title("🩺 Diagnostic Tool: Disease Predictor")
st.write("Enter patient vitals and test results below to assess the likelihood of disease presence.")

st.info("Note: This is a skeleton dashboard. Connect your trained medical classifier to perform actual predictions. This is for educational purposes only and not medical advice.")

st.sidebar.header("Patient Vitals")
age = st.sidebar.slider("Age", 1, 100, 45)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
blood_pressure = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.sidebar.slider("Serum Cholestoral (mg/dl)", 100, 400, 200)

if st.sidebar.button("Run Diagnostics"):
    st.subheader("Diagnostic Results")
    
    # Placeholder logic
    risk = "High" if age > 50 and blood_pressure > 140 else "Low"
    
    if risk == "High":
        st.error("🚨 **Warning: High Likelihood of Disease Presence Detected**")
        st.write("Please consult a healthcare professional for further testing.")
    else:
        st.success("✅ **Clear: Low Likelihood of Disease Presence**")
        st.write("Vitals appear to be within normal ranges.")

st.markdown("---")
st.markdown("Developed as part of the Full ML Roadmap - Supervised Learning Projects.")
