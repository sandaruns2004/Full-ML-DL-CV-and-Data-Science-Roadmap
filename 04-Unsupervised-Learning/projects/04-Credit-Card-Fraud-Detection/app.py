import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("🕵️ Credit Card Fraud Detection System")
st.markdown("Monitor real-time transactions and detect anomalies using Isolation Forest.")

# Generate Dummy Transaction Data (PCA reduced features V1, V2, V3)
@st.cache_data
def load_data():
    np.random.seed(42)
    # 1000 normal transactions
    normal = np.random.normal(0, 1, (1000, 3))
    # 20 fraudulent transactions
    fraud = np.random.uniform(low=-5, high=5, size=(20, 3))
    
    data = np.vstack([normal, fraud])
    df = pd.DataFrame(data, columns=['V1', 'V2', 'V3'])
    df['Amount'] = np.random.exponential(100, 1020)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

df = load_data()

with st.sidebar:
    st.header("Detection Parameters")
    contamination = st.slider("Expected Fraud % (Contamination)", 0.001, 0.05, 0.02, 0.005)
    
# Train Isolation Forest
iso = IsolationForest(contamination=contamination, random_state=42)
# We train on V1, V2, V3, and Amount
df['Prediction'] = iso.fit_predict(df[['V1', 'V2', 'V3', 'Amount']])
df['Anomaly Score'] = iso.decision_function(df[['V1', 'V2', 'V3', 'Amount']])

# -1 is anomaly, 1 is normal
anomalies = df[df['Prediction'] == -1]

st.subheader("System Status")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(df))
col2.metric("Flagged as Fraud", len(anomalies), delta_color="inverse")
col3.metric("Fraud Rate", f"{(len(anomalies)/len(df))*100:.2f}%")

st.markdown("---")
st.subheader("🚨 Recent Flagged Transactions")
if not anomalies.empty:
    # Sort by how anomalous they are (most negative score first)
    st.dataframe(anomalies.sort_values('Anomaly Score').head(10).style.highlight_min(axis=0, color='red'))
else:
    st.success("No anomalies detected currently.")

st.markdown("---")
st.subheader("Anomaly Score Distribution")
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df['Anomaly Score'], bins=50, kde=True, ax=ax, color='teal')
plt.axvline(x=0, color='red', linestyle='--', label='Decision Boundary')
plt.title("Distribution of Isolation Forest Decision Scores")
plt.legend()
st.pyplot(fig)
