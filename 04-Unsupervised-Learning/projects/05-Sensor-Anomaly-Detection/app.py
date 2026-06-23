import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import time

st.set_page_config(page_title="Sensor Anomaly Detection", layout="wide")

st.title("🏭 Industrial Sensor Anomaly Detection")
st.markdown("Monitoring multivariate sensor telemetry using Local Outlier Factor (LOF) to predict mechanical failures.")

# Generate Dummy Time-Series Data
@st.cache_data
def load_data():
    np.random.seed(42)
    time_index = pd.date_range(start="2026-06-01", periods=1000, freq="H")
    
    # Normal operation
    temp = np.random.normal(70, 2, 1000)
    vibration = np.random.normal(0.5, 0.1, 1000)
    pressure = np.random.normal(30, 1, 1000)
    
    # Inject contextual anomalies
    # Temp and Pressure are usually correlated. Let's break the correlation.
    temp[400:405] += 10 # Sudden spike
    vibration[750:760] += 0.8 # Bearing failing
    pressure[200:205] -= 5 # Pressure drop
    
    df = pd.DataFrame({'Temperature': temp, 'Vibration': vibration, 'Pressure': pressure}, index=time_index)
    return df

df = load_data()

with st.sidebar:
    st.header("LOF Parameters")
    n_neighbors = st.slider("Number of Neighbors (k)", 5, 50, 20)
    contamination = st.slider("Contamination", 0.001, 0.05, 0.02, 0.005)

# Train LOF
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
df['Anomaly'] = lof.fit_predict(X_scaled)
df['LOF_Score'] = -lof.negative_outlier_factor_

anomalies = df[df['Anomaly'] == -1]

st.subheader(f"System Overview: {len(anomalies)} Anomalies Detected")

# Plot Time Series
fig, axes = plt.subplots(3, 1, figsize=(15, 8), sharex=True)

# Temperature
axes[0].plot(df.index, df['Temperature'], label='Temperature', color='blue', alpha=0.6)
axes[0].scatter(anomalies.index, anomalies['Temperature'], color='red', label='Anomaly')
axes[0].legend()

# Vibration
axes[1].plot(df.index, df['Vibration'], label='Vibration', color='orange', alpha=0.6)
axes[1].scatter(anomalies.index, anomalies['Vibration'], color='red', label='Anomaly')
axes[1].legend()

# Pressure
axes[2].plot(df.index, df['Pressure'], label='Pressure', color='green', alpha=0.6)
axes[2].scatter(anomalies.index, anomalies['Pressure'], color='red', label='Anomaly')
axes[2].legend()

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")
st.subheader("High-Risk Timeframes")
if not anomalies.empty:
    st.dataframe(anomalies.sort_values('LOF_Score', ascending=False).head(10).style.highlight_max(subset=['LOF_Score'], color='red'))
else:
    st.success("All sensors operating within normal contextual parameters.")
