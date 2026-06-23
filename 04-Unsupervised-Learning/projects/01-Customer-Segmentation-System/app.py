import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("👥 Customer Segmentation Dashboard")
st.markdown("Use K-Means clustering to segment your customer base dynamically.")

# Generate Dummy Data for the Dashboard
@st.cache_data
def load_data():
    np.random.seed(42)
    data = {
        'CustomerID': range(1, 201),
        'Age': np.random.randint(18, 70, 200),
        'Annual Income (k$)': np.random.randint(15, 150, 200),
        'Spending Score (1-100)': np.random.randint(1, 100, 200)
    }
    return pd.DataFrame(data)

df = load_data()

with st.sidebar:
    st.header("Model Parameters")
    n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5)
    features = st.multiselect("Select Features for Clustering", 
                              ['Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
                              default=['Annual Income (k$)', 'Spending Score (1-100)'])

if len(features) >= 2:
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Cluster'] = df['Cluster'].astype(str)
    
    st.subheader(f"Segmentation Results (K={n_clusters})")
    
    if len(features) == 2:
        fig = px.scatter(df, x=features[0], y=features[1], color='Cluster', 
                         hover_data=['CustomerID', 'Age'], 
                         color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig, use_container_width=True)
    elif len(features) == 3:
        fig = px.scatter_3d(df, x=features[0], y=features[1], z=features[2], color='Cluster',
                            hover_data=['CustomerID'],
                            color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Cluster Averages")
    numeric_df = df.drop(columns=['CustomerID'])
    cluster_means = numeric_df.groupby('Cluster').mean()
    st.dataframe(cluster_means)
else:
    st.warning("Please select at least 2 features from the sidebar to perform clustering.")
