import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import time

st.set_page_config(page_title="Dimensionality Reduction Comparison", layout="wide")

st.title("🌌 Dimensionality Reduction Comparison")
st.markdown("Compare PCA, t-SNE, and UMAP on high-dimensional data.")

@st.cache_data
def get_data():
    digits = load_digits()
    X = digits.data
    y = digits.target.astype(str)
    return X, y

X, y = get_data()

with st.sidebar:
    st.header("Algorithm Selection")
    algo = st.radio("Select Algorithm", ["PCA", "t-SNE", "UMAP"])
    
    st.header("Hyperparameters")
    if algo == "PCA":
        st.info("PCA is deterministic and requires no hyperparameter tuning for 2D projection.")
    elif algo == "t-SNE":
        perplexity = st.slider("Perplexity", 5, 50, 30)
        n_iter = st.slider("Iterations", 250, 2000, 1000)
    elif algo == "UMAP":
        n_neighbors = st.slider("n_neighbors", 2, 100, 15)
        min_dist = st.slider("min_dist", 0.0, 0.99, 0.1)

# Execution
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

start_time = time.time()

if algo == "PCA":
    reducer = PCA(n_components=2)
    embedding = reducer.fit_transform(X_scaled)
elif algo == "t-SNE":
    reducer = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
elif algo == "UMAP":
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X_scaled)

end_time = time.time()

# Visualization
df_plot = pd.DataFrame({
    'Dim 1': embedding[:, 0],
    'Dim 2': embedding[:, 1],
    'Label': y
})

st.subheader(f"{algo} Projection (Execution Time: {end_time - start_time:.2f} seconds)")

fig = px.scatter(df_plot, x='Dim 1', y='Dim 2', color='Label', 
                 color_discrete_sequence=px.colors.qualitative.Alphabet,
                 hover_data=['Label'], opacity=0.8)

fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Observations")
if algo == "PCA":
    st.markdown("PCA preserves global variance but fails to separate complex, non-linear clusters like handwritten digits. The digits are mostly mashed together in the center.")
elif algo == "t-SNE":
    st.markdown("t-SNE beautifully separates the digits into distinct islands by preserving local neighborhoods. Notice how it takes significantly longer to run than PCA.")
elif algo == "UMAP":
    st.markdown("UMAP separates the clusters nearly as well as t-SNE but executes much faster. It also attempts to preserve global distances (e.g., visually similar digits like 1 and 7 might be placed closer together globally).")
