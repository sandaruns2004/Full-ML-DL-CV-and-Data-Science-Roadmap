# 🌌 Dimensionality Reduction Dashboard

## Overview
This project provides an interactive playground to compare the three titans of dimensionality reduction: Principal Component Analysis (PCA), t-SNE, and UMAP. It allows users to project high-dimensional datasets (like the Digits dataset) into 2D or 3D and visually observe how each algorithm handles global variance vs. local neighborhood preservation.

## Architecture
```mermaid
graph TD
    A[High-Dimensional Dataset \n e.g. Digits 64-D] --> B[StandardScaler]
    B --> C[PCA \n Linear, Max Variance]
    B --> D[t-SNE \n Non-Linear, KL Divergence]
    B --> E[UMAP \n Topological, Cross-Entropy]
    C --> F[Streamlit Dashboard \n Interactive Plotly Graphs]
    D --> F
    E --> F
    
    style A fill:#4c566a,color:#eceff4
    style F fill:#88c0d0,color:#2e3440
```

## Project Structure
*   `data/`: High-dimensional datasets.
*   `notebooks/`: Algorithm timing comparisons.
*   `src/`: Python scripts for algorithm wrappers.
*   `app.py`: Streamlit dashboard for interactive plotting.

## How to Run
1. Install dependencies: `pip install streamlit scikit-learn umap-learn plotly pandas numpy matplotlib`
2. Run the dashboard: `streamlit run app.py`
