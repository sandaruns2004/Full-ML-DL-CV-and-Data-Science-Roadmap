# 🏠 House Price Prediction (Advanced Ensembles)

## Overview
This project tackles House Price Prediction using advanced ensemble techniques such as Gradient Boosting, LightGBM, and Stacking. These models excel at handling the heterogeneous tabular data typical in real estate.

## Architecture
```mermaid
graph TD
    A[Raw Housing Data] --> B[Data Preprocessing & Feature Engineering]
    B --> C[LightGBM Regressor]
    B --> D[CatBoost Regressor]
    C --> E[Stacking Meta-Learner \n Ridge Regression]
    D --> E
    E --> F[Model Evaluation \n RMSE, R²]
    F --> G[Streamlit Dashboard \n Price Estimator]
    
    style A fill:#4c566a,color:#eceff4
    style G fill:#88c0d0,color:#2e3440
```

## Project Structure
*   `data/`: Contains the housing datasets.
*   `notebooks/`: Jupyter notebooks detailing EDA, EFB (Exclusive Feature Bundling), and model tuning.
*   `src/`: Python scripts for pipelines and inference.
*   `app.py`: Streamlit dashboard for interactive predictions.

## How to Run
1. Install dependencies: `pip install streamlit scikit-learn lightgbm catboost pandas numpy`
2. Navigate to the project directory.
3. Run the dashboard: `streamlit run app.py`
