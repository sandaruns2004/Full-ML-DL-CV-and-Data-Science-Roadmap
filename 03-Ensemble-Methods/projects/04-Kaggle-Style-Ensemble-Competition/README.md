# 🏆 Kaggle-Style Ensemble Competition

## Overview
This project simulates a high-stakes Kaggle tabular data competition. It utilizes advanced Stacking and Blending architectures, combining LightGBM, XGBoost, and CatBoost into a massive meta-ensemble to squeeze every drop of accuracy out of the data.

## Architecture
```mermaid
graph TD
    A[Train / Test Data] --> B[Feature Engineering \n Target Encoding, GroupBy]
    B --> C[Layer 1: XGBoost]
    B --> D[Layer 1: LightGBM]
    B --> E[Layer 1: CatBoost]
    C --> F[Out-Of-Fold Predictions]
    D --> F
    E --> F
    F --> G[Layer 2: Ridge / Logistic Meta-Learner]
    G --> H[Final Prediction]
    
    style A fill:#4c566a,color:#eceff4
    style G fill:#b48ead,color:#2e3440
```

## Project Structure
*   `data/`: Contains competition datasets.
*   `notebooks/`: Extensive notebooks showing OOF prediction generation and hyperparameter tuning (Optuna).
*   `src/`: Python scripts for robust cross-validation frameworks.
*   `app.py`: Streamlit dashboard visualizing the ensemble weightings.

## How to Run
1. Install dependencies: `pip install streamlit scikit-learn lightgbm xgboost catboost optuna`
2. Navigate to the project directory.
3. Run the dashboard: `streamlit run app.py`
