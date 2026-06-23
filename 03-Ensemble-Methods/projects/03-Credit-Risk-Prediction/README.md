# 💳 Credit Risk Prediction (Ensemble)

## Overview
This project focuses on identifying high-risk borrowers using Bagging and Boosting algorithms. It emphasizes robustness to noise and reducing False Positives (approving a bad loan), a critical metric for financial institutions.

## Architecture
```mermaid
graph TD
    A[Financial History Data] --> B[Data Preprocessing \n Imputation, Missing Values]
    B --> C[Random Forest Baseline]
    B --> D[XGBoost / LightGBM]
    C --> E[Model Evaluation \n ROC-AUC, Recall]
    D --> E
    E --> F[Streamlit Dashboard \n Credit Decisioning Tool]
    
    style A fill:#4c566a,color:#eceff4
    style F fill:#ebcb8b,color:#2e3440
```

## Project Structure
*   `data/`: Contains credit risk datasets.
*   `notebooks/`: Jupyter notebooks with EDA, handling class imbalance, and model comparison.
*   `src/`: Python scripts for feature engineering and evaluation.
*   `app.py`: Streamlit dashboard for interactive risk evaluation.

## How to Run
1. Install dependencies: `pip install streamlit scikit-learn xgboost pandas matplotlib seaborn`
2. Navigate to the project directory.
3. Run the dashboard: `streamlit run app.py`
