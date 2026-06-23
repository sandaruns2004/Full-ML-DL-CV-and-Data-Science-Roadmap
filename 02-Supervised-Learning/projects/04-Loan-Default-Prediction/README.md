# 💸 Loan Default Prediction

## Overview
This project builds a classification model to predict whether a borrower will default on a loan. It handles highly imbalanced data and uses robust tree-based ensembles to ensure high recall for detecting defaults.

## Architecture
```mermaid
graph TD
    A[Financial & Demographic Data] --> B[Data Preprocessing \n Imputation, Scaling]
    B --> C[Class Balancing \n SMOTE / Class Weights]
    C --> D[Decision Tree Baseline]
    C --> E[Ensemble Methods \n Random Forest, XGBoost]
    D --> F[Model Evaluation \n ROC-AUC, PR-Curve]
    E --> F
    F --> G[Streamlit Dashboard \n Loan Application Scorer]
    
    style A fill:#4c566a,color:#eceff4
    style G fill:#ebcb8b,color:#2e3440
```

## Project Structure
*   `data/`: Contains the financial datasets.
*   `notebooks/`: Jupyter notebooks with EDA and hyperparameter tuning.
*   `src/`: Python scripts for feature engineering and model pipelines.
*   `app.py`: Streamlit dashboard for interactive loan scoring.

## How to Run
1. Install dependencies: `pip install streamlit scikit-learn pandas matplotlib seaborn xgboost`
2. Navigate to the project directory.
3. Run the dashboard: `streamlit run app.py`
