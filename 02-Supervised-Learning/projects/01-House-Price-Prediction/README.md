# 🏠 House Price Prediction System

## Overview
This project applies Supervised Learning regression techniques (such as Linear Regression, Random Forest, and Gradient Boosting) to predict the price of houses based on various features like square footage, number of bedrooms, and location.

## Architecture
```mermaid
graph TD
    A[Raw Housing Data] --> B[Data Preprocessing & Feature Engineering]
    B --> C[Linear Regression Baseline]
    B --> D[Ensemble Methods \n Random Forest / XGBoost]
    C --> E[Model Evaluation \n RMSE, MAE, R²]
    D --> E
    E --> F[Streamlit Dashboard \n Interactive Price Estimator]
    
    style A fill:#4c566a,color:#eceff4
    style F fill:#88c0d0,color:#2e3440
```

## Project Structure
*   `data/`: Contains the housing datasets (e.g., Ames Housing, Boston Housing).
*   `notebooks/`: Jupyter notebooks detailing EDA, feature selection, and model training.
*   `src/`: Python scripts for data processing pipelines and model inference.
*   `app.py`: Streamlit dashboard for interactive predictions.

## How to Run
1. Install dependencies: `pip install streamlit scikit-learn pandas numpy matplotlib seaborn`
2. Navigate to the project directory.
3. Run the dashboard: `streamlit run app.py`
