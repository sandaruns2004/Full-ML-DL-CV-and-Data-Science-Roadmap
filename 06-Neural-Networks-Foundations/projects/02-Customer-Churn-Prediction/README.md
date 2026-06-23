# 📉 Project 02: Customer Churn Prediction

## 🎯 Goal
Build a business-focused Neural Network to predict whether a customer will leave a subscription service (Churn).

## 📝 Description
While computer vision (MNIST) is fun, the vast majority of machine learning jobs in the real world involve tabular data stored in SQL databases. In this project, you will take a messy dataset of customer demographics, usage metrics, and billing history, and build a deep learning model to flag high-risk customers.

## ✅ Requirements
- Perform Exploratory Data Analysis (EDA) to understand the factors driving churn.
- Preprocess the data (Handle missing values, One-Hot Encoding, StandardScaler).
- Build a PyTorch or TensorFlow model to classify `Churn: 1` vs `Retained: 0`.
- Address **Class Imbalance** (e.g., using weighted loss functions or SMOTE).
- Evaluate the model using business-critical metrics: Precision, Recall, and the F1-Score (Accuracy is misleading in churn prediction).

## 📁 Files
- `data/churn_dataset.csv` (Placeholder)
- `churn_prediction.ipynb`
- `model_deployment_script.py`
