# Customer Retention Prediction

## Business Problem
Our e-commerce platform has been experiencing a high churn rate among newly registered users. Retaining an existing customer is 5x cheaper than acquiring a new one. The goal of this project is to predict whether a customer will be retained or if they will churn, allowing the marketing team to offer targeted incentives.

## Dataset Description
We will use a synthetic dataset simulating e-commerce behavior, consisting of user demographic data, purchase history, website engagement, and customer service interactions. 

## Approach
This project leverages **Bagging (Bootstrap Aggregating)** algorithms. We focus on stabilizing high-variance models (Decision Trees) by training multiple independent models on variations of the data and averaging their predictions.

## Project Structure
- `data/`: Contains the raw and processed datasets.
- `notebooks/`: Jupyter notebooks for EDA, Feature Engineering, and Model Training.
- `src/`: Python scripts for data processing and model inference pipelines.
- `models/`: Saved `joblib` or `pickle` model files.
- `reports/`: Evaluation reports and metric outputs.
- `images/`: Visualizations for feature importance and error analysis.

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the main training notebook in `notebooks/`.
