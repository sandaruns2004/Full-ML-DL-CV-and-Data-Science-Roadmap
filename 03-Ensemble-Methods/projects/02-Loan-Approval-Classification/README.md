# Loan Approval Classification

## Business Problem
Our financial institution needs a reliable, low-variance model to predict whether a loan applicant will default on their loan. False positives (approving a bad loan) cost the bank heavily, while false negatives (rejecting a good customer) damage our reputation and lose potential interest revenue. 

## Dataset Description
We use a mock dataset mimicking credit reports, which includes age, income, employment length, credit utilization, previous defaults, and requested loan amount.

## Approach
This project leverages **Bagging classifiers**, which are uniquely suited for credit risk. While deep decision trees might overfit to noise in credit histories, Bagging averages out these anomalies, leading to highly stable decision boundaries without sacrificing model depth.

## Project Structure
- `data/`: Raw and processed credit data.
- `notebooks/`: Jupyter notebooks for EDA and Model Training.
- `src/`: Python scripts for data ingestion and API endpoints.
- `models/`: Saved model objects.
- `reports/`: Audit and fairness reports.
- `images/`: SHAP plots and ROC curves.

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Explore `notebooks/` to view the training pipeline.
