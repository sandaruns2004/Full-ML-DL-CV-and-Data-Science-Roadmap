# 🕵️ Credit Card Fraud Detection

## Overview
This project uses unsupervised anomaly detection to identify fraudulent credit card transactions. Traditional supervised learning struggles here because fraud is extremely rare (highly imbalanced data). Isolation Forest is perfectly suited to isolate these rare events rapidly in high-dimensional space.

## Architecture
```mermaid
graph TD
    A[Credit Card Transactions \n Highly Imbalanced] --> B[Data Preprocessing \n Scaling]
    B --> C[Isolation Forest \n Build Random Trees]
    B --> D[Local Outlier Factor \n Contextual Anomalies]
    C --> E[Calculate Anomaly Scores]
    D --> E
    E --> F[Threshold Optimization \n Precision-Recall Tradeoff]
    F --> G[Streamlit Dashboard \n Alert Monitoring]
    
    style A fill:#4c566a,color:#eceff4
    style G fill:#bf616a,color:#eceff4
```

## Project Structure
*   `data/`: Transaction logs (e.g., creditcard.csv).
*   `notebooks/`: Comparison between Isolation Forest and LOF.
*   `src/`: Python scripts for model training and inference.
*   `app.py`: Streamlit dashboard for real-time transaction monitoring.

## How to Run
1. Install dependencies: `pip install streamlit scikit-learn pandas numpy matplotlib seaborn`
2. Run the dashboard: `streamlit run app.py`
