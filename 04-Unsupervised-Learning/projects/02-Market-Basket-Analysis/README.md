# 🛒 Market Basket Analysis

## Overview
This project applies the Apriori algorithm and Association Rule Mining to uncover product relationships within customer transaction data. It identifies products frequently bought together, which is critical for recommendation engines, store layout optimization, and cross-selling campaigns.

## Architecture
```mermaid
graph TD
    A[Raw Transaction Data] --> B[Data Preprocessing \n One-Hot Encoding]
    B --> C[Apriori Algorithm \n Find Frequent Itemsets]
    C --> D[Association Rule Mining \n Calculate Support, Confidence, Lift]
    D --> E[Filter Rules \n (e.g., Lift > 1.2)]
    E --> F[Streamlit Dashboard \n Network Visualization]
    
    style A fill:#4c566a,color:#eceff4
    style F fill:#88c0d0,color:#2e3440
```

## Project Structure
*   `data/`: Transaction datasets (e.g., groceries.csv).
*   `notebooks/`: Exploratory Data Analysis and Rule Generation notebooks.
*   `src/`: Python scripts for transaction processing.
*   `app.py`: Streamlit dashboard for exploring rules dynamically.

## How to Run
1. Install dependencies: `pip install streamlit pandas matplotlib seaborn mlxtend networkx`
2. Run the dashboard: `streamlit run app.py`
