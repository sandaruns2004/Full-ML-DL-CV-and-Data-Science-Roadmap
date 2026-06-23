# 🏭 Industrial Sensor Anomaly Detection

## Overview
This project focuses on predictive maintenance in an industrial setting. By applying the Local Outlier Factor (LOF) algorithm to multi-dimensional sensor telemetry (temperature, vibration, pressure), we can detect contextual anomalies—situations where a sensor reading might be globally "normal" but is anomalous relative to the current state of its neighboring sensors.

## Architecture
```mermaid
graph TD
    A[IoT Sensor Telemetry \n (Temp, Vibration, Pressure)] --> B[Time Series Preprocessing \n Moving Averages, Scaling]
    B --> C[Local Outlier Factor \n Calculate Density vs Neighbors]
    C --> D[Thresholding \n Flag High LOF Scores]
    D --> E[Alert Generation \n Warning: Impending Failure]
    E --> F[Streamlit Dashboard \n Real-Time Monitor]
    
    style A fill:#4c566a,color:#eceff4
    style F fill:#88c0d0,color:#2e3440
```

## Project Structure
*   `data/`: Telemetry datasets (e.g., factory_sensors.csv).
*   `notebooks/`: LOF parameter tuning and time-series EDA.
*   `src/`: Python scripts for stream processing and LOF inference.
*   `app.py`: Streamlit dashboard for real-time sensor monitoring.

## How to Run
1. Install dependencies: `pip install streamlit scikit-learn pandas numpy matplotlib seaborn`
2. Run the dashboard: `streamlit run app.py`
