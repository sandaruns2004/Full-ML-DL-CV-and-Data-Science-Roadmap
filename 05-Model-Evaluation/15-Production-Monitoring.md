# 🏭 Production Monitoring and Drift

> **Prerequisites**: Model Evaluation | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [Models Decay Over Time](#1-models-decay-over-time)
2. [Data Drift vs Concept Drift](#2-data-drift-vs-concept-drift)
3. [Monitoring in Industry](#3-monitoring-in-industry)

---

## 1. Models Decay Over Time

### 🟢 Beginner
**Simple Explanation**: 
If you train a model to predict housing prices in 2019, it might have 99% accuracy. If you use that exact same model in 2023, its predictions will be completely wrong because the economy changed, inflation happened, and people moved to different cities. Models don't stay smart forever; they decay as the world changes.

### 🟡 Intermediate
**Data Drift vs Concept Drift**: 
- **Data Drift (Feature Drift)**: The input data changes. For example, a sensor starts breaking and sending noisy data, or a new marketing campaign brings in a completely different demographic of users.
- **Concept Drift**: The actual definition of the target variable changes. For example, during COVID-19, the concept of "normal purchasing behavior" changed entirely. The data might look the same, but the outcome mapping has shifted.

### 🔴 Advanced
**Industry Best Practices (MLOps)**:
In production, you cannot calculate Accuracy or F1-Score in real-time because you don't immediately know the true labels (e.g., you don't know if a loan will default until 6 months later). 

Instead, MLOps Engineers use statistical divergence metrics to measure if the live incoming data looks mathematically different from the data the model was trained on:
- **Population Stability Index (PSI)**
- **Kullback-Leibler (KL) Divergence**
- **Kolmogorov-Smirnov (K-S) Test**

```python
# Conceptual Monitoring Alert System
def detect_drift(reference_data, live_data, threshold=0.05):
    from scipy.stats import ks_2samp
    
    # K-S Test checks if two samples are drawn from the same distribution
    statistic, p_value = ks_2samp(reference_data, live_data)
    
    if p_value < threshold:
        return "ALERT: Data Drift Detected! Retraining required."
    return "Status: Normal"
```
When drift is detected, an automated CI/CD pipeline triggers the model to retrain on the newest data, evaluate itself against a holdout set, and deploy the new version (Shadow Deployment or Canary Release) if the metrics improve.

---

[← Model Comparison and Statistical Significance](14-Model-Comparison.md) | [Back to Index](../README.md) | [Next: Introduction to Neural Networks →](../06-Neural-Networks-Foundations/01-Introduction-To-Neural-Networks.md)
