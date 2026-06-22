# 🚨 Anomaly Detection (Outlier Detection)

> **Prerequisites**: Intro to DS | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 1. Finding the Needle in the Haystack

### 🟢 Beginner
**Simple Explanation**: Anomaly detection is the process of finding the "weird" data points. Imagine a factory producing perfectly round wheels. If a wheel comes out shaped like an oval, the anomaly detection system flags it. 

### 🟡 Intermediate
**Types of Methods**:
- **Statistical**: Uses Z-Scores or Interquartile Range (IQR). Assumes data is normally distributed.
- **Distance-Based**: Uses algorithms like KNN to find points that are far away from any neighbors.
- **Density-Based**: Finds points situated in low-density regions (LOF).

### 🔴 Advanced
**Applications & Challenges**: 
Used heavily in Credit Card Fraud, Cyber Intrusion Detection, and Predictive Maintenance. The main challenge in industry is the massive class imbalance (anomalies might be 0.001% of the data) and the cost of False Positives (alert fatigue).

---

[← Apriori Algorithm](11-Apriori-Algorithm.md) | [Back to Index](../README.md) | [Next: Isolation Forest →](13-Isolation-Forest.md)
