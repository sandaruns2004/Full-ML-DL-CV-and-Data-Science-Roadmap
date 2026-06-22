# 🧭 Model Selection Guide

> **Prerequisites**: Supervised Learning | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 1. How to Choose an Algorithm?

### 🟢 Beginner
**Simple Explanation**: No single model works best for everything (No Free Lunch Theorem). You choose based on your data size and whether you need to explain the model to a human.

### 🟡 Intermediate
**Heuristics**:
- **Small Data / Explainable**: Decision Trees, Logistic Regression.
- **Text Classification**: Naive Bayes, SVM.
- **Tabular Data (Kaggle)**: Random Forest, XGBoost, LightGBM.

### 🔴 Advanced
**Mathematical Complexity**:
Consider training time $O(n^2)$ vs inference time. KNN is $O(1)$ to train but $O(nd)$ to predict, whereas Logistic Regression is $O(d)$ to predict. In low-latency systems (e.g. ad bidding), prediction time is the ultimate constraint.

---

[← Model Building Pipeline](11-Model-Building-Pipeline.md) | [Back to Index](../README.md) | [Next: Introduction To Ensemble Learning →](../03-Ensemble-Methods/01-Introduction-To-Ensemble-Learning.md)
