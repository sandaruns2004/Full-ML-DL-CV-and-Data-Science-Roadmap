# 📉 Precision-Recall Curves

> **Prerequisites**: ROC & AUC | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 📋 Table of Contents
1. [Why not just use ROC?](#1-why-not-just-use-roc)
2. [Evaluating Imbalanced Data](#2-evaluating-imbalanced-data)

---

## 1. Why not just use ROC?

### 🟢 Beginner
**Simple Explanation**: 
If you are looking for needles in a haystack, you care about the needles you found (True Positives), the needles you missed (False Negatives), and the pieces of hay you mistakenly thought were needles (False Positives). You **DO NOT** care about how many pieces of hay you correctly identified as hay (True Negatives). 
ROC curves use True Negatives in their math. Precision-Recall curves ignore True Negatives entirely.

### 🟡 Intermediate
**Workflow and Practical Implementation**: 
The PR Curve plots **Precision** on the Y-axis and **Recall** on the X-axis at different probability thresholds.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

y_prob = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```
The goal is to be in the **Top-Right** corner (High Precision, High Recall).

### 🔴 Advanced
**Average Precision (AP) vs AUC-PR**:
Instead of ROC AUC, we calculate the Area Under the PR Curve (AUC-PR) or Average Precision (AP). 
$AP = \sum_n (R_n - R_{n-1}) P_n$
Where $P_n$ and $R_n$ are the precision and recall at the $n$-th threshold.

**Industry Application**: 
In Credit Card Fraud detection, where anomalies are 0.17% of the data, an ROC AUC might look like 0.98 (excellent), while the PR-AUC might be 0.40 (terrible). Always use PR curves for minority-class detection problems in production systems.

---

[← ROC Curves and AUC](07-ROC-And-AUC.md) | [Back to Index](../README.md) | [Next: Cross Validation →](09-Cross-Validation.md)
