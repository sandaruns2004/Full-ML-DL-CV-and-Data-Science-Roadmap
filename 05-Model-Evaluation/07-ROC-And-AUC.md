# 📈 ROC Curves and AUC

> **Prerequisites**: Confusion Matrix | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 📋 Table of Contents
1. [What is an ROC Curve?](#1-what-is-an-roc-curve)
2. [Threshold Analysis](#2-threshold-analysis)
3. [The Area Under the Curve (AUC)](#3-the-area-under-the-curve)

---

## 1. What is an ROC Curve?

### 🟢 Beginner
**Simple Explanation**: 
Many machine learning models don't just say "Yes" or "No". They give a probability, like "There is a 70% chance this email is spam." 
By default, we say anything > 50% is spam. But what if we change that rule to > 90%? 
The **Receiver Operating Characteristic (ROC)** curve is a plot that shows how well the model works as you slide that threshold from 0% all the way up to 100%.

### 🟡 Intermediate
**Workflow**: 
The ROC curve plots the **True Positive Rate (Recall)** on the Y-axis against the **False Positive Rate** on the X-axis.
- **Top Left Corner (0, 1)**: The perfect model. 100% True Positives and 0% False Positives.
- **Diagonal Line**: A completely random guess (e.g., flipping a coin).

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# You must pass PROBABILITIES, not hard predictions!
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
```

### 🔴 Advanced
**The Area Under the Curve (AUC)**:
The AUC is a single number summarizing the ROC curve.
- $AUC = 1.0$: Perfect classifier.
- $AUC = 0.5$: Worthless classifier.

**Statistical Interpretation**: The AUC represents the probability that the classifier will rank a randomly chosen positive instance higher than a randomly chosen negative instance. 
**Crucial Industry Note**: ROC AUC can be highly misleading when dealing with severely imbalanced datasets. If you have 1,000,000 negatives and 10 positives, a massive change in False Positives barely moves the False Positive Rate on the X-axis. In these cases, Precision-Recall curves are mandatory.
