# 📏 Metrics and Evaluation: Beyond Accuracy

> **Prerequisites**: Supervised Learning, Probabilities | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Accuracy Paradox](#1-the-accuracy-paradox)
2. [Confusion Matrix deep dive](#2-confusion-matrix-deep-dive)
3. [Precision, Recall, and the F-Beta Score](#3-precision-recall-and-the-f-beta-score)
4. [ROC-AUC vs Precision-Recall Curve](#4-roc-auc-vs-precision-recall-curve)
5. [Probabilistic Metrics (Log Loss & Brier Score)](#5-probabilistic-metrics-log-loss--brier-score)
6. [Cohen's Kappa (Inter-rater reliability)](#6-cohens-kappa-inter-rater-reliability)
7. [Regression Metrics (MSE, MAE, Huber)](#7-regression-metrics-mse-mae-huber)
8. [Business Impact of Metrics](#8-business-impact-of-metrics)
9. [Project Ideas & What's Next](#9-project-ideas--whats-next)

---

## 1. The Accuracy Paradox

Imagine a dataset detecting a rare genetic mutation present in exactly 1 in 10,000 people. 
If we build a "dumb" model that simply outputs `0` (No Mutation) for every single patient, its accuracy will be:
$$\text{Accuracy} = \frac{9,999}{10,000} = 99.99\%$$

Despite having 99.99% accuracy, the model is completely useless for its intended purpose. This is the **Accuracy Paradox**. In heavily imbalanced datasets, accuracy provides a false sense of security.

---

## 2. Confusion Matrix deep dive

The foundation of classification metrics is the Confusion Matrix.

| | Predicted Positive ($\hat{y}=1$) | Predicted Negative ($\hat{y}=0$) |
|---|---|---|
| **Actual Positive ($y=1$)** | True Positive (TP) | False Negative (FN) *Type II Error* |
| **Actual Negative ($y=0$)** | False Positive (FP) *Type I Error*| True Negative (TN) |

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Imbalanced mock predictions
y_true = np.array([0]*90 + [1]*10)
y_pred = np.array([0]*85 + [1]*5 + [0]*2 + [1]*8) # 5 FP, 2 FN, 8 TP, 85 TN

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
```

---

## 3. Precision, Recall, and the F-Beta Score

### Precision (Quality)
Of all the items the model *claimed* were positive, how many were *actually* positive?
$$\text{Precision} = \frac{TP}{TP + FP}$$
*Penalty on False Positives.*

### Recall / Sensitivity / True Positive Rate (Quantity)
Of all the *actual* positive items, how many did the model *find*?
$$\text{Recall} = \frac{TP}{TP + FN}$$
*Penalty on False Negatives.*

### F-Beta Score
You cannot maximize both simultaneously (Threshold trade-off). The **F-measure** combines them into a harmonic mean. 
$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

- $\beta = 1$ ($F1$ Score): Precision and Recall are weighted equally.
- $\beta = 2$ ($F2$ Score): Recall is considered twice as important as Precision (e.g., Cancer screening).
- $\beta = 0.5$ ($F0.5$ Score): Precision is considered twice as important as Recall (e.g., YouTube Recommendations).

---

## 4. ROC-AUC vs Precision-Recall Curve

Both curves evaluate a model's performance across *all possible classification thresholds* (from 0.0 to 1.0).

### Receiver Operating Characteristic (ROC)
Plots **True Positive Rate (Recall)** against **False Positive Rate (FPR)**.
$$FPR = \frac{FP}{FP + TN} = 1 - \text{Specificity}$$

**Area Under Curve (AUC)**: Represents the probability that the model will score a randomly chosen positive class higher than a randomly chosen negative class.
- $AUC = 1.0$: Perfect ranking.
- $AUC = 0.5$: Random guessing.

### Precision-Recall (PR) Curve
Plots **Precision** against **Recall**. 
*Crucial difference*: ROC curves can be overly optimistic on highly imbalanced datasets because the massive amount of True Negatives dilutes the False Positive Rate. **PR Curves ignore True Negatives entirely**, focusing only on the minority positive class.

```python
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Mock probabilities
np.random.seed(42)
y_scores_0 = np.random.normal(0.3, 0.1, 90)
y_scores_1 = np.random.normal(0.7, 0.15, 10)
y_scores = np.concatenate([y_scores_0, y_scores_1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ROC
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate (Recall)')
ax1.set_title('ROC Curve')
ax1.legend()
ax1.grid(alpha=0.3)

# PR Curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)
ax2.plot(recall, precision, color='green', lw=2, label=f'Avg Precision = {ap:.3f}')
baseline = len(y_true[y_true==1]) / len(y_true)
ax2.axhline(baseline, color='navy', lw=2, linestyle='--', label=f'Baseline ({baseline:.1f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curve (Imbalanced Data)')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=150)
plt.show()
```

---

## 5. Probabilistic Metrics (Log Loss & Brier Score)

Accuracy only cares about the hard label (e.g., `0` or `1`). Probabilistic metrics care about the model's *confidence*. A model predicting a positive class with 51% confidence vs 99% confidence shouldn't be scored identically.

### Cross-Entropy Loss (Log Loss)
Heavily penalizes confident but wrong predictions.
$$L(y, p) = -(y \log(p) + (1-y) \log(1-p))$$

### Brier Score
Mean Squared Error of the predicted probabilities. Lower is better. Often used in weather forecasting.
$$BS = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$

---

## 6. Cohen's Kappa (Inter-rater reliability)

Cohen's Kappa normalizes accuracy by the accuracy that would have been achieved by random chance.
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
- $p_o$: Observed agreement (Accuracy).
- $p_e$: Expected agreement by chance.
- $\kappa \leq 0$: Worse than chance.
- $\kappa = 1$: Perfect agreement.
Excellent for checking if an annotator is actually good, or if a model is genuinely beating a random baseline on skewed data.

---

## 7. Regression Metrics (MSE, MAE, Huber)

| Metric | Formula | Business Interpretation |
|--------|---------|-------------------------|
| **MAE** (Mean Absolute Error) | $\frac{1}{n}\sum\|y - \hat{y}\|$ | "On average, our model is off by X dollars." Robust to outliers. |
| **MSE** (Mean Squared Error) | $\frac{1}{n}\sum(y - \hat{y})^2$ | "We want to heavily punish catastrophic errors." Sensitive to outliers. |
| **RMSE** (Root Mean Squared Error) | $\sqrt{MSE}$ | MSE translated back into original units (e.g., dollars). |
| **MAPE** (Mean Abs Percentage Error)| $\frac{100}{n}\sum\|\frac{y - \hat{y}}{y}\|$ | "Our model is off by X%". Intuitive for executives. Fails if $y=0$. |
| **$R^2$** (Coefficient of Det.) | $1 - \frac{SS_{res}}{SS_{tot}}$ | "Our model explains X% of the variance in the target." |

### Huber Loss (Smooth L1 Loss)
Combines the best of MAE and MSE. It is quadratic (like MSE) for small errors, and linear (like MAE) for large errors. Prevents exploding gradients during training.
$$L_\delta(y, f(x)) = \begin{cases} \frac{1}{2}(y - f(x))^2 & \text{for } |y - f(x)| \leq \delta \\ \delta|y - f(x)| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

---

## 8. Business Impact of Metrics

Aligning the ML metric with the business Key Performance Indicator (KPI) is the most critical step in applied ML.

1. **Credit Card Fraud Detection**:
   - False Negative: Bank loses $1,000.
   - False Positive: Customer is slightly annoyed, bank texts them.
   - **Optimization Target**: Maximize Recall (or optimize $F_2$ score).
2. **Algorithmic Trading**:
   - False Positive: Algorithm buys a stock that crashes. Massive financial loss.
   - False Negative: Algorithm misses a trade. Opportunity cost, but no actual capital loss.
   - **Optimization Target**: Maximize Precision.
3. **Medical Triage**:
   - A probability model predicting ICU survival chance.
   - **Optimization Target**: Brier Score / Log Loss (Calibrated probabilities matter more than hard thresholds).

---

## 9. Project Ideas & What's Next

### Project Ideas
- 🟢 **Threshold Optimization Engine**: Write a Python class that takes true labels and model probabilities, loops through all 100 thresholds (0.00 to 1.00), and outputs the optimal threshold to maximize the F-Beta score, where Beta is a user-defined argument.

### What's Next
| Next | Why |
|------|-----|
| [Cross-Validation](./02-Cross-Validation.md) | We know how to score models now. How do we ensure those scores are statistically robust and unbiased? |

---

[← Gaussian Mixture Models](../04-Unsupervised-Learning/04-Gaussian-Mixture-Models.md) | [Back to Index](../README.md) | [Next: Cross Validation →](./02-Cross-Validation.md)
