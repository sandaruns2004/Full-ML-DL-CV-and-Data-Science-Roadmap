# 🎢 Validation Curves

> **Prerequisites**: Learning Curves | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 📋 Table of Contents
1. [Tuning Hyperparameters](#1-tuning-hyperparameters)
2. [Finding the Sweet Spot](#2-finding-the-sweet-spot)

---

## 1. Tuning Hyperparameters

### 🟢 Beginner
**Simple Explanation**: 
While a *Learning Curve* changes the amount of data on the X-axis, a *Validation Curve* changes a specific model setting (like the maximum depth of a tree) on the X-axis. It helps us find the "Goldilocks" setting—not too simple, not too complex.

### 🟡 Intermediate
**Workflow**: 
Plot the Training Score and Validation Score for varying values of a single hyperparameter.
- **Left Side of Graph (e.g., Depth=1)**: High error on both train and validation (Underfitting).
- **Right Side of Graph (e.g., Depth=50)**: Perfect score on train, terrible score on validation (Overfitting).
- **The Sweet Spot**: The exact point where the validation score hits its highest peak before dropping off.

### 🔴 Advanced
**Implementation**:

```python
from sklearn.model_selection import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

param_range = np.arange(1, 20, 2)
train_scores, test_scores = validation_curve(
    RandomForestClassifier(), X, y, param_name="max_depth", 
    param_range=param_range, cv=3, scoring="accuracy"
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(param_range, train_mean, label="Training score")
plt.plot(param_range, test_mean, label="Cross-validation score")
plt.title("Validation Curve for Random Forest Max Depth")
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
```
**Statistical Significance**: When plotting validation curves, always plot the variance band (`plt.fill_between` using standard deviation) to ensure the peak is statistically significant and not just a random artifact of the cross-validation folds.

---

[← Learning Curves](10-Learning-Curves.md) | [Back to Index](../README.md) | [Next: Hyperparameter Evaluation →](12-Hyperparameter-Tuning-Evaluation.md)
