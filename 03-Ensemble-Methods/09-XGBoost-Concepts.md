# ⚔️ XGBoost (eXtreme Gradient Boosting)

> **Prerequisites**: Gradient Boosting | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [What makes XGBoost "Extreme"?](#1-what-makes-xgboost-extreme)
2. [Mathematical Formulations & Regularization](#2-mathematical-formulations--regularization)
3. [Python Implementation & Feature Importance](#3-python-implementation--feature-importance)

---

## 1. What makes XGBoost "Extreme"?

### 🟢 Beginner
**Simple Explanation**: XGBoost is Gradient Boosting on steroids. It is extremely fast, handles missing data automatically, and has built-in protections against memorizing data (regularization). It was the dominant model for winning Kaggle competitions for many years.

---

## 2. Mathematical Formulations & Regularization

### 🔴 Advanced
The objective function of XGBoost at step $t$ is:

$$\mathcal{L}^{(t)} = \sum_{i=1}^n L(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$

where $\Omega(f_t)$ is the regularization term penalizing tree complexity:

$$\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$

- $T$ is the number of leaves.
- $w_j$ are the leaf weights.
- $\gamma$ is the complexity parameter (minimum loss reduction to split).
- $\lambda$ is the L2 regularization coefficient.

#### Second-Order Taylor Approximation
To optimize the objective for any arbitrary differentiable loss function, XGBoost uses a second-order Taylor expansion:

$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ L(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$

where:
- $g_i = \frac{\partial L(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}$ (First-order gradient)
- $h_i = \frac{\partial^2 L(y_i, \hat{y}_i^{(t-1)})}{\partial (\hat{y}_i^{(t-1)})^2}$ (Second-order gradient / Hessian)

By solving for the optimal leaf weight $w_j^*$, the optimal objective value (also called the split quality score) for a node containing instance set $I_j$ is:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

The quality gain of a split into Left ($L$) and Right ($R$) subsets is:

$$\text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

---

## 3. Python Implementation & Feature Importance

To run this code, make sure to install XGBoost: `pip install xgboost`.

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,          # L1 regularization
    reg_lambda=1.0,          # L2 regularization
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"XGBoost Test Accuracy: {model.score(X_test, y_test):.4f}")

# Plot Gain-based feature importance
xgb.plot_importance(model, max_num_features=10, importance_type='gain')
plt.title('XGBoost Feature Importance (Gain)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

[← Gradient Boosting (GBM)](08-Gradient-Boosting.md) | [Back to Index](../README.md) | [Next: LightGBM →](10-LightGBM-Concepts.md)
