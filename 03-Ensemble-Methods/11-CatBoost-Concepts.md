# 🐈 CatBoost

> **Prerequisites**: Gradient Boosting | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [What is CatBoost?](#1-what-is-catboost)
2. [Key Innovations](#2-key-innovations)
3. [Python Implementation](#3-python-implementation)

---

## 1. What is CatBoost?

### 🟢 Beginner
**Simple Explanation**: Developed by Yandex. If your data has a lot of category columns (like names, cities, colors), traditional models require you to convert them into numbers first (like One-Hot Encoding). CatBoost handles category columns natively without you needing to do any preprocessing!

---

## 2. Key Innovations

### 🔴 Advanced
CatBoost resolves common gradient boosting issues via three core innovations:

#### 2.1 Native Categorical Handling (Ordered Target Statistics)
CatBoost uses a specialized version of target encoding to convert categorical features into numerical values. Traditional target encoding causes target leakage and prediction shift. CatBoost solves this by computing target statistics on random permutations of the dataset:

$$x_{i,k} \approx \frac{\sum_{j: \sigma(j) < \sigma(i)} \mathbb{1}_{\{x_{\sigma(j),k} = x_{\sigma(i),k}\}} \cdot y_{\sigma(j)} + s \cdot p}{\sum_{j: \sigma(j) < \sigma(i)} \mathbb{1}_{\{x_{\sigma(j),k} = x_{\sigma(i),k}\}} + s}$$

where $\sigma$ is a random permutation of the data, $p$ is a prior, and $s$ is the prior weight. This ensures that the target statistic for sample $i$ only uses target values of samples that appear *before* $i$ in the permutation, eliminating target leakage.

#### 2.2 Ordered Boosting
Traditional boosting has a target leakage problem because the target of a sample is used to calculate its own residual. CatBoost uses "Ordered Boosting" (a permutation-driven approach) to ensure the residual calculation for sample $i$ only uses models trained on samples before $i$.

#### 2.3 Symmetric Trees (Oblivious Trees)
CatBoost builds balanced/symmetric trees, where the same split criterion is used for all nodes at the same level of the tree.
- **Symmetric split**: This acts as a regularizer, preventing overfitting.
- **Fast inference**: The tree structure can be evaluated using simple bitwise operations, which makes model execution extremely fast in production systems!

---

## 3. Python Implementation

To run this code, make sure to install CatBoost: `pip install catboost`.

```python
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train CatBoost
model = CatBoostClassifier(
    iterations=100,
    depth=6,
    learning_rate=0.1,
    l2_leaf_reg=3,
    random_state=42,
    verbose=0
)
model.fit(X_train, y_train)
print(f"CatBoost Test Accuracy: {model.score(X_test, y_test):.4f}")
```

---

[← LightGBM Concepts](./10-LightGBM-Concepts.md) | [Back to Index](../README.md) | [Next: Stacking →](./12-Stacking.md)
