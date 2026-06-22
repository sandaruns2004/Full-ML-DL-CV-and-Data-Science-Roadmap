# ⚡ LightGBM

> **Prerequisites**: XGBoost | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [What is LightGBM?](#1-what-is-lightgbm)
2. [Key Innovations](#2-key-innovations)
3. [Python Implementation](#3-python-implementation)

---

## 1. What is LightGBM?

### 🟢 Beginner
**Simple Explanation**: When datasets get extremely massive (millions of rows), XGBoost can be slow. LightGBM (by Microsoft) uses a clever trick of putting continuous numbers into "bins" (histograms) to speed up calculations, and it grows trees differently to focus on the highest error reduction first.

---

## 2. Key Innovations

### 🔴 Advanced
LightGBM addresses the GBDT training bottlenecks via four key optimizations:

#### 2.1 Leaf-Wise Growth (Best-First)
Unlike XGBoost which grows trees level-by-level (depth-wise), LightGBM grows trees **leaf-wise**. It chooses the leaf with the maximum delta loss to split, regardless of depth. This results in faster convergence and lower loss, but requires strict tuning (e.g. `max_depth` or `num_leaves` constraints) to prevent overfitting.

```
Level-Wise Growth (XGBoost):
       O                 O
      / \      ───>     / \
     O   O             O   O
                      / \ / \
                     O  O O  O (Balanced)

Leaf-Wise Growth (LightGBM):
       O                 O
      / \      ───>     / \
     O   O             O   O
                          /
                         O   
                        /
                       O  (Asymmetric, targets higher loss reduction)
```

#### 2.2 Histogram-Based Splitting
Instead of sorting continuous features to find optimal split points (complexity $O(\#\text{data} \times \#\text{features})$), LightGBM bins continuous features into discrete integer bins (typically 255 bins). This reduces the split-finding complexity to $O(\#\text{bins} \times \#\text{features})$, which is significantly faster and uses less memory.

#### 2.3 Gradient-based One-Side Sampling (GOSS)
In GBDT, instances with small gradients are already well-trained (small errors). GOSS keeps all instances with large gradients and performs random sampling on the instances with small gradients. It introduces a constant multiplier for the sampled data with small gradients to preserve the original data distribution. This significantly accelerates training without sacrificing accuracy.

#### 2.4 Exclusive Feature Bundling (EFB)
High-dimensional sparse datasets often contain mutually exclusive features (they rarely take non-zero values simultaneously, like one-hot encoded columns). EFB bundles these exclusive features into a single feature, drastically reducing the feature dimension without any information loss.

---

## 3. Python Implementation

To run this code, make sure to install LightGBM: `pip install lightgbm`.

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train LightGBM
model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=-1,          # No limit (leaf-wise growth controls this)
    num_leaves=31,         # Max leaves in one tree
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)
model.fit(X_train, y_train)
print(f"LightGBM Test Accuracy: {model.score(X_test, y_test):.4f}")
```

---

[← XGBoost (eXtreme Gradient Boosting)](09-XGBoost-Concepts.md) | [Back to Index](../README.md) | [Next: CatBoost →](11-CatBoost-Concepts.md)
