# 🎛️ Blending

> **Prerequisites**: Stacking | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 📋 Table of Contents
1. [Holdout-Based Ensembling](#1-holdout-based-ensembling)
2. [Stacking vs Blending](#2-stacking-vs-blending)
3. [Python Implementation](#3-python-implementation)

---

## 1. Holdout-Based Ensembling

### 🟢 Beginner
**Simple Explanation**: Blending is very similar to Stacking, but simpler. Instead of running complex, slow cross-validation folds to get predictions for the meta-learner, we just split the training data once. We train our base models on part of the data, make predictions on a "validation set" (which the models haven't seen), and train the meta-learner on those validation predictions.

---

## 2. Stacking vs Blending

### 🟡 Intermediate
- **Stacking**: Uses Out-of-Fold predictions via $K$-Fold cross-validation.
  - *Pros*: Uses data very efficiently (every point is used both to train base models and as a validation point for the meta-learner).
  - *Cons*: Slow and computationally expensive (must train each base model $K$ times).
- **Blending**: Uses a single holdout validation set.
  - *Pros*: Much faster and simpler to implement.
  - *Cons*: Wastes data because the base models are trained on a smaller portion of the dataset, and predictions are only made on a subset.

---

## 3. Python Implementation

Here is an end-to-end classification example of Blending:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()
X_train_full, X_test, y_train_full, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Split training set into train and blend sets
X_train, X_blend, y_train, y_blend = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

# Train base models on the train set
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Create blend features (predictions of base models on the validation/blend set)
blend_features = np.column_stack([
    rf.predict_proba(X_blend)[:, 1],
    gb.predict_proba(X_blend)[:, 1]
])

# Train meta-learner on the blend features
meta = LogisticRegression()
meta.fit(blend_features, y_blend)

# Predict on test set
test_features = np.column_stack([
    rf.predict_proba(X_test)[:, 1],
    gb.predict_proba(X_test)[:, 1]
])
print(f"Blending Test Accuracy: {meta.score(test_features, y_test):.4f}")
```

---

[← Stacking](./12-Stacking.md) | [Back to Index](../README.md) | [Next: Ensemble Selection Strategies →](./14-Model-Selection-For-Ensembles.md)
