# 🎛️ Blending

> **Difficulty**: ⭐⭐⭐☆☆ Advanced | **Prerequisites**: Stacking

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Stacking vs Blending](#3-stacking-vs-blending)
4. [Algorithm Workflow](#4-algorithm-workflow)
5. [Python Implementation](#5-python-implementation)

---

## 1. What Problem Does This Solve?

### 🟢 Beginner
Stacking is very powerful, but doing Cross-Validation (training every model 5 times) is extremely slow and requires a lot of code. Blending is a shortcut to Stacking that gets you 90% of the performance with 10% of the computational cost.

### 🟡 Intermediate
Blending replaces the complex Out-of-Fold (OOF) prediction strategy of Stacking with a simple hold-out validation set. You train the base models once, make predictions on the validation set once, and train the meta-learner on those validation predictions.

### 🔴 Advanced
By avoiding K-Fold CV, Blending significantly reduces the risk of information leak into the meta-learner. However, it requires a large dataset, because you must sacrifice a portion of your training data purely for the hold-out validation set.

---

## 2. Intuition

Instead of making the students take 5 practice tests and averaging their results (Stacking), we just give them one giant mid-term exam (the hold-out set) to figure out who is best at what subject, and use that knowledge for the final exam.

---

## 3. Stacking vs Blending

| Feature | Stacking | Blending |
| :--- | :--- | :--- |
| **Validation Strategy** | Out-of-Fold (K-Fold CV) | Single Hold-Out Set |
| **Data Efficiency** | High (uses all data for training) | Low (must sacrifice a validation set) |
| **Computational Cost** | High (trains base models $K$ times) | Low (trains base models 1 time) |
| **Leakage Risk** | Moderate (if CV isn't perfect) | Very Low |
| **Best Used When...** | Dataset is small to medium | Dataset is massive |

---

## 4. Algorithm Workflow

1. Split the dataset into three parts: **Train**, **Validation (Blend)**, and **Test**.
2. Train all base models (Layer 0) on the **Train** set.
3. Have the base models make predictions on the **Validation (Blend)** set.
4. Train the meta-learner (Layer 1) using the true labels of the Validation set and the predictions from step 3 as features.
5. For final inference on the **Test** set: have base models predict on the Test set, and pass those predictions to the meta-learner.

---

## 5. Python Implementation

Here is an end-to-end classification example of Blending using standard Scikit-Learn:

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

[← Stacking](12-Stacking.md) | [Back to Index](../README.md) | [Next: Model Selection For Ensembles →](14-Model-Selection-For-Ensembles.md)
