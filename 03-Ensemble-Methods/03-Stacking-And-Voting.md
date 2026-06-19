# 🏗️ Stacking & Voting Ensembles

> **Prerequisites**: Random Forest, Boosting | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Voting Classifiers](#1-voting-classifiers)
2. [Stacking (Stacked Generalization)](#2-stacking-stacked-generalization)
3. [Blending](#3-blending)
4. [Implementation](#4-implementation)
5. [Project Ideas & What's Next](#5-project-ideas--whats-next)

---

## 1. Voting Classifiers

### Hard Voting
Majority vote among models:
$$\hat{y} = \text{mode}(h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_K(\mathbf{x}))$$

### Soft Voting
Average predicted probabilities, pick the class with highest average:
$$\hat{y} = \arg\max_c \frac{1}{K}\sum_{k=1}^{K} P_k(y = c | \mathbf{x})$$

Soft voting usually works better (uses confidence information).

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Define base models
estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Hard voting
hard_voting = VotingClassifier(estimators=estimators, voting='hard')
# Soft voting
soft_voting = VotingClassifier(estimators=estimators, voting='soft')

# Compare
print(f"{'Model':<30} {'CV Score':>10}")
print("-" * 42)
for name, model in estimators + [('Hard Voting', hard_voting), ('Soft Voting', soft_voting)]:
    scores = cross_val_score(model, data.data, data.target, cv=5, scoring='accuracy')
    print(f"{name:<30} {scores.mean():>10.4f} ± {scores.std():.4f}")
```

---

## 2. Stacking (Stacked Generalization)

Stacking uses **predictions from base models as features** for a **meta-learner**:

```
Layer 1 (Base Models):
  Model A → prediction_A
  Model B → prediction_B     →  [pred_A, pred_B, pred_C]  →  Meta-learner → Final prediction
  Model C → prediction_C
```

**Key**: Use **cross-validation** for base model predictions to avoid overfitting!

### The Math

1. Split training data into $K$ folds
2. For each base model $m$ and fold $k$:
   - Train on $K-1$ folds
   - Predict on fold $k$
3. Use these out-of-fold predictions as features for the meta-learner
4. Train meta-learner on these new features

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Base models
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(probability=True, random_state=42))
]

# Stacking with Logistic Regression as meta-learner
stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,                    # 5-fold CV for base predictions
    stack_method='predict_proba',  # Use probabilities, not hard predictions
    n_jobs=-1
)

stacking.fit(X_train, y_train)
print(f"Stacking test accuracy: {stacking.score(X_test, y_test):.4f}")

# Compare with individual models
print(f"\nIndividual model scores:")
for name, model in base_estimators:
    model.fit(X_train, y_train)
    print(f"  {name}: {model.score(X_test, y_test):.4f}")
```

---

## 3. Blending

Blending is a simpler version of stacking — uses a **held-out validation set** instead of cross-validation:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X_train_full, X_test, y_train_full, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Split training into train and blend sets
X_train, X_blend, y_train, y_blend = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

# Train base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)

# Create blend features (predictions on blend set)
blend_features = np.column_stack([
    rf.predict_proba(X_blend)[:, 1],
    gb.predict_proba(X_blend)[:, 1]
])

# Train meta-learner on blend features
meta = LogisticRegression()
meta.fit(blend_features, y_blend)

# Predict on test set
test_features = np.column_stack([
    rf.predict_proba(X_test)[:, 1],
    gb.predict_proba(X_test)[:, 1]
])
print(f"Blending accuracy: {meta.score(test_features, y_test):.4f}")
```

---

## 4. Implementation

### When to Use What

| Method | Pros | Cons | Use When |
|--------|------|------|----------|
| **Voting** | Simple, fast | Limited improvement | Quick baseline combination |
| **Stacking** | Most powerful | Complex, slow training | Competitions, maximum accuracy |
| **Blending** | Simpler than stacking | Wastes training data | Moderate improvement needed |

### Tips for Effective Ensembles

1. **Diversity**: Use different algorithm types (tree + linear + distance-based)
2. **Quality**: Each base model should be reasonably good
3. **Simple meta-learner**: Logistic Regression often works best (avoids overfitting)
4. **Cross-validation**: Always use CV for stacking to prevent leakage

---

## 5. Project Ideas & What's Next

### Project Ideas
- 🟢 **Model Comparison Framework** — Compare voting vs stacking
- 🟡 **Kaggle Ensemble Strategy** — Build a multi-layer stacking system
- 🔴 **Auto-Ensemble Builder** — Automatically select and combine best models

### What's Next
| Next | Why |
|------|-----|
| [Clustering](../04-Unsupervised-Learning/01-Clustering.md) | Unsupervised learning |
| [Model Evaluation](../05-Model-Evaluation/01-Metrics-And-Evaluation.md) | Properly evaluate your ensembles |

---

[← Boosting](./02-Boosting.md) | [Back to Index](../README.md) | [Next: Clustering →](../04-Unsupervised-Learning/01-Clustering.md)
