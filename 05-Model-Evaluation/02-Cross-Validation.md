# 🔀 Cross-Validation, Data Splitting & Leakage

> **Prerequisites**: Metrics and Evaluation, Statistical Significance | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Mathematical Justification for CV](#1-the-mathematical-justification-for-cv)
2. [K-Fold & Stratified K-Fold](#2-k-fold--stratified-k-fold)
3. [Group K-Fold (Preventing Data Leakage)](#3-group-k-fold-preventing-data-leakage)
4. [Time Series Split (Forward Chaining)](#4-time-series-split-forward-chaining)
5. [Nested Cross-Validation (The Gold Standard)](#5-nested-cross-validation-the-gold-standard)
6. [Data Leakage Avoidance via Pipelines](#6-data-leakage-avoidance-via-pipelines)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Mathematical Justification for CV

When evaluating a model with a single Train/Test split, the estimate of the test error $\hat{E}$ is highly dependent on the random seed used to split the data. The variance of this estimate $Var(\hat{E})$ is inversely proportional to the size of the test set.

**Cross-Validation** averages the error across $K$ folds:
$$CV_{(K)} = \frac{1}{K} \sum_{i=1}^{K} MSE_i$$

Because we are averaging, the variance of the error estimate decreases. Furthermore, because each model is trained on $\frac{K-1}{K}$ of the data, the bias of the error estimate (compared to the true population error) is also minimized. $K=5$ or $K=10$ have been empirically shown to strike the optimal balance in the Bias-Variance tradeoff of error estimation.

---

## 2. K-Fold & Stratified K-Fold

**Standard K-Fold**: Shuffles data and divides it into $K$ equal-sized blocks.
**Stratified K-Fold**: Enforces that the class distribution inside each fold matches the class distribution of the overall dataset. *Mandatory for classification problems.*

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from matplotlib.patches import Patch

# Visualize Stratified K-Fold
np.random.seed(42)
X = np.random.randn(100, 2)
y = np.array([0]*80 + [1]*20) # 80/20 class imbalance

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fig, ax = plt.subplots(figsize=(10, 3))
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    indices = np.zeros(100)
    indices[train_index] = 1 # 1 = Train
    indices[test_index] = 2  # 2 = Test
    ax.scatter(range(100), [i]*100, c=indices, cmap='coolwarm', marker='s', s=50)

ax.set_yticks(range(5))
ax.set_yticklabels([f'Fold {i}' for i in range(5)])
ax.set_xlabel('Sample Index')
ax.set_title('Stratified K-Fold Data Splitting (Red=Train, Blue=Test)', fontweight='bold')
plt.tight_layout()
plt.savefig('stratified_kfold.png', dpi=150)
plt.show()
```

---

## 3. Group K-Fold (Preventing Data Leakage)

Imagine predicting lung cancer from X-rays. You have 1,000 X-rays, but they come from only 200 distinct patients (5 scans per patient).

If you use standard K-Fold, scans from Patient A will end up in both the training fold and the test fold. The model will just memorize Patient A's anatomy, leading to **massive data leakage** and falsely high accuracy.

**Group K-Fold** ensures that all data points from a specific group (e.g., Patient ID) end up entirely in either the training set or the test set, never both.

```python
from sklearn.model_selection import GroupKFold

groups = np.repeat(np.arange(20), 5) # 20 patients, 5 samples each = 100 samples
gkf = GroupKFold(n_splits=4)

print("Group K-Fold Splitting:")
for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups.intersection(test_groups)
    print(f"Fold {i}: Train Groups: {len(train_groups)}, Test Groups: {len(test_groups)}, Overlap: {len(overlap)}")
# Overlap will ALWAYS be 0!
```

---

## 4. Time Series Split (Forward Chaining)

In time series, the future cannot be used to predict the past. K-Fold randomly shuffles data, which breaks causality.

**Time Series Split** uses an expanding window:
- Fold 1: Train on [Day 1], Test on [Day 2]
- Fold 2: Train on [Day 1, Day 2], Test on [Day 3]
- Fold 3: Train on [Day 1, Day 2, Day 3], Test on [Day 4]

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=4)
# Used specifically for temporal/sequential data ordering
```

---

## 5. Nested Cross-Validation (The Gold Standard)

If you use K-Fold CV to tune hyperparameters (e.g., choosing the best tree depth), the resulting "best CV score" is actually an **overly optimistic** estimate of how the model will perform on completely unseen data. By repeatedly picking the best score on the validation folds, you have effectively leaked validation data into the model selection process.

**Nested CV** solves this by adding an "outer" loop:
1. **Outer Loop**: Splits data into $K_{outer}$ folds to estimate generalization error.
2. **Inner Loop**: For each outer training set, splits it into $K_{inner}$ folds to tune hyperparameters.

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC

# Inner Loop (Hyperparameter Tuning)
param_grid = {'C': [1, 10, 100], 'gamma': [0.01, 0.1]}
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
clf = GridSearchCV(estimator=SVC(), param_grid=param_grid, cv=inner_cv)

# Outer Loop (Performance Estimation)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)

print(f"Unbiased Nested CV accuracy: {nested_score.mean():.3f} +/- {nested_score.std():.3f}")
```

---

## 6. Data Leakage Avoidance via Pipelines

The most common mistake in ML:
1. Scale the entire dataset `X_scaled = StandardScaler().fit_transform(X)`
2. Run `cross_val_score(model, X_scaled, y, cv=5)`

**Why this is wrong**: The `StandardScaler` calculated the mean and standard deviation using the *entire* dataset, including the test folds. Information from the test folds has "leaked" into the training process via the scaling parameters.

**The Fix**: Use `sklearn.pipeline.Pipeline`. It ensures that during each fold of cross-validation, the scaler is `fit()` *only* on the training data of that specific fold.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Robust pipeline combining preprocessing and modeling
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

# CV is now perfectly insulated against leakage
scores = cross_val_score(pipeline, X, y, cv=5)
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Proof of Leakage**: Write a script that uses standard K-Fold vs Nested K-Fold on a random noise dataset (where features have zero actual predictive power). Show that standard CV hyperparameter tuning can artificially "find" patterns and report positive accuracy, while Nested CV correctly reports 50% (random guess).

### What's Next
| Next | Why |
|------|-----|
| [Hyperparameter Tuning](./03-Hyperparameter-Tuning.md) | Now that we can evaluate without bias, how do we efficiently search the hyperparameter space? |

---

[← Metrics And Evaluation](./01-Metrics-And-Evaluation.md) | [Back to Index](../README.md) | [Next: Hyperparameter Tuning →](./03-Hyperparameter-Tuning.md)
