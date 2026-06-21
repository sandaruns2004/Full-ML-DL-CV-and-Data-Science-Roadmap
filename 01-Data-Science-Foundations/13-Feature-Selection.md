# ✂️ Feature Selection Methods

> **Prerequisites**: [Feature Engineering](./12-Feature-Engineering.md) | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents

1. [The Curse of Dimensionality](#1-the-curse-of-dimensionality)
2. [Filter Methods (Statistical)](#2-filter-methods-statistical)
3. [Wrapper Methods (Iterative)](#3-wrapper-methods-iterative)
4. [Embedded Methods (Algorithmic)](#4-embedded-methods-algorithmic)
5. [Summary of When to Use Which](#5-summary-of-when-to-use-which)
6. [What's Next](#6-whats-next)

---

## 1. The Curse of Dimensionality

### 🟢 Beginner

When you learn Data Science, you often think: *"More data is always better. Let's feed the model 1,000 features!"*

This is dangerous due to the **Curse of Dimensionality**:
1. **Overfitting**: With too many features, the model memorizes the noise instead of learning the pattern. It performs perfectly on training data but fails miserably in real life.
2. **Computational Cost**: Training a model on 1,000 features takes exponentially longer than 10 features.
3. **Interpretability**: A model with 5 features can be explained to the CEO. A model with 500 features is a "black box."

**Feature Selection** is the process of scientifically dropping the useless/redundant columns and keeping only the ones that matter.

There are 3 main approaches: **Filter**, **Wrapper**, and **Embedded**.

---

## 2. Filter Methods (Statistical)

### 🟡 Intermediate

Filter methods look at the mathematical properties of the data *before* any ML model is trained. They are extremely fast.

**1. Variance Threshold**
If a feature has the exact same value for 99% of the rows, it is useless for predicting anything. We drop low-variance features.

```python
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

df = pd.DataFrame({
    'Good_Feature': [10, 20, 30, 40, 50],
    'Useless_Feature': [1, 1, 1, 1, 1.1] # Almost zero variance!
})

# Drop features where 80% or more of the values are the same
selector = VarianceThreshold(threshold=0.2)
X_clean = selector.fit_transform(df)
```

**2. Correlation / Collinearity**
If `Feature_A` and `Feature_B` are 95% correlated (e.g., "Year Built" and "Age of House"), you don't need both. They provide the exact same information. Keep one, drop the other.

```python
import numpy as np

# Calculate correlation matrix
corr_matrix = df.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.90
to_drop = [column for column in upper.columns if any(upper[column] > 0.90)]

# Drop them
df_clean = df.drop(df[to_drop], axis=1)
```

---

## 3. Wrapper Methods (Iterative)

### 🔴 Advanced

Wrapper methods actually train a mini Machine Learning model over and over again with different combinations of features to see which combination gets the highest accuracy.

Because they train the model hundreds of times, they are **computationally expensive** but yield the best results.

**1. Recursive Feature Elimination (RFE)**
1. Train the model on all 100 features.
2. Look at which feature was the least important. Drop it.
3. Train the model on the remaining 99 features.
4. Repeat until you are left with the desired number of features.

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# We want to narrow our dataset down to the top 5 most important features
model = RandomForestClassifier()
rfe = RFE(estimator=model, n_features_to_select=5)

# Fit RFE
rfe.fit(X_train, y_train)

# View which features were kept
for i in range(X_train.shape[1]):
    print(f"Column: {i}, Selected: {rfe.support_[i]}, Rank: {rfe.ranking_[i]}")
    
# Transform the dataset
X_train_selected = rfe.transform(X_train)
```

**2. Forward Selection**
Start with 0 features. Try adding each feature one by one and keep the one that improves accuracy the most. Repeat.

---

## 4. Embedded Methods (Algorithmic)

### 🔴 Advanced

Embedded methods are a hybrid. The feature selection is built directly into the ML algorithm itself. As the model trains, it automatically figures out which features to ignore.

**1. L1 Regularization (Lasso Regression)**
Lasso Regression adds a mathematical penalty for having too many features. As it trains, it forces the coefficients (weights) of useless features to become exactly `0.0`, effectively deleting them from the equation.

```python
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# Alpha controls the strictness. Higher alpha = more features dropped.
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# Select features where the Lasso coefficient is NOT zero
selector = SelectFromModel(lasso, prefit=True)
X_train_selected = selector.transform(X_train)
```

**2. Tree-Based Feature Importance**
Algorithms like Random Forest and XGBoost calculate "Feature Importance" out-of-the-box. They track how often a feature was used to split a decision tree.

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Plot the built-in feature importances
importances = model.feature_importances_

import matplotlib.pyplot as plt
plt.bar(range(X_train.shape[1]), importances)
plt.title("Random Forest Feature Importances")
plt.show()
```

---

## 5. Summary of When to Use Which

| Method | Speed | Accuracy | Best For | Example |
|--------|-------|----------|----------|---------|
| **Filter** | ⚡ Very Fast | ⭐ Low | Initial cleanup of massive datasets (1,000+ columns) | Removing 99% correlated features |
| **Wrapper** | 🐢 Very Slow | ⭐⭐⭐ High | Small/Medium datasets where accuracy is critical | RFE (Recursive Feature Elimination) |
| **Embedded** | 🏃 Medium | ⭐⭐ Medium | Large datasets, done simultaneously with training | Lasso Regression, XGBoost Importance |

**Pro Tip Pipeline**: 
In reality, you use them together. Use a *Filter* method to drop the obvious garbage and reduce 1,000 columns to 100. Then use a *Wrapper/Embedded* method to reduce the 100 down to the final 20.

---

## 6. What's Next

Feature Selection handles datasets with too many columns. But what do you do when your dataset has a wildly uneven number of rows per category? (e.g., 99% of transactions are legitimate, 1% are fraud).

| Next Topic | Why |
|------------|-----|
| [Imbalanced Data](./14-Imbalanced-Data.md) | Learn how to fix datasets where one class severely outnumbers the other, preventing your model from just guessing the majority class 100% of the time. |

---

[← Previous: Feature Engineering](./12-Feature-Engineering.md) | [Back to Main Index](../README.md) | [Next: Imbalanced Data →](./14-Imbalanced-Data.md)
