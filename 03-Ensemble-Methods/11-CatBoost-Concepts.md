# 🐈 CatBoost (Categorical Boosting)

> **Difficulty**: ⭐⭐⭐⭐⭐ Expert | **Prerequisites**: LightGBM, Gradient Boosting | **Estimated Reading Time**: 25 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Core Mathematics](#3-core-mathematics)
4. [Visual Explanation](#4-visual-explanation)
5. [Algorithm Workflow](#5-algorithm-workflow)
6. [From Scratch Implementation](#6-from-scratch-implementation)
7. [NumPy Implementation](#7-numpy-implementation)
8. [Scikit-Learn Implementation](#8-scikit-learn-implementation)
9. [Hyperparameter Deep Dive](#9-hyperparameter-deep-dive)
10. [Visualization Lab](#10-visualization-lab)
11. [Failure Cases](#11-failure-cases)
12. [Industry Applications](#12-industry-applications)
13. [Interview Preparation](#13-interview-preparation)
14. [Hands-On Exercises](#14-hands-on-exercises)
15. [Further Reading](#15-further-reading)

---

## 1. What Problem Does This Solve?

Both XGBoost and LightGBM are brilliant, but they share two weaknesses:
1. **Target Leakage (Prediction Shift)**: When computing the gradients (residuals) for a data point, standard boosting uses the model trained *on that same data point*, which introduces subtle overfitting.
2. **Categorical Features**: They require categorical variables (like City, Color, UserID) to be heavily pre-processed (One-Hot Encoded, which explodes dimensionality, or Target Encoded, which causes massive data leakage).

**CatBoost**, created by Yandex (the "Google of Russia"), solves both problems natively via:
1. **Ordered Boosting**: A mathematical trick to ensure that the model predicting a data point's residual was *never trained on that data point*.
2. **Ordered Target Encoding**: A highly sophisticated, built-in way to process categorical features without causing target leakage.

**Use Cases:**
- Datasets dominating by high-cardinality categorical features (e.g., zip codes, user IDs).
- When you want top-tier performance out-of-the-box with zero hyperparameter tuning.

---

## 2. Intuition

### 🟢 Beginner
Imagine trying to guess the price of a car based on its `Brand` (Categorical). If you calculate the average price of all Fords to help you guess the price of *this specific* Ford, you are "cheating" because you used *this* Ford's price to calculate the overall average.
CatBoost artificially pretends the data arrived in a sequence over time. To calculate the average price of Fords for car #5, it only looks at Fords #1, #2, #3, and #4. This stops the cheating (target leakage).

### 🟡 Intermediate
Standard Target Encoding replaces a category like "New York" with the mean target value of all "New York" rows. This leaks information.
CatBoost randomly permutes (shuffles) the dataset. For row $k$, it replaces the category with the average target of only the rows *before* $k$ that share the same category. 

### 🔴 Advanced
**Prediction Shift**: In standard Gradient Boosting, the distribution of the estimated gradients is shifted from the true distribution of gradients on unseen data, because the model $F_{m-1}$ was trained on the exact same $X$ that we are now evaluating to find the residuals.
**Ordered Boosting**: CatBoost trains $N$ separate models (where $N$ is the dataset size). Model $k$ is trained only on the first $k$ instances. When evaluating the residual for instance $k+1$, it uses Model $k$. This completely eliminates Prediction Shift.

---

## 3. Core Mathematics

### 3.1 Ordered Target Encoding
Let $x_{i,k}$ be the categorical feature value for row $i$. The encoded value $\hat{x}_{i,k}$ is:
$$ \hat{x}_{i,k} = \frac{\sum_{j=1}^{i-1} [x_{j,k} == x_{i,k}] \cdot y_j + a \cdot P}{\sum_{j=1}^{i-1} [x_{j,k} == x_{i,k}] + a} $$
Where:
- $y_j$ is the target.
- $P$ is the prior (usually the mean target of the whole dataset).
- $a$ is a smoothing parameter (weight of the prior).
*Notice the sum only goes up to $i-1$. This prevents target leakage.*

### 3.2 Oblivious Trees (Symmetric Trees)
CatBoost exclusively builds **Oblivious Trees**. This means that at a given depth, the exact same splitting criterion (e.g., `Age > 30`) is applied across *all* nodes in that level.
This makes the trees perfectly balanced, highly resistant to overfitting, and extremely fast to execute during inference (it just becomes a simple bitwise evaluation).

---

## 4. Visual Explanation

```mermaid
flowchart TD
    subgraph Standard Tree
        R1[Root: Age > 30] --> L1[City == NY]
        R1 --> R2[Income > 50k]
    end

    subgraph Oblivious Tree (CatBoost)
        R3[Root: Age > 30] --> L3[City == NY]
        R3 --> R4[City == NY]
        L3 --> L5[Income > 50k]
        L3 --> L6[Income > 50k]
        R4 --> R5[Income > 50k]
        R4 --> R6[Income > 50k]
    end
```
*Notice how in the Oblivious Tree, the same rule is applied horizontally across the entire level.*

---

## 5. Algorithm Workflow

1. Perform random permutations of the dataset (usually multiple different shuffles).
2. For categorical features, apply Ordered Target Encoding based on the current permutation.
3. For Ordered Boosting, conceptually maintain a separate model for every prefix of the permutation (implemented via a clever tree structure to save memory).
4. Grow Oblivious Trees: Select the single best split feature/threshold and apply it to all leaves at the current depth.
5. Update the ensemble.

---

## 6. From Scratch Implementation

*Like LightGBM, CatBoost's magic relies on complex C++ data structures for permutation hashing and bitwise tree evaluation.*

---

## 7. NumPy Implementation

*(See section 6).*

---

## 8. CatBoost Implementation

*Note: You must `pip install catboost`*

```python
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Pass the indices or names of the categorical columns
cat_features_indices = [0, 2, 5] 

# CatBoost requires almost ZERO hyperparameter tuning. The defaults are famously excellent.
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='Accuracy',
    random_seed=42,
    verbose=100
)

model.fit(
    X_train, y_train,
    cat_features=cat_features_indices,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)

preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
```

---

## 9. Hyperparameter Deep Dive

One of CatBoost's biggest selling points is that it requires almost no tuning. However, if you must:
- **`iterations`**: Number of trees. Set high and use early stopping.
- **`depth`**: Depth of the oblivious tree. (Default 6 is usually perfect. Don't go above 10).
- **`learning_rate`**: Automatically set by CatBoost based on dataset size! But you can override it.
- **`l2_leaf_reg`**: L2 regularization. Increase to reduce overfitting.

---

## 10. Visualization Lab

*CatBoost has incredible built-in plotting.*

```python
# Assuming 'model' is trained as above, in a Jupyter Notebook:
model.plot_tree(tree_idx=0)

# To view feature importances:
importances = model.get_feature_importance()
print(importances)
```

---

## 11. Failure Cases

**Sparse Data (Text TF-IDF)**
CatBoost is designed for dense, tabular data with many categorical columns. It performs poorly (and very slowly) on massive, sparse numerical matrices (like word counts). Use LightGBM for that.

**Training Time on purely numerical data**
If your dataset has zero categorical features, CatBoost's overhead from Ordered Boosting makes it slower to train than LightGBM, often without a significant accuracy boost.

---

## 12. Industry Applications

- **Weather Forecasting**: Yandex heavily uses CatBoost for its weather prediction services, which involve massive amounts of categorical sensor and location data.
- **Ride-Sharing ETA**: Predicting travel times using categorical features like Day of Week, Weather Status, and Road ID.

---

## 13. Interview Preparation

### Beginner
**Q: Why use CatBoost over XGBoost?**
> A: If the dataset has many categorical features, CatBoost handles them natively and beautifully without requiring you to manually One-Hot Encode or Target Encode them. It also yields better out-of-the-box performance without hyperparameter tuning.

### Intermediate
**Q: What is an Oblivious Tree?**
> A: A perfectly balanced decision tree where the exact same splitting rule is applied to all nodes at a specific depth level. 

### Advanced
**Q: Explain how CatBoost prevents Target Leakage.**
> A: It uses "Ordered Boosting" and "Ordered Target Encoding". By randomly shuffling the data and ensuring that row $i$ only ever uses information from rows $1$ through $i-1$, it prevents the model from "seeing into the future" or utilizing its own label during training.

---

## 14. Hands-On Exercises

**Easy**: Find a dataset with categorical features. Train a RandomForest using One-Hot Encoding, and a CatBoost using native `cat_features`. Compare the lines of code required and the final accuracy.
**Medium**: Test CatBoost's default performance. Run CatBoost with default parameters vs XGBoost with default parameters on 3 different tabular datasets. Record the results.
**Hard**: Implement a rudimentary "Ordered Target Encoder" in pure Pandas for a single categorical column. Shuffle the dataframe, iterate through it, and calculate the mean target of the category using *only* the rows that appeared before the current row.

---

## 15. Further Reading

- [CatBoost: unbiased boosting with categorical features (Prokhorenkova et al., 2018)](https://arxiv.org/abs/1706.09516)
- [Official CatBoost Documentation](https://catboost.ai/en/docs/)

---

[← LightGBM Concepts](10-LightGBM-Concepts.md) | [Return to Ensemble Index](../README.md)
