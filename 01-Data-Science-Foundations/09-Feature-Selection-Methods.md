# 📉 Feature Selection Methods

> **Prerequisites**: Data Preprocessing | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Filter Methods](#1-filter-methods)
2. [Wrapper Methods](#2-wrapper-methods)
3. [Embedded Methods](#3-embedded-methods)
4. [Summary](#summary)

---

Feature selection is the process of reducing the number of input variables when developing a predictive model. It is desirable to reduce the number of input variables to both reduce the computational cost of modeling and, in some cases, to improve the performance of the model.

In this document, we will explore the mathematics and implementations of the three main categories of feature selection: **Filter Methods**, **Wrapper Methods**, and **Embedded Methods**.

---

## 1. Filter Methods

Filter methods evaluate the relevance of the features outside of the predictive models and subsequently select only the features that meet some criteria. They are typically fast and rely on statistical tests.

### 1.1 Mutual Information

Mutual Information (MI) measures the dependence between two variables. It quantifies the "amount of information" (in units such as shannons or bits) obtained about one random variable through observing the other random variable.

For discrete variables $X$ and $Y$, the mutual information $I(X;Y)$ is defined as:

$$ I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log \left( \frac{p(x,y)}{p(x)p(y)} \right) $$

Where:
*   $p(x,y)$ is the joint probability mass function of $X$ and $Y$.
*   $p(x)$ and $p(y)$ are the marginal probability mass functions of $X$ and $Y$ respectively.

If $X$ and $Y$ are independent, their joint probability is the product of their marginal probabilities ($p(x,y) = p(x)p(y)$), making the log term $\log(1) = 0$, so $I(X;Y) = 0$.

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.datasets import make_classification

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=2, random_state=42)
feature_names = [f"Feature_{i}" for i in range(10)]
df = pd.DataFrame(X, columns=feature_names)

# Calculate Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=42)

# Create a DataFrame for visualization
mi_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values(by='MI_Score', ascending=False)
print("Mutual Information Scores:\n", mi_df.head())

# Select the top 3 features using SelectKBest
selector = SelectKBest(score_func=mutual_info_classif, k=3)
X_selected = selector.fit_transform(X, y)
selected_features = np.array(feature_names)[selector.get_support()]
print("\nSelected Features (Top 3):", selected_features)
```

### 1.2 ANOVA F-Value

Analysis of Variance (ANOVA) is used to analyze the differences among group means in a sample. The F-statistic in ANOVA is the ratio of two variances: the variance between the group means and the variance within the groups.

$$ F = \frac{\text{Between-group variance}}{\text{Within-group variance}} = \frac{\frac{\sum_{i=1}^{k} n_i (\bar{x}_i - \bar{x})^2}{k-1}}{\frac{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (x_{ij} - \bar{x}_i)^2}{N-k}} $$

A larger F-value indicates that the means of the groups are more spread out relative to the variance within the groups, suggesting the feature is highly discriminatory for the target classes.

```python
from sklearn.feature_selection import f_classif

f_values, p_values = f_classif(X, y)
anova_df = pd.DataFrame({'Feature': feature_names, 'F_Value': f_values, 'P_Value': p_values})
anova_df = anova_df.sort_values(by='F_Value', ascending=False)
print("ANOVA F-Values:\n", anova_df.head())
```

---

## 2. Wrapper Methods

Wrapper methods consider the selection of a set of features as a search problem, where different combinations are prepared, evaluated and compared to other combinations. A predictive model is used to evaluate a combination of features and assign a score based on model accuracy.

### 2.1 Recursive Feature Elimination (RFE)

RFE works by searching for a subset of features by starting with all features in the training dataset and successfully removing features until the desired number remains.

This is achieved by fitting the given machine learning algorithm used in the core of the model, ranking features by importance, discarding the least important features, and re-fitting the model.

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Initialize the base estimator
rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42)

# Initialize RFE to select the top 4 features
rfe = RFE(estimator=rf_estimator, n_features_to_select=4, step=1)
rfe.fit(X, y)

# Print the results
print("Feature Ranking:")
for i in range(X.shape[1]):
    print(f"{feature_names[i]}: Rank {rfe.ranking_[i]} {'(Selected)' if rfe.support_[i] else ''}")
```

### 2.2 Boruta Algorithm

Boruta is an all-relevant feature selection method. It works as a wrapper algorithm around Random Forest.

1.  **Shadow Features**: It creates copies of all original features.
2.  **Permutation**: It randomly permutes the values of these shadow features to remove their correlation with the target.
3.  **Importance**: It trains a Random Forest and evaluates the importance (usually Mean Decrease Gini) of all features (original + shadow).
4.  **Z-Score**: It compares the importance of each original feature to the maximum importance of any shadow feature using a statistical test (Z-score). If an original feature is significantly better than the best shadow feature, it is marked as "important".

While Scikit-Learn doesn't have Boruta built-in, you can use the `boruta` package or implement a simplified concept.

---

## 3. Embedded Methods

Embedded methods learn which features best contribute to the accuracy of the model while the model is being created. The most common type of embedded feature selection methods are regularization methods.

### 3.1 L1 Regularization (Lasso)

Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds an $L_1$ penalty to the loss function.

$$ Loss = \sum_{i=1}^{n} (y_i - \sum_{j=1}^{p} x_{ij}\beta_j)^2 + \lambda \sum_{j=1}^{p} |\beta_j| $$

The unique property of the $L_1$ norm is that it produces sparse solutions. Because the absolute value function is not differentiable at zero, the gradient descent optimization will often force some coefficients exactly to zero. Features with a coefficient of zero are effectively eliminated.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the data (essential for regularized linear models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Logistic Regression with L1 penalty (Lasso)
# C is the inverse of regularization strength (C = 1/lambda)
lasso_model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
lasso_model.fit(X_scaled, y)

# Identify selected features (non-zero coefficients)
selected_l1 = np.array(feature_names)[lasso_model.coef_[0] != 0]
print(f"Features selected by L1 Regularization: {selected_l1}")
print(f"Coefficients: {lasso_model.coef_[0]}")
```

### 3.2 Tree-Based Feature Importance

Decision Trees and Ensembles (Random Forest, XGBoost) naturally calculate feature importance during training.

In a Random Forest, the importance of a feature is calculated as the (normalized) total reduction of the criterion (e.g., Gini Impurity) brought by that feature. It is also known as the Gini importance.

$$ \text{Importance}(f) = \frac{\sum_{trees} \sum_{nodes \text{ splitting on } f} \Delta \text{Impurity}}{\text{Number of trees}} $$

```python
# Re-using the random forest from the RFE section
rf_estimator.fit(X, y)

# Get feature importances
importances = rf_estimator.feature_importances_

# Visualize
import matplotlib.pyplot as plt

indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in indices]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), sorted_features, rotation=45)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
# plt.show() # Uncomment to show plot
```

## Summary

*   **Filter Methods** (Mutual Info, ANOVA): Fast, scalable, independent of the model. Good for initial screening.
*   **Wrapper Methods** (RFE, Sequential Feature Selection): Highly accurate as they optimize for the specific model, but computationally expensive and prone to overfitting on small datasets.
*   **Embedded Methods** (Lasso, Tree Importance): Good balance of accuracy and computational cost, as selection is built into the model training.

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Apply Mutual Information to a dataset and plot a bar chart of the top 5 features.
- 🟡 **Intermediate**: Compare the selected features and resulting model accuracy of RFE vs Lasso on a high-dimensional dataset.

### What's Next
| Next | Why |
|------|-----|
| [Imbalanced Data](./10-Imbalanced-Data.md) | Learn how to handle datasets where one class dominates. |

---

[← Data Visualization Mastery](./08-Data-Visualization-Mastery.md) | [Back to Index](../README.md) | [Next: Imbalanced Data →](./10-Imbalanced-Data.md)
