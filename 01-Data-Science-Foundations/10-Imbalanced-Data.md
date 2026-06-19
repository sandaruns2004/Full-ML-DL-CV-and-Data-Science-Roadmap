# ⚖️ Handling Imbalanced Data

> **Prerequisites**: Statistical Inference | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Metrics for Imbalanced Data](#1-metrics-for-imbalanced-data)
2. [Resampling Techniques](#2-resampling-techniques)
3. [Synthetic Data Generation](#3-synthetic-data-generation)
4. [Algorithmic Approaches (Cost-Sensitive Learning)](#4-algorithmic-approaches-cost-sensitive-learning)
5. [Summary of Best Practices](#summary-of-best-practices)

---

In many real-world classification problems, the distribution of classes is not uniform. When one class (the **minority class**) is significantly smaller than the other (the **majority class**), the dataset is considered imbalanced. 

Examples include fraud detection (99.9% legitimate, 0.1% fraud), rare disease diagnosis, or manufacturing defect detection.

Standard machine learning algorithms are designed to maximize overall accuracy. Given a 99:1 imbalance, a model that simply predicts the majority class every single time will achieve 99% accuracy, while completely failing at its actual objective (identifying the rare event).

---

## 1. Metrics for Imbalanced Data

When dealing with imbalanced datasets, **Accuracy is a misleading metric**. Instead, rely on:

*   **Precision**: Out of all positive predictions, how many were actually positive? $TP / (TP + FP)$
*   **Recall (Sensitivity)**: Out of all actual positives, how many did we find? $TP / (TP + FN)$
*   **F1-Score**: The harmonic mean of Precision and Recall. $2 \times \frac{Precision \times Recall}{Precision + Recall}$
*   **ROC-AUC**: The Area Under the Receiver Operating Characteristic curve. Evaluates how well the model separates the classes at various thresholds.
*   **PR-AUC**: Area Under the Precision-Recall curve. Often preferred over ROC-AUC for highly imbalanced data.

---

## 2. Resampling Techniques

Resampling involves altering the dataset to achieve a more balanced distribution. This happens *before* model training.

### 2.1 Random Undersampling

Undersampling randomly removes examples from the majority class until it matches the size of the minority class.
*   **Pros**: Reduces training time, prevents the model from ignoring the minority class.
*   **Cons**: Throws away potentially useful information.

```python
import pandas as pd
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler

# Generate a highly imbalanced dataset (95% class 0, 5% class 1)
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0, 
                           n_clusters_per_class=1, weights=[0.95], flip_y=0, random_state=42)

print(f"Original class distribution: 0: {sum(y==0)}, 1: {sum(y==1)}")

# Apply Random Undersampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

print(f"Resampled class distribution: 0: {sum(y_resampled==0)}, 1: {sum(y_resampled==1)}")
```

### 2.2 Random Oversampling

Oversampling randomly duplicates examples from the minority class until it matches the majority class.
*   **Pros**: No information is lost.
*   **Cons**: Increases training time, highly prone to **overfitting** (the model just memorizes the duplicated minority examples).

---

## 3. Synthetic Data Generation

Instead of simply duplicating data, we can synthesize new, artificial data points based on the underlying distribution of the minority class.

### 3.1 SMOTE (Synthetic Minority Over-sampling Technique)

SMOTE creates synthetic samples by interpolating between existing minority instances.

**The SMOTE Algorithm:**
1.  Choose a minority class sample $x_i$.
2.  Find its $k$ nearest neighbors belonging to the same minority class.
3.  Randomly select one of these neighbors, say $\hat{x}_i$.
4.  Create a new synthetic sample $x_{new}$ at a random point along the line segment connecting $x_i$ and $\hat{x}_i$:
    $$ x_{new} = x_i + \lambda (\hat{x}_i - x_i) $$
    where $\lambda$ is a random number between 0 and 1.

```python
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Apply SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print(f"SMOTE class distribution: 0: {sum(y_smote==0)}, 1: {sum(y_smote==1)}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[y==0][:, 0], X[y==0][:, 1], label='Class 0', alpha=0.5, s=10)
ax1.scatter(X[y==1][:, 0], X[y==1][:, 1], label='Class 1', alpha=0.8, s=10)
ax1.set_title("Original Imbalanced Data")

ax2.scatter(X_smote[y_smote==0][:, 0], X_smote[y_smote==0][:, 1], label='Class 0', alpha=0.5, s=10)
ax2.scatter(X_smote[y_smote==1][:, 0], X_smote[y_smote==1][:, 1], label='Class 1', alpha=0.8, s=10)
ax2.set_title("Data after SMOTE")
# plt.show() # Uncomment to view
```

### 3.2 ADASYN (Adaptive Synthetic Sampling)

ADASYN is an improvement over SMOTE. While SMOTE generates an equal number of synthetic samples for each original minority sample, ADASYN calculates a weight for each minority sample based on the level of difficulty in learning it.

Difficulty is determined by the ratio of majority-class neighbors in the $k$-nearest neighbors of the minority point. ADASYN generates more synthetic data in regions of the feature space where the density of minority examples is low and majority examples is high (the decision boundary).

---

## 4. Algorithmic Approaches (Cost-Sensitive Learning)

Instead of changing the dataset, we can change the learning algorithm to penalize errors on the minority class more heavily than errors on the majority class.

### 4.1 Class Weights

Most algorithms in Scikit-Learn support a `class_weight` parameter. 

If we define the loss function as $L(y, \hat{y})$, cost-sensitive learning modifies it to:
$$ Loss = \sum_{i=1}^{n} w_{y_i} L(y_i, \hat{y}_i) $$
where $w_{y_i}$ is the weight assigned to the class of example $i$.

Typically, weights are inversely proportional to class frequencies:
$$ w_j = \frac{n_{samples}}{n_{classes} \times n_{samples\_in\_class\_j}} $$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Train standard model
model_standard = LogisticRegression()
model_standard.fit(X_train, y_train)
y_pred_std = model_standard.predict(X_test)
print("Standard Logistic Regression:")
print(classification_report(y_test, y_pred_std))

# 2. Train cost-sensitive model
# 'balanced' automatically computes weights based on inverse frequency
model_weighted = LogisticRegression(class_weight='balanced')
model_weighted.fit(X_train, y_train)
y_pred_weighted = model_weighted.predict(X_test)
print("\nCost-Sensitive (Balanced) Logistic Regression:")
print(classification_report(y_test, y_pred_weighted))
```

*Notice in the output that the `recall` for class 1 increases dramatically in the weighted model, even if it costs some precision.*

### 4.2 Focal Loss

Focal Loss is widely used in deep learning for highly imbalanced datasets (e.g., Object Detection). It modifies the standard cross-entropy loss to dynamically scale based on prediction confidence. 

Standard Cross-Entropy for binary classification:
$$ CE(p_t) = -\log(p_t) $$
Where $p_t$ is the model's estimated probability for the true class.

Focal Loss adds a modulating factor $(1 - p_t)^\gamma$:
$$ FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) $$

*   $\alpha_t$ balances the importance of positive/negative examples (similar to class weights).
*   $\gamma$ (focusing parameter) reduces the loss contribution from easy examples ($p_t \approx 1$). If an example is easy to classify, $(1 - p_t)^\gamma \approx 0$, so its loss approaches 0, forcing the model to focus on the hard, misclassified examples (which are usually the minority class).

---

## Summary of Best Practices

1.  **Never evaluate on synthetic data**: Only apply SMOTE/Undersampling to the **training set**. The validation and test sets must maintain the original, natural imbalance to provide an accurate evaluation.
2.  **Start with Class Weights**: Cost-sensitive learning is usually the fastest and easiest method to implement.
3.  **Try SMOTE + Undersampling**: A common pipeline is to use SMOTE to oversample the minority class slightly, and then undersample the majority class, achieving a balanced dataset without creating too many artificial points.
4.  **Use PR-AUC**: Rely on Precision-Recall curves rather than ROC curves for model selection.

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Apply class weights to a Logistic Regression model on a skewed dataset and observe the precision/recall changes.
- 🟡 **Intermediate**: Build a pipeline that applies SMOTE to the training set only, trains an XGBoost model, and evaluates on the untouched test set using PR-AUC.

### What's Next
| Next | Why |
|------|-----|
| [Linear Regression](../02-Supervised-Learning/01-Linear-Regression.md) | Dive into Supervised Learning algorithms. |

---

[← Feature Selection Methods](./09-Feature-Selection-Methods.md) | [Back to Index](../README.md) | [Next: Linear Regression →](../02-Supervised-Learning/01-Linear-Regression.md)
