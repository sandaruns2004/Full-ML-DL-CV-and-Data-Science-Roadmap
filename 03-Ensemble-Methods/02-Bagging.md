# 🎒 Bagging (Bootstrap Aggregating)

> **Difficulty:** ⭐⭐☆☆☆ Beginner | **Prerequisites:** Decision Trees, Bias-Variance Tradeoff | **Estimated Reading Time:** 25 minutes

---

## 📋 Table of Contents

1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Core Mathematics](#3-core-mathematics)
4. [Visual Explanation](#4-visual-explanation)
5. [Algorithm Workflow](#5-algorithm-workflow)
6. [From Scratch Implementation](#6-from-scratch-implementation)
7. [Scikit-Learn Implementation](#7-scikit-learn-implementation)
8. [Hyperparameter Deep Dive](#8-hyperparameter-deep-dive)
9. [Failure Cases](#9-failure-cases)
10. [Industry Applications](#10-industry-applications)

---

## 1. What Problem Does This Solve?

### 🟢 Beginner
Single machine learning models, especially deep decision trees, tend to memorize the training data. This means they perform perfectly on the data they were trained on, but terribly on new, unseen data. This is called **overfitting**. Bagging solves this by training many different models and averaging their predictions, which "smooths out" the overconfident mistakes of any single model.

### 🟡 Intermediate
In the context of the Bias-Variance tradeoff, unpruned decision trees have low bias but **high variance**. Small changes in the training data can lead to drastically different tree structures and predictions. Bagging reduces this variance without increasing the bias. It achieves this by introducing randomness into the training process through dataset sampling.

### 🔴 Advanced
Bagging exploits the statistical property that averaging independent models reduces their variance by a factor of $N$ (the number of models). While models trained on the same data are not perfectly independent, Bagging uses bootstrap sampling (sampling with replacement) to create slightly different datasets. This decorrelates the base learners just enough so that their aggregated prediction is significantly more robust and generalized than any individual estimator.

---

## 2. Intuition

Imagine you have a jar of jelly beans and you need to guess the exact number.

If you ask one person, their guess might be completely wrong. They might overestimate or underestimate heavily based on their limited perspective.

However, if you ask 1,000 independent people to guess, and you take the average of all their guesses, the final aggregated number will be shockingly close to the true amount. This is the **"Wisdom of the Crowds."**

Bagging applies this to machine learning:
1. **The Crowd:** A large number of models (typically Decision Trees).
2. **Different Perspectives:** We give each model a slightly different slice of the dataset (Bootstrap Sampling).
3. **The Final Answer:** We let all models vote on the final prediction (Aggregation).

---

## 3. Core Mathematics

### Variance Reduction
If we have $B$ perfectly independent models, each with variance $\sigma^2$, the variance of their average is:

$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^{B} f_b(x)\right) = \frac{\sigma^2}{B}$$

However, in Bagging, the models are trained on variations of the same dataset, so they are somewhat correlated. If the correlation between any two models is $\rho$, the variance of the ensemble is:

$$\text{Var}_{bagging} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As the number of models $B \to \infty$, the second term approaches 0, leaving the variance bounded by $\rho\sigma^2$. This shows why minimizing correlation between models is critical.

### Out-of-Bag (OOB) Estimation
Because we sample *with replacement*, some rows of data are picked multiple times, and some are never picked. 

For a dataset of size $N$, the probability of a specific point *not* being chosen in a single draw is $1 - \frac{1}{N}$. For $N$ draws, the probability that it is never chosen is:

$$\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = \frac{1}{e} \approx 0.368$$

This means ~36.8% of the data is left out of the training set for any given tree. These "Out-of-Bag" samples act as a built-in validation set, allowing us to evaluate the ensemble's performance without needing a separate hold-out test set!

---

## 4. Visual Explanation

```mermaid
flowchart TD
    Data[(Original Dataset \n N samples)] --> S1[Bootstrap Sample 1 \n N samples with replacement]
    Data --> S2[Bootstrap Sample 2 \n N samples with replacement]
    Data --> S3[Bootstrap Sample B \n N samples with replacement]

    S1 --> M1(Base Model 1 \n e.g., Decision Tree)
    S2 --> M2(Base Model 2 \n e.g., Decision Tree)
    S3 --> M3(Base Model B \n e.g., Decision Tree)

    M1 --> V{Aggregation}
    M2 --> V
    M3 --> V

    V -->|Classification| F1[Majority Vote]
    V -->|Regression| F2[Average]
```

---

## 5. Algorithm Workflow

1. **Bootstrap Sampling:** Given a dataset $D$ of size $N$, create $B$ bootstrap samples by randomly drawing $N$ elements from $D$ *with replacement*.
2. **Model Training:** Train a separate base model (usually an unpruned decision tree) on each of the $B$ bootstrap samples. These models are trained completely independently and in parallel.
3. **Prediction (Classification):** For a new instance, pass it through all $B$ models. Each model casts a "vote" for a class. The ensemble returns the class with the most votes (Majority Voting).
4. **Prediction (Regression):** For a new instance, pass it through all $B$ models. The ensemble returns the mathematical average of all $B$ predictions.

---

## 6. From Scratch Implementation

Here is a simple, educational implementation of a Bagging classifier using Python.

```python
import numpy as np
from collections import Counter

class DummyTree:
    """A simple dummy model to act as our base learner."""
    def fit(self, X, y):
        # In reality, this would build a tree.
        # Here we just memorize the most common class.
        self.prediction = Counter(y).most_common(1)[0][0]
        
    def predict(self, X):
        return [self.prediction] * len(X)

class BaggingClassifierFromScratch:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.models = []
        
    def fit(self, X, y):
        n_samples = len(X)
        self.models = []
        
        for _ in range(self.n_estimators):
            # Bootstrap sample: random indices with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # Train and store base model
            model = DummyTree() # Replace with real DecisionTree
            model.fit(X_sample, y_sample)
            self.models.append(model)
            
    def predict(self, X):
        # Get predictions from all models
        # Shape: (n_estimators, n_samples)
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Aggregate via majority vote
        final_preds = []
        for i in range(len(X)):
            votes = predictions[:, i]
            majority = Counter(votes).most_common(1)[0][0]
            final_preds.append(majority)
            
        return np.array(final_preds)
```

---

## 7. Scikit-Learn Implementation

In production, you should rely on `sklearn.ensemble.BaggingClassifier` or `BaggingRegressor`.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Create dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Initialize Bagging Classifier
# oob_score=True allows us to use Out-of-Bag samples for validation
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(), # Base estimator
    n_estimators=100,                   # Number of trees
    max_samples=1.0,                    # Size of bootstrap sample (1.0 = N)
    bootstrap=True,                     # Use replacement
    n_jobs=-1,                          # Use all CPU cores (parallel training)
    oob_score=True,                     # Compute OOB error
    random_state=42
)

# 3. Train
bagging_clf.fit(X_train, y_train)

# 4. Evaluate
print(f"OOB Score: {bagging_clf.oob_score_:.4f}")

y_pred = bagging_clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

---

## 8. Hyperparameter Deep Dive

| Parameter | What it does | Default | Best Practices |
|-----------|--------------|---------|----------------|
| `n_estimators` | Number of base models in the ensemble. | 10 | Higher is better for variance reduction, but diminishing returns occur after ~100. It increases computational cost. |
| `max_samples` | Proportion of data to draw for each bootstrap sample. | 1.0 (100%) | Keep at 1.0. Lowering it reduces correlation between trees but sacrifices individual tree strength. |
| `max_features`| Proportion of features to use per model (Feature Bagging). | 1.0 (100%) | Useful if you have many features. This effectively moves the algorithm closer to Random Forests. |
| `bootstrap` | Whether to sample with replacement. | `True` | Keep `True` for standard Bagging. Setting to `False` creates "Pasting" (training on sub-samples without replacement). |
| `oob_score` | Calculate validation score on Out-of-Bag samples. | `False` | Set to `True` to get a free validation metric without sacrificing training data. |
| `n_jobs` | Number of CPU cores to use. | `None` (1) | Set to `-1` to use all cores. Bagging is embarrassingly parallel! |

---

## 9. Failure Cases

Bagging is powerful, but it has specific scenarios where it fails:

1. **High Bias Base Models:** Bagging reduces variance, NOT bias. If your base model is too simple (like a Linear Regression or a Tree with `max_depth=1`), Bagging will not improve the performance significantly. It requires strong, complex base learners (like deep trees) to work.
2. **Highly Correlated Features:** If a dataset has one massively dominant feature, every single bootstrap tree will split on that feature first. As a result, all the trees will look very similar (high correlation $\rho$). When trees are highly correlated, Bagging's variance reduction fails. (This is exactly the problem Random Forests were invented to solve).
3. **Loss of Interpretability:** A single Decision Tree is easily interpreted by plotting its rules. An ensemble of 500 bagged trees is effectively a black box.

---

## 10. Industry Applications

While standard Bagging is often superseded by Random Forests or Boosting in modern applications, its principles are still widely used:

* **Customer Retention:** Aggregating predictions from different subsets of customer history to predict churn likelihood with high stability.
* **Credit Scoring:** Using Bagging to ensure that small fluctuations in a user's financial data don't lead to wild swings in their credit approval status.
* **Medical Diagnosis:** Reducing the variance of diagnostic models on small clinical trial datasets where standard hold-out testing is impossible (using OOB evaluations instead).

---

[← Introduction To Ensemble Learning](01-Introduction-To-Ensemble-Learning.md) | [Back to Index](../README.md) | [Next: Random Forest →](03-Random-Forest.md)
