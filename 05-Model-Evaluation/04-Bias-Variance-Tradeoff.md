# ⚖️ The Bias-Variance Tradeoff

> **Prerequisites**: Supervised Learning | **Difficulty**: ⭐⭐☆☆☆ Elementary

---

## 📋 Table of Contents
1. [Mathematical Decomposition of Expected Error](#1-mathematical-decomposition-of-expected-error)
2. [The Tradeoff](#2-the-tradeoff)
3. [Learning Curves](#3-learning-curves)

---

The Bias-Variance Tradeoff is the central problem in supervised machine learning. It is the tension between a model's ability to capture the true underlying patterns in the training data (low bias) and its ability to generalize to unseen data without capturing noise (low variance).

Ideally, we want a model that accurately captures the regularities in its training data, but also generalizes well to unseen data. Unfortunately, it is typically impossible to do both simultaneously.

---

## 1. Mathematical Decomposition of Expected Error

Assume we have a target variable $y$ that is generated from an underlying function $f(x)$ plus some random noise $\epsilon$:
$$ y = f(x) + \epsilon $$

Where the noise has a mean of zero and a variance of $\sigma^2$:
$$ E[\epsilon] = 0, \quad Var(\epsilon) = \sigma^2 $$

We want to find a model $\hat{f}(x)$ that approximates the true function $f(x)$ as closely as possible. 

If we evaluate our model on an unseen data point $(x, y)$, we want to measure the Expected Prediction Error (EPE), which is the expected value of the squared difference between the prediction $\hat{f}(x)$ and the true value $y$:
$$ E[(y - \hat{f}(x))^2] $$

Through algebraic expansion, this Expected Error can be mathematically decomposed into three distinct components:

$$ E[(y - \hat{f}(x))^2] = \text{Bias}[\hat{f}(x)]^2 + \text{Var}[\hat{f}(x)] + \sigma^2 $$

Let's define each component:

### 1.1 Bias Squared: $\text{Bias}[\hat{f}(x)]^2$
$$ \text{Bias}[\hat{f}(x)] = E[\hat{f}(x)] - f(x) $$
*   **Definition**: The difference between the average prediction of our model and the correct value. 
*   **Intuition**: High bias implies the model is too simple to capture the underlying relationships. It makes strong, incorrect assumptions.
*   **Symptom**: **Underfitting**. The model performs poorly on *both* the training set and the test set. (e.g., using a linear line to fit a sine wave).

### 1.2 Variance: $\text{Var}[\hat{f}(x)]$
$$ \text{Var}[\hat{f}(x)] = E[\hat{f}(x)^2] - E[\hat{f}(x)]^2 $$
*   **Definition**: The variability of the model prediction for a given data point over different training sets.
*   **Intuition**: High variance implies the model is highly sensitive to the specific noise in the training data. It memorizes the training data.
*   **Symptom**: **Overfitting**. The model performs exceptionally well on the training set but poorly on the test set. (e.g., a highly complex polynomial that passes through every single training point).

### 1.3 Irreducible Error: $\sigma^2$
*   **Definition**: The variance of the noise term $\epsilon$.
*   **Intuition**: This is the fundamental, inherent noise in the problem itself (e.g., measurement error, unobserved variables).
*   **Fact**: No matter how good the model is, the Expected Error can never fall below $\sigma^2$.

---

## 2. The Tradeoff

*   **Simple Models** (e.g., Linear Regression, Shallow Decision Trees): Tend to have **High Bias** and **Low Variance**. They don't change much if you swap out the training data, but they miss complex patterns.
*   **Complex Models** (e.g., Deep Neural Networks, Deep Random Forests, High-degree Polynomials): Tend to have **Low Bias** and **High Variance**. They can fit the training data perfectly, but wildly change their predictions if the training data changes slightly.

As model complexity increases, Bias decreases and Variance increases. The optimal model complexity is the point that minimizes the total Expected Error.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate underlying true function + noise
np.random.seed(0)
def true_fun(X):
    return np.cos(1.5 * np.pi * X)

n_samples = 30
X = np.sort(np.random.rand(n_samples))
y = true_fun(X) + np.random.randn(n_samples) * 0.1
X_test = np.linspace(0, 1, 100)

# Compare degrees
degrees = [1, 4, 15] # Underfit (High Bias), Good Fit, Overfit (High Variance)

plt.figure(figsize=(14, 4))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = make_pipeline(polynomial_features, linear_regression)
    pipeline.fit(X[:, np.newaxis], y)
    
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, edgecolor='b', s=20, label="Samples")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title(f"Degree {degrees[i]}")

# plt.show() # Uncomment to view
```

---

## 3. Learning Curves

Learning curves plot the model's performance on the training set and validation set as a function of the training set size (number of instances). They are excellent tools for diagnosing Bias vs. Variance.

### 3.1 Diagnosing High Bias (Underfitting)
*   Both the training error and validation error are **high**.
*   The curves converge quickly.
*   **Solution**: Adding more training data will **NOT** help. You need a more complex model, more features, or less regularization.

### 3.2 Diagnosing High Variance (Overfitting)
*   There is a **large gap** between the training error (low) and validation error (high).
*   The training error remains low, but the validation error refuses to drop.
*   **Solution**: Adding more training data **WILL** help. Alternatively, you need a simpler model, fewer features, or more regularization.

```python
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# Example usage:
# plot_learning_curve(LinearRegression(), "Learning Curves (Linear Reg)", X[:, np.newaxis], y, cv=5)
```

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Generate a synthetic dataset and plot the learning curves for a highly complex model (like PolynomialFeatures degree 15) vs a simple model.
- 🟡 **Intermediate**: Implement the bias-variance decomposition mathematically using bootstrapping on a small dataset.

### What's Next
| Next | Why |
|------|-----|
| [Interpretability Explainability](./05-Interpretability-Explainability.md) | Learn how to interpret complex models. |

---

[← Hyperparameter Tuning](./03-Hyperparameter-Tuning.md) | [Back to Index](../README.md) | [Next: Interpretability Explainability →](./05-Interpretability-Explainability.md)
