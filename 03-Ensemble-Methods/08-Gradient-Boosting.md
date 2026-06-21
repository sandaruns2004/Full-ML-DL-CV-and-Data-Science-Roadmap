# 📉 Gradient Boosting (GBM)

> **Prerequisites**: AdaBoost, Calculus Foundations | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 📋 Table of Contents
1. [Residual Learning Intuition](#1-residual-learning-intuition)
2. [Mathematical Working Mechanism](#2-mathematical-working-mechanism)
3. [Stochastic Gradient Boosting](#3-stochastic-gradient-boosting)
4. [Python Demonstration: Fitting Residuals](#4-python-demonstration-fitting-residuals)

---

## 1. Residual Learning Intuition

### 🟢 Beginner
**Simple Explanation**: Instead of tweaking sample weights (like AdaBoost), Gradient Boosting tries to predict the *errors* (residuals) of the previous model. 

Imagine you are trying to guess a house price ($300k).
1. Model 1 predicts $200k. (Error = $100k)
2. Model 2 is trained to predict the *error* ($100k), and predicts $80k.
3. The combined model predicts $200k + $80k = $280k. (New Error = $20k)
4. Model 3 is trained to predict the new error ($20k)...
This sequence continues, and we sum the predictions up!

---

## 2. Mathematical Working Mechanism

### 🟡 Intermediate
1. **Initialize the model** with a constant value (typically the mean of the target variable for regression):

   $$F_0(x) = \arg\min_\gamma \sum_{i=1}^N L(y_i, \gamma)$$

2. **Iterate** for $m = 1$ to $M$:
   - For each sample $i$, compute the **pseudo-residuals** (negative gradients of the loss function $L$ with respect to the prediction):
   
     $$r_{im} = -\left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}$$
     
     *(Note: Under Squared Error Loss $L(y, F) = \frac{1}{2}(y - F)^2$, the pseudo-residual is exactly the residual: $r_{im} = y_i - F_{m-1}(x_i)$.)*
   - Fit a base learner (e.g. regression tree) $h_m(x)$ to the pseudo-residuals $r_{im}$.
   - Compute the optimal step size (leaf values) $\gamma_m$ to minimize the loss.
   - Update the model:
   
     $$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
     
     *(where $\eta$ is the learning rate or shrinkage factor, typically between 0.01 and 0.2).*

---

## 3. Stochastic Gradient Boosting

To improve generalizability, at each iteration, the base tree is fit on a random subset of the training data (e.g. 80%) sampled without replacement. This is known as **Stochastic Gradient Boosting** and introduces bagging-like regularization.

---

## 4. Python Demonstration: Fitting Residuals

Here is an end-to-end regression example showing how Gradient Boosting fits residuals sequentially:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Generate non-linear data
np.random.seed(42)
X = np.sort(np.random.uniform(0, 5, 100)).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.randn(100) * 0.2

n_estimators = 5
learning_rate = 0.5
trees = []
predictions = np.zeros(len(y))
residuals_history = [y.copy()]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
X_plot = np.linspace(0, 5, 200).reshape(-1, 1)

for i in range(n_estimators):
    # Compute residuals
    residuals = y - predictions
    residuals_history.append(residuals.copy())
    
    # Fit tree to residuals
    tree = DecisionTreeRegressor(max_depth=2)
    tree.fit(X, residuals)
    trees.append(tree)
    
    # Update predictions
    predictions += learning_rate * tree.predict(X)
    
    # Plot predictions
    ax = axes.flat[i]
    ax.scatter(X, y, alpha=0.3, s=15, color='#36A2EB')
    ax.plot(X_plot, sum(learning_rate * t.predict(X_plot) for t in trees), 
            'r-', linewidth=2, label='Ensemble Model')
    ax.plot(X_plot, np.sin(X_plot.ravel()), 'g--', alpha=0.5, label='True Function')
    mse = np.mean((y - predictions)**2)
    ax.set_title(f'Step {i+1} (MSE: {mse:.4f})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Show residuals shrinking in last panel
ax = axes.flat[5]
for idx, res in enumerate(residuals_history[:5]):
    ax.plot(np.sort(X.ravel()), res[np.argsort(X.ravel())], alpha=0.7, label=f'Step {idx}')
ax.set_title('Residuals Shrinking Over Steps', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('Gradient Boosting: Sequential Residual Fitting', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

[← AdaBoost](./07-AdaBoost.md) | [Back to Index](../README.md) | [Next: XGBoost →](./09-XGBoost-Concepts.md)
