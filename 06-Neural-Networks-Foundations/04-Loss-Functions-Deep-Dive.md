# 📉 Loss Functions Deep Dive

> **Prerequisites**: Backpropagation, Calculus | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [What is a Loss Function?](#1-what-is-a-loss-function)
2. [Regression Loss Functions](#2-regression-loss-functions)
3. [Classification Loss Functions (Cross-Entropy)](#3-classification-loss-functions-cross-entropy)
4. [Handling Imbalanced Data: Focal Loss](#4-handling-imbalanced-data-focal-loss)
5. [Self-Supervised & Embedding Loss: Contrastive Loss](#5-self-supervised--embedding-loss-contrastive-loss)
6. [Visualizing Loss Landscapes](#6-visualizing-loss-landscapes)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. What is a Loss Function?

A **Loss Function** $L(y, \hat{y})$ measures how "wrong" a single prediction $\hat{y}$ is compared to the true label $y$. 
The **Cost Function** $J(\mathbf{W}, \mathbf{b})$ is the average of the loss functions across the entire training dataset.

In neural networks, the choice of loss function fundamentally dictates the geometric space the network learns to map inputs into.

---

## 2. Regression Loss Functions

Used when predicting continuous numerical values (e.g., House Prices, Stock Values).

### 2.1 Mean Squared Error (MSE / L2 Loss)
The standard for regression. Penalizes large errors heavily because of the square.
$$L(y, \hat{y}) = (y - \hat{y})^2$$

- **Pros**: Easy to differentiate, yields a single global minimum for linear regression.
- **Cons**: Extremely sensitive to outliers. A single massive outlier will drag the entire regression line towards it.

### 2.2 Mean Absolute Error (MAE / L1 Loss)
$$L(y, \hat{y}) = |y - \hat{y}|$$

- **Pros**: Robust to outliers.
- **Cons**: The gradient is the same magnitude regardless of how close the prediction is to the target. It is also non-differentiable exactly at $0$ (though we can use subgradients).

### 2.3 Huber Loss
Combines the best of MSE and MAE. It is quadratic (MSE) for small errors and linear (MAE) for large errors.

$$L(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

---

## 3. Classification Loss Functions (Cross-Entropy)

Used when predicting discrete classes. 

### Why not use MSE for Classification?
If we use Sigmoid activation with MSE, the loss landscape becomes non-convex (it gets wavy, full of local minima). Cross-Entropy paired with Sigmoid/Softmax mathematically cancels out the annoying exponential terms, resulting in a beautiful, convex loss surface (for linear models) or at least a much smoother one for deep networks.

### 3.1 Binary Cross-Entropy (BCE)
Used for binary classification (Output layer has 1 neuron with Sigmoid).
$$L(y, \hat{y}) = - \left[ y \log(\hat{y}) + (1-y) \log(1-\hat{y}) \right]$$

**Intuition**:
- If true class $y=1$, the formula collapses to $- \log(\hat{y})$. To minimize this, $\hat{y}$ must be pushed towards 1.
- If true class $y=0$, the formula collapses to $- \log(1-\hat{y})$. To minimize this, $\hat{y}$ must be pushed towards 0.

### 3.2 Categorical Cross-Entropy (CCE)
Used for multi-class classification where classes are mutually exclusive (Output layer has $K$ neurons with Softmax).
$$L(\mathbf{y}, \mathbf{\hat{y}}) = - \sum_{k=1}^{K} y_k \log(\hat{y}_k)$$

Because $\mathbf{y}$ is one-hot encoded, only the true class index $c$ is non-zero (it equals 1). Thus, the loss simplifies to:
$$L = - \log(\hat{y}_c)$$
The network only explicitly cares about maximizing the predicted probability of the *correct* class. (Because Softmax probabilities must sum to 1, pushing the correct class up naturally forces the incorrect classes down).

---

## 4. Handling Imbalanced Data: Focal Loss

Invented by Facebook AI Research (FAIR) in 2017 for the RetinaNet object detector.

**The Problem**: In an image, 99% of the bounding boxes are background (easy negative examples). Even if the BCE loss for each easy background is tiny (e.g., 0.01), summing up 100,000 of them overwhelms the loss from the 3 rare, actual objects.

**The Solution**: Focal Loss dynamically scales the cross-entropy loss based on the prediction confidence.
$$L_{focal} = - (1 - \hat{p})^\gamma \log(\hat{p})$$
*(where $\hat{p}$ is the predicted probability for the true class, and $\gamma \ge 0$ is a focusing parameter, typically 2).*

**How it works**:
- If an example is easy and the network predicts $\hat{p} = 0.9$, the scaling factor becomes $(1 - 0.9)^2 = 0.01$. The loss is crushed down to nearly zero.
- If an example is hard and the network predicts $\hat{p} = 0.1$, the scaling factor becomes $(1 - 0.1)^2 = 0.81$. The loss remains high.
- The network is forced to ignore easy examples and focus entirely on the hard ones.

---

## 5. Self-Supervised & Embedding Loss: Contrastive Loss

Used in Facial Recognition, Sentence Transformers, and Self-Supervised Learning (SimCLR). Instead of predicting a class, the network learns to map inputs into a high-dimensional vector space where similar items are close together, and dissimilar items are far apart.

### 5.1 Triplet Loss
You provide three inputs:
1. **Anchor (A)**: An image of Person 1.
2. **Positive (P)**: A different image of Person 1.
3. **Negative (N)**: An image of Person 2.

The network embeds them into vectors. Let $D(x, y)$ be the Euclidean distance between two vectors.
$$L = \max(D(A, P) - D(A, N) + \alpha, 0)$$
*(where $\alpha$ is a margin, e.g., 0.2).*

**Goal**: Force the distance between Anchor and Negative to be greater than the distance between Anchor and Positive by at least the margin $\alpha$.

### 5.2 InfoNCE (Used in Contrastive Learning)
Used heavily in modern Large Language Models (for text embeddings) and Vision (CLIP). Given an anchor, 1 positive example, and $K$ negative examples:
$$L = - \log \frac{\exp(\text{sim}(A, P) / \tau)}{\exp(\text{sim}(A, P) / \tau) + \sum_{k=1}^K \exp(\text{sim}(A, N_k) / \tau)}$$
*(where $\text{sim}$ is cosine similarity, and $\tau$ is a temperature hyperparameter).*

---

## 6. Visualizing Loss Landscapes

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Regression Losses
y_true = 0
y_pred = np.linspace(-3, 3, 200)

mse = (y_true - y_pred)**2
mae = np.abs(y_true - y_pred)

delta = 1.0
huber = np.where(np.abs(y_true - y_pred) <= delta, 
                 0.5 * (y_true - y_pred)**2, 
                 delta * np.abs(y_true - y_pred) - 0.5 * delta**2)

# 2. Classification Losses (for true class y=1)
p = np.linspace(0.01, 0.99, 200)
ce = -np.log(p)

gamma = 2.0
focal = -(1 - p)**gamma * np.log(p)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot Regression
axes[0].plot(y_pred, mse, label='MSE (L2)', lw=2)
axes[0].plot(y_pred, mae, label='MAE (L1)', lw=2)
axes[0].plot(y_pred, huber, label='Huber (delta=1)', lw=2, linestyle='--')
axes[0].set_title('Regression Losses (Target = 0)', fontweight='bold')
axes[0].set_xlabel('Prediction')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_ylim(-0.5, 9)

# Plot Classification
axes[1].plot(p, ce, label='Cross-Entropy', lw=2)
axes[1].plot(p, focal, label='Focal Loss (gamma=2)', lw=2, color='red')
axes[1].set_title('Classification Losses (True Class = 1)', fontweight='bold')
axes[1].set_xlabel('Predicted Probability (p)')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('loss_functions.png', dpi=150)
plt.show()
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Focal Loss Implementation**: Write a custom PyTorch loss function for Focal Loss. Use it to train a model on a highly imbalanced dataset (e.g., Credit Card Fraud, where 99.8% of cases are class 0) and compare the F1 score against standard BCE Loss.

### What's Next
| Next | Why |
|------|-----|
| [Optimizers Deep Dive](./05-Optimizers-Deep-Dive.md) | We've defined the Loss. Now we need the algorithms (SGD, Adam) to navigate the loss landscape efficiently. |

---

[← Activation Functions](./03-Activation-Functions.md) | [Back to Index](../README.md) | [Next: Optimizers Deep Dive →](./05-Optimizers-Deep-Dive.md)
