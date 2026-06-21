# 📉 Loss Functions

---

## 📋 Table of Contents
1. [Beginner: Measuring Success and Error](#1-beginner-measuring-success-and-error)
2. [Intermediate: Python Implementations of Core Losses](#2-intermediate-python-implementations-of-core-losses)
3. [Advanced: Probabilistic Derivations & Numerical Stability](#3-advanced-probabilistic-derivations--numerical-stability)

---

## 1. Beginner: Measuring Success and Error

### Simple Explanation
A **Loss Function** (also known as an objective function or error function) measures how far off the neural network's predictions are from the actual correct answers. 
During training, the loss function acts as a guide: the network's goal is to minimize this loss, adjusting its weights until the error is as close to zero as possible.

### Real-World Analogy: Archery Practice
Imagine you are practicing archery:
- The bullseye represents the true target ($y$).
- The arrow's landing spot is the network's prediction ($\hat{y}$).
- The **Loss** is the distance between the arrow and the bullseye. If you hit the center, your loss is $0$. If you miss by a foot, you calculate your mistake and adjust your stance (weights) for the next shot.

---

## 2. Intermediate: Python Implementations of Core Losses

Let us write the most widely used loss functions in pure NumPy, making sure we handle numerical limits (like division by zero or log of zero).

```python
import numpy as np

# 1. Mean Squared Error (MSE) - Regression
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

# 2. Mean Absolute Error (MAE) - Regression
def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

# 3. Binary Cross-Entropy (BCE) - Binary Classification
def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Clip predictions to avoid log(0) and log(1)
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

# 4. Categorical Cross-Entropy (CCE) - Multi-class
def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Assumes y_true is one-hot encoded and y_pred contains probability distributions
    y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    return float(-np.sum(y_true * np.log(y_pred)) / y_true.shape[0])
```

---

## 3. Advanced: Probabilistic Derivations & Numerical Stability

### Probabilistic Interpretation via Maximum Likelihood Estimation (MLE)

Many loss functions are not just arbitrary formulas; they are directly derived from probability theory.

#### 1. MSE derived from Gaussian Assumption
Suppose we assume the target $y$ is generated from a linear/nonlinear model $f(\mathbf{x}; \mathbf{w})$ with added zero-mean Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$:
$$y = f(\mathbf{x}; \mathbf{w}) + \epsilon$$

The conditional probability of $y$ given $\mathbf{x}$ is:
$$p(y \mid \mathbf{x}; \mathbf{w}) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - f(\mathbf{x}; \mathbf{w}))^2}{2\sigma^2}\right)$$

For $m$ independent observations, the Likelihood $L(\mathbf{w})$ is:
$$L(\mathbf{w}) = \prod_{i=1}^m p(y^{(i)} \mid \mathbf{x}^{(i)}; \mathbf{w})$$

Taking the Log-Likelihood:
$$\log L(\mathbf{w}) = -\frac{m}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^m \left(y^{(i)} - f(\mathbf{x}^{(i)}; \mathbf{w})\right)^2$$

To maximize the Likelihood (MLE), we discard constants and minimize the negative log-likelihood (NLL):
$$\text{argmin}_{\mathbf{w}} -\log L(\mathbf{w}) = \text{argmin}_{\mathbf{w}} \sum_{i=1}^m \left(y^{(i)} - f(\mathbf{x}^{(i)}; \mathbf{w})\right)^2$$
Which is exactly the **Sum of Squared Errors** (the core of Mean Squared Error).

#### 2. BCE derived from Bernoulli Assumption
In binary classification, let the label $y \in \{0, 1\}$ follow a Bernoulli distribution, where the probability of $y=1$ is given by our network output $\hat{y} = f(\mathbf{x}; \mathbf{w})$:
$$p(y \mid \mathbf{x}; \mathbf{w}) = \hat{y}^y (1 - \hat{y})^{1 - y}$$

The Log-Likelihood for $m$ independent data points is:
$$\log L(\mathbf{w}) = \sum_{i=1}^m \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

Minimizing the Negative Log-Likelihood (NLL) gives:
$$\mathcal{L}_{\text{BCE}} = - \frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$
Which is exactly the **Binary Cross-Entropy Loss**.

### Numerical Stability: The LogSumExp Trick

When computing Categorical Cross-Entropy, we compute $\log(\text{Softmax}(\mathbf{z}))$. If some elements in the raw logit vector $\mathbf{z}$ are very large (e.g. $z_i = 1000$), computing $e^{1000}$ results in numerical overflow (`inf`). If they are very negative (e.g., $z_i = -1000$), computing $e^{-1000}$ results in underflow (`0.0`).

To avoid this, we use the **LogSumExp** formulation:
$$\log \left( \sum_{i} e^{z_i} \right) = c + \log \left( \sum_{i} e^{z_i - c} \right)$$
where $c = \max_i z_i$.

By choosing $c = \max(\mathbf{z})$, the largest term in the exponent becomes $z_i - c = 0$, making $e^0 = 1$. This guarantees that the exponent terms do not overflow or underflow.

In practice (and in PyTorch), you should never apply Softmax and then take Log-loss sequentially. Instead, use a unified activation-loss function (like `nn.CrossEntropyLoss` in PyTorch) that operates directly on the raw logits using this LogSumExp stabilization.

---

[← Previous: Forward Propagation](./04-Forward-Propagation.md) | [Back to Index](./README.md) | [Next: Gradient Descent →](./06-Gradient-Descent.md)
