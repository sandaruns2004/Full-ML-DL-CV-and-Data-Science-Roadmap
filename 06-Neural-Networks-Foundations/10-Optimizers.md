# 🏎️ Optimizers

---

## 📋 Table of Contents
1. [Beginner: Driving Down the Mountain Faster](#1-beginner-driving-down-the-mountain-faster)
2. [Intermediate: Python Implementations of SGD, Momentum, and Adam](#2-intermediate-python-implementations-of-sgd-momentum-and-adam)
3. [Advanced: Proofs, Bias Corrections, and Decoupled Weight Decay](#3-advanced-proofs-bias-corrections-and-decoupled-weight-decay)

---

## 1. Beginner: Driving Down the Mountain Faster

### Simple Explanation
Once we calculate the direction that reduces our error (the gradient), we need to update our weights. A simple step-by-step adjustment (Stochastic Gradient Descent) works, but it can be slow and get stuck easily. An **Optimizer** is an advanced algorithm that adjusts the learning rate for each weight dynamically, using historical patterns to speed up training and navigate complex error landscapes.

### Real-World Analogy: Heavy Ball vs. Smart Brakes
- **Standard SGD**: Imagine walking down a rocky mountain. You look only at your feet, take a step directly down the slope, and repeat. You easily get stuck in small potholes (local minima).
- **Momentum**: Imagine placing a heavy bowling ball on the mountain. As it rolls, it builds up speed. Its momentum allows it to roll right through small potholes and continue down the mountain.
- **Adam**: Imagine driving down the mountain in a smart car. It measures the slope, remembers past curves (momentum), and dynamically applies the brakes or accelerates individual wheels depending on how slippery the path is (adaptive rates).

---

## 2. Intermediate: Python Implementations of SGD, Momentum, and Adam

Let us implement the update step for SGD, Momentum, and Adam in pure NumPy.

```python
import numpy as np

# Mock parameters and gradients
np.random.seed(42)
W = np.random.randn(5, 5)
dW = np.random.randn(5, 5)

# 1. Stochastic Gradient Descent (SGD)
def sgd_update(W: np.ndarray, dW: np.ndarray, lr: float) -> np.ndarray:
    return W - lr * dW

# 2. SGD with Momentum
class MomentumOptimizer:
    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        self.lr = lr
        self.beta = beta
        self.v = None
        
    def update(self, W: np.ndarray, dW: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(W)
        self.v = self.beta * self.v + (1 - self.beta) * dW
        return W - self.lr * self.v

# 3. Adam (Adaptive Moment Estimation)
class AdamOptimizer:
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, W: np.ndarray, dW: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(W)
            self.v = np.zeros_like(W)
            
        self.t += 1
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * dW
        # Update biased second raw moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dW ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = self.m / (1 - self.beta1 ** self.t)
        # Compute bias-corrected second raw moment estimate
        v_hat = self.v / (1 - self.beta2 ** self.t)
        
        # Update parameters
        return W - (self.lr / (np.sqrt(v_hat) + self.epsilon)) * m_hat
```

---

## 3. Advanced: Proofs, Bias Corrections, and Decoupled Weight Decay

### Derivation of Adam's Bias Correction
When we initialize the first and second moment vectors, we set them to zero: $\mathbf{m}_0 = \mathbf{0}, \mathbf{v}_0 = \mathbf{0}$.
Because they start at 0, the updates are biased toward 0, especially during the first few training steps (when $\beta_1$ and $\beta_2$ are close to 1). We need to correct this bias.

Let us derive the correction for the first moment $\mathbf{m}_t$ (the second moment derivation follows the same steps).
The expansion of $\mathbf{m}_t$ starting from $\mathbf{m}_0 = \mathbf{0}$ is:
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} \mathbf{g}_i$$

We want to find the expectation $\mathbb{E}[\mathbf{m}_t]$ in relation to the true expectation of the gradients $\mathbb{E}[\mathbf{g}_i]$ (assuming the gradients come from a stationary distribution, so $\mathbb{E}[\mathbf{g}_i] = \mathbb{E}[\mathbf{g}_t]$ for all $i$):

$$\mathbb{E}[\mathbf{m}_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} \mathbf{g}_i \right]$$
$$\mathbb{E}[\mathbf{m}_t] = \mathbb{E}[\mathbf{g}_t] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i}$$

The summation is a finite geometric series:
$$\sum_{i=1}^t \beta_1^{t-i} = \frac{1 - \beta_1^t}{1 - \beta_1}$$

Substituting this back in:
$$\mathbb{E}[\mathbf{m}_t] = \mathbb{E}[\mathbf{g}_t] (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[\mathbf{g}_t] (1 - \beta_1^t)$$

Therefore, to ensure the expectation of our estimator $\hat{\mathbf{m}}_t$ equals the true expectation of the gradient $\mathbb{E}[\mathbf{g}_t]$, we must scale it:
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}$$

As $t \to \infty$, the term $\beta_1^t \to 0$, and the correction term $1 - \beta_1^t$ converges to 1, meaning the bias correction is only active during the initial steps of training.

### Adam vs. AdamW (Decoupled Weight Decay)
In standard SGD, L2 Regularization (adding a penalty term $\frac{\lambda}{2} \|\mathbf{w}\|^2$ to the loss) is mathematically equivalent to Weight Decay:
$$\mathbf{w}_{t+1} = (1 - \eta \lambda) \mathbf{w}_t - \eta \nabla \mathcal{L}_0(\mathbf{w}_t)$$

However, in adaptive optimizers like Adam, L2 regularization is added directly to the gradients before dividing by the second moment:
$$\mathbf{g}_t = \nabla \mathcal{L}_0(\mathbf{w}_t) + \lambda \mathbf{w}_t$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t(\mathbf{g}_t)$$

This means the regularization term is scaled by $\frac{1}{\sqrt{\mathbf{v}_t}}$. Weights that have large, frequent gradients will be regularized *less* than weights with small gradients.
**AdamW** solves this by applying weight decay directly to the parameters, decoupling it from the gradient step:
$$\mathbf{w}_{t+1} = (1 - \eta \lambda) \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t(\nabla \mathcal{L}_0(\mathbf{w}_t))$$
This ensures all weights are regularized equally, which is why AdamW is the default optimizer for modern architectures like LLMs and Transformers.

---

[← Weight Initialization](09-Weight-Initialization.md) | [Back to Index](../README.md) | [Next: Vanishing and Exploding Gradients →](11-Vanishing-And-Exploding-Gradients.md)
