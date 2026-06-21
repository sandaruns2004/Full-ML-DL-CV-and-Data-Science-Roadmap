# ⚖️ Weight Initialization

---

## 📋 Table of Contents
1. [Beginner: The Importance of the Starting Line](#1-beginner-the-importance-of-the-starting-line)
2. [Intermediate: Python Implementation of Xavier and He](#2-intermediate-python-implementation-of-xavier-and-he)
3. [Advanced: Mathematical Derivation of Variance Propagation](#3-advanced-mathematical-derivation-of-variance-propagation)

---

## 1. Beginner: The Importance of the Starting Line

### Simple Explanation
Before a neural network starts training, we must assign initial values to its weights and biases. Setting them to arbitrary numbers like all zeros or too large numbers can break the training process completely.
**Weight Initialization** is the practice of setting the starting values of neural network weights to small, random values that are scaled mathematically. This ensures that information propagates stably through the network without disappearing (vanishing) or growing out of control (exploding).

### Real-World Analogy: Tuning a Radio
Imagine you are trying to tune a radio to a faint station:
- If you start with the volume set to $0$, you hear absolutely nothing (Zero Initialization).
- If you start with the volume set to maximum ($100$), the speaker screeches and blows out (Exploding Gradients).
- You want to start with the volume dial set to a moderate level where you can hear the signal and tune it safely (Proper Weight Initialization).

---

## 2. Intermediate: Python Implementation of Xavier and He

We will implement two of the most popular weight initialization methods in pure NumPy:
1. **Xavier (Glorot) Initialization**: Used for layers with Sigmoid or Tanh activations.
2. **He (Kaiming) Initialization**: Used for layers with ReLU or Leaky ReLU activations.

```python
import numpy as np

# Set random seed
np.random.seed(42)

# Xavier (Glorot) Normal Initialization
def xavier_normal_init(fan_in: int, fan_out: int) -> np.ndarray:
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

# Xavier (Glorot) Uniform Initialization
def xavier_uniform_init(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

# He (Kaiming) Normal Initialization
def he_normal_init(fan_in: int) -> np.ndarray:
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std

# He (Kaiming) Uniform Initialization
def he_uniform_init(fan_in: int) -> np.ndarray:
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))

# Let's verify for a layer with 500 inputs and 300 outputs
fan_in, fan_out = 500, 300
w_xavier = xavier_normal_init(fan_in, fan_out)
w_he = he_normal_init(fan_in)

print(f"Xavier standard deviation: {np.std(w_xavier):.4f} (Expected: {np.sqrt(2.0 / (500 + 300)):.4f})")
print(f"He standard deviation: {np.std(w_he):.4f} (Expected: {np.sqrt(2.0 / 500):.4f})")
```

---

## 3. Advanced: Mathematical Derivation of Variance Propagation

Let us derive why the weights must be scaled relative to the size of the layer.

### The Linear Step Setup
Let the output of a linear layer with $n$ inputs be:
$$y = \sum_{i=1}^{n} w_i x_i$$
We assume:
1. $w_i$ and $x_i$ are independent and identically distributed (i.i.d.).
2. The weights $w_i$ are initialized with mean $\mathbb{E}[w_i] = 0$.
3. The inputs $x_i$ also have mean $\mathbb{E}[x_i] = 0$.

### Variance Derivation
The variance of the sum of independent random variables is the sum of their individual variances:
$$\text{Var}(y) = \sum_{i=1}^{n} \text{Var}(w_i x_i)$$

For two independent variables $A$ and $B$:
$$\text{Var}(AB) = \mathbb{E}[A]^2 \text{Var}(B) + \mathbb{E}[B]^2 \text{Var}(A) + \text{Var}(A)\text{Var}(B)$$

Since $\mathbb{E}[w_i] = 0$ and $\mathbb{E}[x_i] = 0$:
$$\text{Var}(w_i x_i) = \text{Var}(w_i)\text{Var}(x_i)$$

Substituting this back into the sum:
$$\text{Var}(y) = n \cdot \text{Var}(w_i)\text{Var}(x_i)$$

To prevent the variance from growing or shrinking as it propagates through the network, we want:
$$\text{Var}(y) = \text{Var}(x_i)$$

This requires:
$$n \cdot \text{Var}(w_i) = 1 \implies \text{Var}(w_i) = \frac{1}{n}$$

### 1. Xavier Initialization (Sigmoid/Tanh)
Assuming the activation function is linear near 0 (like Tanh), the variance of the activation equals the variance of the pre-activation. By averaging `fan_in` (forward pass) and `fan_out` (backward pass), we get the Xavier variance:
$$\text{Var}(W) = \frac{2}{\text{fan\_in} + \text{fan\_out}}$$

### 2. He Initialization (ReLU)
If using the ReLU activation function, all negative inputs are set to zero:
$$a = \max(0, y)$$
Because the input distribution is symmetric around 0, ReLU zeros out exactly half the distribution, which cuts the output variance in half:
$$\text{Var}(a) = \frac{1}{2} \text{Var}(y)$$

To maintain a constant variance through the activation layer, we must double the variance of the weights:
$$\text{Var}(W) = 2 \cdot \frac{1}{n} = \frac{2}{\text{fan\_in}}$$

Without this adjustment, the signal variance would drop by $0.5$ at every ReLU layer, leading to vanishing gradients in very deep networks.

---

[← Previous: Multi-Layer Perceptron](./08-Multi-Layer-Perceptron.md) | [Back to Index](./README.md) | [Next: Optimizers →](./10-Optimizers.md)
