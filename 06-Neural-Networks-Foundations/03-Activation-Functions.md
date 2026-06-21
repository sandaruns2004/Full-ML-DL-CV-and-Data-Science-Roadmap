# ⚡ Activation Functions

---

## 📋 Table of Contents
1. [Beginner: The Gatekeepers of Neural Networks](#1-beginner-the-gatekeepers-of-neural-networks)
2. [Intermediate: Python Implementation & Derivatives](#2-intermediate-python-implementation--derivatives)
3. [Advanced: Mathematical Derivations & Research Insights](#3-advanced-mathematical-derivations--research-insights)

---

## 1. Beginner: The Gatekeepers of Neural Networks

### Simple Explanation
In a neural network, a neuron receives inputs, multiplies them by weights, adds a bias, and gets a single number (the weighted sum). If we passed this number directly to the next layer, our network would only be performing linear equations (like drawing a straight line).
An **activation function** is a non-linear mathematical operation applied to the weighted sum. It decides whether and to what extent a neuron should "fire" (send its signal to the next layer). Non-linear activations allow the network to bend, twist, and curve the space, enabling it to learn complex, non-linear patterns (like curves, circles, and shapes).

### Real-World Analogy: The Audition Judge
Imagine a talent show with a judge:
- **Linear Transformation**: The judge scores the contestant on different categories (voice quality, stage presence) and sums up the points.
- **Activation Function (Sigmoid-like)**: The judge has a threshold. If the contestant gets a score below the threshold, they get a $0$ (rejected). If they get a score above it, they get a $1$ (passed). The transition might be a smooth curve where the closer they are to passing, the more enthusiastic the judge's recommendation.
- **ReLU-like**: If the contestant's score is negative or zero, they get ignored ($0$). If they get a positive score, their pass probability is directly proportional to their score ($z$).

### Visual Intuition
Without non-linear activations, stacking multiple layers is equivalent to a single linear layer. The non-linearities bend the space so that non-linearly separable classes (like XOR or circular clusters) can be cleanly separated.

---

## 2. Intermediate: Python Implementation & Derivatives

Let us implement the most common activation functions and their derivatives in pure NumPy.

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1 - s)

# 2. Tanh
def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(z)**2

# 3. ReLU (Rectified Linear Unit)
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0, 1.0, 0.0)

# 4. Leaky ReLU
def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, 1.0, alpha)

# 5. Softmax (Numerical stability: subtract max)
def softmax(z: np.ndarray) -> np.ndarray:
    # Assumes z is shape (batch_size, num_classes) or (num_classes,)
    if z.ndim == 1:
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)
    else:
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
```

### Visualizing Functions & Derivatives
Using matplotlib, we can visualize how these functions transform inputs and how their gradients change across the domain.

```python
# Generate inputs
z = np.linspace(-5, 5, 200)

plt.figure(figsize=(12, 8))

# Plot Sigmoid
plt.subplot(2, 2, 1)
plt.plot(z, sigmoid(z), label="Sigmoid", color="blue", lw=2)
plt.plot(z, sigmoid_derivative(z), label="Derivative", color="cyan", linestyle="--")
plt.title("Sigmoid & Derivative")
plt.grid(True)
plt.legend()

# Plot Tanh
plt.subplot(2, 2, 2)
plt.plot(z, tanh(z), label="Tanh", color="red", lw=2)
plt.plot(z, tanh_derivative(z), label="Derivative", color="orange", linestyle="--")
plt.title("Tanh & Derivative")
plt.grid(True)
plt.legend()

# Plot ReLU
plt.subplot(2, 2, 3)
plt.plot(z, relu(z), label="ReLU", color="green", lw=2)
plt.plot(z, relu_derivative(z), label="Derivative", color="lime", linestyle="--")
plt.title("ReLU & Derivative")
plt.grid(True)
plt.legend()

# Plot Leaky ReLU
plt.subplot(2, 2, 4)
plt.plot(z, leaky_relu(z), label="Leaky ReLU", color="purple", lw=2)
plt.plot(z, leaky_relu_derivative(z), label="Derivative", color="violet", linestyle="--")
plt.title("Leaky ReLU & Derivative")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
```

---

## 3. Advanced: Mathematical Derivations & Research Insights

### Derivative Derivations

#### 1. Sigmoid Derivative
Let $\sigma(z) = \frac{1}{1 + e^{-z}}$. Applying the quotient rule:
$$\sigma'(z) = \frac{d}{dz} (1 + e^{-z})^{-1} = -1(1 + e^{-z})^{-2} \cdot (-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}$$
$$\sigma'(z) = \left(\frac{1}{1 + e^{-z}}\right) \left(\frac{e^{-z}}{1 + e^{-z}}\right) = \sigma(z) \cdot (1 - \sigma(z))$$

#### 2. Tanh Derivative
Let $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$. Since $\tanh(z) = \frac{\sinh(z)}{\cosh(z)}$:
$$\tanh'(z) = \frac{\cosh(z)\cosh(z) - \sinh(z)\sinh(z)}{\cosh^2(z)} = \frac{\cosh^2(z) - \sinh^2(z)}{\cosh^2(z)} = \frac{1}{\cosh^2(z)} = 1 - \tanh^2(z)$$

#### 3. Softmax Jacobian Derivation
The Softmax function for index $i$ is defined as $a_i = \frac{e^{z_i}}{\sum_k e^{z_k}}$. 
Since the output $a_i$ depends on all elements of the input vector $\mathbf{z}$, the derivative $\frac{\partial a_i}{\partial z_j}$ is a Jacobian matrix.

Using the Quotient Rule:
$$\frac{\partial a_i}{\partial z_j} = \frac{\frac{\partial}{\partial z_j}(e^{z_i}) \cdot \sum_k e^{z_k} - e^{z_i} \cdot \frac{\partial}{\partial z_j}(\sum_k e^{z_k})}{\left(\sum_k e^{z_k}\right)^2}$$

**Case 1: $i = j$**
$$\frac{\partial a_i}{\partial z_i} = \frac{e^{z_i} \sum_k e^{z_k} - e^{z_i} e^{z_i}}{\left(\sum_k e^{z_k}\right)^2} = \frac{e^{z_i}}{\sum_k e^{z_k}} - \left(\frac{e^{z_i}}{\sum_k e^{z_k}}\right)^2 = a_i(1 - a_i)$$

**Case 2: $i \neq j$**
$$\frac{\partial a_i}{\partial z_j} = \frac{0 - e^{z_i} e^{z_j}}{\left(\sum_k e^{z_k}\right)^2} = -\left(\frac{e^{z_i}}{\sum_k e^{z_k}}\right)\left(\frac{e^{z_j}}{\sum_k e^{z_k}}\right) = -a_i a_j$$

In Kronecker Delta notation ($\delta_{ij} = 1$ if $i=j$, else $0$):
$$\frac{\partial a_i}{\partial z_j} = a_i (\delta_{ij} - a_j)$$

### Research Insights
- **Sparsity vs. Information Loss**: ReLU introduces sparsity (zero-activations), which reduces memory overhead and computation. However, it can lead to the **Dying ReLU** problem (where neurons get locked into the negative half-space with a zero gradient). Leaky ReLU and ELU address this by keeping a small gradient channel alive.
- **Smoothness (GELU & Swish)**: Standard ReLU has a sharp non-differentiable point at $z=0$. State-of-the-art architectures (like Transformers) use **GELU** ($z \cdot \Phi(z)$) or **Swish** ($z \cdot \sigma(\beta z)$), which are smooth, non-monotonic curves that dip below zero. This smooth landscape significantly improves optimization convergence rates.

---

[← Previous: Perceptron And Biological Analogy](./02-Perceptron-And-Biological-Analogy.md) | [Back to Index](./README.md) | [Next: Forward Propagation →](./04-Forward-Propagation.md)
