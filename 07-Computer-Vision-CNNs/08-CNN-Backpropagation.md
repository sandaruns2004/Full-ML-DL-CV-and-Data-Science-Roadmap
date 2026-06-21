# 🔄 CNN Backpropagation

---

## 📋 Table of Contents
1. [Beginner: Feedback Across Shared Patterns](#1-beginner-feedback-across-shared-patterns)
2. [Intermediate: Spatial Chain Rule Walkthrough](#2-intermediate-spatial-chain-rule-walkthrough)
3. [Advanced: Derivation of Convolutional Gradients](#3-advanced-derivation-of-convolutional-gradients)

---

## 1. Beginner: Feedback Across Shared Patterns

### Simple Intuition
During backpropagation in a standard MLP, each connection has its own individual weight. In a Convolutional Neural Network, we use **Weight Sharing**—the same filter is used across the entire image. 
When calculating the gradient for a shared filter weight, we must sum up the gradients from all the locations where that filter was applied.

### Real-World Analogy: The Quality Inspector
Imagine a factory quality inspector checking 100 chocolate bars using the same stencil:
- If the stencil has a defect (e.g. a small hole), it will leave a mark on all 100 chocolate bars.
- To figure out how to fix the stencil (weight update), the inspector doesn't just look at one chocolate bar. They look at all 100 bars, sum up the errors, and make a single adjustment to the stencil so it prints correctly on the next batch.

---

## 2. Intermediate: Spatial Chain Rule Walkthrough

Let us write a NumPy script to demonstrate how gradients are routed back to the weights of a 2D convolution kernel.

```python
import numpy as np

# Inputs: 3x3 image, 2x2 kernel. Stride=1, Padding=0.
x = np.array([
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 4.0],
    [2.0, 1.0, 0.0]
])

w = np.array([
    [0.5, -0.5],
    [1.0,  0.0]
])

# Forward pass
# Output shape: (2, 2)
z = np.array([
    [np.sum(x[0:2, 0:2] * w), np.sum(x[0:2, 1:3] * w)],
    [np.sum(x[1:3, 0:2] * w), np.sum(x[1:3, 1:3] * w)]
])

# Upstream gradient of loss with respect to output z (e.g., from MSE loss)
dL_dz = np.array([
    [1.0, -1.0],
    [2.0,  0.0]
])

# Backward pass to find gradient of weights dL_dw
# Each kernel weight is applied at multiple locations, so we sum the gradients
dL_dw = np.zeros_like(w)
for i in range(2):
    for j in range(2):
        dL_dw += dL_dz[i, j] * x[i:i+2, j:j+2]

print("Gradient dL_dw:\n", dL_dw)
```

---

## 3. Advanced: Derivation of Convolutional Gradients

Let us derive the mathematical equations for backpropagation through a convolutional layer.

Let:
- $\mathbf{X}$ be the input matrix.
- $\mathbf{W}$ be the kernel filter of size $K \times K$.
- $\mathbf{Z} = \mathbf{X} * \mathbf{W}$ be the output feature map.
- $\mathcal{L}$ be the final scalar loss.

### 1. Gradient with respect to the weights ($\mathbf{W}$)
Since the weight $W_{m,n}$ is used to calculate all output elements $Z_{i,j}$, we apply the multivariate chain rule:
$$\frac{\partial \mathcal{L}}{\partial W_{m,n}} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial Z_{i,j}} \frac{\partial Z_{i,j}}{\partial W_{m,n}}$$

Since $Z_{i,j} = \sum_{m'} \sum_{n'} X_{i+m', j+n'} W_{m',n'}$, the local derivative is:
$$\frac{\partial Z_{i,j}}{\partial W_{m,n}} = X_{i+m, j+n}$$

Substituting this in:
$$\frac{\partial \mathcal{L}}{\partial W_{m,n}} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial Z_{i,j}} X_{i+m, j+n}$$
This is equivalent to a cross-correlation between the input $\mathbf{X}$ and the upstream gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}}$.

### 2. Gradient with respect to the inputs ($\mathbf{X}$)
To pass the gradient backward to the previous layer, we calculate the derivative of the loss with respect to input $X_{p,q}$:
$$\frac{\partial \mathcal{L}}{\partial X_{p,q}} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial Z_{i,j}} \frac{\partial Z_{i,j}}{\partial X_{p,q}}$$

Since $p = i+m \implies m = p-i$ and $q = j+n \implies n = q-j$:
$$\frac{\partial \mathcal{L}}{\partial X_{p,q}} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial Z_{i,j}} W_{p-i, q-j}$$
This equation is a mathematically formal convolution (which flips the kernel weights $\mathbf{W}$) between the upstream gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}}$ and the weights $\mathbf{W}$.

---

[← Previous: Edge Detection And Feature Maps](./07-Edge-Detection-And-Feature-Maps.md) | [Back to Index](./README.md) | [Next: LeNet And AlexNet →](./09-LeNet-And-AlexNet.md)
