# ⚖️ Weight Initialization Deep Dive

> **Prerequisites**: Backpropagation, Regularization | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Curse of Bad Initialization](#1-the-curse-of-bad-initialization)
2. [Zero & Random Normal Initialization (Why they fail)](#2-zero--random-normal-initialization-why-they-fail)
3. [The Core Goal: Preserving Variance](#3-the-core-goal-preserving-variance)
4. [Xavier (Glorot) Initialization (For Sigmoid/Tanh)](#4-xavier-glorot-initialization-for-sigmoidtanh)
5. [He (Kaiming) Initialization (For ReLU)](#5-he-kaiming-initialization-for-relu)
6. [LSUV: Layer-Sequential Unit-Variance](#6-lsuv-layer-sequential-unit-variance)
7. [Visualizing Variance Across Layers](#7-visualizing-variance-across-layers)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. The Curse of Bad Initialization

In deep networks, gradients must flow backwards through dozens or hundreds of layers. 
- If weights are initialized too small, the signal shrinks exponentially as it passes through the network. This causes **Vanishing Gradients**, where early layers never learn.
- If weights are initialized too large, the signal grows exponentially. This causes **Exploding Gradients**, where the loss becomes `NaN` (infinity).

Proper weight initialization ensures that the signal (forward pass) and the gradients (backward pass) maintain a stable variance throughout the entire network.

---

## 2. Zero & Random Normal Initialization (Why they fail)

### 2.1 Zero Initialization
If all weights are initialized to $0$, every neuron in a layer computes the exact same output. During backpropagation, they all receive the exact same gradient. Because they update identically, the network acts as if it only has a single neuron per layer. 
**Rule**: *Never initialize weights to zero. (Biases can be initialized to zero, though).*

### 2.2 Random Normal Initialization
We might initialize weights from a standard normal distribution: $\mathbf{W} \sim \mathcal{N}(0, 1)$.
Assume we have an input vector $\mathbf{x}$ of size $n$, and we compute the dot product:
$$z = \sum_{i=1}^{n} w_i x_i$$
Since both $w_i$ and $x_i$ have a variance of 1 (and assuming they have mean 0), the variance of their product $w_i x_i$ is 1. The variance of the sum of $n$ independent terms is the sum of their variances:
$$\text{Var}(z) = n$$

If $n = 500$ (a standard layer width), the output $z$ has a variance of $500$! The outputs explode instantly. When passed into a Sigmoid/Tanh, they immediately saturate to $1$ or $-1$, killing the gradient completely.

---

## 3. The Core Goal: Preserving Variance

To stop the variance from growing to $n$, we need the variance of the outputs to equal the variance of the inputs:
$$\text{Var}(z) = \text{Var}(x)$$
Since $\text{Var}(z) = n \cdot \text{Var}(w) \cdot \text{Var}(x)$, we can preserve variance if we force:
$$n \cdot \text{Var}(w) = 1 \implies \text{Var}(w) = \frac{1}{n}$$

This is the fundamental mathematical basis for all modern initialization strategies: **The variance of the weights should be inversely proportional to the number of incoming connections (`fan_in`).**

---

## 4. Xavier (Glorot) Initialization (For Sigmoid/Tanh)

Invented by Xavier Glorot and Yoshua Bengio in 2010.

If we are using **Sigmoid** or **Tanh** activations, they are approximately linear near $0$. Therefore, if we keep the activations centered around 0 with variance 1, the signal passes through the activation function unchanged.

Glorot proposed drawing weights from a distribution with variance $\frac{1}{n_{in}}$.
To compromise between the forward pass (which depends on `fan_in`) and the backward pass (which depends on `fan_out`), the standard Xavier variance uses the average:
$$\text{Var}(\mathbf{W}) = \frac{2}{\text{fan\_in} + \text{fan\_out}}$$

**Sampling from a Normal Distribution**:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}\right)$$

**Sampling from a Uniform Distribution** (where bounds are $\pm \sqrt{3 \times \text{Var}}$):
$$W \sim U\left(-\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}, \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}\right)$$

---

## 5. He (Kaiming) Initialization (For ReLU)

Invented by Kaiming He et al. in 2015 for training the famous ResNet.

Xavier initialization breaks down when we use **ReLU** activations. ReLU zeros out exactly half of the normal distribution (all negative numbers). Because half the data is discarded, the variance of the outputs is essentially cut in half at every layer.

To compensate for discarding half the variance, Kaiming He proposed simply multiplying the Xavier variance by $2$.

$$\text{Var}(\mathbf{W}) = \frac{2}{\text{fan\_in}}$$

**Sampling from a Normal Distribution**:
$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{\text{fan\_in}}}\right)$$

Using He initialization is what finally allowed researchers to train networks deeper than 30 layers without them dying.

---

## 6. LSUV: Layer-Sequential Unit-Variance

What if we don't want to rely on mathematical approximations of variance? What if we have weird, complex architectures?

**LSUV (Layer-Sequential Unit-Variance)** is an empirical, data-driven initialization method introduced in 2015.

**Algorithm**:
1. Initialize all weights with orthonormal matrices (e.g., using SVD).
2. Take a small batch of training data and pass it through the first layer.
3. Calculate the variance of the output of that layer.
4. Divide the weights of that layer by $\sqrt{\text{Variance}}$. Now the output variance is exactly 1.0!
5. Move to the next layer and repeat.

LSUV physically forces every single layer in the network to have an output variance of 1.0 for the first batch of data, guaranteeing perfect signal propagation regardless of the activation function or architecture used.

---

## 7. Visualizing Variance Across Layers

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
dims = [500] * 10  # A 10-layer network, 500 neurons each

def simulate_network(init_scale, activation='tanh'):
    x = np.random.randn(1000, 500) # 1000 samples
    variances = []
    
    for i in range(10):
        # Initialize weights
        if init_scale == 'xavier':
            w = np.random.randn(500, 500) * np.sqrt(1.0 / 500)
        elif init_scale == 'he':
            w = np.random.randn(500, 500) * np.sqrt(2.0 / 500)
        else:
            w = np.random.randn(500, 500) * init_scale
            
        z = np.dot(x, w)
        if activation == 'tanh':
            x = np.tanh(z)
        elif activation == 'relu':
            x = np.maximum(0, z)
            
        variances.append(np.var(x))
    return variances

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Tanh Network
axes[0].plot(simulate_network(0.01, 'tanh'), label='Scale=0.01 (Vanishing)', marker='o')
axes[0].plot(simulate_network(0.1, 'tanh'), label='Scale=0.1 (Exploding)', marker='o')
axes[0].plot(simulate_network('xavier', 'tanh'), label='Xavier (Stable)', marker='o', lw=3, color='black')
axes[0].set_title('Variance across 10 layers (Tanh Activation)', fontweight='bold')
axes[0].set_xlabel('Layer')
axes[0].set_ylabel('Variance')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(alpha=0.3)

# ReLU Network
axes[1].plot(simulate_network('xavier', 'relu'), label='Xavier (Vanishing in ReLU)', marker='o')
axes[1].plot(simulate_network('he', 'relu'), label='He (Stable in ReLU)', marker='o', lw=3, color='black')
axes[1].set_title('Variance across 10 layers (ReLU Activation)', fontweight='bold')
axes[1].set_xlabel('Layer')
axes[1].set_ylabel('Variance')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('weight_initialization_variance.png', dpi=150)
plt.show()
```

---

## 8. Project Ideas & What's Next

### Project Ideas
- 🟢 **LSUV Implementation**: Write a simple PyTorch function that takes a `nn.Sequential` model and a single batch of `X`. Iterate over the modules. If the module is a `nn.Linear` or `nn.Conv2d`, run the forward pass, calculate `torch.var(out)`, and scale the module's weights `module.weight.data /= torch.sqrt(var)`.

### What's Next
| Next | Why |
|------|-----|
| [Frameworks: Keras & PyTorch](./08-Frameworks-Keras-PyTorch.md) | We've learned all the deep theory (Activations, Losses, Optimizers, Initialization). Now let's put it together in code. |

---

[← Regularization Techniques](./06-Regularization-Techniques.md) | [Back to Index](../README.md) | [Next: Frameworks Keras PyTorch →](./08-Frameworks-Keras-PyTorch.md)
