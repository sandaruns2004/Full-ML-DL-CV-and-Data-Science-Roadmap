# ⚡ Activation Functions

> **Prerequisites**: Perceptron & MLP, Calculus | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Why We Need Activation Functions](#1-why-we-need-activation-functions)
2. [The Classical Era: Sigmoid & Tanh](#2-the-classical-era-sigmoid--tanh)
3. [The Modern Era: ReLU and its Variants](#3-the-modern-era-relu-and-its-variants)
4. [State of the Art: GELU & Swish](#4-state-of-the-art-gelu--swish)
5. [Output Layer Activations: Softmax](#5-output-layer-activations-softmax)
6. [Visualizing All Functions and Derivatives](#6-visualizing-all-functions-and-derivatives)
7. [Cheat Sheet: Which Activation to Use?](#7-cheat-sheet-which-activation-to-use)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. Why We Need Activation Functions

Without activation functions, Neural Networks are completely useless for complex tasks.
If a network only consists of linear transformations ($\mathbf{Z} = \mathbf{X} \mathbf{W} + \mathbf{b}$), stacking 100 layers together is mathematically equivalent to just one single layer. 

$$\mathbf{W}^{[3]} (\mathbf{W}^{[2]} (\mathbf{W}^{[1]} \mathbf{X})) = (\mathbf{W}^{[3]} \mathbf{W}^{[2]} \mathbf{W}^{[1]}) \mathbf{X} = \mathbf{W}_{\text{combined}} \mathbf{X}$$

Non-linear activation functions bend and fold the multi-dimensional space, allowing the network to learn arbitrary decision boundaries and approximate any function (Universal Approximation Theorem).

---

## 2. The Classical Era: Sigmoid & Tanh

Before 2012, neural networks almost exclusively used these two functions in their hidden layers.

### 2.1 Sigmoid (Logistic) Function
Maps any real-valued number into the range $(0, 1)$.

- **Formula**: $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Derivative**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
- **Range**: $(0, 1)$

**Why we stopped using it for hidden layers**:
1. **Vanishing Gradients**: The maximum value of the derivative is $0.25$ (at $z=0$). During backpropagation, multiplying by values $< 0.25$ across multiple layers exponentially shrinks the gradient to exactly $0$.
2. **Not Zero-Centered**: Outputs are always positive, causing undesirable zig-zagging dynamics during gradient descent.

*Where it's used today*: ONLY in the final output layer for binary classification.

### 2.2 Tanh (Hyperbolic Tangent)
A scaled and shifted version of the Sigmoid function that solves the zero-centered problem.

- **Formula**: $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
- **Derivative**: $\tanh'(z) = 1 - \tanh^2(z)$
- **Range**: $(-1, 1)$

**Pros/Cons**: It is zero-centered, making optimization easier than Sigmoid. However, it still suffers heavily from the vanishing gradient problem when $z$ is large (positive or negative).

*Where it's used today*: Inside RNNs and LSTMs.

---

## 3. The Modern Era: ReLU and its Variants

In 2012, AlexNet revolutionized Deep Learning by switching to ReLU, allowing deep networks to finally train without gradients vanishing.

### 3.1 ReLU (Rectified Linear Unit)
- **Formula**: $f(z) = \max(0, z)$
- **Derivative**: $f'(z) = 1$ if $z > 0$, else $0$
- **Range**: $[0, \infty)$

**Pros**: 
1. Solves the vanishing gradient problem (derivative is exactly 1 for positive inputs).
2. Extremely fast to compute.
3. Induces sparsity (many neurons output exactly 0).

**The Dying ReLU Problem**:
If a large gradient updates a weight such that a neuron's input is always negative, that neuron will output $0$ forever. Its gradient will be $0$ forever. It is effectively "dead."

### 3.2 Leaky ReLU
Created to fix the Dying ReLU problem by allowing a small, non-zero gradient when $z < 0$.

- **Formula**: $f(z) = \max(\alpha z, z)$ where $\alpha$ is usually $0.01$.
- **Derivative**: $f'(z) = 1$ if $z > 0$, else $\alpha$

### 3.3 ELU (Exponential Linear Unit)
Instead of a sharp angle like Leaky ReLU, ELU curves smoothly for negative values, making it robust to noise.

- **Formula**: $f(z) = z$ if $z > 0$, else $\alpha(e^z - 1)$
- **Derivative**: $f'(z) = 1$ if $z > 0$, else $f(z) + \alpha$

---

## 4. State of the Art: GELU & Swish

In the era of Transformers (BERT, GPT) and massive CNNs (EfficientNet), researchers found functions that slightly outperform ReLU.

### 4.1 GELU (Gaussian Error Linear Unit)
The standard activation function used in **BERT, GPT-3, and Vision Transformers (ViTs)**.
It weighs the input by its probability under a Gaussian distribution.

- **Formula**: $f(z) = z \cdot \Phi(z)$, where $\Phi(z)$ is the standard Gaussian cumulative distribution function.
- **Approximation**: $f(z) \approx 0.5z \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (z + 0.044715z^3)\right)\right)$

**Why it works**: It provides a smoother transition near zero than ReLU, combining properties of dropout, zoneout, and ReLUs.

### 4.2 Swish / SiLU (Sigmoid Linear Unit)
Discovered by Google Brain using automated search techniques. Used in **YOLOv8 and EfficientNet**.

- **Formula**: $f(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$
- **Derivative**: $f'(z) = f(z) + \sigma(z)(1 - f(z))$

**Why it works**: Like GELU, it is non-monotonic (it dips slightly below zero before going up). This negative dip helps pull weights back if they go too far negative, self-stabilizing the training.

---

## 5. Output Layer Activations: Softmax

Used exclusively in the final layer for **Multi-Class Classification**. It converts a vector of raw scores (logits) into a probability distribution that sums to 1.

- **Formula**: $\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$

**The Softmax Derivative (Jacobian):**
Because the output of Softmax node $i$ depends on the input of node $j$ (due to the denominator sum), the derivative forms a matrix (Jacobian):
$$ \frac{\partial S_i}{\partial z_j} = \begin{cases} S_i(1 - S_i) & \text{if } i = j \\ -S_i S_j & \text{if } i \neq j \end{cases} $$

---

## 6. Visualizing All Functions and Derivatives

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-4, 4, 200)

# Definitions
sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
relu = np.maximum(0, z)
leaky_relu = np.maximum(0.1*z, z)
elu = np.where(z > 0, z, 1.0 * (np.exp(z) - 1))
swish = z * sigmoid
gelu = 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))

# Plotting
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
funcs = [
    ('Sigmoid', sigmoid, 'Vanishing Gradients'),
    ('Tanh', tanh, 'Zero-centered, still vanishes'),
    ('ReLU', relu, 'The Modern Standard'),
    ('Leaky ReLU', leaky_relu, 'Fixes Dying ReLU'),
    ('Swish / SiLU', swish, 'Non-monotonic dip (YOLO/EfficientNet)'),
    ('GELU', gelu, 'The Transformer Standard (GPT/BERT)')
]

for ax, (name, y, desc) in zip(axes.flat, funcs):
    ax.plot(z, y, lw=3, color='#2196F3')
    ax.axhline(0, color='black', lw=1, alpha=0.5)
    ax.axvline(0, color='black', lw=1, alpha=0.5)
    ax.set_title(f"{name}\n{desc}", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim(-1.5, 3)

plt.tight_layout()
plt.savefig('all_activations.png', dpi=150)
plt.show()
```

---

## 7. Cheat Sheet: Which Activation to Use?

| Task / Layer | Recommended Activation |
|---|---|
| **Default for Hidden Layers (MLP/CNN)** | ReLU |
| **If your network is dying (all 0s)** | Leaky ReLU |
| **Transformers (LLMs, ViTs)** | GELU |
| **State-of-the-art CNNs (EfficientNet)** | Swish (SiLU) |
| **Recurrent Neural Networks (RNN/LSTM)** | Tanh (for hidden state), Sigmoid (for gates) |
| **Output: Binary Classification** | Sigmoid |
| **Output: Multi-Class Classification** | Softmax |
| **Output: Regression** | Linear (None) |

---

## 8. Project Ideas & What's Next

### Project Ideas
- 🟢 **Swish vs ReLU Benchmark**: Build a 4-layer MLP using PyTorch. Train it on FashionMNIST once using `nn.ReLU()` and once using `nn.SiLU()`. Plot the training loss curves on the same graph to see which converges faster.

### What's Next
| Next | Why |
|------|-----|
| [Loss Functions Deep Dive](./04-Loss-Functions-Deep-Dive.md) | We know how to activate neurons, but how do we mathematically penalize them when they are wrong? |
| [Optimizers Deep Dive](./05-Optimizers-Deep-Dive.md) | Once we have the loss, how do we best update the weights? (Adam, RMSProp). |

---

[← Backpropagation](./02-Backpropagation.md) | [Back to Index](../README.md) | [Next: Loss Functions Deep Dive →](./04-Loss-Functions-Deep-Dive.md)
