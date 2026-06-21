# 🧠 Introduction to Neural Networks

---

## 📋 Table of Contents
1. [Beginner: Simple Explanation & Real-World Analogy](#1-beginner-simple-explanation--real-world-analogy)
2. [Intermediate: Structure and Basic Workflow](#2-intermediate-structure-and-basic-workflow)
3. [Advanced: Mathematical Formulations & Tensor Layouts](#3-advanced-mathematical-formulations--tensor-layouts)

---

## 1. Beginner: Simple Explanation & Real-World Analogy

### What is a Neural Network?
An Artificial Neural Network (ANN) is a computational model inspired by the structure and function of biological brains. At its core, it is a machine designed to learn patterns from data. If you show it enough examples of something (e.g., pictures of cats and dogs), it will learn to recognize the underlying differences and correctly classify new, unseen examples.

### Real-World Analogy: The Committee Decision
Imagine a company trying to decide whether to invest in a new product. Instead of one person deciding, there is a **committee**:
1. **The Inputs (Market Analysts)**: Each analyst gathers a specific piece of information (e.g., consumer demand, production cost, competitor activity).
2. **The Committee Members (Neurons)**: Each member listens to all the analysts. However, they don't value everyone's opinion equally. A member might trust the "consumer demand" analyst more than the "competitor activity" analyst. These preferences are **weights**.
3. **The Final Board (Output Layer)**: The committee members debate and vote. If their collective output exceeds a certain threshold, the company invests; otherwise, it does not.

### Visual Intuition
Here is the conceptual diagram of a single neuron:

```
Inputs (x)       Weights (w)
   x₁ ----------->  w₁  ----\
   x₂ ----------->  w₂  -----> ∑ (Weighted Sum) ---> Activation Function (f) ---> Output (y)
   x₃ ----------->  w₃  ----/
                     b  ---/ (Bias)
```

---

## 2. Intermediate: Structure and Basic Workflow

A neural network is organized into layers of computational nodes (neurons):
1. **Input Layer**: Receives the raw features from the dataset.
2. **Hidden Layer(s)**: Intermediate layers that extract higher-level representations.
3. **Output Layer**: Produces the final prediction.

### NumPy Walkthrough: Simple Input-to-Output Flow
Below is a simple Python snippet demonstrating how an input vector passes through a single-layer network using matrix math:

```python
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Input features (e.g., 3 features: temperature, humidity, wind speed)
X = np.array([0.5, -0.2, 0.1])

# Weights for a single output neuron (3 inputs -> 1 output)
W = np.random.randn(3)
# Bias term
b = 0.1

# 1. Weighted Sum (Linear Step)
z = np.dot(X, W) + b

# 2. Firing (Non-linear Activation - Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

output = sigmoid(z)
print(f"Linear input (z): {z:.4f}")
print(f"Network Output (y_hat): {output:.4f}")
```

---

## 3. Advanced: Mathematical Formulations & Tensor Layouts

Let us define the structure of a fully-connected feedforward neural network mathematically.

### Mathematical Notation
Let $L$ be the number of layers in the network. For any layer $l \in \{1, 2, \dots, L\}$:
- $n_l$ denotes the number of neurons in layer $l$.
- $\mathbf{W}^{[l]} \in \mathbb{R}^{n_{l-1} \times n_l}$ is the weight matrix, where $W^{[l]}_{j, k}$ represents the weight connecting neuron $j$ in layer $l-1$ to neuron $k$ in layer $l$.
- $\mathbf{b}^{[l]} \in \mathbb{R}^{1 \times n_l}$ is the bias vector of layer $l$.
- $\mathbf{z}^{[l]} \in \mathbb{R}^{1 \times n_l}$ is the pre-activation vector.
- $\mathbf{a}^{[l]} \in \mathbb{R}^{1 \times n_l}$ is the activation vector, with $\mathbf{a}^{[0]} = \mathbf{x}$ (the input vector).

For a batch of $m$ training samples, we stack the vectors into matrices:
- $\mathbf{X} \in \mathbb{R}^{m \times n_0}$ is the input matrix.
- $\mathbf{Z}^{[l]} \in \mathbb{R}^{m \times n_l}$ is the linear pre-activation matrix.
- $\mathbf{A}^{[l]} \in \mathbb{R}^{m \times n_l}$ is the activation matrix.

The computation for layer $l$ is defined as:
$$\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \mathbf{W}^{[l]} + \mathbf{b}^{[l]}$$
$$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$
where $g^{[l]}$ is the element-wise non-linear activation function of layer $l$.

### Industry Practices
- **Batching**: Always process data in mini-batches rather than single instances. Modern GPU architectures are optimized to execute matrix-matrix multiplications ($\mathbf{A}^{[l-1]} \mathbf{W}^{[l]}$) in parallel rather than matrix-vector loops.
- **Precision**: Use 32-bit floating point precision (`float32` or `FP32`) or mixed precision (`FP16`/`BF16`) for training. 64-bit float (`double`) is rarely used in deep learning because the performance overhead outweighs any utility of double precision.

---

[← Back to Index](./README.md) | [Next: Perceptron And Biological Analogy →](./02-Perceptron-And-Biological-Analogy.md)
