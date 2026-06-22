# 🧠 Multi-Layer Perceptron (MLP)

---

## 📋 Table of Contents
1. [Beginner: Stacking Computational Layers](#1-beginner-stacking-computational-layers)
2. [Intermediate: Object-Oriented NumPy MLP Class](#2-intermediate-object-oriented-numpy-mlp-class)
3. [Advanced: The Universal Approximation Theorem & Representation Depth](#3-advanced-the-universal-approximation-theorem--representation-depth)

---

## 1. Beginner: Stacking Computational Layers

### Simple Explanation
A **Multi-Layer Perceptron (MLP)** is a type of feedforward artificial neural network. It consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Except for the input nodes, each node is a neuron that uses a non-linear activation function. 
By stacking these layers, the network can learn increasingly complex, abstract representations of the input data.

### Real-World Analogy: Factory Assembly Line
Imagine a factory line that builds custom cars:
- **Input Layer**: Raw components (metal sheets, engine parts, glass).
- **Hidden Layer 1 (Stamping)**: Cuts the metal into chassis, doors, and hood shapes.
- **Hidden Layer 2 (Welding & Painting)**: Joins the stamped pieces and paints them.
- **Output Layer (Final Assembly)**: Adds the engine and wheels, producing the final car.
No single worker does everything. The raw components are transformed step-by-step through a hierarchy of tasks. Stacking layers in a neural network works the same way: early layers extract basic details (like edges in an image), while deeper layers combine those details to recognize complex objects (like faces or cars).

---

## 2. Intermediate: Object-Oriented NumPy MLP Class

Let us build a modular, multi-layer neural network class in pure NumPy that allows us to define any layer configuration.

```python
import numpy as np

class DenseLayer:
    def __init__(self, in_features: int, out_features: int):
        # Xavier-style initialization
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / (in_features + out_features))
        self.b = np.zeros((1, out_features))
        self.A_prev = None
        self.Z = None
        self.dW = None
        self.db = None

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b
        # Let's use ReLU activation in hidden layers (we'll implement this inline)
        return np.maximum(0, self.Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        # Derivative of ReLU: 1 if Z > 0, else 0
        dZ = dA * (self.Z > 0)
        m = self.A_prev.shape[0]
        self.dW = np.dot(self.A_prev.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m
        return np.dot(dZ, self.W.T)

class NumPyMLP:
    def __init__(self, layer_sizes: list[int]):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(DenseLayer(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers[:-1]:
            out = layer.forward(out)
        # For the final layer, let's bypass the inline ReLU and use a Sigmoid activation
        last_layer = self.layers[-1]
        last_layer.A_prev = out
        last_layer.Z = np.dot(out, last_layer.W) + last_layer.b
        return 1.0 / (1.0 + np.exp(-np.clip(last_layer.Z, -500, 500)))

    def backward(self, dA_last: np.ndarray):
        # Backward pass starting from the output layer
        # Output layer activation is sigmoid, derivative is: A * (1 - A)
        last_layer = self.layers[-1]
        A_out = 1.0 / (1.0 + np.exp(-np.clip(last_layer.Z, -500, 500)))
        dZ = dA_last * A_out * (1 - A_out)
        m = last_layer.A_prev.shape[0]
        last_layer.dW = np.dot(last_layer.A_prev.T, dZ) / m
        last_layer.db = np.sum(dZ, axis=0, keepdims=True) / m
        dA = np.dot(dZ, last_layer.W.T)
        
        # Backprop through hidden layers
        for layer in reversed(self.layers[:-1]):
            dA = layer.backward(dA)

    def update_params(self, lr: float):
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

# Quick Check
mlp = NumPyMLP([2, 4, 1])  # 2 inputs, 4 hidden, 1 output
X_sample = np.array([[0.1, 0.9], [0.8, 0.2]])
y_pred = mlp.forward(X_sample)
print("Predicted probabilities:\n", y_pred)
```

---

## 3. Advanced: The Universal Approximation Theorem & Representation Depth

### The Universal Approximation Theorem (UAT)
Formulated by George Cybenko (1989) for sigmoid activation functions, and later generalized by Kurt Hornik (1991) to arbitrary non-constant activation functions, the **Universal Approximation Theorem** states:

> Let $g(\cdot)$ be a non-constant, bounded, and continuous activation function. Let $I_n$ denote the $n$-dimensional unit hypercube $[0, 1]^n$. The space of continuous functions on $I_n$ is denoted by $C(I_n)$. Given any $f \in C(I_n)$ and $\epsilon > 0$, there exists a integer $N$, real constants $v_i, b_i$, and vectors $\mathbf{w}_i \in \mathbb{R}^n$ such that:
> $$F(\mathbf{x}) = \sum_{i=1}^N v_i g(\mathbf{w}_i^T \mathbf{x} + b_i)$$
> satisfies $|F(\mathbf{x}) - f(\mathbf{x})| < \epsilon$ for all $\mathbf{x} \in I_n$.

In simple terms: a neural network with a **single hidden layer** containing a finite number of neurons can approximate any continuous function to arbitrary precision.

### The Caveat and the Case for Depth
While the UAT proves that a shallow network *can* represent any function, it does not guarantee:
1. That our optimization algorithms (like gradient descent) can actually find those parameters.
2. That the required number of neurons $N$ will be practical. In the worst case, $N$ grows exponentially with the input dimension $n$ ($O(2^n)$).

#### Why Deep Networks Win
Instead of using one extremely wide layer, modern neural networks use **many deep, narrower layers**.
- **Hierarchical Feature Extraction**: Deep networks compose simple features (lines) into intermediate features (shapes) and then into complex features (objects). 
- **Parameter Efficiency**: Research shows that approximating certain complex functions requires exponentially fewer parameters in a deep network than in a shallow one. A function that requires $O(2^n)$ neurons in a shallow network might only require $O(n^2)$ parameters in a deep network.

---

[← Backpropagation](07-Backpropagation.md) | [Back to Index](../README.md) | [Next: Weight Initialization →](09-Weight-Initialization.md)
