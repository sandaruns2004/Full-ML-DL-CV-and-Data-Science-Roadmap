# 💻 Neural Network from Scratch (NumPy)

---

## 📋 Table of Contents
1. [Introduction](#1-introduction)
2. [Module Architecture](#2-module-architecture)
3. [NumPy Implementation Code](#3-numpy-implementation-code)
4. [Training the Network on a Non-Linear Dataset](#4-training-the-network-on-a-non-linear-dataset)

---

## 1. Introduction
Building a neural network framework from scratch using only matrix algebra forces you to understand every detail of forward flow, gradient propagation, and parameter updates. This module provides a fully modular, object-oriented, type-hinted neural network framework written in pure NumPy.

---

## 2. Module Architecture

The architecture mimics modern deep learning libraries:
- `Layer`: Abstract base class defining `forward` and `backward` interfaces.
- `Dense`: A fully connected layer that manages weights ($\mathbf{W}$) and biases ($\mathbf{b}$).
- `Activation`: Applies element-wise non-linear functions (ReLU, Sigmoid, Softmax).
- `Loss`: Computes loss values and initial backward gradients (Binary Cross Entropy, Categorical Cross Entropy).
- `Optimizer`: Manages parameter updates using computed gradients (SGD, Adam).
- `Model`: Sequences layers together and manages training loops.

---

## 3. NumPy Implementation Code

Save or run this code in a Python 3.10+ environment.

```python
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

class Layer:
    """Base interface for all network layers."""
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Dense(Layer):
    """Fully Connected (Dense) Layer."""
    def __init__(self, in_features: int, out_features: int, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        # He (Kaiming) normal initialization
        self.W: np.ndarray = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b: np.ndarray = np.zeros((1, out_features))
        
        self.inputs: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.dot(inputs, self.W) + self.b

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        assert self.inputs is not None, "Forward pass must be executed before backward pass."
        m = self.inputs.shape[0]
        self.dW = np.dot(self.inputs.T, output_gradient) / m
        self.db = np.sum(output_gradient, axis=0, keepdims=True) / m
        return np.dot(output_gradient, self.W.T)

class ReLU(Layer):
    """ReLU Activation Layer."""
    def __init__(self):
        self.inputs: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        assert self.inputs is not None
        return output_gradient * (self.inputs > 0)

class Sigmoid(Layer):
    """Sigmoid Activation Layer."""
    def __init__(self):
        self.output: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-np.clip(inputs, -500, 500)))
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        assert self.output is not None
        return output_gradient * self.output * (1.0 - self.output)

class BinaryCrossEntropy:
    """Binary Cross Entropy Loss Function."""
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        return - (y_true / y_pred - (1.0 - y_true) / (1.0 - y_pred))

class Adam:
    """Adam Optimizer."""
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: Dict[int, Tuple[np.ndarray, np.ndarray]] = {} # id(layer) -> (mW, mb)
        self.v: Dict[int, Tuple[np.ndarray, np.ndarray]] = {} # id(layer) -> (vW, vb)
        self.t = 0

    def step(self, layers: List[Layer]):
        self.t += 1
        for layer in layers:
            if not isinstance(layer, Dense):
                continue
                
            layer_id = id(layer)
            if layer_id not in self.m:
                self.m[layer_id] = (np.zeros_like(layer.W), np.zeros_like(layer.b))
                self.v[layer_id] = (np.zeros_like(layer.W), np.zeros_like(layer.b))
                
            mW, mb = self.m[layer_id]
            vW, vb = self.v[layer_id]
            
            # Update biased first moment
            mW = self.beta1 * mW + (1.0 - self.beta1) * layer.dW
            mb = self.beta1 * mb + (1.0 - self.beta1) * layer.db
            
            # Update biased second moment
            vW = self.beta2 * vW + (1.0 - self.beta2) * (layer.dW ** 2)
            vb = self.beta2 * vb + (1.0 - self.beta2) * (layer.db ** 2)
            
            # Save moments
            self.m[layer_id] = (mW, mb)
            self.v[layer_id] = (vW, vb)
            
            # Bias correction
            mW_corrected = mW / (1.0 - self.beta1 ** self.t)
            mb_corrected = mb / (1.0 - self.beta1 ** self.t)
            vW_corrected = vW / (1.0 - self.beta2 ** self.t)
            vb_corrected = vb / (1.0 - self.beta2 ** self.t)
            
            # Update parameters
            layer.W -= (self.lr / (np.sqrt(vW_corrected) + self.eps)) * mW_corrected
            layer.b -= (self.lr / (np.sqrt(vb_corrected) + self.eps)) * mb_corrected
```

---

## 4. Training the Network on a Non-Linear Dataset

Let's test this framework on a circular, non-linearly separable dataset.

```python
# Generate circular dataset
np.random.seed(42)
num_samples = 200
r = np.random.uniform(0.0, 1.0, num_samples)
theta = np.random.uniform(0.0, 2 * np.pi, num_samples)
X = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
y = (r > 0.5).astype(float).reshape(-1, 1)

# Build a 2-layer MLP model: 2 inputs -> 8 hidden -> 1 output
layers: List[Layer] = [
    Dense(2, 8, seed=42),
    ReLU(),
    Dense(8, 1, seed=43),
    Sigmoid()
]

loss_fn = BinaryCrossEntropy()
optimizer = Adam(lr=0.05)

# Training loop
epochs = 300
for epoch in range(epochs):
    # Forward propagation
    out = X
    for layer in layers:
        out = layer.forward(out)
        
    # Compute loss
    loss = loss_fn.compute(y, out)
    
    # Backward propagation
    grad = loss_fn.gradient(y, out)
    for layer in reversed(layers):
        grad = layer.backward(grad)
        
    # Optimizer step
    optimizer.step(layers)
    
    if (epoch + 1) % 50 == 0:
        # Calculate training accuracy
        predictions = (out > 0.5).astype(float)
        accuracy = np.mean(predictions == y)
        print(f"Epoch {epoch+1:03d} | Loss: {loss:.4f} | Accuracy: {accuracy * 100:.1f}%")
```

---

[← Previous: Computational Graphs](./12-Computational-Graphs.md) | [Back to Index](./README.md) | [Next: Neural Network PyTorch Implementation →](./14-Neural-Network-PyTorch-Implementation.md)
