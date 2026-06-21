# 🧠 The Perceptron & Biological Analogy

---

## 📋 Table of Contents
1. [Beginner: The Biological Analogy & Rosenblatt's Perceptron](#1-beginner-the-biological-analogy--rosenblatts-perceptron)
2. [Intermediate: Perceptron Learning Algorithm from Scratch](#2-intermediate-perceptron-learning-algorithm-from-scratch)
3. [Advanced: Convergence Theorem & The XOR Limitation Proof](#3-advanced-convergence-theorem--the-xor-limitation-proof)

---

## 1. Beginner: The Biological Analogy & Rosenblatt's Perceptron

### Biological Analogy
Artificial neural networks draw inspiration from biological neurons:
- **Dendrites**: Receive electrical signals from other cells.
- **Soma (Cell Body)**: Processes/sums up the incoming signals.
- **Axon**: Transmits the signal to other neurons if the sum exceeds a threshold.

The **McCulloch-Pitts Neuron (1943)** was the first mathematical simplification of this process. It took binary inputs and summed them, firing a binary output if the sum was above a threshold. However, it did not have a learning mechanism.

### Rosenblatt's Perceptron
In 1957, Frank Rosenblatt introduced the **Perceptron**, adding learnable continuous **weights**.

A single Perceptron multiplies each input by its weight, sums them together with a **bias**, and passes the result through a step function:

$$z = w_1 x_1 + w_2 x_2 + b = \mathbf{w}^T\mathbf{x} + b$$

$$\hat{y} = \text{step}(z) = \begin{cases} 1 & \text{if } z \geq 0 \\ 0 & \text{if } z < 0 \end{cases}$$

### Visual Intuition
The equation $\mathbf{w}^T\mathbf{x} + b = 0$ defines a straight line (in 2D space) or a hyperplane (in higher dimensions). This boundary splits the space into two halves. Points on one side are classified as $1$, and points on the other side as $0$. This is called a **linear decision boundary**.

---

## 2. Intermediate: Perceptron Learning Algorithm from Scratch

When the Perceptron makes a mistake during training, it updates its weights and bias using the **Perceptron Learning Rule**:

$$\mathbf{w}^{(new)} = \mathbf{w}^{(old)} + \eta (y - \hat{y}) \mathbf{x}$$
$$b^{(new)} = b^{(old)} + \eta (y - \hat{y})$$

Where:
- $y$ is the true label (0 or 1).
- $\hat{y}$ is the predicted label (0 or 1).
- $\eta$ is the learning rate ($0 < \eta \le 1$).

If the prediction is correct ($y = \hat{y}$), the weight adjustment is $0$.

### NumPy Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_dim: int, lr: float = 0.1, epochs: int = 100):
        self.lr = lr
        self.epochs = epochs
        self.w = np.zeros(input_dim)
        self.b = 0.0
        
    def predict(self, x: np.ndarray) -> int:
        z = np.dot(x, self.w) + self.b
        return 1 if z >= 0 else 0
        
    def train(self, X: np.ndarray, y: np.ndarray):
        for epoch in range(self.epochs):
            errors = 0
            for xi, yi in zip(X, y):
                prediction = self.predict(xi)
                update = self.lr * (yi - prediction)
                self.w += update * xi
                self.b += update
                if update != 0.0:
                    errors += 1
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}!")
                break

# Logical AND Dataset
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

clf = Perceptron(input_dim=2, lr=0.1, epochs=20)
clf.train(X_and, y_and)
print(f"Trained weights: {clf.w}, bias: {clf.b}")
```

---

## 3. Advanced: Convergence Theorem & The XOR Limitation Proof

### Perceptron Convergence Theorem
If a dataset is **linearly separable** (meaning a hyperplane exists that completely separates the classes), the Perceptron learning algorithm is guaranteed to converge and find a separating hyperplane in a finite number of updates.

#### Proof Sketch:
Let the dataset be separated by an optimal hyperplane with weights $\mathbf{w}^*$ such that $\|\mathbf{w}^*\| = 1$. There exists some margin $\gamma > 0$ such that for all samples $i$:
$$y_i (\mathbf{w}^{*T} \mathbf{x}_i) \ge \gamma$$ (where targets are rewritten as $y_i \in \{-1, +1\}$).
Assume the maximum radius of the inputs is $R = \max_i \|\mathbf{x}_i\|$. 
It can be shown that the number of updates $k$ is bounded by:
$$k \le \frac{R^2}{\gamma^2}$$
Thus, the algorithm must terminate in a finite number of steps.

### Proof of XOR Limitation (Minsky & Papert, 1969)
The XOR problem is historically significant as it caused the first "AI Winter". A single perceptron cannot solve the XOR problem because the XOR data distribution is not linearly separable.

#### Mathematical Proof:
Let the inputs be $x_1, x_2 \in \{0, 1\}$. We want a set of weights $w_1, w_2$ and bias $b$ satisfying:

1. For $(0,0) \to 0$:  $b < 0$
2. For $(0,1) \to 1$:  $w_2 + b \ge 0 \implies w_2 \ge -b$
3. For $(1,0) \to 1$:  $w_1 + b \ge 0 \implies w_1 \ge -b$
4. For $(1,1) \to 0$:  $w_1 + w_2 + b < 0$

Summing equations (2) and (3) gives:
$$w_1 + w_2 \ge -2b$$

If we substitute this into (4):
$$-2b + b \le w_1 + w_2 + b < 0 \implies -b < 0 \implies b > 0$$

However, this contradicts equation (1), which states that $b < 0$. Thus, no such weights and bias can exist. To solve XOR, we must stack multiple linear perceptrons with non-linear activation functions between them.

---

[← Previous: Introduction To Neural Networks](./01-Introduction-To-Neural-Networks.md) | [Back to Index](./README.md) | [Next: Activation Functions →](./03-Activation-Functions.md)
