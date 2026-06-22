# 🔄 Backpropagation

---

## 📋 Table of Contents
1. [Beginner: The Chain of Feedback](#1-beginner-the-chain-of-feedback)
2. [Intermediate: Step-by-Step Derivation and Chain Rule](#2-intermediate-step-by-step-derivation-and-chain-rule)
3. [Advanced: Matrix Calculus, Jacobians, and Numerical Stability](#3-advanced-matrix-calculus-jacobians-and-numerical-stability)

---

## 1. Beginner: The Chain of Feedback

### Simple Explanation
**Backpropagation** (short for "backward propagation of errors") is the algorithm that calculates how much each weight and bias in a neural network contributed to the final output error. It is essentially the calculus **Chain Rule** applied systematically to the network's layers, working backward from the output layer to the input layer.

### Real-World Analogy: The Whispering Game (with a Boss)
Imagine a chain of employees at a company:
- The CEO (Input) gives a general instruction.
- The Manager (Hidden Layer 1) interprets it and tells the Team Lead (Hidden Layer 2).
- The Team Lead tells the Intern (Output Layer) who executes the task.
- The Client evaluates the final product and finds a mistake (Loss/Error).
- To fix the mistake, the Client gives feedback to the Intern. The Intern explains, "I only did what the Team Lead told me." So, the feedback flows **backward**: the Intern passes the feedback to the Team Lead, the Team Lead passes it to the Manager, and the Manager passes it to the CEO. Each person learns exactly how their specific interpretation contributed to the final mistake and adjusts their work.

---

## 2. Intermediate: Step-by-Step Derivation and Chain Rule

In a neural network, layers are connected sequentially. To update a weight $W^{[l]}$ at layer $l$, we need to find the partial derivative of the loss $L$ with respect to that weight: $\frac{\partial L}{\partial W^{[l]}}$.

By applying the chain rule, we decompose this derivative:

$$\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial A^{[l]}} \cdot \frac{\partial A^{[l]}}{\partial Z^{[l]}} \cdot \frac{\partial Z^{[l]}}{\partial W^{[l]}}$$

Let us define the intermediate error term (commonly called $\delta^{[l]}$ or $dZ^{[l]}$) as:
$$dZ^{[l]} = \frac{\partial L}{\partial Z^{[l]}}$$

### The 4 Fundamental Equations of Backpropagation
For a batch of $m$ examples, the matrix equations are:

1. **Error at the Output Layer ($L$)**:
   $$d\mathbf{Z}^{[L]} = \mathbf{A}^{[L]} - \mathbf{Y}$$
   *(Note: This holds true for the combination of Sigmoid + Binary Cross-Entropy, or Softmax + Categorical Cross-Entropy).*

2. **Error at Hidden Layer ($l$)**:
   $$d\mathbf{Z}^{[l]} = \left( d\mathbf{Z}^{[l+1]} (\mathbf{W}^{[l+1]})^T \right) \odot g'^{[l]}(\mathbf{Z}^{[l]})$$
   *(where $\odot$ denotes the element-wise/Hadamard product, and $g'^{[l]}$ is the derivative of the activation function).*

3. **Gradient of Weights ($l$)**:
   $$d\mathbf{W}^{[l]} = \frac{1}{m} (\mathbf{A}^{[l-1]})^T d\mathbf{Z}^{[l]}$$

4. **Gradient of Biases ($l$)**:
   $$d\mathbf{b}^{[l]} = \frac{1}{m} \sum_{\text{rows}} d\mathbf{Z}^{[l]}$$

### NumPy Walkthrough: A Single Backprop Step
```python
import numpy as np

# Mock inputs and outputs
np.random.seed(42)
A_prev = np.random.randn(4, 3) # Batch size 4, input features 3
W = np.random.randn(3, 2)      # 3 inputs -> 2 outputs
b = np.random.randn(1, 2)
Z = np.dot(A_prev, W) + b
A = 1 / (1 + np.exp(-Z))        # Sigmoid activation
Y = np.array([[0, 1], [1, 0], [1, 1], [0, 0]]) # True labels

# Upstream gradient of loss with respect to activation (BCE loss derivative)
dA = - (Y / A - (1 - Y) / (1 - A)) / A.shape[0]

# Local gradient of Sigmoid: A * (1 - A)
dZ = dA * A * (1 - A)

# Gradients for parameters
dW = np.dot(A_prev.T, dZ)
db = np.sum(dZ, axis=0, keepdims=True)
dA_prev = np.dot(dZ, W.T)

print("dW Shape:", dW.shape)
print("db Shape:", db.shape)
print("dA_prev Shape:", dA_prev.shape)
```

---

## 3. Advanced: Matrix Calculus, Jacobians, and Numerical Stability

### Jacobian Vector Products (JVP)
When calculating derivatives with respect to vectors, the intermediate derivatives are matrices (Jacobians). 
For instance, let $\mathbf{a} \in \mathbb{R}^n$ and $\mathbf{z} \in \mathbb{R}^n$. The derivative $\frac{\partial \mathbf{a}}{\partial \mathbf{z}}$ is an $n \times n$ Jacobian matrix $\mathbf{J}$ where $J_{ij} = \frac{\partial a_i}{\partial z_j}$.

During backpropagation, we do not compute and store this entire matrix because it would take too much memory. Instead, we compute the **Jacobian-Vector Product (JVP)** directly.
If we have an incoming upstream gradient vector $\mathbf{v} = \frac{\partial L}{\partial \mathbf{a}}$, the downstream gradient vector is:
$$\mathbf{v}^T \mathbf{J} = \left[ \sum_{i} v_i \frac{\partial a_i}{\partial z_1}, \dots, \sum_{i} v_i \frac{\partial a_i}{\partial z_n} \right]$$

For element-wise functions (like Sigmoid or ReLU), the Jacobian is a diagonal matrix:
$$J_{ij} = 0 \quad \text{for } i \neq j$$
Thus, the Jacobian-Vector Product collapses to an element-wise product:
$$\mathbf{v} \odot g'(\mathbf{z})$$

### Numerical Stability: Gradient Clipping
When gradients flow backward through many layers, they are multiplied repeatedly. If the eigenvalues of the weight matrices are greater than 1, the gradient norms can grow exponentially, leading to **exploding gradients** (value becomes `NaN` or `inf`).

To prevent this, we use **Gradient Clipping**. There are two main strategies:
1. **Clip by Value**: Force each gradient element to stay within a range $[-c, c]$:
   $$g \leftarrow \max(-c, \min(c, g))$$
2. **Clip by Norm**: Scale the entire gradient vector if its L2 norm exceeds a threshold $c$:
   $$g \leftarrow g \cdot \frac{c}{\max(c, \|g\|_2)}$$
   Clipping by norm preserves the direction of the gradient vector in multi-dimensional space, which helps keep training stable.

---

[← Gradient Descent](06-Gradient-Descent.md) | [Back to Index](../README.md) | [Next: Multi-Layer Perceptron (MLP) →](08-Multi-Layer-Perceptron.md)
