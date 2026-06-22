# 🕸️ Computational Graphs & Autograd

---

## 📋 Table of Contents
1. [Beginner: Flowcharts of Math](#1-beginner-flowcharts-of-math)
2. [Intermediate: Building a Mini-Autograd Engine in Python](#2-intermediate-building-a-mini-autograd-engine-in-python)
3. [Advanced: Topological Sorting & Dynamic vs. Static Graphs](#3-advanced-topological-sorting--dynamic-vs-static-graphs)

---

## 1. Beginner: Flowcharts of Math

### Simple Explanation
A **Computational Graph** is a way of expressing mathematical equations as a flowchart or network. 
- **Nodes** represent the variables or values (like numbers or arrays).
- **Edges** represent the operations (like addition, multiplication, or activation functions) that connect these values.

By breaking down a complex neural network into a simple flowchart, we can easily trace the calculations forward to get our prediction, and then trace them backward to see how each input contributed to the final output.

### Real-World Analogy: The Business Recipe
Imagine a small business baking and selling cakes:
- **Flour ($x$)** costs $2.
- **Sugar ($y$)** costs $1$.
- **Baking step ($a$)**: You mix them ($x + y = 3$).
- **Selling step ($b$)**: You package and sell the cake at a markup ($a \times 2 = 6$).
- **Tax step ($L$)**: The state takes a flat tax ($b - 0.5 = 5.5$).

This recipe is a forward pass. If the final profit ($L$) drops, you can look backward through this flowchart to calculate exactly how a change in flour prices ($x$) would impact your final profit ($L$).

---

## 2. Intermediate: Building a Mini-Autograd Engine in Python

Let us build a simple, fully-functional automatic differentiation (autograd) node in pure Python, similar to PyTorch's backend or Andrej Karpathy's `micrograd`.

```python
class Value:
    def __init__(self, data: float, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Build topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Backprop step
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# Test the autograd engine
# Equation: L = (x * w) + b, activated by ReLU
x = Value(2.0)
w = Value(-3.0)
b = Value(1.0)

z = (x * w) + b
L = z.relu()

# Run backpropagation
L.backward()

print(f"Forward output L: {L.data}") # Expected: max(0, 2 * -3 + 1) = max(0, -5) = 0.0
print(f"Gradient of w: {w.grad}")   # Expected: 0.0 (since ReLU output was 0)
```

---

## 3. Advanced: Topological Sorting & Dynamic vs. Static Graphs

### Topological Sorting
To execute backpropagation correctly, we must process the nodes in the reverse order of how they were computed. This is done using **Topological Sorting**.
A topological sort of a directed acyclic graph (DAG) is a linear ordering of its vertices such that for every directed edge $u \to v$, vertex $u$ comes before $v$ in the ordering.
During backpropagation, we:
1. Build a topological sort of the graph starting from the output node (the Loss).
2. Set the output gradient $\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1.0$.
3. Iterate backward through the sorted list of nodes, calling their local gradient function `_backward()`.

### Dynamic vs. Static Graphs
In the deep learning ecosystem, there are two primary methods for building and executing computational graphs:

| Feature | Static Graphs (TensorFlow 1.x, Caffe) | Dynamic Graphs (PyTorch, JAX) |
|---|---|---|
| **Compilation** | **Define-and-Run**: The graph structure is defined once, compiled, and run repeatedly. | **Define-by-Run**: The graph is constructed on-the-fly during the forward pass. |
| **Flexibility** | Difficult to use python loops or conditionals inside the network. | Easy to use native Python control flow (loops, `if` statements). |
| **Optimization** | Highly optimized. The compiler can merge nodes and optimize memory before execution. | Harder to optimize globally on the CPU/GPU without a compilation step (like `torch.compile`). |
| **Debugging** | Difficult. Error stacks do not map to the Python line that caused the bug. | Easy. Standard Python debuggers (`pdb`) work out of the box. |

Modern frameworks have converged: PyTorch uses a dynamic graph by default for development and debugging, but offers a compilation step (`torch.compile`) to generate optimized static graphs for deployment.

---

[← Vanishing and Exploding Gradients](11-Vanishing-And-Exploding-Gradients.md) | [Back to Index](../README.md) | [Next: Neural Network from Scratch (NumPy) →](13-Neural-Network-From-Scratch-Numpy.md)
