# 📉 Gradient Descent

---

## 📋 Table of Contents
1. [Beginner: Descending a Mountain in the Dark](#1-beginner-descending-a-mountain-in-the-dark)
2. [Intermediate: Batch, Mini-batch, and Stochastic Implementations](#2-intermediate-batch-mini-batch-and-stochastic-implementations)
3. [Advanced: Learning Rate Curvature & Optimization Landscapes](#3-advanced-learning-rate-curvature--optimization-landscapes)

---

## 1. Beginner: Descending a Mountain in the Dark

### Simple Explanation
Once we have our loss function (which tells us how wrong we are), we need a systematic way to adjust our weights to reduce the error. **Gradient Descent** is the optimization algorithm used to train neural networks. It calculates the direction of steepest increase in error (the gradient) and takes a step in the opposite direction to minimize the error.

### Real-World Analogy: Foggy Mountain Descent
Imagine you are at the top of a mountain shrouded in thick fog. You want to reach the village at the absolute bottom of the valley, but you cannot see more than a foot in front of you:
- You feel the slope of the ground under your feet.
- To go down, you take a step in the direction where the ground slopes downward.
- If you take steps that are too small, it will take you years to reach the bottom (small learning rate).
- If you take giant, blind leaps, you might fly off the edge of a cliff or jump right over the valley to another peak (large learning rate).

---

## 2. Intermediate: Batch, Mini-batch, and Stochastic Implementations

There are three ways to feed data into Gradient Descent to update weights:

1. **Batch Gradient Descent**: Computes the gradient of the cost function with respect to the parameters for the *entire* dataset.
2. **Stochastic Gradient Descent (SGD)**: Computes the gradient and updates parameters using a *single* training example at a time.
3. **Mini-batch Gradient Descent**: Computes the gradient and updates parameters on small, random subsets of data (typically size 32, 64, 128, or 256). This is the industry standard.

### NumPy Comparison Code

```python
import numpy as np

# Mock Dataset: 1000 samples, 10 features
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.random.randn(1000, 1)

# Parameters
w = np.zeros((10, 1))
b = 0.0
lr = 0.01

def compute_gradients(x_batch, y_batch, w_val, b_val):
    m = x_batch.shape[0]
    predictions = np.dot(x_batch, w_val) + b_val
    error = predictions - y_batch
    dw = np.dot(x_batch.T, error) / m
    db = np.sum(error) / m
    return dw, db

# 1. Batch Gradient Descent (1 Epoch)
dw_batch, db_batch = compute_gradients(X, y, w, b)
w_updated_batch = w - lr * dw_batch
b_updated_batch = b - lr * db_batch
print("Batch GD completed successfully.")

# 2. Stochastic Gradient Descent (1 Epoch)
w_sgd, b_sgd = w.copy(), b
for i in range(len(X)):
    xi = X[i:i+1]
    yi = y[i:i+1]
    dw_i, db_i = compute_gradients(xi, yi, w_sgd, b_sgd)
    w_sgd -= lr * dw_i
    b_sgd -= lr * db_i
print("SGD completed successfully.")

# 3. Mini-batch Gradient Descent (1 Epoch, batch_size=32)
w_mb, b_mb = w.copy(), b
batch_size = 32
for i in range(0, len(X), batch_size):
    x_mini = X[i:i+batch_size]
    y_mini = y[i:i+batch_size]
    dw_mini, db_mini = compute_gradients(x_mini, y_mini, w_mb, b_mb)
    w_mb -= lr * dw_mini
    b_mb -= lr * db_mini
print("Mini-batch GD completed successfully.")
```

---

## 3. Advanced: Learning Rate Curvature & Optimization Landscapes

### Mathematical Formulation
Let $f(\mathbf{w})$ be our objective function. The Taylor expansion around $\mathbf{w}_t$ is:
$$f(\mathbf{w}) \approx f(\mathbf{w}_t) + \nabla f(\mathbf{w}_t)^T (\mathbf{w} - \mathbf{w}_t) + \frac{1}{2} (\mathbf{w} - \mathbf{w}_t)^T \mathbf{H} (\mathbf{w} - \mathbf{w}_t)$$
where $\mathbf{H}$ is the Hessian matrix of second-order derivatives.

Using gradient descent with step size $\eta$:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t)$$

Substituting this back into the Taylor expansion:
$$f(\mathbf{w}_{t+1}) \approx f(\mathbf{w}_t) - \eta \|\nabla f(\mathbf{w}_t)\|^2 + \frac{1}{2} \eta^2 \nabla f(\mathbf{w}_t)^T \mathbf{H} \nabla f(\mathbf{w}_t)$$

For the function value to decrease ($f(\mathbf{w}_{t+1}) < f(\mathbf{w}_t)$), we need:
$$\eta < \frac{2 \nabla f(\mathbf{w}_t)^T \nabla f(\mathbf{w}_t)}{\nabla f(\mathbf{w}_t)^T \mathbf{H} \nabla f(\mathbf{w}_t)}$$

If the maximum eigenvalue of the Hessian $\mathbf{H}$ is $\lambda_{\max}$, then the learning rate must satisfy:
$$\eta < \frac{2}{\lambda_{\max}}$$
If $\eta$ exceeds this bound, the updates will overshoot, and the loss will explode.

### Optimization Challenges

- **Saddle Points**: In high dimensions, local minima are rare. Instead, optimization is dominated by **saddle points**—regions where the gradient is zero, but some directions curve up while others curve down. Simple SGD can get trapped in flat plateaus around saddle points.
- **Pathologies of High Curvature**: In narrow ravines, the gradient is perpendicular to the direction of the valley bottom. The optimization trajectory bounces back and forth between the ravine walls instead of traveling down the valley.

---

[← Loss Functions](05-Loss-Functions.md) | [Back to Index](../README.md) | [Next: Backpropagation →](07-Backpropagation.md)
