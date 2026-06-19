# 📐 Mathematical Foundations for Machine Learning

> **Prerequisites**: Basic algebra | **Difficulty**: ⭐⭐☆☆☆ Elementary

---

## 📋 Table of Contents

1. [Why Math Matters for ML](#1-why-math-matters-for-ml)
2. [Linear Algebra](#2-linear-algebra)
3. [Calculus](#3-calculus)
4. [Probability & Statistics](#4-probability--statistics)
5. [Optimization Theory](#5-optimization-theory)
6. [Information Theory](#6-information-theory)
7. [Project Ideas](#7-project-ideas)
8. [What's Next](#8-whats-next)

---

## 1. Why Math Matters for ML

Machine Learning is **applied mathematics**. Every algorithm is a mathematical procedure:

| ML Concept | Mathematical Foundation |
|------------|------------------------|
| Data representation | Vectors, matrices, tensors |
| Linear Regression | Matrix multiplication, least squares |
| Gradient Descent | Partial derivatives, chain rule |
| PCA | Eigenvalues, eigenvectors |
| Neural Networks | Chain rule, matrix calculus |
| Bayesian methods | Probability, Bayes' theorem |
| Loss functions | Optimization theory |
| Decision Trees | Information theory (entropy) |

---

## 2. Linear Algebra

Linear algebra is the **language of data** in ML. Every dataset is a matrix, every data point is a vector.

### 2.1 Scalars, Vectors, and Matrices

**Scalar**: A single number.

$$a = 5, \quad b = 3.14, \quad c = -2.7$$

**Vector**: An ordered list of numbers (1D array). A vector in $\mathbb{R}^n$ has $n$ components.

$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

In ML, a **feature vector** represents one data point:
$$\mathbf{x} = \begin{bmatrix} \text{age} \\ \text{income} \\ \text{credit\_score} \end{bmatrix} = \begin{bmatrix} 25 \\ 50000 \\ 720 \end{bmatrix}$$

**Matrix**: A 2D array of numbers. An $m \times n$ matrix has $m$ rows and $n$ columns.

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

In ML, the **design matrix** $\mathbf{X}$ stores the entire dataset:
- Rows = data points (samples)
- Columns = features

$$\mathbf{X} = \begin{bmatrix} — \mathbf{x}_1^T — \\ — \mathbf{x}_2^T — \\ \vdots \\ — \mathbf{x}_m^T — \end{bmatrix} \in \mathbb{R}^{m \times n}$$

**Tensor**: An $n$-dimensional array. A 3D tensor could represent a color image (height × width × channels).

```python
import numpy as np

# Scalar
a = 5

# Vector (feature vector for a house)
x = np.array([1500, 3, 2, 1985])  # [sqft, bedrooms, bathrooms, year]

# Matrix (dataset of 4 houses)
X = np.array([
    [1500, 3, 2, 1985],
    [2000, 4, 3, 2000],
    [1200, 2, 1, 1970],
    [1800, 3, 2, 1995]
])
print(f"Dataset shape: {X.shape}")  # (4, 4) — 4 samples, 4 features

# Tensor (color image: 28x28 pixels, 3 color channels)
image = np.random.rand(28, 28, 3)
print(f"Image tensor shape: {image.shape}")  # (28, 28, 3)

# Batch of images (32 images for training)
batch = np.random.rand(32, 28, 28, 3)
print(f"Batch tensor shape: {batch.shape}")  # (32, 28, 28, 3)
```

### 2.2 Vector Operations

**Vector Addition**: Element-wise addition.

$$\mathbf{a} + \mathbf{b} = \begin{bmatrix} a_1 + b_1 \\ a_2 + b_2 \\ \vdots \\ a_n + b_n \end{bmatrix}$$

**Scalar Multiplication**: Multiply every element by a scalar.

$$c \cdot \mathbf{a} = \begin{bmatrix} c \cdot a_1 \\ c \cdot a_2 \\ \vdots \\ c \cdot a_n \end{bmatrix}$$

**Dot Product**: Measures similarity between vectors. This is THE most important operation in ML.

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n$$

**Geometric interpretation**: $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta$

- If $\theta = 0°$: vectors point same direction, dot product is maximized
- If $\theta = 90°$: vectors are orthogonal (perpendicular), dot product = 0
- If $\theta = 180°$: vectors point opposite directions, dot product is minimized

**In ML**: A linear model computes $\hat{y} = \mathbf{w} \cdot \mathbf{x} + b$ — the dot product of weights and features!

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
print(f"Dot product: {dot}")

# Magnitude (L2 norm)
magnitude_a = np.linalg.norm(a)  # sqrt(1² + 2² + 3²) = 3.742
print(f"|a| = {magnitude_a:.3f}")

# Angle between vectors
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.arccos(cos_theta)
print(f"Angle: {np.degrees(theta):.2f}°")

# Cosine similarity (used in NLP!)
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
print(f"Cosine similarity: {sim[0][0]:.4f}")
```

### 2.3 Vector Norms (Measuring Size)

Norms measure the "size" or "length" of a vector.

**L1 Norm** (Manhattan distance): $\|\mathbf{x}\|_1 = \sum_{i=1}^{n} |x_i|$

**L2 Norm** (Euclidean distance): $\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^{n} x_i^2}$

**L∞ Norm** (Max norm): $\|\mathbf{x}\|_\infty = \max_i |x_i|$

**Lp Norm** (General): $\|\mathbf{x}\|_p = \left(\sum_{i=1}^{n} |x_i|^p\right)^{1/p}$

**In ML**:
- L1 norm → Lasso regularization (produces sparse models)
- L2 norm → Ridge regularization (prevents large weights)
- Euclidean distance → KNN, K-Means clustering

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([3, -4, 1, -2])

l1 = np.linalg.norm(x, 1)    # |3| + |-4| + |1| + |-2| = 10
l2 = np.linalg.norm(x, 2)    # sqrt(9 + 16 + 1 + 4) = 5.477
linf = np.linalg.norm(x, np.inf)  # max(3, 4, 1, 2) = 4

print(f"L1 norm:  {l1}")
print(f"L2 norm:  {l2:.3f}")
print(f"L∞ norm: {linf}")

# Visualize unit balls for different norms
theta = np.linspace(0, 2*np.pi, 1000)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, p, name in zip(axes, [1, 2, np.inf], ['L1 (Diamond)', 'L2 (Circle)', 'L∞ (Square)']):
    if p == 1:
        x_vals = np.cos(theta)
        y_vals = np.sin(theta)
        r = np.abs(x_vals) + np.abs(y_vals)
        x_vals /= r
        y_vals /= r
    elif p == 2:
        x_vals = np.cos(theta)
        y_vals = np.sin(theta)
    else:
        t = np.linspace(-1, 1, 100)
        x_vals = np.concatenate([t, np.ones(100), t[::-1], -np.ones(100)])
        y_vals = np.concatenate([-np.ones(100), t, np.ones(100), t[::-1]])
    
    ax.plot(x_vals, y_vals, linewidth=2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{name} Unit Ball', fontsize=12)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

plt.suptitle('Unit Balls for Different Norms', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('norm_unit_balls.png', dpi=150)
plt.show()
```

### 2.4 Matrix Operations

**Matrix Multiplication**: If $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$, then $\mathbf{C} = \mathbf{AB} \in \mathbb{R}^{m \times p}$.

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

Each entry is the **dot product** of the $i$-th row of $\mathbf{A}$ and the $j$-th column of $\mathbf{B}$.

**Why this matters in ML**: A neural network layer computes $\mathbf{y} = \mathbf{Wx} + \mathbf{b}$ — matrix-vector multiplication!

**Transpose**: Flip rows and columns. $(\mathbf{A}^T)_{ij} = A_{ji}$

**Inverse**: $\mathbf{A}^{-1}$ satisfies $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$. Used to solve linear systems: $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$

**Determinant**: $\det(\mathbf{A})$ — measures how much $\mathbf{A}$ scales volumes. If $\det(\mathbf{A}) = 0$, the matrix is **singular** (non-invertible).

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B
print(f"A @ B =\n{C}")
# [[1*5+2*7, 1*6+2*8],   [[19, 22],
#  [3*5+4*7, 3*6+4*8]] =  [43, 50]]

# Transpose
print(f"A^T =\n{A.T}")

# Determinant
det_A = np.linalg.det(A)
print(f"det(A) = {det_A:.1f}")  # 1*4 - 2*3 = -2

# Inverse
A_inv = np.linalg.inv(A)
print(f"A^(-1) =\n{A_inv}")
print(f"A @ A^(-1) =\n{(A @ A_inv).round(10)}")  # Identity matrix

# Solving Ax = b (THE fundamental operation in linear regression)
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(f"Solution: x = {x}")  # [1, 2]
print(f"Verification: A @ x = {A @ x}")  # [5, 11]
```

### 2.5 Eigenvalues and Eigenvectors

An **eigenvector** of matrix $\mathbf{A}$ is a vector $\mathbf{v}$ such that multiplying by $\mathbf{A}$ only scales it:

$$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$$

where $\lambda$ is the **eigenvalue** (the scaling factor).

**Geometric intuition**: Eigenvectors define the "natural axes" of a transformation. The eigenvalue tells you how much stretching/compression happens along that axis.

**In ML**:
- **PCA**: Uses eigenvectors to find the directions of maximum variance in data
- **Google's PageRank**: Finds the dominant eigenvector of a link matrix
- **Spectral Clustering**: Uses eigenvectors of similarity matrices

**Finding eigenvalues**: Solve $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$ (the characteristic equation).

For $\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$:

$$\det\begin{bmatrix} 2-\lambda & 1 \\ 1 & 2-\lambda \end{bmatrix} = (2-\lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 3)(\lambda - 1) = 0$$

So $\lambda_1 = 3$ and $\lambda_2 = 1$.

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2, 1], [1, 2]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")     # [3, 1]
print(f"Eigenvectors:\n{eigenvectors}")  # Each column is an eigenvector

# Verify: A @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    Av = A @ v
    lam_v = lam * v
    print(f"\nEigenvector {i+1}: v = {v}")
    print(f"  A @ v   = {Av}")
    print(f"  λ * v   = {lam_v}")
    print(f"  Equal?  {np.allclose(Av, lam_v)}")

# Visualize eigenvectors as directions in 2D
fig, ax = plt.subplots(figsize=(8, 8))

# Plot original and transformed vectors
angles = np.linspace(0, 2*np.pi, 100)
circle = np.array([np.cos(angles), np.sin(angles)])
transformed = A @ circle

ax.plot(circle[0], circle[1], 'b-', alpha=0.3, label='Unit circle')
ax.plot(transformed[0], transformed[1], 'r-', alpha=0.3, label='Transformed')

# Plot eigenvectors
for i in range(2):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    ax.arrow(0, 0, v[0]*lam, v[1]*lam, head_width=0.1, head_length=0.1,
             fc=f'C{i+2}', ec=f'C{i+2}', linewidth=2)
    ax.annotate(f'λ={lam:.0f}, v={v.round(2)}', xy=(v[0]*lam, v[1]*lam),
                fontsize=10, ha='center')

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend()
ax.set_title('Eigenvectors and Eigenvalues Visualization')
plt.tight_layout()
plt.savefig('eigenvectors.png', dpi=150)
plt.show()
```

### 2.6 Singular Value Decomposition (SVD)

SVD factors **any** matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ into three matrices:

$$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: Left singular vectors (orthonormal)
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$: Diagonal matrix of singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $\mathbf{V}^T \in \mathbb{R}^{n \times n}$: Right singular vectors (orthonormal)

**In ML**: SVD is used in:
- **Dimensionality reduction**: Keep only top-$k$ singular values
- **Recommendation systems**: Matrix factorization (Netflix Prize!)
- **Data compression**: Low-rank approximation
- **Computing pseudo-inverse**: For underdetermined systems

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U, S, Vt = np.linalg.svd(A)
print(f"U (left singular vectors):\n{U.round(3)}")
print(f"S (singular values): {S.round(3)}")
print(f"Vt (right singular vectors):\n{Vt.round(3)}")

# Reconstruct A from SVD
Sigma = np.zeros_like(A, dtype=float)
np.fill_diagonal(Sigma, S)
A_reconstructed = U @ Sigma @ Vt
print(f"\nReconstructed A:\n{A_reconstructed.round(10)}")

# Low-rank approximation (keep only k=2 singular values)
k = 2
A_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"\nRank-2 approximation:\n{A_approx.round(3)}")

# Frobenius norm of error
error = np.linalg.norm(A - A_approx, 'fro')
print(f"Approximation error: {error:.6f}")
```

---

## 3. Calculus

Calculus is the mathematics of **change** — and ML is all about minimizing change (loss) through gradient descent.

### 3.1 Derivatives (Single Variable)

The derivative of $f(x)$ measures the **instantaneous rate of change**:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

**Key derivative rules**:

| Rule | Formula | Example |
|------|---------|---------|
| Power rule | $\frac{d}{dx}x^n = nx^{n-1}$ | $\frac{d}{dx}x^3 = 3x^2$ |
| Constant | $\frac{d}{dx}c = 0$ | $\frac{d}{dx}5 = 0$ |
| Sum | $\frac{d}{dx}[f + g] = f' + g'$ | |
| Product | $\frac{d}{dx}[fg] = f'g + fg'$ | |
| Chain rule | $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$ | |
| Exponential | $\frac{d}{dx}e^x = e^x$ | |
| Logarithm | $\frac{d}{dx}\ln(x) = \frac{1}{x}$ | |

**The Chain Rule** is the most important for ML (it's how backpropagation works!):

$$\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$$

**Example**: Let $f(x) = (3x^2 + 1)^5$. Let $u = 3x^2 + 1$, so $f = u^5$.

$$\frac{df}{dx} = \frac{df}{du} \cdot \frac{du}{dx} = 5u^4 \cdot 6x = 30x(3x^2 + 1)^4$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Numerical derivative
def numerical_derivative(f, x, h=1e-7):
    """Compute derivative using central differences (more accurate)."""
    return (f(x + h) - f(x - h)) / (2 * h)

# Example: f(x) = x³ - 3x² + 2x
f = lambda x: x**3 - 3*x**2 + 2*x
f_prime = lambda x: 3*x**2 - 6*x + 2  # Analytical derivative

x = np.linspace(-1, 4, 200)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, f(x), 'b-', linewidth=2, label='f(x) = x³ - 3x² + 2x')
ax.plot(x, f_prime(x), 'r--', linewidth=2, label="f'(x) = 3x² - 6x + 2")

# Mark critical points (where f'(x) = 0)
# 3x² - 6x + 2 = 0 → x = (6 ± √12) / 6
x_crit = [(6 - np.sqrt(12))/6, (6 + np.sqrt(12))/6]
for xc in x_crit:
    ax.plot(xc, f(xc), 'go', markersize=10)
    ax.annotate(f'Critical: ({xc:.2f}, {f(xc):.2f})', xy=(xc, f(xc)),
                xytext=(xc+0.3, f(xc)+0.5), fontsize=10,
                arrowprops=dict(arrowstyle='->', color='green'))

ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_title('Function and its Derivative', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
plt.tight_layout()
plt.savefig('derivatives.png', dpi=150)
plt.show()

# Verify numerical vs analytical derivative
x_test = 2.0
numerical = numerical_derivative(f, x_test)
analytical = f_prime(x_test)
print(f"At x = {x_test}:")
print(f"  Numerical derivative:  {numerical:.10f}")
print(f"  Analytical derivative: {analytical:.10f}")
print(f"  Error: {abs(numerical - analytical):.2e}")
```

### 3.2 Partial Derivatives and Gradients

For functions of multiple variables, the **partial derivative** measures change along one variable while holding others constant:

$$\frac{\partial f}{\partial x_i}$$

The **gradient** is the vector of ALL partial derivatives:

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**The gradient points in the direction of steepest ascent.** To minimize a function (like a loss function), we move in the **opposite direction** — this is **gradient descent**!

**Example**: $f(x, y) = x^2 + 2y^2$

$$\nabla f = \begin{bmatrix} 2x \\ 4y \end{bmatrix}$$

At point $(1, 1)$: $\nabla f = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$

```python
import numpy as np
import matplotlib.pyplot as plt

# Function: f(x, y) = x² + 2y²
def f(x, y):
    return x**2 + 2*y**2

# Gradient: ∇f = [2x, 4y]
def grad_f(x, y):
    return np.array([2*x, 4*y])

# Visualize the function and gradient field
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Contour plot with gradient arrows
cs = axes[0].contour(X, Y, Z, levels=15, cmap='viridis')
axes[0].clabel(cs, inline=True, fontsize=8)

# Gradient field (subsample for clarity)
xx = np.linspace(-3, 3, 12)
yy = np.linspace(-3, 3, 12)
XX, YY = np.meshgrid(xx, yy)
GX = 2 * XX
GY = 4 * YY
axes[0].quiver(XX, YY, -GX, -GY, color='red', alpha=0.6, scale=50)
axes[0].set_title('Contour Plot + Negative Gradient (Descent Direction)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].plot(0, 0, 'r*', markersize=15, label='Minimum')
axes[0].legend()

# 3D surface plot
ax3d = fig.add_subplot(122, projection='3d')
ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax3d.set_xlabel('x')
ax3d.set_ylabel('y')
ax3d.set_zlabel('f(x, y)')
ax3d.set_title('3D Surface: f(x, y) = x² + 2y²')

plt.tight_layout()
plt.savefig('gradient_visualization.png', dpi=150)
plt.show()
```

### 3.3 The Chain Rule (Backpropagation Foundation)

The chain rule is the **mathematical backbone of neural networks**. Given a composition of functions:

$$y = f(g(h(x)))$$

The derivative is:

$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

**In a neural network** with layers $z_1 = W_1 x$, $a_1 = \sigma(z_1)$, $z_2 = W_2 a_1$, $L = \text{loss}(z_2, y)$:

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial z_2} \cdot \frac{\partial z_2}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$$

This is computed **backwards** through the network — hence "backpropagation."

```python
import numpy as np

# Simulate a simple 2-layer neural network forward and backward pass
# Input: x = 2
# Layer 1: z1 = 3x + 1, a1 = relu(z1)
# Layer 2: z2 = 2*a1 - 1, output = z2
# Loss: L = (z2 - target)^2

x = 2.0
target = 10.0
w1, b1 = 3.0, 1.0   # Layer 1 weights
w2, b2 = 2.0, -1.0   # Layer 2 weights

# Forward pass
z1 = w1 * x + b1          # z1 = 3*2 + 1 = 7
a1 = max(0, z1)            # relu(7) = 7
z2 = w2 * a1 + b2          # z2 = 2*7 - 1 = 13
L = (z2 - target) ** 2     # L = (13 - 10)^2 = 9

print("=== Forward Pass ===")
print(f"z1 = {z1}, a1 = {a1}, z2 = {z2}, Loss = {L}")

# Backward pass (chain rule!)
dL_dz2 = 2 * (z2 - target)        # dL/dz2 = 2(13-10) = 6
dz2_da1 = w2                        # dz2/da1 = 2
da1_dz1 = 1.0 if z1 > 0 else 0.0  # relu derivative = 1 if z1 > 0
dz1_dw1 = x                         # dz1/dw1 = x = 2
dz1_dx = w1                         # dz1/dx = w1 = 3

# Chain rule: dL/dw1 = dL/dz2 * dz2/da1 * da1/dz1 * dz1/dw1
dL_dw1 = dL_dz2 * dz2_da1 * da1_dz1 * dz1_dw1
dL_dw2 = dL_dz2 * a1
dL_db2 = dL_dz2

print("\n=== Backward Pass (Chain Rule) ===")
print(f"dL/dz2 = {dL_dz2}")
print(f"dL/dw2 = {dL_dw2}")
print(f"dL/dw1 = {dL_dw1}")

# Gradient descent update
lr = 0.01
w1_new = w1 - lr * dL_dw1
w2_new = w2 - lr * dL_dw2
print(f"\n=== Update (lr={lr}) ===")
print(f"w1: {w1} → {w1_new}")
print(f"w2: {w2} → {w2_new}")
```

### 3.4 Important ML Functions and Their Derivatives

| Function | $f(x)$ | $f'(x)$ | ML Use |
|----------|---------|----------|--------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $f(x)(1-f(x))$ | Logistic regression, binary output |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - f(x)^2$ | RNN activations |
| ReLU | $\max(0, x)$ | $\begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ | Deep learning activations |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | Complex (Jacobian) | Multi-class classification |
| Cross-entropy | $-\sum y_i \log(\hat{y}_i)$ | $\hat{y}_i - y_i$ (with softmax) | Classification loss |
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | $\frac{2}{n}(\hat{y}_i - y_i)$ | Regression loss |

```python
import numpy as np
import matplotlib.pyplot as plt

# Define activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# Plot all activation functions
x = np.linspace(-5, 5, 200)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

functions = [
    ('Sigmoid', sigmoid, sigmoid_derivative),
    ('Tanh', tanh, tanh_derivative),
    ('ReLU', relu, relu_derivative),
    ('Leaky ReLU', leaky_relu, leaky_relu_derivative)
]

for i, (name, func, deriv) in enumerate(functions):
    # Function
    axes[0, i].plot(x, func(x), 'b-', linewidth=2)
    axes[0, i].set_title(f'{name}', fontsize=13, fontweight='bold')
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].axhline(y=0, color='k', linewidth=0.5)
    axes[0, i].axvline(x=0, color='k', linewidth=0.5)
    axes[0, i].set_ylabel('f(x)')
    
    # Derivative
    axes[1, i].plot(x, deriv(x), 'r-', linewidth=2)
    axes[1, i].set_title(f'{name} Derivative', fontsize=13)
    axes[1, i].grid(True, alpha=0.3)
    axes[1, i].axhline(y=0, color='k', linewidth=0.5)
    axes[1, i].axvline(x=0, color='k', linewidth=0.5)
    axes[1, i].set_ylabel("f'(x)")
    axes[1, i].set_xlabel('x')

plt.suptitle('Activation Functions and Their Derivatives', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('activation_functions.png', dpi=150)
plt.show()
```

---

## 4. Probability & Statistics

### 4.1 Probability Basics

**Sample Space** ($\Omega$): The set of all possible outcomes.

**Event**: A subset of the sample space.

**Probability axioms**:
1. $P(A) \geq 0$ for any event $A$
2. $P(\Omega) = 1$ (something must happen)
3. If $A$ and $B$ are mutually exclusive: $P(A \cup B) = P(A) + P(B)$

**Conditional Probability**: The probability of $A$ given $B$ has occurred:

$$P(A|B) = \frac{P(A \cap B)}{P(B)}$$

**Independence**: $A$ and $B$ are independent if $P(A \cap B) = P(A) \cdot P(B)$, which means $P(A|B) = P(A)$.

### 4.2 Bayes' Theorem

Bayes' theorem lets us **update our beliefs** given new evidence. This is the foundation of Bayesian ML.

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

| Term | Name | Interpretation |
|------|------|----------------|
| $P(A\|B)$ | Posterior | Updated belief after seeing evidence |
| $P(B\|A)$ | Likelihood | How likely is the evidence given our hypothesis |
| $P(A)$ | Prior | Our initial belief before seeing evidence |
| $P(B)$ | Evidence | How likely is the evidence overall |

**Example: Medical Test**

A disease affects 1% of the population. A test has:
- 95% sensitivity (true positive rate): $P(\text{positive} | \text{disease}) = 0.95$
- 90% specificity (true negative rate): $P(\text{negative} | \text{no disease}) = 0.90$

If you test positive, what's the probability you actually have the disease?

$$P(\text{disease} | \text{positive}) = \frac{P(\text{positive} | \text{disease}) \cdot P(\text{disease})}{P(\text{positive})}$$

$$P(\text{positive}) = P(\text{pos} | \text{disease}) \cdot P(\text{disease}) + P(\text{pos} | \text{no disease}) \cdot P(\text{no disease})$$

$$= 0.95 \times 0.01 + 0.10 \times 0.99 = 0.0095 + 0.099 = 0.1085$$

$$P(\text{disease} | \text{positive}) = \frac{0.95 \times 0.01}{0.1085} \approx 0.0876 \approx 8.8\%$$

**Surprising!** Even with a positive test, there's only an 8.8% chance you have the disease!

```python
import numpy as np

# Bayes' theorem implementation
def bayes_theorem(prior, likelihood, evidence):
    """P(A|B) = P(B|A) * P(A) / P(B)"""
    return (likelihood * prior) / evidence

# Medical test example
p_disease = 0.01
p_no_disease = 0.99
p_pos_given_disease = 0.95
p_pos_given_no_disease = 0.10

# Total probability of positive test
p_positive = p_pos_given_disease * p_disease + p_pos_given_no_disease * p_no_disease

# Bayes' theorem
p_disease_given_pos = bayes_theorem(p_disease, p_pos_given_disease, p_positive)
print(f"P(disease | positive test) = {p_disease_given_pos:.4f} = {p_disease_given_pos*100:.1f}%")
print(f"Despite the test being 95% accurate, only {p_disease_given_pos*100:.1f}% chance of disease!")

# Multiple tests (Bayesian updating)
p = p_disease  # Start with prior
for test in range(1, 6):
    p_pos = p_pos_given_disease * p + p_pos_given_no_disease * (1 - p)
    p = bayes_theorem(p, p_pos_given_disease, p_pos)
    print(f"After {test} positive test(s): P(disease) = {p:.4f} = {p*100:.1f}%")
```

### 4.3 Probability Distributions

#### Discrete Distributions

**Bernoulli**: Single trial with probability $p$ of success.
$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

**Binomial**: Number of successes in $n$ independent Bernoulli trials.
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

**Poisson**: Number of events in a fixed interval. $P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$

#### Continuous Distributions

**Uniform**: Equal probability over $[a, b]$.
$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

**Normal (Gaussian)**: The most important distribution in ML.
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- $\mu$ = mean (center)
- $\sigma$ = standard deviation (spread)
- $\sigma^2$ = variance

**Why Gaussian matters**: Central Limit Theorem states that the sum of many independent random variables tends toward a Gaussian, regardless of their individual distributions!

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Bernoulli
p = 0.7
x = [0, 1]
probs = [1-p, p]
axes[0, 0].bar(x, probs, color=['#FF6384', '#36A2EB'], width=0.3)
axes[0, 0].set_title(f'Bernoulli (p={p})')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_ylabel('P(X=x)')

# 2. Binomial
n, p = 20, 0.5
x = np.arange(0, n+1)
probs = stats.binom.pmf(x, n, p)
axes[0, 1].bar(x, probs, color='#36A2EB', edgecolor='white')
axes[0, 1].set_title(f'Binomial (n={n}, p={p})')
axes[0, 1].set_xlabel('k (successes)')

# 3. Poisson
for lam in [1, 4, 10]:
    x = np.arange(0, 20)
    probs = stats.poisson.pmf(x, lam)
    axes[0, 2].plot(x, probs, 'o-', label=f'λ={lam}')
axes[0, 2].set_title('Poisson Distribution')
axes[0, 2].legend()

# 4. Normal (different means and variances)
x = np.linspace(-8, 8, 200)
for mu, sigma in [(0, 1), (0, 2), (2, 1), (-2, 0.5)]:
    axes[1, 0].plot(x, stats.norm.pdf(x, mu, sigma), 
                    label=f'μ={mu}, σ={sigma}', linewidth=2)
axes[1, 0].set_title('Normal Distribution')
axes[1, 0].legend()

# 5. Central Limit Theorem demonstration
sample_sizes = [1, 5, 30, 100]
colors = ['#FF6384', '#FF9F40', '#36A2EB', '#4BC0C0']
for n, color in zip(sample_sizes, colors):
    means = [np.mean(np.random.exponential(1, n)) for _ in range(1000)]
    axes[1, 1].hist(means, bins=30, alpha=0.5, color=color, label=f'n={n}', density=True)
axes[1, 1].set_title('Central Limit Theorem\n(Exponential → Normal)')
axes[1, 1].legend()

# 6. Uniform
a, b = 2, 8
x = np.linspace(0, 10, 200)
axes[1, 2].plot(x, stats.uniform.pdf(x, a, b-a), 'b-', linewidth=2)
axes[1, 2].fill_between(x, stats.uniform.pdf(x, a, b-a), alpha=0.3)
axes[1, 2].set_title(f'Uniform [{a}, {b}]')
axes[1, 2].set_ylim(0, 0.25)

plt.suptitle('Common Probability Distributions', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('distributions.png', dpi=150)
plt.show()
```

### 4.4 Descriptive Statistics

```python
import numpy as np
from scipy import stats

np.random.seed(42)
data = np.random.normal(100, 15, 1000)  # IQ-like distribution

# Measures of central tendency
print("=== Central Tendency ===")
print(f"Mean:    {np.mean(data):.2f}")
print(f"Median:  {np.median(data):.2f}")
print(f"Mode:    {stats.mode(data.round(), keepdims=True).mode[0]:.2f}")

# Measures of spread
print("\n=== Spread ===")
print(f"Variance (σ²):    {np.var(data):.2f}")
print(f"Std Dev (σ):      {np.std(data):.2f}")
print(f"Range:            {np.ptp(data):.2f}")
print(f"IQR:              {np.percentile(data, 75) - np.percentile(data, 25):.2f}")

# Measures of shape
print("\n=== Shape ===")
print(f"Skewness: {stats.skew(data):.4f}")    # 0 = symmetric
print(f"Kurtosis: {stats.kurtosis(data):.4f}")  # 0 = normal-like tails

# Percentiles
print("\n=== Percentiles ===")
for p in [25, 50, 75, 90, 95, 99]:
    print(f"  {p}th percentile: {np.percentile(data, p):.2f}")
```

### 4.5 Maximum Likelihood Estimation (MLE)

MLE is the most common method to estimate model parameters. Given data $\mathbf{x} = (x_1, \ldots, x_n)$, find $\theta$ that maximizes the likelihood:

$$\hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^{n} P(x_i | \theta)$$

In practice, we maximize the **log-likelihood** (sums are easier than products):

$$\hat{\theta}_{MLE} = \arg\max_\theta \sum_{i=1}^{n} \log P(x_i | \theta)$$

**Example**: MLE for Gaussian parameters.

Given data $x_1, \ldots, x_n$ from $\mathcal{N}(\mu, \sigma^2)$:

$$\log L = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2$$

Taking derivatives and setting to zero:

$$\hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} x_i = \bar{x}$$

$$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

The MLE estimates are simply the sample mean and variance!

```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Generate data from a known distribution
np.random.seed(42)
true_mu, true_sigma = 5.0, 2.0
data = np.random.normal(true_mu, true_sigma, 200)

# Method 1: Analytical MLE
mu_mle = np.mean(data)
sigma_mle = np.std(data)  # Note: np.std uses 1/n (MLE), not 1/(n-1)
print(f"True parameters:     μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates:       μ={mu_mle:.4f}, σ={sigma_mle:.4f}")

# Method 2: Numerical MLE (using optimization)
def neg_log_likelihood(params, data):
    mu, sigma = params
    if sigma <= 0:
        return 1e10
    n = len(data)
    ll = -n/2 * np.log(2*np.pi) - n * np.log(sigma) - \
         np.sum((data - mu)**2) / (2 * sigma**2)
    return -ll  # Negative because we minimize

result = minimize(neg_log_likelihood, x0=[0, 1], args=(data,), method='Nelder-Mead')
print(f"Numerical MLE:       μ={result.x[0]:.4f}, σ={result.x[1]:.4f}")

# Visualize the likelihood surface
mu_range = np.linspace(3, 7, 100)
sigma_range = np.linspace(1, 3, 100)
MU, SIGMA = np.meshgrid(mu_range, sigma_range)
LL = np.zeros_like(MU)

for i in range(len(sigma_range)):
    for j in range(len(mu_range)):
        LL[i, j] = -neg_log_likelihood([MU[i, j], SIGMA[i, j]], data)

fig, ax = plt.subplots(figsize=(10, 8))
cs = ax.contourf(MU, SIGMA, LL, levels=50, cmap='viridis')
ax.plot(mu_mle, sigma_mle, 'r*', markersize=15, label='MLE')
ax.plot(true_mu, true_sigma, 'wx', markersize=15, markeredgewidth=3, label='True')
ax.set_xlabel('μ', fontsize=14)
ax.set_ylabel('σ', fontsize=14)
ax.set_title('Log-Likelihood Surface', fontsize=14)
ax.legend(fontsize=12)
plt.colorbar(cs, label='Log-likelihood')
plt.tight_layout()
plt.savefig('mle_surface.png', dpi=150)
plt.show()
```

---

## 5. Optimization Theory

### 5.1 Gradient Descent

Gradient descent is how ML models **learn**. The idea is simple:

1. Start with random parameters $\theta$
2. Compute the gradient of the loss: $\nabla L(\theta)$
3. Update: $\theta \leftarrow \theta - \alpha \nabla L(\theta)$
4. Repeat until convergence

where $\alpha$ is the **learning rate**.

**Intuition**: Imagine you're on a mountain in fog and want to reach the valley. You feel the slope under your feet and step downhill. That's gradient descent!

```python
import numpy as np
import matplotlib.pyplot as plt

# Minimize f(x) = x^4 - 3x^3 + 2
def f(x):
    return x**4 - 3*x**3 + 2

def df(x):
    return 4*x**3 - 9*x**2

# Gradient descent
def gradient_descent(f, df, x0, lr=0.01, n_steps=100):
    path = [x0]
    x = x0
    for _ in range(n_steps):
        x = x - lr * df(x)
        path.append(x)
    return np.array(path)

# Run with different learning rates
x = np.linspace(-1, 3.5, 200)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
learning_rates = [0.001, 0.01, 0.05]

for ax, lr in zip(axes, learning_rates):
    path = gradient_descent(f, df, x0=3.0, lr=lr, n_steps=50)
    
    ax.plot(x, f(x), 'b-', linewidth=2)
    ax.plot(path, f(path), 'ro-', markersize=4, alpha=0.7)
    ax.plot(path[0], f(path[0]), 'go', markersize=10, label='Start')
    ax.plot(path[-1], f(path[-1]), 'r*', markersize=15, label=f'End: x={path[-1]:.3f}')
    ax.set_title(f'Learning Rate = {lr}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

plt.suptitle('Gradient Descent with Different Learning Rates', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=150)
plt.show()
```

### 5.2 Gradient Descent Variants

| Variant | Update Rule | Pros | Cons |
|---------|-------------|------|------|
| **Batch GD** | Use ALL data points | Stable convergence | Slow for large datasets |
| **Stochastic GD** | Use 1 random data point | Fast, can escape local minima | Very noisy |
| **Mini-batch GD** | Use a batch of $b$ data points | Best of both worlds | Need to choose batch size |
| **Momentum** | $v_t = \beta v_{t-1} + \nabla L$, then $\theta -= \alpha v_t$ | Faster convergence | Extra hyperparameter |
| **Adam** | Adaptive learning rates per parameter | Works great in practice | Slight overhead |

### 5.3 Convexity

A function is **convex** if a line segment between any two points on the function lies above the function:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y), \quad \forall \lambda \in [0, 1]$$

**Why it matters**: Convex functions have a single global minimum — gradient descent is guaranteed to find it!

- Linear regression loss (MSE) → **convex** ✓
- Logistic regression loss → **convex** ✓
- Neural network loss → **non-convex** ✗ (many local minima)

---

## 6. Information Theory

Information theory quantifies **uncertainty** and **information content**. It's used in:
- Decision trees (entropy, information gain)
- Cross-entropy loss
- KL divergence

### 6.1 Entropy

**Entropy** measures the average uncertainty in a random variable:

$$H(X) = -\sum_{i=1}^{n} P(x_i) \log_2 P(x_i)$$

- **Low entropy**: Data is predictable (e.g., a biased coin with P(heads) = 0.99)
- **High entropy**: Data is unpredictable (e.g., a fair coin with P(heads) = 0.5)

### 6.2 Cross-Entropy

Measures the difference between two distributions $P$ (true) and $Q$ (predicted):

$$H(P, Q) = -\sum_{i} P(x_i) \log Q(x_i)$$

**This is the most common loss function for classification in ML!**

### 6.3 KL Divergence

Measures how different distribution $Q$ is from $P$:

$$D_{KL}(P \| Q) = \sum_{i} P(x_i) \log \frac{P(x_i)}{Q(x_i)} = H(P, Q) - H(P)$$

Note: $D_{KL} \geq 0$ and $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ (not symmetric!)

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(probs):
    """Compute Shannon entropy."""
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs))

def cross_entropy(p, q):
    """Compute cross-entropy H(p, q)."""
    return -np.sum(p * np.log2(q + 1e-10))

def kl_divergence(p, q):
    """Compute KL divergence D_KL(p || q)."""
    mask = p > 0
    return np.sum(p[mask] * np.log2(p[mask] / (q[mask] + 1e-10)))

# Entropy of a coin with different biases
p_heads = np.linspace(0.01, 0.99, 100)
entropies = [entropy(np.array([p, 1-p])) for p in p_heads]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(p_heads, entropies, 'b-', linewidth=2)
axes[0].set_xlabel('P(Heads)', fontsize=12)
axes[0].set_ylabel('Entropy (bits)', fontsize=12)
axes[0].set_title('Entropy of a Coin Flip', fontsize=14)
axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Maximum entropy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# KL Divergence visualization
p = np.array([0.2, 0.5, 0.3])  # True distribution
q_range = np.linspace(0.01, 0.98, 50)
kl_values = []
for q1 in q_range:
    q = np.array([q1, (1-q1)/2, (1-q1)/2])  # Vary first probability
    kl_values.append(kl_divergence(p, q))

axes[1].plot(q_range, kl_values, 'r-', linewidth=2)
axes[1].set_xlabel('q₁', fontsize=12)
axes[1].set_ylabel('D_KL(P || Q)', fontsize=12)
axes[1].set_title('KL Divergence as Q varies', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('information_theory.png', dpi=150)
plt.show()

# Example calculations
print("=== Information Theory Examples ===")
fair_coin = np.array([0.5, 0.5])
biased_coin = np.array([0.9, 0.1])
print(f"Fair coin entropy:   {entropy(fair_coin):.4f} bits")
print(f"Biased coin entropy: {entropy(biased_coin):.4f} bits")
print(f"KL(fair || biased):  {kl_divergence(fair_coin, biased_coin):.4f}")
print(f"KL(biased || fair):  {kl_divergence(biased_coin, fair_coin):.4f}")
```

---

## 7. Project Ideas

### 🟢 Project 1: Gradient Descent Visualizer (Beginner)
Build an interactive gradient descent visualizer for 2D functions.
- Implement batch, stochastic, and mini-batch GD
- Visualize convergence paths with different learning rates
- **Skills**: Calculus, NumPy, Matplotlib

### 🟡 Project 2: Probability Calculator (Intermediate)
Build a Bayesian inference engine that updates probabilities as evidence arrives.
- Implement Bayes' theorem with multiple evidence updates
- Visualize prior → posterior evolution
- **Skills**: Probability, Bayes' theorem

### 🔴 Project 3: Linear Algebra Decomposition Library (Advanced)
Implement SVD, eigendecomposition, and QR factorization from scratch.
- Compare with NumPy implementations
- Apply to image compression
- **Skills**: Linear algebra, numerical methods

---

## 8. What's Next

| Next Topic | Why |
|------------|-----|
| [Probability And Statistics](./03-Probability-And-Statistics.md) | Continue the prerequisites by mastering probability |
| [What Is Data Science And ML](../01-Data-Science-Foundations/01-What-Is-Data-Science-And-ML.md) | Apply this math to real ML algorithms |
| [Linear Regression](../02-Supervised-Learning/01-Linear-Regression.md) | See how linear algebra solves regression |

### 🛤️ Learning Pathway
```
You are here: Mathematical Foundations
                    │
                    ▼
           Probability And Statistics
                    │
                    ▼
          SQL For Data Science
```

---

> **Key Takeaway**: You don't need to memorize all these formulas. Understanding the *intuition* behind each concept is what matters. The code implementations help build that intuition.

---

[← Python Essentials](./01-Python-Essentials.md) | [Back to Index](../README.md) | [Next: Probability And Statistics →](./03-Probability-And-Statistics.md)
