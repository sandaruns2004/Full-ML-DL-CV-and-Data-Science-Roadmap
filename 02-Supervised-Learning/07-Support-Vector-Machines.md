# ⚔️ Support Vector Machines (SVM)

> **Prerequisites**: Logistic Regression | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 1. Linear & Kernel SVM

### 🟢 Beginner
**Simple Explanation**: Imagine drawing a wide road between two groups of dots. SVM tries to make this road as wide as possible without touching the dots.

### 🟡 Intermediate
**Working Mechanism**: It finds the optimal hyperplane that maximizes the margin between classes. Support vectors are the points closest to the boundary.

### 🔴 Advanced
**Mathematics**:
Optimization Objective: Minimize $\frac{1}{2} ||w||^2$ subject to $y_i(w \cdot x_i + b) \ge 1$.
**Kernel Trick**: Uses functions (Polynomial, RBF) to implicitly map data into high-dimensional space without explicitly computing coordinates, using $K(x, x') = \phi(x)^T \phi(x')$.

---

[← Decision Trees](06-Decision-Trees.md) | [Back to Index](../README.md) | [Next: Naive Bayes →](08-Naive-Bayes.md)
