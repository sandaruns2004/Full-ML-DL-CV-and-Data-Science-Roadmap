# 📈 Polynomial Regression

> **Prerequisites:** Linear Regression
>
> **Difficulty:** ⭐⭐☆☆☆
>
> **Estimated Reading Time:** 15 minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Mathematics](#3-mathematics)
4. [Visual Explanation](#4-visual-explanation)
5. [Algorithm Workflow](#5-algorithm-workflow)
6. [From Scratch Implementation](#6-from-scratch-implementation)
7. [NumPy Implementation](#7-numpy-implementation)
8. [Scikit-Learn Implementation](#8-scikit-learn-implementation)
9. [Hyperparameter Deep Dive](#9-hyperparameter-deep-dive)
10. [Visualization Lab](#10-visualization-lab)
11. [Failure Cases](#11-failure-cases)
12. [Industry Applications](#12-industry-applications)

14. [Exercises](#14-exercises)


---

# 1. What Problem Does This Solve?

### 🟢 Beginner
Linear Regression draws a straight line. But what if your data curves? For example, human height grows rapidly from ages 0-15, slows down from 15-20, and then stops completely. If you draw a straight line through that, it will predict that a 90-year-old is 15 feet tall! Polynomial Regression allows you to bend the line to fit curves.

### 🟡 Intermediate
Polynomial Regression is technically just Multiple Linear Regression in disguise. When the true relationship between features and target is non-linear, we artificially create new features by raising existing features to a power (e.g., squaring them or cubing them). This allows a linear model to fit a non-linear boundary.

### 🔴 Advanced
Polynomial expansion is a fundamental technique for increasing the dimensionality of your feature space. However, it suffers heavily from the **Curse of Dimensionality**. If you have 10 features and you expand them to degree 4, you instantly create hundreds of highly correlated, redundant features, which can mathematically destabilize the closed-form Normal Equation and lead to massive overfitting.

---

# 2. Intuition

Imagine you have a single slider variable: $x$. A straight line can only ever be a straight line: $y = mx + b$.

But what if, behind the scenes, you secretly calculate $x^2$ and feed it to the model as a *brand new variable* $z$? 
Now the model is solving: $y = w_1 x + w_2 z + b$.
Because the model doesn't know that $z$ is secretly $x^2$, it just treats it like a normal linear regression problem in a 2D space. But when you plot the result back onto the original 1D graph of $x$, the line magically forms a perfect parabola!

---

# 3. Mathematics

### 3.1 Feature Expansion
To fit a $d$-degree polynomial on a single feature $x$, we create a matrix of expanded features:
$$ X_{\text{poly}} = [1, x, x^2, x^3, \dots, x^d] $$

### 3.2 The Linear Hypothesis
Once the features are expanded, the math is **identical** to standard Linear Regression.
$$ \hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_d x^d $$

Notice that the equation is non-linear with respect to the input data $x$, but it is **strictly linear** with respect to the parameters $\theta$. This is why it is still considered "Linear" Regression—we are still just finding optimal weights for terms!

### 3.3 The Curse of Dimensionality
If you have two features $a$ and $b$, and you want a degree-3 expansion, you don't just get $a^3$ and $b^3$. You get all interaction terms:
$$ [1, a, b, a^2, b^2, ab, a^3, b^3, a^2b, ab^2] $$
The number of features explodes factorially!

---

# 4. Visual Explanation

```mermaid
flowchart LR
    Data[1D Feature: X] --> PF[PolynomialFeatures]
    PF -->|Square it| X2[X^2]
    PF -->|Cube it| X3[X^3]
    
    Data --> Reg[Linear Regression Model]
    X2 --> Reg
    X3 --> Reg
    
    Reg --> Final[🏆 Curved Decision Boundary]
```

---

# 5. Algorithm Workflow

1. **Input Data**: You have an input matrix $X$.
2. **Transform**: Pass $X$ through a Polynomial feature transformer to generate new columns of squared, cubed, and interaction terms.
3. **Scale**: **CRITICAL STEP**. Because $10^3 = 1000$, the new features will have wildly different scales. You MUST apply standard scaling.
4. **Train**: Pass the expanded, scaled matrix to a standard Linear Regression model.
5. **Predict**: When predicting new data, you must apply the exact same transformation and scaling before predicting.

---

# 6. From Scratch Implementation

```python
import numpy as np

class PolynomialFeatureTransformer:
    def __init__(self, degree=2):
        self.degree = degree
        
    def transform(self, X):
        # Taking a 1D array and expanding it
        X_poly = [X]
        for d in range(2, self.degree + 1):
            X_poly.append(X ** d)
        # Stack them horizontally
        return np.column_stack(X_poly)

# Assume SimpleLinearRegression from Chapter 2 is available
# X_poly = PolynomialFeatureTransformer(degree=2).transform(X)
# model.fit(X_poly, y)
```

---

# 7. NumPy Implementation

Vectorized polynomial expansion for multiple features is tricky from scratch because of the interaction terms. This is why we almost always rely on Scikit-Learn's `PolynomialFeatures`.

---

# 8. Scikit-Learn Implementation

```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([[1], [4], [9], [16], [25]]) # Perfect parabola

# 2. Create a Pipeline! (Best Practice)
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])

# 3. Train
poly_model.fit(X, y)

# 4. Predict
print(f"Prediction for 6: {poly_model.predict([[6]])[0][0]:.2f}") # Should be ~36
```

---

# 9. Hyperparameter Deep Dive

- **`degree`**: The highest power to raise your features to.
  - *Degree 1*: Standard straight line.
  - *Degree 2*: Parabola (U-shape).
  - *Degree 3*: S-curve.
  - *Degree 20*: Madness (Massive Overfitting).
- **`interaction_only`**: If `True`, it will only calculate interaction features like $a \times b$, but not $a^2$ or $b^2$. Useful if you suspect features multiply together (like width $\times$ height) but don't curve individually.

---

# 10. Visualization Lab

*See how changing the degree changes the fit.*

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1) # Non-linear data

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5, label='Data')

# Plot Degree 1 (Underfit)
model_1 = LinearRegression().fit(X, y)
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
plt.plot(X_plot, model_1.predict(X_plot), color='red', label='Degree 1')

# Plot Degree 2 (Perfect fit)
poly_2 = PolynomialFeatures(degree=2, include_bias=False)
model_2 = LinearRegression().fit(poly_2.fit_transform(X), y)
plt.plot(X_plot, model_2.predict(poly_2.transform(X_plot)), color='green', linewidth=3, label='Degree 2')

# Plot Degree 300 (Overfit)
poly_300 = PolynomialFeatures(degree=300, include_bias=False)
model_300 = LinearRegression().fit(poly_300.fit_transform(X), y)
plt.plot(X_plot, model_300.predict(poly_300.transform(X_plot)), color='purple', label='Degree 300 (Overfit)')

plt.ylim(0, 10)
plt.legend()
plt.title("Polynomial Regression Degrees")
plt.show()
```

---

# 11. Failure Cases

### Runge's Phenomenon (Extreme Overfitting)
If you set the `degree` too high (e.g., 20), the model will force the curve to snake up and down violently to perfectly hit every single training data point. When given new data, the curve will output numbers in the millions or billions. 
*Fix: Heavily restrict the degree to 2 or 3, or use heavily regularized models like Ridge/Lasso.*

---

# 12. Industry Applications

- **Physics & Engineering**: Modeling trajectories (which are inherently parabolic due to gravity).
- **Economics**: Yield curve modeling.
- **Epidemiology**: Early stages of viral spread can often be modeled effectively with low-degree polynomial curves before exponential growth kicks in.

---

# 14. Exercises

### Easy
Run the visualization code above. Change the `degree=300` to `degree=3` and watch how the purple line behaves.

### Medium
Using a dataset of Housing Prices, apply a `PolynomialFeatures(degree=2)` transformation. Before applying standard scaling, print out the `.head()` of the dataset to see how large the squared "Square Footage" column has become.

### Hard
Polynomial expansion is actually a manual version of the **Kernel Trick**. Explain mathematically how a Support Vector Machine with a Polynomial Kernel achieves the exact same result as Polynomial Regression without ever actually calculating the expanded $X_{poly}$ matrix.

---

[← Linear Regression](02-Linear-Regression.md) | [Back to Index](../README.md) | [Next: Logistic Regression →](04-Logistic-Regression.md)
