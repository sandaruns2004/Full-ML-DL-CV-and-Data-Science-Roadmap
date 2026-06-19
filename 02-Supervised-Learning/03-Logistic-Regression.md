# 🎯 Logistic Regression — Classification with Deep Math

> **Prerequisites**: Linear Regression | **Difficulty**: ⭐⭐☆☆☆ Elementary

---

## 📋 Table of Contents
1. [From Regression to Classification](#1-from-regression-to-classification)
2. [The Sigmoid Function](#2-the-sigmoid-function)
3. [Mathematical Derivation (MLE)](#3-mathematical-derivation-mle)
4. [Gradient Descent for Logistic Regression](#4-gradient-descent-for-logistic-regression)
5. [Decision Boundary](#5-decision-boundary)
6. [Multi-Class Classification](#6-multi-class-classification)
7. [Implementation from Scratch](#7-implementation-from-scratch)
8. [scikit-learn Implementation](#8-scikit-learn-implementation)
9. [Regularized Logistic Regression](#9-regularized-logistic-regression)
10. [Project Ideas](#10-project-ideas)
11. [What's Next](#11-whats-next)

---

## 1. From Regression to Classification

> **🧠 Why is it called "Regression" if it's used for Classification?** 
> It's an unfortunate historical naming convention! It uses the underlying math of linear *regression* (drawing a line), but then wraps that line in a special mathematical function to output probabilities instead of raw numbers. Think of it as a classification algorithm with a regression engine inside.

Linear regression predicts continuous values. But what if we need to predict **categories** (spam/not-spam, cat/dog)?

**Problem with linear regression for classification**:
- Output can be any real number, but we need probabilities $[0, 1]$
- No natural threshold for decision making

**Solution**: Wrap the linear function in a **sigmoid** to squash output to $[0, 1]$:

$$P(y=1|x) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}$$

---

## 2. The Sigmoid Function and Log-Odds

> **🧠 ELI5 Analogy (Log-Odds):** Imagine a sports game. If a team has a 75% chance of winning, the *odds* are 3-to-1 (75/25). Odds can go from 0 to infinity. If we take the *logarithm* of those odds, the scale stretches from negative infinity to positive infinity. Linear regression *loves* predicting numbers from negative to positive infinity. So, Logistic Regression uses a linear equation to predict the *log-odds* of an event, and the **Sigmoid function** acts as a translator, turning those log-odds back into a simple 0% to 100% probability!

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties**:
- Output range: $(0, 1)$ — perfect for probabilities
- $\sigma(0) = 0.5$ — the decision boundary
- $\sigma(\infty) \to 1$, $\sigma(-\infty) \to 0$
- **Derivative**: $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-8, 8, 200)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sigmoid
axes[0].plot(z, sigmoid(z), 'b-', linewidth=3)
axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision boundary (0.5)')
axes[0].axvline(x=0, color='gray', linestyle=':', alpha=0.5)
axes[0].fill_between(z, 0, sigmoid(z), where=(sigmoid(z) > 0.5), alpha=0.1, color='green', label='Predict 1')
axes[0].fill_between(z, 0, sigmoid(z), where=(sigmoid(z) <= 0.5), alpha=0.1, color='red', label='Predict 0')
axes[0].set_title('Sigmoid Function σ(z)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('z = w·x + b')
axes[0].set_ylabel('σ(z) = P(y=1)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Derivative
axes[1].plot(z, sigmoid(z) * (1 - sigmoid(z)), 'r-', linewidth=3)
axes[1].set_title("Sigmoid Derivative σ'(z) = σ(z)(1-σ(z))", fontsize=14, fontweight='bold')
axes[1].set_xlabel('z')
axes[1].set_ylabel("σ'(z)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sigmoid.png', dpi=150)
plt.show()
```

---

## 3. Mathematical Derivation (MLE)

### 3.1 Likelihood Function

Given binary labels $y_i \in \{0, 1\}$ and predicted probabilities $\hat{p}_i = \sigma(\mathbf{w}^T\mathbf{x}_i)$:

$$P(y_i | \mathbf{x}_i) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1-y_i}$$

**Likelihood** (probability of all data):
$$L(\mathbf{w}) = \prod_{i=1}^{n} \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1-y_i}$$

### 3.2 Log-Likelihood

$$\ell(\mathbf{w}) = \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]$$

### 3.3 Loss Function (Binary Cross-Entropy)

We MINIMIZE the negative log-likelihood:

$$J(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]$$

**Why this loss works**:
- When $y=1$: Loss = $-\log(\hat{p})$ → penalizes low predictions heavily
- When $y=0$: Loss = $-\log(1-\hat{p})$ → penalizes high predictions heavily
- Loss = 0 when prediction is perfect

### 3.4 Gradient Derivation

$$\frac{\partial J}{\partial w_j} = \frac{1}{n}\sum_{i=1}^{n} (\hat{p}_i - y_i) x_{ij}$$

In matrix form: $\nabla J = \frac{1}{n}\mathbf{X}^T(\hat{\mathbf{p}} - \mathbf{y})$

**Remarkably**, this has the same form as linear regression's gradient! The sigmoid is baked into $\hat{p}$.

---

## 4. Gradient Descent for Logistic Regression

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
# Generate 2D classification data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Gradient Descent for Logistic Regression
n_samples, n_features = X.shape
weights = np.zeros(n_features)
bias = 0
lr = 0.1
losses = []

for epoch in range(200):
    # Forward pass
    z = X @ weights + bias
    predictions = sigmoid(z)
    
    # Binary cross-entropy loss
    loss = -np.mean(y * np.log(predictions + 1e-8) + (1 - y) * np.log(1 - predictions + 1e-8))
    losses.append(loss)
    
    # Gradients
    dw = (1/n_samples) * X.T @ (predictions - y)
    db = (1/n_samples) * np.sum(predictions - y)
    
    # Update
    weights -= lr * dw
    bias -= lr * db

accuracy = np.mean((predictions >= 0.5).astype(int) == y)
print(f"Accuracy: {accuracy:.2%}")
print(f"Weights: {weights.round(4)}")
print(f"Bias: {bias:.4f}")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ weights + bias).reshape(xx.shape)

axes[0].contourf(xx, yy, Z, levels=50, cmap='RdYlBu', alpha=0.5)
axes[0].scatter(X[y==0, 0], X[y==0, 1], c='red', edgecolor='black', s=40, label='Class 0')
axes[0].scatter(X[y==1, 0], X[y==1, 1], c='blue', edgecolor='black', s=40, label='Class 1')
axes[0].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
axes[0].set_title('Decision Boundary', fontsize=13, fontweight='bold')
axes[0].legend()

# Loss curve
axes[1].plot(losses, color='#FF6384', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Binary Cross-Entropy Loss')
axes[1].set_title('Training Loss', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_regression.png', dpi=150)
plt.show()
```

---

## 5. Decision Boundary

The decision boundary is where $P(y=1) = 0.5$, which means $\mathbf{w}^T\mathbf{x} + b = 0$.

For 2D: $w_1 x_1 + w_2 x_2 + b = 0 \implies x_2 = -\frac{w_1}{w_2}x_1 - \frac{b}{w_2}$

This is a **straight line** (hyperplane in higher dimensions).

---

## 6. Multi-Class Classification

### 6.1 Softmax Regression

For $K$ classes, the **softmax** function converts $K$ raw scores into probabilities:

$$P(y = k | \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{where } z_k = \mathbf{w}_k^T \mathbf{x} + b_k$$

**Properties**: All probabilities sum to 1, each $\in (0, 1)$.

**Loss**: Categorical Cross-Entropy:
$$J = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \log(\hat{p}_{ik})$$

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load iris (3 classes)
iris = load_iris()
X, y = iris.data[:, :2], iris.target  # Use only 2 features for visualization

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Multinomial logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

print(f"Accuracy: {model.score(X_test, y_test):.2%}")
print(f"\n{classification_report(y_test, model.predict(X_test), target_names=iris.target_names)}")

# Visualize multi-class decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10, 7))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black', s=50)
ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])
ax.set_title('Multi-Class Logistic Regression Decision Boundaries', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Class')
plt.tight_layout()
plt.savefig('multiclass_lr.png', dpi=150)
plt.show()
```

---

## 7. Implementation from Scratch

```python
import numpy as np

class LogisticRegressionFromScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []
        
        for _ in range(self.n_iterations):
            z = X @ self.weights + self.bias
            predictions = self._sigmoid(z)
            
            # Loss
            loss = -np.mean(y * np.log(predictions + 1e-8) + 
                          (1-y) * np.log(1 - predictions + 1e-8))
            self.losses.append(loss)
            
            # Gradients
            dw = (1/n_samples) * X.T @ (predictions - y)
            db = (1/n_samples) * np.sum(predictions - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict_proba(self, X):
        return self._sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=300, n_features=5, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegressionFromScratch(learning_rate=0.1, n_iterations=500)
model.fit(X_train, y_train)
print(f"Train accuracy: {model.score(X_train, y_train):.2%}")
print(f"Test accuracy:  {model.score(X_test, y_test):.2%}")
```

---

## 8. scikit-learn Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load breast cancer dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train
model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)

# Evaluate
y_pred = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

print(f"Accuracy: {model.score(X_test_s, y_test):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_proba):.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=data.target_names)}")
```

---

## 9. Regularized Logistic Regression

In scikit-learn, `C` is the **inverse** of regularization strength ($C = 1/\alpha$).

- Large `C` → less regularization → more complex model
- Small `C` → more regularization → simpler model

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
for C in C_values:
    model = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=1000)
    model.fit(X_train_s, y_train)
    train_acc = model.score(X_train_s, y_train)
    test_acc = model.score(X_test_s, y_test)
    n_nonzero = np.sum(np.abs(model.coef_) > 0.01)
    print(f"C={C:>6}: Train={train_acc:.4f}, Test={test_acc:.4f}, Non-zero={n_nonzero}")
```

---

## 10. Project Ideas

### 🟢 Email Spam Classifier (Beginner)
- **Dataset:** SMS Spam Collection Dataset or classical Enron Email dataset.
- **Task:** Build a binary classifier to predict if an email is "Spam" or "Ham".
- **Skills:** Basic Natural Language Processing (NLP) concepts using `CountVectorizer` or `TfidfVectorizer`, applying Logistic Regression on text data.
- **Output:** A confusion matrix showing False Positives and False Negatives, along with the top words that indicate spam based on model coefficients.

### 🟡 Credit Card Fraud Detection (Intermediate)
- **Dataset:** Kaggle Credit Card Fraud Detection dataset (highly imbalanced).
- **Task:** Identify fraudulent transactions out of hundreds of thousands of legitimate ones.
- **Skills:** Handling extreme class imbalance using techniques like SMOTE or class weights (`class_weight='balanced'`), threshold tuning to maximize recall.
- **Output:** Precision-Recall curve analysis rather than relying solely on accuracy.

### 🔴 Multi-Class Image Classifier (Advanced)
- **Dataset:** MNIST handwritten digits or Fashion-MNIST.
- **Task:** Classify 28x28 pixel images into one of 10 categories using Softmax Regression (Multinomial Logistic Regression).
- **Skills:** Image flattening, pixel scaling, multi-class handling (`multi_class='multinomial'`), optimizing solver choice (e.g., `saga`).
- **Output:** Visual grid of model predictions showing where the model gets confused (e.g., misclassifying '7' as '1').

---

## 11. What's Next

You've transitioned from predicting continuous numbers to discrete classes. You now understand the fundamental algorithm underlying much of modern classification.

| Next Topic | Why it's Important |
|------------|--------------------|
| [K-Nearest Neighbors](./04-KNN.md) | Explore a completely different, distance-based intuition for classification that doesn't rely on drawing a rigid mathematical boundary. |
| [Metrics & Evaluation](../05-Model-Evaluation/01-Metrics-And-Evaluation.md) | Accuracy is rarely enough. Deep dive into Precision, Recall, F1-Score, and ROC-AUC curves. |
| [Imbalanced Data](../01-Data-Science-Foundations/10-Imbalanced-Data.md) | Learn robust techniques for when one class drastically outnumbers the others (e.g., Fraud, Rare Diseases). |

---

[← Polynomial And Regularization](./02-Polynomial-And-Regularization.md) | [Back to Index](../README.md) | [Next: KNN →](./04-KNN.md)
