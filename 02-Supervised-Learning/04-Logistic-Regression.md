# 📊 Logistic Regression

> **Prerequisites:** Linear Regression, Probabilities
>
> **Difficulty:** ⭐⭐☆☆☆
>
> **Estimated Reading Time:** 20 minutes

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
13. [Interview Preparation](#13-interview-preparation)
14. [Exercises](#14-exercises)
15. [Further Reading](#15-further-reading)

---

# 1. What Problem Does This Solve?

### 🟢 Beginner
Linear Regression predicts a number (e.g., house price = \$450,000). But what if you want to predict a **Yes or No** question? Will this patient survive? Will this email be marked as spam? Will this customer click on the ad?
You can't predict a price for these. You need a percentage. Logistic Regression is the algorithm that predicts the **probability** (from 0% to 100%) that something belongs to a certain category.

### 🟡 Intermediate
Despite having "Regression" in the name, Logistic Regression is fundamentally a **Classification** algorithm. It is used for binary classification (0 or 1). Instead of drawing a straight line through points, it draws an S-shaped curve that bounds all predictions strictly between 0.0 and 1.0. If the output is > 0.5, it predicts class 1. Otherwise, class 0.

### 🔴 Advanced
Logistic Regression serves as the fundamental unit (a single neuron/perceptron) of modern Deep Learning architectures. The mathematics of logistic regression—specifically the Sigmoid activation function and the Binary Cross-Entropy (Log Loss) cost function—are the exact same mathematical constructs used in the output layers of advanced classification Neural Networks.

---

# 2. Intuition

Imagine you try to use Linear Regression to classify Malignant (1) vs Benign (0) tumors based on tumor size. 
You draw a straight line through the data. For a while, it works: tumors smaller than 2cm output a number $< 0.5$ (Benign), and tumors $> 2$cm output a number $> 0.5$ (Malignant).

But then, a massive 15cm tumor is added to the dataset. The straight line tilts wildly upward to accommodate it. Now, 3cm tumors that *should* be malignant are suddenly outputting $0.3$ and being classified as benign! Furthermore, the 15cm tumor outputs a "probability" of $2.5$ (250%), which makes no sense.

Logistic Regression solves this by putting a "ceiling" at 1 and a "floor" at 0. It squashes the straight line into an S-curve, ensuring no matter how massive the outlier is, the probability approaches, but never exceeds, 1.0.

---

# 3. Mathematics

### 3.1 The Sigmoid Function
How do we squash a linear line $z = \theta^T X$ into the range $(0, 1)$? We pass it through the **Sigmoid (Logistic) Function**:
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
If $z$ is a huge positive number, $e^{-z}$ becomes $0$, and the output is $\frac{1}{1} = 1$.
If $z$ is a huge negative number, $e^{-z}$ becomes $\infty$, and the output is $\frac{1}{\infty} = 0$.

### 3.2 The Hypothesis
$$ \hat{y} = \sigma(\theta^T X) $$

### 3.3 Log Loss (Binary Cross-Entropy)
We cannot use Mean Squared Error (MSE) because wrapping MSE around the Sigmoid function creates a non-convex, wavy cost function full of local minima. 
Instead, we use **Log Loss**, which heavily penalizes being confidently wrong:
$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] $$
- If $y=1$, the right half disappears. We want $\log(\hat{y})$ to be close to 1.
- If $y=0$, the left half disappears. We want $\log(1-\hat{y})$ to be close to 1.

### 3.4 Gradient Descent
Miraculously, the derivative of Log Loss with the Sigmoid function cancels out beautifully, resulting in the exact same gradient formula as Linear Regression!
$$ \theta_j = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)} $$

---

# 4. Visual Explanation

```mermaid
flowchart TD
    Data[Input Features X] --> Linear[Compute Z = X * Theta]
    Linear --> Sigmoid[Pass Z through Sigmoid Function]
    Sigmoid --> Prob[Output Probability between 0 and 1]
    
    Prob --> Threshold{Is Prob >= 0.5?}
    Threshold -- Yes --> C1[Predict Class 1]
    Threshold -- No --> C0[Predict Class 0]
    
    Prob --> Cost[Calculate Log Loss]
    Cost --> Backprop[Gradient Descent Updates Theta]
```

---

# 5. Algorithm Workflow

1. **Calculate Linear Equation**: Multiply inputs by weights.
2. **Squash**: Pass the result through the sigmoid function to get a probability.
3. **Threshold**: Convert the probability to a discrete class label (usually using 0.5 as the cutoff).
4. **Evaluate Error**: Calculate the Log Loss using the true labels.
5. **Update**: Use Gradient Descent to adjust weights.

---

# 6. From Scratch Implementation

```python
import math

class SimpleLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = 0.0
        self.b = 0.0
        
    def _sigmoid(self, z):
        # Clip z to prevent math overflow errors
        z = max(min(z, 250), -250)
        return 1 / (1 + math.exp(-z))
        
    def fit(self, X, y):
        m = len(X)
        for _ in range(self.epochs):
            # Forward pass
            y_pred = [self._sigmoid(self.w * X[i] + self.b) for i in range(m)]
            
            # Gradients (Same as Linear Regression!)
            dw = (1/m) * sum((y_pred[i] - y[i]) * X[i] for i in range(m))
            db = (1/m) * sum((y_pred[i] - y[i]) for i in range(m))
            
            # Update
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
    def predict_proba(self, X):
        return [self._sigmoid(self.w * x + self.b) for x in X]
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probs]
```

---

# 7. NumPy Implementation

```python
import numpy as np

class VectorizedLogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        
    def fit(self, X, y):
        X_b = np.c_[np.ones((len(X), 1)), X]
        self.weights = np.zeros((X_b.shape[1], 1))
        m = len(X)
        y = y.reshape(-1, 1)
        
        for _ in range(self.epochs):
            z = X_b.dot(self.weights)
            y_pred = 1 / (1 + np.exp(-z)) # Vectorized Sigmoid
            
            gradients = (1/m) * X_b.T.dot(y_pred - y)
            self.weights -= self.lr * gradients
            
    def predict(self, X):
        X_b = np.c_[np.ones((len(X), 1)), X]
        z = X_b.dot(self.weights)
        probs = 1 / (1 + np.exp(-z))
        return (probs >= 0.5).astype(int)
```

---

# 8. Scikit-Learn Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# 1. Data (Tumor size predicting Malignant 1 or Benign 0)
X = np.array([[1.1], [1.5], [2.2], [2.9], [4.5], [5.1], [6.2]])
y = np.array([0, 0, 0, 1, 1, 1, 1])

# 2. Train
model = LogisticRegression(penalty='l2', C=1.0) # Built-in L2 Regularization!
model.fit(X, y)

# 3. Predict Probabilities
probs = model.predict_proba([[2.5]])
print(f"Prob Benign: {probs[0][0]:.2f}, Prob Malignant: {probs[0][1]:.2f}")

# 4. Evaluate
preds = model.predict(X)
print(f"Accuracy: {accuracy_score(y, preds)}")
print(f"Confusion Matrix:\n{confusion_matrix(y, preds)}")
```

---

# 9. Hyperparameter Deep Dive

- **`penalty`**: Logistic Regression mathematically risks pushing weights to infinity to perfectly separate classes (if the data is perfectly separable). Scikit-Learn applies `l2` regularization by default to prevent this.
- **`C`**: The inverse of regularization strength (just like SVMs). 
  - *Small `C` (e.g. 0.01)*: Strong regularization. Forces weights to be tiny. High Bias, Low Variance.
  - *Large `C` (e.g. 100)*: Weak regularization. Allows weights to grow. Low Bias, High Variance.
- **`solver`**: The algorithm used to find the minimum of the cost function. `lbfgs` is the default, `liblinear` is better for small datasets, and `saga` is best for massive datasets and L1 penalties.
- **`class_weight`**: Set to `'balanced'` if you have highly imbalanced data (e.g., 99% non-spam, 1% spam). It forces the algorithm to penalize errors on the minority class more heavily.

---

# 10. Visualization Lab

*Visualizing the S-Curve.*

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = (X > 0).astype(int).ravel() # Step function

model = LogisticRegression().fit(X, y)
probs = model.predict_proba(X)[:, 1]

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', alpha=0.3, label='Actual Data (0 or 1)')
plt.plot(X, probs, color='red', linewidth=3, label='Sigmoid Probability Curve')
plt.axhline(0.5, color='black', linestyle='--', label='Decision Threshold')
plt.title("Logistic Regression S-Curve")
plt.xlabel("Feature Value")
plt.ylabel("Probability of Class 1")
plt.legend()
plt.grid(True)
plt.show()
```

---

# 11. Failure Cases

### Non-Linear Boundaries
Logistic regression draws a strictly straight decision boundary (a hyperplane) across the feature space. If the classes form a circle or a donut shape, Logistic Regression will fail completely.
*Fix: Use Polynomial feature expansion before passing to Logistic Regression, or use a non-linear model like Random Forest.*

### Outliers with L1/None Regularization
While the sigmoid function handles outliers better than Linear Regression, extreme outliers can still heavily distort the probability curve if regularization is turned off.

---

# 12. Industry Applications

- **Credit Scoring**: Predicting the probability that a user will default on a loan. Used universally because the output is a literal calibrated probability, not just a binary label.
- **Medicine**: Predicting disease presence given a set of biomarkers.
- **Marketing**: Click-through rate (CTR) prediction for online advertisements.

---

# 13. Interview Preparation

### Beginner Questions
**Q: Why do we call it Logistic Regression if it's used for Classification?**
> A: Because underneath the hood, it is still calculating a continuous regression line ($y = mx + b$), it just squashes that continuous line into a bounded curve (the Logistic function) and thresholds it to make a class prediction.

### Intermediate Questions
**Q: Why don't we use Mean Squared Error (MSE) for Logistic Regression?**
> A: Because plugging the non-linear Sigmoid function into the MSE formula results in a "wavy", non-convex cost function with multiple local minimums. Gradient Descent would get stuck. Log Loss guarantees a perfectly convex bowl.

### Advanced Questions
**Q: What happens to the weights of a Logistic Regression model if the data is *perfectly* linearly separable and there is no regularization?**
> A: The weights will grow to infinity! To maximize the probability of correct classes to exactly 1.0 and 0.0, the Sigmoid curve needs to become a perfect step-function. The only way to make Sigmoid a perfect step-function is to multiply the inputs by infinitely large weights. This is why Scikit-learn forces L2 regularization (`C=1.0`) by default.

---

# 14. Exercises

### Easy
Use the `predict_proba` method to find out what probability the model outputs for an input of exactly `0.0` in the Visualization Lab above.

### Medium
Plot the ROC Curve (Receiver Operating Characteristic) for a Logistic Regression model trained on the breast cancer dataset, and calculate the AUC.

### Hard
Implement **Softmax Regression** (Multinomial Logistic Regression) from scratch to handle more than 2 classes. Hint: The sigmoid function is replaced by the Softmax function $e^{z_i} / \sum e^{z_j}$.

---

# 15. Further Reading

- *Pattern Recognition and Machine Learning (Bishop)* - Chapter 4
- *Hands-On Machine Learning* - Chapter 4 (Logistic Regression section)
- Scikit-Learn Documentation: `sklearn.linear_model.LogisticRegression`

---

[← Previous Chapter (Polynomial Regression)](./03-Polynomial-Regression.md) | [Next Chapter (KNN) →](./05-KNN.md)

[Return to Root Index](../README.md)
