# 📈 Learning Curves

> **Prerequisites**: Bias-Variance Tradeoff | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 📋 Table of Contents
1. [Diagnosing Model Health](#1-diagnosing-model-health)
2. [Visualizing Learning Curves](#2-visualizing-learning-curves)

---

## 1. Diagnosing Model Health

### 🟢 Beginner
**Simple Explanation**: 
A learning curve shows how your model improves as you give it more and more data to study. 
- If you give it 10 examples, it memorizes them easily but fails the test.
- If you give it 10,000 examples, it struggles to memorize them all, but it learns the *real* pattern, so its test score improves.

### 🟡 Intermediate
**Workflow and Diagnosis**: 
We plot the Training Error and the Validation Error on the Y-axis against the Number of Training Examples on the X-axis.
- **High Bias (Underfitting)**: Both training and validation curves quickly flatline at a high error rate. Adding more data *will not help*. You need a more complex model.
- **High Variance (Overfitting)**: Training error is very low, validation error is very high, and there is a huge gap between them. Adding more data *will help* close the gap.

### 🔴 Advanced
**Implementation**:

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    estimator=model, X=X, y=y, train_sizes=np.linspace(0.1, 1.0, 10), cv=5
)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.plot(train_sizes, train_mean, label='Training Score')
plt.plot(train_sizes, val_mean, label='Validation Score')
plt.title('Learning Curve')
plt.xlabel('Number of Training Examples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```
**Industry Context**: Compute costs. If the learning curve has already asymptoted at 50,000 rows, spending $10,000 to acquire and label 500,000 more rows is a waste of money.

---

[← Cross Validation](09-Cross-Validation.md) | [Back to Index](../README.md) | [Next: Validation Curves →](11-Validation-Curves.md)
