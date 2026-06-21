# 📈 AdaBoost (Adaptive Boosting)

> **Prerequisites**: Boosting Introduction | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 📋 Table of Contents
1. [Sample Weights & Weak Learners](#1-sample-weights--weak-learners)
2. [The AdaBoost Algorithm](#2-the-adaboost-algorithm)
3. [scikit-learn Implementation & Visualization](#3-scikit-learn-implementation--visualization)

---

## 1. Sample Weights & Weak Learners

### 🟢 Beginner
**Simple Explanation**: AdaBoost assigns "weights" to data points, which act as a priority score.
At the beginning, all points have equal priority. If a point is misclassified, its weight increases, making it a "bigger target" for the next tree. The final prediction is a weighted vote where models that performed better get a louder voice.

---

## 2. The AdaBoost Algorithm

### 🟡 Intermediate
AdaBoost usually uses "Decision Stumps" (trees with a maximum depth of 1) as its weak learners.

**Algorithm Steps**:
1. Initialize sample weights $w_i = \frac{1}{N}$ for $i = 1, \dots, N$.
2. For $t = 1, 2, \dots, T$:
   - Train a weak learner $h_t(x)$ on the weighted dataset.
   - Compute its weighted error:
   
     $$\epsilon_t = \frac{\sum_{i: h_t(x_i) \neq y_i} w_i}{\sum_{i=1}^N w_i}$$
     
   - Compute the model's voting weight $\alpha_t$:
   
     $$\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$$
     
   - Update the sample weights:
   
     $$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))$$
     
     *(where $y_i \in \{-1, 1\}$ and $h_t(x_i) \in \{-1, 1\}$. If correct, exponent is negative, weight decreases. If wrong, exponent is positive, weight increases.)*
   - Re-normalize weights so $\sum w_i^{(t+1)} = 1$.
3. The final ensemble prediction is:

   $$H(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t h_t(x)\right)$$

---

## 3. scikit-learn Implementation & Visualization

Here is how you can train AdaBoost and visualize how its test accuracy evolves as more weak learners are added:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                           n_clusters_per_class=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoost with decision stumps
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_train, y_train)

# Show staged test accuracy scores
staged_scores = list(ada.staged_score(X_test, y_test))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Learning curve
axes[0].plot(range(1, len(staged_scores)+1), staged_scores, 'b-', linewidth=2)
axes[0].set_xlabel('Number of Estimators')
axes[0].set_ylabel('Test Accuracy')
axes[0].set_title('AdaBoost Learning Curve', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Decision boundary
xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 200),
                      np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 200))
Z = ada.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[1].contourf(xx, yy, Z, cmap='RdBu', alpha=0.3)
axes[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], c='red', edgecolor='black', s=30)
axes[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], c='blue', edgecolor='black', s=30)
axes[1].set_title(f'AdaBoost Decision Boundary (acc={ada.score(X_test, y_test):.2%})', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

[← Boosting Introduction](./06-Boosting-Introduction.md) | [Back to Index](../README.md) | [Next: Gradient Boosting →](./08-Gradient-Boosting.md)
