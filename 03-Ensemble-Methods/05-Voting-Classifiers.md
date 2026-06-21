# 🗳️ Voting Classifiers

> **Prerequisites**: Supervised Learning Classifiers | **Difficulty**: ⭐☆☆☆☆ Beginner

---

## 📋 Table of Contents
1. [Hard Voting vs Soft Voting](#1-hard-voting-vs-soft-voting)
2. [scikit-learn Implementation](#2-scikit-learn-implementation)
3. [When to Use Voting Classifiers](#3-when-to-use-voting-classifiers)

---

## 1. Hard Voting vs Soft Voting

### 🟢 Beginner
**Simple Explanation**: Imagine you want to diagnose a medical scan. You ask a Logistic Regression model, a Support Vector Machine, and a Decision Tree. 
- **Hard Voting** is majority rules: if 2 models say "Cancer" and 1 says "No Cancer", the final decision is "Cancer".
- **Soft Voting** takes their confidence percentages: if they say "Cancer with 51% confidence", "No Cancer with 90% confidence", and "No Cancer with 80% confidence", the average probability points strongly to "No Cancer".

### 🟡 Intermediate
**Working Mechanisms**:
- **Hard Voting**: Each model casts exactly 1 vote. The class with the mode of predictions wins.
  
  $$\hat{y} = \text{mode}(h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_K(\mathbf{x}))$$

- **Soft Voting**: Averages the predicted probabilities for each class across all models and chooses the class with the highest average probability. This method gives more weight to highly confident predictions.
  
  $$\hat{y} = \arg\max_c \frac{1}{K}\sum_{k=1}^{K} P_k(y = c | \mathbf{x})$$

### 🔴 Advanced
**Weighted Voting**:
You can apply weights to individual models if you know some are more reliable than others:

$$\hat{y} = \arg\max_i \sum_{j=1}^m w_j p_{ij}$$

where $w_j$ is the weight of the $j$-th model, and $p_{ij}$ is the probability predicted by model $j$ for class $i$. These weights can be optimized using grid search or meta-learners.

---

## 2. scikit-learn Implementation

Here is how you can set up and compare Hard and Soft Voting Classifiers against their base models:

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score

# Load dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Define diverse base models
estimators = [
    ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42))
]

# Hard voting
hard_voting = VotingClassifier(estimators=estimators, voting='hard')
# Soft voting
soft_voting = VotingClassifier(estimators=estimators, voting='soft')

# Compare CV scores
print(f"{'Model':<30} {'CV Score':>10}")
print("-" * 42)
for name, model in estimators + [('Hard Voting', hard_voting), ('Soft Voting', soft_voting)]:
    scores = cross_val_score(model, data.data, data.target, cv=5, scoring='accuracy')
    print(f"{name:<30} {scores.mean():>10.4f} ± {scores.std():.4f}")
```

---

## 3. When to Use Voting Classifiers

- **Quick Baselines**: Great for combining models when you don't have the time to train a complex meta-learner (Stacking).
- **Diversity**: For voting to work effectively, base models must be **diverse** (e.g. mixing tree models, linear models, and distance-based classifiers) so they make different, uncorrelated errors. If they make the same errors, the voting classifier will not improve performance.

---

[← Extra Trees](./04-Extra-Trees.md) | [Back to Index](../README.md) | [Next: Boosting Introduction →](./06-Boosting-Introduction.md)
