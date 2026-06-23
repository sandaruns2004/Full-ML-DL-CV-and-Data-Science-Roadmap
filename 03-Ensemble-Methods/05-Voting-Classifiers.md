# 🗳️ Voting Classifiers

> **Difficulty**: ⭐☆☆☆☆ Beginner | **Prerequisites**: Supervised Learning Classifiers

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Core Mathematics](#3-core-mathematics)
4. [Scikit-Learn Implementation](#4-scikit-learn-implementation)
5. [Industry Applications](#5-industry-applications)

---

## 1. What Problem Does This Solve?

### 🟢 Beginner
You have trained multiple different models (like a Logistic Regression, a Support Vector Machine, and a Random Forest), but none of them are perfectly accurate on their own. Instead of picking just one and throwing the rest away, a Voting Classifier combines all of them together. It allows the models to vote on the final answer, leveraging the strengths of each model to produce a more reliable prediction.

### 🟡 Intermediate
Voting Classifiers are a straightforward ensemble technique to combine conceptually different machine learning classifiers. They are highly effective when the base models are **diverse**. If the base models make different, uncorrelated errors (e.g., a tree model vs a linear model), their combined voting will often outperform the best individual model in the group.

### 🔴 Advanced
A Voting Classifier is essentially a manual, unweighted (or statically weighted) ensemble. While not as sophisticated as Stacking (where a meta-learner learns the optimal combination weights), it serves as an extremely fast baseline for blending heterogeneous models, optimizing the overall bias-variance profile of the combined predictions.

---

## 2. Intuition

Imagine you want to diagnose a medical scan. You ask three different doctors.
- **Hard Voting** is majority rules: if 2 doctors say "Cancer" and 1 says "No Cancer", the final decision is "Cancer".
- **Soft Voting** takes their confidence percentages: if they say "Cancer with 51% confidence", "No Cancer with 90% confidence", and "No Cancer with 80% confidence", the average probability points strongly to "No Cancer". Soft voting gives more weight to highly confident predictions.

---

## 3. Core Mathematics

### Hard Voting
Each model casts exactly 1 vote. The class with the mode (most frequent) of predictions wins.
  
$$ \hat{y} = \text{mode}(h_1(\mathbf{x}), h_2(\mathbf{x}), \ldots, h_K(\mathbf{x})) $$

### Soft Voting
Averages the predicted probabilities for each class across all models and chooses the class with the highest average probability. 
  
$$ \hat{y} = \arg\max_c \frac{1}{K}\sum_{k=1}^{K} P_k(y = c | \mathbf{x}) $$

### Weighted Voting
You can apply weights to individual models if you know some are more reliable than others:

$$ \hat{y} = \arg\max_i \sum_{j=1}^m w_j p_{ij} $$

where $w_j$ is the weight of the $j$-th model, and $p_{ij}$ is the probability predicted by model $j$ for class $i$.

---

## 4. Scikit-Learn Implementation

Here is how you can set up and compare Hard and Soft Voting Classifiers:

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

## 5. Industry Applications

- **Quick Baselines**: Great for combining models when you don't have the time to train a complex meta-learner (Stacking).
- **Competitions**: Kaggle competitors often use soft-voting at the very end of a competition to average out the predictions of their top 3 or 4 completely different architectures (e.g. XGBoost + Neural Network + Random Forest).

---

[← Extra Trees (Extremely Randomized Trees)](04-Extra-Trees.md) | [Back to Index](../README.md) | [Next: Introduction to Boosting →](06-Boosting-Introduction.md)
