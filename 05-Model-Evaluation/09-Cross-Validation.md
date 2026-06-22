# 🔀 Cross Validation

> **Prerequisites**: Train/Test Split | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Flaw in Train/Test Split](#1-the-flaw-in-train-test-split)
2. [K-Fold Cross Validation](#2-k-fold-cross-validation)
3. [Stratified K-Fold and LOOCV](#3-stratified-k-fold-and-loocv)

---

## 1. The Flaw in Train/Test Split

### 🟢 Beginner
**Simple Explanation**: 
If you randomly split a deck of cards into "Train" and "Test", you might accidentally put all the Aces into the Test pile. Your model would never learn what an Ace looks like, and it would fail the test. 
To fix this, we shuffle the deck, split it into 5 piles, and take turns using each pile as the "Test" pile while learning from the other 4. This is called **Cross Validation**.

### 🟡 Intermediate
**K-Fold Workflow**: 
1. Divide data into $K$ equal subsets (folds).
2. For $i=1$ to $K$:
   - Train the model on $K-1$ folds.
   - Evaluate the model on fold $i$.
3. Average the $K$ evaluation scores.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
# cv=5 means 5-fold cross validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Scores for each fold: {scores}")
print(f"Average Accuracy: {scores.mean():.2f} +/- {scores.std():.2f}")
```

### 🔴 Advanced
**Advanced Variations**:
- **Stratified K-Fold**: Ensures that the ratio of classes (e.g., 90% dogs, 10% cats) is exactly preserved in every single fold. Mandatory for imbalanced classification.
- **Leave-One-Out (LOOCV)**: $K$ is set to the total number of data points $N$. You train on $N-1$ points and test on 1 point. Extremely computationally expensive, but theoretically unbiased.
- **Time Series Split**: You cannot use K-Fold on Time Series data because you would be predicting the past using the future. You must use expanding window splits.

---

[← Precision-Recall Curves](08-Precision-Recall-Curves.md) | [Back to Index](../README.md) | [Next: Learning Curves →](10-Learning-Curves.md)
