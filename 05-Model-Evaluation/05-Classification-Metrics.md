# 📊 Classification Metrics

> **Prerequisites**: Model Evaluation Concepts | **Difficulty**: ⭐⭐☆☆☆ Intermediate

---

## 📋 Table of Contents
1. [Beyond Accuracy](#1-beyond-accuracy)
2. [Precision, Recall, and F1-Score](#2-precision-recall-and-f1-score)
3. [When to use which metric?](#3-when-to-use-which-metric)

---

## 1. Beyond Accuracy

### 🟢 Beginner
**Simple Explanation**: 
Accuracy is simply "How many did I get right out of all my guesses?". 
However, Accuracy can be **very dangerous**. Imagine a disease that only affects 1 in 10,000 people. If I build a completely useless model that just says "NO ONE has the disease", my model will be 99.99% accurate! But it is completely useless for finding sick people.

### 🟡 Intermediate
**Practical Implementation**:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 0 = Healthy, 1 = Sick
y_true = [0, 0, 0, 0, 1, 1]
y_pred = [0, 0, 0, 1, 0, 1]

print(f"Accuracy: {accuracy_score(y_true, y_pred):.2f}")
print(f"Precision: {precision_score(y_true, y_pred):.2f}")
print(f"Recall: {recall_score(y_true, y_pred):.2f}")
print(f"F1 Score: {f1_score(y_true, y_pred):.2f}")
```

**Metrics Breakdown**:
- **Precision**: Out of all the people I *said* were sick, how many were *actually* sick? (Minimizes False Positives)
- **Recall (Sensitivity)**: Out of all the people who are *actually* sick, how many did I *find*? (Minimizes False Negatives)
- **Specificity**: Out of all the healthy people, how many did I correctly identify as healthy?

### 🔴 Advanced
**The F1-Score and Harmonic Means**:
The F1-Score is the Harmonic Mean of Precision and Recall. We use the harmonic mean instead of a simple average because it heavily punishes extreme values. If a model has a Recall of 1.0 but a Precision of 0.01, a simple average would be ~0.5. The harmonic mean forces the F1-score down to ~0.02, accurately reflecting that the model is performing poorly.

**Formula**:
$F_1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$

**Beta Parameter**: The generalized $F_\beta$ score allows you to weigh Recall higher than Precision (or vice versa). $F_2$ weighs recall twice as high as precision.
