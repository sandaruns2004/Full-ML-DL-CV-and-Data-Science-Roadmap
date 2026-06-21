# ⚖️ Imbalanced Classification Evaluation

> **Prerequisites**: Classification Metrics | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 📋 Table of Contents
1. [The Accuracy Paradox](#1-the-accuracy-paradox)
2. [Balanced Accuracy and Resampling](#2-balanced-accuracy-and-resampling)

---

## 1. The Accuracy Paradox

### 🟢 Beginner
**Simple Explanation**: 
If 99% of emails are normal and 1% are viruses, a model that simply predicts "Normal" every single time is 99% accurate. This is the **Accuracy Paradox**. The model looks amazing on paper, but it is completely failing at its job of finding viruses.

### 🟡 Intermediate
**Workflow and Metrics**: 
When evaluating imbalanced datasets, you **must** abandon standard Accuracy. 
- Use **Precision** and **Recall**. 
- Use **F1-Score** (or $F_\beta$ if you want to penalize False Negatives more than False Positives).
- Use **Balanced Accuracy**: This calculates the accuracy for each class individually, and then averages them. So a model that predicts "Normal" every time would have 100% accuracy on Normal, but 0% accuracy on Viruses, giving a Balanced Accuracy of 50%.

```python
from sklearn.metrics import balanced_accuracy_score

# 99 Normal, 1 Virus
y_true = [0]*99 + [1]
y_pred = [0]*100 # Model predicts Normal for everything

print(f"Accuracy: {accuracy_score(y_true, y_pred)}") # 0.99
print(f"Balanced Accuracy: {balanced_accuracy_score(y_true, y_pred)}") # 0.50
```

### 🔴 Advanced
**Class Weights vs Resampling Evaluation**:
When handling class imbalance, you can use SMOTE (Synthetic Minority Over-sampling Technique) during training. **CRITICAL WARNING**: Never use SMOTE on your Validation or Test sets! If you evaluate your model on SMOTE-generated data, you are testing your model's performance on fake, synthetic data, which completely breaks real-world generalization. You only oversample the `X_train` dataset.
Instead of SMOTE, modern industry systems usually prefer applying `class_weights='balanced'` in the model algorithm directly, which modifies the loss function to heavily penalize errors made on the minority class.
