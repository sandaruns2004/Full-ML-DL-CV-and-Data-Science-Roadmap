# ⚖️ Data Ethics and Fairness in Machine Learning

> **Prerequisites**: Machine Learning | **Difficulty**: ⭐⭐☆☆☆ Elementary

---

## 📋 Table of Contents
1. [Types of Bias in Machine Learning](#1-types-of-bias-in-machine-learning)
2. [Mathematical Metrics for Fairness](#2-mathematical-metrics-for-fairness)
3. [Bias Mitigation Strategies](#3-bias-mitigation-strategies)
4. [Implementation Example (AIF360)](#4-implementation-example-aif360)
5. [Summary](#summary)

---

As Machine Learning models are increasingly deployed in high-stakes domains like criminal justice, hiring, lending, and healthcare, the consequences of biased algorithms become severe. An algorithm that achieves 95% accuracy overall might still systematically discriminate against a specific demographic.

**Fairness** in ML is not just a social or philosophical issue; it is a mathematical and engineering problem that requires specific metrics and mitigation strategies.

---

## 1. Types of Bias in Machine Learning

Bias does not usually come from a malicious engineer; it enters the system through the data or the formulation of the problem.

*   **Historical Bias**: The data perfectly reflects the world, but the world is historically biased (e.g., a resume screening tool trained on past data will prefer men for engineering roles because historically more men held those roles).
*   **Representation Bias**: The training data is not representative of the population (e.g., facial recognition trained mostly on light-skinned faces will perform poorly on dark-skinned faces).
*   **Measurement Bias**: The proxy used to measure a target is flawed. (e.g., predicting "healthcare needs" by predicting "healthcare costs". Because minority populations historically spent less on healthcare due to systemic issues, the algorithm wrongly concludes they are healthier and need less care).
*   **Aggregation Bias**: A one-size-fits-all model is applied to groups with different underlying distributions (e.g., HbA1c levels for diagnosing diabetes differ naturally across ethnicities, but a single threshold is used).

---

## 2. Mathematical Metrics for Fairness

There is no single mathematical definition of "fairness." In fact, it has been mathematically proven that several common definitions of fairness are mutually exclusive—you cannot satisfy them all simultaneously if the base rates differ between groups.

Let $Y$ be the true label, $\hat{Y}$ be the model's prediction, and $A$ be a protected attribute (e.g., race, gender).

### 2.1 Demographic Parity (Statistical Parity)
**Concept**: The likelihood of a positive outcome should be the same regardless of the protected attribute.
$$ P(\hat{Y}=1 | A=0) = P(\hat{Y}=1 | A=1) $$
*   **Pros**: Forces equal outcomes.
*   **Cons**: Ignores the actual true label $Y$. If group A actually has a higher rate of defaulting on loans, forcing Demographic Parity might require granting loans to unqualified individuals in group B.

### 2.2 Equal Opportunity (True Positive Rate Parity)
**Concept**: If an individual actually deserves the positive outcome ($Y=1$), they should have the same probability of getting it regardless of their group.
$$ P(\hat{Y}=1 | A=0, Y=1) = P(\hat{Y}=1 | A=1, Y=1) $$
*   **Pros**: Focuses on fairness for the qualified individuals. Often legally required.
*   **Cons**: Does not consider the False Positive Rate (giving loans to unqualified people).

### 2.3 Equalized Odds
**Concept**: Both the True Positive Rate and the False Positive Rate must be identical across groups.
$$ P(\hat{Y}=1 | A=0, Y=y) = P(\hat{Y}=1 | A=1, Y=y) \quad \text{for } y \in \{0, 1\} $$

### 2.4 Disparate Impact (The 80% Rule)
A legal concept used in US employment law. It states that the selection rate for any race, sex, or ethnic group must not be less than 4/5ths (80%) of the rate for the group with the highest rate.
$$ \frac{P(\hat{Y}=1 | \text{Unprivileged Group})}{P(\hat{Y}=1 | \text{Privileged Group})} \geq 0.8 $$

---

## 3. Bias Mitigation Strategies

Mitigation can happen at three stages in the ML pipeline.

### 3.1 Pre-Processing (Fixing the Data)
Modifying the training data before the model is trained.
*   **Reweighting**: Assign different weights to training examples to simulate a balanced dataset. (e.g., give a higher weight to successful minority applicants).
*   **Optimized Pre-processing**: Mathematically transform the feature space so that it becomes independent of the protected attribute $A$, while preserving as much information as possible about $Y$.
*   **Removing Proxies**: Simply removing "Race" from the dataset doesn't work. Models will learn to infer race from highly correlated proxy variables like "Zip Code." These proxies must be identified and neutralized.

### 3.2 In-Processing (Fixing the Model)
Modifying the learning algorithm to penalize unfairness during training.
*   **Adversarial Debiasing**: Train two models simultaneously. Model 1 tries to predict the target $Y$. Model 2 (the adversary) looks at Model 1's predictions and tries to guess the protected attribute $A$. Model 1's loss function includes a penalty if Model 2 is successful. This forces Model 1 to make predictions that reveal no information about the protected attribute.
*   **Regularization**: Add a fairness constraint penalty to the standard loss function.

### 3.3 Post-Processing (Fixing the Predictions)
Modifying the output of the trained model.
*   **Threshold Optimization**: A standard classifier uses a 0.5 threshold for all groups. Post-processing can find different thresholds for different groups to achieve Equal Opportunity or Equalized Odds. (e.g., threshold of 0.55 for Group A and 0.45 for Group B).

---

## 4. Implementation Example (AIF360)

IBM's **AI Fairness 360 (AIF360)** is the standard library for evaluating and mitigating bias.

```python
# Conceptual example using AIF360
from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# 1. Load data with defined privileged/unprivileged groups
dataset = AdultDataset()
privileged_groups = [{'sex': 1}] # e.g., Male
unprivileged_groups = [{'sex': 0}] # e.g., Female

# 2. Evaluate initial bias
# (Assume model_predictions is a dataset object containing the model's output)
metric = ClassificationMetric(dataset, model_predictions, 
                              unprivileged_groups=unprivileged_groups,
                              privileged_groups=privileged_groups)

print(f"Disparate Impact: {metric.disparate_impact()}") 
# A value < 0.8 indicates bias against the unprivileged group

# 3. Mitigate bias using Reweighing (Pre-processing)
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

# Transform the dataset to have new sample weights that balance the groups
dataset_transf = RW.fit_transform(dataset)

# You would now train your model using dataset_transf.instance_weights
```

## Summary
Building fair machine learning systems requires active intervention. You cannot rely on "blind" algorithms that simply ignore protected attributes. Evaluating fairness metrics (Demographic Parity, Equal Opportunity) and applying mitigation strategies (Reweighting, Adversarial Debiasing) must become a standard part of the MLOps pipeline.

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Load a dataset, train a Logistic Regression model, and calculate the Disparate Impact metric for a protected group.
- 🟡 **Intermediate**: Use the AIF360 library to apply the Reweighing algorithm and compare the model's fairness before and after mitigation.

### What's Next
| Next | Why |
|------|-----|
| [ML Pipeline](../15-ML-In-Production/01-ML-Pipeline.md) | Learn how to deploy models into production correctly. |

---

[← Big Data & Distributed ML](05-Big-Data-And-Distributed-ML.md) | [Back to Index](../README.md) | [Next: ML Pipeline →](../15-ML-In-Production/01-ML-Pipeline.md)
