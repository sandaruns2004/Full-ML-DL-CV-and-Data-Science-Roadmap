# 🌓 Semi-Supervised Learning

> **Prerequisites**: Supervised Learning, Unsupervised Learning | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Labeling Bottleneck](#1-the-labeling-bottleneck)
2. [What is Semi-Supervised Learning (SSL)?](#2-what-is-semi-supervised-learning-ssl)
3. [Pseudo-Labeling (Self-Training)](#3-pseudo-labeling-self-training)
4. [Consistency Regularization](#4-consistency-regularization)
5. [Graph-Based SSL](#5-graph-based-ssl)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The Labeling Bottleneck

In modern Machine Learning, getting **data** is easy. Getting **labeled data** is incredibly expensive.
- **Easy**: Scraping 1 million images from the internet.
- **Hard**: Paying human experts to draw precise bounding boxes around cancerous tumors in 1 million medical scans.

Supervised learning requires massive labeled datasets. Unsupervised learning requires no labels but rarely solves specific prediction tasks. How do we bridge the gap?

---

## 2. What is Semi-Supervised Learning (SSL)?

Semi-Supervised Learning (SSL) sits directly between Supervised and Unsupervised learning. It operates on a dataset containing a **small amount of labeled data** and a **massive amount of unlabeled data**.

The core assumption of SSL:
> **The unlabeled data contains underlying structural information about the data distribution that can help the model draw better decision boundaries.**

### The Cluster Assumption
SSL assumes that if points are in the same local cluster, they likely share the same label. Unlabeled data helps us discover where these clusters actually are.

### The Low-Density Separation Assumption
SSL assumes that the decision boundary between classes should pass through low-density regions (empty space). Unlabeled data shows us where the high-density and low-density regions are.

---

## 3. Pseudo-Labeling (Self-Training)

The simplest and most intuitive SSL approach is **Pseudo-Labeling** (also known as Self-Training).

### The Algorithm:
1. **Train**: Train a supervised model $M$ on the small labeled dataset $L$.
2. **Predict**: Use model $M$ to predict the labels for the unlabeled dataset $U$.
3. **Filter**: Select only the predictions where the model is highly confident (e.g., probability $> 0.95$). These are called **pseudo-labels**.
4. **Combine**: Add these high-confidence pseudo-labeled examples to the original training set $L$.
5. **Retrain**: Retrain the model on the expanded dataset.
6. **Repeat**: Loop until the model stops finding high-confidence predictions.

### Scikit-Learn Implementation
Scikit-Learn provides a built-in `SelfTrainingClassifier` that wraps around any standard estimator.

```python
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier

# X_labeled, y_labeled: Small labeled set
# X_unlabeled: Large unlabeled set. Labels must be set to -1 for sklearn to recognize them.

# Combine datasets
import numpy as np
X_train = np.vstack([X_labeled, X_unlabeled])
y_train = np.concatenate([y_labeled, [-1] * len(X_unlabeled)])

# Define the base model
base_model = RandomForestClassifier()

# Wrap it in the Self-Training module
# threshold=0.8 means only predictions with >80% confidence will be added as pseudo-labels
ssl_model = SelfTrainingClassifier(base_model, threshold=0.8)

# Fit on the combined data
ssl_model.fit(X_train, y_train)

# View how many unlabeled samples were successfully labeled
print(f"Labeled during training: {ssl_model.transduction_[ssl_model.transduction_ != -1].shape[0]}")
```

*Danger*: If the initial model makes confident mistakes, those mistakes get added to the training set and heavily reinforced in the next loop. This is known as **Confirmation Bias**.

---

## 4. Consistency Regularization

Consistency Regularization is the state-of-the-art approach for SSL in Deep Learning (used in papers like *FixMatch* and *UDA*).

### The Core Idea:
If we take an image of a dog and apply a slight augmentation (flip it, crop it, add noise), its semantic meaning doesn't change—it is still a dog. Therefore, the neural network should output the **exact same prediction** for the original image and the augmented image, even if we don't know the true label!

### The Algorithm (FixMatch style):
For a batch of **unlabeled** images:
1. Generate a **weak augmentation** (e.g., a simple horizontal flip) and a **strong augmentation** (e.g., extreme color distortion, cutout, noise).
2. Pass the weak augmentation through the network to get a prediction. If the network is highly confident ($>0.95$), generate a pseudo-label.
3. Pass the strong augmentation through the network.
4. **The Loss**: Calculate the Cross-Entropy loss between the network's prediction on the *strong* augmentation and the pseudo-label from the *weak* augmentation.

By forcing the network to make consistent predictions across extreme augmentations of unlabeled data, the network learns incredibly robust, invariant features.

---

## 5. Graph-Based SSL

If we represent our dataset as a graph (where each node is a data point, and edges connect data points that are mathematically close to each other), we can use **Label Propagation**.

### Label Spreading Algorithm
1. Construct a similarity graph where edge weights represent the distance between points (e.g., using RBF Kernel).
2. The few labeled nodes act as "sources" of color (e.g., Class A is Red, Class B is Blue).
3. The color "flows" or diffuses through the edges of the graph to the unlabeled nodes.
4. Nodes connected tightly to Red nodes will become Red. The flow stops when the network reaches an equilibrium.

```python
from sklearn.semi_supervised import LabelSpreading

# Uses an RBF kernel to measure similarity between points
label_prop_model = LabelSpreading(kernel='rbf', gamma=20, n_neighbors=7, alpha=0.2)

# Fit on the combined dataset (unlabeled points are marked with -1)
label_prop_model.fit(X_train, y_train)
```

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Pseudo-Labeling Playground**: Load the Digits dataset (`sklearn.datasets.load_digits`). Keep only 5% of the labels and set the rest to `-1`. Train a baseline Logistic Regression model on the 5%. Then, wrap it in a `SelfTrainingClassifier` and train it on the combined set. Compare the test accuracy of both!

### What's Next
| Next | Why |
|------|-----|
| [Multi-Task Learning](./09-Multi-Task-Learning.md) | We just learned how to learn with fewer labels. Next, we learn how to predict multiple different labels simultaneously using the same network. |

---

[← Knowledge Distillation](07-Knowledge-Distillation.md) | [Back to Index](../README.md) | [Next: Multi-Task Learning →](09-Multi-Task-Learning.md)
