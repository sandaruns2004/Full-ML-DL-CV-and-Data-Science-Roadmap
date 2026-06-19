# 📉 Knowledge Distillation

> **Prerequisites**: CNNs, Neural Network Foundations | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Problem: Massive Models](#1-the-problem-massive-models)
2. [What is Knowledge Distillation?](#2-what-is-knowledge-distillation)
3. [Soft Targets vs. Hard Targets](#3-soft-targets-vs-hard-targets)
4. [Temperature Scaling](#4-temperature-scaling)
5. [PyTorch Implementation](#5-pytorch-implementation)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The Problem: Massive Models

Modern Deep Learning models (like GPT-4, Vision Transformers, or massive ResNets) are incredibly accurate, but they are practically impossible to deploy on edge devices like mobile phones, IoT sensors, or cheap embedded hardware due to memory and compute constraints.

We need a way to take the performance of a massive model and compress it into a tiny model that can run fast and cheap.

---

## 2. What is Knowledge Distillation?

**Knowledge Distillation (KD)** is a model compression technique introduced by Geoffrey Hinton in 2015. 

The core idea is the **Teacher-Student paradigm**:
- **Teacher Model**: A large, cumbersome, highly accurate model (or ensemble of models) trained on the dataset.
- **Student Model**: A small, lightweight model trained to mimic the behavior of the Teacher model.

Instead of training the Student on the original labels (e.g., `Dog=1, Cat=0, Car=0`), we train the Student to output the exact same probability distribution as the Teacher model (e.g., `Dog=0.85, Cat=0.14, Car=0.01`).

---

## 3. Soft Targets vs. Hard Targets

Why not just train the small model on the original data? 

Because the Teacher model outputs **Soft Targets**.
- **Hard Target (Ground Truth)**: `[1.0, 0.0, 0.0]` (It is definitely a Dog).
- **Soft Target (Teacher Output)**: `[0.85, 0.14, 0.01]`

The Soft Target contains **dark knowledge**. It tells the Student model that a Dog looks *a little bit* like a Cat (0.14), but looks *nothing* like a Car (0.01). This rich similarity information helps the Student generalize much better than if it was only given the Hard Targets.

---

## 4. Temperature Scaling

Normally, a well-trained model will output a probability distribution very close to the hard targets (e.g., `[0.999, 0.001, 0.000]`). This hides the dark knowledge!

To reveal the dark knowledge, we apply a **Temperature ($T$)** to the Softmax function during distillation:

$$q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- **$T = 1$**: Standard Softmax.
- **$T > 1$**: "Softens" the distribution, making the hidden probabilities larger and easier for the Student to learn from.

Both the Teacher and Student use the same $T > 1$ during the distillation process. During inference, the Student uses $T = 1$.

The total loss is a combination of:
1. **Distillation Loss (KL Divergence)**: Difference between Student and Teacher soft targets.
2. **Student Loss (Cross-Entropy)**: Difference between Student output (T=1) and true hard targets.

---

## 5. PyTorch Implementation

Here is how you distill a large neural network into a small one in PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits, true_labels, T, alpha):
    """
    Computes the Knowledge Distillation loss.
    - alpha: Weight for distillation loss vs true label loss
    - T: Temperature
    """
    # 1. Distillation Loss (Soft Targets)
    # Note: KLDivLoss in PyTorch expects log probabilities for inputs, and probabilities for targets.
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_probs = F.log_softmax(student_logits / T, dim=1)
    
    # Multiply by T^2 to scale the gradients back to the same magnitude
    distil_loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean') * (T * T)
    
    # 2. Standard Loss (Hard Targets)
    student_loss = F.cross_entropy(student_logits, true_labels)
    
    # Combine losses
    total_loss = (alpha * distil_loss) + ((1 - alpha) * student_loss)
    return total_loss

# --- Example Training Loop ---
# teacher_model = ResNet50 (Pre-trained and frozen)
# student_model = MobileNetV2 (To be trained)

def train_student(teacher, student, dataloader, optimizer, T=3.0, alpha=0.5):
    teacher.eval() # Teacher is always in eval mode
    student.train()
    
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        
        # Forward passes
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        student_logits = student(inputs)
        
        # Calculate KD Loss
        loss = distillation_loss(student_logits, teacher_logits, labels, T, alpha)
        
        # Backward pass
        loss.backward()
        optimizer.step()
```

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Train a large MLP on MNIST (Teacher). Then train a tiny MLP (Student) using only cross-entropy. Finally, train the tiny MLP using Knowledge Distillation and compare the accuracies.
- 🟡 **Intermediate**: Distill a pre-trained ResNet-18 into a custom 3-layer CNN on the CIFAR-10 dataset.

### What's Next
| Next | Why |
|------|-----|
| [AB Testing](../14-DS-Advanced/01-AB-Testing.md) | Shift focus to Advanced Data Science topics. |

---

[← Time Series Deep Dive](./06-Time-Series-Deep-Dive.md) | [Back to Index](../README.md) | [Next: Semi Supervised Learning →](./08-Semi-Supervised-Learning.md)
