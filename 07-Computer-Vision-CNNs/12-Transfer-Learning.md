# ♻️ Transfer Learning

---

## 📋 Table of Contents
1. [Beginner: Standing on the Shoulders of Visual Giants](#1-beginner-standing-on-the-shoulders-of-visual-giants)
2. [Intermediate: PyTorch Implementation of Transfer Learning](#2-intermediate-pytorch-implementation-of-transfer-learning)
3. [Advanced: Catastrophic Forgetting & Discriminative Learning Rates](#3-advanced-catastrophic-forgetting--discriminative-learning-rates)

---

## 1. Beginner: Standing on the Shoulders of Visual Giants

### Simple Intuition
Training a state-of-the-art CNN from scratch requires millions of labeled images and weeks of GPU computing time. **Transfer Learning** is the process of taking a neural network that has already been trained on a massive dataset (like ImageNet, which contains 1.2 million images of 1,000 different objects) and adapting it to solve a new, custom task. 
Instead of starting from scratch, we reuse the generic visual features (edges, textures, shapes) that the model has already learned.

### Real-World Analogy: Learning to Drive a Truck
If you already know how to drive a sedan, you do not start learning to drive a cargo truck from scratch. You already know:
- How to steer the wheel.
- How to use the brakes and gas pedals.
- How to read road signs and follow traffic laws.
You only need to adapt your existing driving skills (transfer learning) to handle the truck's larger size and weight.

---

## 2. Intermediate: PyTorch Implementation of Transfer Learning

We will write a PyTorch script demonstrating the two primary transfer learning strategies using a pre-trained ResNet-18 model:
1. **Feature Extraction**: Freezing the convolutional layers and training only a new classification head.
2. **Fine-Tuning**: Continuing to train the entire network with a small learning rate.

```python
import torch
import torch.nn as nn
from torchvision import models

# Load ResNet-18 with default pre-trained ImageNet weights
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# ====================================================
# Strategy 1: Feature Extraction (Freeze early layers)
# ====================================================
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected classification head
# ResNet-18 has a final layer named 'fc' which maps 512 features to 1000 classes
num_features = model.fc.in_features
# We replace it with a new layer for our custom task (e.g. 2 classes: Ants vs Bees)
# The new layer has requires_grad=True by default!
model.fc = nn.Linear(num_features, 2)

# Verify only the final fc layer will be trained
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable layer: {name}")
```

---

## 3. Advanced: Catastrophic Forgetting & Discriminative Learning Rates

### Catastrophic Forgetting
When we replace the pre-trained classification head with a randomly initialized layer, the new head will output large errors during the first few epochs. 
If we do not freeze the early layers, the backpropagated gradients will be very large. These massive gradients can destroy the delicate, pre-trained weights in the convolutional layers. This is called **Catastrophic Forgetting** (or network degradation).

### Mitigation Best Practices

#### 1. Two-Step Warm-up:
- **Step 1**: Freeze the pre-trained layers. Train *only* the new classification head for 3-5 epochs until its weights stabilize.
- **Step 2**: Unfreeze the entire network and train with a very small learning rate ($\approx 10^{-5}$).

#### 2. Discriminative Learning Rates
Instead of using a single learning rate for the entire network, we assign smaller learning rates to the early layers (which detect generic features like edges) and larger learning rates to the deeper layers and head (which detect task-specific shapes).

In PyTorch, we implement this by passing list dictionaries to the optimizer:

```python
import torch.optim as optim

# Split parameters into groups
head_params = model.fc.parameters()
base_params = [param for name, param in model.named_parameters() if 'fc' not in name]

# Optimizer with different learning rates
optimizer = optim.AdamW([
    {'params': base_params, 'lr': 1e-5}, # Slow updates for visual features
    {'params': head_params, 'lr': 1e-3}  # Fast updates for new classification head
], weight_decay=1e-4)
```

---

[← Previous: Modern CNN Architectures](./11-Modern-CNN-Architectures.md) | [Back to Index](./README.md) | [Next: Data Augmentation →](./13-Data-Augmentation.md)
