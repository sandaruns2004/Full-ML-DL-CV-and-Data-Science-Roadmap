# 🧰 Deep Learning Methods & Advanced Implementations

> **Prerequisites**: PyTorch Basics | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [Transfer Learning & Fine-Tuning (Math & Practice)](#1-transfer-learning--fine-tuning-math--practice)
2. [Advanced Data Augmentation (Mixup & CutMix)](#2-advanced-data-augmentation-mixup--cutmix)
3. [Writing Custom Loss Functions in PyTorch (Focal Loss)](#3-writing-custom-loss-functions-in-pytorch-focal-loss)
4. [Custom PyTorch Callbacks (Early Stopping)](#4-custom-pytorch-callbacks-early-stopping)
5. [Project Ideas & What's Next](#5-project-ideas--whats-next)

---

## 1. Transfer Learning & Fine-Tuning (Math & Practice)

Instead of initializing weights $W \sim \mathcal{N}(0, \sigma^2)$, we initialize them with weights pre-trained on a massive dataset (like ImageNet).

**Why it works mathematically:**
The early layers of a deep network learn universal feature extractors (Gabor filters, edge detectors). We want to reuse these filters and only retrain the final hyperplane (the classification head) to separate our specific classes.

### The PyTorch Implementation

```python
import torch
import torch.nn as nn
from torchvision import models

# 1. Load Pre-trained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Freeze all base layers (requires_grad = False)
# This means dW is not computed for these layers during backward()
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the final Fully Connected (fc) layer
num_ftrs = model.fc.in_features
# By default, newly created modules have requires_grad=True
model.fc = nn.Linear(num_ftrs, 2) # e.g., 2 classes: Cats vs Dogs

# 4. Optimizer: ONLY pass the parameters of the new fc layer!
import torch.optim as optim
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# To Fine-Tune later:
# Unfreeze the last block, lower the learning rate drastically (e.g., 1e-5), 
# and pass those parameters to a new optimizer.
```

---

## 2. Advanced Data Augmentation (Mixup & CutMix)

Standard augmentation (rotations, flips) does not change the label. **Mixup** and **CutMix** are modern techniques that fundamentally alter the training math by creating synthetic examples and interpolating labels.

### Mixup Formulation (Zhang et al., 2017)
Given two random training examples $(x_i, y_i)$ and $(x_j, y_j)$, we create a virtual training example:
$$\tilde{x} = \lambda x_i + (1-\lambda) x_j$$
$$\tilde{y} = \lambda y_i + (1-\lambda) y_j$$
Where $\lambda \sim \text{Beta}(\alpha, \alpha)$.

This forces the network to learn linear behavior *between* training examples, preventing overconfidence and massive overfitting.

```python
import numpy as np
import torch

def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Modified loss function for Mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Usage in training loop:
# inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
# outputs = model(inputs)
# loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
```

---

## 3. Writing Custom Loss Functions in PyTorch (Focal Loss)

**Focal Loss (Lin et al., 2017)** solves the extreme class imbalance problem (e.g., 1 anomaly for every 1000 normal items). Standard Cross-Entropy gets overwhelmed by the massive number of easy negatives.

Focal Loss adds a modulating factor $(1 - p_t)^\gamma$ to the Cross-Entropy loss.
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
- When an example is misclassified ($p_t$ is small), the modulating factor is near 1 (normal loss).
- When an example is correctly classified ($p_t \approx 1$), the factor goes to 0, down-weighting "easy" examples.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Compute standard BCE loss (without reduction)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-BCE_loss) # Because BCE = -log(pt), so exp(-BCE) = pt
        
        # Apply Focal formulation
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        return torch.mean(F_loss)

# Usage:
# criterion = FocalLoss(gamma=2.0)
# loss = criterion(outputs, targets)
```

---

## 4. Custom PyTorch Callbacks (Early Stopping)

Unlike Keras, PyTorch doesn't hide the training loop. You must implement logic like Early Stopping manually.

```python
class EarlyStopping:
    def __init__(self, patience=7, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # Save model when validation loss decreases
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Usage inside PyTorch Epoch Loop:
# early_stopping = EarlyStopping(patience=10)
# early_stopping(val_loss, model)
# if early_stopping.early_stop:
#     print("Early stopping triggered")
#     break
```

---

## 5. Project Ideas & What's Next

### Project Ideas
- 🟢 **Focal Loss vs BCE**: Create a heavily imbalanced synthetic dataset (99% class 0, 1% class 1). Train an MLP using standard BCE Loss, and another using Focal Loss. Compare their Precision-Recall curves.

### What's Next
| Next | Why |
|------|-----|
| [Advanced Deep Learning: CNNs](../07-CNNs/01-Convolution-Mathematics.md) | How do we apply these neural networks to images? We need convolutions. |

---

[← Frameworks Keras PyTorch](./08-Frameworks-Keras-PyTorch.md) | [Back to Index](../README.md) | [Next: Convolution Mathematics →](../07-CNNs/01-Convolution-Mathematics.md)
