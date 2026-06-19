# ♻️ Transfer Learning & Fine-Tuning

> **Prerequisites**: CNN Architectures | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Philosophy of Transfer Learning](#1-the-philosophy-of-transfer-learning)
2. [Feature Extraction vs. Fine-Tuning](#2-feature-extraction-vs-fine-tuning)
3. [When to use which strategy?](#3-when-to-use-which-strategy)
4. [Implementation in PyTorch (`torchvision`)](#4-implementation-in-pytorch-torchvision)
5. [Catastrophic Forgetting & Learning Rates](#5-catastrophic-forgetting--learning-rates)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The Philosophy of Transfer Learning

Training a ResNet-50 from scratch requires over 1.2 million labeled images (ImageNet) and weeks of continuous GPU compute time. If you want to build an app to classify "Hotdog vs. Not Hotdog", you don't have 1.2 million hotdog images.

**Transfer Learning** solves this.
The early layers of a CNN learn generic, universal features:
- Layer 1: Edges, gradients, colors.
- Layer 2: Textures, corners, circles.
- Layer 3: Parts of objects (wheels, eyes, fur).

These features are universally applicable to *any* image task. A network trained to classify 1,000 different objects has already learned the perfect universal visual feature extractor. We can download these pre-trained weights and just swap out the final classification head!

---

## 2. Feature Extraction vs. Fine-Tuning

There are two primary ways to perform transfer learning.

### 2.1 Feature Extraction (Freezing)
We take the pre-trained network, remove the final fully-connected (Dense) layer, and replace it with a new one initialized randomly.
Crucially, we **freeze** the weights of all the convolutional layers. During backpropagation, only the weights of our brand new classification head are updated.

- **Pros**: Blazing fast training. Requires very little data. Impossible to overfit the conv layers.
- **Cons**: The conv layers are stuck looking for ImageNet features. If your task is radically different (e.g., medical X-Rays), the extracted features might be suboptimal.

### 2.2 Fine-Tuning (Unfreezing)
We start the same way: replace the final layer. However, we do **not** freeze the convolutional layers. We allow the entire network to continue training on our new dataset.

- **Pros**: Achieves maximum possible accuracy because the conv layers adapt specifically to your new data domain.
- **Cons**: Slow. Requires much more data to prevent overfitting. Extremely susceptible to **Catastrophic Forgetting** (destroying the pre-trained weights with large gradients).

---

## 3. When to use which strategy?

Your strategy depends on two factors: **Size of your dataset** and **Similarity to ImageNet**.

| Dataset Size | Similarity to ImageNet | Recommended Strategy |
|--------------|------------------------|----------------------|
| **Small** | **High** (e.g., dogs vs cats) | **Feature Extraction**. Fine-tuning will overfit. |
| **Small** | **Low** (e.g., cell microscopy) | **Feature Extraction** (mostly). Maybe fine-tune the top 1-2 conv layers very carefully. |
| **Large** | **High** | **Fine-Tuning**. The network will optimize perfectly for your specific classes. |
| **Large** | **Low** | **Fine-Tuning**. Since you have lots of data, you can safely reshape the entire network's understanding of visual features. |

---

## 4. Implementation in PyTorch (`torchvision`)

Let's do Transfer Learning using a pre-trained ResNet-18 to classify Ants vs. Bees.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

# 1. Download the pre-trained model
# (weights=models.ResNet18_Weights.DEFAULT pulls the best available ImageNet weights)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# ==========================================
# STRATEGY 1: Feature Extraction (Freeze)
# ==========================================
# Freeze all parameters in the network
for param in model.parameters():
    param.requires_grad = False

# ==========================================
# STRATEGY 2: Fine-Tuning (Unfreeze)
# ==========================================
# Do nothing! By default, requires_grad = True for all parameters.

# ==========================================
# Common Step: Replace the final head
# ==========================================
# ResNet's final layer is called 'fc' (fully connected).
num_ftrs = model.fc.in_features # For ResNet-18, this is 512

# We replace it with a new layer that outputs 2 classes (Ants, Bees).
# Newly created layers have requires_grad=True by default!
model.fc = nn.Linear(num_ftrs, 2)

# Verify what will be trained:
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Training: {name}")

# If we chose Strategy 1, the optimizer will only update model.fc.weight and model.fc.bias!
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Now, we just pass this into our standard training loop!
```

---

## 5. Catastrophic Forgetting & Learning Rates

If you decide to **Fine-Tune** the entire network, you must be extremely careful.

When training starts, your new classification head is randomly initialized. It will produce massive errors, sending massive gradients shooting backward through the network. If these massive gradients hit the delicate, perfectly tuned pre-trained convolutional layers, they will completely shatter the pre-trained weights. This is called **Catastrophic Forgetting**.

### Best Practices for Fine-Tuning:

1. **Warm-up (The Two-Step Method)**:
   - *Step 1*: Freeze the conv layers. Train *only* the new classification head for 5-10 epochs until it converges reasonably well.
   - *Step 2*: Unfreeze the conv layers. Now that the head isn't producing massive errors, train the entire network.

2. **Micro Learning Rates**:
   - When you unfreeze the network, you must drop the learning rate drastically (e.g., $1e-5$ or $1e-6$). We only want to gently nudge the convolutional weights, not rewrite them.

3. **Discriminative Learning Rates (Advanced)**:
   - Use different learning rates for different parts of the network. 
   - Early layers (edges/colors) get $1e-6$.
   - Middle layers (shapes) get $1e-5$.
   - The new custom head gets $1e-3$.

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Medical Image Transfer Learning**: Download a Kaggle dataset of Pneumonia Chest X-Rays. Use `torchvision.models.efficientnet_b0`. Try training it using Feature Extraction, and then try the Two-Step Fine-Tuning method. Compare the final validation accuracy of both approaches.

### What's Next
| Next | Why |
|------|-----|
| [Data Augmentation](./05-Data-Augmentation.md) | We've reduced our data requirements using Transfer Learning. Now let's artificially multiply our dataset size using Augmentations. |

---

[← Modern CNN Architectures](./03-Modern-CNN-Architectures.md) | [Back to Index](../README.md) | [Next: Data Augmentation →](./05-Data-Augmentation.md)
