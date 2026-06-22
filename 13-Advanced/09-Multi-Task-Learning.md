# 🤹 Multi-Task Learning (MTL)

> **Prerequisites**: Neural Networks | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [Why Learn Multiple Tasks at Once?](#1-why-learn-multiple-tasks-at-once)
2. [Hard Parameter Sharing](#2-hard-parameter-sharing)
3. [Soft Parameter Sharing](#3-soft-parameter-sharing)
4. [The Challenge: Loss Balancing](#4-the-challenge-loss-balancing)
5. [Project Ideas & What's Next](#5-project-ideas--whats-next)

---

## 1. Why Learn Multiple Tasks at Once?

Standard Machine Learning trains one model for one specific task. 
**Multi-Task Learning (MTL)** trains a single model to solve multiple tasks simultaneously.

**Why do this?**
1. **Regularization**: If a network is forced to learn features that are useful for multiple distinct tasks, it is much less likely to overfit to the noise of just one task. It learns more generalizable, robust representations.
2. **Computational Efficiency**: A self-driving car needs to detect pedestrians, read speed limit signs, and calculate lane boundaries. Running 3 separate ResNets at 60 FPS is computationally impossible. Running 1 ResNet backbone that splits into 3 lightweight heads is highly efficient.
3. **Data Amplification**: If Task A has 100,000 labels and Task B only has 1,000 labels, learning them together allows Task B to "borrow" the rich feature extraction capabilities learned from Task A's data.

---

## 2. Hard Parameter Sharing

This is the most common and simple form of MTL in deep learning.

**Architecture**:
The network has a shared "backbone" (e.g., a CNN for images or a Transformer for text) that processes the input. The output of this backbone is a generic feature vector. This vector is then fed into multiple distinct, task-specific "heads" (fully connected layers).

```text
       Input Image
            |
      [Shared CNN Backbone]
            |
       Shared Features
       /      |      \
 [Head 1]  [Head 2]  [Head 3]
    |         |         |
 Pedestrian  Sign      Lane
 Detection   Reading   Boundary
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torchvision.models as models

class MultiTaskCarVision(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Shared Backbone (ResNet18, removing the final classification layer)
        resnet = models.resnet18(pretrained=True)
        self.shared_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Task Specific Heads
        # Task 1: Binary Classification (Is there a pedestrian?)
        self.pedestrian_head = nn.Linear(512, 1)
        
        # Task 2: Multi-class Classification (Speed Limit Sign: 10, 20, 30... 100)
        self.sign_head = nn.Linear(512, 10)
        
        # Task 3: Regression (Distance to the car ahead in meters)
        self.distance_head = nn.Linear(512, 1)

    def forward(self, x):
        # Extract generic features
        features = self.shared_backbone(x)
        features = torch.flatten(features, 1)
        
        # Pass features through task-specific heads
        ped_pred = torch.sigmoid(self.pedestrian_head(features))
        sign_pred = self.sign_head(features)
        dist_pred = self.distance_head(features)
        
        return ped_pred, sign_pred, dist_pred
```

---

## 3. Soft Parameter Sharing

In Soft Parameter Sharing, every task has its own complete neural network with its own parameters. However, the models are trained together, and a regularization penalty is added to the loss function to encourage the parameters of the different models to be similar to each other.

*Example:* If Model A and Model B are learning similar tasks, their weights are penalized if the L2 distance between $W_A$ and $W_B$ becomes too large.

This is much less common than Hard Parameter Sharing because it does not provide the computational speedup of a shared backbone.

---

## 4. The Challenge: Loss Balancing

The hardest part of Multi-Task Learning is combining the losses.
If we just sum them: $L_{total} = L_{pedestrian} + L_{sign} + L_{distance}$, we run into massive problems:

1. **Scale Differences**: The distance regression loss (MSE) might be $50.0$, while the pedestrian classification loss (BCE) might be $0.3$. The gradients from the regression task will completely overwhelm the network, and it will ignore the pedestrian task.
2. **Gradient Conflict (Negative Transfer)**: Task A might push the shared weights in one direction, while Task B pushes them in the exact opposite direction. The shared backbone learns nothing useful, and performance drops below what single-task models would achieve.

### Solution: Dynamic Weighting
Instead of a simple sum, we weight the losses: 
$$L_{total} = w_1 L_1 + w_2 L_2 + w_3 L_3$$

Advanced MTL systems dynamically adjust $w_1, w_2, w_3$ during training:
- **Uncertainty Weighting**: Tasks with higher uncertainty (higher variance in loss) are given lower weights.
- **Gradient Normalization (GradNorm)**: The weights are adjusted dynamically so that the gradient magnitudes from all tasks are roughly equal when they flow back into the shared backbone.

---

## 5. Project Ideas & What's Next

### Project Ideas
- 🟢 **Age and Gender Prediction**: Find a dataset containing face images labeled with both Age (continuous) and Gender (binary). Build a Hard Parameter Sharing CNN that predicts both simultaneously. Experiment with manual loss weighting: `loss = 1.0 * gender_loss + 0.1 * age_loss` to balance the BCE and MSE scales.

### What's Next
| Next | Why |
|------|-----|
| [ML in Production Phase](../15-ML-In-Production/01-Model-Deployment-Strategies.md) | Now that you've mastered advanced neural network architectures, it's time to learn how to deploy them to the real world. |

---

[← Semi-Supervised Learning](08-Semi-Supervised-Learning.md) | [Back to Index](../README.md) | [Next: A/B Testing →](../14-DS-Advanced/01-AB-Testing.md)
