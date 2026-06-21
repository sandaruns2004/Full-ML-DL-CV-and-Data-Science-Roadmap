# 🏛️ Modern CNN Architectures

---

## 📋 Table of Contents
1. [Beginner: The Gradient Highways (Skip Connections)](#1-beginner-the-gradient-highways-skip-connections)
2. [Intermediate: PyTorch ResNet Block Implementation](#2-intermediate-pytorch-resnet-block-implementation)
3. [Advanced: Dense Connectivity and Compound Scaling Laws](#3-advanced-dense-connectivity-and-compound-scaling-laws)

---

## 1. Beginner: The Gradient Highways (Skip Connections)

### Simple Intuition
As neural networks grow deeper (e.g. 50 or 100 layers), they start performing *worse*. This is because gradients shrink to zero as they travel backward through dozens of layers (vanishing gradients). 
**ResNet (Residual Networks)** solved this by adding **Skip Connections** (or residual connections). These connections allow the signal to bypass some layers, creating a shortcut for both the forward information flow and the backward gradients.

### Real-World Analogy: The Subway Express
Imagine walking down a long, congested street with 100 checkpoints (VGG layers):
- At each checkpoint, you get delayed and lose energy. By the end, you are exhausted.
- Now imagine there is an express train running parallel to the street (Skip Connection). You can hop on the train, skip the congested checkpoints, and arrive at the end of the street full of energy. The skip connection acts as an express lane, keeping the gradient signal strong.

---

## 2. Intermediate: PyTorch ResNet Block Implementation

Let us write a standard Residual Block in PyTorch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # Conv 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv 2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        # If input and output shapes mismatch due to stride or channel changes
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Add the input shortcut before the final ReLU activation
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Quick test
block = ResNetBasicBlock(in_channels=64, out_channels=128, stride=2)
x_in = torch.randn(1, 64, 32, 32)
out_block = block(x_in)
print("Input shape:", x_in.shape)
print("Output shape:", out_block.shape) # Expected: (1, 128, 16, 16)
```

---

## 3. Advanced: Dense Connectivity and Compound Scaling Laws

### DenseNet (Dense Connectivity)
Rather than adding the input to the output ($F(x) + x$) like ResNet, **DenseNet** concatenates the feature maps:
$$x_l = H_l([x_0, x_1, \dots, x_{l-1}])$$
In a DenseNet block, every layer receives all preceding feature maps as inputs. This maximizes feature reuse and improves gradient flow, allowing the network to train with fewer channels (called a low **growth rate**).

### EfficientNet and Compound Scaling
When researchers want to improve a network's accuracy, they traditionally scale one of three dimensions:
1. **Depth ($d$)**: Make the network deeper (e.g. ResNet-50 to ResNet-101).
2. **Width ($w$)**: Increase the number of channels/filters.
3. **Resolution ($r$)**: Train on larger input images.

In 2019, Tan and Le (Google) proved that scaling these dimensions arbitrarily leads to diminishing returns. They proposed **Compound Scaling**, which scales all three dimensions together using a fixed compound coefficient $\phi$:
$$\text{Depth: } d = \alpha^\phi$$
$$\text{Width: } w = \beta^\phi$$
$$\text{Resolution: } r = \gamma^\phi$$
$$\text{subject to } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \quad \text{and } \alpha, \beta, \gamma \ge 1$$

Using this method, **EfficientNet** achieved state-of-the-art accuracy on ImageNet while using 8.4x fewer parameters and running 6.1x faster than existing models.

---

[← Previous: VGG-Net](./10-VGG-Net.md) | [Back to Index](./README.md) | [Next: Transfer Learning →](./12-Transfer-Learning.md)
