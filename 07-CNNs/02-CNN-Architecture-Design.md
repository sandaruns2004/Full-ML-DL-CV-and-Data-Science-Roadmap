# 🏛️ CNN Architecture Design: VGG to ResNet

> **Prerequisites**: Convolution Mathematics | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Golden Rule of CNN Design](#1-the-golden-rule-of-cnn-design)
2. [LeNet-5 (1998): The Pioneer](#2-lenet-5-1998-the-pioneer)
3. [AlexNet (2012): The Deep Learning Boom](#3-alexnet-2012-the-deep-learning-boom)
4. [VGG-16 (2014): The Power of $3 \times 3$](#4-vgg-16-2014-the-power-of-3-times-3)
5. [Inception / GoogLeNet (2014): Going Wide](#5-inception--googlenet-2014-going-wide)
6. [ResNet (2015): The Residual Revolution](#6-resnet-2015-the-residual-revolution)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Golden Rule of CNN Design

As you build deeper architectures, you generally follow one strict pattern:
**As the spatial dimensions (Height $\times$ Width) decrease, the number of Channels must increase.**

When you apply a Pooling layer, you destroy 75% of the spatial information (a $2 \times 2$ pool reduces 4 pixels to 1). To prevent massive information loss, you typically double the number of filters (channels) in the subsequent Convolutional layer. You are trading spatial resolution for semantic depth.

---

## 2. LeNet-5 (1998): The Pioneer

Designed by Yann LeCun to recognize handwritten zip codes on mail.

**Architecture**:
- $32 \times 32$ Grayscale Input
- Conv ($5 \times 5$, 6 filters) $\rightarrow$ Average Pool
- Conv ($5 \times 5$, 16 filters) $\rightarrow$ Average Pool
- Fully Connected (120) $\rightarrow$ Fully Connected (84) $\rightarrow$ Output (10)

**Legacy**: Established the core pattern: `Conv -> Pool -> Conv -> Pool -> FC`.

### LeNet-5 PyTorch Implementation
Here is the complete from-scratch implementation of LeNet-5:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        # Input: 1 channel (Grayscale), 32x32 image
        
        # 1. First Convolutional Layer: 6 filters of size 5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        # 2. First Pooling Layer: 2x2 Average Pool
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 3. Second Convolutional Layer: 16 filters of size 5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # 4. Second Pooling Layer: 2x2 Average Pool
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Calculate flattened dimension: 
        # 32x32 -> Conv5x5 -> 28x28 -> Pool2x2 -> 14x14 -> Conv5x5 -> 10x10 -> Pool2x2 -> 5x5
        # 16 channels * 5 * 5 = 400
        
        # 5. Fully Connected Layers
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        # Apply Conv -> Tanh -> Pool
        x = self.pool1(torch.tanh(self.conv1(x)))
        x = self.pool2(torch.tanh(self.conv2(x)))
        
        # Flatten the spatial dimensions for the fully connected layers
        x = torch.flatten(x, 1) # Flatten starting from dimension 1 (keep batch dimension)
        
        # Apply FC -> Tanh
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        
        # Output layer (no activation here, CrossEntropyLoss handles Softmax)
        logits = self.fc3(x)
        return logits

# Example usage
model = LeNet5()
dummy_input = torch.randn(1, 1, 32, 32) # (Batch_Size, Channels, Height, Width)
output = model(dummy_input)
print(f"Output shape: {output.shape}") # Should be (1, 10)
```

---

## 3. AlexNet (2012): The Deep Learning Boom

Designed by Alex Krizhevsky, it crushed the ImageNet competition, sparking the modern AI revolution.

**Key Innovations**:
1. **ReLU Activation**: Replaced Tanh, massively speeding up training and reducing vanishing gradients.
2. **Dropout**: Used in the fully connected layers to prevent overfitting.
3. **GPU Training**: The network was so large (60M parameters) it had to be split across two 3GB GTX 580 GPUs.

*Note: AlexNet used large $11 \times 11$ and $5 \times 5$ filters in the early layers, which we now know is mathematically inefficient.*

---

## 4. VGG-16 (2014): The Power of $3 \times 3$

VGG (Visual Geometry Group at Oxford) standardized CNN design.

**The Insight**:
Instead of using large $7 \times 7$ filters, VGG used exclusively small $3 \times 3$ filters with `stride=1` and `padding='same'`.
As proven in [Convolution Mathematics](./01-Convolution-Mathematics.md), stacking three $3 \times 3$ filters yields the exact same receptive field as one $7 \times 7$ filter, but requires 45% fewer parameters and includes three ReLU activations instead of one, allowing it to learn more complex non-linear functions.

**VGG Block Design**:
```python
x = Conv2d(3x3, padding=1, out_channels=C)(x)
x = ReLU()(x)
x = Conv2d(3x3, padding=1, out_channels=C)(x)
x = ReLU()(x)
x = MaxPool2d(2x2, stride=2)(x) # Spatial dimensions halve. Next block will have 2*C channels.
```
VGG stacks these blocks repeatedly, growing from 64 $\rightarrow$ 128 $\rightarrow$ 256 $\rightarrow$ 512 channels.

---

## 5. Inception / GoogLeNet (2014): Going Wide

Instead of just going deep, Inception went *wide*.

**The Problem**:
Should you use a $1 \times 1$, $3 \times 3$, or $5 \times 5$ filter at a given layer? 
**The Inception Solution**: Use all of them simultaneously! 

An Inception Block passes the input through a $1 \times 1$, a $3 \times 3$, a $5 \times 5$, and a Max Pool, and then concatenates all their outputs together along the channel dimension. The network learns for itself which filter sizes are most useful.

### The Magic of $1 \times 1$ Convolutions
A $1 \times 1$ convolution sounds useless (it doesn't look at neighboring pixels). However, it looks across the *channels*. If you have a 512-channel input and apply sixty-four $1 \times 1$ filters, you compress the depth from 512 down to 64. 
Inception uses $1 \times 1$ convolutions as "bottlenecks" to massively reduce computation before applying the expensive $3 \times 3$ and $5 \times 5$ filters.

---

## 6. ResNet (2015): The Residual Revolution

Created by Kaiming He et al., ResNet won ImageNet by training an unheard-of 152-layer network.

**The Degradation Problem**:
Before ResNet, researchers noticed that a 56-layer VGG performed *worse* on both training and test sets than a 20-layer VGG. The gradients were vanishing. 

**The Residual Block**:
Instead of forcing the layers to learn the underlying mapping $H(x)$, ResNet forces them to learn the *residual* (the difference): $F(x) = H(x) - x$.
It does this by adding a **Skip Connection**:
$$Output = \text{ReLU}(F(x) + x)$$

If a layer is completely useless, the network can easily set the weights of $F(x)$ to $0$, and the block just becomes an Identity function ($Output = x$). The gradients flow flawlessly backwards through the "$+ x$" highway.

### PyTorch ResNet Block Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main Path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip Connection Path
        self.shortcut = nn.Sequential()
        # If stride != 1, the spatial dimensions shrink. We must shrink the skip connection too!
        # If channels change, we must use a 1x1 conv to match channels.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main Path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add Skip Connection
        out += self.shortcut(x)
        
        # Final Activation
        out = F.relu(out)
        return out
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **ResNet-18 from Scratch**: Look up the official ResNet-18 architecture diagram. Using the `ResidualBlock` defined above, write the full `ResNet18` class in PyTorch. It consists of an initial $7 \times 7$ conv, a MaxPool, and then 4 distinct sequential layers containing 2 Residual Blocks each.

### What's Next
| Next | Why |
|------|-----|
| [Modern CNN Architectures](./03-Modern-CNN-Architectures.md) | ResNet is from 2015. How do modern networks achieve better accuracy while using 10x less battery power on mobile phones? |

---

[← Convolution Mathematics](./01-Convolution-Mathematics.md) | [Back to Index](../README.md) | [Next: Modern CNN Architectures →](./03-Modern-CNN-Architectures.md)
