# 🏛️ VGG-Net

---

## 📋 Table of Contents
1. [Beginner: The Power of Small Filters](#1-beginner-the-power-of-small-filters)
2. [Intermediate: PyTorch Implementation of VGG-16](#2-intermediate-pytorch-implementation-of-vgg-16)
3. [Advanced: Receptive Field Equivalence and Parameter Comparison](#3-advanced-receptive-field-equivalence-and-parameter-comparison)

---

## 1. Beginner: The Power of Small Filters

### Simple Intuition
Older architectures (like AlexNet) used large filters (like $11 \times 11$ or $5 \times 5$) in early layers. The creators of **VGG-Net** (Visual Geometry Group, Oxford, 2014) realized that using a stack of multiple small $3 \times 3$ filters was far better than using a single large filter. 

VGG standardized CNN design by stacking simple blocks:
`[Conv 3x3] -> [ReLU] -> [Conv 3x3] -> [ReLU] -> [MaxPool 2x2]`

### Real-World Analogy: Building with Lego
Imagine trying to build a curved arch:
- If you use a single large, rigid block (large filter), the arch will be blocky and simple.
- If you stack multiple small, interlocking Lego bricks (small filters), you can construct a smoother, more detailed curve. Stacking small filters allows the network to learn more complex shapes with fewer parameters.

---

## 2. Intermediate: PyTorch Implementation of VGG-16

Let us implement VGG-16 in PyTorch. VGG-16 consists of 13 convolutional layers and 3 fully connected layers.

```python
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Adaptive pooling forces spatial dim to 7x7 regardless of input size
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Verify architecture output shape
vgg = VGG16()
dummy_input = torch.randn(1, 3, 224, 224)
out = vgg(dummy_input)
print("VGG-16 Output Shape:", out.shape) # Expected: (1, 1000)
```

---

## 3. Advanced: Receptive Field Equivalence and Parameter Comparison

Let us mathematically prove the equivalence of stacking three $3 \times 3$ convolution layers instead of a single $7 \times 7$ layer.

### Receptive Field Proof
Recall the receptive field formula for stride=1:
$$r_l = r_{l-1} + (k_l - 1)$$

Let the base receptive field be $r_0 = 1$ (raw input pixels).
1. After the first $3 \times 3$ layer:
   $$r_1 = 1 + (3 - 1) = 3$$
2. After the second $3 \times 3$ layer:
   $$r_2 = 3 + (3 - 1) = 5$$
3. After the third $3 \times 3$ layer:
   $$r_3 = 5 + (3 - 1) = 7$$

Thus, three stacked $3 \times 3$ layers cover a receptive field of $7 \times 7$.

### Parameter Count Savings
Let $C$ be the number of input and output channels.
- **Single $7 \times 7$ layer parameters**:
  $$\text{Params} = 7 \times 7 \times C \times C = 49C^2$$
- **Three $3 \times 3$ layers parameters**:
  $$\text{Params} = 3 \times (3 \times 3 \times C \times C) = 27C^2$$

### Advantages:
1. **Parameter Reduction**: Stacking three $3 \times 3$ layers reduces the parameter count by:
   $$\frac{49C^2 - 27C^2}{49C^2} \approx 45\%$$
   This makes the model much less prone to overfitting and faster to train.
2. **More Non-linearities**: Each layer is followed by a ReLU activation function. Stacking three layers gives the network three non-linear decision boundaries instead of one, allowing it to learn more complex representations.

---

[← Previous: LeNet And AlexNet](./09-LeNet-And-AlexNet.md) | [Back to Index](./README.md) | [Next: Modern CNN Architectures →](./11-Modern-CNN-Architectures.md)
