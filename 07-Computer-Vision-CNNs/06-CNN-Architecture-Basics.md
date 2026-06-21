# 🏛️ CNN Architecture Basics

---

## 📋 Table of Contents
1. [Beginner: The Hierarchical Visual Assembly](#1-beginner-the-hierarchical-visual-assembly)
2. [Intermediate: Simple PyTorch CNN Class](#2-intermediate-simple-pytorch-cnn-class)
3. [Advanced: Receptive Fields and Parameter Counting](#3-advanced-receptive-fields-and-parameter-counting)

---

## 1. Beginner: The Hierarchical Visual Assembly

### Simple Intuition
A single convolutional layer extracts simple, local features like vertical edges or colors. To recognize complex structures (like human faces or cars), we must stack multiple convolutional layers in sequence, followed by pooling layers to reduce dimension.
As the feature maps pass deeper into the network:
1. The spatial dimensions (Height & Width) decrease.
2. The semantic channels (representing different features) increase.

### Real-World Analogy: Identifying a House
- **First Layer (Simple Cells)**: Detects individual lines, orientation angles, and basic colors.
- **Second Layer (Composite Cells)**: Combines the lines to detect windows, doors, and walls.
- **Final Layers (Fully Connected)**: Combines windows, doors, and walls to decide: "This is a House."

---

## 2. Intermediate: Simple PyTorch CNN Class

Let us build a simple CNN class in PyTorch containing two convolution layers, pooling layers, and a final fully connected classification head.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Input: 3 channels (RGB), 32x32 image (e.g. CIFAR-10)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dimensions tracking: 
        # 32x32 -> conv1 -> 32x32 -> pool -> 16x16
        # 16x16 -> conv2 -> 16x16 -> pool -> 8x8
        # Flattened shape: 32 channels * 8 * 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # Flatten keeping batch dimension
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Example output shape
model = SimpleCNN()
x_batch = torch.randn(4, 3, 32, 32)
outputs = model(x_batch)
print("Input shape:", x_batch.shape)
print("Output logits shape:", outputs.shape) # Expected: (4, 10)
```

---

## 3. Advanced: Receptive Fields and Parameter Counting

### Calculating the Receptive Field (RF)
The Receptive Field is the spatial area of the original input image that influences a single neuron's activation at a given layer.
The formula for the receptive field of layer $l$, denoted by $r_l$, is:
$$r_l = r_{l-1} + (k_l - 1) \cdot j_{l-1}$$
where:
- $k_l$ is the kernel size of layer $l$.
- $j_{l-1}$ is the cumulative stride up to layer $l-1$, calculated as:
  $$j_l = j_{l-1} \cdot s_l \quad (\text{with } j_0 = 1)$$
- $s_l$ is the stride of layer $l$.

#### Stacking Convolutions vs. Large Kernels
Consider a stack of two $3 \times 3$ convolution layers (stride=1):
- Layer 1: $r_1 = 1 + (3 - 1) \cdot 1 = 3$
- Layer 2: $r_2 = 3 + (3 - 1) \cdot 1 = 5$

Two stacked $3 \times 3$ layers cover a receptive field of $5 \times 5$, but use fewer parameters:
- **Single $5 \times 5$ layer (C channels)**: $5 \times 5 \times C \times C = 25C^2$ parameters.
- **Two $3 \times 3$ layers (C channels)**: $2 \times (3 \times 3 \times C \times C) = 18C^2$ parameters.
This reduction in parameters, coupled with the extra activation function in between, is why deep learning favors stacked smaller filters over single large filters.

### Parameter Counting: FC vs. Conv
Let's compare the parameter counts of a Fully Connected layer and a Convolutional layer.

1. **Fully Connected Layer**:
   Connecting $M$ input channels to $N$ output channels:
   $$\text{Parameters} = M \times N + N \quad (\text{weights} + \text{biases})$$

2. **Convolutional Layer**:
   Connecting $M$ input channels to $N$ output channels with kernel size $K \times K$:
   $$\text{Parameters} = N \times (M \times K \times K) + N$$

Because $K$ is typically small (e.g. 3) and independent of image height or width, convolutional layers require exponentially fewer parameters, preventing overfitting.

---

[← Previous: Pooling Layers](./05-Pooling-Layers.md) | [Back to Index](./README.md) | [Next: Edge Detection And Feature Maps →](./07-Edge-Detection-And-Feature-Maps.md)
