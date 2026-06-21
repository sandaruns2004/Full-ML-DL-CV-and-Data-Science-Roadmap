# 🔷 Pooling Layers

---

## 📋 Table of Contents
1. [Beginner: Zooming Out for the Big Picture](#1-beginner-zooming-out-for-the-big-picture)
2. [Intermediate: Pooling in PyTorch](#2-intermediate-pooling-in-pytorch)
3. [Advanced: Backpropagation Gradients Routing through Pooling](#3-advanced-backpropagation-gradients-routing-through-pooling)

---

## 1. Beginner: Zooming Out for the Big Picture

### Simple Intuition
As we build deeper networks, we want to look at larger regions of the image. **Pooling** layers are used to downsample the spatial dimensions (Height and Width) of the feature maps, reducing the computational cost and memory footprint.
- **Max Pooling**: Looks at a small patch (usually $2 \times 2$) and outputs only the maximum value.
- **Average Pooling**: Computes the average value of all pixels in the patch.

### Real-World Analogy: Squinting Your Eyes
Imagine looking at a mosaic painting:
- Up close, you see individual tiles.
- If you step back or squint your eyes, the fine details disappear, but the large shapes and structures (like trees or houses) become clearer. 
Pooling works the same way: it discards tiny pixel-level details and preserves the most important features, introducing **translation invariance** (meaning if an edge shifts by a pixel or two, the max pooling output remains the same).

---

## 2. Intermediate: Pooling in PyTorch

Let us test Max Pooling and Average Pooling in PyTorch.

```python
import torch
import torch.nn as nn

# Create a mock feature map: 1 image, 1 channel, 4x4 matrix
x = torch.tensor([[
    [1.0, 2.0, 3.0, 0.0],
    [5.0, 8.0, 2.0, 1.0],
    [0.0, 2.0, 4.0, 2.0],
    [3.0, 1.0, 1.0, 5.0]
]]).unsqueeze(0) # Shape: (1, 1, 4, 4)

# 1. Max Pooling (2x2, stride 2)
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
out_max = max_pool(x)
print("Max Pool Output:\n", out_max) # Expected Max in each 2x2 quadrant: [[8, 3], [3, 5]]

# 2. Average Pooling (2x2, stride 2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
out_avg = avg_pool(x)
print("Average Pool Output:\n", out_avg)
```

---

## 3. Advanced: Backpropagation Gradients Routing through Pooling

Since pooling layers do not have learnable weights, they do not update parameters. However, they must propagate the upstream loss gradients backward to the previous layer.

### 1. Max Pooling Backward Pass (Routing)
During the forward pass of Max Pooling, the layer must record the index of the maximum value (the **switches** or mask) in each window.
During the backward pass:
- The incoming gradient from the downstream layer is routed **entirely** to the input index that had the maximum value during the forward pass.
- All other inputs in the window receive a gradient of $0$.

$$\frac{\partial \mathcal{L}}{\partial x_i} = \begin{cases} \frac{\partial \mathcal{L}}{\partial y} & \text{if } x_i = \max(X) \\ 0 & \text{otherwise} \end{cases}$$

### 2. Average Pooling Backward Pass (Distribution)
During the forward pass, Average Pooling computes the mean of the inputs in a window of size $N$ (e.g. $N=4$ for a $2 \times 2$ pool):
$$y = \frac{1}{N} \sum_{i=1}^{N} x_i$$

During the backward pass:
- The incoming gradient $\frac{\partial \mathcal{L}}{\partial y}$ is divided equally and distributed to all inputs in the window:
$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{N} \frac{\partial \mathcal{L}}{\partial y}$$

---

[← Previous: Padding And Strides](./04-Padding-And-Strides.md) | [Back to Index](./README.md) | [Next: CNN Architecture Basics →](./06-CNN-Architecture-Basics.md)
