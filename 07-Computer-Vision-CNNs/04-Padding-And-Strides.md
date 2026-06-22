# 🔷 Padding and Strides

---

## 📋 Table of Contents
1. [Beginner: Borders and Step Sizes](#1-beginner-borders-and-step-sizes)
2. [Intermediate: Dimension Calculations in PyTorch](#2-intermediate-dimension-calculations-in-pytorch)
3. [Advanced: Derivation of the Spatial Output Formula](#3-advanced-derivation-of-the-spatial-output-formula)

---

## 1. Beginner: Borders and Step Sizes

### Simple Intuition
- **The Border Problem**: Every time we pass an image through a convolution layer, the output gets slightly smaller. The pixels on the border are only touched once, while center pixels are touched multiple times. Information at the borders is lost.
- **Padding**: To prevent the image from shrinking and keep border information, we add a border of extra pixels (usually zeros) around the outside of the image.
  - **Valid Padding**: No padding. The image shrinks.
  - **Same Padding**: Add enough padding so the output shape matches the input shape.
- **Stride**: The step size of the sliding window. A stride of 1 means moving one pixel at a time. A stride of 2 means jumping two pixels at a time, which cuts the output width and height roughly in half.

### Real-World Analogy: Reading a Newspaper
- **Padding**: Imagine a newspaper page with a wide white margin around the edge of the text. This margin ensures you don't miss words printed near the edge of the paper.
- **Stride**: Imagine reading a text but skipping every second word (stride 2) to read through the page faster.

---

## 2. Intermediate: Dimension Calculations in PyTorch

Let us write a script to compute the output shape of a convolution layer in PyTorch given different padding and stride parameters.

```python
import torch
import torch.nn as nn

# Input: Batch size 1, 3 channels, 32x32 image (e.g. CIFAR-10)
x = torch.randn(1, 3, 32, 32)

# 1. Valid Padding (p=0), Stride=1
conv_valid = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)
out_valid = conv_valid(x)
print("Valid Padding (p=0) output shape:", out_valid.shape) # Expected: (1, 16, 30, 30)

# 2. Same Padding (p=1), Stride=1
conv_same = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
out_same = conv_same(x)
print("Same Padding (p=1) output shape:", out_same.shape)   # Expected: (1, 16, 32, 32)

# 3. Same Padding, Stride=2
conv_strided = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
out_strided = conv_strided(x)
print("Strided (s=2) output shape:", out_strided.shape)       # Expected: (1, 16, 16, 16)
```

---

## 3. Advanced: Derivation of the Spatial Output Formula

Let us derive the formula for the spatial output size ($W_{\text{out}}$) of a convolution layer.

Given:
- Input dimension width: $W$
- Kernel filter width: $K$
- Padding on both sides: $P$
- Stride: $S$

### Derivation Steps:
1. With padding $P$ applied to both sides of the input, the effective width of the padded input becomes:
   $$W_{\text{padded}} = W + 2P$$
2. A kernel of width $K$ is placed at the leftmost position. The remaining space to slide the kernel is:
   $$\text{Remaining Space} = (W + 2P) - K$$
3. The kernel slides across the remaining space in steps of size $S$. The number of steps the kernel can take is:
   $$\text{Number of Steps} = \frac{(W + 2P - K)}{S}$$
4. Since the kernel must fit entirely within the bounds of the padded input, we take the floor of this division ($\lfloor \cdot \rfloor$) to handle cases where the kernel does not align perfectly at the right edge:
   $$\text{Valid Steps} = \left\lfloor \frac{W + 2P - K}{S} \right\rfloor$$
5. Adding the initial placement step at the leftmost position ($+ 1$), we get the final output width:
   $$W_{\text{out}} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

---

[← The Convolution Operation](03-Convolution-Operation.md) | [Back to Index](../README.md) | [Next: Pooling Layers →](05-Pooling-Layers.md)
