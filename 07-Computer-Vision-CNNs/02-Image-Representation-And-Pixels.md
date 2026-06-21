# 🖼️ Image Representation and Pixels

---

## 📋 Table of Contents
1. [Beginner: What are Pixels and Color Channels?](#1-beginner-what-are-pixels-and-color-channels)
2. [Interactive: Tensor Layouts (HWC vs. CHW) in PyTorch](#2-interactive-tensor-layouts-hwc-vs-chw-in-pytorch)
3. [Advanced: Mathematical Color Formulations and Normalization](#3-advanced-mathematical-color-formulations-and-normalization)

---

## 1. Beginner: What are Pixels and Color Channels?

### Simple Intuition
To a computer, an image is nothing more than a grid of numbers. Each coordinate in the grid is called a **pixel** (short for "picture element"). 
- **Grayscale Images**: Each pixel contains a single number representing brightness, typically from $0$ (pure black) to $255$ (pure white).
- **Color Images (RGB)**: Each pixel contains three numbers, representing the intensity of **Red, Green, and Blue** light. These three values are combined to display millions of colors.

### Real-World Analogy: Pointillism Painting
Imagine a painter using a technique called *pointillism*, painting a canvas entirely out of tiny colored dots. From a distance, the dots merge to form a detailed landscape. Up close, it is just a grid of colored dots. A computer image is exactly the same: a canvas of tiny colored pixels.

---

## 2. Interactive: Tensor Layouts (HWC vs. CHW) in PyTorch

In computer vision, image dimensions are formatted in two main ways:
1. **HWC**: Height × Width × Channels. This is the default format for libraries like OpenCV, NumPy, and PIL.
2. **CHW**: Channels × Height × Width. This is the format required by **PyTorch** for GPU operations.

### PyTorch Layout Conversion Code

```python
import torch
import numpy as np

# Create a mock HWC image in NumPy: 128x128 pixels, 3 channels (RGB)
hwc_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
print("NumPy HWC Image Shape:", hwc_image.shape)

# Convert to PyTorch Tensor
tensor_image = torch.from_numpy(hwc_image)

# Convert HWC -> CHW using permute (channel transposition)
chw_tensor = tensor_image.permute(2, 0, 1)
print("PyTorch CHW Tensor Shape:", chw_tensor.shape)

# Add a batch dimension: PyTorch expects input tensors to be (B, C, H, W)
batch_tensor = chw_tensor.unsqueeze(0)
print("PyTorch Batch Tensor Shape:", batch_tensor.shape)
```

---

## 3. Advanced: Mathematical Color Formulations and Normalization

### Color Conversion Math
Converting RGB to Grayscale is not a simple average: $\frac{R+G+B}{3}$. 
Because human eyes are more sensitive to green and red light than blue light, we use a weighted sum (the **Luminance** formula):
$$Y = 0.299R + 0.587G + 0.114B$$

### Normalization and Standardization
For optimal training convergence, we convert pixel values from the range $[0, 255]$ to floating point numbers in the range $[0.0, 1.0]$:
$$x_{\text{normalized}} = \frac{x}{255.0}$$

Then, we apply **Standardization** using the mean ($\mu$) and standard deviation ($\sigma$) of the dataset (e.g. ImageNet statistics):
$$x_{\text{standardized}} = \frac{x_{\text{normalized}} - \mu}{\sigma}$$
This centers the features around zero with a standard deviation of 1, which helps prevent gradients from vanishing or exploding early in training.

---

[← Previous: Introduction To Computer Vision](./01-Introduction-To-Computer-Vision.md) | [Back to Index](./README.md) | [Next: Convolution Operation →](./03-Convolution-Operation.md)
