# 🔷 The Convolution Operation

---

## 📋 Table of Contents
1. [Beginner: Sliding Windows and Image Filters](#1-beginner-sliding-windows-and-image-filters)
2. [Intermediate: 2D Convolution in NumPy](#2-intermediate-2d-convolution-in-numpy)
3. [Advanced: Cross-Correlation vs. Convolution & 3D Tensors](#3-advanced-cross-correlation-vs-convolution--3d-tensors)

---

## 1. Beginner: Sliding Windows and Image Filters

### Simple Intuition
A **Convolution** is a mathematical operation that combines two sources of information: an input image and a smaller grid of numbers called a **filter** (or **kernel**). 
The filter slides across the image from left to right, top to bottom. At each position, it multiplies the numbers in the filter by the pixel values it is resting on, and sums them up to create a single pixel in a new image, called a **Feature Map**.

### Real-World Analogy: Flashlight Search
Imagine looking at a painting in a dark room with a small, square flashlight:
- The flashlight beam only illuminates a small portion of the painting at a time (e.g. $3 \times 3$ pixels).
- As you sweep the flashlight across the painting (sliding window), you register features in your mind. If you find a vertical line, you write down its location.
- The map of all locations where you found vertical lines is the output feature map.

### Visual Representation

```
Input Image (5x5)            Filter (3x3)           Output Feature Map (3x3)
  1  1  1  0  0
  0  1  1  1  0                 1  0  1
  0  0  1  1  1  ★  matmul  -->  0  1  0  ------->     4  3  4
  0  0  1  1  0                 1  0  1
  0  1  1  0  0
```

---

## 2. Intermediate: 2D Convolution in NumPy

Let us write a function in pure NumPy to compute 2D convolution on a single-channel grayscale image.

```python
import numpy as np

def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Computes a 2D valid convolution on a single-channel image.
    
    Args:
        image: NumPy array of shape (H, W)
        kernel: NumPy array of shape (Kh, Kw)
        
    Returns:
        NumPy array of shape (H - Kh + 1, W - Kw + 1)
    """
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    
    # Calculate output shape
    out_h = img_h - k_h + 1
    out_w = img_w - k_w + 1
    output = np.zeros((out_h, out_w))
    
    # Slide the window
    for i in range(out_h):
        for j in range(out_w):
            # Extract local window patch
            patch = image[i:i+k_h, j:j+k_w]
            # Element-wise multiply and sum
            output[i, j] = np.sum(patch * kernel)
            
    return output

# Test the function with a vertical edge detector
image = np.array([
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0]
], dtype=np.float32)

vertical_kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [3, 0, -3]
], dtype=np.float32)

feature_map = convolve2d(image, vertical_kernel)
print("Input Image:\n", image)
print("Output Feature Map:\n", feature_map)
```

---

## 3. Advanced: Cross-Correlation vs. Convolution & 3D Tensors

### Mathematical Formulation
In signal processing, the continuous 2D convolution of an image $I$ and a filter $K$ is defined as:
$$(I * K)(x, y) = \iint I(u, v) K(x - u, y - v) \, du \, dv$$
Note that the filter is **flipped** ($K(x-u, y-v)$) before multiplying.

In Deep Learning libraries (like PyTorch), we actually implement **Cross-Correlation**, which does not flip the filter:
$$\text{Output}(i, j) = \sum_{m} \sum_{n} I(i + m, j + n) K(m, n)$$
For simplicity, the deep learning community calls this "convolution". Since the filter weights are learned during training, whether the filter is flipped or not is mathematically irrelevant because the network learns the flipped weights anyway.

### 3.4 3D Convolutions on Color Tensors
When processing color images, the input has shape $(C, H, W)$ (typically 3 channels).
A convolution filter must match the input channel depth. Therefore, the filter has shape $(C, K_h, K_w)$.

The convolution sums across the channels:
$$\text{Output}(i, j) = \sum_{c=1}^{C} \sum_{m=0}^{K_h-1} \sum_{n=0}^{K_w-1} I(c, i + m, j + n) K(c, m, n) + b$$
This sum yields a single value per sliding position, collapsing the channels. If we want $D$ output channels, we apply $D$ independent filters, resulting in an output tensor of shape $(D, H_{\text{out}}, W_{\text{out}})$.

---

[← Previous: Image Representation And Pixels](./02-Image-Representation-And-Pixels.md) | [Back to Index](./README.md) | [Next: Padding And Strides →](./04-Padding-And-Strides.md)
