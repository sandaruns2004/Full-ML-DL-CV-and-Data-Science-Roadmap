# 🖼️ Edge Detection & Feature Maps

---

## 📋 Table of Contents
1. [Beginner: Finding the Outlines](#1-beginner-finding-the-outlines)
2. [Intermediate: Sobel Filter Implementation in Python](#2-intermediate-sobel-filter-implementation-in-python)
3. [Advanced: Gradient Vector Fields and Orientation](#3-advanced-gradient-vector-fields-and-orientation)

---

## 1. Beginner: Finding the Outlines

### Simple Intuition
Before a network can recognize objects (like faces or cars), it must find their boundaries. An **edge** is a boundary in an image where the brightness changes sharply. By detecting edges, we can extract the structural outlines of objects while ignoring irrelevant background details.

### Real-World Analogy: Coloring Book
Imagine turning a photograph into a coloring book page. You trace along the outlines of the people and objects, discarding the solid color fills. Those outlines represent the edges of the image. The early layers of a CNN act like someone drawing a coloring book, finding simple edges at different angles.

---

## 2. Intermediate: Sobel Filter Implementation in Python

We can detect horizontal and vertical edges manually using **Sobel Operators**. The Sobel kernels are:

$$G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

Let us write a NumPy script to compute Sobel edge detection.

```python
import numpy as np

def apply_sobel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies Sobel horizontal and vertical edge detection.
    
    Args:
        image: Grayscale image array of shape (H, W)
        
    Returns:
        Horizontal gradients, Vertical gradients
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    h, w = image.shape
    grad_x = np.zeros((h - 2, w - 2))
    grad_y = np.zeros((h - 2, w - 2))
    
    for i in range(h - 2):
        for j in range(w - 2):
            patch = image[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(patch * Gx)
            grad_y[i, j] = np.sum(patch * Gy)
            
    return grad_x, grad_y

# Simple test image: vertical split of white and black
test_img = np.zeros((8, 8))
test_img[:, 4:] = 255.0

gx, gy = apply_sobel(test_img)
print("Horizontal Edge Gradients Gx:\n", np.round(gx))
```

---

## 3. Advanced: Gradient Vector Fields and Orientation

### Mathematical Formulations
For a continuous 2D image function $I(x, y)$, the spatial gradient is a vector field:
$$\nabla I = \begin{bmatrix} \frac{\partial I}{\partial x} \\ \frac{\partial I}{\partial y} \end{bmatrix} = \begin{bmatrix} G_x \\ G_y \end{bmatrix}$$

We can calculate the **Gradient Magnitude** (the strength of the edge) and the **Gradient Orientation** (the direction of the change in brightness) at any pixel coordinate:

1. **Gradient Magnitude**:
   $$M(x, y) = \|\nabla I\| = \sqrt{G_x^2 + G_y^2}$$

2. **Gradient Orientation**:
   $$\theta(x, y) = \arctan2(G_y, G_x)$$

In CNNs, the filter weights are not fixed Sobel filters; they are learned parameters. The network automatically learns Sobel-like edge detectors in the first layer and combines them to extract complex shapes in deeper layers.

---

[← CNN Architecture Basics](06-CNN-Architecture-Basics.md) | [Back to Index](../README.md) | [Next: CNN Backpropagation →](08-CNN-Backpropagation.md)
