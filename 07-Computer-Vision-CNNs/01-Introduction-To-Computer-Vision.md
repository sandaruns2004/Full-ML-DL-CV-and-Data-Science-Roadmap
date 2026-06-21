# 👁️ Introduction to Computer Vision

---

## 📋 Table of Contents
1. [Beginner: The Human vs. Machine Vision Analogy](#1-beginner-the-human-vs-machine-vision-analogy)
2. [Intermediate: Vision Pipelines & OpenCV Basics](#2-intermediate-vision-pipelines--opencv-basics)
3. [Advanced: Biological Visual Pathways & Mathematical Fields](#3-advanced-biological-visual-pathways--mathematical-fields)

---

## 1. Beginner: The Human vs. Machine Vision Analogy

### Simple Intuition
**Computer Vision (CV)** is the subfield of artificial intelligence that focuses on enabling computers to see, identify, and process images and videos in the same way that human vision does, and then provide appropriate outputs.

### Real-World Analogy: The Eye and the Brain
- **The Camera (Eye)**: Your eyes act as lenses that capture light and project it onto the retina.
- **Visual Cortex (Brain)**: The back of your brain processes the electrical signals from the retina, finding boundaries, shapes, and recognizing what objects are in front of you.

In computer vision, the **camera** acts as the eye, capturing grid values of light intensity, and the **neural network** acts as the visual cortex, processing those values to identify objects.

---

## 2. Intermediate: Vision Pipelines & OpenCV Basics

Let us write a simple Python script using `OpenCV` to load, display, and inspect the properties of an image.

```python
import cv2
import numpy as np

# Load a grayscale or color image (assumes we have a sample 'image.jpg' in directory)
# For testing, we will generate a synthetic random image array:
img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# Check dimensions
height, width, channels = img.shape
print(f"Image Dimensions: {width}x{height} with {channels} channels")

# Convert color spaces: OpenCV loads images in BGR format by default
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(f"Grayscale shape: {img_gray.shape}")
```

---

## 3. Advanced: Biological Visual Pathways & Mathematical Fields

### Biological Insights (Hubel & Wiesel)
In 1959, David Hubel and Torsten Wiesel ran landmark experiments on the visual cortex of cats. They discovered that:
- Neurons in the primary visual cortex (V1) do not respond to complex shapes directly.
- Instead, individual neurons fire in response to **simple features** at specific orientations (e.g. horizontal, vertical, or diagonal edges).
- This hierarchically builds up: simple cells feed into complex cells, which feed into hypercomplex cells to construct shape recognition.

This discovery is the direct foundation of **Convolutional Neural Networks (CNNs)**, where early layers extract simple edges and deep layers combine them to detect objects.

### Mathematical Formulation
An image can be represented as a continuous 2D function $f(x, y)$, where $x$ and $y$ are spatial coordinates, and the value of $f$ at any point $(x, y)$ is the intensity (or color vector) of the image.
A digital image is a discretized version of this function:
$$I[i, j] = f(i \cdot \Delta x, j \cdot \Delta y)$$
where $i \in \{0, \dots, H-1\}$ and $j \in \{0, \dots, W-1\}$ represent index dimensions of the tensor grid.

---

[← Back to Index](./README.md) | [Next: Image Representation And Pixels →](./02-Image-Representation-And-Pixels.md)
