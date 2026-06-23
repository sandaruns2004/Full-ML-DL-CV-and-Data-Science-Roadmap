# 🛠️ OpenCV Masterclass

> **Prerequisites**: Image Processing Fundamentals | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [What is OpenCV?](#1-what-is-opencv)
2. [I/O and Basic Operations](#2-io-and-basic-operations)
3. [Thresholding & Masking](#3-thresholding--masking)
4. [Morphological Operations (Erosion/Dilation)](#4-morphological-operations-erosiondilation)
5. [Edge Detection (Canny) & Contours](#5-edge-detection-canny--contours)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. What is OpenCV?

**OpenCV (Open Source Computer Vision Library)** is the most widely used classical computer vision library in the world. Originally developed by Intel in 1999, it is written in highly optimized C/C++ but has excellent Python bindings.

While Deep Learning (PyTorch/TensorFlow) is used for complex pattern recognition, OpenCV is used for:
1. Reading/Writing video streams and images.
2. High-speed geometric transformations (resizing, rotating).
3. Extracting regions of interest (bounding boxes) before sending them to a Neural Network.

*Note: In Python, OpenCV is imported as `cv2`.*

---

## 2. I/O and Basic Operations

Images in OpenCV are stored as NumPy arrays.

```python
import cv2
import numpy as np

# 1. Read an image (Loads as BGR by default!)
img = cv2.imread('image.jpg')

# 2. Check its dimensions
print(img.shape) # e.g., (1080, 1920, 3) -> (Height, Width, Channels)

# 3. Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 4. Resize the image (Width first in cv2.resize!)
resized = cv2.resize(img, (800, 600)) 

# 5. Draw a red rectangle (Point 1, Point 2, Color in BGR, Thickness)
cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 2)

# 6. Show the image
cv2.imshow('My Window', img)
cv2.waitKey(0) # Waits infinitely until the user presses a key
cv2.destroyAllWindows()
```

---

## 3. Thresholding & Masking

Thresholding converts a grayscale image into a **Binary Image** (strictly black and white). 
This is the first step in isolating objects from the background.

```python
# Any pixel > 127 becomes 255 (White). Everything else becomes 0 (Black).
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

**Adaptive Thresholding**: If an image has a shadow cast across half of it, a global threshold of 127 will fail. Adaptive thresholding calculates a different threshold for small, local regions of the image.

**Masking**: Once you have a binary image (the "mask"), you can use `cv2.bitwise_and(img, img, mask=thresh)` to "cut out" the object from the original color image, leaving the background completely black.

---

## 4. Morphological Operations (Erosion/Dilation)

When you threshold an image, the resulting white blob often has holes in it, or random white "noise" pixels floating around it. Morphological operations fix this.

1. **Erosion**: A kernel (e.g., a $5 \times 5$ square of 1s) slides over the image. A pixel is only kept white if *all* pixels under the kernel are white.
   - *Result*: Shrinks the white blob. Instantly destroys tiny floating noise pixels.
2. **Dilation**: The opposite. A pixel becomes white if *at least one* pixel under the kernel is white.
   - *Result*: Expands the white blob. Fills in black holes inside the object.

**Closing** (Dilation followed by Erosion) is incredibly useful for closing small black holes inside a solid white object without changing the overall size of the object.

```python
kernel = np.ones((5,5), np.uint8)
# Close small holes inside the foreground object
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
```

---

## 5. Edge Detection (Canny) & Contours

### Canny Edge Detection
Developed by John F. Canny in 1986, this is a multi-stage algorithm that finds the structural outlines of objects.
It uses a Gaussian filter to remove noise, calculates the intensity gradients (Sobel), and uses hysteresis thresholding to link strong edges together.

```python
# Lower bound 100, Upper bound 200
edges = cv2.Canny(gray, 100, 200)
```

### Contours
Once you have edges or a binary mask, you can find **Contours** (the continuous curve joining all the continuous points along a boundary).

Finding contours allows you to count the number of objects, calculate their surface area, and draw exact bounding boxes around them.

```python
# Find contours on a binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # Calculate the area
    area = cv2.contourArea(cnt)
    
    # Ignore tiny noise artifacts
    if area > 500:
        # Get the bounding box coordinates
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Draw a green bounding box on the original color image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Document Scanner**: Write a script that takes a photo of a piece of paper on a desk. Convert to grayscale, apply Gaussian Blur, and use Canny Edge Detection. Find the largest contour with exactly 4 corners. Use `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to warp the image so the paper appears perfectly flat and top-down, just like a scanner app!

### What's Next
| Next | Why |
|------|-----|
| [Image Classification Pipeline](./03-Image-Classification-Pipeline.md) | Let's put everything we've learned together into a massive, end-to-end PyTorch vision pipeline. |

---

[← Image Processing Fundamentals](01-Image-Processing-Fundamentals.md) | [Back to Index](../README.md) | [Next: End-to-End Image Classification Pipeline →](03-Image-Classification-Pipeline.md)
