# 🖼️ Image Processing Fundamentals

> **Prerequisites**: Linear Algebra | **Difficulty**: ⭐⭐☆☆☆ Beginner

---

## 📋 Table of Contents
1. [What is an Image to a Computer?](#1-what-is-an-image-to-a-computer)
2. [Color Spaces (RGB vs BGR vs HSV)](#2-color-spaces-rgb-vs-bgr-vs-hsv)
3. [Image Histograms & Equalization](#3-image-histograms--equalization)
4. [Classical Computer Vision vs. Deep Learning](#4-classical-computer-vision-vs-deep-learning)
5. [Project Ideas & What's Next](#5-project-ideas--whats-next)

---

## 1. What is an Image to a Computer?

To a computer, an image is simply a multi-dimensional matrix (tensor) of numbers.

- **Grayscale Images**: A 2D matrix of shape `(Height, Width)`. Each pixel is a single number representing the light intensity.
  - Usually represented as an 8-bit unsigned integer (`uint8`), meaning values range from $0$ (pure black) to $255$ (pure white).
- **Color Images**: A 3D matrix of shape `(Height, Width, Channels)`. 
  - Standard images have 3 channels (Red, Green, Blue). A $1920 \times 1080$ image is a matrix of size $1920 \times 1080 \times 3$, containing over 6.2 million individual numbers.

### The Coordinate System
Unlike standard Cartesian graphs where $(0,0)$ is the bottom-left, image processing universally uses the top-left corner as the origin $(0,0)$.
- The $x$-axis moves to the **right** (Width).
- The $y$-axis moves **down** (Height).

---

## 2. Color Spaces (RGB vs BGR vs HSV)

A **Color Space** is a specific mathematical organization of colors.

### 2.1 RGB (Red, Green, Blue)
The standard color space used by monitors and the human eye. Colors are created by adding light.
- `[255, 0, 0]` = Pure Red
- `[255, 255, 255]` = Pure White
- `[0, 0, 0]` = Pure Black

### 2.2 BGR (Blue, Green, Red)
Exactly the same as RGB, but the channels are stored in reverse order.
**Why do we care?** Because **OpenCV** (the industry-standard image processing library) loads images in **BGR** format by default for historical hardware reasons. If you feed a BGR image into a Deep Learning model trained on RGB images, performance will tank because the network thinks the sky is orange and apples are blue!
*Always convert `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before passing to PyTorch.*

### 2.3 HSV (Hue, Saturation, Value)
RGB is terrible for isolating objects by color. If a red apple is in a shadow, its RGB values might be `[50, 0, 0]`. If it is in the sun, it might be `[255, 100, 100]`. Finding a single "rule" for the color red in RGB is nearly impossible.

**HSV separates color from lighting:**
- **Hue**: The actual color type (0-179 in OpenCV). Red is around 0 or 179. Green is 60. Blue is 120.
- **Saturation**: The intensity of the color (0-255). 0 is gray, 255 is pure neon.
- **Value**: The brightness (0-255). 0 is pitch black, 255 is fully illuminated.

If you want to track a red ball, convert the image to HSV and just look for pixels where `Hue ≈ 0`, completely ignoring the `Value` channel to become immune to shadows!

---

## 3. Image Histograms & Equalization

### Image Histograms
A histogram is a graph showing the frequency of every pixel intensity value ($0-255$) in an image.
- If the histogram is clumped on the left ($0-50$), the image is extremely dark/underexposed.
- If it is clumped on the right ($200-255$), the image is overexposed.

### Histogram Equalization
If you take a photo at night, the contrast is terrible. The pixels might only range from $10$ to $40$. 
**Histogram Equalization** is a mathematical algorithm that forcibly stretches these pixel values to cover the entire $0-255$ range, drastically improving the contrast and revealing hidden details. 

*(Adaptive Histogram Equalization (CLAHE) does this locally in small patches to prevent blowing out already-bright areas).*

---

## 4. Classical Computer Vision vs. Deep Learning

Before Deep Learning took over in 2012, researchers spent decades manually engineering mathematical filters to extract features from images. This is called **Classical Computer Vision**.

### Classical CV Workflow (e.g., finding a face):
1. Convert to grayscale.
2. Apply a Gaussian Blur to remove static noise.
3. Apply a Sobel Filter to find all the edges.
4. Run a Hough Transform to find circles (eyes).
5. Write massive `if/else` logic trees based on the geometric distances between the circles to classify it as a "Face".

**Pros**: Requires no training data. Completely interpretable math. Runs instantly on CPUs.
**Cons**: Fails completely if the face is tilted, in shadows, partially obscured, or wearing glasses.

### Deep Learning Workflow:
1. Collect 10,000 images of faces and 10,000 images of not-faces.
2. Feed them into a ResNet-50.
3. The network automatically learns its own filters via backpropagation.

**Pros**: Robust to almost infinite variations (lighting, angle, occlusion).
**Cons**: Requires massive amounts of labeled data and expensive GPUs. A "black box".

*Modern Computer Vision heavily mixes both. We use Classical CV (OpenCV) to pre-process, crop, and normalize the images before feeding them into Deep Learning models.*

---

## 5. Project Ideas & What's Next

### Project Ideas
- 🟢 **The Invisibility Cloak**: Use your webcam and OpenCV. Capture a "background" frame. Then, continuously convert the video feed to HSV. Use an `inRange` mask to detect a specific color (like a blue blanket). Use `cv2.bitwise_and` to replace the blanket pixels with the original background pixels!

### What's Next
| Next | Why |
|------|-----|
| [OpenCV Masterclass](./02-OpenCV-Masterclass.md) | Now that we understand what images are, let's learn how to manipulate them using the most powerful CV library in the world. |

---

[← Neural Style Transfer](../10-Generative/06-Neural-Style-Transfer.md) | [Back to Index](../README.md) | [Next: OpenCV Masterclass →](./02-OpenCV-Masterclass.md)
