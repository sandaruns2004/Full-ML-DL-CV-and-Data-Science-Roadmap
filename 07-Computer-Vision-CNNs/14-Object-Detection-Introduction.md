# 🎯 Object Detection Introduction

---

## 📋 Table of Contents
1. [Beginner: Finding What and Where](#1-beginner-finding-what-and-where)
2. [Intermediate: Implementing Intersection over Union (IoU)](#2-intermediate-implementing-intersection-over-union-iou)
3. [Advanced: Object Detection Paradigms (YOLO vs. Faster R-CNN)](#3-advanced-object-detection-paradigms-yolo-vs-faster-r-cnn)

---

## 1. Beginner: Finding What and Where

### Simple Intuition
- **Image Classification**: Answers the question: *"Is there an object in this image?"* (e.g. Yes, a dog).
- **Object Detection**: Answers two questions: *"What objects are in this image, and where are they located?"* It outputs the class name along with a rectangular **bounding box** that wraps around each detected object.

### Real-World Analogy: Spotting Animals
Imagine playing a game of "I Spy" looking at a crowded playground:
- Classification is saying: "I spy a dog!"
- Object Detection is walking over and pointing your finger, drawing an imaginary box in the air around the dog to show exactly where it is.

---

## 2. Intermediate: Implementing Intersection over Union (IoU)

To evaluate how well our predicted bounding box fits the actual true box, we use **Intersection over Union (IoU)**.
IoU measures the ratio of the overlapping area between the two boxes to the total combined area of both boxes.

$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

An IoU score $> 0.5$ is generally considered a decent prediction, while $> 0.7$ is excellent.

### PyTorch IoU Implementation

```python
import torch

def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: Tensor of shape (4,) representing [x1, y1, x2, y2] (top-left, bottom-right)
        box2: Tensor of shape (4,) representing [x1, y1, x2, y2]
        
    Returns:
        IoU value as float
    """
    # 1. Determine the coordinates of the intersection rectangle
    x_left = max(box1[0].item(), box2[0].item())
    y_top = max(box1[1].item(), box2[1].item())
    x_right = min(box1[2].item(), box2[2].item())
    y_bottom = min(box1[3].item(), box2[3].item())
    
    # Compute intersection area
    intersection_area = max(0.0, x_right - x_left) * max(0.0, y_bottom - y_top)
    
    # 2. Compute area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 3. Compute union area
    union_area = box1_area.item() + box2_area.item() - intersection_area
    
    # Return IoU
    return float(intersection_area / union_area) if union_area > 0 else 0.0

# Quick test
b1 = torch.tensor([0.0, 0.0, 2.0, 2.0]) # Area: 4
b2 = torch.tensor([1.0, 1.0, 3.0, 3.0]) # Area: 4, Overlap: 1x1 = 1
print("Computed IoU:", compute_iou(b1, b2)) # Expected: 1 / (4 + 4 - 1) = 1/7 ≈ 0.1428
```

---

## 3. Advanced: Object Detection Paradigms (YOLO vs. Faster R-CNN)

Object detection models are split into two primary architectures:

### 1. Two-Stage Detectors (e.g. Faster R-CNN)
Two-stage models separate localization and classification:
- **Stage 1 (Region Proposal Network - RPN)**: Scans the image to propose candidate regions (regions of interest - RoIs) that are likely to contain objects.
- **Stage 2 (Classifier Head)**: Extracts feature maps from the proposed regions and classifies them, refining the bounding box coordinates.
- **Tradeoff**: Highly accurate but computationally heavy and slow, making real-time deployment difficult.

### 2. One-Stage Detectors (e.g. YOLO - You Only Look Once)
One-stage models treat detection as a single regression problem:
- The input image is split into a grid (e.g. $7 \times 7$).
- A single convolutional network processes the image once and directly predicts class probabilities and bounding box coordinates for each grid cell.
- **Tradeoff**: Blazing fast and capable of running in real-time ($60+$ FPS) on video feeds, but historically had lower accuracy on small, clustered objects compared to two-stage models.

---

[← Previous: Data Augmentation](./13-Data-Augmentation.md) | [Back to Index](./README.md) | [Next: Image Segmentation Introduction →](./15-Image-Segmentation-Introduction.md)
