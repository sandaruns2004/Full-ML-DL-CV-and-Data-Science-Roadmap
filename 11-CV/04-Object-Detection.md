# 🎯 Object Detection

> **Prerequisites**: CNNs, Image Classification | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [Classification vs. Object Detection](#1-classification-vs-object-detection)
2. [Intersection Over Union (IoU) & mAP](#2-intersection-over-union-iou--map)
3. [Two-Stage Detectors (R-CNN Family)](#3-two-stage-detectors-r-cnn-family)
4. [One-Stage Detectors (YOLO Family)](#4-one-stage-detectors-yolo-family)
5. [Non-Maximum Suppression (NMS)](#5-non-maximum-suppression-nms)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. Classification vs. Object Detection

- **Image Classification**: "What is the main subject in this image?" (Outputs a single class).
- **Classification + Localization**: "What is the subject, and where is it?" (Outputs a class + 4 coordinates for a bounding box: $x_{min}, y_{min}, x_{max}, y_{max}$).
- **Object Detection**: "Find *all* the objects in this image, classify each one, and draw a bounding box around each one."

Object detection is vastly more difficult because the network doesn't know in advance how many objects are in the image. It could be 0, or it could be 100.

---

## 2. Intersection Over Union (IoU) & mAP

### Intersection Over Union (IoU)
How do we know if our predicted bounding box is accurate compared to the ground truth box drawn by a human labeler? We use **IoU**.

$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

- $\text{IoU} = 1.0$: Perfect overlap.
- $\text{IoU} \geq 0.5$: The standard threshold to be considered a "True Positive" detection.
- $\text{IoU} < 0.5$: Considered a "False Positive".

### Mean Average Precision (mAP)
The standard metric for evaluating object detectors.
1. Calculate the Precision-Recall curve for a specific class (e.g., "Dog") at a specific IoU threshold (e.g., $0.5$).
2. The **Average Precision (AP)** is the area under that curve.
3. The **mAP** is the mean of the APs calculated across *all* classes.

In modern competitions (like COCO), mAP is averaged across multiple IoU thresholds (from 0.50 to 0.95 in steps of 0.05) to heavily reward perfectly tight bounding boxes.

---

## 3. Two-Stage Detectors (R-CNN Family)

Pioneered by Ross Girshick. These algorithms prioritize accuracy over speed.

### R-CNN (2014)
1. **Stage 1 (Region Proposal)**: Use a classical algorithm (Selective Search) to generate 2,000 "proposals" (areas that might contain an object).
2. **Stage 2 (Classification)**: Crop those 2,000 regions, resize them, and pass *all 2,000* individually through a CNN (like AlexNet) to classify them.
*Flaw*: Extremely slow (takes 47 seconds per image).

### Faster R-CNN (2015)
1. Pass the entire image through a CNN *once* to get a feature map.
2. **Stage 1**: A **Region Proposal Network (RPN)** slides over the feature map and predicts which areas contain objects using "Anchor Boxes" (pre-defined shapes like tall, wide, or square).
3. **Stage 2**: Using **ROI Pooling**, crop those specific areas from the feature map and classify them.
*Result*: Massive speedup. Can run at ~5-10 FPS. Still the gold standard for high-accuracy medical and satellite imagery.

---

## 4. One-Stage Detectors (YOLO Family)

Pioneered by Joseph Redmon (2015). **You Only Look Once (YOLO)** prioritizes real-time speed.

Instead of proposing regions, YOLO frames object detection as a massive, single-shot regression problem.

**The YOLO Architecture**:
1. Divide the input image into an $S \times S$ grid (e.g., $13 \times 13$).
2. **The Golden Rule**: If the *center* of an object falls into a specific grid cell, that grid cell is solely responsible for detecting that object.
3. Each grid cell predicts $B$ bounding boxes. For each box, it outputs 5 values:
   - $x, y$: The center coordinates (relative to the grid cell).
   - $w, h$: Width and height (relative to the whole image).
   - **Confidence Score**: $Pr(\text{Object}) \times \text{IoU}$. How confident the model is that there is an object here, multiplied by how tight it thinks the box is.
4. Each grid cell also predicts $C$ conditional class probabilities (e.g., is it a dog, car, or person?).

The final output tensor size is $S \times S \times (B \times 5 + C)$.

By passing the image through the CNN exactly once, YOLO achieves blazing fast speeds (45 to 150 FPS), making it the standard for self-driving cars, robotics, and CCTV monitoring.

---

## 5. Non-Maximum Suppression (NMS)

Because YOLO divides the image into a grid, multiple adjacent grid cells might think they contain the center of a large object. The network will output 5 different bounding boxes all clustered tightly around the exact same dog.

**Non-Maximum Suppression (NMS)** is the algorithmic cleanup step applied to the raw outputs:
1. Discard all boxes with a Confidence Score below a certain threshold (e.g., $< 0.4$).
2. Pick the box with the highest confidence and output it as an official prediction.
3. Calculate the IoU of this chosen box with all other remaining boxes for that specific class.
4. If the $\text{IoU} > 0.5$, discard the other box (we assume it's detecting the same object).
5. Repeat until no overlapping boxes are left.

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Ultralytics YOLOv8**: Do not build YOLO from scratch. The math is notoriously finicky. Install the `ultralytics` package. Download a pre-trained `yolov8n.pt` model. Write a 10-line Python script using OpenCV to capture your webcam feed, pass the frames to the YOLO model, and display the resulting bounding boxes in real-time.

### What's Next
| Next | Why |
|------|-----|
| [Image Segmentation](./05-Image-Segmentation.md) | Bounding boxes are great, but they include a lot of background pixels. How do we trace the exact pixel-perfect outline of an object? |

---

[← Image Classification Pipeline](./03-Image-Classification-Pipeline.md) | [Back to Index](../README.md) | [Next: Image Segmentation →](./05-Image-Segmentation.md)
