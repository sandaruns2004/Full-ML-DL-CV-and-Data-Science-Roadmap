# 🤸 Pose Estimation

> **Prerequisites**: CNNs, Object Detection | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [What is Pose Estimation?](#1-what-is-pose-estimation)
2. [Top-Down vs. Bottom-Up Approaches](#2-top-down-vs-bottom-up-approaches)
3. [Heatmaps vs. Regression](#3-heatmaps-vs-regression)
4. [OpenPose Architecture](#4-openpose-architecture)
5. [Evaluation Metrics (PCK, OKS)](#5-evaluation-metrics-pck-oks)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. What is Pose Estimation?

Object detection draws a bounding box around a person. **Human Pose Estimation (HPE)** goes a step further: it locates specific **Keypoints** (joints) on the human body (e.g., Left Shoulder, Right Elbow, Left Knee) to understand posture, movement, and biomechanics.

Most datasets (like MS COCO) define a human skeleton using **17 Keypoints**:
- `0`: Nose
- `1-4`: Eyes and Ears
- `5-10`: Shoulders, Elbows, Wrists
- `11-16`: Hips, Knees, Ankles

---

## 2. Top-Down vs. Bottom-Up Approaches

If there are 5 people in an image, how do we find all their joints and assign the correct arm to the correct person?

### Top-Down Approach (e.g., HRNet, Mask R-CNN)
1. Use an Object Detector (like YOLO or Faster R-CNN) to find all the people.
2. Crop out each person.
3. Run a Pose Estimation network on each cropped image to find that specific person's 17 keypoints.
*Pros*: Extremely accurate.
*Cons*: Slow. If there are 20 people in an image, you have to run the pose network 20 times. Fails if the Object Detector misses a person.

### Bottom-Up Approach (e.g., OpenPose)
1. Run the entire image through the network *once*.
2. The network detects *every single joint* in the entire image simultaneously (e.g., it finds 10 Left Shoulders and 10 Left Elbows).
3. A grouping algorithm (Part Affinity Fields) figures out which shoulder connects to which elbow to assemble the 5 distinct human skeletons.
*Pros*: Blazing fast (runtime is independent of the number of people).
*Cons*: Slightly less accurate, especially on small/far-away people.

---

## 3. Heatmaps vs. Regression

How does the Neural Network actually output a keypoint location?

**Direct Regression**:
The network outputs raw $(x, y)$ coordinates for the 17 joints.
- *Problem*: It's highly inaccurate. Predicting an exact floating-point coordinate from a high-dimensional feature map is mathematically difficult for CNNs.

**Heatmap Prediction (The Standard)**:
Instead of outputting coordinates, the network outputs 17 different **Heatmaps** (one for each joint type, e.g., a "Left Elbow Heatmap"). 
- A heatmap is a 2D matrix where the value is $1.0$ exactly at the joint, and smoothly decays to $0.0$ as you move away (a 2D Gaussian).
- *To get the final coordinate*: Simply find the $(x, y)$ pixel with the maximum value (argmax) in the heatmap!

---

## 4. OpenPose Architecture

Created by Carnegie Mellon University (CMU), OpenPose was the first real-time multi-person system.

1. **Feature Extraction**: An image is passed through a CNN (VGG-19) to generate a set of deep features.
2. **Branch 1 (Confidence Maps)**: Predicts the heatmaps for all the joints (where are the elbows?).
3. **Branch 2 (Part Affinity Fields - PAFs)**: Predicts 2D vector fields that point from one joint to another (e.g., a vector pointing from the shoulder to the elbow). This tells the grouping algorithm *how* the joints are connected, solving the multi-person assembly problem.

---

## 5. Evaluation Metrics (PCK, OKS)

Because human annotators aren't perfect (where exactly is the mathematical center of a knee?), we need forgiving metrics.

### Percentage of Correct Keypoints (PCK)
A keypoint prediction is considered "Correct" if it falls within a certain distance threshold of the ground truth. The threshold is usually defined relative to the person's size (e.g., the distance between their left shoulder and right hip).

### Object Keypoint Similarity (OKS)
The COCO standard metric. It acts like IoU but for keypoints.
It calculates the distance between the predicted keypoint and the ground truth, but penalizes errors differently depending on the joint.
- *Example*: A 2-inch error on the Nose is heavily penalized (noses are small and rigid). A 2-inch error on the Hip is barely penalized (the hip is a massive, ambiguous joint).

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **MediaPipe Pushup Counter**: Google's `mediapipe` library offers incredibly fast, pre-trained Pose Estimation that runs on CPUs. Write a Python script using OpenCV and MediaPipe to track your webcam. Calculate the mathematical angle between your Shoulder, Elbow, and Wrist. If the angle goes below 90 degrees and back to 180, count $1$ pushup!

### What's Next
| Next | Why |
|------|-----|
| [Image Generation](./07-Image-Generation.md) | We've spent this entire course analyzing and deconstructing images. Now, it's time to create them from scratch. |

---

[← Image Segmentation](./05-Image-Segmentation.md) | [Back to Index](../README.md) | [Next: Image Generation →](./07-Image-Generation.md)
