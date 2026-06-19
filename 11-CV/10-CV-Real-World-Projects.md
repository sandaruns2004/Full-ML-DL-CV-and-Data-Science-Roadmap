# 🏆 Computer Vision: Real-World Capstone Projects

> **Prerequisites**: All previous CV Modules | **Difficulty**: ⭐⭐⭐⭐⭐ Expert

---

## 📋 Table of Contents
1. [The Capstone Philosophy](#1-the-capstone-philosophy)
2. [Project 1: Automated Quality Assurance System](#2-project-1-automated-quality-assurance-system)
3. [Project 2: Smart Security Camera (Edge AI)](#3-project-2-smart-security-camera-edge-ai)
4. [Project 3: Medical Image Segmentation](#4-project-3-medical-image-segmentation)
5. [Project 4: Automated License Plate Recognition (ALPR)](#5-project-4-automated-license-plate-recognition-alpr)
6. [Conclusion to Phase 3](#6-conclusion-to-phase-3)

---

## 1. The Capstone Philosophy

Reading about architectures and mathematics is essential, but you don't truly understand Deep Learning until you deploy a system that processes real-world, noisy data.

The following projects are designed to be portfolio pieces. They simulate actual tasks you would be assigned as a Senior Computer Vision Engineer in the industry. Choose at least one to complete before moving on to the next phase of the curriculum.

---

## 2. Project 1: Automated Quality Assurance System

**Industry**: Manufacturing / Supply Chain
**The Goal**: Detect microscopic defects (scratches, dents, missing components) on a high-speed factory assembly line.

**Requirements**:
1. **Dataset**: Download the [MVTec AD dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/) (the industry standard for anomaly detection).
2. **Architecture**: You cannot use a standard classifier because defects are extremely rare (you might only have 5 pictures of scratched metal). Instead, implement a **PatchCore** or **PaDiM** architecture using pre-trained ResNet features to detect anomalies without *any* anomalous training data (unsupervised anomaly detection).
3. **Speed**: The factory line moves fast. Your system must process 30 frames per second.

**Skills Demonstrated**: Transfer Learning, Feature Extraction, Unsupervised Learning, Optimization.

---

## 3. Project 2: Smart Security Camera (Edge AI)

**Industry**: Smart Home / Physical Security
**The Goal**: Turn a cheap Raspberry Pi and webcam into a smart security system that alerts your phone when it detects a *person* or *dog*, but ignores *cars* driving by.

**Requirements**:
1. **Model**: Download a highly compressed model like `YOLOv8n` (nano).
2. **Optimization**: Standard PyTorch models are too heavy for a Raspberry Pi. Export the model to **ONNX** or use **OpenVINO** to accelerate CPU inference.
3. **Logic**: Write a Python script using OpenCV to capture the webcam stream. If the model detects `class=person` with confidence $>0.60$ for 3 consecutive frames, trigger an alert.
4. **Data Handling**: Use OpenCV to draw the bounding box and timestamp on the frame, and save the image locally as evidence.

**Skills Demonstrated**: Object Detection (YOLO), OpenCV Video Streams, Edge Deployment (ONNX), Real-time Logic.

---

## 4. Project 3: Medical Image Segmentation

**Industry**: Healthcare / Biotech
**The Goal**: Automatically segment lung tumors from CT scans to assist radiologists in measuring tumor volume over time.

**Requirements**:
1. **Dataset**: Download the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) dataset (specifically Lung or Liver).
2. **Data Processing**: Medical images are stored in `.nii` or `.dcm` formats, not `.jpg`. You must learn how to load volumetric medical data using `nibabel` or `SimpleITK`, extract 2D slices, and normalize the Hounsfield units.
3. **Architecture**: Train a **U-Net** or **UNet++**.
4. **Loss**: You must use **Dice Loss** or Focal Loss to handle the extreme class imbalance (the tumor is only 1% of the pixels in the image).
5. **Metric**: Achieve a Dice Score of at least $0.80$ on the validation set.

**Skills Demonstrated**: Medical Image Processing, Semantic Segmentation, Class Imbalance, U-Net.

---

## 5. Project 4: Automated License Plate Recognition (ALPR)

**Industry**: Traffic Management / Law Enforcement
**The Goal**: Read the license plate text off cars moving through a toll booth.

**Requirements**:
1. **Stage 1 (Detection)**: Train an Object Detector (e.g., Faster R-CNN or YOLO) to draw a bounding box precisely around the license plate in the image.
2. **Stage 2 (Cropping & Processing)**: Use OpenCV to crop the bounding box, convert to grayscale, and apply Adaptive Thresholding to make the text pop.
3. **Stage 3 (OCR)**: Feed the processed crop into an Optical Character Recognition (OCR) engine like **Tesseract OCR** or **EasyOCR** to extract the actual text strings.
4. **Evaluation**: Your system should output `"CA-1ABC123"`.

**Skills Demonstrated**: Multi-stage Pipelines, Object Detection, Classical CV pipelines, Optical Character Recognition.

---

## 6. Conclusion to Phase 3

Congratulations! You have successfully mastered Computer Vision. 

You have built the math from the ground up (Convolutions, Padding, Strides), explored the history of architectures (VGG to ConvNeXt), mastered modern training tricks (AMP, CutMix), and expanded your toolkit to cover Segmentation, Detection, Pose, and 3D Vision.

You are now ready to tackle the other half of the Deep Learning revolution: Sequences, Language, and Attention.

---

## 7. What's Next
| Next Topic | Why |
|------------|-----|
| [Foundation Models CV](./11-Foundation-Models-CV.md) | Explore state-of-the-art vision models like SAM, DINO, and CLIP. |

---

[← 3D Vision](./09-3D-Vision.md) | [Back to Index](../README.md) | [Next: Foundation Models CV →](./11-Foundation-Models-CV.md)
