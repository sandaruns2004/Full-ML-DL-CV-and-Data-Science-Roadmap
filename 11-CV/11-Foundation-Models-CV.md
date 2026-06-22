# 🏗️ Foundation Models for Computer Vision

> **Prerequisites**: CNNs, Vision Transformers | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Promptable Segmentation: Segment Anything Model (SAM)](#1-promptable-segmentation-segment-anything-model-sam)
2. [Self-Supervised Vision: DINO and DINOv2](#2-self-supervised-vision-dino-and-dinov2)
3. [Using Foundation Models](#3-using-foundation-models)

---

In Natural Language Processing (NLP), "Foundation Models" like GPT-3 and BERT revolutionized the field. Instead of training a new model from scratch for every task, developers simply download a massive, pre-trained foundation model and fine-tune it or prompt it for their specific use case.

Computer Vision (CV) historically lagged behind, relying on task-specific models (e.g., a specific YOLO model trained only to detect cars). However, recent breakthroughs have brought the Foundation Model paradigm to CV.

The two most prominent paradigms for Vision Foundation Models are **Promptable Segmentation** and **Self-Supervised Representation Learning**.

---

## 1. Promptable Segmentation: Segment Anything Model (SAM)

Meta's **Segment Anything Model (SAM)** is designed to be the ultimate, generalized image segmentation model. Prior to SAM, if you wanted to segment medical cells, you needed to train a U-Net on a medical cell dataset. SAM can segment medical cells, street signs, or cats out-of-the-box without any task-specific training.

### The "Promptable" Paradigm
SAM borrows the concept of "prompting" from LLMs. Instead of text prompts, SAM accepts visual prompts to indicate *what* to segment:
*   **Point Prompts**: A user clicks a pixel (e.g., on a cat's nose).
*   **Box Prompts**: A user draws a bounding box roughly around an object.
*   **Text Prompts**: "Find the red car." (CLIP is used under the hood).
*   **Mask Prompts**: A coarse, low-resolution mask from a previous frame.

### SAM Architecture
SAM consists of three components:
1.  **Image Encoder**: A heavy Vision Transformer (ViT) (e.g., ViT-H). It processes the high-resolution image once and outputs an image embedding. This is computationally expensive but only happens once per image.
2.  **Prompt Encoder**: A lightweight network that encodes the points, boxes, or text into prompt embeddings.
3.  **Mask Decoder**: A lightweight Transformer decoder block that takes the image embedding and the prompt embeddings and predicts the segmentation mask. 

Because the Mask Decoder is incredibly lightweight, it can run in real-time in a web browser. The heavy image embedding is calculated once on a server, and the user can interactively click around the image to generate masks instantly.

### Resolving Ambiguity
If a user clicks on a person's shirt, do they want to segment the shirt, or the whole person? SAM solves this ambiguity by predicting **multiple valid masks** (usually 3: sub-part, part, whole object) and an associated confidence score for each.

---

## 2. Self-Supervised Vision: DINO and DINOv2

How do you train a foundation model without human labels? 

In NLP, models are trained to predict the next word (Self-Supervised Learning). For images, there are no "next words". Contrastive learning methods like CLIP require millions of text descriptions. 

Meta's **DINO** (Self-**DI**stillation with **NO** labels) trains high-quality vision models entirely on images without any labels or text pairs.

### The DINO Architecture (Knowledge Distillation)
DINO uses a Teacher-Student architecture. Both the Teacher and the Student are Vision Transformers (ViTs) with the exact same architecture, but different weights.

The goal is to force the Student network to match the output of the Teacher network.

### How it learns without labels:
1.  **Cropping**: Take an image. Create two different augmented views:
    *   *Global Views*: Two large crops covering >50% of the image.
    *   *Local Views*: Several small crops covering <50% of the image.
2.  **Pass through networks**:
    *   Pass the *Global Views* through the **Teacher**.
    *   Pass *ALL views* (Global + Local) through the **Student**.
3.  **The Objective**: The Student must predict what the Teacher outputted. Specifically, the Student looking at a small *Local View* (e.g., just the dog's ear) must predict the global representation produced by the Teacher looking at the *Global View* (the whole dog).
4.  **No Backpropagation for Teacher**: The Student's weights are updated using standard gradient descent (Cross-Entropy loss against the Teacher's output). The Teacher is **not** updated with backpropagation. Instead, the Teacher's weights are updated via an **Exponential Moving Average (EMA)** of the Student's weights.

$$ \theta_{teacher} \leftarrow \lambda \theta_{teacher} + (1 - \lambda) \theta_{student} $$
*(where $\lambda$ follows a cosine schedule from 0.996 to 1)*

### DINOv2 and Semantic Layouts
DINOv2 scaled this approach massively. The resulting ViT embeddings contain explicitly semantic layouts. Without any training, PCA applied to the patch embeddings of a DINOv2 model will naturally separate objects from backgrounds, and even correspond the same parts across different images (e.g., the front left wheel of a car in image A will have the same embedding as the front left wheel of a completely different car in image B).

This makes DINOv2 an incredible backbone for downstream tasks like depth estimation, semantic segmentation, and image retrieval.

---

## 3. Using Foundation Models

Instead of training a ResNet from scratch, modern CV pipelines often look like this:

```python
import torch

# 1. Load a pre-trained Foundation Model (e.g., DINOv2)
# We load the model but freeze its weights
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2.eval() # Set to evaluation mode
for param in dinov2.parameters():
    param.requires_grad = False # Freeze weights

# 2. Extract features for your dataset
# image shape: [batch_size, 3, 224, 224]
with torch.no_grad():
    features = dinov2(images) # Shape: [batch_size, 768]

# 3. Train a very small, fast classifier on top of the features
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(max_iter=1000)
classifier.fit(features.cpu().numpy(), labels)
```

Because foundation models have already learned excellent representations of the visual world, this simple Logistic Regression on top of DINOv2 features will often outperform a ResNet50 trained from scratch on your specific dataset, while taking a fraction of the time to train.

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Load DINOv2 using PyTorch Hub and extract embeddings for 5 different images.
- 🟡 **Intermediate**: Use SAM to segment an image using 3 different visual prompts (point, box, text) and visualize the output masks.

### What's Next
| Next | Why |
|------|-----|
| [Text Preprocessing](../12-NLP/01-Text-Preprocessing.md) | Shift your focus to Natural Language Processing fundamentals. |

---

[← CV Projects](10-CV-Real-World-Projects.md) | [Back to Index](../README.md) | [Next: Text Preprocessing →](../12-NLP/01-Text-Preprocessing.md)
