# 🧩 Image Segmentation

> **Prerequisites**: CNNs, Object Detection | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [Semantic vs. Instance Segmentation](#1-semantic-vs-instance-segmentation)
2. [Transposed Convolutions (Up-Sampling)](#2-transposed-convolutions-up-sampling)
3. [The U-Net Architecture](#3-the-u-net-architecture)
4. [Mask R-CNN (Instance Segmentation)](#4-mask-r-cnn-instance-segmentation)
5. [Evaluation Metrics (Dice Loss, Boundary IoU)](#5-evaluation-metrics-dice-loss-boundary-iou)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. Semantic vs. Instance Segmentation

While Object Detection draws rough bounding boxes around objects, **Segmentation** classifies images at the *pixel level*.

### Semantic Segmentation
Every single pixel in the image is assigned a class label (e.g., $0=\text{Background}$, $1=\text{Person}$, $2=\text{Car}$).
- **The Catch**: It has no concept of individual objects. If two people are standing next to each other, Semantic Segmentation merges them into one giant "Person Blob".

### Instance Segmentation
Identifies individual objects *and* their pixel-perfect boundaries. It differentiates between "Person 1" and "Person 2".
- This is significantly harder, usually requiring a multi-stage approach combining Object Detection with Segmentation.

---

## 2. Transposed Convolutions (Up-Sampling)

In standard CNNs (ResNet, VGG), we use MaxPooling and Strided Convolutions to *down-sample* the spatial dimensions, compressing the image into a tiny vector of high-level features.

To perform pixel-level segmentation, we must take that tiny vector and *up-sample* it back to the original image dimensions.

**The Transposed Convolution** (sometimes incorrectly called Deconvolution):
Instead of reducing a $3 \times 3$ patch of pixels into $1$ value, a Transposed Convolution takes $1$ value and multiplies it by a learnable filter to broadcast it into a $3 \times 3$ patch.
By overlapping these patches, the network learns how to mathematically reconstruct spatial resolution from compressed feature maps.

---

## 3. The U-Net Architecture

Originally designed for Biomedical Image Segmentation (Ronneberger et al., 2015), U-Net is the undisputed king of Semantic Segmentation.

It has a symmetrical "U" shape consisting of three parts:

1. **The Contracting Path (Encoder)**: A standard CNN (e.g., ResNet-34) that repeatedly applies Convolutions and Max Pooling. It captures deep semantic *context* ("what is in this image?"), but loses spatial *resolution* ("where exactly is it?").
2. **The Expanding Path (Decoder)**: Uses Transposed Convolutions to up-sample the tiny feature maps back to the original image size.
3. **Skip Connections**: The brilliance of U-Net. The Decoder alone cannot guess where the exact pixel boundaries of an object are. Skip connections take the high-resolution feature maps from the Encoder and **concatenate** them directly into the Decoder. This provides the Decoder with the exact spatial details needed to draw perfect pixel-level boundaries.

### U-Net PyTorch Sketch

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=1):
        super(UNet, self).__init__()
        
        # 1. ENCODER
        self.enc_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2) # Halves spatial size
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        # 2. BOTTLENECK
        self.bottleneck = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # 3. DECODER
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Note: 128 (from upconv) + 128 (from skip connection) = 256 input channels!
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # 64 + 64 = 128
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # Output mask (1x1 conv to map to classes)
        self.final_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip1 = self.enc_conv1(x)
        x = self.pool1(skip1)
        
        skip2 = self.enc_conv2(x)
        x = self.pool2(skip2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with Skip Connections!
        x = self.upconv2(x)
        x = torch.cat([x, skip2], dim=1) # The U-Net magic
        x = self.dec_conv2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, skip1], dim=1) # The U-Net magic
        x = self.dec_conv1(x)
        
        return self.final_conv(x)
```

---

## 4. Mask R-CNN (Instance Segmentation)

To solve *Instance Segmentation*, Kaiming He et al. (2017) created **Mask R-CNN**. It is an extension of the Faster R-CNN object detector.

1. **Stage 1**: The Region Proposal Network proposes bounding boxes around objects.
2. **Stage 2**: ROI Align extracts feature maps for each bounding box.
3. **Stage 3 (The Split)**: 
   - Branch A: Classifies the object and tightens the bounding box.
   - **Branch B (The Mask Head)**: A tiny Fully Convolutional Network (FCN) processes the bounding box feature map and outputs a binary mask for the object inside that box.

Because the mask is generated *per bounding box*, Mask R-CNN inherently distinguishes between "Person 1" and "Person 2".

---

## 5. Evaluation Metrics (Dice Loss, Boundary IoU)

Standard Cross-Entropy Loss often fails in segmentation because of **Class Imbalance**. If you are segmenting a tumor in an MRI, 99% of the pixels are healthy tissue (background) and 1% is the tumor. The network will just predict "Background" everywhere and get 99% accuracy!

### Dice Loss (F1-Score for Pixels)
Dice Loss directly optimizes the overlap between the predicted mask and the ground truth mask, completely ignoring the massive background area.
$$\text{Dice Coefficient} = \frac{2 \times |X \cap Y|}{|X| + |Y|}$$
(where $X$ is the predicted mask pixels, and $Y$ is the ground truth mask pixels).

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Carvana Image Masking**: Download the Carvana dataset from Kaggle. Use `segmentation_models_pytorch` (a fantastic library built on top of `timm`) to initialize a U-Net with a pre-trained EfficientNet encoder. Train it using Dice Loss to automatically remove the background from images of cars.

### What's Next
| Next | Why |
|------|-----|
| [Pose Estimation](./06-Pose-Estimation.md) | We can find objects and draw their outlines. But can we track the human skeleton and understand posture? |

---

[← Object Detection](./04-Object-Detection.md) | [Back to Index](../README.md) | [Next: Pose Estimation →](./06-Pose-Estimation.md)
