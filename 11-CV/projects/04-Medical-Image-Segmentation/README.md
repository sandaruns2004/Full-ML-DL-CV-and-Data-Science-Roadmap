# 🩻 Project 04: Medical Image Segmentation

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Goal**: Pixel-perfect U-Net segmentation.

Leave bounding boxes behind. Build a pixel-perfect semantic segmentation model for healthcare using the U-Net architecture.

---

## The Implementation Roadmap

### 🟢 Beginner: The Data Pipeline
1. **Dataset**: Download the 2018 Data Science Bowl dataset (Nuclei) or a publicly available MRI tumor dataset.
2. **Dataloader**: Write a PyTorch Dataset class that loads the image and its corresponding binary mask. Crucially, apply the exact same resizing/cropping transforms to both the image and the mask.

### 🟡 Intermediate: The Architecture
3. **Build U-Net**: Implement the U-Net architecture from scratch in PyTorch. 
4. **Skip Connections**: Ensure that your forward pass correctly concatenates the high-resolution feature maps from the Encoder path into the Decoder path.
5. **Loss Function**: Binary Cross Entropy often fails if the tumor only occupies 2% of the image. Implement the **Dice Loss** or **IoU Loss** to force the network to focus on the boundary.

### 🔴 Advanced: Evaluation & Augmentation
6. **Extreme Augmentation**: Medical datasets are small. Implement aggressive data augmentations (Elastic Deformations, random rotations, flips) to prevent overfitting.
7. **Evaluation Dashboard**: Write an evaluation script that randomly samples 5 images from the test set and plots a matplotlib grid showing: `Original Image | True Mask | Model Prediction` side-by-side.
