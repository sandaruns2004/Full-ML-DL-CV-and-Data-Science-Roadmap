# 🎨 Data Augmentation

---

## 📋 Table of Contents
1. [Beginner: Creating New Training Examples on the Fly](#1-beginner-creating-new-training-examples-on-the-fly)
2. [Intermediate: Implementing Image Augmentation in PyTorch](#2-intermediate-implementing-image-augmentation-in-pytorch)
3. [Advanced: Mathematical Formulations of Affine Transformations](#3-advanced-mathematical-formulations-of-affine-transformations)

---

## 1. Beginner: Creating New Training Examples on the Fly

### Simple Intuition
Deep learning models need a lot of training data. If you train a model on a small dataset (e.g. 500 images of cars), the network will quickly memorize the exact pixel arrangements of those specific images and fail to recognize new cars (**overfitting**).
**Data Augmentation** is the process of applying random mathematical transformations (like rotating, zooming, flipping, or color shifting) to your existing images on the fly during training. This creates infinite variations of your dataset, forcing the model to learn the actual concepts (e.g. wheels, windows) rather than memorizing pixels.

### Real-World Analogy: Teaching a Toddler
Imagine teaching a child to recognize a "Cup":
- If you only show them a single blue cup resting upright on a table, they might think only blue, upright objects are cups.
- To teach them properly, you show them the cup from different angles, upside down, far away, and in different colors. 
Data augmentation does exactly this for a neural network.

---

## 2. Intermediate: Implementing Image Augmentation in PyTorch

In PyTorch, we apply data augmentation using `torchvision.transforms.v2`. 
> **Important Rule**: Data augmentation must **only** be applied to the **training set**. The validation and test sets must remain unaugmented (only resized and normalized) so we can accurately measure the model's real-world performance.

```python
from torchvision.transforms import v2
import torch

# 1. Training Augmentation Pipeline
train_transform = v2.Compose([
    # Resize the image
    v2.Resize((256, 256)),
    # Extract a random crop with zoom variation
    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    # Randomly flip the image horizontally (50% probability)
    v2.RandomHorizontalFlip(p=0.5),
    # Randomly jitter brightness, contrast, and saturation
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # Convert image to float tensor and scale to [0.0, 1.0]
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # Normalize with standard ImageNet statistics
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Validation / Test Pipeline (No random transformations!)
val_transform = v2.Compose([
    v2.Resize((256, 256)),
    v2.CenterCrop((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 3. Advanced: Mathematical Formulations of Affine Transformations

Spatial augmentations (rotation, translation, scaling, shearing) are mathematically represented as **Affine Transformations**. An affine transformation maps a coordinate $(x, y)$ in the input image to a new coordinate $(x', y')$ in the augmented image.

### Homogeneous Coordinates
To perform translation alongside rotation and scaling using a single matrix multiplication, we represent coordinates in **homogeneous coordinates** by adding a dummy dimension:
$$\mathbf{p} = \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}, \quad \mathbf{p}' = \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix}$$

The general affine transformation is:
$$\mathbf{p}' = \mathbf{M} \mathbf{p} = \begin{bmatrix} a & b & tx \\ c & d & ty \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

### Standard Transformation Matrices

1. **Translation** (shifting by $t_x, t_y$):
   $$\mathbf{M}_{\text{trans}} = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$$

2. **Scaling** (scaling by $s_x, s_y$):
   $$\mathbf{M}_{\text{scale}} = \begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

3. **Rotation** (rotating counter-clockwise by angle $\theta$ around the origin):
   $$\mathbf{M}_{\text{rot}} = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

### Image Resampling
When applying these transformations, the mapped coordinates $(x', y')$ may not land precisely on integer pixel grids. To resolve this, libraries apply **interpolation** techniques:
- **Nearest Neighbor**: Sets the value to the closest integer pixel coordinate. Fast but causes blocky artifacts.
- **Bilinear Interpolation**: Calculates the weighted average of the four nearest pixel values based on distance. This produces smooth results and is the default for most vision training tasks.

---

[← Transfer Learning](12-Transfer-Learning.md) | [Back to Index](../README.md) | [Next: Object Detection Introduction →](14-Object-Detection-Introduction.md)
