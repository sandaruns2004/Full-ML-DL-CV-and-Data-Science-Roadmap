# 🕶️ 3D Vision & Point Clouds

> **Prerequisites**: CNNs | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [Beyond Pixels: Representing 3D Data](#1-beyond-pixels-representing-3d-data)
2. [PointNets: Deep Learning on Point Clouds](#2-pointnets-deep-learning-on-point-clouds)
3. [Voxel-Based Networks](#3-voxel-based-networks)
4. [Neural Radiance Fields (NeRFs)](#4-neural-radiance-fields-nerfs)
5. [Project Ideas & What's Next](#5-project-ideas--whats-next)

---

## 1. Beyond Pixels: Representing 3D Data

Standard cameras project a 3D world onto a flat 2D grid of RGB pixels. This loses all depth information.
Self-driving cars and robotics require absolute spatial awareness, so they use **LiDAR** (Light Detection and Ranging) and **RGB-D cameras** (Depth cameras) to capture true 3D data.

How do we represent 3D data for a Neural Network?
1. **Voxels**: A 3D grid of cubes (like Minecraft). Easiest to apply 3D Convolutions to, but wildly memory inefficient (most of the 3D space is just empty air).
2. **Point Clouds**: A raw list of $(X, Y, Z)$ coordinates floating in space. Extremely memory efficient, but unstructured (no grid).
3. **Meshes**: Vertices and polygonal faces. Great for graphics, terrible for Deep Learning.

---

## 2. PointNets: Deep Learning on Point Clouds

A point cloud is an unordered set of $N$ points, each with an $(x, y, z)$ coordinate.
Standard CNNs fail here because there is no grid to slide a filter over. Furthermore, if you shuffle the list of points, the 3D object doesn't change, so the network MUST be permutation-invariant!

**PointNet (2017)** solved this elegantly:
1. Pass *each individual point* through a shared Multi-Layer Perceptron (MLP) to map the 3D coordinate into a high-dimensional feature space (e.g., 1024 dimensions).
2. Apply a symmetric **Max Pooling** operation across all $N$ points. Because Max Pooling just takes the largest value regardless of order, it perfectly solves the permutation-invariance problem!
3. The result is a single "Global Feature Vector" that describes the entire 3D shape, which is passed into a classifier.

**PointNet++ (2017)**:
The original PointNet looks at all points independently until the final max pool. It has no concept of "local neighborhoods" (like a 3x3 convolution has for pixels). PointNet++ solves this by applying PointNet recursively on nested partitioning of the input point set. It samples a set of "centroid" points, groups neighboring points around them within a spherical radius, and applies a mini-PointNet to each local group. This captures fine-grained local structures.

---

## 3. Voxel-Based Networks

To use convolutions, we must convert point clouds into Voxels.
Because 99% of Voxels are empty space, we use **Sparse Convolutions**.
A Sparse Convolution algorithm keeps a hash map of only the "active" voxels (voxels that actually contain points). It only performs the math on those specific voxels, completely skipping the empty air. This allows Voxel networks to process massive outdoor LiDAR scans in real-time.

---

## 4. Neural Radiance Fields (NeRFs)

Introduced in 2020, NeRFs revolutionized 3D generation.

Instead of outputting a 3D mesh, a NeRF uses a Neural Network to implicitly encode a 3D scene inside its weights.

**How it works**:
1. You take 50 photos of a statue from different angles.
2. You ask the network: "If I am standing at coordinate $(x,y,z)$ and looking in direction $(\theta, \phi)$, what color and opacity will I see?"
3. The network (a massive MLP) takes the 5D coordinate and outputs an RGB color and a density value.
4. Using classical Ray Marching, you shoot millions of rays through the scene, query the network at points along the ray, and mathematically composite the colors to render a 2D image.

The network is trained by rendering an image from the exact angle of one of your 50 photos, and calculating the pixel loss between the render and the real photo. Once trained, you can render the scene from *any* completely novel angle!

---

## 5. 3D Gaussian Splatting (2023)

While NeRFs produce incredible quality, evaluating a massive MLP millions of times for every single frame is incredibly slow. In 2023, **3D Gaussian Splatting** overtook NeRFs as the state-of-the-art.

Instead of an implicit neural network, Gaussian Splatting uses an explicit representation: millions of 3D Gaussian blobs floating in space.
1. Each blob has a position $(x, y, z)$, a covariance matrix (defining its 3D shape/scale/rotation), an opacity, and spherical harmonics (which define how its color changes depending on the viewing angle).
2. To render an image, you mathematically "splat" (project) these 3D blobs onto the 2D camera plane.
3. Because the splatting math is completely differentiable, you can start with a random cloud of Gaussians and use gradient descent to optimize their positions, shapes, and colors until the rendered image matches your reference photos.

*Result*: Photorealistic 3D rendering at 100+ Frames Per Second on a standard GPU, completely eliminating the need for ray-marching MLPs!

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Instant-NGP**: NVIDIA released an open-source library called `instant-ngp` that trains a NeRF in 5 seconds instead of 5 hours using hash encoding. Take a 30-second video walking slowly around an object in your room. Use `ffmpeg` to extract the frames, and follow the `instant-ngp` tutorial to train a photorealistic 3D model of your object!

### What's Next
| Next | Why |
|------|-----|
| [Real-World CV Projects](./10-CV-Real-World-Projects.md) | You've learned everything from pixels to 3D point clouds. Let's finish this phase with a capstone project. |

---

[← Video Analysis](08-Video-Analysis.md) | [Back to Index](../README.md) | [Next: CV Projects →](10-CV-Real-World-Projects.md)
