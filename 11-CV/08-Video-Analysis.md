# 🎞️ Video Analysis & 3D CNNs

> **Prerequisites**: CNNs, Sequence Models | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [The Challenge of Video Data](#1-the-challenge-of-video-data)
2. [CNN-RNN Hybrid Models](#2-cnn-rnn-hybrid-models)
3. [Two-Stream Networks (Optical Flow)](#3-two-stream-networks-optical-flow)
4. [3D Convolutions (I3D)](#4-3d-convolutions-i3d)
5. [Action Recognition vs. Action Detection](#5-action-recognition-vs-action-detection)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The Challenge of Video Data

A video is just a sequence of images (frames). Mathematically, it is a 4D tensor: $(Batch, Time, Channels, Height, Width)$.

**The Complexity Explosion**:
If a ResNet-50 requires 4GB of VRAM to train on a batch of 32 images...
Training it on a batch of 32 videos, where each video has 60 frames, requires $4 \times 60 = 240$ GB of VRAM!
Because of this, video analysis requires heavy temporal down-sampling (e.g., only analyzing 1 frame every second) and massive compute clusters.

---

## 2. CNN-RNN Hybrid Models

The most intuitive way to process a video:
1. Pass every individual frame through a standard 2D CNN (like ResNet) to extract a feature vector.
2. If the video has 100 frames, you now have a sequence of 100 feature vectors.
3. Feed this sequence into an LSTM or GRU to model the progression over time.

*Pros*: Very easy to implement. You can use pre-trained ImageNet CNNs instantly.
*Cons*: It doesn't truly capture temporal motion. The CNN processes a frame completely independently of the previous frame. It has no concept of "speed" or "direction" until the very end of the network.

---

## 3. Two-Stream Networks (Optical Flow)

To force the network to understand motion, researchers developed the **Two-Stream Architecture**.

1. **Spatial Stream**: A standard CNN that looks at a single RGB frame to understand *what* is in the video (e.g., a basketball, a hoop).
2. **Temporal Stream**: A second CNN that does NOT look at RGB pixels. It looks at **Optical Flow** frames. Optical Flow is a classical computer vision algorithm that calculates a 2D vector field showing exactly which pixels moved, and in what direction, between Frame $t$ and Frame $t+1$.
    - **Lucas-Kanade Method**: A popular differential method for optical flow estimation. It assumes that the flow is essentially constant in a local neighbourhood of the pixel under consideration, and solves the basic optical flow equations for all the pixels in that neighbourhood by the least squares criterion.
    - By explicitly passing this motion data to the CNN, the network doesn't have to learn how to deduce motion from raw pixels; it gets the motion map for free.
3. **Fusion**: The outputs of both CNNs are concatenated and passed into a final classifier.

*Result*: Extremely high accuracy for Action Recognition (e.g., classifying "playing basketball" vs "running"). However, computing dense optical flow for every frame at runtime is extremely computationally expensive.

---

## 4. 3D Convolutions (I3D)

Why use 2D CNNs and RNNs when we can just make the Convolution operation 3D?

A standard $3 \times 3$ filter slides across Height and Width.
A **3D Filter** ($3 \times 3 \times 3$) slides across Height, Width, AND Time!
When a 3D filter is applied, it simultaneously multiplies pixels across 3 consecutive frames, directly learning spatio-temporal features (e.g., it learns the exact visual pattern of a hand swiping left).

**The I3D Breakthrough (Inflated 3D ConvNets)**:
Training a deep 3D CNN from scratch is nearly impossible because it requires millions of labeled videos.
DeepMind solved this by "inflating" pre-trained 2D ImageNet weights. They took a standard 2D Inception network and literally stretched its $3 \times 3$ weights out into a $3 \times 3 \times 3$ cube by duplicating the values. This allowed the 3D CNN to start with perfect knowledge of objects, drastically reducing the video data needed to train.

---

## 4.5 SlowFast Networks

Developed by Facebook AI Research, the **SlowFast architecture** approaches video by treating spatial and temporal information asymmetrically, mimicking the human visual system (which has parvocellular cells for high-res color and magnocellular cells for high-speed motion).

It uses two distinct parallel pathways:
1. **The Slow Pathway**: Operates at a low frame rate (e.g., processes 1 frame every second). It has a large number of channels to capture highly detailed spatial semantics (colors, textures, identities).
2. **The Fast Pathway**: Operates at a high frame rate (e.g., processes 15 frames every second). Crucially, it has a very low channel capacity (e.g., only 1/8th of the channels of the Slow pathway). This makes it extremely lightweight, allowing it to capture rapid motion without slowing down the entire system.
3. **Lateral Connections**: The Fast pathway continually fuses its motion features into the Slow pathway at various depths of the network.

---

## 5. Action Recognition vs. Action Detection

- **Action Recognition**: The video is 5 seconds long. Assign one label to the whole video (e.g., "Swimming").
- **Temporal Action Detection**: The video is 2 hours long. Output the exact timestamps where an action occurs (e.g., "Goal scored from 14:02 to 14:06").
- **Spatio-Temporal Action Detection**: Output the exact timestamps AND draw a bounding box around the person performing the action in every frame.

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **SlowFast Networks**: Facebook AI designed the SlowFast architecture, which processes a video using two pathways: a "Slow" pathway (processes 1 frame per second to capture deep spatial semantics) and a "Fast" pathway (processes 15 frames per second using very cheap, lightweight convolutions to capture rapid motion). Read the original SlowFast paper and write a one-page summary of how lateral connections fuse the two pathways!

### What's Next
| Next | Why |
|------|-----|
| [3D Vision](./09-3D-Vision.md) | Videos add the dimension of time. Now let's add the dimension of depth. How do self-driving cars "see" in 3D? |

---

[← Image Generation](./07-Image-Generation.md) | [Back to Index](../README.md) | [Next: 3D Vision →](./09-3D-Vision.md)
