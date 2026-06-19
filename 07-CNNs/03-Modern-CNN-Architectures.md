# 🚀 Modern CNN Architectures

> **Prerequisites**: CNN Architecture Design (VGG, ResNet) | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Shift in Focus: Efficiency over Depth](#1-the-shift-in-focus-efficiency-over-depth)
2. [MobileNet (2017): Depthwise Separable Convolutions](#2-mobilenet-2017-depthwise-separable-convolutions)
3. [MobileNetV2 (2018): Inverted Residuals](#3-mobilenetv2-2018-inverted-residuals)
4. [EfficientNet (2019): Compound Scaling](#4-efficientnet-2019-compound-scaling)
5. [ConvNeXt (2022): The CNN strikes back](#5-convnext-2022-the-cnn-strikes-back)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The Shift in Focus: Efficiency over Depth

ResNet proved we could train extremely deep networks. The ensuing years saw a race to build larger ResNets (ResNet-152, ResNet-200), resulting in massive models that required server-grade GPUs to run inference.

However, the real-world deployment of Deep Learning shifted to **Edge Devices** (mobile phones, IoT sensors, self-driving cars). These devices have strict constraints on battery life, memory, and inference latency.

The modern era of CNNs focuses on mathematical optimizations to achieve ResNet-level accuracy using $10 \times$ fewer parameters and FLOPS.

---

## 2. MobileNet (2017): Depthwise Separable Convolutions

Created by Google, MobileNet introduced a fundamental rethinking of the convolution operation to dramatically reduce computation.

### Standard Convolution Cost
A standard Convolution filters spatial dimensions AND mixes channels simultaneously.
Let's assume an input of $32 \times 32$ with $64$ channels. We want to apply a $3 \times 3$ filter and output $128$ channels.
**Parameters**: $3 \times 3 \times 64 \times 128 = 73,728$

### The Depthwise Separable Solution
MobileNet splits this into two completely separate steps:

**1. Depthwise Convolution (Spatial filtering only):**
Apply a *single* $3 \times 3$ filter strictly to each individual channel. No mixing across channels.
- **Parameters**: $3 \times 3 \times 64 = 576$

**2. Pointwise Convolution (Channel mixing only):**
Apply a $1 \times 1$ convolution to mix the 64 channels into 128 channels.
- **Parameters**: $1 \times 1 \times 64 \times 128 = 8,192$

**Total Parameters**: $576 + 8,192 = 8,768$
**Reduction**: $73,728 \rightarrow 8,768$ ($\approx 8.4 \times$ fewer parameters!)

This mathematical trick performs almost exactly the same function as a standard convolution but is radically cheaper to run on mobile processors.

---

## 3. MobileNetV2 (2018): Inverted Residuals

Standard ResNets use "bottleneck" blocks. They use a $1 \times 1$ convolution to *compress* the channels, apply a $3 \times 3$ filter, and then use another $1 \times 1$ to *expand* the channels back out. This is a wide $\rightarrow$ narrow $\rightarrow$ wide design.

MobileNetV2 flipped this upside down.
Because Depthwise Convolutions are so ridiculously cheap, MobileNetV2 uses an **Inverted Residual Block**:
1. Take a narrow input (e.g., 24 channels).
2. **Expand** it using a $1 \times 1$ pointwise conv (e.g., to 144 channels).
3. Apply a cheap $3 \times 3$ depthwise conv in this high-dimensional space.
4. **Project** it back down using a $1 \times 1$ pointwise conv (e.g., back to 24 channels).

Narrow $\rightarrow$ Wide $\rightarrow$ Narrow. This uses minimal memory while still allowing the non-linearities to operate in a high-dimensional feature space.

---

## 4. EfficientNet (2019): Compound Scaling

Also from Google, EfficientNet formalized the process of making networks larger.

To get better accuracy, you can scale a network in three ways:
1. **Width**: Add more channels per layer.
2. **Depth**: Add more layers.
3. **Resolution**: Feed it higher-resolution input images (e.g., $512 \times 512$ instead of $224 \times 224$).

Historically, engineers just picked one arbitrarily (e.g., ResNet-50 to ResNet-152 scales strictly Depth).

EfficientNet proved that scaling one dimension without the others quickly yields diminishing returns. They developed a **Compound Scaling Method** that mathematically balances scaling Width, Depth, and Resolution simultaneously using a constant ratio ($\phi$).

They used Neural Architecture Search to find a tiny, optimal baseline network (EfficientNet-B0), and then used their Compound Scaling equation to scale it up to B1, B2... B7. 
**Result**: EfficientNet-B7 achieved state-of-the-art ImageNet accuracy while being 8.4x smaller and 6.1x faster than the best ResNet variant at the time.

---

## 5. ConvNeXt (2022): The CNN strikes back

In 2020/2021, Vision Transformers (ViTs) overtook CNNs as the absolute state-of-the-art for image classification, fueled by their success in NLP. ViTs require massive amounts of data to train.

In 2022, Meta (Facebook AI) asked a simple question: *What if we take a standard ResNet-50 and apply all the architectural tricks developed for Transformers to it?*

They created **ConvNeXt**. To upgrade ResNet, they made several changes:
1. **Macro design**: Changed the ratio of blocks in the stages to match Swin Transformers.
2. **ResNeXt-ify**: Swapped standard convolutions for Depthwise Separable Convolutions.
3. **Inverted Bottleneck**: Flipped the ResNet block to match MobileNetV2 / Transformer MLPs (Narrow $\rightarrow$ Wide $\rightarrow$ Narrow).
4. **Large Kernels**: Increased the $3 \times 3$ filters to massive $7 \times 7$ depthwise filters (to mimic the global attention of Transformers).
5. **Micro design**: Replaced ReLU with GELU, replaced BatchNorm with LayerNorm, and reduced the number of activation functions overall.

**The Result**: ConvNeXt completely bridged the gap, matching or outperforming Vision Transformers of the same size while retaining the simplicity and efficiency of pure CNNs.

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Implement Depthwise Separable Conv**: Write a custom `nn.Module` in PyTorch that replicates the MobileNet `DepthwiseSeparableConv`. Hint: Use `nn.Conv2d(..., groups=in_channels)` to achieve the depthwise step! Verify its parameter count against a standard `nn.Conv2d`.

### What's Next
| Next | Why |
|------|-----|
| [Transfer Learning](./04-Transfer-Learning.md) | We just learned about incredibly powerful, pre-trained architectures. Now let's learn how to download them and adapt them to our own custom datasets in minutes. |

---

[← CNN Architecture Design](./02-CNN-Architecture-Design.md) | [Back to Index](../README.md) | [Next: Transfer Learning →](./04-Transfer-Learning.md)
