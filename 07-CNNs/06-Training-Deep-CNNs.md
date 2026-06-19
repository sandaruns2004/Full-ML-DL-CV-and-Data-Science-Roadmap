# 🏋️‍♂️ Training Deep CNNs: The Modern Recipe

> **Prerequisites**: Optimizers, Transfer Learning, Augmentation | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The "Bag of Tricks" for Image Classification](#1-the-bag-of-tricks-for-image-classification)
2. [Mixed Precision Training (AMP)](#2-mixed-precision-training-amp)
3. [Gradient Accumulation](#3-gradient-accumulation)
4. [Learning Rate Warmup & Schedulers](#4-learning-rate-warmup--schedulers)
5. [Exponential Moving Average (EMA) of Weights](#5-exponential-moving-average-ema-of-weights)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The "Bag of Tricks" for Image Classification

In 2018, Amazon Web Services published a famous paper titled *Bag of Tricks for Image Classification*. They demonstrated that simply by tweaking the *training process*—without changing the ResNet-50 architecture at all—they could boost its accuracy on ImageNet from 75.3% to 79.29% (a massive leap).

The modern recipe for training deep CNNs isn't just `loss.backward()` and `optimizer.step()`. It involves a highly orchestrated symphony of memory optimization, mathematical tricks, and scheduling.

---

## 2. Mixed Precision Training (AMP)

Traditionally, Deep Learning uses 32-bit floating-point numbers (`float32`) for all computations.
However, modern GPUs have specialized hardware (Tensor Cores) designed to perform massive matrix multiplications using 16-bit floating-point numbers (`float16` or `bfloat16`) at **$2\times$ to $4\times$ the speed**.

**The Catch**: `float16` cannot represent very small numbers. The tiny gradients calculated during backpropagation will often underflow to `0.0`, destroying the training process.

**Automatic Mixed Precision (AMP)** solves this:
1. It stores the Master Weights in `float32`.
2. During the Forward Pass, it downcasts inputs and weights to `float16`, performing ultra-fast math.
3. During the Backward Pass, it calculates gradients in `float16`.
4. **Gradient Scaling**: It multiplies the loss by a massive number (e.g., $65536$) *before* backpropagation to prevent the gradients from underflowing to zero, then un-scales them before updating the `float32` Master Weights.

**PyTorch Implementation:**
```python
scaler = torch.cuda.amp.GradScaler() # Handles the gradient scaling

for inputs, labels in dataloader:
    optimizer.zero_grad()
    
    # Enable AMP for the forward pass
    with torch.cuda.amp.autocast():
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
    # Scales loss and calls backward
    scaler.scale(loss).backward()
    
    # Unscales gradients and updates weights
    scaler.step(optimizer)
    scaler.update()
```

---

## 3. Gradient Accumulation

To get stable gradients, you need a large batch size (e.g., 256). But what if a batch size of 256 requires 24GB of VRAM, and your GPU only has 8GB? You can only fit a batch size of 64.

**Gradient Accumulation** mathematically simulates a larger batch size without using extra VRAM.
Instead of updating the weights after every forward pass, we accumulate the gradients over multiple smaller forward passes, and only step the optimizer after we've simulated the large batch.

**PyTorch Implementation:**
```python
accumulation_steps = 4  # 64 batch size * 4 steps = 256 effective batch size!

for i, (inputs, labels) in enumerate(dataloader):
    # Forward pass
    outputs = model(inputs)
    # Divide loss by accumulation steps so the gradients average out correctly
    loss = criterion(outputs, labels) / accumulation_steps
    
    # Accumulate gradients (notice we do NOT zero_grad here!)
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad() # Now we clear them!
```

---

## 4. Learning Rate Warmup & Schedulers

When using AdamW or large batch sizes, the early gradients are highly noisy and unstable. If you start with a high learning rate, the network will take a massive step in the wrong direction, potentially ruining the carefully crafted initialization (or pre-trained weights).

**Linear Warmup**:
Start the learning rate at $0.0$, and linearly increase it to your target learning rate $\eta_{max}$ over the first 5 epochs (or 1,000 steps).

**Cosine Annealing**:
Once you reach $\eta_{max}$, smoothly decay the learning rate following a cosine curve down to $0.0$ by the end of training.

*This combination (Linear Warmup + Cosine Decay) is the undisputed industry standard for training all modern models (CNNs, LLMs, Diffusion).*

---

## 5. Exponential Moving Average (EMA) of Weights

The optimizer step constantly pushes the weights around, causing the loss to jitter slightly up and down. The final weights at the exact final epoch might not actually be the best possible representation of the learned function—they might just be at the top of a jitter.

**EMA** creates a "shadow copy" of the model.
Every step, the shadow model updates its weights to be a massive 99.9% of its *own* previous weights, and only 0.1% of the *current training model's* weights.

$$W_{ema} = 0.999 \times W_{ema} + 0.001 \times W_{train}$$

This results in an incredibly smooth, averaged set of weights. When training is finished, we throw away the training model and **use the EMA model for validation and deployment**. This consistently adds a free $0.5\% - 1.0\%$ boost to accuracy.

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **The Ultimate Training Loop**: Take the standard PyTorch training loop written in [Frameworks](../06-Neural-Networks-Foundations/08-Frameworks-Keras-PyTorch.md), and upgrade it. Add AMP (autocast + GradScaler), Gradient Accumulation (steps=4), and a `OneCycleLR` scheduler. Observe the GPU VRAM usage and training speed differences!

### What's Next
| Next | Why |
|------|-----|
| [Image Processing Fundamentals](../11-CV/01-Image-Processing-Fundamentals.md) | We've mastered Neural Networks. Now it's time to build the complete Computer Vision foundation, starting with classical image manipulation. |

---

[← Data Augmentation](./05-Data-Augmentation.md) | [Back to Index](../README.md) | [Next: RNN Fundamentals →](../08-Sequence-Models/01-RNN-Fundamentals.md)
