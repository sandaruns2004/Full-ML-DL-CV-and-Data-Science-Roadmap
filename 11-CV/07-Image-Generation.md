# 🪄 Image Generation (GANs & Diffusion)

> **Prerequisites**: CNNs, U-Net | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [Discriminative vs. Generative Models](#1-discriminative-vs-generative-models)
2. [Generative Adversarial Networks (GANs)](#2-generative-adversarial-networks-gans)
3. [The Mathematics of GAN Training](#3-the-mathematics-of-gan-training)
4. [Denoising Diffusion Probabilistic Models (DDPMs)](#4-denoising-diffusion-probabilistic-models-ddpms)
5. [Latent Diffusion (Stable Diffusion)](#5-latent-diffusion-stable-diffusion)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. Discriminative vs. Generative Models

- **Discriminative Models** (ResNet, YOLO): Given an image $X$, predict the label $Y$. Mathematically, they learn $P(Y|X)$.
- **Generative Models** (GANs, Diffusion): Learn the underlying distribution of the data itself, $P(X)$. They figure out the complex mathematical boundaries of "what makes an image look like a real photograph" so they can sample new, fake images from that distribution.

---

## 2. Generative Adversarial Networks (GANs)

Invented by Ian Goodfellow (2014), GANs frame image generation as a counterfeiter vs. police officer game.

It consists of two separate Neural Networks:
1. **The Generator ($G$)**: Takes a vector of random noise ($z$) and tries to up-sample it into a realistic image. (The Counterfeiter).
2. **The Discriminator ($D$)**: A standard binary image classifier (CNN) that looks at an image and predicts if it is "Real" (from the training set) or "Fake" (produced by the Generator). (The Police Officer).

They are trained simultaneously in a zero-sum game. The Generator gets better at faking images, forcing the Discriminator to get better at spotting fakes, which forces the Generator to create even better fakes.

---

## 3. The Mathematics of GAN Training

**The Value Function $V(D, G)$**:
$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))] $$

**How to Train**:
1. **Train $D$**: Feed it real images (target label $= 1$). Feed it fake images from $G$ (target label $= 0$). Calculate standard Binary Cross-Entropy (BCE) loss. Update $D$'s weights.
2. **Train $G$**: Generate fake images. Feed them to $D$. Calculate BCE loss *as if the target label were 1* (because $G$ wants to fool $D$). **Freeze $D$'s weights**, and backpropagate the error all the way back to $G$'s weights. Update $G$.

**Mode Collapse**:
GANs are notoriously unstable to train. A common failure mode is "Mode Collapse", where the Generator figures out how to draw one perfectly realistic image of a cat. It then stops learning and just outputs that exact same cat image every single time, completely tricking the Discriminator but failing to learn the true diversity of the dataset.

---

## 4. Denoising Diffusion Probabilistic Models (DDPMs)

In 2020, Diffusion models dethroned GANs. They are much more stable to train and produce significantly higher-quality images (Midjourney, DALL-E).

Diffusion works on a simple principle derived from thermodynamics:
1. **Forward Process (Adding Noise)**: Take a crisp, real image of a dog. Over $T$ steps (e.g., 1000 steps), slowly add Gaussian noise to the image until it is 100% pure static. (This requires no neural network, just a mathematical formula).
2. **Reverse Process (Denoising)**: Train a Neural Network to reverse the process. Give the network a noisy image at step $t$, and ask it to predict the exact noise that was added at step $t-1$.

**The Architecture**:
The network used for the Reverse Process is almost always a **U-Net** (because the input and output are both images of the exact same size).

**Inference (Generation)**:
Start with a $256 \times 256$ matrix of pure, random static. Pass it through the trained U-Net 1000 times. Each time, the U-Net subtracts a tiny bit of noise, hallucinating details along the way, until a photorealistic image emerges.

---

## 5. Latent Diffusion (Stable Diffusion)

Standard Diffusion is incredibly slow and VRAM-hungry because the U-Net has to operate on high-resolution $512 \times 512 \times 3$ images for 1000 steps.

**Stable Diffusion** solved this by performing diffusion in the *Latent Space*.
1. Train a **Variational Autoencoder (VAE)** to compress a $512 \times 512 \times 3$ image into a tiny $64 \times 64 \times 4$ "latent" matrix.
2. Perform the entire Diffusion Process (Forward and Reverse) on these tiny $64 \times 64$ matrices.
3. Once the tiny latent image is generated, use the VAE Decoder to expand it back into a crisp $512 \times 512$ image.

Because the U-Net is operating on matrices that are 48x smaller, Stable Diffusion can be run on consumer GPUs (e.g., an RTX 3060) in seconds.

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Hugging Face Diffusers**: Don't write Diffusion math from scratch. Install the `diffusers` library. Download the `runwayml/stable-diffusion-v1-5` model. Write a script that takes a text prompt ("A cyberpunk cat sitting on a neon car") and passes it through the model to generate the image. Explore how changing the `guidance_scale` and `num_inference_steps` alters the output!

### What's Next
| Next | Why |
|------|-----|
| [Video Analysis](./08-Video-Analysis.md) | We've mastered 2D images. What happens when we add the dimension of time and process continuous video streams? |

---

[← Pose Estimation](./06-Pose-Estimation.md) | [Back to Index](../README.md) | [Next: Video Analysis →](./08-Video-Analysis.md)
