# ☁️ Diffusion Models

> **Prerequisites**: Generative Models, U-Net Architecture | **Difficulty**: ⭐⭐⭐⭐⭐ Expert

---

## 📋 Table of Contents
1. [The Death of GANs](#1-the-death-of-gans)
2. [What is a Diffusion Model?](#2-what-is-a-diffusion-model)
3. [The Forward Process (Destroying Data)](#3-the-forward-process-destroying-data)
4. [The Reverse Process (Creating Data)](#4-the-reverse-process-creating-data)
5. [The U-Net Backbone](#5-the-u-net-backbone)
6. [Classifier-Free Guidance (CFG)](#6-classifier-free-guidance-cfg)
7. [Latent Diffusion (Stable Diffusion)](#7-latent-diffusion-stable-diffusion)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. The Death of GANs

From 2014 to 2020, Generative Adversarial Networks (GANs) were the undisputed kings of image generation. However, GANs have massive problems:
- **Mode Collapse**: They often only learn to generate a small subset of the training data.
- **Training Instability**: The Generator and Discriminator must be perfectly balanced, or the loss diverges to infinity.
- **Inflexibility**: Conditioning a GAN on highly complex text prompts (like "An astronaut riding a horse on Mars in the style of Picasso") is extremely difficult.

In 2020, **Denoising Diffusion Probabilistic Models (DDPMs)** burst onto the scene. By 2022, Diffusion Models (DALL-E 2, Midjourney, Stable Diffusion) had completely replaced GANs as the state-of-the-art for image, audio, and even video generation. 
They offer perfect training stability, massive diversity, and unmatched text-conditioning capabilities.

---

## 2. What is a Diffusion Model?

Diffusion models are inspired by non-equilibrium thermodynamics. If you drop a drop of blue dye into a glass of water, the dye slowly diffuses until the water is a uniform light blue. This process is easy to simulate, but reversing it (turning light blue water back into a single concentrated drop) seems impossible.

Diffusion Models learn to reverse this thermodynamic process for data.

**The Core Idea**:
1. Take a pristine, high-quality image.
2. Slowly add Gaussian noise to it over 1000 tiny steps, until it becomes pure TV static (pure noise). This is the **Forward Process**.
3. Train a Neural Network to predict and remove the noise that was added at each step. This is the **Reverse Process**.
4. To generate a new image, start with pure TV static and run the Reverse Process 1000 times. Out pops a photorealistic image!

---

## 3. The Forward Process (Destroying Data)

The Forward Process $q$ is a fixed Markov Chain. We don't train a neural network for this part; it's purely mathematical.

Given a real image $\mathbf{x}_0$, we add a small amount of Gaussian noise $\boldsymbol{\epsilon}$ at step $t$ to get a slightly noisier image $\mathbf{x}_t$. The variance of the noise is controlled by a schedule $\beta_1, \dots, \beta_T$.

$$ q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\mathbf{x}_{t-1}, \beta_t \mathbf{I}) $$

Instead of calculating this sequentially 1000 times, a beautiful mathematical property of Gaussians allows us to jump directly from $\mathbf{x}_0$ to any timestep $\mathbf{x}_t$ in one shot!

Let $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$.
$$ \mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon} $$
Where $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$.

At $t=1000$, $\bar{\alpha}_T \approx 0$, so $\mathbf{x}_{1000} \approx \boldsymbol{\epsilon}$ (pure standard normal noise).

---

## 4. The Reverse Process (Creating Data)

If we knew the exact reverse distribution $q(\mathbf{x}_{t-1} | \mathbf{x}_t)$, we could start with noise and denoise it back into an image. But calculating that probability requires knowing the entire distribution of all possible images in the universe, which is impossible.

So, we train a Neural Network $\boldsymbol{\epsilon}_\theta$ to approximate it!

We don't actually train the network to predict the clean image $\mathbf{x}_0$. Instead, **we train the network to predict the noise $\boldsymbol{\epsilon}$ that was added to the image.**

### The Loss Function
The loss function is unbelievably simple. It's just Mean Squared Error (MSE) between the actual noise we added, and the noise the network predicted!

$$ \mathcal{L}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ || \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) ||^2 \right] $$

**The Training Algorithm**:
1. Pick a random real image $\mathbf{x}_0$.
2. Pick a random timestep $t$ between 1 and 1000.
3. Sample random noise $\boldsymbol{\epsilon}$.
4. Corrupt the image to get $\mathbf{x}_t$.
5. Pass $\mathbf{x}_t$ and $t$ into the Neural Network.
6. Calculate MSE loss between the network's output and $\boldsymbol{\epsilon}$.

---

## 5. The U-Net Backbone

What architecture do we use for the Neural Network $\boldsymbol{\epsilon}_\theta$? 
The standard choice is a **U-Net**.

A U-Net takes an image, downsamples it through several Convolutional/ResNet blocks to a bottleneck (encoding spatial features), and then upsamples it back to the original resolution, using **Skip Connections** to preserve high-frequency details.

Crucially, the network needs to know *what timestep $t$ it is currently denoising*. (Denoising step 900 requires removing heavy static; denoising step 10 requires removing slight film grain). 
We encode the integer $t$ using Sinusoidal Positional Embeddings (just like Transformers) and inject it into every ResNet block of the U-Net.

---

## 6. Classifier-Free Guidance (CFG)

How do we tell the Diffusion Model to generate a "Cat"?
We inject text embeddings (e.g., from CLIP) into the U-Net using Cross-Attention layers.

However, sometimes the model ignores the text prompt and just generates a generic, high-quality image to minimize its loss. To force the model to listen to the prompt, we use **Classifier-Free Guidance (CFG)**.

During training, we randomly drop the text prompt 10% of the time (replacing it with an empty string).
During generation, we run the U-Net *twice* at every step:
1. Once with the text prompt: $\boldsymbol{\epsilon}_{cond}$
2. Once with an empty prompt: $\boldsymbol{\epsilon}_{uncond}$

We calculate the final noise prediction by pushing the unconditional prediction in the direction of the conditional prediction, scaled by a Guidance Scale $w$ (usually ~7.5):

$$ \hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_{uncond} + w \cdot (\boldsymbol{\epsilon}_{cond} - \boldsymbol{\epsilon}_{uncond}) $$

This mathematically forces the image to heavily align with the text prompt!

---

## 7. Latent Diffusion (Stable Diffusion)

Running a U-Net 1000 times on a $1024 \times 1024$ pixel image requires supercomputers.

In 2022, researchers invented **Latent Diffusion** (which became **Stable Diffusion**).
Instead of running the diffusion process in pixel space, they:
1. Trained a powerful Variational Autoencoder (VAE) to compress $512 \times 512$ images into a $64 \times 64 \times 4$ latent tensor.
2. Ran the entire Diffusion process (Forward and Reverse) on these tiny latent tensors!
3. Passed the final, denoised latent tensor through the VAE Decoder to get the $512 \times 512$ image back.

This reduced the computational cost by 99%, allowing state-of-the-art image generation to run on consumer graphics cards with just 8GB of VRAM!

---

## 8. Project Ideas & What's Next

### Project Ideas
- 🟢 **Hugging Face Diffusers**: Use the `diffusers` library in Python to load `StableDiffusionPipeline`. Write a script that takes a text prompt and generates an image using CFG scale = 7.5 and 50 inference steps.
- 🟡 **Train a DDPM from Scratch**: Build a tiny U-Net and train a DDPM on the $32 \times 32$ CIFAR-10 dataset or MNIST. Write the forward process logic, the MSE loss, and the reverse sampling loop. It is incredibly satisfying to watch the model turn pure static into a recognizable digit over 1000 steps.

### What's Next
| Next | Why |
|------|-----|
| [Neural Style Transfer](./06-Neural-Style-Transfer.md) | Before Diffusion and GANs ruled the world, researchers discovered how to use pretrained CNNs to rip the "style" from a Van Gogh painting and apply it to a photograph. We'll explore this classic computer vision technique. |

---

[← Advanced GANs](04-Advanced-GANs.md) | [Back to Index](../README.md) | [Next: Neural Style Transfer →](06-Neural-Style-Transfer.md)
