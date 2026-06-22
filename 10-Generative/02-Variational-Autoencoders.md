# 🎲 Variational Autoencoders (VAEs)

> **Prerequisites**: Autoencoders, Probability Distributions | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Problem with Standard Autoencoders](#1-the-problem-with-standard-autoencoders)
2. [What is a Variational Autoencoder?](#2-what-is-a-variational-autoencoder)
3. [Deep Mathematics: The ELBO Derivation](#3-deep-mathematics-the-elbo-derivation)
4. [The Reparameterization Trick](#4-the-reparameterization-trick)
5. [From-Scratch Implementation (PyTorch)](#5-from-scratch-implementation-pytorch)
6. [Visualizing the Latent Space](#6-visualizing-the-latent-space)
7. [Beta-VAEs and Disentanglement](#7-beta-vaes-and-disentanglement)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. The Problem with Standard Autoencoders

A standard Autoencoder (AE) compresses an image into a low-dimensional **latent vector** $\mathbf{z}$ and then reconstructs it.
While great for compression and denoising, standard AEs are **terrible at generation**.

Why? Because the latent space is **discrete and disjointed**. 
If you feed a picture of a "7" into an AE, it might map it to the coordinate $[5.2, -3.1]$. If you feed an "8", it maps to $[12.9, 8.4]$. 
If you sample a random point from the latent space, say $[0, 0]$, and ask the Decoder to generate an image, it will likely output pure garbage. The AE never learned what the space *between* the numbers looks like.

To generate new, realistic data, we need a latent space that is **continuous, dense, and structured**.

---

## 2. What is a Variational Autoencoder?

A Variational Autoencoder (VAE) forces the latent space to behave like a standard probability distribution (usually a Gaussian/Normal distribution).

Instead of mapping an input image to a **single point** $\mathbf{z}$, the VAE Encoder maps the input image to a **probability distribution** over the latent space.

For every image, the Encoder outputs two vectors:
1. $\boldsymbol{\mu}$ (Mean): Where the distribution is centered.
2. $\boldsymbol{\sigma}$ (Standard Deviation): How spread out the distribution is.

We then randomly sample a point $\mathbf{z}$ from this specific distribution: $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$. 
Finally, the Decoder takes this sampled point and reconstructs the image.

### The Two Losses of a VAE
To train this, we need two loss functions:
1. **Reconstruction Loss** (e.g., MSE or Binary Cross-Entropy): Forces the decoded image to look like the original input.
2. **Kullback-Leibler (KL) Divergence**: Forces the Encoder's output distribution $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ to be as close as possible to a Standard Normal distribution $\mathcal{N}(0, 1)$.

The KL Divergence acts as a **regularizer**. Without it, the Encoder would just learn to output $\boldsymbol{\sigma} = 0$ (turning it back into a standard AE to minimize reconstruction loss). The KL loss forces the distributions to overlap, creating a smooth, continuous latent space where interpolating between two points yields a realistic transition.

---

## 3. Deep Mathematics: The ELBO Derivation

Let $\mathbf{x}$ be the data (images) and $\mathbf{z}$ be the latent variables. 
We want to maximize the probability of our data, $p(\mathbf{x})$.

$$ p(\mathbf{x}) = \int p(\mathbf{x}|\mathbf{z})p(\mathbf{z}) d\mathbf{z} $$

This integral is completely intractable. To solve this, we introduce an approximation to the true posterior, which we call our Encoder: $q_{\phi}(\mathbf{z}|\mathbf{x})$.

Using Jensen's Inequality, we can derive a lower bound on the log-likelihood of our data. This is called the **Evidence Lower Bound (ELBO)**:

$$ \log p(\mathbf{x}) \ge \mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})} [\log p_{\theta}(\mathbf{x}|\mathbf{z})] - \mathbb{D}_{KL}(q_{\phi}(\mathbf{z}|\mathbf{x}) || p(\mathbf{z})) $$

Let's break this down:
- The left side, $\log p(\mathbf{x})$, is what we want to maximize.
- The first term on the right, $\mathbb{E} [\log p_{\theta}(\mathbf{x}|\mathbf{z})]$, is the **Reconstruction Error**. (How likely is the Decoder to generate $\mathbf{x}$ given $\mathbf{z}$?).
- The second term, $\mathbb{D}_{KL}(q_{\phi} || p)$, is the **KL Divergence**. It measures how far our Encoder's distribution $q_{\phi}(\mathbf{z}|\mathbf{x})$ is from our prior $p(\mathbf{z})$ (which we assume is $\mathcal{N}(0, I)$).

### Closed-Form KL Divergence
Because we assume both distributions are Gaussian, the KL Divergence has a beautiful, closed-form analytical solution. We don't need to estimate it; we can calculate it exactly!

$$ \mathbb{D}_{KL} = - \frac{1}{2} \sum_{i=1}^{k} \left( 1 + \log(\sigma_i^2) - \mu_i^2 - \sigma_i^2 \right) $$

*(Note: In practice, neural networks predict the log variance ($\log(\sigma^2)$) instead of standard deviation for numerical stability, as variance must be positive, but a neural network output can be negative).*

---

## 4. The Reparameterization Trick

We have a massive problem. To train the VAE via backpropagation, gradients must flow from the Decoder, through the latent space, to the Encoder.

However, the latent space involves **random sampling**: $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$. 
**You cannot backpropagate through a random node!** The gradient of randomness is undefined.

**The Solution:** The Reparameterization Trick (Kingma & Welling, 2013).

Instead of sampling $\mathbf{z}$ directly from the complex distribution, we sample a random noise variable $\boldsymbol{\epsilon}$ from a Standard Normal distribution:
$$ \boldsymbol{\epsilon} \sim \mathcal{N}(0, 1) $$

Then, we deterministically scale and shift it using the Encoder's outputs:
$$ \mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon} $$

Now, the randomness ($\boldsymbol{\epsilon}$) is an external input that we don't need gradients for! The path from $\mathbf{z}$ to $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$ is entirely deterministic (just addition and multiplication), so backpropagation works perfectly.

---

## 5. From-Scratch Implementation (PyTorch)

Let's build a complete VAE in PyTorch to generate handwritten digits (MNIST).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # ENCODER
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # The encoder outputs TWO vectors: mean and log_variance
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # DECODER
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)
    
    def reparameterize(self, mu, logvar):
        # 1. Convert log variance to standard deviation
        std = torch.exp(0.5 * logvar)
        # 2. Sample standard normal noise
        eps = torch.randn_like(std)
        # 3. Scale and shift (Reparameterization Trick!)
        return mu + eps * std
        
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # Use sigmoid because MNIST pixels are normalized between 0 and 1
        return torch.sigmoid(self.fc4(h3))
        
    def forward(self, x):
        # Flatten image
        x_flat = x.view(-1, 784)
        
        # Encode to get distribution parameters
        mu, logvar = self.encode(x_flat)
        
        # Sample z
        z = self.reparameterize(mu, logvar)
        
        # Decode z to reconstruct image
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar

# The Loss Function (Negative ELBO)
def vae_loss(recon_x, x, mu, logvar):
    # 1. Reconstruction Loss (Binary Cross Entropy)
    # Using sum reduction to match the math equation
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # 2. KL Divergence Loss
    # Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss is the sum
    return BCE + KLD

# Test the model
model = VAE()
dummy_image = torch.rand(32, 1, 28, 28) # Batch of 32 MNIST images
recon, mu, logvar = model(dummy_image)
loss = vae_loss(recon, dummy_image, mu, logvar)
print(f"Reconstruction Shape: {recon.shape}") # (32, 784)
print(f"Total Loss (BCE + KLD): {loss.item():.4f}")
```

---

## 6. Visualizing the Latent Space

Because a VAE forces the latent space to be continuous, you can do **Latent Space Interpolation**. 

If you take the latent vector for a "2" ($\mathbf{z}_A$) and the latent vector for a "7" ($\mathbf{z}_B$), and draw a straight line between them in the 20-dimensional space, every point along that line can be decoded into an image. 

You won't see two overlapping images. Instead, you'll see a smooth, morphing animation where the top curve of the "2" slowly straightens out to become the top bar of the "7".

Furthermore, because the prior is $\mathcal{N}(0, 1)$, we can generate **completely new, fake images** just by sampling random vectors from a standard normal distribution and feeding them directly into the Decoder!

---

## 7. Beta-VAEs and Disentanglement

A major area of VAE research is **Disentangled Representations**. 
If you train a VAE on faces, you want the individual dimensions of the latent space to control specific, understandable features. 
- Dimension 1: Controls smile/frown.
- Dimension 2: Controls hair color.
- Dimension 3: Controls rotation.

A standard VAE entangles these features. To fix this, DeepMind introduced the **$\beta$-VAE**. 
They simply multiply the KL Divergence term by a constant $\beta > 1$.

$$ \mathcal{L} = \text{Reconstruction} + \beta \cdot \mathbb{D}_{KL} $$

By heavily punishing the KL divergence (e.g., $\beta=4$), the model is forced to be extremely efficient with its latent dimensions, naturally leading to individual neurons specializing in single, disentangled visual concepts.

---

## 8. Project Ideas & What's Next

### Project Ideas
- 🟢 **Generate Celeb Faces**: Download the `CelebA` dataset. Upgrade the PyTorch VAE above to use `Conv2d` layers in the Encoder and `ConvTranspose2d` layers in the Decoder. Train it to generate new celebrity faces.
- 🟡 **Latent Space Arithmetic**: Once trained on faces, find the average latent vector for "people with glasses" and subtract the vector for "people without glasses". Add this "glasses vector" to the latent vector of a person without glasses, and the Decoder will generate that same person wearing glasses!

### What's Next
| Next | Why |
|------|-----|
| [Advanced GANs](./04-Advanced-GANs.md) | VAEs are mathematically beautiful and stable, but their generated images are often **blurry** due to the MSE/BCE reconstruction loss. To get ultra-sharp, photorealistic images, we need Generative Adversarial Networks (GANs). |

---

[← Autoencoders](01-Autoencoders.md) | [Back to Index](../README.md) | [Next: GAN Fundamentals →](03-GAN-Fundamentals.md)
