# ⚔️ Generative Adversarial Networks (GANs): Fundamentals

> **Prerequisites**: PyTorch Basics, Binary Cross Entropy | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [What is a GAN? The Counterfeiter and the Cop](#1-what-is-a-gan-the-counterfeiter-and-the-cop)
2. [The Minimax Game: Mathematics of GANs](#2-the-minimax-game-mathematics-of-gans)
3. [Building a Simple GAN in PyTorch](#3-building-a-simple-gan-in-pytorch)
4. [Training Dynamics and Pitfalls](#4-training-dynamics-and-pitfalls)

---

## 1. What is a GAN? The Counterfeiter and the Cop

Introduced by Ian Goodfellow in 2014, **Generative Adversarial Networks (GANs)** represent a paradigm shift in generative AI. Instead of trying to explicitly model the probability distribution of the data (like VAEs do), GANs learn to generate data through a game between two competing neural networks:

1. **The Generator ($G$):** The "counterfeiter". It takes random noise as input and tries to generate fake data that looks real.
2. **The Discriminator ($D$):** The "cop". It takes data (either real data from the dataset or fake data from the generator) and tries to predict whether it is real ($1$) or fake ($0$).

They are trained together in an adversarial zero-sum game. As $D$ gets better at catching fakes, $G$ must get better at forging data to fool $D$.

---

## 2. The Minimax Game: Mathematics of GANs

The objective function of a GAN is a **minimax game**. The Discriminator wants to maximize the objective (maximize its accuracy), while the Generator wants to minimize it (minimize the Discriminator's accuracy on fake data).

The Value function $V(D, G)$ is defined as:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

Let's break this down:
- $D(x)$: The discriminator's estimate of the probability that real data instance $x$ is real.
- $E_{x}[\log D(x)]$: The discriminator wants to maximize this by outputting $1$ for real data.
- $D(G(z))$: The discriminator's estimate of the probability that a fake instance $G(z)$ is real.
- $E_{z}[\log(1 - D(G(z)))]$: The discriminator wants to maximize this by outputting $0$ for fake data.
- The Generator $G$ wants to minimize $E_{z}[\log(1 - D(G(z)))]$ by forcing $D(G(z))$ to be close to $1$ (fooling the discriminator).

### The Non-Saturating Generator Loss
In practice, minimizing $\log(1 - D(G(z)))$ causes vanishing gradients early in training when the Discriminator is very good and the Generator is very bad. 
Instead, we train the generator to **maximize $\log(D(G(z)))$**. This provides much stronger gradients early in training.

---

## 3. Building a Simple GAN in PyTorch

Let's implement a simple GAN to generate 2D points or flattened 1D images.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. The Generator (Noise z -> Fake Data x_hat)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Tanh() # Output scaled between [-1, 1]
        )
        
    def forward(self, z):
        return self.net(z)

# 2. The Discriminator (Data x -> Probability of being Real)
class Discriminator(nn.Module):
    def __init__(self, input_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # Output probability [0, 1]
        )
        
    def forward(self, x):
        return self.net(x)

# Setup
G = Generator()
D = Discriminator()

# Use Binary Cross Entropy Loss
criterion = nn.BCELoss()

# Optimizers (Adam is standard for GANs, usually with a lower beta1)
opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

def train_step(real_data, batch_size):
    # Create labels
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)
    
    # ---------------------
    #  Train Discriminator
    # ---------------------
    opt_D.zero_grad()
    
    # Loss on real data (Max log(D(x)))
    pred_real = D(real_data)
    loss_D_real = criterion(pred_real, real_labels)
    
    # Loss on fake data (Max log(1 - D(G(z))))
    z = torch.randn(batch_size, 100)
    fake_data = G(z)
    # Note: We detach fake_data so we don't backprop through G here
    pred_fake = D(fake_data.detach())
    loss_D_fake = criterion(pred_fake, fake_labels)
    
    loss_D = loss_D_real + loss_D_fake
    loss_D.backward()
    opt_D.step()
    
    # -----------------
    #  Train Generator
    # -----------------
    opt_G.zero_grad()
    
    # We want D to classify fake data as Real (Max log(D(G(z))))
    # Notice we don't detach fake_data here!
    pred_fake_for_G = D(fake_data)
    loss_G = criterion(pred_fake_for_G, real_labels) # Using real labels to fool D!
    
    loss_G.backward()
    opt_G.step()
    
    return loss_D.item(), loss_G.item()
```

---

## 4. Training Dynamics and Pitfalls

Training GANs is notoriously unstable. Unlike standard supervised learning where loss smoothly decreases, GAN training is an oscillating balance of power.

### Common Failure Modes
1. **Mode Collapse:** The Generator discovers one specific output that consistently fools the Discriminator. Instead of generating diverse outputs, it produces the exact same output (or a small set of outputs) over and over again.
2. **Vanishing Gradients:** If the Discriminator becomes too good too quickly, it outputs exactly $0$ for fakes and $1$ for reals. The gradients vanish, and the Generator learns nothing.
3. **Non-Convergence:** The model parameters oscillate wildly and never reach a Nash Equilibrium.

### How to Stabilize GANs
- **LeakyReLU:** Use LeakyReLU in the Discriminator to prevent dead gradients.
- **Batch Normalization:** Essential in deeper architectures like DCGAN.
- **Label Smoothing:** Instead of target labels of `1.0` for real, use `0.9`. This prevents the discriminator from becoming overconfident.
- **Wasserstein Loss:** A completely different mathematical formulation (WGAN) that solves vanishing gradients and provides a meaningful loss metric.

---

## 5. Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Implement a basic GAN to generate 1D data points following a normal distribution.
- 🟡 **Intermediate**: Train a GAN on the Fashion-MNIST dataset. Implement label smoothing to stabilize the training process.

### What's Next
Basic GANs use dense layers, which are terrible for generating high-resolution images. In the next section, we look at Advanced GANs, including Deep Convolutional GANs (DCGAN) and Conditional GANs.

| Next Topic | Why |
|------------|-----|
| [Advanced GANs](./04-Advanced-GANs.md) | How to use convolutions to generate high-resolution images and how to control what the GAN generates. |

---

[← Variational Autoencoders](./02-Variational-Autoencoders.md) | [Back to Index](../README.md) | [Next: Advanced GANs →](./04-Advanced-GANs.md)
