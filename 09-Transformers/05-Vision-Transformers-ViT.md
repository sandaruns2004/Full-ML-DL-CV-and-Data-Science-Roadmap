# 👁️ Vision Transformers (ViT)

> **Prerequisites**: Transformer Architecture, CNN Fundamentals | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Downfall of CNNs?](#1-the-downfall-of-cnns)
2. [An Image is Worth 16x16 Words](#2-an-image-is-worth-16x16-words)
3. [The ViT Architecture Deep Dive](#3-the-vit-architecture-deep-dive)
4. [Deep Mathematics of Patch Embeddings](#4-deep-mathematics-of-patch-embeddings)
5. [From-Scratch Implementation (PyTorch)](#5-from-scratch-implementation-pytorch)
6. [Why ViTs Work: Global vs Local Receptive Fields](#6-why-vits-work-global-vs-local-receptive-fields)
7. [Advanced Variants (Swin, DeiT)](#7-advanced-variants-swin-deit)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. The Downfall of CNNs?

For nearly a decade (2012-2020), Convolutional Neural Networks (CNNs) like ResNet and EfficientNet completely dominated Computer Vision. CNNs have a built-in "inductive bias"—they inherently assume that pixels close to each other are related (locality) and that a cat in the top-left is the same as a cat in the bottom-right (translation invariance).

In 2020, Google researchers published **"An Image is Worth 16x16 Words"**, introducing the **Vision Transformer (ViT)**. 

Their radical idea: **Throw away convolutions entirely.** If a standard Transformer can learn the grammar of language just by looking at sequences of words, maybe it can learn the grammar of images by looking at sequences of image patches. Given enough data, the Transformer can learn locality and translation invariance from scratch without needing them hardcoded as convolutions!

---

## 2. An Image is Worth 16x16 Words

The standard Transformer expects a sequence of 1D word embeddings (e.g., $N=512$ words, each $d=768$ dimensions). 
An image is a 3D tensor: $H \times W \times C$ (Height $\times$ Width $\times$ Channels). How do we feed an image into a Transformer?

**The Solution: Patching**
1. Take an image (e.g., $224 \times 224 \times 3$).
2. Chop it up into a grid of non-overlapping patches (e.g., $16 \times 16$ pixel patches).
   - This creates a $14 \times 14$ grid of patches ($196$ total patches).
3. Flatten each $16 \times 16 \times 3$ patch into a 1D vector of length $768$.
4. Pass these 196 vectors through a Linear Layer to get 196 **Patch Embeddings**.
5. Prepend a special `[CLS]` token (making it 197 tokens).
6. Add Positional Embeddings to all 197 tokens (so the model knows where each patch came from).
7. Feed this sequence into a standard Transformer Encoder!

---

## 3. The ViT Architecture Deep Dive

Let's break down the exact pipeline of a Vision Transformer classifying an image of a dog.

### 3.1 The [CLS] Token
Just like in BERT, we add an extra, learnable embedding vector at the very beginning of the sequence (token 0). Since all tokens communicate via Self-Attention, this `[CLS]` token will aggregate information from all 196 image patches. After the final Transformer layer, we throw away the 196 patch outputs and only pass the `[CLS]` output into a classification head (an MLP) to predict "Dog".

### 3.2 1D Positional Embeddings
Unlike CNNs, Transformers process all patches simultaneously. Without positional embeddings, the model wouldn't know if a patch came from the top-left or bottom-right. 
ViT uses **Learnable 1D Positional Embeddings**. Surprisingly, the model naturally learns to arrange these 1D embeddings into a 2D grid structure purely from the data!

### 3.3 The Transformer Encoder
The Encoder is exactly the same as the original NLP Transformer. It consists of alternating layers of:
- Multi-Head Self Attention (MSA)
- Multilayer Perceptron (MLP)
- Layer Normalization (LN) applied *before* every block (Pre-Norm).

$$ z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1} $$
$$ z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l $$

---

## 4. Deep Mathematics of Patch Embeddings

Let's formalize the patch extraction and embedding process.

Let the input image be $\mathbf{x} \in \mathbb{R}^{H \times W \times C}$.
We extract patches $\mathbf{x}_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where:
- $P$ is the patch size (e.g., 16).
- $N = \frac{HW}{P^2}$ is the total number of patches (sequence length).

Each flattened patch is multiplied by a learnable embedding matrix $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$, where $D$ is the constant latent vector size used throughout the Transformer (e.g., 768).

$$ \mathbf{z}_0 = [\mathbf{x}_{class}; \mathbf{x}_{p}^1 \mathbf{E}; \mathbf{x}_{p}^2 \mathbf{E}; \dots ; \mathbf{x}_{p}^N \mathbf{E}] + \mathbf{E}_{pos} $$

Where:
- $\mathbf{x}_{class} \in \mathbb{R}^{1 \times D}$ is the learnable `[CLS]` token.
- $\mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}$ are the learnable positional embeddings.
- $\mathbf{z}_0$ is the final input sequence to the Transformer.

*(Fun fact: Instead of manually chopping and flattening, you can implement the entire patching and linear projection step simultaneously using a single 2D Convolution with `kernel_size=P` and `stride=P`!)*

---

## 5. From-Scratch Implementation (PyTorch)

Let's implement the Patch Embedding and a basic ViT in PyTorch.

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # We use a Conv2d to efficiently chop patches and project them to embed_dim!
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width) -> (B, 3, 224, 224)
        x = self.proj(x) 
        # Output shape: (B, embed_dim, H/patch_size, W/patch_size) -> (B, 768, 14, 14)
        
        # Flatten the spatial dimensions: (B, 768, 196)
        x = x.flatten(2) 
        
        # Transpose to get sequence length in the middle: (B, 196, 768)
        x = x.transpose(1, 2) 
        return x

class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable Positional Embeddings (num_patches + 1 for CLS)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # The Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads, 
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Get Patch Embeddings
        x = self.patch_embed(x) # (B, 196, 768)
        
        # 2. Add CLS Token to every sequence in the batch
        cls_tokens = self.cls_token.expand(B, -1, -1) # (B, 1, 768)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 197, 768)
        
        # 3. Add Positional Embeddings
        x = x + self.pos_embed
        
        # 4. Pass through Transformer
        x = self.transformer(x) # (B, 197, 768)
        
        # 5. Extract the CLS token output (the 0th token)
        cls_output = x[:, 0] # (B, 768)
        
        # 6. Classify
        return self.mlp_head(cls_output) # (B, 1000)

# Test it!
model = SimpleViT()
dummy_image = torch.randn(2, 3, 224, 224) # Batch of 2 images
output = model(dummy_image)
print(f"Output shape: {output.shape}") # Should be (2, 1000)
```

---

## 6. Why ViTs Work: Global vs Local Receptive Fields

Why do ViTs eventually outperform CNNs on huge datasets (like JFT-300M)?

- **CNNs have a Local Receptive Field**: In early layers, a CNN can only look at a $3 \times 3$ pixel area. It takes many layers to build up a "global" view of the image.
- **ViTs have a Global Receptive Field**: In the very first Self-Attention layer, Patch 1 (top-left) can directly communicate with Patch 196 (bottom-right). The model has global context instantly.

**The Data Hunger Problem**: Because CNNs have hardcoded biases (locality), they train well on small datasets (like ImageNet's 1M images). Because ViTs learn *everything* from scratch, they perform worse than CNNs on small datasets, but their ceiling is much higher when trained on massive datasets (100M+ images).

---

## 7. Advanced Variants (Swin, DeiT)

The original ViT was great, but highly computationally expensive ($O(N^2)$ attention scaling) and required massive Google-scale datasets to train. Researchers quickly improved it:

### 7.1 DeiT (Data-efficient Image Transformers)
Meta (Facebook) figured out how to train a ViT on just ImageNet (1.2M images) without it overfitting. They used heavy data augmentation and **Knowledge Distillation** (using a pre-trained CNN as a "teacher" to guide the ViT).

### 7.2 Swin Transformer (Hierarchical ViT)
Standard ViTs keep the sequence length constant (196 patches) throughout all 12 layers. This makes them bad for dense prediction tasks like Object Detection or Segmentation, which require high-resolution feature maps.
Swin (Shifted Window) Transformer reintroduces CNN-like hierarchy. It starts with small patches, calculates attention only within local "windows", and gradually merges patches together in deeper layers. Swin won the ICCV 2021 best paper award and is widely used for modern computer vision.

---

## 8. Project Ideas & What's Next

### Project Ideas
- 🟢 **Image Classification with Hugging Face**: Use `transformers.ViTForImageClassification` to load `google/vit-base-patch16-224` and classify your own uploaded images.
- 🟡 **Attention Map Visualization**: Extract the self-attention weights from the last layer of a pre-trained ViT. Plot a heatmap over the original image to see exactly which patches the `[CLS]` token is looking at when making its decision. You'll see it perfectly highlights the subject of the image!

### What's Next
| Next | Why |
|------|-----|
| [LLM Fine-Tuning & RLHF](./06-LLM-Fine-Tuning-And-RLHF.md) | We've covered Encoder (BERT), Decoder (GPT), and Vision (ViT) Transformers. Now let's see how modern Decoder models are fine-tuned to become helpful chatbots like ChatGPT. |

---

[← GPT And Decoder Models](./04-GPT-And-Decoder-Models.md) | [Back to Index](../README.md) | [Next: LLM Fine Tuning And RLHF →](./06-LLM-Fine-Tuning-And-RLHF.md)
