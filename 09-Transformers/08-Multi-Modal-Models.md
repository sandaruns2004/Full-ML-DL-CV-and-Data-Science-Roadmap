# 🖼️ Multi-Modal Models

> **Prerequisites**: Transformers, CNNs | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Contrastive Alignment: CLIP](#1-contrastive-alignment-clip)
2. [Projection Integration: LLaVA](#2-projection-integration-llava)
3. [Cross-Attention Fusion: Flamingo](#3-cross-attention-fusion-flamingo)
4. [Summary of Paradigms](#summary-of-paradigms)

---

Traditional neural networks operate on a single modality: text (LLMs), images (CNNs/ViTs), or audio (Wav2Vec). However, human intelligence is inherently multi-modal. We understand the world by combining sights, sounds, and text.

**Multi-Modal Models (VLM: Vision-Language Models)** bridge these domains, allowing a single model to take images and text as input and generate text (or images) as output.

There are three primary architectural paradigms for building multi-modal models:
1.  **Contrastive Alignment** (e.g., CLIP)
2.  **Projection/Adapter Integration** (e.g., LLaVA)
3.  **Cross-Attention Fusion** (e.g., Flamingo)

---

## 1. Contrastive Alignment: CLIP

OpenAI's **CLIP** (Contrastive Language-Image Pretraining) is arguably the most important foundation model for modern multi-modal AI. It doesn't generate text; instead, it learns a shared embedded space where images and their corresponding text descriptions are close together.

### Architecture
CLIP consists of two separate encoders:
1.  **Image Encoder**: Usually a Vision Transformer (ViT) or ResNet.
2.  **Text Encoder**: A standard Transformer encoder.

### Contrastive Loss (InfoNCE)
Given a batch of $N$ (image, text) pairs, CLIP computes the image embeddings $I_1, ..., I_N$ and text embeddings $T_1, ..., T_N$.

It then calculates the cosine similarity for all $N \times N$ possible pairings. The objective is to maximize the cosine similarity of the $N$ correct pairs (the diagonal of the matrix) while minimizing the cosine similarity of the $N^2 - N$ incorrect pairs.

$$ \text{Similarity Matrix} = I \times T^T $$

The loss function applies symmetric Cross-Entropy over the rows and columns of this similarity matrix.

```python
import torch
import torch.nn.functional as F

def clip_loss(image_embeddings, text_embeddings, temperature):
    # Normalize embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Calculate cosine similarity scaled by temperature
    logits = torch.matmul(image_embeddings, text_embeddings.T) * torch.exp(temperature)
    
    # Create labels (diagonal elements are the correct pairs)
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size).to(logits.device)
    
    # Symmetric loss
    loss_i = F.cross_entropy(logits, labels) # Image to Text
    loss_t = F.cross_entropy(logits.T, labels) # Text to Image
    
    return (loss_i + loss_t) / 2
```

**Why is CLIP so important?** The pre-trained CLIP image encoder is used as the "eyes" for almost every modern Visual Language Model (VLM) and Image Generator (Stable Diffusion).

---

## 2. Projection Integration: LLaVA

**LLaVA** (Large Language-and-Vision Assistant) is an architecture that gives a standard text-only LLM the ability to see. It is highly efficient because it leverages pre-trained models.

### Architecture
1.  **Vision Encoder**: A frozen, pre-trained CLIP ViT.
2.  **Language Model**: A frozen, pre-trained LLM (like Llama).
3.  **Projection Layer**: A trainable small neural network (usually a 2-layer MLP) that bridges the two.

### How it works
1. An image is passed through the frozen CLIP ViT, outputting a grid of visual features (e.g., 256 tokens of dimension 1024).
2. The trainable Projection Layer translates these 1024-dimensional visual tokens into the dimension expected by the LLM (e.g., 4096-dimensional text tokens).
3. These "visual text tokens" are concatenated with the actual text prompt tokens.
4. The LLM processes the combined sequence and generates a response.

During training, the LLM and the Vision Encoder are kept frozen. **Only the Projection Layer is updated**. This makes LLaVA incredibly cheap to train while yielding state-of-the-art results.

---

## 3. Cross-Attention Fusion: Flamingo

DeepMind's **Flamingo** takes a different approach to injecting visual information into an LLM. Instead of concatenating visual tokens at the input, it injects them deep inside the Transformer layers using Cross-Attention.

### Architecture
1.  **Vision Encoder**: A frozen pre-trained Normalizer-Free ResNet (NFNet).
2.  **Perceiver Resampler**: A module that takes the variable number of image features from the vision encoder and compresses them into a fixed number of visual tokens (e.g., 64 tokens) using cross-attention.
3.  **Gated Cross-Attention Dense Layers**: Inserted between the frozen self-attention and FFN layers of a pre-trained LLM.

### The Gated Cross-Attention Mechanism
Inside the LLM, the text tokens attend to the visual tokens via Cross-Attention:
*   Queries ($Q$) come from the text tokens.
*   Keys ($K$) and Values ($V$) come from the visual tokens.

To prevent destroying the LLM's pre-trained knowledge upon initialization, a `tanh` gate initialized to 0 multiplies the output of the cross-attention layer:

$$ y = x + \tanh(\alpha) \cdot \text{CrossAttention}(x, VisualTokens) $$

Initially, $\alpha=0$, so $\tanh(0)=0$. The cross-attention output is completely ignored, and the model behaves exactly like the original text-only LLM. As training progresses, $\alpha$ is learned, allowing the model to slowly integrate visual information.

---

## Summary of Paradigms

| Paradigm | Example | How it works | Primary Use Case |
| :--- | :--- | :--- | :--- |
| **Contrastive** | CLIP, ALIGN | Maps images and text to a shared space. | Zero-shot image classification, Image search, Stable Diffusion text conditioning. |
| **Projection** | LLaVA, BLIP-2 | Translates visual features into text embeddings and concatenates them. | Visual Question Answering, Image Captioning (efficient to train). |
| **Cross-Attention** | Flamingo, IDEFICS | Text tokens attend to visual tokens deep within the LLM layers. | Few-shot in-context visual learning. |

---

## Project Ideas & What's Next

### Project Ideas
- 🟢 **Beginner**: Load a pre-trained CLIP model using Hugging Face and write a script to match an image against 5 text prompts.
- 🟡 **Intermediate**: Implement a simple Projection Layer (2-layer MLP) and connect it to a dummy ViT output and a dummy LLM input embedding layer.

### What's Next
| Next | Why |
|------|-----|
| [RAG And Vector Databases](./09-RAG-And-Vector-Databases.md) | Learn how to connect LLMs to external data sources. |

---

[← Mixture of Experts (MoE)](07-Mixture-Of-Experts.md) | [Back to Index](../README.md) | [Next: RAG & Vector Databases →](09-RAG-And-Vector-Databases.md)
