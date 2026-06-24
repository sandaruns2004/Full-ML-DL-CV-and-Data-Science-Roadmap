# 14 - Multimodal Transformers

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Prerequisites**: 06-Transformer-Architecture | **Estimated Reading Time**: 20 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [The Universal API: Embeddings](#2-the-universal-api-embeddings)
3. [Contrastive Language-Image Pre-training (CLIP)](#3-contrastive-language-image-pre-training-clip)
4. [Vision-Language Models (VLMs)](#4-vision-language-models-vlms)
5. [Any-to-Any Modality (Gemini)](#5-any-to-any-modality-gemini)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

Until now, we have treated Transformers strictly as text-processing engines. We tokenize words, turn them into vectors, and use Self-Attention.

### 🟢 Beginner
If you show an AI a picture of a Stop Sign, and ask: *"What does this sign mean?"*, a text-only LLM cannot see the image. A standard Image Classification CNN (like ResNet) can see the image, but it cannot generate conversational text explaining *why* you should stop. We need an AI that can seamlessly blend vision and language.

### 🟡 Intermediate
To build a system like GPT-4V or Gemini, we cannot build separate disconnected networks. We need a way to mathematically force text concepts (the word "Dog") and visual concepts (pixels showing a golden retriever) into the exact same mathematical space.

### 🔴 Advanced
**Multimodal Transformers** treat the Transformer not as a text engine, but as a universal routing engine. Self-Attention doesn't care if the input vector came from a word, a 16x16 pixel patch, or a 1-second audio waveform. As long as the data can be projected into a $d_{model}$-dimensional vector, the Transformer can find the relationships between them.

---

# 2. The Universal API: Embeddings

The key to Multimodality is realizing that the `[batch_size, seq_len, d_model]` tensor shape is the universal API of AI.

*   **Text**: Chop a sentence into 10 subwords. Project them into 10 vectors of size 512. Sequence Length = 10.
*   **Image**: Chop a 256x256 image into a grid of 16x16 patches (like a puzzle). There are 256 patches. Flatten each patch into a vector, and project it to size 512. Sequence Length = 256.
*   **Audio**: Take a waveform, compute a mel-spectrogram, and slice it into 50 time-bins. Project each bin to size 512. Sequence Length = 50.

If we want an AI to caption an image, we simply concatenate the Image Sequence and the Text Sequence into one massive sequence:
`[Image Patch 1, Image Patch 2, ..., <START_TEXT>, What, is, this, ?]`

The Self-Attention mechanism will dynamically calculate dot products between the word "What" and the Image Patches, allowing the text to explicitly *attend* to specific regions of the image.

---

# 3. Contrastive Language-Image Pre-training (CLIP)

Before we can concatenate text and images, we must ensure they speak the same mathematical language.

OpenAI solved this in 2021 with **CLIP**.
They took 400 million pairs of (Image, Text Caption) from the internet.

1.  Pass the Image through an Image Encoder (a CNN or ViT). It outputs an Image Vector.
2.  Pass the Text through a Text Encoder (BERT-style). It outputs a Text Vector.
3.  Calculate the Cosine Similarity between the two vectors.
4.  **The Loss Function (Contrastive Loss)**: Maximize the similarity between the Image and its *correct* caption, and minimize the similarity between the Image and all other *incorrect* captions in the batch.

The result? The word "Dog" and a picture of a Dog map to the exact same coordinates in 512-dimensional space. CLIP became the mathematical bridge between Vision and Language.

---

# 4. Vision-Language Models (VLMs)

Once you have a bridge like CLIP, you can build a massive Vision-Language Model (like LLaVA or Flamingo).

**Architecture:**
1.  **Vision Encoder**: A pre-trained CLIP model processes the user's uploaded image into visual tokens.
2.  **Projection Layer**: A small linear layer that maps the CLIP visual tokens directly into the input dimension of a large LLM (like Llama-2 or GPT-3).
3.  **The LLM (The Brain)**: The LLM receives the visual tokens just like they were text tokens, followed by the user's prompt tokens. It generates the text response autoregressively.

Because the LLM was pre-trained on trillions of text tokens, it already possesses incredible reasoning. By simply "showing" it the image tokens through the projection layer, the LLM instantly gains the ability to visually reason.

---

# 5. Any-to-Any Modality (Gemini)

While models like LLaVA bolt a Vision Encoder onto a Text LLM, the absolute state-of-the-art is **Natively Multimodal** models like Google's **Gemini**.

Instead of training a Text model and stitching on an Image model later, Gemini was trained from Day 1 on a mixture of interleaved Text, Image, Audio, and Video data. 

*   A video is treated as a sequence of images (frames).
*   Audio waveforms are interleaved directly with the text tokens.

This allows the Self-Attention mechanism to learn natively how a spoken word, a visual frame, and a written transcript interact, unlocking capabilities like real-time video reasoning and highly nuanced audio tone detection (like hearing sarcasm, which a text transcript misses).

---

# 6. Key Takeaways

*   The Transformer's `[seq_len, d_model]` architecture makes it a **Universal Routing Engine** for any modality.
*   **CLIP** bridges Vision and Language by mapping images and text into the exact same mathematical embedding space using Contrastive Loss.
*   **VLMs** (Vision-Language Models) work by projecting visual tokens directly into the context window of a pre-trained LLM.
*   **Native Multimodality** (like Gemini) trains the Transformer on all data types simultaneously from scratch, yielding superior cross-modal reasoning.

---

# 7. Next Topic

We have discussed how images can be chopped up and fed into a Transformer as part of a multimodal pipeline.

Next, we will look deeply at how this "image chopping" actually works, and how the **Vision Transformer (ViT)** completely dethroned the CNN as the king of Computer Vision.

[← RAG](13-Retrieval-Augmented-Generation-RAG.md) | [Back to Index](README.md) | [Next Topic: Vision Transformers (ViT) →](15-Vision-Transformers-ViT.md)
