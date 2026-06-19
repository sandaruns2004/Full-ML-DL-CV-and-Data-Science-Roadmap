# 🪞 Self-Supervised Learning (SSL) & Contrastive Learning

> **Prerequisites**: Convolutional Neural Networks, Deep Learning | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Data Labeling Bottleneck](#1-the-data-labeling-bottleneck)
2. [What is Self-Supervised Learning?](#2-what-is-self-supervised-learning)
3. [Contrastive Learning Framework](#3-contrastive-learning-framework)
4. [SimCLR: A Simple Framework for Contrastive Learning](#4-simclr-a-simple-framework-for-contrastive-learning)
5. [BYOL: Bootstrap Your Own Latent](#5-byol-bootstrap-your-own-latent)
6. [Masked Image Modeling (MAE & DINO)](#6-masked-image-modeling-mae--dino)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Data Labeling Bottleneck

The massive success of ResNet, VGG, and early CNNs relied on **ImageNet**—a dataset of 1.2 million images painstakingly labeled by human workers across 1,000 categories.

But what if you are a medical startup with 10 million MRI scans, and hiring expert doctors to label all of them would cost $50 million? 
Supervised learning hits a hard bottleneck: **Data is infinite, but labels are expensive.**

For years, researchers tried Unsupervised Learning (like Autoencoders) to extract features, but the resulting representations were poor for downstream tasks.

---

## 2. What is Self-Supervised Learning?

**Self-Supervised Learning (SSL)** is the holy grail of modern AI. It allows models to learn massive amounts of knowledge from *unlabeled data* by inventing a pretext task where the data itself provides the supervision.

We already saw SSL in NLP: **BERT's Masked Language Modeling** (hiding a word and asking the network to guess it) and **GPT's Next Word Prediction**. 
The text itself acts as the label! This is why NLP models were able to scale to trillions of words.

Applying SSL to Computer Vision took much longer, but eventually revolutionized the field through two main paradigms: **Contrastive Learning** and **Masked Image Modeling**.

---

## 3. Contrastive Learning Framework

The core philosophy of Contrastive Learning is simple:
> *A network should output similar mathematical vectors for images that show the same thing, and vastly different vectors for images that show different things.*

But without labels, how do we know which images show the same thing?
**Data Augmentation!**

If we take a picture of a dog, and we aggressively crop it, flip it, and apply a massive color jitter (turning it black and white), *it is still a picture of a dog*. 

**The Contrastive Task**:
1. Take an image $x$. Create two different augmented versions of it: $x_A$ and $x_B$ (these are called a **Positive Pair**).
2. Take a completely different image $y$ from the dataset. (This forms a **Negative Pair** with $x$).
3. Pass all images through a CNN encoder to get vector embeddings.
4. **The Loss Function**: Maximize the Cosine Similarity between the vectors of the Positive Pair ($x_A$ and $x_B$), while minimizing the similarity between the Negative Pairs ($x_A$ and $y$).

By doing this across millions of unlabeled images, the CNN is forced to ignore background noise, color shifts, and cropping, and strictly learn to identify the core semantic objects in the images.

---

## 4. SimCLR: A Simple Framework for Contrastive Learning

In 2020, Google published **SimCLR**, a massive breakthrough in contrastive learning.

SimCLR proved that to make contrastive learning work, you need two things:
1. **Extremely aggressive data augmentations**: A combination of random cropping, massive color distortion, and Gaussian blur is strictly required.
2. **Massive Batch Sizes**: SimCLR used a batch size of 8,192. Why? Because in a batch of 8,192, there is 1 Positive Pair and 8,191 Negative Pairs. Having a massive number of negative examples forces the network to learn incredibly discriminative features.

### The InfoNCE Loss (NT-Xent)
SimCLR uses a variation of Cross-Entropy called InfoNCE (Noise-Contrastive Estimation) loss.
For a positive pair $(i, j)$, the loss is:

$$ \ell_{i,j} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau)} $$

Where $\text{sim}()$ is cosine similarity, and $\tau$ is a temperature scaling parameter. The numerator pulls the positive pair together. The denominator pushes all $2N-1$ other images in the batch (the negatives) away.

---

## 5. BYOL: Bootstrap Your Own Latent

SimCLR was amazing, but requiring a batch size of 8,192 meant you needed Google's TPU pods to train it. Standard researchers with a single GPU couldn't use it.

Shortly after, DeepMind released **BYOL (Bootstrap Your Own Latent)**.
BYOL achieved state-of-the-art results **without using Negative Pairs at all!**

If you only use Positive Pairs, the network should suffer from **Representation Collapse** (it just learns to output a vector of all zeros for every single image, which perfectly maximizes similarity and drops the loss to 0).

**How BYOL prevents collapse:**
It uses two neural networks: an **Online** network and a **Target** network.
1. The Online network learns normally via gradient descent.
2. The Target network does *not* use gradient descent. Instead, its weights are a slow Exponential Moving Average (EMA) of the Online network's weights.
3. The Online network takes augmented image $x_A$ and tries to predict the Target network's output for augmented image $x_B$.

This asymmetric architecture creates a self-reinforcing bootstrap loop that magically prevents collapse and trains perfectly on small batch sizes!

---

## 6. Masked Image Modeling (MAE & DINO)

When Vision Transformers (ViTs) became popular, researchers realized they could replicate BERT's exact success in Computer Vision.

**MAE (Masked Autoencoders)** by Facebook AI:
1. Take an image and chop it into patches.
2. Randomly mask out **75%** of the patches! (Feed the network an image that is mostly missing).
3. Pass the remaining 25% through a ViT Encoder.
4. Use a small ViT Decoder to reconstruct the missing 75% of the original pixels.
5. The loss is simply the Mean Squared Error of the reconstructed pixels.

To successfully reconstruct 75% of a missing dog, the model must develop a deep, holistic understanding of what dogs look like, purely through self-supervision.

Once pre-trained, you throw away the Decoder, keep the Encoder, and fine-tune it on a small dataset (e.g., 100 labeled images) to achieve incredible classification accuracy.

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Lightly Library**: The `lightly` Python library provides out-of-the-box implementations of SimCLR, BYOL, and others. Use it to pre-train a ResNet-18 on the unlabeled `CIFAR-10` dataset using SimCLR. Then, freeze the ResNet, attach a Linear layer, and train it on just 1% of the CIFAR-10 labels. Watch it achieve high accuracy with almost no labeled data!
- 🟡 **Custom Augmentation Pipeline**: Write a custom PyTorch `Dataset` class that implements the SimCLR augmentation strategy using `torchvision.transforms` (RandomResizedCrop, RandomApply(ColorJitter), RandomGrayscale, RandomApply(GaussianBlur)). Verify that your positive pairs look radically different to the human eye, but contain the same semantic object.

### What's Next
| Next | Why |
|------|-----|
| [Meta-Learning](./04-Meta-Learning.md) | Self-Supervised Learning allows us to train without labels. **Meta-Learning** allows us to learn how to learn, so a model can master a new task using only 1 or 2 examples! |

---

[← Graph Neural Networks](./02-Graph-Neural-Networks.md) | [Back to Index](../README.md) | [Next: Meta Learning →](./04-Meta-Learning.md)
