# 🛡️ Regularization Techniques

> **Prerequisites**: Backpropagation, Optimizers | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Overfitting Crisis in Deep Learning](#1-the-overfitting-crisis-in-deep-learning)
2. [Weight Decay (L1 & L2 Regularization)](#2-weight-decay-l1--l2-regularization)
3. [Dropout & Its Variants](#3-dropout--its-variants)
4. [Normalization Layers (Batch, Layer, Group)](#4-normalization-layers-batch-layer-group)
5. [Early Stopping & Label Smoothing](#5-early-stopping--label-smoothing)
6. [Data Augmentation as Regularization](#6-data-augmentation-as-regularization)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Overfitting Crisis in Deep Learning

Deep Neural Networks are universal function approximators with millions (or billions) of parameters. They possess enough capacity to completely memorize the training dataset, including the noise.

**The Bias-Variance Tradeoff in DL:**
In classical Machine Learning, we try to find the "sweet spot" of model complexity to balance bias and variance. In Deep Learning, we operate in the **over-parameterized regime**. We deliberately build models with massive capacity (low bias, high variance) and then heavily restrict that capacity using **Regularization**.

> **Regularization** is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error.

---

## 2. Weight Decay (L1 & L2 Regularization)

We add a penalty term to the Loss function to force the network to keep its weights small. Small weights lead to a simpler, smoother function that is less likely to overfit.

### 2.1 L2 Regularization (Ridge)
$$J = L(y, \hat{y}) + \frac{\lambda}{2m} \sum_{l=1}^{L} ||\mathbf{W}^{[l]}||_F^2$$
*(where $||\cdot||_F$ is the Frobenius norm, the sum of squared matrix elements).*

During backpropagation, the gradient becomes:
$$d\mathbf{W}^{[l]} = \nabla L(\mathbf{W}^{[l]}) + \frac{\lambda}{m} \mathbf{W}^{[l]}$$
Thus, during the gradient descent update step:
$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \alpha \left( \nabla L(\mathbf{W}^{[l]}) + \frac{\lambda}{m} \mathbf{W}^{[l]} \right)$$
$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} \left(1 - \alpha \frac{\lambda}{m} \right) - \alpha \nabla L(\mathbf{W}^{[l]})$$
Notice that the weight is actively decayed by a factor of $(1 - \alpha \frac{\lambda}{m})$ before taking the gradient step. This is why L2 regularization is synonymous with **Weight Decay** in standard SGD.

### 2.2 L1 Regularization (Lasso)
$$J = L(y, \hat{y}) + \frac{\lambda}{m} \sum_{l=1}^{L} |\mathbf{W}^{[l]}|$$
L1 regularization drives weights exactly to $0$, resulting in a sparse network. It acts as an implicit feature selector. However, because it is non-differentiable at 0, and because GPUs struggle with sparse matrix operations, it is used less frequently in Deep Learning than L2.

---

## 3. Dropout & Its Variants

Invented by Srivastava & Hinton (2014), **Dropout** is arguably the most powerful structural regularization technique for Neural Networks.

### 3.1 Standard Dropout
**The Mechanism**: During *training*, at each forward pass, every neuron in a layer has a probability $p$ of being temporarily dropped (set to 0). 
- If $p=0.5$, half the neurons are dead for that specific batch.
- On the next batch, a completely different, random set of neurons is dropped.

**The Math (Inverted Dropout)**:
To ensure the expected value of the activations remains the same during testing (when dropout is turned off), we scale the remaining active neurons during training by $\frac{1}{1-p}$.
$$a_{train} = a \cdot \frac{D}{1-p} \quad (\text{where } D \sim \text{Bernoulli}(1-p))$$
$$a_{test} = a$$

**Why it works (Ensemble Theory)**:
If a network has $N$ neurons, Dropout effectively trains $2^N$ different sub-networks that all share weights. At test time, turning Dropout off effectively averages the predictions of all $2^N$ sub-networks. It prevents complex co-adaptations between specific neurons.

### 3.2 Spatial Dropout (For CNNs)
In CNNs, adjacent pixels in feature maps are highly correlated. Standard dropout randomly dropping single pixels does nothing—the network just infers the missing pixel from its neighbors. **Spatial Dropout** drops entire 2D feature maps (channels) instead of individual pixels.

### 3.3 DropConnect
Instead of dropping activations (neurons), DropConnect randomly drops the *weights* connecting them.

---

## 4. Normalization Layers (Batch, Layer, Group)

While primarily an optimization technique to speed up training, Normalization layers act as strong regularizers.

### 4.1 Batch Normalization
Normalizes the activations $Z$ of a layer *across the batch dimension*.
$$ \hat{z} = \frac{z - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}} $$
$$ \tilde{z} = \gamma \hat{z} + \beta $$

**Regularization Effect**: Because the mean and variance are calculated over a random mini-batch, the exact output for a single image depends on the *other* images in that batch. This injects noise into the forward pass, acting similarly to Dropout. In fact, if you use Batch Norm, you usually don't need Dropout!

### 4.2 Layer Normalization
Batch Norm fails if the batch size is too small (e.g., $B=2$), or in RNNs where sequence lengths vary.
**Layer Normalization** computes the mean and variance *across the features* for a single sample, completely independent of the batch dimension. It is the standard normalization used in **Transformers**.

### 4.3 Group Normalization
Divides the channels into groups and computes mean and variance within each group. Used in advanced CNNs (like ResNeXt or Object Detectors) where batch sizes are forced to be very small due to high-resolution images.

---

## 5. Early Stopping & Label Smoothing

### 5.1 Early Stopping
The simplest and most effective regularization technique. We monitor the validation loss during training. When the validation loss stops improving and starts going up (indicating overfitting), we simply stop training and restore the model weights from the best epoch.

### 5.2 Label Smoothing
In standard Categorical Cross-Entropy, the target vector is a one-hot vector: $[0, 1, 0, 0]$.
This forces the network to be *infinitely confident* (it wants to output $p=1.0$ for the true class, which means the logit $z$ must approach $\infty$). This causes over-confidence and overfitting.

**Label Smoothing** redistributes a small amount of probability $\epsilon$ to all other classes:
$$y_{smoothed} = \begin{cases} 1 - \epsilon & \text{for the correct class} \\ \frac{\epsilon}{K-1} & \text{for the incorrect classes} \end{cases}$$
*(e.g., $[0.05, 0.85, 0.05, 0.05]$).* This forces the model to be slightly less certain, improving generalization.

---

## 6. Data Augmentation as Regularization

The ultimate way to prevent overfitting is to get more data. If you can't get more data, fake it.

By applying random transformations (rotations, flips, color jitter, cropping) to your training data every epoch, the network never sees the exact same image twice. This forces the network to learn invariant features (e.g., "a cat is a cat regardless of rotation") rather than memorizing exact pixel values.

*(Advanced augmentations like MixUp and CutMix will be covered deeply in the CNN section).*

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Overfitting a Tiny Dataset**: Take a dataset of just 50 images. Build a wide MLP and train it without regularization until it reaches 100% training accuracy and 10% validation accuracy. Then, add Dropout (p=0.5) and L2 Weight Decay. Observe how the training accuracy drops, but the validation accuracy rises.
- 🟡 **Batch Norm vs Layer Norm**: Implement both in PyTorch from scratch using raw tensors and custom `nn.Module` classes. Verify your output matches PyTorch's native `nn.BatchNorm1d` and `nn.LayerNorm`.

### What's Next
| Next | Why |
|------|-----|
| [Weight Initialization](./07-Weight-Initialization.md) | Regularization restricts the weights during training. But how do we set the weights before training even begins? |

---

[← Optimizers Deep Dive](./05-Optimizers-Deep-Dive.md) | [Back to Index](../README.md) | [Next: Weight Initialization →](./07-Weight-Initialization.md)
