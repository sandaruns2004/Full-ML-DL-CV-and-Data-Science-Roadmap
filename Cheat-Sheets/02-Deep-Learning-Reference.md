# 📄 Cheat Sheet: Deep Learning Reference

> A quick-reference guide for Activations, Loss Functions, Optimizers, and Architectures.

---

## 1. Activation Functions

| Function | Equation | Range | Where to Use | Pros / Cons |
|----------|----------|-------|--------------|-------------|
| **Sigmoid** | $\frac{1}{1 + e^{-x}}$ | $(0, 1)$ | Output layer of Binary Classification. | Causes vanishing gradients in deep networks. Not zero-centered. |
| **Softmax** | $\frac{e^{x_i}}{\sum e^{x_j}}$ | $(0, 1)$ | Output layer of Multi-class Classification. | Probabilities sum to 1. Exaggerates the largest value. |
| **Tanh** | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $(-1, 1)$ | Hidden layers (rarely), RNN hidden states. | Zero-centered (better than Sigmoid), but still suffers from vanishing gradients. |
| **ReLU** | $\max(0, x)$ | $[0, \infty)$ | Default for CNN/MLP Hidden Layers. | Extremely fast. Solves vanishing gradients for $x>0$. Can cause "Dead Neurons" if $x \le 0$. |
| **Leaky ReLU** | $\max(\alpha x, x)$ | $(-\infty, \infty)$ | Deep CNNs, GAN Discriminators. | Fixes the "Dead Neuron" problem by allowing a tiny gradient ($\alpha \approx 0.01$) when $x < 0$. |
| **GELU** | $x \Phi(x)$ | $(-\infty, \infty)$ | Default for Transformers (BERT, GPT). | Smoother than ReLU. Empirically performs better in NLP and modern Vision Transformers. |

---

## 2. Loss Functions

| Task | PyTorch Loss | Math Concept | When to Use |
|------|--------------|--------------|-------------|
| **Regression** | `nn.MSELoss()` | Mean Squared Error | Standard regression. Heavily penalizes large outliers. |
| **Regression** | `nn.L1Loss()` | Mean Absolute Error | Robust regression. Use if your data has extreme, unremovable outliers. |
| **Binary Classification** | `nn.BCELoss()` | Binary Cross Entropy | Predicting Yes/No (1 output node). Requires a Sigmoid activation right before it. |
| **Binary Classification** | `nn.BCEWithLogitsLoss()` | BCE + Sigmoid | Same as above, but mathematically more stable. Network output should NOT have a sigmoid. |
| **Multi-class Classification** | `nn.CrossEntropyLoss()` | LogSoftmax + NLLLoss | Predicting 1 out of K mutually exclusive classes. Network output should NOT have a softmax. |
| **Multi-label Classification** | `nn.BCEWithLogitsLoss()` | Multi-Binary Cross Entropy | Predicting multiple independent tags (e.g., Image contains Dog AND Car). |

---

## 3. Optimizers

| Optimizer | Mechanics | When to Use |
|-----------|-----------|-------------|
| **SGD** | $w = w - \alpha \nabla L$ | The baseline. Rarely used without Momentum. |
| **SGD + Momentum** | Accumulates a moving average of past gradients to push through saddle points. | Often used in Computer Vision (ResNets). Can generalize slightly better than Adam if tuned perfectly. |
| **Adam** | Adaptive Moment Estimation. Maintains separate learning rates for every single weight based on past gradients and squared gradients. | **The Default.** Use this 95% of the time. Extremely fast convergence, less sensitive to learning rate choice. |
| **AdamW** | Adam + decoupled Weight Decay (L2 Regularization). | Use instead of Adam for Transformers (BERT, GPT, ViT). Better regularization. |

---

## 4. Architecture Selection Guide

| Data Type | Standard Architecture | State-of-the-Art |
|-----------|-----------------------|------------------|
| **Tabular Data** (CSV, SQL) | Random Forest / XGBoost | XGBoost / LightGBM |
| **Images** (Classification) | ResNet-50 | Vision Transformer (ViT), ConvNeXt |
| **Images** (Object Detection) | Faster R-CNN, YOLOv5 | YOLOv8, DETR (Detection Transformer) |
| **Images** (Segmentation) | U-Net | Mask R-CNN, SAM (Segment Anything) |
| **Sequential** (Time Series) | LSTM / GRU | Temporal Convolutional Networks (TCN), PatchTST |
| **Text** (Classification) | BERT / RoBERTa | LLama 3 / GPT-4o (Zero-shot) |
| **Text** (Generation) | GPT-2 | Llama 3 / Mistral / Claude 3 |

---

[← ML Algorithms Comparison](./01-ML-Algorithms-Comparison.md) | [Back to Index](../README.md) | [Next: Machine Learning Quiz →](../Quizzes/01-Machine-Learning-Quiz.md)
