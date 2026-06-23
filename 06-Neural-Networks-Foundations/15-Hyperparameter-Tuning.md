# 🧠 15 - Hyperparameter Tuning

---

## 📋 Table of Contents
1. [Parameters vs Hyperparameters](#parameters-vs-hyperparameters)
2. [The Most Important Dials](#the-most-important-dials)
3. [Tuning Strategies](#tuning-strategies)
4. [What's Next](#whats-next)

---

## 🎛️ Parameters vs Hyperparameters

In Deep Learning, there are two distinct types of variables:

1. **Parameters:** Variables the network learns automatically through Backpropagation (e.g., Weights and Biases).
2. **Hyperparameters:** Variables *you* must set manually before training begins. They dictate how the network learns and what its architecture looks like.

Hyperparameter tuning is part science, part dark art. Finding the optimal settings requires intuition, experimentation, and a lot of computing power.

*(Note: Use the [Hyperparameter Experimentation Lab](./notebooks/06-Hyperparameter-Experimentation-Lab.ipynb) to interactively test how changing these dials affects a model's final accuracy).*

---

## 🎚️ The Most Important Dials

If you are trying to improve your model's accuracy, here are the hyperparameters you should tune, ordered from most important to least important.

### 1. Learning Rate ($\alpha$)
**Importance:** Critical (Tune this first, always).
**What it does:** Determines the size of the steps the optimizer takes down the loss landscape.
**How to tune:** 
- Use a logarithmic scale. Try `0.1`, `0.01`, `0.001` (1e-3), `0.0001`. 
- If using Adam, start at `1e-3`. If using SGD, start at `0.1` or `0.01`.
- If the loss explodes to `NaN`, it's too high. If the loss barely moves over 10 epochs, it's too low.

### 2. Batch Size
**Importance:** High.
**What it does:** The number of samples the network looks at before performing a weight update.
**How to tune:**
- Always use powers of 2 (due to how GPUs manage memory): `16`, `32`, `64`, `128`, `256`.
- **Large Batch (e.g., 256):** Trains faster, uses more GPU memory, gradient estimates are very accurate.
- **Small Batch (e.g., 32):** Adds "noise" to the training path, which actually acts as a form of Regularization and helps the model generalize better to new data. `32` is often considered the golden default.

### 3. Number of Hidden Units (Width)
**Importance:** Medium.
**What it does:** The number of neurons in a specific hidden layer.
**How to tune:**
- If the network is underfitting (training accuracy is bad), make it wider. It needs more capacity to learn.
- Keep decreasing the width in deeper layers. (e.g., Layer 1: 512, Layer 2: 256, Layer 3: 128). This creates a funnel that compresses the data into dense representations.

### 4. Network Depth (Layers)
**Importance:** Medium.
**What it does:** The total number of hidden layers in the network.
**How to tune:**
- Start shallow (1 or 2 hidden layers). If it underfits, add another. 
- Deep networks (50+ layers) are almost exclusively used for Computer Vision (CNNs) and NLP (Transformers). For standard tabular data, 2 to 4 layers is usually the maximum before performance degrades.

### 5. Epochs
**Importance:** Low (Use Early Stopping instead).
**What it does:** The number of times the network loops through the *entire* training dataset.
**How to tune:**
- Don't guess. Set epochs to an absurdly high number (e.g., 1000) and use an **Early Stopping** script to automatically kill the training when the Validation Loss stops improving.

---

## 🔬 Tuning Strategies

You have 5 different dials. How do you find the perfect combination without spending 10 years manually testing them all?

### 1. Babysitting (Manual Trial & Error)
Used when you have very limited computing power. You train a model for a few epochs, watch the loss, manually tweak the learning rate, and try again. It builds great intuition but is highly inefficient.

### 2. Grid Search
Used when you have lots of compute. You define a list of options for every hyperparameter (e.g., LR: [0.1, 0.01], Batch: [32, 64]). The computer tests every single possible combination. 
*Warning: If you have 5 hyperparameters with 5 options each, that's $5^5 = 3,125$ models to train. It suffers from the curse of dimensionality.*

### 3. Random Search
The industry standard for basic tuning. Instead of testing every combination in a rigid grid, you tell the computer to pick random combinations within specific ranges. 
*Why it's better than Grid Search:* It explores a much wider variety of the crucial hyperparameters (like Learning Rate) rather than repeatedly testing identical learning rates with slightly different, less important parameters.

### 4. Bayesian Optimization
The advanced, modern approach. It uses machine learning to tune machine learning. The algorithm looks at the results of the past few trials and mathematically predicts which combination of hyperparameters has the highest probability of improving the score next. (Libraries like `Optuna` or `Ray Tune` do this automatically).

---

## 🚀 What's Next

### Key Takeaways
- Hyperparameters dictate the architecture and learning mechanics of the network.
- **Learning Rate** is the single most important parameter to tune.
- Always use powers of 2 for **Batch Size** (32 is a great default).
- Avoid manual tuning when possible; use tools like Random Search or Optuna.

### Common Mistakes
- **Tuning everything at once:** If you change the learning rate, the batch size, and the architecture all in one try, and the model improves, you have no idea *which* change actually caused the improvement. 

### Practical Recommendations
- My standard workflow: Build a simple 2-layer network. Set Batch Size to 32. Use the Adam optimizer. Tune *only* the Learning Rate until the loss drops nicely. Then add Dropout or adjust the network width to fine-tune.

### Next Topic
You now know all the theory. You know the architecture, the math, the optimizers, the regularizers, and how to tune the dials. 

It is time to put everything together. We are going to build a complete Neural Network from scratch using nothing but Python and NumPy. 

[← Previous Topic](./14-Batch-Normalization.md) | [Next Topic: Building a NN from Scratch →](./16-Building-A-Neural-Network-From-Scratch.md)
