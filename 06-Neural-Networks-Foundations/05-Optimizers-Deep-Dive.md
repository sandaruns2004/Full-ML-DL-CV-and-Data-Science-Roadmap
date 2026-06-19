# 🏎️ Optimizers Deep Dive

> **Prerequisites**: Backpropagation, Loss Functions | **Difficulty**: ⭐⭐⭐⭐⭐ Advanced

---

## 📋 Table of Contents
1. [The Goal of Optimization](#1-the-goal-of-optimization)
2. [First-Order Methods (Gradient Descent)](#2-first-order-methods-gradient-descent)
3. [Momentum & Nesterov](#3-momentum--nesterov)
4. [Adaptive Learning Rates (AdaGrad & RMSprop)](#4-adaptive-learning-rates-adagrad--rmsprop)
5. [The King: Adam & AdamW](#5-the-king-adam--adamw)
6. [Large Batch Training: LAMB & LARS](#6-large-batch-training-lamb--lars)
7. [Learning Rate Schedulers](#7-learning-rate-schedulers)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. The Goal of Optimization

Backpropagation calculates the gradient $\nabla L(\theta)$ for every parameter in our network. Optimization is the process of using those gradients to update the parameters $\theta$ to minimize the loss $L$.

A Neural Network loss landscape is highly non-convex, filled with:
- **Local Minima**: Valleys that are not the lowest point.
- **Saddle Points**: Areas where the gradient is zero, but it's a minimum in one direction and a maximum in another.
- **Ravines**: Narrow valleys where the surface curves steeply in one dimension but is flat in another.

A good optimizer navigates these obstacles quickly without getting stuck.

---

## 2. First-Order Methods (Gradient Descent)

### 2.1 Standard Gradient Descent (Batch GD)
Calculates the gradient using the *entire* dataset.
$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$
*(where $\eta$ is the learning rate).*

**Problem**: Requires passing the entire dataset through memory for a single update. Impossibly slow for deep learning.

### 2.2 Stochastic Gradient Descent (SGD)
Calculates the gradient using exactly *one* example.
**Problem**: Extremely noisy. Bounces around the minimum erratically. Cannot utilize GPU vectorization effectively.

### 2.3 Mini-Batch SGD
The standard approach. Calculates the gradient on a small batch $B$ (e.g., 32, 64, 256).
- Utilizes fast matrix multiplication on GPUs.
- The inherent noise of the mini-batch actually helps the model escape sharp local minima, acting as implicit regularization.

---

## 3. Momentum & Nesterov

### 3.1 SGD with Momentum
Imagine a ball rolling down a hill. It builds up momentum. 
In steep ravines, standard SGD oscillates wildly side-to-side without moving forward. Momentum dampens the oscillations (because they cancel out) and amplifies the consistent forward direction.

We maintain a velocity vector $v$:
$$v_{t} = \beta v_{t-1} + (1 - \beta) \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_{t}$$
*(Typically $\beta = 0.9$)*.

### 3.2 Nesterov Accelerated Gradient (NAG)
Standard momentum computes the gradient at the *current* position, then adds momentum. 
Nesterov looks ahead: it first jumps by the momentum, computes the gradient at that *future* position, and uses that to correct the course. This prevents the ball from rolling too fast past the minimum.

$$v_{t} = \beta v_{t-1} + (1 - \beta) \nabla L(\theta_t - \beta v_{t-1})$$
$$\theta_{t+1} = \theta_t - \eta v_{t}$$

---

## 4. Adaptive Learning Rates (AdaGrad & RMSprop)

SGD uses the same global learning rate $\eta$ for all parameters. But some features are rare (need large updates when seen) and some are frequent (need small updates).

### 4.1 AdaGrad
Scales the learning rate inversely proportional to the square root of the sum of all historical squared gradients.
$$G_t = G_{t-1} + (\nabla L(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L(\theta_t)$$

**Problem**: Because $G_t$ only grows, the learning rate eventually shrinks to zero, and the model stops learning prematurely.

### 4.2 RMSprop (Root Mean Square Propagation)
Invented by Geoffrey Hinton. Fixes AdaGrad by changing the strict sum into an exponentially decaying average.
$$E[\nabla^2]_t = \beta E[\nabla^2]_{t-1} + (1 - \beta) (\nabla L(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[\nabla^2]_t + \epsilon}} \nabla L(\theta_t)$$
*(Typically $\beta = 0.999$)*.

---

## 5. The King: Adam & AdamW

### 5.1 Adam (Adaptive Moment Estimation)
Introduced in 2014, Adam combines the best of Momentum (First Moment) and RMSprop (Second Moment).

1. Update biased first moment (Mean):
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(\theta_t)$$
2. Update biased second moment (Variance):
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(\theta_t))^2$$
3. Compute **Bias-Corrected** moments (because $m_0, v_0$ are initialized to 0, they are biased toward zero early in training):
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
4. Update parameters:
   $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
*(Defaults: $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$)*.

### 5.2 AdamW (Adam with decoupled Weight Decay)
In standard SGD, L2 Regularization (adding $\frac{\lambda}{2}\|\theta\|^2$ to the loss) is mathematically identical to Weight Decay ($\theta_{t+1} = (1 - \lambda)\theta_t - \dots$).
In Adam, because of the adaptive denominators, L2 Regularization and Weight Decay are **NOT** the same. L2 regularization gets scaled by $\hat{v}_t$, meaning weights with large gradients are barely regularized!

AdamW fixes this by decoupling the weight decay step from the gradient update:
$$\theta_{t+1} = \theta_t - \eta \lambda \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$
**AdamW is currently the state-of-the-art default for training LLMs and Vision Transformers.**

---

## 6. Large Batch Training: LAMB & LARS

When training giant models (like BERT or GPT) across thousands of GPUs, researchers want to use massive batch sizes (e.g., 32,000 or 65,000) so the GPUs don't sit idle. 

However, standard Adam diverges at these extreme batch sizes because the "trust ratio" between the layer weights and the layer gradients breaks down.

### LARS (Layer-wise Adaptive Rate Scaling)
Used to train ResNet-50 in minutes using batch sizes of 32K. It calculates a separate learning rate for *each layer* based on the ratio of the weight norm to the gradient norm:
$$\eta^{(l)} = \eta \times \frac{\|\theta^{(l)}\|}{\|\nabla L(\theta^{(l)})\| + \lambda \|\theta^{(l)}\|}$$

### LAMB (Layer-wise Adaptive Moments optimizer for Batch training)
Combines AdamW with the layer-wise trust ratio of LARS. This is the optimizer used by Google to train BERT in 76 minutes (down from 3 days) using a batch size of 32,768.

---

## 7. Learning Rate Schedulers

No matter what optimizer you use, keeping the learning rate $\eta$ static is a bad idea.

1. **Step Decay**: Multiply $\eta$ by 0.1 every $N$ epochs.
2. **Cosine Annealing**: Smoothly decay the learning rate following a cosine curve from $\eta_{max}$ to $\eta_{min}$. Allows the model to aggressively explore early, and carefully fine-tune late.
3. **Linear Warmup**: Adam often requires a "warmup" period where the learning rate scales linearly from $0$ to $\eta_{max}$ over the first few thousand steps. This prevents the model from diverging early while the momentum moving averages ($m_t, v_t$) are still stabilizing.

*Modern LLM standard: AdamW + Linear Warmup + Cosine Decay.*

---

## 8. Project Ideas & What's Next

### Project Ideas
- 🟢 **Optimizer Race**: Create a highly non-convex 2D mathematical function (like the Beale function or Ackley function). Implement basic SGD, Momentum, and Adam in pure Python. Plot their trajectories on a contour map to see who finds the global minimum first.
- 🟡 **PyTorch Scheduler Analysis**: Train a ResNet on CIFAR-10. Train it once with a static learning rate of $1e-3$. Train it again using `torch.optim.lr_scheduler.CosineAnnealingLR`. Compare the final accuracy and the smoothness of the validation curves.

### What's Next
| Next | Why |
|------|-----|
| [Regularization Techniques](./06-Regularization-Techniques.md) | Now that we can optimize and fit our data perfectly, we must ensure we aren't *overfitting*. |
| [Weight Initialization](./07-Weight-Initialization.md) | Optimization assumes our network starts in a sensible state. If we initialize badly, even Adam can't save us. |

---

[← Loss Functions Deep Dive](./04-Loss-Functions-Deep-Dive.md) | [Back to Index](../README.md) | [Next: Regularization Techniques →](./06-Regularization-Techniques.md)
