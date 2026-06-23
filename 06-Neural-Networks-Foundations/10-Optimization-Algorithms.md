# 🧠 10 - Optimization Algorithms

---

## 📋 Table of Contents
1. [The Problem with Standard Gradient Descent](#the-problem-with-standard-gradient-descent)
2. [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
3. [SGD with Momentum](#sgd-with-momentum)
4. [RMSProp](#rmsprop)
5. [Adam (Adaptive Moment Estimation)](#adam-adaptive-moment-estimation)
6. [AdamW (Adam with Weight Decay)](#adamw-adam-with-weight-decay)
7. [What's Next](#whats-next)

---

## 🐌 The Problem with Standard Gradient Descent

Standard Gradient Descent (often called Batch Gradient Descent) computes the gradient using the *entire* dataset before taking a single step. 

If you have 1 million images, the network must perform forward propagation and backpropagation 1 million times just to update the weights slightly by $\alpha$. 
This is incredibly slow and computationally impossible for modern datasets. Furthermore, if the cost landscape has a small pothole (local minimum), the network will get permanently stuck in it.

To train deep learning models, we need optimizers that are faster, smarter, and capable of navigating treacherous mathematical landscapes.

*(Note: Run the [Optimization Algorithms Visualizer](./projects/05-Optimization-Algorithm-Comparison/) to watch these algorithms race down a 3D loss landscape!)*

---

## 1. Stochastic Gradient Descent (SGD)

**Why it was invented:** To solve the slowness of standard Gradient Descent.

Instead of looking at 1 million images to take one step, SGD looks at exactly **1 image**, calculates the gradient, and takes a step. It repeats this 1 million times. 

**Tradeoffs:**
Because it updates the weights based on a single image, the steps are extremely noisy. The path down the mountain looks like a drunk person staggering—it zig-zags wildly. While it eventually reaches the bottom, the zig-zagging is inefficient.

### Mini-Batch SGD (The Industry Standard)
In practice, nobody uses pure SGD (1 image). We use **Mini-Batch SGD**. We look at a small batch of images (e.g., 32 or 64) at a time. This averages out the noise, providing a much smoother path down the mountain while still being lightning fast.

---

## 2. SGD with Momentum

**Why it was invented:** To stop SGD from zig-zagging and getting stuck in potholes.

Imagine pushing a heavy bowling ball down a hill. As it rolls, it builds up **momentum**. If it hits a small pothole, its momentum carries it right over the hole. If the hill curves slightly left and right, the momentum keeps the ball rolling straight down the main slope.

**How it works mathematically:** 
Instead of only using the *current* gradient to take a step, SGD with Momentum calculates a moving average of *past* gradients. If past gradients were all pointing South, and the current gradient suddenly points East, Momentum says "No, keep going mostly South."

**Tradeoffs:**
- Much faster convergence than standard SGD.
- Very good at escaping local minima.
- Sometimes it builds up *too much* momentum and rolls right past the global minimum, requiring a smaller learning rate as training ends.

---

## 3. RMSProp

**Why it was invented:** To handle features that occur at different frequencies. (Invented by Geoffrey Hinton during a Coursera lecture!)

Imagine a mountain that is very steep on the North-South axis, but very flat on the East-West axis. Standard SGD will bounce violently North-South while making almost zero progress East-West. 

**How it works mathematically:**
RMSProp keeps track of the squared gradients for every single weight. 
- If a weight has been experiencing massive, violent gradients (North-South), RMSProp automatically **shrinks** its learning rate.
- If a weight has been experiencing tiny gradients (East-West), RMSProp automatically **boosts** its learning rate.

Every single weight in the network gets its own, custom, adaptable learning rate.

---

## 4. Adam (Adaptive Moment Estimation)

**Why it was invented:** To combine the best of both worlds. 

Adam is essentially **RMSProp + Momentum**. 

**How it works mathematically:**
1. It keeps track of the moving average of past gradients (Momentum) to keep the ball rolling smoothly.
2. It keeps track of the squared gradients (RMSProp) to give every weight its own custom learning rate.

**Tradeoffs:**
- **Strengths:** Adam is incredibly robust. It works flawlessly out-of-the-box for almost any deep learning problem with very little hyperparameter tuning. It converges extremely fast.
- **Weaknesses:** In some specific cases (like Image Classification), while Adam converges faster, a perfectly tuned SGD with Momentum will often achieve a slightly better final accuracy.

---

## 5. AdamW (Adam with Weight Decay)

**Why it was invented:** Adam had a mathematical flaw when it came to regularization (L2 Weight Decay). 

Standard Adam applies regularization in a way that interacts poorly with its adaptive learning rates, leading to models that don't generalize as well as SGD. AdamW decoupled the weight decay from the gradient updates, fixing the math.

**Typical Usage:** 
AdamW has largely replaced standard Adam. If you are training a modern Transformer (like GPT or BERT), you are almost certainly using AdamW.

---

## 🚀 What's Next

### Key Takeaways
- **SGD (Mini-Batch):** Takes steps based on a small chunk of data. Noisy but fast.
- **Momentum:** Remembers past steps to smooth out the zig-zagging.
- **Adam:** The undisputed king of optimizers for general tasks. Combines momentum with per-weight adaptive learning rates.

### Common Mistakes
- **Spending hours tuning learning rates for Adam:** Adam is designed to be highly forgiving. If Adam isn't learning with the default learning rate (usually `1e-3`), you likely have a bug in your architecture or data, not your optimizer. 

### Practical Recommendations
- **Default Choice:** Always start your project using **Adam** (or **AdamW**). 
- If you are trying to squeeze out the absolute final 1% of accuracy for a computer vision paper, switch to **SGD with Momentum** and spend time carefully tuning the learning rate schedule.

### Next Topic
We now have the engine (Backpropagation) and the steering wheel (Optimization Algorithms). But before we start the car, we have to make sure the starting position is correct. How do we set the initial random weights of the network?

[← Previous Topic](./09-Backpropagation.md) | [Next Topic: Weight Initialization →](./11-Weight-Initialization.md)
