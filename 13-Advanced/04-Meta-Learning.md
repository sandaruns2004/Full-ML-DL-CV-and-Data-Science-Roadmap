# 🧠 Meta-Learning (Learning to Learn)

> **Prerequisites**: Deep Learning Fundamentals, Few-Shot Learning | **Difficulty**: ⭐⭐⭐⭐⭐ Expert

---

## 📋 Table of Contents
1. [The Problem with Standard Deep Learning](#1-the-problem-with-standard-deep-learning)
2. [What is Meta-Learning?](#2-what-is-meta-learning)
3. [N-Way K-Shot Classification](#3-n-way-k-shot-classification)
4. [Metric-Based Meta-Learning (Prototypical Networks)](#4-metric-based-meta-learning-prototypical-networks)
5. [Optimization-Based Meta-Learning (MAML)](#5-optimization-based-meta-learning-maml)
6. [From-Scratch Implementation (MAML PyTorch)](#6-from-scratch-implementation-maml-pytorch)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Problem with Standard Deep Learning

Standard Deep Learning (like a ResNet trained on ImageNet) requires thousands of examples per class.
If you train a ResNet to classify 100 different breeds of dogs, and tomorrow you want it to classify a brand new breed it has never seen, you have to:
1. Collect 1,000 photos of the new breed.
2. Add a new neuron to the output layer.
3. Fine-tune the network (which might cause catastrophic forgetting of the previous 100 breeds).

Humans don't work like this. If you show a toddler exactly *one* picture of a platypus, they can instantly recognize a platypus in real life forever. This is **Few-Shot Learning**.

---

## 2. What is Meta-Learning?

**Meta-Learning** is the science of "Learning to Learn".
Instead of training a neural network on *data points* (images), we train a neural network on *entire tasks*.

The goal is to train a model that, when exposed to a brand new task, can learn that task perfectly using only 1 or 2 gradient updates and a handful of examples.

**The Meta-Training Loop:**
1. **Sample a Task**: Pick 5 random classes from your training dataset (e.g., Cats, Dogs, Cars, Planes, Boats).
2. **Support Set**: Take a tiny amount of data (e.g., 5 images per class) to train on.
3. **Query Set**: Take another small amount of data from those same 5 classes to test on.
4. Update the *Meta-Weights* based on how well the model learned the task from the Support Set and generalized to the Query Set.
5. Repeat for thousands of different random tasks!

---

## 3. N-Way K-Shot Classification

Meta-learning tasks are universally described using the "N-Way K-Shot" terminology.

- **N-Way**: How many classes are in this specific task?
- **K-Shot**: How many examples per class does the model get to learn from?

A **5-Way 1-Shot** task means: 
"Here are 5 totally random classes. Here is exactly 1 picture of each class. Now look at this new picture (the Query) and tell me which of the 5 classes it belongs to."

This is incredibly difficult, which is why standard cross-entropy loss fails, and we need specialized Meta-Learning architectures.

---

## 4. Metric-Based Meta-Learning (Prototypical Networks)

One way to solve Few-Shot learning is by learning a highly structured metric space (similar to Contrastive Learning).

**Prototypical Networks** (Snell et al., 2017) are beautifully simple and highly effective:
1. Pass the entire Support Set through a CNN to get embeddings.
2. For each class in the task, calculate its **Prototype**: the mathematical average (mean) of all the embeddings of the images in that class.
3. Pass a Query image through the CNN to get its embedding.
4. Use a distance metric (usually Euclidean distance) to see which class Prototype the Query embedding is closest to.
5. Apply a softmax over the negative distances to get probabilities, and calculate Cross-Entropy loss.

During Meta-Training, the CNN is forced to learn an embedding space where images of the same class tightly cluster together, regardless of what the class actually is!

---

## 5. Optimization-Based Meta-Learning (MAML)

What if we don't want to use distance metrics? What if we want to use standard gradient descent, but we only have 1 data point?

**MAML (Model-Agnostic Meta-Learning)** by Chelsea Finn et al. (2017) is the most famous meta-learning algorithm ever created.

MAML doesn't learn specific features. **MAML learns an initialization**. 
The goal of MAML is to find a set of initial neural network weights $\theta$ that are highly sensitive to gradients.

If you start at $\theta$, and you take *one single gradient step* using 1 picture of a platypus, the weights instantly snap to the perfect configuration for classifying platypuses.

### The Math of MAML (Double Gradient)
MAML requires calculating the derivative of a derivative.

1. Start with meta-weights $\theta$.
2. **Inner Loop**: For a specific task $T_i$, calculate the gradient on the Support Set and take a step to get task-specific weights $\theta_i'$:
   $$ \theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{T_i}(\theta) $$
3. **Outer Loop**: Calculate the loss of those *new* weights $\theta_i'$ on the Query Set. Then, update the original meta-weights $\theta$ to minimize that Query loss!
   $$ \theta \leftarrow \theta - \beta \nabla_\theta \mathcal{L}_{T_i}(\theta_i') $$

Because $\theta_i'$ was calculated using $\nabla_\theta$, taking the derivative of $\theta_i'$ with respect to $\theta$ requires a Hessian matrix (second-order derivatives). This forces $\theta$ to move to a spot in the loss landscape where a single step $\alpha$ leads to massive generalization.

---

## 6. From-Scratch Implementation (MAML PyTorch)

Implementing MAML from scratch requires deep PyTorch knowledge because we must manually apply gradients without using `optimizer.step()` for the Inner Loop.

Fortunately, the `higher` library by Facebook automates this. But for educational purposes, here is the manual Inner Loop logic:

```python
import torch
import torch.nn as nn
from collections import OrderedDict

def maml_inner_loop(model, support_x, support_y, lr_inner=0.01):
    """
    Performs 1 gradient step on the support set, manually updating weights
    without altering the global model parameters (so we can backprop through it).
    """
    # 1. Forward pass on support set
    logits = model(support_x)
    loss = nn.CrossEntropyLoss()(logits, support_y)
    
    # 2. Get gradients for all model parameters
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    
    # 3. Manually create the updated weights (theta_prime)
    fast_weights = OrderedDict()
    for (name, param), grad in zip(model.named_parameters(), grads):
        # theta' = theta - alpha * grad
        fast_weights[name] = param - lr_inner * grad
        
    return fast_weights

def forward_with_fast_weights(model, x, fast_weights):
    """
    A custom forward pass that uses our manually calculated weights
    instead of the model's actual weights.
    """
    # This requires a custom model definition where you can pass weights in,
    # or using torch.func.functional_call (in newer PyTorch versions).
    pass 

# --- Pseudocode for the Outer Loop ---
# 1. fast_weights = maml_inner_loop(model, support_x, support_y)
# 2. query_logits = forward_with_fast_weights(model, query_x, fast_weights)
# 3. meta_loss = CrossEntropyLoss()(query_logits, query_y)
# 4. meta_optimizer.zero_grad()
# 5. meta_loss.backward()  <-- This computes the second-order derivative!
# 6. meta_optimizer.step() <-- Updates the global initialization theta
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Omniglot Prototypical Network**: The `Omniglot` dataset is the "MNIST of Meta-Learning" (1623 handwritten characters from 50 different alphabets, but only 20 examples per character). Write a PyTorch Prototypical Network, train it on 1200 characters, and evaluate its 5-Way 1-Shot accuracy on characters it has never seen.
- 🟡 **First-Order MAML (FOMAML)**: Calculating the second derivative in MAML is extremely memory intensive. Implement FOMAML, which simply drops the second-derivative calculation and assumes it is 0. Compare its speed and accuracy against full MAML on a toy Sine Wave regression task.

### What's Next
| Next | Why |
|------|-----|
| [Recommender Systems](./05-Recommender-Systems.md) | We've looked at esoteric learning paradigms. Now let's return to the real world and look at the technology driving 80% of Netflix's views and Amazon's sales: Deep Recommender Systems. |

---

[← Self-Supervised Learning](03-Self-Supervised-Learning.md) | [Back to Index](../README.md) | [Next: Deep Recommender Systems →](05-Recommender-Systems.md)
