# 🧠 07 - Loss Functions

---

## 📋 Table of Contents
1. [What is a Loss Function?](#what-is-a-loss-function)
2. [Regression Loss Functions](#regression-loss-functions)
   - [Mean Absolute Error (MAE)](#mean-absolute-error-mae-l1-loss)
   - [Mean Squared Error (MSE)](#mean-squared-error-mse-l2-loss)
   - [Huber Loss](#huber-loss)
3. [Classification Loss Functions](#classification-loss-functions)
   - [Binary Cross-Entropy (BCE)](#binary-cross-entropy-bce)
   - [Categorical Cross-Entropy (CCE)](#categorical-cross-entropy-cce)
4. [What's Next](#whats-next)

---

## 🎯 What is a Loss Function?

During Forward Propagation, the network makes a prediction ($\hat{y}$). 

Unless the network is already perfectly trained, this prediction will probably be wrong. It will differ from the true label ($y$). 

A **Loss Function** (or Cost Function) is a mathematical ruler used to measure *exactly how wrong* the prediction is. 
- If the loss is high, the network is doing a terrible job.
- If the loss is close to zero, the network is doing a great job.

The entire goal of training a neural network is to **minimize this loss**.

*Terminology Note:* "Loss" usually refers to the error on a single data point. "Cost" usually refers to the average loss over the entire dataset. In practice, the terms are used interchangeably.

Different problems require different rulers. You wouldn't use a thermometer to measure distance. Similarly, you use different loss functions for Regression vs Classification.

---

## 📈 Regression Loss Functions

Regression problems are when the neural network is predicting a continuous number (e.g., House Price, Temperature, Stock Value).

### Mean Absolute Error (MAE / L1 Loss)

**Intuition:** The simplest way to measure error. Just calculate the absolute difference between the prediction and the reality. If you predict $300,000 and the house costs $350,000, your error is $50,000.

**Formula:** 
$$ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

**Strengths:** Very robust to outliers. If one house prediction is off by a billion dollars, MAE doesn't completely panic; it treats it linearly.
**Weaknesses:** The derivative is a constant steep slope, which means even when the network is very close to the right answer, it takes large gradient steps, making it hard to settle exactly on the target.

### Mean Squared Error (MSE / L2 Loss)

**Intuition:** Instead of taking the absolute value, square the difference. If you are off by 2, your penalty is 4. If you are off by 10, your penalty is 100. 

**Formula:** 
$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

**Strengths:** Because of the curve of a square ($x^2$), the penalty shrinks drastically as you get closer to the truth. This makes the gradient small near the minimum, allowing the network to settle perfectly.
**Weaknesses:** Highly sensitive to outliers. Because errors are squared, being off by a huge amount results in an astronomical penalty, dragging the whole network's weights to try and fix that one outlier.

### Huber Loss

**Intuition:** The best of both worlds. It acts like MSE when the error is small, but if the error gets too large, it switches to acting like MAE. 

**Strengths:** Smooth convergence near the target (like MSE) but ignores crazy outliers (like MAE).
**Weaknesses:** Requires you to tune a hyperparameter ($\delta$) that determines exactly when to switch from MSE to MAE.

---

## 🏷️ Classification Loss Functions

Classification problems are when the network is predicting probabilities of categories (e.g., Spam vs Not Spam, Cat vs Dog). 

You *cannot* use MSE for classification. Squaring probabilities breaks the math required for efficient gradient descent in classification spaces. Instead, we use Entropy.

### Binary Cross-Entropy (BCE)

**Intuition:** Used when there are only two classes (0 or 1). 
Imagine a true label is 1. 
- If the network predicts 0.99, the penalty is almost zero.
- If the network predicts 0.10, the penalty is very high.
- If the network predicts 0.0001, the penalty goes to **infinity**. 

Cross-Entropy heavily penalizes a network for being *confidently wrong*. 

**Formula:**
$$ \text{Loss} = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] $$

*(Don't let the math scare you. If $y=1$, the right half cancels out. If $y=0$, the left half cancels out. It's just a log scale penalty).*

**Typical Setup:**
Output Layer Activation: **Sigmoid**
Loss Function: **BCE**

### Categorical Cross-Entropy (CCE)

**Intuition:** Used when there are 3 or more classes (e.g., MNIST digit classification: 0 through 9).
It works exactly like BCE, but compares vectors of probabilities. The network outputs a probability for every class (summing to 100%), and CCE compares that distribution to the true distribution (where the correct class is 100% and the rest are 0%).

**Typical Setup:**
Output Layer Activation: **Softmax**
Loss Function: **CCE**

---

## 🚀 What's Next

### Key Takeaways
- **Regression Default:** Use Mean Squared Error (MSE).
- **Binary Classification Default:** Use Sigmoid activation + Binary Cross Entropy (BCE).
- **Multi-Class Default:** Use Softmax activation + Categorical Cross Entropy (CCE).
- Loss functions act as the "ruler" measuring how wrong the network is.

### Common Mistakes
- **Using MSE for Classification:** Mathematically, using MSE with Sigmoid creates non-convex loss landscapes (hilly terrain with false bottoms). Gradient descent will get stuck. Always use Cross-Entropy for probabilities.

### Practical Recommendations
- When using PyTorch, `nn.CrossEntropyLoss` actually applies the Softmax function *for you* internally. A common bug is manually applying Softmax to your output layer and then using `nn.CrossEntropyLoss`, essentially applying Softmax twice and ruining the math.

### Next Topic
We have a prediction, and we have a loss function that tells us how bad that prediction is. Now, how do we use that loss to actually fix the weights? We need to traverse the landscape of the loss function using Optimization.

[← Previous Topic](./06-Forward-Propagation.md) | [Next Topic: Gradient Descent →](./08-Gradient-Descent.md)
