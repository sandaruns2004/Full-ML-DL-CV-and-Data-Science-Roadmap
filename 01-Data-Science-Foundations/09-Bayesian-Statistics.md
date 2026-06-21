# 🔮 Bayesian Statistics

> **Prerequisites**: [Inferential Statistics](./08-Inferential-Statistics.md) | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents

1. [Frequentist vs. Bayesian Thinking](#1-frequentist-vs-bayesian-thinking)
2. [Bayes' Theorem Explained](#2-bayes-theorem-explained)
3. [The Three Pillars: Prior, Likelihood, Posterior](#3-the-three-pillars-prior-likelihood-posterior)
4. [Naïve Bayes Classifier (A Practical ML Application)](#4-naïve-bayes-classifier-a-practical-ml-application)
5. [Markov Chain Monte Carlo (MCMC)](#5-markov-chain-monte-carlo-mcmc)
6. [What's Next](#6-whats-next)

---

## 1. Frequentist vs. Bayesian Thinking

### 🟢 Beginner

In statistics, there are two warring factions:

**1. Frequentists (The Traditionalists)**
- Probability is the long-run frequency of events.
- "If I flip this coin infinite times, 50% will be heads. Therefore, the probability of heads is 0.5."
- Uses p-values, t-tests, and confidence intervals.
- The parameter (e.g., the true conversion rate) is fixed, but unknown.

**2. Bayesians (The Updaters)**
- Probability is a measure of belief or certainty.
- "Based on my prior knowledge, I think the probability of heads is 0.5. After flipping it and getting 10 heads in a row, I am updating my belief to think this is a rigged coin."
- Uses Prior, Likelihood, and Posterior distributions.
- The parameter itself is a random variable with a distribution.

> **Analogy:** You lose your phone in the house. A Frequentist will search every room systematically, assuming the phone is in one fixed spot. A Bayesian will start searching the bedroom first because they know they usually leave it there (Prior belief), and update their search path as they clear rooms.

---

## 2. Bayes' Theorem Explained

### 🟡 Intermediate

Bayes' Theorem provides the mathematical rule for updating your beliefs when you see new evidence.

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

Where:
- **$P(A)$**: Probability of A happening (Before any evidence).
- **$P(B)$**: Probability of the evidence happening overall.
- **$P(B|A)$**: Probability of seeing the evidence B, if A is true.
- **$P(A|B)$**: Probability of A being true, given the new evidence B.

**The Classic Example: Medical Testing**

Suppose 1% of the population has a rare disease: $P(\text{Disease}) = 0.01$.
You take a test that is 99% accurate: $P(\text{Positive}|\text{Disease}) = 0.99$.
The test also has a 5% false positive rate: $P(\text{Positive}|\text{No Disease}) = 0.05$.

You test Positive. What is the actual probability you have the disease?
Most people guess 99%. Let's use Bayes' Theorem:

$$P(\text{Disease}|\text{Positive}) = \frac{P(\text{Positive}|\text{Disease}) \times P(\text{Disease})}{P(\text{Positive})}$$

To find $P(\text{Positive})$ (The total chance of testing positive), we add true positives and false positives:
$P(\text{Positive}) = (0.99 \times 0.01) + (0.05 \times 0.99) = 0.0099 + 0.0495 = 0.0594$

$$P(\text{Disease}|\text{Positive}) = \frac{0.99 \times 0.01}{0.0594} = 0.166 \text{ or } 16.6\%$$

Even with a 99% accurate test, because the disease is so rare, a positive test only means you have a **16.6% chance** of actually having it. This is why doctors run follow-up tests!

---

## 3. The Three Pillars: Prior, Likelihood, Posterior

### 🔴 Advanced

In Bayesian Machine Learning, we rewrite Bayes' Theorem slightly differently:

$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

1. **Prior Distribution**: What we believed before seeing data. (e.g., "Conversion rates are usually around 2%").
2. **Likelihood**: How well the observed data explains our hypothesis. (e.g., "We saw 5 conversions out of 100 visitors").
3. **Posterior Distribution**: Our updated belief after combining Prior and Likelihood.

If you have very little data, your Posterior will look a lot like your Prior. 
If you have massive amounts of data, the data overwhelms the Prior, and the Posterior will look entirely like the Likelihood (Bayesian and Frequentist results converge!).

---

## 4. Naïve Bayes Classifier (A Practical ML Application)

### 🟡 Intermediate

The **Naïve Bayes Classifier** is a machine learning algorithm based directly on Bayes' Theorem. It is famously used for spam filtering and text classification.

It is called "Naïve" because it makes a massive assumption: It assumes every feature is completely independent of every other feature (which is rarely true in real life, but the algorithm still works incredibly well).

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd

# Load data
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
# "Gaussian" Naive Bayes assumes the features follow a normal distribution
model = GaussianNB()

# Train the model (It calculates Priors and Likelihoods for every class)
model.fit(X_train, y_train)

# Predict (It calculates the Posterior for each class and picks the highest)
accuracy = model.score(X_test, y_test)
print(f"Naïve Bayes Accuracy: {accuracy:.2%}")
```

---

## 5. Markov Chain Monte Carlo (MCMC)

### 🔴 Advanced

For complex real-world problems, calculating the denominator of Bayes' Theorem ($P(B)$) requires solving impossible multidimensional integrals. 

For decades, Bayesian statistics was mathematically beautiful but computationally impossible. Then computers got faster, and **MCMC** was born.

**MCMC** is a class of algorithms that allows us to sample from a complex distribution without actually knowing its exact mathematical formula. Instead of solving the impossible math equation, MCMC simulates a random "walker" exploring the parameter space. Where the walker spends the most time is where the highest probability lies.

Libraries like `PyMC` and `Stan` use MCMC (specifically, the No-U-Turn Sampler or NUTS) to perform modern Bayesian inference.

```python
# Conceptual example of modern Bayesian modeling using PyMC
# (Requires: pip install pymc)

"""
import pymc as pm

with pm.Model() as conversion_model:
    # 1. Define Priors (We think conversion rate is around 2-5%)
    true_rate_A = pm.Beta('rate_A', alpha=2, beta=50)
    true_rate_B = pm.Beta('rate_B', alpha=2, beta=50)
    
    # 2. Define Likelihood (The data we actually observed)
    obs_A = pm.Binomial('obs_A', n=1000, p=true_rate_A, observed=45)
    obs_B = pm.Binomial('obs_B', n=1000, p=true_rate_B, observed=62)
    
    # 3. Calculate the difference (What we actually care about)
    diff = pm.Deterministic('difference', true_rate_B - true_rate_A)
    
    # 4. Run MCMC to find the Posterior distribution
    trace = pm.sample(2000, return_inferencedata=True)
    
# We can now say: "There is a 98.4% probability that Version B is better than Version A."
# This is much easier for business leaders to understand than p-values!
"""
```

---

## 6. What's Next

We have now covered the core of Data Science mathematics. It is time to move away from equations and focus on how to present data to human beings.

| Next Topic | Why |
|------------|-----|
| [Data Visualization](./10-Data-Visualization.md) | Learn how to create compelling visual narratives using Matplotlib, Seaborn, and Plotly. |

---

[← Previous: Inferential Statistics](./08-Inferential-Statistics.md) | [Back to Main Index](../README.md) | [Next: Data Visualization →](./10-Data-Visualization.md)
