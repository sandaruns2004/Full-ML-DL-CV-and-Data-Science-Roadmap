# 🎲 Probability and Distributions

> **Prerequisites**: [Descriptive Statistics](./08-Descriptive-Statistics.md) | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents

1. [Foundations of Probability](#1-foundations-of-probability)
2. [Conditional Probability & Independence](#2-conditional-probability--independence)
3. [The Normal Distribution (Gaussian)](#3-the-normal-distribution-gaussian)
4. [Discrete Distributions (Binomial, Poisson)](#4-discrete-distributions-binomial-poisson)
5. [Continuous Distributions (Uniform, Exponential)](#5-continuous-distributions-uniform-exponential)
6. [Central Limit Theorem (The Most Important Math in ML)](#6-central-limit-theorem)
7. [What's Next](#7-whats-next)

---

## 1. Foundations of Probability

### 🟢 Beginner

Machine Learning is entirely built on probability. A spam classifier doesn't say "This IS spam." It says "There is a 98% probability this is spam."

**Basic Terms:**
- **Experiment**: A process with an uncertain outcome (e.g., rolling a die).
- **Sample Space ($S$)**: All possible outcomes. For a die, $S = \{1, 2, 3, 4, 5, 6\}$.
- **Event ($A$)**: A subset of outcomes you care about. Event "rolling an even number" = $\{2, 4, 6\}$.

**Probability Formula:**
If all outcomes are equally likely:
$$P(A) = \frac{\text{Number of favorable outcomes}}{\text{Total number of possible outcomes}}$$

For rolling an even number: $P(A) = \frac{3}{6} = 0.5$ (or 50%).

---

## 2. Conditional Probability & Independence

### 🟡 Intermediate

**Independence**
Two events are independent if the outcome of one does NOT affect the other.
Example: Flipping a coin twice. Getting heads on the first flip doesn't change the odds of getting heads on the second.
$$P(A \text{ and } B) = P(A) \times P(B)$$

**Conditional Probability**
What is the probability of event A happening, *given* that event B already happened?
Example: What is the probability a patient has lung cancer ($A$), *given* that they are a smoker ($B$)?

$$P(A|B) = \frac{P(A \text{ and } B)}{P(B)}$$

---

## 3. The Normal Distribution (Gaussian)

### 🟡 Intermediate

The **Normal Distribution** (Bell Curve) is the king of distributions. It appears everywhere in nature: human heights, blood pressure, IQ scores, and measurement errors.

**Key Properties:**
1. It is symmetrical (Mean = Median = Mode).
2. It is completely defined by two parameters: Mean ($\mu$) and Standard Deviation ($\sigma$).
3. **The Empirical Rule (68-95-99.7 Rule):**
   - **68%** of data falls within 1 standard deviation ($1\sigma$) of the mean.
   - **95%** falls within $2\sigma$.
   - **99.7%** falls within $3\sigma$.

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Generate a normal distribution: Mean=0, Std=1
mu = 0
sigma = 1
x = np.linspace(-4, 4, 100)
y = stats.norm.pdf(x, mu, sigma) # PDF = Probability Density Function

plt.plot(x, y)
plt.title('The Standard Normal Distribution')
plt.fill_between(x, y, where=(x > -1) & (x < 1), alpha=0.3, color='blue', label='68%')
plt.legend()
plt.show()
```

### 🔴 Advanced: Z-Scores Standardization

In Machine Learning, we often **Standardize** our data so that it becomes a Standard Normal Distribution ($\mu=0, \sigma=1$). We do this by calculating the Z-score for every data point:

$$Z = \frac{X - \mu}{\sigma}$$

This allows ML algorithms (like Neural Networks or SVMs) to process features of different scales (like Age vs Income) on the same playing field.

---

## 4. Discrete Distributions

### 🟡 Intermediate

Discrete distributions deal with data that can only take specific, countable values (e.g., number of clicks, number of heads).

**1. The Binomial Distribution**
Models the number of successes in $n$ independent yes/no trials.
*Example: If I flip a coin 10 times, what is the probability of getting exactly 7 heads?*

Parameters:
- $n$ = number of trials
- $p$ = probability of success on a single trial

```python
from scipy.stats import binom

# Probability of getting exactly 7 heads in 10 flips (p=0.5)
n = 10
p = 0.5
prob_7 = binom.pmf(7, n, p) # PMF = Probability Mass Function
print(f"P(7 heads) = {prob_7:.4f}") # 0.1172
```

**2. The Poisson Distribution**
Models the number of events occurring in a fixed interval of time or space.
*Example: How many customer support calls will we receive in the next hour?*

Parameter:
- $\lambda$ (Lambda) = expected (average) number of events.

```python
from scipy.stats import poisson

# If we average 5 calls an hour, what is the probability of getting exactly 8 calls?
lambda_val = 5
prob_8 = poisson.pmf(8, lambda_val)
print(f"P(8 calls) = {prob_8:.4f}") # 0.0653
```

---

## 5. Continuous Distributions

### 🟡 Intermediate

Continuous distributions deal with data that can take any value within a range (e.g., time, height, money).

**1. The Uniform Distribution**
Every outcome in a range has an equal probability. A flat rectangle.
*Example: A random number generator picking a decimal between 0 and 1.*

**2. The Exponential Distribution**
Models the amount of time until a specific event occurs. It is closely related to the Poisson distribution.
*Example: How long will a machine run before it breaks down?*

```python
from scipy.stats import expon

# If an earthquake happens on average every 10 years (scale=10)
# What is the probability the next one happens within 5 years?
scale = 10
# CDF = Cumulative Distribution Function (Probability of X <= 5)
prob_within_5 = expon.cdf(5, scale=scale) 
print(f"Probability = {prob_within_5:.4f}") # 0.3935
```

---

## 6. Central Limit Theorem (The Most Important Math in ML)

### 🔴 Advanced

The **Central Limit Theorem (CLT)** is the reason why the Normal Distribution is so important.

**The Theorem:**
Take ANY dataset. It doesn't matter how weirdly shaped it is (bimodal, skewed, uniform). If you randomly draw large enough samples from that dataset, calculate the mean of each sample, and plot those means... **the resulting plot of the means will form a perfect Normal Bell Curve.**

This is magical. It means that even if our real-world data is extremely messy and non-normal, we can still use normal-distribution statistics to make highly accurate predictions about the population mean.

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Create a heavily skewed population (NOT normal)
population = np.random.exponential(scale=2, size=100000)

# 2. Take 1,000 samples (each of size 50) and find their means
sample_means = []
for _ in range(1000):
    sample = np.random.choice(population, size=50)
    sample_means.append(np.mean(sample))

# 3. Plot the sample means
plt.hist(sample_means, bins=30, edgecolor='black')
plt.title('Distribution of Sample Means (It is Normal!)')
plt.show()
```

---

## 7. What's Next

Now that we understand probability distributions and the Central Limit Theorem, we can finally stop just *describing* data and start *proving* things about the real world using Hypothesis Testing.

| Next Topic | Why |
|------------|-----|
| [Inferential Statistics](./10-Inferential-Statistics.md) | Learn A/B Testing, p-values, and how to prove that your ML model is actually better than random chance. |

---

[← Descriptive Statistics](08-Descriptive-Statistics.md) | [Back to Index](../README.md) | [Next: Inferential Statistics & Hypothesis Testing →](10-Inferential-Statistics.md)
