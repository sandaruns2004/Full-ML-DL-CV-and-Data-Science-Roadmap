# ⚖️ Inferential Statistics & Hypothesis Testing

> **Prerequisites**: [Probability & Distributions](./09-Probability-And-Distributions.md) | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents

1. [Descriptive vs. Inferential Statistics](#1-descriptive-vs-inferential-statistics)
2. [Populations and Samples](#2-populations-and-samples)
3. [Hypothesis Testing (The Core Framework)](#3-hypothesis-testing-the-core-framework)
4. [P-Values Explained](#4-p-values-explained)
5. [A/B Testing (T-Tests)](#5-ab-testing-t-tests)
6. [Type I & Type II Errors](#6-type-i--type-ii-errors)
7. [What's Next](#7-whats-next)

---

## 1. Descriptive vs. Inferential Statistics

### 🟢 Beginner

- **Descriptive Statistics** (Mean, Median, Std Dev) tells you what happened in your *existing* dataset.
- **Inferential Statistics** uses that dataset to make mathematical predictions or generalizations about a larger group you *don't* have data for.

**Example**:
You survey 1,000 voters.
- *Descriptive*: "52% of the people we surveyed said they will vote for Candidate A."
- *Inferential*: "Based on this survey, we are 95% confident that between 49% and 55% of the *entire country* will vote for Candidate A."

---

## 2. Populations and Samples

### 🟢 Beginner

Because you usually cannot collect data on everyone or everything, you must use samples.

- **Population ($N$)**: Every possible member of the group you are studying (e.g., all 330 million people in the US).
- **Sample ($n$)**: The subset you actually measured (e.g., the 1,000 people you surveyed).

**The goal of Inferential Statistics is to calculate a statistic from the Sample to estimate the true parameter of the Population.**

### 🟡 Intermediate

**Sampling Bias**: If your sample is not truly random, your inferences will be wrong. 
*Example: Surveying people leaving a gym about their exercise habits will not give you a true representation of the national average.*

---

## 3. Hypothesis Testing (The Core Framework)

### 🟡 Intermediate

Hypothesis testing is the scientific method applied to data. It is how tech companies run experiments (A/B testing) to see if a new feature actually improves metrics, or if the improvement was just random luck.

**The 4 Steps of Hypothesis Testing:**

1. **State the Null Hypothesis ($H_0$)**: The baseline assumption. Usually states that "nothing happened" or "there is no difference."
2. **State the Alternative Hypothesis ($H_a$)**: What you are trying to prove. "The new feature increased sales."
3. **Choose a Significance Level ($\alpha$)**: Usually 0.05 (5%). This is your threshold for proof.
4. **Calculate a Test Statistic & P-Value**: Run the math. If the p-value is less than $\alpha$, you reject the Null Hypothesis.

---

## 4. P-Values Explained

### 🟡 Intermediate

The p-value is the most misunderstood metric in statistics.

**What a p-value IS:**
The probability of seeing the data you collected (or more extreme data) IF the Null Hypothesis was completely true.

**What a p-value is NOT:**
It is NOT the probability that your hypothesis is correct.

**Example:**
You test a new website layout. The Null Hypothesis ($H_0$) says "The new layout does not change conversion rates."
You run the test and get a **p-value of 0.03**.

*Interpretation:* "If the new layout truly had zero effect, there is only a 3% chance we would see conversion numbers this high just by random luck."
Since 3% is less than our 5% threshold ($\alpha = 0.05$), we reject the idea that it was random luck. We conclude the new layout actually works!

---

## 5. A/B Testing (T-Tests)

### 🔴 Advanced

A T-Test compares the means of two different groups to see if they are statistically significantly different from each other. This is the math behind A/B testing.

**Independent Two-Sample T-Test Example in Python:**
Does Website Version B generate more revenue per user than Version A?

```python
import numpy as np
from scipy import stats

# Simulated revenue data for two website versions
np.random.seed(42)
version_a_revenue = np.random.normal(loc=50.0, scale=10.0, size=500) # Mean 50
version_b_revenue = np.random.normal(loc=52.0, scale=10.0, size=500) # Mean 52

# Run an independent t-test
t_stat, p_value = stats.ttest_ind(version_a_revenue, version_b_revenue)

print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Result is statistically significant! Version B is different.")
else:
    print("Result is not significant. We cannot prove Version B is different.")
```

**Other types of tests you should know:**
- **Paired T-Test**: Comparing the *same* subjects before and after an intervention (e.g., blood pressure before and after medicine).
- **ANOVA (Analysis of Variance)**: Like a t-test, but for comparing 3 or more groups.
- **Chi-Square Test**: Used for comparing categorical data (e.g., is there a relationship between gender and political party?).

---

## 6. Type I & Type II Errors

### 🔴 Advanced

In hypothesis testing, you will sometimes make the wrong decision. 

| | We Reject the Null Hypothesis (Say there IS an effect) | We Fail to Reject the Null (Say there is NO effect) |
|---|---|---|
| **Null is actually TRUE (No effect)** | ❌ **Type I Error** (False Positive) | ✅ Correct Decision |
| **Null is actually FALSE (Real effect)**| ✅ Correct Decision | ❌ **Type II Error** (False Negative) |

- **Type I Error ($\alpha$)**: You launch the new website thinking it's better, but it actually isn't. (A false alarm). You control this directly by setting your $\alpha$ level to 0.05.
- **Type II Error ($\beta$)**: The new website was actually better, but your test didn't detect it, so you didn't launch it. (A missed opportunity). 

**Statistical Power ($1 - \beta$)**: The probability that you correctly detect an effect if one truly exists. To increase your power, you need a larger sample size!

---

## 7. What's Next

Frequentist statistics (p-values, t-tests) is the standard in science, but it has flaws. It only looks at the data from the current experiment. What if you already have prior knowledge? That's where Bayesian Statistics comes in.

| Next Topic | Why |
|------------|-----|
| [Bayesian Statistics](./11-Bayesian-Statistics.md) | Learn how to update your probability estimates as new evidence arrives. |

---

[← Probability and Distributions](09-Probability-And-Distributions.md) | [Back to Index](../README.md) | [Next: Bayesian Statistics →](11-Bayesian-Statistics.md)
