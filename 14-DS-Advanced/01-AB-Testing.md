# 🧪 A/B Testing & Experimental Design

> **Prerequisites**: Statistical Inference, Hypothesis Testing | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Why Predictive Models Aren't Enough](#1-why-predictive-models-arent-enough)
2. [The Anatomy of an A/B Test](#2-the-anatomy-of-an-ab-test)
3. [Sample Size & Minimum Detectable Effect (MDE)](#3-sample-size--minimum-detectable-effect-mde)
4. [Frequentist vs. Bayesian A/B Testing](#4-frequentist-vs-bayesian-ab-testing)
5. [Common Pitfalls (Peeking & Novelty Effects)](#5-common-pitfalls-peeking--novelty-effects)
6. [Library Implementation (Statsmodels)](#6-library-implementation-statsmodels)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. Why Predictive Models Aren't Enough

You just built a state-of-the-art Recommender System. Offline metrics show that its AUC-ROC is 5% better than the old model. You deploy it to production.

A week later, the CEO asks: *"Did your new AI model actually make us more money?"*

You cannot answer this question just by looking at revenue. Maybe revenue went up because it was a holiday weekend. Maybe it went down because of a server outage. 

To definitively prove that your model caused a change in user behavior, you must run a randomized controlled experiment: an **A/B Test**. A/B testing is the gold standard of Data Science; it bridges the gap between AI research and actual business value.

---

## 2. The Anatomy of an A/B Test

An A/B test involves randomly splitting your users into two groups:
- **Control Group (A)**: Receives the existing experience (e.g., the old Recommender System).
- **Variant Group (B)**: Receives the new experience (e.g., the Deep Learning model).

### Core Metrics
Before starting the test, you must clearly define:
1. **The Primary Metric (OEC)**: Overall Evaluation Criterion. The one number you are trying to maximize (e.g., Click-Through Rate or Revenue per User).
2. **Guardrail Metrics**: Metrics that *must not degrade*. (e.g., Page Load Time, Uninstall Rate). If your new model increases clicks by 10% but increases app crashes by 5%, the test is a failure.

---

## 3. Sample Size & Minimum Detectable Effect (MDE)

You cannot just run a test for 2 days, see that Variant B is winning by 2%, and stop the test. That is statistical suicide. 

Before the test starts, you must mathematically calculate exactly how many users you need. This requires four parameters:
1. **Baseline Conversion Rate**: What is the current metric for Group A? (e.g., 5.0%).
2. **Minimum Detectable Effect (MDE)**: What is the smallest relative change you care about detecting? (e.g., A 10% relative lift, meaning you want to detect if the rate goes from 5.0% to 5.5%).
3. **Statistical Power ($1 - \beta$)**: The probability of detecting a difference if one actually exists. Industry standard is **80%**.
4. **Significance Level ($\alpha$)**: The probability of finding a fake difference (False Positive). Industry standard is **5%** ($p = 0.05$).

Plugging these into a Power Analysis formula will tell you: *"You need exactly 32,450 users per variant."* You must run the test until you hit this number!

---

## 4. Frequentist vs. Bayesian A/B Testing

### Frequentist (The Classic Approach)
Uses $p$-values. The null hypothesis ($H_0$) states there is no difference between A and B. 
If $p < 0.05$, you reject $H_0$. 
*Problem*: Non-technical stakeholders don't understand $p$-values. They ask "What is the probability that B is better than A?" and the Frequentist approach mathematically cannot answer that question.

### Bayesian (The Modern Approach)
Used by companies like VWO and Optimizely. Instead of $p$-values, it uses Bayes' Theorem to calculate the exact probability distribution of the metrics.
It can directly output: *"There is a 94.2% chance that Variant B is better than Variant A, and the expected revenue increase is between \$1.20 and \$1.80 per user."*
Bayesian testing is much more intuitive for business leaders and handles "peeking" much better than Frequentist testing.

---

## 5. Common Pitfalls (Peeking & Novelty Effects)

**1. The Peeking Problem (p-hacking)**
If you check your Frequentist A/B test dashboard every single day, and stop the test the moment $p < 0.05$, you have invalidated your results. Because of random statistical noise, almost all A/B tests will temporarily show significance at some point. You must wait until the target sample size is reached.

**2. Novelty Effect**
When you launch a drastic UI redesign, users might click on the new buttons just because they look new. Variant B will win massively in Week 1, and then drop below Variant A in Week 3. Always run tests long enough to outlast the novelty effect.

**3. Day of Week Effect**
Never run a test for 10 days. User behavior on weekends is vastly different than on Tuesdays. Always run tests in multiples of 7 days (e.g., 14 days, 21 days).

---

## 6. Library Implementation (Statsmodels)

Let's run a Frequentist A/B test analyzing Click-Through Rate (CTR) using Python.

```python
import numpy as np
import statsmodels.stats.api as sms
import scipy.stats as stats

# --- 1. Power Analysis (Before the Test) ---
baseline_rate = 0.10  # Current CTR is 10%
mde_relative = 0.05   # We want to detect a 5% relative lift (10.0% -> 10.5%)

# Calculate absolute effect size
effect_size = sms.proportion_effectsize(baseline_rate, baseline_rate * (1 + mde_relative))

# Calculate required sample size
required_n = sms.NormalIndPower().solve_power(
    effect_size, 
    power=0.80, 
    alpha=0.05, 
    ratio=1
)
print(f"Required sample size per group: {int(np.ceil(required_n)):,}")
# Output: Required sample size per group: 61,048

# --- 2. Analyzing the Results (After the Test) ---
# We ran the test and collected the data:
conversions_A = 6150
total_A = 61050

conversions_B = 6480
total_B = 61050

# Perform a Two-Proportion Z-Test
z_stat, p_value = sms.proportions_ztest(
    [conversions_B, conversions_A], 
    [total_B, total_A], 
    alternative='larger' # Testing if B is strictly > A
)

print(f"\nVariant A CTR: {conversions_A/total_A:.4%}")
print(f"Variant B CTR: {conversions_B/total_B:.4%}")
print(f"Z-Statistic: {z_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: STATISTICALLY SIGNIFICANT! Deploy Variant B.")
else:
    print("Result: Not significant. Keep Variant A.")
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Bayesian Calculator**: Write a Python script using the `scipy.stats.beta` distribution. Input the exact same conversions/totals as the code block above, and calculate the exact probability that Variant B's true conversion rate is greater than Variant A's. Plot the two overlapping Beta distributions using `matplotlib`.
- 🟡 **Multi-Armed Bandits**: Sometimes A/B testing is too slow. Implement a **Thompson Sampling** algorithm (a type of Reinforcement Learning). Create 3 simulated slot machines (Variant A, B, C) with hidden win probabilities. Write a loop where the algorithm dynamically routes more traffic to the winning variant in real-time, minimizing "regret".

### What's Next
| Next | Why |
|------|-----|
| [Causal Inference](./02-Causal-Inference.md) | A/B testing is perfect when you *can* randomize users. But what if you can't? What if a government passes a new law, and you want to know its effect on the economy? You can't A/B test a law. You need Causal Inference. |

---

[← Multi Task Learning](../13-Advanced/09-Multi-Task-Learning.md) | [Back to Index](../README.md) | [Next: Causal Inference →](./02-Causal-Inference.md)
