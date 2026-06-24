# Advanced Statistics For Data Science

## What Problem Does This Solve?

Imagine you are a Data Scientist at an e-commerce company. The marketing team runs a campaign and observes that the conversion rate increased from 2.0% to 2.2%. They are thrilled and want to roll out the campaign globally, which will cost $500,000. 

They ask you: *"Is this 0.2% increase real, or did we just get lucky with the people who visited the site today?"*

Basic descriptive statistics (mean, median, mode) tell you what happened in the past. **Advanced inferential statistics tell you what you can expect to happen in the future**, and most importantly, how confident you can be about it. It allows us to quantify uncertainty.

Without advanced statistics, businesses make multi-million dollar decisions based on random noise.

---

## Why Basic Statistics Is Not Enough

Basic statistics summarizes data. If you have 10,000 customers, you can easily calculate the average revenue per user (ARPU). But what if you only have a sample of 1,000 customers, and you want to infer the ARPU for the remaining 9,000? 

Basic statistics breaks down when:
1. We only have a sample, not the entire population.
2. We need to compare two groups to see if they are truly different (e.g., A/B testing).
3. We need to know if an effect is large enough to matter to the business.

Advanced statistics bridges the gap between *observation* and *decision*.

---

## Intuition: The Sampling Distribution

Before we dive into formulas, let's build intuition using a story.

Suppose you want to know the average height of all adults in your city. It's impossible to measure everyone. 
1. You measure 100 people at random. The average is 170cm.
2. Your friend measures a *different* 100 people. Their average is 172cm.
3. Another friend measures 100 people. Their average is 169cm.

Every time you take a sample, the average will be slightly different. But if you plotted all these different averages on a histogram, they would form a beautiful bell curve (Normal Distribution) centered exactly on the *true* city average.

This is the **Sampling Distribution**. It is the most important concept in statistics because it tells us how much our sample average is likely to bounce around just by pure luck.

---

## Core Concepts

### 1. Confidence Intervals

A confidence interval is a range of values that we are reasonably sure contains the true population parameter.

Instead of saying: *"The average user spends $50."*
We say: *"We are 95% confident that the average user spends between $46 and $54."*

**Business Value:** It prevents executives from overreacting to small changes. If the marketing team's new campaign generated $51 on average, but the confidence interval is [$45, $57], we know the result is not statistically reliable.

#### Mathematics
$$ CI = \bar{x} \pm Z \cdot \frac{s}{\sqrt{n}} $$
Where:
- $\bar{x}$ = sample mean
- $Z$ = Z-score (confidence level, e.g., 1.96 for 95%)
- $s$ = sample standard deviation
- $n$ = sample size

### 2. Hypothesis Testing Review

Hypothesis testing is how we prove that a finding is not just random noise.

1. **Null Hypothesis ($H_0$)**: The status quo. (e.g., The new marketing campaign has no effect).
2. **Alternative Hypothesis ($H_1$)**: What you want to prove. (e.g., The new campaign increases conversion).
3. **p-value**: The probability of seeing our data *if* the null hypothesis were true.

If the p-value is very small (typically < 0.05), we say: *"It's highly unlikely we would see this data by pure chance. Therefore, the new campaign probably worked."*

### 3. Statistical Power

The p-value protects us from False Positives (Type I Error). But what about False Negatives (Type II Error)? 

**Statistical Power** is the probability that our test will correctly reject the null hypothesis when it is false. In business terms: *If the marketing campaign actually works, what is the probability our test will detect it?*

If your test has low power (usually because your sample size is too small), you might abandon a profitable campaign just because your test couldn't detect its effect. Industry standard is usually 80% power.

### 4. Effect Size

Statistical significance (p-value) tells you if an effect exists.
**Effect Size** tells you if the effect matters.

If you have 10 million users, even a tiny increase in conversion of 0.001% might be statistically significant (p < 0.05). But is it worth the engineering effort to deploy the new code? Probably not.

**Cohen's d** is a common metric for effect size:
$$ d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}} $$

---

## Visualizations

Imagine two distributions representing the Control Group (blue) and Treatment Group (orange).

1. **Low Effect Size:** The curves overlap almost completely. Even if statistically significant (due to massive sample size), the real-world difference is negligible.
2. **High Effect Size:** The curves are distinctly separated. The treatment clearly moved the needle.

*(In a real notebook environment, we would use Seaborn/Matplotlib to plot these overlapping KDE plots).*

---

## From Scratch Implementation: Confidence Intervals

Let's calculate a 95% confidence interval for user session lengths using pure Python.

```python
import numpy as np
import scipy.stats as stats

# Synthetic data: Session lengths in seconds
np.random.seed(42)
sessions = np.random.normal(loc=120, scale=30, size=500)

mean = np.mean(sessions)
std_err = stats.sem(sessions) # Standard error of the mean
confidence_level = 0.95
degrees_freedom = len(sessions) - 1

# Calculate interval using the t-distribution
ci = stats.t.interval(confidence_level, degrees_freedom, mean, std_err)

print(f"Sample Mean: {mean:.2f} seconds")
print(f"95% Confidence Interval: [{ci[0]:.2f}, {ci[1]:.2f}] seconds")
```

---

## Common Failure Cases

1. **P-Hacking**: Running multiple tests and only reporting the ones with p < 0.05. If you run 20 random tests, 1 will be significant just by chance.
2. **Ignoring Effect Size**: Deploying a complex model that improves accuracy by 0.1% just because it is statistically significant.
3. **Underpowered Tests**: Running a test for too short a time, finding no significance, and incorrectly concluding the feature doesn't work.

---

## Industry Applications

- **Pharmaceuticals**: Clinical trials require immense statistical rigor (high power, strict p-values) because human lives are at stake.
- **E-Commerce**: Companies like Amazon and Booking.com run thousands of tests daily. They use rigorous power calculations to determine exactly how long a test needs to run.
- **Manufacturing**: Six Sigma quality control relies heavily on sampling distributions to detect defects on assembly lines.

---

## Key Takeaways

1. Advanced statistics is the science of making decisions under uncertainty.
2. The Sampling Distribution is the engine that makes inference possible.
3. P-values tell you if an effect is real; Effect Size tells you if it matters.
4. Statistical Power ensures you have enough data to detect an effect if it exists.

## Next Topic

Now that we understand the mathematical foundation of uncertainty, how do we actively design experiments to test our business hypotheses? 

Navigation:

[Back to Index](./README.md) | [Next Topic: Experimental Design & A/B Testing →](./02-Experimental-Design-And_AB_Testing.md)
