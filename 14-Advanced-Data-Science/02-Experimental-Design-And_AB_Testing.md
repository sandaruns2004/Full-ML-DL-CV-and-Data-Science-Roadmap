# Experimental Design and A/B Testing

## What Problem Does This Solve?

Imagine you work at Netflix. The design team has created a new, sleeker thumbnail style for TV shows. They launch it on a Tuesday, and by Thursday, user engagement has dropped by 5%. 

The design team argues: *"It was a holiday on Wednesday, people were outside! It's not the thumbnails!"*
The product team argues: *"The new thumbnails are confusing users!"*

Who is right? Without a controlled experiment, it is impossible to know. You cannot compare Tuesday to Thursday because time itself is a confounding variable.

**A/B Testing (Experimental Design)** solves this by testing both variations *at the exact same time*, on randomly selected user groups. This isolates the variable you are testing, allowing you to definitively prove cause and effect.

---

## Why Companies Run Experiments

At companies like Google, Amazon, and Netflix, almost no feature goes live without an A/B test. 

- **Risk Mitigation:** A bad feature can cost millions. In 2012, an engineer at a large tech company deployed a small UI change that caused a massive drop in revenue. Because it was an A/B test, it only affected 1% of users, saving the company from catastrophic losses.
- **Data-Driven Culture:** Experiments end arguments. Instead of the Highest Paid Person's Opinion (HiPPO) deciding what ships, the data decides.
- **Marginal Gains:** Amazon found that every 100ms of latency cost them 1% in sales. Finding these small, counter-intuitive insights is only possible through rigorous experimentation.

---

## Core Concepts

### 1. Randomization
The bedrock of any experiment. You must assign users to Group A or Group B randomly. If you assign all mobile users to Group A and desktop users to Group B, your test is invalid. Randomization ensures that *on average*, both groups are identical in every way (age, income, behavior) except for the specific feature you are testing.

### 2. Control and Treatment Groups
- **Control Group (A):** The baseline. They see the existing version of the product.
- **Treatment Group (B):** The variant. They see the new feature.

### 3. The Metric of Interest (OEC)
Before starting, you must define the Overall Evaluation Criterion (OEC). What are you trying to improve? Click-through rate? Revenue per user? Retention? You cannot change the goalpost after the test starts.

### 4. Statistical Significance
As discussed in the previous chapter, if Group B performs 2% better than Group A, we must calculate the p-value. If p < 0.05, we conclude the difference is statistically significant and not just due to the random noise of the sampling distribution.

---

## Workflow: Running an A/B Test

1. **Formulate a Hypothesis:** *"Changing the 'Buy' button from Green to Red will increase click-through rate by 5% because it stands out more against the background."*
2. **Power Analysis:** Calculate how many users you need to detect a 5% difference with 80% power.
3. **Randomization & Execution:** Run the test, randomly assigning users to Control or Treatment.
4. **Data Collection:** Wait until the required sample size is reached. Do NOT stop the test early just because you see a significant p-value (this is called "peeking" and increases false positives).
5. **Statistical Analysis:** Run a t-test (for continuous metrics like revenue) or a Z-test (for proportions like click-through rate).
6. **Decision:** Ship the feature, iterate, or discard it.

---

## From Scratch Implementation: A/B Test Analysis

Let's simulate a click-through rate A/B test for an email campaign.

```python
import numpy as np
import scipy.stats as stats

# Control Group (A): 10,000 emails sent, 400 clicks
# Treatment Group (B): 10,000 emails sent, 450 clicks

n_A = 10000
clicks_A = 400
p_A = clicks_A / n_A  # 4.0%

n_B = 10000
clicks_B = 450
p_B = clicks_B / n_B  # 4.5%

# Calculate the pooled probability
p_pooled = (clicks_A + clicks_B) / (n_A + n_B)

# Calculate the Standard Error
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_A + 1/n_B))

# Calculate the Z-score
z_score = (p_B - p_A) / se

# Calculate the p-value (two-tailed test)
p_value = stats.norm.sf(abs(z_score)) * 2

print(f"Control Conversion Rate: {p_A*100:.2f}%")
print(f"Treatment Conversion Rate: {p_B*100:.2f}%")
print(f"Z-Score: {z_score:.2f}")
print(f"P-Value: {p_value:.4f}")

if p_value < 0.05:
    print("Result is statistically significant! Ship the treatment.")
else:
    print("Result is not significant. Do not ship.")
```

---

## Common Failure Cases

1. **Peeking at Results (Early Stopping):** Checking the p-value every day and stopping the test the moment it drops below 0.05. This guarantees a high false-positive rate. You must commit to a sample size beforehand.
2. **Simpson's Paradox:** A trend appears in different groups of data but disappears or reverses when these groups are combined. Always check for underlying segments (e.g., mobile vs desktop).
3. **Novelty Effect:** Users might click a new button just because it's new, temporarily inflating the metrics. The effect fades over time. Run tests long enough to capture stable behavior.
4. **Network Effects:** In social networks (like Facebook or Uber), treating one user affects the control group. (e.g., If you give an Uber driver a new routing algorithm, they arrive faster, which takes away rides from control group drivers).

---

## Industry Applications

- **Netflix:** Tests every single piece of artwork. A single title might have 10 different thumbnails, and different users see different variations to maximize viewing time.
- **Google:** Famously tested 41 shades of blue for their toolbar links to see which generated the most clicks, resulting in an extra $200M a year in revenue.
- **Amazon:** Tests UI layouts, recommendation algorithm placements, and even the text of their marketing emails constantly.

---

## Key Takeaways

1. A/B testing is the gold standard for determining causality in business.
2. Randomization ensures the control and treatment groups are comparable.
3. Never stop an experiment early just because you found statistical significance.
4. Beware of the novelty effect and network effects that can contaminate your results.

## Next Topic

A/B testing is perfect when you can randomly assign users. But what if you *can't* run an experiment? What if you want to know if smoking causes cancer, or if a new government policy reduced crime? You cannot randomly force people to smoke. 

For these situations, we need the advanced techniques of **Causal Inference**.

Navigation:

[← Previous Topic](./01-Advanced-Statistics-For-Data-Science.md) | [Back to Index](./README.md) | [Next Topic: Causal Inference →](./03-Causal-Inference.md)
