# Causal Inference

## What Problem Does This Solve?

Imagine you are analyzing data for a large retail company. You notice a massive spike in ice cream sales every July. You also notice a massive spike in shark attacks every July. 

A naive machine learning model might conclude: *"Eating ice cream causes shark attacks. If we stop selling ice cream, we can save lives!"*

This is the classic **Correlation vs. Causation** problem. The two variables move together, but neither causes the other. A hidden third variable (a *confounding variable*)—in this case, summer weather—causes both people to buy ice cream and people to swim in the ocean (where sharks live).

While A/B testing solves this by forcing randomization, **you cannot always run an A/B test.** 
- You cannot randomly force half your customers to smoke to see if it causes cancer.
- You cannot randomly launch a massive marketing campaign in half a city but not the other half (billboards are visible to everyone).
- You cannot randomly assign users a premium subscription to see if it increases their engagement; they have to choose to buy it themselves.

**Causal Inference** is the set of statistical methods used to estimate the causal impact of an intervention when A/B testing is impossible, unethical, or too expensive.

---

## Core Concepts

### 1. The Fundamental Problem of Causal Inference: Counterfactuals
If you take an aspirin and your headache goes away, did the aspirin cure it? Or was it going to go away on its own anyway?

You can only observe one reality: You took the aspirin and felt better.
You cannot observe the **Counterfactual** reality: You *didn't* take the aspirin, but everything else in the universe remained exactly the same.

Because we can never observe the counterfactual, causal inference attempts to simulate it using data.

### 2. Confounding Variables
A confounding variable is an outside influence that changes the effect of a dependent and independent variable.

**Business Example:** 
You launch a loyalty program. You notice that users in the loyalty program spend 3x more money than users who are not. 
Did the loyalty program *cause* them to spend more?
Probably not. The *confounding variable* is "pre-existing brand loyalty." Users who already loved your brand opted into the program, and they were always going to spend more money anyway. The program itself might have zero causal effect.

### 3. Propensity Score Matching
How do we solve the loyalty program problem? 

We use **Propensity Score Matching (PSM)**. 
1. We train a logistic regression model to predict the *probability* (propensity) of a user joining the loyalty program based on their past behavior (age, past spend, app usage).
2. For every user who joined the program, we find a user who *didn't* join, but who had the exact same propensity score (they looked exactly the same on paper).
3. We compare the future spend of these two matched groups. 

By artificially matching identical users, we simulate an A/B test without actually running one!

### 4. Natural Experiments (Difference-in-Differences)
Sometimes the world runs an experiment for you. 
Suppose a new tax law is passed in California but not in Nevada. You can compare the economic changes in California to Nevada. However, Nevada might naturally be growing faster than California anyway.

**Difference-in-Differences (DiD)** calculates the normal difference between the two states *before* the law, and compares it to the difference *after* the law. If the gap suddenly widens, that widening is the causal effect of the law.

---

## Workflow: Establishing Causality from Observational Data

1. **Draw a Causal Graph (DAG):** Draw out your assumptions. A -> B, but does C -> A? You must map out potential confounders.
2. **Identify the Treatment and Outcome:** What is the intervention (loyalty program)? What is the outcome (revenue)?
3. **Control for Confounders:** Use techniques like Propensity Score Matching or regression to mathematically hold confounding variables constant.
4. **Estimate the Average Treatment Effect (ATE):** Calculate the difference in outcome between the treated and untreated (matched) groups.

---

## Library Implementation: Causal Inference with DoWhy

Microsoft's `dowhy` library provides a unified framework for causal inference.

```python
import pandas as pd
import numpy as np
from dowhy import CausalModel

# Synthetic observational data
# Confounder (X) causes both Treatment (T) and Outcome (Y)
np.random.seed(42)
X = np.random.normal(0, 1, 1000)  # Pre-existing brand loyalty
T = np.where(X + np.random.normal(0, 0.5, 1000) > 0, 1, 0) # Joins loyalty program
Y = 5 * T + 2 * X + np.random.normal(0, 1, 1000) # Spends money

df = pd.DataFrame({'Loyalty_Score': X, 'Joined_Program': T, 'Revenue': Y})

# Step 1: Model the causal mechanism
model = CausalModel(
    data=df,
    treatment='Joined_Program',
    outcome='Revenue',
    common_causes=['Loyalty_Score']
)

# Step 2: Identify the causal effect
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Step 3: Estimate the causal effect using Propensity Score Matching
estimate = model.estimate_effect(
    identified_estimand,
    method_name="backdoor.propensity_score_matching"
)

print(estimate)
# The output will show the causal effect is ~5.0, correctly isolating it from the confounder (2*X).
```

---

## Common Failure Cases

1. **Unobserved Confounders:** PSM only works on variables you can measure. If "emotional attachment to brand" is driving purchases, but you don't have a column for it in your database, your causal estimate will be biased.
2. **Over-controlling:** Controlling for a variable that was actually caused by the treatment. (e.g., Controlling for "items added to cart" when estimating the effect of a website redesign on sales. The redesign *caused* them to add items to the cart, so controlling for it hides the effect).
3. **Assuming Correlation is Enough:** "Our model predicts churn with 95% accuracy. Therefore, if we change the variables the model looks at, we can stop churn." No. Predictive models exploit correlations; changing a correlated variable (like "days since last login") does not cause the user to stay.

---

## Industry Applications

- **Uber/Lyft:** Used to measure the impact of new pricing algorithms in specific cities when true randomization is impossible due to network effects.
- **Healthcare:** Estimating the effectiveness of a new drug based on electronic health records (observational data) when clinical trials are too slow.
- **Marketing:** Measuring the true ROI of television advertising (since you can't A/B test a TV commercial).

---

## Key Takeaways

1. Correlation is not causation. Confounding variables are the enemy.
2. We can never observe the counterfactual, so we must simulate it.
3. Propensity Score Matching allows us to create synthetic control groups from observational data.
4. Causal inference is required when A/B testing is impossible or unethical.

## Next Topic

We've covered how to measure causality at a specific point in time. But what happens when our data isn't a static snapshot, but a continuous stream over days, months, and years? We must learn how to handle the dimension of time.

Navigation:

[← Previous Topic](./02-Experimental-Design-And_AB_Testing.md) | [Back to Index](./README.md) | [Next Topic: Time Series Fundamentals →](./04-Time-Series-Fundamentals.md)
