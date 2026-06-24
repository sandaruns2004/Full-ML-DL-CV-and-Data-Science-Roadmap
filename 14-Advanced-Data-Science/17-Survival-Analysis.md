# ⏳ Survival Analysis

> **Prerequisites**: Probability, Linear Regression | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Problem with "Time-to-Event" Data](#1-the-problem-with-time-to-event-data)
2. [The Concept of Censoring](#2-the-concept-of-censoring)
3. [The Survival Function & Kaplan-Meier](#3-the-survival-function--kaplan-meier)
4. [The Hazard Function & Cox Proportional Hazards](#4-the-hazard-function--cox-proportional-hazards)
5. [Library Implementation (Lifelines)](#5-library-implementation-lifelines)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. The Problem with "Time-to-Event" Data

In business and medicine, we often want to predict exactly *when* something will happen:
- When will this cancer patient relapse?
- When will this machine part break down (Predictive Maintenance)?
- When will this Netflix subscriber cancel their subscription (Customer Churn)?

If we want to predict "Days until Churn", a naive approach is to use standard Linear Regression. 
However, Linear Regression fails spectacularly for time-to-event data because of a phenomenon called **Censoring**.

---

## 2. The Concept of Censoring

Imagine tracking 100 new SaaS customers for a 1-year study.
- Customer A churns after 3 months. (Event observed: $T=3$)
- Customer B churns after 8 months. (Event observed: $T=8$)
- Customer C is still an active, paying subscriber at the end of the 12-month study!

What do we do with Customer C?
- We can't say their time-to-churn is $12$ months, because they haven't churned yet! They might stay for 50 months.
- We can't just throw them out of the dataset. If we delete all our most loyal customers from the data, our model will be massively biased and assume everyone churns quickly.

Customer C's data is **Right-Censored**. We know their survival time is *at least* 12 months ($T > 12$), but we don't know the exact value. Survival Analysis is specifically designed to handle this mathematical inequality.

---

## 3. The Survival Function & Kaplan-Meier

The fundamental concept is the **Survival Function**, $S(t)$.
It gives the probability that the event has NOT happened by time $t$.
- $S(0) = 1.0$ (At day 0, 100% of people have survived).
- As $t \rightarrow \infty$, $S(t) \rightarrow 0$.

### The Kaplan-Meier Estimator
How do we calculate $S(t)$ with censored data? We use the non-parametric Kaplan-Meier estimator.
It calculates the conditional probability of surviving day $t$, given that you survived up to day $t-1$.

$$ \hat{S}(t) = \prod_{t_i \le t} \left( 1 - \frac{d_i}{n_i} \right) $$
Where:
- $d_i$ = number of people who churned at time $t_i$.
- $n_i$ = number of people "at risk" (still active) right before time $t_i$.

*(Crucially, when a user is censored at month 12, they simply drop out of the $n_i$ "at risk" pool for month 13, without counting as a churn!)*

When you plot the Kaplan-Meier curve, it looks like a staircase stepping down over time. By plotting the curve for "Users with Premium Tier" vs "Users with Basic Tier", you can visually see which group survives longer.

---

## 4. The Hazard Function & Cox Proportional Hazards

Kaplan-Meier is great for plotting, but it cannot handle continuous features (like Age or Income). 
To understand how different variables *impact* survival time, we need a regression model.

We model the **Hazard Function**, $h(t)$. 
While Survival $S(t)$ is the probability of surviving past $t$, the Hazard $h(t)$ is the instantaneous *risk* of dying exactly at time $t$, given you survived up until $t$.

### Cox Proportional Hazards Model
Invented by Sir David Cox in 1972, this is the most widely used survival model.
It assumes the hazard of user $i$ is:
$$ h_i(t) = h_0(t) \times \exp(\beta_1 X_{i1} + \beta_2 X_{i2} + \dots) $$

- $h_0(t)$ is the "baseline hazard" (the risk for an average person). It can be any crazy shape over time.
- The $\exp(\beta X)$ term acts as a multiplier.

If the coefficient for the `Discount_Code_Applied` feature is $-0.69$:
$\exp(-0.69) \approx 0.5$. This means users with a discount code have $0.5\times$ the hazard rate (they are 50% less likely to churn at any given moment) compared to the baseline.

---

## 5. Library Implementation (Lifelines)

The industry standard library for Survival Analysis in Python is `lifelines`.
Let's fit a Cox model to a famous dataset of Telco Customer Churn.

```python
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.datasets import load_telco_churn

# 1. Load Data
# This dataset has 'tenure' (months) and 'Churn' (1/0)
df = load_telco_churn()

# Select a few features to model
features = ['tenure', 'Churn', 'MonthlyCharges', 'SeniorCitizen', 'Partner', 'Dependents']
df = df[features]

# Encode Yes/No to 1/0
df = pd.get_dummies(df, drop_first=True)

# 2. Instantiate and Fit the Cox Proportional Hazards Model
cph = CoxPHFitter()

# We must tell it which column is the Time (duration) and which is the Event (observed/censored)
cph.fit(df, duration_col='tenure', event_col='Churn_Yes')

# 3. View the Results
cph.print_summary()

# 4. Predict Survival Curves for specific individuals
# Let's predict the survival curve for the first 3 customers in the dataset
predictions = cph.predict_survival_function(df.head(3))
# We can easily plot these curves using predictions.plot()
```

*Interpreting the Summary:*
If the `MonthlyCharges` coefficient is positive, it means higher monthly charges increase the hazard rate (risk of churn). If `Partner_Yes` is negative, it means having a partner decreases the hazard rate.

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Kaplan-Meier Visualizer**: Download the `lifelines` Telco Churn dataset. Use the `KaplanMeierFitter` to plot two overlapping survival curves: one for users with `InternetService_Fiber_optic` and one for `InternetService_DSL`. Visually determine which service has higher retention.
- 🟡 **Predictive Maintenance**: Find the NASA Turbofan Engine Degradation dataset. Use a Random Survival Forest (via the `scikit-survival` library) to predict the Remaining Useful Life (RUL) of the aircraft engines based on sensor readings.

### What's Next
| Next | Why |
|------|-----|
| [Geospatial Analysis](./04-Geospatial-Analysis.md) | We've mastered modeling "When" things happen (Time Series & Survival). Now let's master modeling "Where" things happen using Geospatial mapping and coordinates! |

---

[← Causal Inference](02-Causal-Inference.md) | [Back to Index](../README.md) | [Next: Geospatial Analysis & Mapping →](04-Geospatial-Analysis.md)
