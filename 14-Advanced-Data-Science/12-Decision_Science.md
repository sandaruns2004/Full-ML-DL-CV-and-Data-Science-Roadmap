# Decision Science

## What Problem Does This Solve?

A Machine Learning Engineer builds a model that predicts customer churn with 85% accuracy. They deploy the model, and it correctly flags 10,000 customers who are about to leave. 

Now what?

Machine Learning predicts the future, but it does not tell you *what to do about it*. Should you offer them a 10% discount? A 50% discount? A free phone? Nothing at all? 

If you offer a 50% discount to all 10,000 users, you might save them, but the company loses millions of dollars in margin. If you offer them nothing, they churn, and the company loses millions in future revenue.

**Decision Science** bridges the gap between predictive models and business actions. It calculates the optimal action to maximize the **Expected Value** for the business.

---

## Core Concepts

### 1. Expected Value (EV)
The foundation of Decision Science. It is the probability of an event multiplied by its financial payoff.

$$ EV = \sum (Probability_i \times Payoff_i) $$

**Example:**
An ML model predicts a user has a 20% chance of churning. Their Customer Lifetime Value (CLV) is $1,000. 
- The expected loss if we do nothing is $200.
- We have an intervention (a phone call) that costs $50 and reduces the churn probability to 5%.
- New expected loss: 5% * $1000 = $50.
- Net benefit of the phone call: $200 (old loss) - $50 (new loss) - $50 (cost of call) = **+$100 EV.**

Because the EV is positive, we should make the phone call. If the EV was negative, we should do nothing, even though the user might churn.

### 2. Risk Analysis and Uncertainty
Models are never 100% accurate. What if the churn probability is actually a confidence interval between 10% and 30%?

Decision scientists use **Monte Carlo Simulations** to run 10,000 parallel universes, drawing random probabilities from the confidence intervals. 
This generates a risk profile: *"We expect this intervention to generate $1M in profit, but there is a 5% chance it results in a $200k loss."* Executives need this risk profile to make informed choices.

### 3. Scenario Planning (Optimization)
Often, you have constraints. You only have 100 customer service agents to make phone calls, but 10,000 at-risk customers. Who do you call?

You cannot just call the people with the highest churn probability (they might be low-value users). You cannot just call the highest-value users (they might not be churning). 
You must optimize for the intersection of High Probability of Churn + High CLV + High Probability of Responding to the Call. This is solved using Linear Programming and Uplift Modeling.

---

## Workflow: From Prediction to Decision

1. **Prediction:** ML model outputs $P(Churn)$.
2. **Valuation:** Data team calculates the Customer Lifetime Value ($CLV$).
3. **Causal Estimate:** A/B testing provides the $Uplift$ (how much a specific intervention changes $P(Churn)$).
4. **Cost:** Finance provides the cost of the intervention ($C$).
5. **Decision Engine:** An automated script calculates: $EV = (P \times CLV) - ((P - Uplift) \times CLV) - C$. 
6. **Action:** If $EV > 0$, the system automatically triggers an API to send the discount email.

---

## From Scratch Implementation: Expected Value Calculation

```python
import numpy as np
import pandas as pd

# 1. Output from our ML models and BI systems
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4],
    'churn_probability': [0.90, 0.85, 0.20, 0.15], # ML Model
    'clv': [100, 2000, 5000, 50]                   # BI System
})

# 2. Known parameters from past A/B tests
intervention_cost = 50
intervention_uplift = 0.40 # Reduces churn probability by absolute 40%

# 3. Calculate the Expected Value of doing NOTHING
customers['ev_do_nothing'] = -(customers['churn_probability'] * customers['clv'])

# 4. Calculate the Expected Value of INTERVENING
customers['new_churn_prob'] = np.maximum(0, customers['churn_probability'] - intervention_uplift)
customers['ev_intervene'] = -(customers['new_churn_prob'] * customers['clv']) - intervention_cost

# 5. Calculate the Net ROI of the intervention
customers['net_roi'] = customers['ev_intervene'] - customers['ev_do_nothing']

# 6. The Decision Logic
customers['action'] = np.where(customers['net_roi'] > 0, 'Send Discount', 'Do Nothing')

print(customers[['customer_id', 'churn_probability', 'clv', 'net_roi', 'action']])
```

**Output Analysis:**
- Customer 1: 90% chance of churning, but low value ($100). The $50 intervention costs more than it saves. **Action: Do Nothing.**
- Customer 2: 85% chance of churning, high value ($2000). The $50 intervention has massive ROI. **Action: Send Discount.**

---

## Common Failure Cases

1. **Ignoring the Cost of Action:** Assuming emails are "free." If you spam users with 50 emails a week trying to prevent churn, they will unsubscribe, destroying their future LTV. Every action has a cost.
2. **Treating Predictions as Ground Truth:** A Random Forest predicting 0.85 does not mean the user will exactly 85% churn. It is a probabilistic estimate. Point estimates are dangerous in decision science; you should use probability distributions.
3. **The Optimizer's Curse:** If you perfectly optimize your supply chain to maximize profit based on historical data, you make it incredibly fragile. A single unpredictable event (like a pandemic) breaks the perfectly optimized system.

---

## Industry Applications

- **Insurance:** Actuaries use decision science to set premium prices. If they price too high, they lose customers. If they price too low, a hurricane bankrupts the company.
- **Airlines:** Yield Management. Should we sell this seat today for $100, or wait until tomorrow and hope a desperate business traveler buys it for $500? Decision science calculates the exact EV of holding the seat.

---

## Key Takeaways

1. Machine Learning predicts the future; Decision Science decides what to do about it.
2. All business decisions should be framed as maximizing Expected Value.
3. You must combine ML probabilities, CLV, and intervention costs to make an automated decision.
4. Monte Carlo simulations help quantify the risk and uncertainty of decisions.

## Next Topic

Decision Science is the brain of the operation. But how do we package all of this—the data pipelines, the ML models, the decision engine, and the UI—into something people can actually use? We must learn how to build Data Products.

Navigation:

[← Previous Topic](./11-Feature_Stores_And_Modern_Data_Platforms.md) | [Back to Index](./README.md) | [Next Topic: Data Product Thinking →](./13-Data_Product_Thinking.md)
