# 🛒 Association Rule Mining (Apriori)

> **Prerequisites**: Basic Set Theory, Probability | **Difficulty**: ⭐⭐☆☆☆ Beginner-Intermediate

---

## 📋 Table of Contents
1. [Learning Progression: Association Rules](#1-learning-progression-association-rules)
2. [The Core Metrics: Support, Confidence, Lift](#2-the-core-metrics-support-confidence-lift)
3. [The Apriori Algorithm (Step-by-Step)](#3-the-apriori-algorithm-step-by-step)
4. [Python Implementation (mlxtend)](#4-python-implementation-mlxtend)
5. [Real-World Applications](#5-real-world-applications)
6. [Mini-Project: Market Basket Analysis](#6-mini-project-market-basket-analysis)

---

## 1. Learning Progression: Association Rules

### 🟢 Beginner (Intuition & Analogy)
Imagine you manage a grocery store. You want to figure out which items people frequently buy together so you can place them next to each other on the shelves to increase sales.
You look at 100 receipts (transactions). You notice a pattern: **"If a customer buys Diapers, they usually also buy Beer."**
This is an **Association Rule**. It's an unsupervised learning technique that discovers hidden patterns and relationships in large datasets, most commonly transaction data.

### 🟡 Intermediate (Concept & Reasoning)
An Association Rule consists of two parts:
- **Antecedent (If)**: The item(s) found in the data (e.g., Diapers).
- **Consequent (Then)**: The item(s) found in combination with the antecedent (e.g., Beer).

We write this as: `{Diapers} -> {Beer}`.

However, we can't just make rules out of thin air. We need statistical metrics to prove that the rule is actually significant and not just a random coincidence. To do this, we measure three things: **Support**, **Confidence**, and **Lift**.

### 🔴 Advanced (Algorithmic Depth)
Generating all possible combinations of items in a store with 10,000 products is computationally impossible ($2^{10000}$ subsets). 
The **Apriori Algorithm** solves this using the "Apriori Property": *All non-empty subsets of a frequent itemset must also be frequent.*
If `{Milk, Bread}` is a frequent combination, then `{Milk}` and `{Bread}` individually MUST also be frequent. Conversely, if `{Beer}` is rarely bought, then `{Beer, Diapers}` will also be rarely bought. Apriori uses this property to heavily prune the search space.

---

## 2. The Core Metrics: Support, Confidence, Lift

Let $X$ be the Antecedent and $Y$ be the Consequent. Let $N$ be the total number of transactions.

### 1. Support
How frequently does the itemset appear in the dataset?
$$ \text{Support}(X) = \frac{\text{Transactions containing } X}{N} $$
$$ \text{Support}(X \rightarrow Y) = \frac{\text{Transactions containing both } X \text{ and } Y}{N} $$
*Use case*: Filters out uninteresting, rarely bought items.

### 2. Confidence
Given that a customer bought $X$, how likely are they to buy $Y$? It is a conditional probability $P(Y|X)$.
$$ \text{Confidence}(X \rightarrow Y) = \frac{\text{Support}(X \rightarrow Y)}{\text{Support}(X)} $$
*Use case*: Measures the reliability of the rule. If Confidence is 0.8, then 80% of people who bought $X$ also bought $Y$.

### 3. Lift
How much more likely is $Y$ to be bought when $X$ is bought, compared to $Y$'s normal baseline popularity?
$$ \text{Lift}(X \rightarrow Y) = \frac{\text{Confidence}(X \rightarrow Y)}{\text{Support}(Y)} $$
*Use case*: 
- **Lift > 1**: $X$ and $Y$ are positively correlated (Buying $X$ increases the chance of buying $Y$).
- **Lift = 1**: $X$ and $Y$ are completely independent.
- **Lift < 1**: $X$ and $Y$ are negatively correlated (Buying $X$ means you probably *won't* buy $Y$).

---

## 3. The Apriori Algorithm (Step-by-Step)

Let's assume a minimum Support threshold of 20%.

1. **Step 1 (1-itemsets)**: Calculate the Support for all single items. Drop any items that fall below 20%.
2. **Step 2 (2-itemsets)**: Take the remaining items and create all possible pairs (2-itemsets). Calculate their Support. Drop pairs below 20%.
3. **Step 3 (3-itemsets)**: Create all possible triplets from the remaining 2-itemsets. Calculate Support. Drop those below 20%.
4. **Repeat** until no more itemsets can be generated.
5. **Rule Generation**: For the final frequent itemsets, generate all possible rules ($X \rightarrow Y$) and calculate Confidence and Lift. Filter out rules that don't meet your minimum Confidence/Lift thresholds.

---

## 4. Python Implementation (`mlxtend`)

*Note: Scikit-learn does not have an implementation for Association Rules. We use the popular `mlxtend` library.*

To install: `pip install mlxtend`

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# 1. Create a dummy transaction dataset
dataset = [
    ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
    ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
    ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
    ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
]

# 2. Convert to One-Hot Encoded format
from mlxtend.preprocessing import TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

print(df.head(2)) # True/False matrix of transactions
```

---

## 5. Real-World Applications

- 🛒 **Retail Layout**: Placing positively correlated items near each other (Chips and Salsa).
- 🎁 **Cross-Selling**: E-commerce "Frequently Bought Together" recommendations (Amazon).
- ⚕️ **Healthcare**: Finding associations between symptoms, patient demographics, and diseases (e.g., `{Symptom A, Demographic B} -> {Disease C}`).
- 🌐 **Web Usage Mining**: Analyzing clickstreams to see which web pages are frequently visited together in a single session.

---

## 6. Mini-Project: Market Basket Analysis

**Problem**: We have a one-hot encoded dataset of supermarket transactions. We need to find the strongest rules to drive our new "Buy One Get One" (BOGO) promotion strategy.

```python
# Assuming 'df' from the previous snippet is our transaction matrix

# 1. Find Frequent Itemsets using Apriori
# min_support=0.6 means the itemset must appear in 60% of transactions
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

print("Frequent Itemsets:")
print(frequent_itemsets.sort_values(by='support', ascending=False).head(5))

# 2. Generate Association Rules
# We use Lift as our metric, and require a minimum lift of 1.2
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# 3. Filter and Sort the Rules
# Let's find rules with High Confidence (> 0.8) AND High Lift
strong_rules = rules[(rules['confidence'] > 0.8) & (rules['lift'] > 1.2)]
strong_rules = strong_rules.sort_values(by='lift', ascending=False)

print("\nStrong Association Rules:")
print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Example Output Interpretation:
# If you see: {Eggs} -> {Kidney Beans} | Support: 0.8 | Confidence: 1.0 | Lift: 1.0
# Wait, Lift is 1.0. This means Kidney Beans are bought in 100% of transactions anyway, 
# so buying Eggs doesn't actually *increase* the likelihood of buying Kidney Beans. 
# They are independent! This is why Lift is the most important metric.

# If you see: {Onion} -> {Eggs} | Support: 0.6 | Confidence: 1.0 | Lift: 1.25
# This means everyone who bought Onions ALSO bought Eggs, AND buying Onions 
# increases the likelihood of buying Eggs by 25% over the baseline!
```

### Actionable Business Insight
Based on the high lift of `{Onion} -> {Eggs}`, the supermarket should bundle Onions and Eggs together in a promotion, or place the Onion bin right next to the Egg coolers to encourage impulse cross-buying.

---

[← Anomaly Detection](./07-Anomaly-Detection.md) | [Back to Index](../README.md)
