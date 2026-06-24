# Customer Analytics

## What Problem Does This Solve?

It costs 5 to 25 times more to acquire a new customer than to retain an existing one. 

If a subscription business is losing 10% of its customers every month (Churn), it doesn't matter how good their marketing is—they are filling a leaky bucket and will eventually go bankrupt.

Furthermore, not all customers are equal. The Pareto Principle (80/20 rule) usually applies: 80% of your revenue comes from 20% of your customers. If you treat all customers exactly the same, you will waste money offering discounts to highly loyal customers, and you will lose at-risk customers because you didn't intervene in time.

**Customer Analytics** is the application of data science to understand, segment, and predict customer behavior to maximize revenue and retention.

---

## Core Concepts

### 1. Cohort Analysis
A cohort is simply a group of users who share a common characteristic, usually the date they signed up. 

Instead of looking at a useless vanity metric like "Total Active Users" (which always goes up if you spend enough on marketing), we look at retention by cohort. 
*Of the 1,000 people who signed up in January, how many were still here in February? In March?* 

This generates a **Retention Matrix** (a triangle-shaped heatmap). If retention is degrading for newer cohorts, your product is getting worse, even if total users are increasing.

### 2. Customer Segmentation (RFM Analysis)
We use Unsupervised Learning (like K-Means Clustering) to group customers. The most common business framework for this is **RFM**:
- **Recency:** How recently did the customer purchase?
- **Frequency:** How often do they purchase?
- **Monetary Value:** How much do they spend?

By scoring users 1-5 on these metrics, we can create segments:
- **Champions (5-5-5):** Bought recently, buy often, spend the most. (Reward them, don't give them discounts).
- **At Risk (1-5-5):** Used to spend a lot and buy often, but haven't been seen in a long time. (Send them a massive "We miss you" discount immediately).

### 3. Churn Prediction
Churn is when a customer cancels their subscription or stops buying. 
We use Supervised Learning (like Random Forests or XGBoost) to predict the *probability* that a user will churn in the next 30 days.

We feed the model features like:
- Days since last login
- Number of customer support tickets opened
- Usage drops in the last 7 days

If the model predicts a user has an 85% probability of churning, the system automatically sends them an intervention (e.g., a free month of service or a call from an account manager).

### 4. Customer Lifetime Value (CLV / LTV)
How much total profit will a user generate for the company before they inevitably churn? 

If the CLV of a user is $100, and the marketing team is spending $120 to acquire them (CAC - Customer Acquisition Cost), the company is losing $20 on every sale.

Predicting CLV early (e.g., predicting their lifetime value based on just their first 7 days of behavior) allows companies to know exactly how much they can afford to spend on marketing.

---

## Workflow: Building a Churn Prediction System

1. **Define Churn:** In a subscription business (Netflix), churn is easy: they clicked "cancel." In e-commerce (Amazon), it's hard: how long does a user have to go without buying to be considered "churned"? 90 days? 180 days?
2. **Feature Engineering:** This is 90% of the work. You must aggregate time-series data into static features (e.g., `spend_last_30_days`, `spend_change_vs_previous_month`).
3. **Train the Model:** Use XGBoost or LightGBM.
4. **Determine the Intervention threshold:** Don't just target everyone with >50% probability. Interventions cost money. Target the users where the Expected Value of the intervention is positive.

---

## From Scratch Implementation: RFM Segmentation

```python
import pandas as pd

# Synthetic transaction data
data = {
    'CustomerID': [1, 1, 2, 3, 3, 3, 4],
    'InvoiceDate': ['2023-10-01', '2023-10-25', '2023-01-15', '2023-09-01', '2023-10-05', '2023-10-20', '2022-12-01'],
    'Amount': [50, 75, 200, 20, 30, 25, 500]
}
df = pd.DataFrame(data)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Assume 'today' is 2023-11-01
today = pd.to_datetime('2023-11-01')

# Calculate RFM
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (today - x.max()).days, # Recency
    'CustomerID': 'count',                           # Frequency
    'Amount': 'sum'                                  # Monetary
})

rfm.columns = ['Recency_Days', 'Frequency', 'Monetary_Total']

print(rfm)
# Customer 4 spent $500 (High M), but 335 days ago (Bad R) -> At Risk of Churn.
# Customer 3 bought 3 times recently (Good R, Good F), but low value (Low M) -> Active but low tier.
```

---

## Common Failure Cases

1. **Targeting the Unsaveable:** A model might predict a user is 99% likely to churn. But if they are moving to a country where your service doesn't exist, a 20% discount email won't save them. You must distinguish between "Probability of Churn" and "Probability of Responding to Intervention" (Uplift Modeling).
2. **Data Leakage in Churn:** Including a feature like `canceled_auto_renew_flag`. If the model sees this, it achieves 100% accuracy, but it's useless because the user has already decided to churn. You must predict churn *before* the user takes decisive action.
3. **Ignoring Profitability:** Saving a churning customer who constantly contacts support and returns 50% of their purchases might actually cost the company money.

---

## Industry Applications

- **Telecommunications:** Verizon and AT&T have massive churn prediction teams. If you call to cancel, the system predicts your value and gives the operator a dynamic maximum discount they are allowed to offer you to stay.
- **Mobile Gaming:** Free-to-play games predict the CLV of a new player within 24 hours. If they are flagged as a "Whale" (high potential spender), the game alters its difficulty and offers to keep them engaged.

---

## Key Takeaways

1. Retaining customers is vastly more profitable than acquiring new ones.
2. RFM is the most common heuristic for segmenting customers based on value.
3. Cohort analysis exposes the true health of product retention over time.
4. Churn prediction models trigger automated, personalized interventions to save at-risk revenue.

## Next Topic

Customer Analytics looks at the humans buying the product. Now we must look at the product itself, and how the marketing team drives humans to it in the first place.

Navigation:

[← Previous Topic](./07-Recommender-Systems.md) | [Back to Index](./README.md) | [Next Topic: Marketing & Product Analytics →](./09-Marketing-And_Product_Analytics.md)
