# Time Series Fundamentals

## What Problem Does This Solve?

Standard machine learning models (like Random Forests or Neural Networks) assume that your data points are **independent and identically distributed (i.i.d.)**. 

If you are predicting whether an email is spam, it doesn't matter if the previous email was spam. The emails are independent.

But what if you are predicting tomorrow's temperature? Or tomorrow's stock price? Or next week's ice cream sales? 
If it is $30^\circ C$ today, it is highly likely to be around $30^\circ C$ tomorrow. **The data points are not independent; they are correlated through time.** 

Time Series Analysis allows us to model this temporal dependence to forecast the future, detect anomalies, and understand underlying patterns that traditional ML models ignore.

---

## What Makes Time Series Different?

If you shuffle the rows in a normal dataset (like a housing price dataset), your ML model will still learn the same patterns.
If you shuffle the rows in a Time Series dataset, **you destroy the data.** The order is the most important feature.

A Time Series is essentially any dataset where the $X$-axis is time and the $Y$-axis is a numeric value. 

---

## Core Concepts: The Components of Time Series

Every time series can be mathematically decomposed into four distinct components. Understanding these is the key to forecasting.

### 1. Trend ($T_t$)
The long-term progression of the series. 
- Is the company's revenue generally going up year over year? That's an **Upward Trend**.
- Is the cost of computing power generally going down? That's a **Downward Trend**.
Trends don't have to be linear; they can be exponential (like user growth in a viral app).

### 2. Seasonality ($S_t$)
Predictable, repeating fluctuations that occur within a fixed timeframe.
- **Daily:** Electricity usage spikes every evening.
- **Weekly:** B2B software usage drops every weekend.
- **Yearly:** Retail sales spike every December for the holidays.
Seasonality has a known, fixed frequency.

### 3. Cyclic Behavior ($C_t$)
Rises and falls that are *not* of a fixed frequency.
- Economic recessions (they happen every 5 to 10 years, but you don't know exactly when).
- The Bitcoin halving cycle.
Cyclic behavior is much harder to predict than seasonality because the duration of the cycle varies.

### 4. Noise / Remainder ($R_t$)
The random, unpredictable fluctuations in the data. 
- A sudden spike in traffic because a celebrity tweeted about your product.
- A drop in sales because of a random server outage.
If you build a perfect model, the only thing left over (the residuals) should be pure noise.

---

## Additive vs. Multiplicative Decomposition

When we combine these components, they usually take one of two forms:

**Additive Model:** 
$$ Y_t = T_t + S_t + R_t $$
Use this when the seasonal variation is relatively constant over time. (e.g., Ice cream sales always increase by exactly 100 units in the summer, regardless of whether base sales are 1,000 or 10,000).

**Multiplicative Model:**
$$ Y_t = T_t \times S_t \times R_t $$
Use this when the seasonal variation increases as the trend increases. (e.g., Airline passenger numbers increase by 20% every summer. As the airline grows over 10 years, that 20% represents a much larger absolute number of people).

---

## Visualizing Time Series Decomposition

The best way to understand a time series is to plot it and decompose it. 
*(In a real notebook, we would use `statsmodels.tsa.seasonal_decompose` to visualize this).*

1. **Original Data:** A jagged line moving upwards.
2. **Trend:** A smooth line cutting through the middle of the jagged data.
3. **Seasonality:** A perfect, repeating wave (e.g., oscillating between +50 and -50).
4. **Noise:** Random static hovering around 0.

---

## From Scratch Implementation: Generating Time Series Data

Let's generate a synthetic time series that combines Trend, Seasonality, and Noise.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create 365 days of data
time = np.arange(365)

# 1. Trend: A slow, linear increase
trend = time * 0.1

# 2. Seasonality: A sine wave with a 30-day frequency
seasonality = 10 * np.sin(2 * np.pi * time / 30)

# 3. Noise: Random normal distribution
np.random.seed(42)
noise = np.random.normal(0, 2, 365)

# Combine them additively
time_series = trend + seasonality + noise

# In a notebook, we would plot this:
# plt.plot(time, time_series)
# plt.title("Synthetic Time Series: Trend + Seasonality + Noise")
# plt.show()
```

---

## Common Failure Cases

1. **Stationarity Ignorance:** Many statistical forecasting models (like ARIMA) require the data to be *stationary* (meaning the mean and variance don't change over time). If you feed raw, trending data into these models, they will fail catastrophically.
2. **Look-ahead Bias:** Accidentally using data from the future to predict the past. (e.g., doing a train/test split randomly instead of chronologically).
3. **Overfitting to Noise:** Building a complex Neural Network that perfectly memorizes the random noise in the training set, resulting in terrible real-world forecasts.

---

## Industry Applications

- **Retail (Walmart, Target):** Inventory management. Knowing exactly how many units of a specific product to stock in a specific store on a specific Tuesday.
- **Energy Sector:** Forecasting electricity demand to know exactly how many power plants to turn on.
- **Finance:** Quantitative trading algorithms analyzing millisecond-level stock price movements.

---

## Key Takeaways

1. Time series data cannot be randomly shuffled; the order contains the information.
2. Every time series is composed of Trend, Seasonality, Cycles, and Noise.
3. Seasonality is predictable and fixed; Cyclic behavior is unpredictable.
4. Before forecasting, you must decompose the series to understand its underlying structure.

## Next Topic

Now that we understand the anatomy of a time series, how do we actually predict the future? We will explore moving averages, statistical models, and modern forecasting algorithms.

Navigation:

[← Previous Topic](./03-Causal-Inference.md) | [Back to Index](./README.md) | [Next Topic: Time Series Forecasting →](./05-Time-Series-Forecasting.md)
