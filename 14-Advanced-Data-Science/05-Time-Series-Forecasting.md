# Time Series Forecasting

## What Problem Does This Solve?

Now that we know how to decompose a time series into its core components (Trend, Seasonality, Noise), how do we actually predict what will happen tomorrow, next month, or next year?

Forecasting is the mathematical process of extrapolating past behavior into the future. 
- **Supply Chain:** How many winter coats do we need to manufacture in August to meet December's demand?
- **Server Infrastructure:** How much AWS compute capacity will we need on Black Friday?
- **Finance:** What will our cash flow look like next quarter?

If you guess too high, you waste millions on unused inventory or servers. If you guess too low, your website crashes or you run out of stock. Accurate forecasting directly translates to massive cost savings.

---

## Core Forecasting Methods

There is no "one size fits all" forecasting model. You must choose the model based on the shape of your data.

### 1. Moving Average (MA)
The simplest possible forecast. You predict tomorrow's value by taking the average of the last $N$ days.

- **When it works:** When your data has no trend and no seasonality, and you just want to smooth out random noise.
- **When it fails:** It cannot predict trends. If sales are growing every day, a moving average will *always* under-predict the future, because it is dragged down by older, lower numbers.

### 2. Exponential Smoothing (Holt-Winters)
Moving averages treat the data from 10 days ago with the same importance as the data from yesterday. **Exponential Smoothing** assigns exponentially decreasing weights to older observations. 

The **Holt-Winters** variant is incredibly powerful because it explicitly models three things:
1. **Level:** The baseline value.
2. **Trend:** Is it going up or down?
3. **Seasonality:** The repeating pattern.

- **When it works:** Excellent for retail sales data with clear trends and annual seasonality. It is extremely fast to compute.
- **When it fails:** Struggles with complex, multiple seasonalities (e.g., a dataset that has both a weekly cycle and a yearly cycle).

### 3. ARIMA (AutoRegressive Integrated Moving Average)
The gold standard of classical statistics. 

1. **AR (AutoRegressive):** Predicts the future based on a linear combination of past values.
2. **I (Integrated):** Differencing the data (subtracting today from tomorrow) to make the time series *stationary* (removing trend).
3. **MA (Moving Average):** Predicts the future based on past forecast errors.

- **When it works:** Excellent for financial data and macroeconomic indicators where you need strict mathematical proofs of your models.
- **When it fails:** Setting the hyperparameters ($p, d, q$) is notoriously difficult. It assumes linear relationships and breaks down entirely if the data has strong, complex seasonality (unless you use the SARIMA variant).

### 4. Facebook Prophet
An open-source library built by Meta. It treats forecasting not as an autoregressive problem, but as a **curve-fitting exercise**. It models time series as an additive model (Trend + Seasonality + Holidays).

- **When it works:** It is the ultimate "out-of-the-box" tool for business data. It automatically handles missing data, outliers, daily/weekly/yearly seasonality, and holiday spikes (e.g., Thanksgiving).
- **When it fails:** It is heavily reliant on historical patterns. If your business undergoes a massive structural change (like the COVID-19 pandemic), Prophet's curve-fitting will fail because the old curves no longer apply.

---

## Workflow: Building a Forecasting Model

1. **Plot the Data:** Always look at your data first. Is there a trend? Seasonality?
2. **Train/Test Split:** You **cannot** use `train_test_split` with random shuffling. You must split chronologically (e.g., train on 2018-2022, test on 2023).
3. **Establish a Baseline:** Build a Naive forecast (tomorrow will be exactly the same as today). If your complex AI model can't beat the Naive forecast, throw the AI model away.
4. **Train Models:** Train Holt-Winters, ARIMA, and Prophet.
5. **Evaluate:** Use metrics like **MAPE** (Mean Absolute Percentage Error) or **RMSE** (Root Mean Squared Error).
6. **Cross-Validation:** Use "Rolling Origin" cross-validation (training on an expanding window of time).

---

## Library Implementation: Forecasting with Prophet

```python
import pandas as pd
from prophet import Prophet

# Prophet requires strict column names: 'ds' (datestamp) and 'y' (value)
df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=365*3, freq='D'),
    'y': np.random.normal(100, 10, 365*3) + np.linspace(0, 50, 365*3) # Trend + Noise
})

# 1. Initialize the model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True)

# 2. Add specific holidays if needed
model.add_country_holidays(country_name='US')

# 3. Fit the model
model.fit(df)

# 4. Create a dataframe extending 90 days into the future
future = model.make_future_dataframe(periods=90)

# 5. Predict
forecast = model.predict(future)

# 6. Plotting (in a notebook)
# fig1 = model.plot(forecast)
# fig2 = model.plot_components(forecast)
```

---

## Common Failure Cases

1. **Non-Stationarity in ARIMA:** Forgetting to difference the data. If the data has a trend, the mean is changing, violating ARIMA's core assumption.
2. **Black Swan Events:** Models trained on 2019 data failed catastrophically in March 2020. Statistical models do not read the news; they only know history. You must manually intervene or retrain models quickly during structural shifts.
3. **Forecast Drift:** A model trained 6 months ago is likely useless today. Time series models degrade in production faster than any other type of ML model.

---

## Industry Applications

- **Uber:** Uses advanced time series forecasting to predict rider demand in specific city blocks to implement Surge Pricing.
- **Zillow:** Forecasts housing prices in different neighborhoods.
- **Cloud Providers:** AWS and Azure use forecasting to determine exactly when to spin up or shut down physical servers in data centers to save electricity.

---

## Key Takeaways

1. Never use random train/test splits for time series data. Time flows in one direction.
2. Holt-Winters is fast and reliable for standard seasonal data.
3. ARIMA is mathematically rigorous but hard to tune.
4. Prophet is the best tool for business data with holidays and multiple seasonalities.
5. Time series models degrade quickly in production.

## Next Topic

Forecasting is about predicting the expected normal behavior. But what happens when the actual data severely deviates from the forecast? How do we catch hackers, broken sensors, and credit card thieves in real-time?

Navigation:

[← Previous Topic](./04-Time-Series-Fundamentals.md) | [Back to Index](./README.md) | [Next Topic: Anomaly Detection →](./06-Anomaly-Detection.md)
