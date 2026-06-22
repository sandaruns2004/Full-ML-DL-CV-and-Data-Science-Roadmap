# 📈 Time Series Deep Dive (ARIMA to Transformers)

> **Prerequisites**: Regression, Sequence Models | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Unique Nature of Time Series](#1-the-unique-nature-of-time-series)
2. [Classical Approaches (ARIMA)](#2-classical-approaches-arima)
3. [Facebook Prophet (Decomposition)](#3-facebook-prophet-decomposition)
4. [Deep Learning for Time Series (LSTMs & CNNs)](#4-deep-learning-for-time-series-lstms--cnns)
5. [The N-BEATS Architecture](#5-the-n-beats-architecture)
6. [Temporal Fusion Transformers (TFT)](#6-temporal-fusion-transformers-tft)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Unique Nature of Time Series

A Time Series is a sequence of data points indexed in time order (e.g., daily stock prices, hourly temperature, minute-by-minute website traffic).

Unlike standard tabular data, time series data violates the core assumption of Machine Learning: **IID (Independent and Identically Distributed)**. 
Today's stock price is absolutely dependent on yesterday's stock price.

**Key Time Series Components:**
1. **Trend**: The long-term progression of the series (e.g., GDP going up over decades).
2. **Seasonality**: Repeated, predictable fluctuations (e.g., Ice cream sales spike every summer).
3. **Cyclical**: Fluctuations that don't have a fixed period (e.g., Economic recessions).
4. **Noise**: Random, unpredictable variation.

---

## 2. Classical Approaches (ARIMA)

For decades, statisticians used **ARIMA** (AutoRegressive Integrated Moving Average) to forecast time series.

ARIMA combines three mathematical concepts:
1. **AR (AutoRegressive) - $p$**: Predicts the current value based on a linear combination of $p$ past values.
2. **I (Integrated) - $d$**: Subtracts current values from previous values $d$ times to make the time series "stationary" (removing trend so the mean/variance is constant over time).
3. **MA (Moving Average) - $q$**: Predicts the current value based on past forecasting errors.

While mathematically sound, ARIMA requires rigorous statistical testing (Dickey-Fuller tests) to manually find the right $p, d, q$ parameters, and it struggles heavily with complex multiple seasonalities (e.g., daily AND yearly patterns simultaneously).

---

## 3. Facebook Prophet (Decomposition)

In 2017, Facebook open-sourced **Prophet**. It replaced complex ARIMA math with an intuitive generalized additive model (GAM).

Prophet decomposes a time series into three separate functions:
$$ y(t) = g(t) + s(t) + h(t) + \epsilon_t $$
1. $g(t)$: **Trend** (Piecewise linear growth).
2. $s(t)$: **Seasonality** (Fourier series to model weekly/yearly cycles).
3. $h(t)$: **Holidays** (Manual list of dates that cause irregular spikes, like Black Friday).

**Why Prophet is loved by industry:**
It works out-of-the-box with missing data, handles outliers gracefully, and doesn't require the data to be stationary. It allows business analysts to tune parameters intuitively (e.g., "increase yearly seasonality flexibility").

---

## 4. Deep Learning for Time Series (LSTMs & CNNs)

Deep learning can model time series without any manual feature engineering.

**Data Preparation (Sliding Windows):**
To feed a time series into an LSTM, you must create a "Sliding Window". 
If window size = 5:
- Input 1: `[t_1, t_2, t_3, t_4, t_5]` $\rightarrow$ Target 1: `t_6`
- Input 2: `[t_2, t_3, t_4, t_5, t_6]` $\rightarrow$ Target 2: `t_7`

**1D CNNs (WaveNet / TCN):**
LSTMs are slow. Temporal Convolutional Networks (TCNs) use **Dilated 1D Convolutions**. By increasing the dilation rate at each layer, the CNN can look incredibly far back in time without requiring infinite layers, completely outperforming LSTMs in both speed and accuracy for tasks like Audio synthesis (WaveNet).

---

## 5. The N-BEATS Architecture

In 2019, Element AI published **N-BEATS** (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting). It proved that simple Multi-Layer Perceptrons (MLPs) could beat highly complex LSTMs and Statistical models.

N-BEATS uses a deep stack of fully connected blocks. 
Crucially, it is **interpretable**. 
- Some blocks in the network are mathematically constrained to only output polynomial curves (acting as Trend predictors).
- Other blocks are constrained to output Fourier harmonics (acting as Seasonality predictors).

The outputs of all blocks are summed together to create the final forecast. It bridged the gap between Deep Learning power and Classical interpretability.

---

## 6. Temporal Fusion Transformers (TFT)

The current king of Time Series forecasting is the **Temporal Fusion Transformer** (Google, 2020).

Time series data in the real world is incredibly messy. You often have multiple types of variables at once:
1. **Past Targets**: (e.g., yesterday's sales).
2. **Known Future Inputs**: (e.g., We know what day of the week tomorrow is; we know tomorrow is a holiday).
3. **Static Metadata**: (e.g., The store's location, the product category).

Standard LSTMs struggle to separate these. The TFT handles them perfectly:
- Uses an LSTM encoder to process the past sequence.
- Uses Gated Residual Networks (GRNs) to skip irrelevant features.
- Uses a **Transformer Attention Mechanism** to find long-term dependencies.
- Outputs **Quantile Forecasts** (e.g., Instead of predicting exactly 100 sales, it predicts a 10th percentile of 80 and a 90th percentile of 120, giving you a confidence interval).

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Prophet Stock Baseline**: Download 5 years of daily Bitcoin prices. Use the `prophet` Python library to train a model. Plot the forecast for the next 30 days and extract the generated trend and seasonality components.
- 🟡 **PyTorch Forecasting**: The `pytorch-forecasting` library makes advanced models easy. Build a Temporal Fusion Transformer to predict Energy Demand using a dataset that includes weather forecasts, day of the week, and historical energy usage. Evaluate the P10, P50, and P90 quantile predictions.

### What's Next
| Next | Why |
|------|-----|
| [Knowledge Distillation](./07-Knowledge-Distillation.md) | Shift focus to model compression techniques for deploying models to edge devices. |

---

[← Deep Recommender Systems](05-Recommender-Systems.md) | [Back to Index](../README.md) | [Next: Knowledge Distillation →](07-Knowledge-Distillation.md)
