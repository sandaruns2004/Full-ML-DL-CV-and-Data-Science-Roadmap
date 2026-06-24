# 20 - Time-Series Deep Dive

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Prerequisites**: Deep Learning Foundations | **Estimated Reading Time**: 25 Minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [What is Time-Series Data?](#2-what-is-time-series-data)
3. [Classical Methods (ARIMA)](#3-classical-methods-arima)
4. [Deep Learning (LSTMs & TCNs)](#4-deep-learning-lstms--tcns)
5. [Time-Series Transformers (PatchTST)](#5-time-series-transformers-patchtst)
6. [Key Takeaways](#6-key-takeaways)
7. [Next Topic](#7-next-topic)

---

# 1. What Problem Does This Solve?

Most Machine Learning models operate on static, independent data points. If you show a CNN a picture of a dog, it predicts "Dog". The prediction does not depend on the picture you showed it 5 minutes ago.

### 🟢 Beginner
But in the real world, data changes over time. You cannot predict the stock market by looking at a single price from today. You must look at the sequence of prices over the last 30 days.

### 🟡 Intermediate
Forecasting the future based on a historical sequence of numbers is called **Time-Series Forecasting**. It is used everywhere: predicting electricity demand, weather forecasting, algorithmic trading, and ICU patient monitoring.

### 🔴 Advanced
Time-Series forecasting is mathematically unique because it introduces **Autocorrelation** (the past influences the future) and **Seasonality** (repeating cyclical patterns). While classical statistical models (like ARIMA) were the gold standard for decades, modern forecasting relies on Temporal Convolutional Networks (TCNs) and specialized Time-Series Transformers (like PatchTST) to predict hundreds of variables simultaneously over long horizons.

---

# 2. What is Time-Series Data?

A Time-Series is a sequence of data points indexed in time order.

**Univariate vs. Multivariate:**
*   **Univariate:** Tracking a single variable over time (e.g., The daily closing price of Apple stock).
*   **Multivariate:** Tracking multiple variables simultaneously (e.g., The daily closing price, trading volume, inflation rate, and CEO sentiment score). Multivariate forecasting is exponentially harder but far more accurate.

**Key Components:**
Time-Series data can usually be decomposed into three underlying mathematical signals:
1.  **Trend:** A long-term increase or decrease (e.g., Global population growth).
2.  **Seasonality:** A repeating, predictable cycle (e.g., Ice cream sales spike every Summer and drop every Winter).
3.  **Noise (Residual):** The random, unpredictable chaos left over after removing the trend and seasonality.

---

# 3. Classical Methods (ARIMA)

Before Neural Networks, statisticians ruled this field. The most famous classical algorithm is **ARIMA (AutoRegressive Integrated Moving Average)**.

*   **AR (AutoRegressive):** Predicts the next value based on a linear combination of the previous values. (If it rained yesterday, it's likely to rain today).
*   **I (Integrated):** Deals with "Trend" by calculating the difference between consecutive days instead of the absolute value.
*   **MA (Moving Average):** Predicts the next value based on the historical *errors* (noise) of the model's past predictions.

*Pros:* It is mathematically interpretable and works perfectly on simple, univariate datasets with strong seasonality.
*Cons:* It cannot handle Multivariate data easily, and it fails completely on complex, chaotic, non-linear relationships.

---

# 4. Deep Learning (LSTMs & TCNs)

To model chaotic, non-linear, multivariate data, we use Deep Learning.

### Recurrent Neural Networks (LSTMs)
As we saw in the NLP module, LSTMs maintain a "Hidden State" (memory) as they process a sequence. You feed the LSTM Day 1, then Day 2, then Day 3. The memory updates at each step, and outputs Day 4. 
*Flaw:* LSTMs are slow to train because they must be processed sequentially. They also suffer from "catastrophic forgetting" on very long time horizons (e.g., trying to remember a pattern from Day 1 when predicting Day 500).

### Temporal Convolutional Networks (TCNs)
Why not use a CNN? TCNs take 1D Convolutions and apply **Dilations** (gaps between the filter points). 
By stacking layers of dilated convolutions, the receptive field grows exponentially. A TCN can look at 10,000 historical data points simultaneously, completely avoiding the sequential bottleneck of LSTMs. TCNs are faster to train and often beat LSTMs on raw accuracy.

---

# 5. Time-Series Transformers (PatchTST)

If Transformers revolutionized NLP by processing entire sequences in parallel using Attention, why not use them for Time-Series?

Early attempts failed. In NLP, words contain deep semantic meaning. In Time-Series, a single number (e.g., `14.5`) has no semantic meaning on its own. If you feed 1,000 individual numbers into a Transformer, the Attention mechanism gets confused by the noise.

**The Breakthrough: PatchTST (2023)**
Instead of feeding individual numbers to the Transformer, we split the time-series into "Patches" (e.g., chunks of 16 days). 
Each 16-day patch is treated as a single "token" (just like a word in a sentence). 
The Transformer calculates the Attention *between the patches*. This allows the model to capture local context (the shape of the 16-day patch) while also looking globally at the long-term historical relationship between patches. 

PatchTST is currently the State-of-the-Art (SOTA) for long-horizon multivariate forecasting.

---

# 6. Key Takeaways

*   **Time-Series Data** is sequential data that exhibits Trend, Seasonality, and Noise.
*   **ARIMA** is the classical statistical gold-standard for simple univariate forecasting.
*   **LSTMs** handle non-linear forecasting by maintaining a continuous memory state, but are slow and struggle with very long horizons.
*   **TCNs (Temporal Convolutional Networks)** use dilated 1D convolutions to process massive historical sequences in parallel.
*   **PatchTST** applies Transformer Attention to "chunks" of time-series data, currently achieving State-of-the-Art results in multivariate forecasting.

---

# 7. Next Topic

We have mastered supervised learning, forecasting, and generative models. But there is a niche area of Machine Learning where we have some labeled data, and a massive amount of unlabeled data. 

How do we combine the certainty of supervised learning with the scale of self-supervised learning?

In the next lesson, we will explore **Semi-Supervised Learning**.

[← Recommender Systems](19-Recommender-Systems.md) | [Back to Index](README.md) | [Next Topic: Semi-Supervised Learning →](21-Semi-Supervised-Learning.md)
