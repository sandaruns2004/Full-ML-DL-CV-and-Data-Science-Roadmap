# 🕵️ Model Monitoring & Drift Detection

> **Prerequisites**: Model Deployment, Statistics | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Silent Model Failure](#1-silent-model-failure)
2. [Concept Drift vs. Data Drift](#2-concept-drift-vs-data-drift)
3. [Statistical Tests for Drift Detection](#3-statistical-tests-for-drift-detection)
4. [The Monitoring Stack (Prometheus & Grafana)](#4-the-monitoring-stack-prometheus--grafana)
5. [Library Implementation (Evidently AI)](#5-library-implementation-evidently-ai)
6. [Project Ideas & What's Next](#6-project-ideas--whats-next)

---

## 1. Silent Model Failure

In traditional software, when something breaks, it usually throws a massive error (e.g., `500 Internal Server Error` or a Python `KeyError`). You get paged, and you fix it immediately.

Machine Learning models **fail silently**.
If a User Interface update changes the "Age" input field from an integer (`35`) to a string (`"35-40"`), the preprocessing script might silently map the string to an unknown token or a `0`. The Neural Network will still happily execute a forward pass and return a prediction (e.g., `Loan Approved`). 

No errors are thrown. The server returns a `200 OK`. But the company is now issuing thousands of fraudulent loans because the model is hallucinating on bad data.

This is why **Model Monitoring** is absolutely critical. You must monitor the data *before* it hits the model, and monitor the predictions *after* they leave the model.

---

## 2. Concept Drift vs. Data Drift

When monitoring ML systems, we look for two distinct types of degradation over time.

### Data Drift (Covariate Shift)
The statistical distribution of the *input features* $P(X)$ changes over time, but the underlying rules of the world haven't changed.
- *Example*: You train a Face ID model on pictures of people taken during the day. In winter, people start unlocking their phones in the dark. The pixel brightness distribution shifted massively. The model fails.

### Concept Drift
The underlying relationship between the features and the target $P(Y|X)$ changes. The rules of the world have changed.
- *Example*: In 2019, a user buying massive amounts of toilet paper might be flagged as a restaurant supplier or anomalous. In March 2020 (COVID-19), everyone bought massive amounts of toilet paper. The model's definition of "normal behavior" is now completely wrong, even though the input feature data format hasn't changed.

---

## 3. Statistical Tests for Drift Detection

How do we mathematically prove that Data Drift has occurred? We compare the distribution of the Reference Data (the data the model was trained on) to the Current Data (the data hitting the production API today).

We use non-parametric statistical tests:
1. **Kolmogorov-Smirnov (KS) Test**: Used for continuous numerical features. It measures the maximum distance between the empirical cumulative distribution functions of the two datasets. If the distance is large, the distributions have drifted.
2. **Chi-Square Test**: Used for categorical features. Compares the expected frequencies of categories in the training set to the observed frequencies in production.
3. **Population Stability Index (PSI)**: An industry standard specifically designed for credit scoring and finance to measure how much a population's bucketed distribution has shifted over time.

---

## 4. The Monitoring Stack (Prometheus & Grafana)

In an enterprise environment, we need a visual dashboard to monitor the model 24/7.

**1. The Exporter (Python)**
Inside your FastAPI prediction endpoint, every time a prediction is made, you send metrics to a time-series database. (e.g., `Total Predictions Made: 1`, `Prediction Value: 0.85`, `Inference Time: 45ms`).

**2. Prometheus (The Database)**
Prometheus is an open-source time-series database. It "scrapes" your FastAPI endpoint every 10 seconds, pulling all the recorded metrics and storing them efficiently over time.

**3. Grafana (The Dashboard)**
Grafana connects to Prometheus. You build beautiful, real-time visual dashboards. You can set up a Grafana Alert: *"If the percentage of Loan Approvals drops below 5% for more than 15 minutes, send an automated message to the Data Science Slack channel."*

---

## 5. Library Implementation (Evidently AI)

Building statistical drift tests from scratch is tedious. The `evidently` Python library is the open-source standard for generating automated Data Drift reports.

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# 1. Load Data
data = fetch_california_housing(as_frame=True)
df = data.frame

# 2. Simulate Reference vs Current Production Data
# Let's say the model was trained on the first 10,000 houses
reference_data = df.iloc[:10000]

# Now let's simulate the current data hitting the API.
# We will artificially inject Data Drift into the 'MedInc' (Median Income) feature 
# by multiplying it by 1.5 (simulating massive inflation in the area).
current_data = df.iloc[10000:].copy()
current_data['MedInc'] = current_data['MedInc'] * 1.5

# 3. Run Evidently AI Drift Detection
# We configure a report to run the DataDriftPreset metrics suite
drift_report = Report(metrics=[DataDriftPreset()])

# Compare the training data vs production data
drift_report.run(reference_data=reference_data, current_data=current_data)

# 4. Export the Report
# This generates a beautiful interactive HTML dashboard highlighting exactly 
# which features drifted, the P-values of the statistical tests, and histograms!
drift_report.save_html("data_drift_report.html")
print("Report generated! Open 'data_drift_report.html' in your browser.")
```

---

## 6. Project Ideas & What's Next

### Project Ideas
- 🟢 **Evidently Drift Dashboard**: Follow the code above, but apply it to a classification dataset like the Iris dataset. Intentionally shuffle the labels in the `current_data` to simulate Concept Drift. Use Evidently's `ClassificationPreset` to generate a report showing how precision/recall collapse due to the drift.
- 🟡 **FastAPI + Prometheus**: Use the `prometheus_client` Python library inside a FastAPI app. Create a `Counter` that increments every time a prediction is made, and a `Histogram` that records the time taken to run the ML model. Run a local Prometheus server via Docker to scrape your API and view the raw metrics.

### What's Next
| Next | Why |
|------|-----|
| [Edge ML & Optimization](./05-Edge-ML-And-Optimization.md) | We've deployed models to massive cloud servers. But what if we need the model to run completely offline on a tiny Raspberry Pi or an iPhone? We need to compress the model using Edge ML techniques like Quantization and Pruning. |

---

[← MLOps](./03-MLOps.md) | [Back to Index](../README.md) | [Next: Edge ML And Optimization →](./05-Edge-ML-And-Optimization.md)
