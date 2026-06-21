# 🔬 Introduction to Data Science

> **Prerequisites**: None | **Difficulty**: ⭐☆☆☆☆ Beginner

---

## 📋 Table of Contents

1. [What Is Data Science?](#1-what-is-data-science)
2. [Data Science vs AI vs ML vs Deep Learning](#2-data-science-vs-ai-vs-ml-vs-deep-learning)
3. [The Data Science Lifecycle](#3-the-data-science-lifecycle)
4. [Roles in Data Science Teams](#4-roles-in-data-science-teams)
5. [Industry Applications](#5-industry-applications)
6. [The Data Science Toolkit](#6-the-data-science-toolkit)
7. [Key Terminology](#7-key-terminology)
8. [Your First Data Science Workflow](#8-your-first-data-science-workflow)
9. [What's Next](#9-whats-next)

---

## 1. What Is Data Science?

### 🟢 Beginner

**Everyday Analogy**: Imagine you own a bakery. You notice that you sell more croissants on rainy Mondays. Data Science is the art of discovering patterns like this — except instead of gut feeling, you use *data* and *math* to prove it, and then make decisions based on those patterns (like baking extra croissants when rain is forecasted).

Data Science is the discipline of extracting **meaningful insights** from data using a combination of:
- **Mathematics & Statistics** — to find patterns
- **Programming** — to process and analyze data
- **Domain Knowledge** — to ask the right questions

> **"Data Science is the sexiest job of the 21st century."** — Harvard Business Review, 2012

### 🟡 Intermediate

Data Science exists at the intersection of three core fields:

```
        ┌──────────────────┐
        │   Domain         │
        │   Knowledge      │
        │  (Business,      │
        │   Healthcare,    │
        │   Finance...)    │
        └───────┬──────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
┌───▼───┐  ┌───▼───┐  ┌────▼────┐
│ Math  │  │ DATA  │  │Computer │
│  &    │──│SCIENCE│──│ Science │
│Stats  │  │       │  │   &     │
│       │  │       │  │ Coding  │
└───────┘  └───────┘  └─────────┘
```

**What makes Data Science different from traditional analytics?**

| Traditional Analytics | Data Science |
|----------------------|--------------|
| What happened? (Descriptive) | What will happen? (Predictive) |
| Dashboards & reports | Models & algorithms |
| SQL & Excel | Python, R, ML libraries |
| Backward-looking | Forward-looking |
| Pre-defined questions | Exploratory discovery |

### 🔴 Advanced

**The Scientific Method in Data Science (CRISP-DM)**:

The Cross-Industry Standard Process for Data Mining (CRISP-DM) is the most widely adopted framework in the industry:

1. **Business Understanding** — Define objectives, success criteria, assess the situation
2. **Data Understanding** — Collect, describe, explore, verify data quality
3. **Data Preparation** — Select, clean, construct, integrate, format data
4. **Modeling** — Select techniques, build models, assess models
5. **Evaluation** — Evaluate results against business objectives
6. **Deployment** — Plan deployment, monitoring, maintenance, final report

This is an **iterative** process — you will frequently loop back to earlier stages as you discover new information.

---

## 2. Data Science vs AI vs ML vs Deep Learning

### 🟢 Beginner

Think of these as nested circles — each one fits inside the next:

```
┌─────────────────────────────────────────────────────────────┐
│                   Data Science (DS)                         │
│  The discipline of extracting insights from data.           │
│  Uses math, stats, programming, and domain expertise.       │
│                                                             │
│       ┌──────────────────────────────────────────────┐      │
│       │           Artificial Intelligence (AI)       │      │
│       │  Systems that mimic human intelligence.      │      │
│       │                                              │      │
│       │   ┌──────────────────────────────────────┐   │      │
│       │   │         Machine Learning (ML)        │   │      │
│       │   │  Algorithms that learn from data     │   │      │
│       │   │  without explicit programming.       │   │      │
│       │   │                                      │   │      │
│       │   │   ┌──────────────────────────────┐   │   │      │
│       │   │   │      Deep Learning (DL)      │   │   │      │
│       │   │   │  Neural networks with many   │   │   │      │
│       │   │   │  layers (Deep Neural Nets).  │   │   │      │
│       │   │   └──────────────────────────────┘   │   │      │
│       │   └──────────────────────────────────────┘   │      │
│       └──────────────────────────────────────────────┘      │
│                                                             │
│   Other DS Tools: Data Eng, Big Data, BI, SQL, Viz, Stats   │
└─────────────────────────────────────────────────────────────┘
```

| Term | What It Does | Example |
|------|-------------|---------|
| **Data Science** | Extracts insights from data | "Which customers are likely to leave?" |
| **AI** | Creates intelligent agents | Self-driving cars, chess bots |
| **ML** | Learns patterns from data | Spam filter, price predictor |
| **Deep Learning** | Learns complex hierarchical features | ChatGPT, image recognition |

### 🟡 Intermediate

**Key distinction**: A Data Scientist might use ML as a tool, but they might also solve a business problem with just a SQL query and a bar chart. ML is not always the answer.

| When to Use ML | When NOT to Use ML |
|----------------|-------------------|
| Complex patterns (speech, vision) | Simple rules work perfectly |
| Massive quality data available | Very little or poor quality data |
| Environment changes over time | 100% correctness required |
| Human expertise is scarce | Interpretability is legally required |

### 🔴 Advanced

**The Modern Data Science Stack**:
```
┌─────────────────────────────────────┐
│     Applications                     │
│  scikit-learn │ PyTorch │ TensorFlow │
├─────────────────────────────────────┤
│     Data Layer                       │
│  Pandas │ Spark │ Dask │ Polars     │
├─────────────────────────────────────┤
│     Computation Layer                │
│  NumPy │ SciPy │ CuPy (GPU)        │
├─────────────────────────────────────┤
│     Visualization Layer              │
│  Matplotlib │ Seaborn │ Plotly      │
├─────────────────────────────────────┤
│     Infrastructure                   │
│  Docker │ MLflow │ Airflow │ K8s    │
└─────────────────────────────────────┘
```

---

## 3. The Data Science Lifecycle

### 🟢 Beginner

**Everyday Analogy**: Data Science is like cooking. You don't just throw ingredients into a pot. You:
1. **Decide what to cook** (Define the problem)
2. **Shop for ingredients** (Collect data)
3. **Wash and chop** (Clean data)
4. **Taste and adjust** (Explore and analyze)
5. **Cook the meal** (Build models)
6. **Serve and get feedback** (Deploy and monitor)

### 🟡 Intermediate

The end-to-end Data Science workflow:

```
┌────────────┐   ┌─────────────┐   ┌──────────────┐   ┌───────────┐
│ 1. Define  │──→│ 2. Collect   │──→│ 3. Clean     │──→│ 4. Explore│
│   Problem  │   │    Data      │   │    Data      │   │   (EDA)   │
└────────────┘   └─────────────┘   └──────────────┘   └─────┬─────┘
                                                              │
┌────────────┐   ┌─────────────┐   ┌──────────────┐         │
│ 7. Deploy  │←──│ 6. Evaluate │←──│ 5. Model &   │←────────┘
│  & Monitor │   │             │   │   Analyze    │
└────────────┘   └─────────────┘   └──────────────┘
```

| Phase | Key Activities | Tools |
|-------|---------------|-------|
| Define Problem | Business question, success metrics, KPIs | Stakeholder interviews |
| Collect Data | SQL queries, APIs, web scraping, files | SQL, requests, BeautifulSoup |
| Clean Data | Handle missing values, fix types, remove duplicates | Pandas, NumPy |
| Explore (EDA) | Visualize distributions, correlations, outliers | Matplotlib, Seaborn |
| Model & Analyze | Feature engineering, build models, hypothesis testing | scikit-learn, scipy |
| Evaluate | Accuracy, precision/recall, business impact | scikit-learn metrics |
| Deploy & Monitor | API endpoints, dashboards, data drift detection | FastAPI, MLflow |

### 🔴 Advanced

**Time allocation in real-world Data Science projects** (approximate):

```
Data Collection        ██████████░░░░░░░░░░  ~20%
Data Cleaning          ████████████████░░░░  ~40%
EDA & Visualization    ██████░░░░░░░░░░░░░░  ~15%
Modeling               ████░░░░░░░░░░░░░░░░  ~10%
Evaluation & Tuning    ███░░░░░░░░░░░░░░░░░  ~8%
Deployment             ██░░░░░░░░░░░░░░░░░░  ~7%
```

> **"In reality, 80% of the time is spent on data preparation."** This is why this Data Science Foundations module is so important — it covers the 80%.

---

## 4. Roles in Data Science Teams

### 🟢 Beginner

| Role | What They Do | Analogy |
|------|-------------|---------|
| **Data Analyst** | Answers business questions with data | Reporter/Journalist |
| **Data Scientist** | Builds predictive models | Detective/Researcher |
| **Data Engineer** | Builds data pipelines and infrastructure | Plumber/Electrician |
| **ML Engineer** | Deploys and scales ML models | Factory Manager |
| **Business Analyst** | Translates business needs to data questions | Translator |

### 🟡 Intermediate

| Role | Key Skills | Tools | Salary Range (USD) |
|------|-----------|-------|-------------------|
| Data Analyst | SQL, Excel, Visualization | Tableau, Power BI, SQL | $55K–$85K |
| Data Scientist | Python, Statistics, ML | scikit-learn, Pandas, Jupyter | $90K–$150K |
| Data Engineer | ETL, Cloud, Databases | Spark, Airflow, AWS/GCP | $100K–$160K |
| ML Engineer | MLOps, APIs, Production ML | Docker, Kubernetes, MLflow | $110K–$170K |
| Research Scientist | Deep Learning, Papers | PyTorch, TensorFlow, LaTeX | $120K–$200K+ |

### 🔴 Advanced

**The Modern Data Team Structure**:
```
                    ┌───────────────┐
                    │  VP of Data   │
                    │   / CTO       │
                    └───────┬───────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼───────┐  ┌────────▼────────┐  ┌───────▼───────┐
│ Data Science  │  │ Data Engineering│  │   Analytics   │
│    Team       │  │     Team        │  │    Team       │
│               │  │                 │  │               │
│ • Scientists  │  │ • Data Eng.     │  │ • Analysts    │
│ • ML Eng.     │  │ • Platform Eng. │  │ • BI Eng.     │
│ • Research    │  │ • DevOps        │  │ • Reporting   │
└───────────────┘  └─────────────────┘  └───────────────┘
```

---

## 5. Industry Applications

### 🟢 Beginner

Data Science is used **everywhere**. Here are some examples you encounter daily:

| Industry | Application | Impact |
|----------|------------|--------|
| **E-commerce** | "Recommended for you" | Amazon: 35% of revenue from recommendations |
| **Healthcare** | Disease prediction from X-rays | AI matches radiologist accuracy |
| **Finance** | Credit scoring, fraud detection | Banks save billions in fraud prevention |
| **Social Media** | News feed ranking | Facebook/Instagram: content relevance |
| **Transportation** | Route optimization, surge pricing | Uber: dynamic pricing algorithms |
| **Sports** | Player performance analytics | Moneyball: Oakland A's success story |
| **Entertainment** | Content recommendations | Netflix: 80% of watched content from recs |

### 🟡 Intermediate

**Case Study: Netflix Recommendation System**

Netflix's recommendation engine saves the company an estimated **$1 billion per year** by reducing churn. Here's how:

1. **Data Collection**: What you watch, when, how long, what you pause, rewind, rate
2. **Collaborative Filtering**: "Users similar to you also watched..."
3. **Content-Based Filtering**: "Based on the genre, actors, director of what you liked..."
4. **A/B Testing**: Thousands of experiments on thumbnails, ranking algorithms
5. **Personalization**: Even the artwork shown for the same movie varies per user

### 🔴 Advanced

**Emerging DS Applications**:
- **Autonomous Vehicles**: Sensor fusion, real-time object detection
- **Drug Discovery**: Molecular property prediction, clinical trial optimization
- **Climate Science**: Weather forecasting with transformer architectures
- **Precision Agriculture**: Satellite imagery + ML for crop yield prediction
- **Cybersecurity**: Anomaly detection for intrusion detection systems
- **Generative AI**: LLMs (GPT, Claude), diffusion models (Stable Diffusion)

---

## 6. The Data Science Toolkit

### 🟢 Beginner

| Tool | Purpose | Learn It When |
|------|---------|--------------|
| **Python** | Programming language for DS | Start here! |
| **Jupyter Notebook** | Interactive coding environment | Start here! |
| **Pandas** | Data manipulation | After Python basics |
| **NumPy** | Numerical computing | After Python basics |
| **Matplotlib/Seaborn** | Data visualization | With Pandas |
| **SQL** | Database queries | After Pandas |
| **scikit-learn** | Machine learning | After statistics |

### 🟡 Intermediate

| Category | Tools |
|----------|-------|
| **Data Manipulation** | Pandas, NumPy, Polars, Dask |
| **Visualization** | Matplotlib, Seaborn, Plotly, Tableau |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM |
| **Deep Learning** | PyTorch, TensorFlow, JAX |
| **NLP** | Hugging Face Transformers, spaCy |
| **Big Data** | PySpark, Hadoop, Hive |
| **MLOps** | MLflow, DVC, Weights & Biases |
| **Cloud** | AWS SageMaker, GCP Vertex AI, Azure ML |

---

## 7. Key Terminology

### 🟢 Beginner

| Term | Simple Definition | Example |
|------|------------------|---------|
| **Dataset** | A collection of data, usually in a table | A spreadsheet of customer records |
| **Feature** | A measurable property (column) | Age, income, zip code |
| **Target/Label** | What you're trying to predict | Will they buy? (Yes/No) |
| **Row/Sample** | One data point (record) | One customer's information |
| **Model** | A mathematical function learned from data | A formula that predicts house prices |
| **Training** | Teaching a model using historical data | Showing the model 10,000 examples |
| **Prediction** | Using the trained model on new data | Predicting tomorrow's sales |

### 🟡 Intermediate

| Term | Definition |
|------|-----------|
| **Overfitting** | Model memorizes training data, fails on new data |
| **Underfitting** | Model is too simple to capture patterns |
| **Bias** | Systematic error from wrong assumptions |
| **Variance** | Sensitivity to small fluctuations in training data |
| **Hyperparameter** | Settings you choose before training (learning rate, tree depth) |
| **Cross-Validation** | Technique to evaluate model on multiple data splits |
| **Feature Engineering** | Creating new features from existing data |
| **Data Leakage** | Using information during training that won't be available at prediction time |

---

## 8. Your First Data Science Workflow

### 🟢 Beginner

Let's walk through a complete, simple Data Science workflow using Python:

```python
# ============================================================
# YOUR FIRST DATA SCIENCE WORKFLOW
# Problem: Understand the Iris flower dataset
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Step 1: Load Data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Step 2: Explore Data
print("=" * 50)
print("STEP 2: DATA EXPLORATION")
print("=" * 50)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nStatistics:\n{df.describe().round(2)}")
print(f"\nSpecies counts:\n{df['species'].value_counts()}")
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Step 3: Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribution of a feature
sns.histplot(data=df, x='sepal length (cm)', hue='species', kde=True, ax=axes[0])
axes[0].set_title('Sepal Length Distribution by Species')

# Relationship between features
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', 
                hue='species', s=60, ax=axes[1])
axes[1].set_title('Sepal vs Petal Length')

# Correlation
corr = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[2])
axes[2].set_title('Feature Correlations')

plt.tight_layout()
plt.savefig('first_ds_workflow.png', dpi=150)
plt.show()

# Step 4: Simple Analysis
print("\n" + "=" * 50)
print("STEP 4: KEY INSIGHTS")
print("=" * 50)
for species in df['species'].unique():
    subset = df[df['species'] == species]
    avg_petal = subset['petal length (cm)'].mean()
    print(f"  {species}: avg petal length = {avg_petal:.2f} cm")

print("\nInsight: Petal length is the strongest separator between species!")
print("This is a feature that would be very useful for a classification model.")
```

### 🟡 Intermediate

```python
# ============================================================
# COMPLETE ML PIPELINE: Iris Classification
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Prepare data
X = df.drop('species', axis=1)
y = iris.target

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=iris.target_names)}")
```

### 🔴 Advanced

**Understanding the Pipeline Mathematically**:

1. **StandardScaler**: $X_{std} = \frac{X - \mu}{\sigma}$ — centers data around 0 with unit variance
2. **Random Forest**: An ensemble of $B$ decision trees, each trained on a bootstrap sample. Final prediction is the mode (classification) or mean (regression) across all trees:
   $$\hat{y} = \text{mode}\{T_1(x), T_2(x), \ldots, T_B(x)\}$$
3. **Accuracy**: $\frac{\text{Correct Predictions}}{\text{Total Predictions}}$ — but be cautious with imbalanced datasets!

---

## 9. What's Next

| Next Topic | Why |
|------------|-----|
| [Python for Data Science](./02-Python-for-Data-Science.md) | Master the programming tools used throughout this module |
| [Data Collection](./05-Data-Collection.md) | Learn how to acquire data from various sources |
| [Exploratory Data Analysis](./07-Exploratory-Data-Analysis.md) | Understand your data before building models |

---

[Back to Main Index](../README.md) | [Next: Python for Data Science →](./02-Python-for-Data-Science.md)
