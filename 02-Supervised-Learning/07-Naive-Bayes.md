# 📊 Naive Bayes Classifier

> **Prerequisites**: Probability & Statistics | **Difficulty**: ⭐⭐☆☆☆ Elementary

---

## 📋 Table of Contents
1. [Bayes' Theorem Review](#1-bayes-theorem-review)
2. [The "Naive" Assumption](#2-the-naive-assumption)
3. [Types of Naive Bayes](#3-types-of-naive-bayes)
4. [Mathematical Derivation](#4-mathematical-derivation)
5. [Implementation from Scratch](#5-implementation-from-scratch)
6. [scikit-learn Implementation](#6-scikit-learn-implementation)
7. [Text Classification with Naive Bayes](#7-text-classification-with-naive-bayes)
8. [Practical Guide](#8-practical-guide)
9. [Project Ideas & What's Next](#9-project-ideas--whats-next)

---

## 1. Bayes' Theorem Review

$$P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y) \cdot P(y)}{P(\mathbf{x})}$$

| Term | Name | ML Interpretation |
|------|------|-------------------|
| $P(y\|\mathbf{x})$ | **Posterior** | Probability of class given features |
| $P(\mathbf{x}\|y)$ | **Likelihood** | Probability of features given class |
| $P(y)$ | **Prior** | Base rate of each class |
| $P(\mathbf{x})$ | **Evidence** | Normalizing constant (same for all classes) |

Since $P(\mathbf{x})$ is the same for all classes, we only need:

$$\hat{y} = \arg\max_y P(y) \cdot P(\mathbf{x}|y)$$

---

## 2. The "Naive" Assumption

The **naive** assumption is that features are **conditionally independent** given the class:

$$P(\mathbf{x}|y) = P(x_1|y) \cdot P(x_2|y) \cdots P(x_n|y) = \prod_{j=1}^{n} P(x_j|y)$$

This simplifies computation enormously:

$$\hat{y} = \arg\max_y P(y) \prod_{j=1}^{n} P(x_j|y)$$

**Is this assumption realistic?** Almost never! But Naive Bayes works surprisingly well in practice, especially for text classification.

---

## 3. Types of Naive Bayes

### 3.1 Gaussian Naive Bayes

Assumes features follow a **normal distribution** within each class:

$$P(x_j|y=c) = \frac{1}{\sqrt{2\pi\sigma_{jc}^2}} \exp\left(-\frac{(x_j - \mu_{jc})^2}{2\sigma_{jc}^2}\right)$$

where $\mu_{jc}$ and $\sigma_{jc}$ are the mean and std of feature $j$ in class $c$.

**Use for**: Continuous features.

### 3.2 Multinomial Naive Bayes

Assumes features are **counts** (e.g., word frequencies):

$$P(x_j|y=c) = \frac{N_{jc} + \alpha}{N_c + \alpha n}$$

where $N_{jc}$ is the count of feature $j$ in class $c$, and $\alpha$ is the **Laplace smoothing** parameter.

**Use for**: Text classification (word counts, TF-IDF).

### 3.3 Bernoulli Naive Bayes

Assumes features are **binary** (present/absent):

$$P(x_j|y=c) = p_{jc}^{x_j}(1-p_{jc})^{1-x_j}$$

**Use for**: Binary features (e.g., word presence/absence).

---

## 4. Mathematical Derivation

### Complete Example: Email Spam Classification

**Training data**:
| Email | Contains "free" | Contains "money" | Contains "meeting" | Spam? |
|-------|:-:|:-:|:-:|:-:|
| 1 | 1 | 1 | 0 | Yes |
| 2 | 1 | 0 | 0 | Yes |
| 3 | 0 | 1 | 0 | Yes |
| 4 | 0 | 0 | 1 | No |
| 5 | 0 | 0 | 1 | No |
| 6 | 1 | 0 | 1 | No |

**Priors**: $P(\text{spam}) = 3/6 = 0.5$, $P(\text{not spam}) = 3/6 = 0.5$

**Likelihoods** (with Laplace smoothing, $\alpha=1$):

| Feature | P(feature=1 \| spam) | P(feature=1 \| not spam) |
|---------|:---:|:---:|
| "free" | $(2+1)/(3+2) = 0.6$ | $(1+1)/(3+2) = 0.4$ |
| "money" | $(2+1)/(3+2) = 0.6$ | $(0+1)/(3+2) = 0.2$ |
| "meeting" | $(0+1)/(3+2) = 0.2$ | $(2+1)/(3+2) = 0.6$ |

**New email**: Contains "free"=1, "money"=1, "meeting"=0

$$P(\text{spam}|\mathbf{x}) \propto P(\text{spam}) \cdot P(\text{free}=1|\text{spam}) \cdot P(\text{money}=1|\text{spam}) \cdot P(\text{meeting}=0|\text{spam})$$
$$= 0.5 \times 0.6 \times 0.6 \times 0.8 = 0.144$$

$$P(\text{not spam}|\mathbf{x}) \propto 0.5 \times 0.4 \times 0.2 \times 0.4 = 0.016$$

**Normalize**: $P(\text{spam}|\mathbf{x}) = \frac{0.144}{0.144 + 0.016} = 0.9 = 90\%$

**Prediction**: SPAM! ✓

```python
import numpy as np

# Manual Naive Bayes computation
print("=" * 50)
print("MANUAL NAIVE BAYES — SPAM DETECTION")
print("=" * 50)

# Priors
p_spam = 0.5
p_not_spam = 0.5

# Likelihoods (with Laplace smoothing)
# P(feature=1 | class)
likelihoods_spam = {'free': 0.6, 'money': 0.6, 'meeting': 0.2}
likelihoods_not = {'free': 0.4, 'money': 0.2, 'meeting': 0.6}

# New email: free=1, money=1, meeting=0
email = {'free': 1, 'money': 1, 'meeting': 0}

# Compute unnormalized posteriors
log_p_spam = np.log(p_spam)
log_p_not = np.log(p_not_spam)

for feature, present in email.items():
    if present:
        log_p_spam += np.log(likelihoods_spam[feature])
        log_p_not += np.log(likelihoods_not[feature])
    else:
        log_p_spam += np.log(1 - likelihoods_spam[feature])
        log_p_not += np.log(1 - likelihoods_not[feature])

# Normalize (in log space)
max_log = max(log_p_spam, log_p_not)
p_spam_given_x = np.exp(log_p_spam - max_log) / (np.exp(log_p_spam - max_log) + np.exp(log_p_not - max_log))

print(f"P(spam | email) = {p_spam_given_x:.4f} = {p_spam_given_x*100:.1f}%")
print(f"Prediction: {'SPAM' if p_spam_given_x > 0.5 else 'NOT SPAM'}")
```

---

## 5. Implementation from Scratch

```python
import numpy as np

class GaussianNBFromScratch:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        
        # Compute mean, variance, and prior for each class
        self.means = np.zeros((self.n_classes, self.n_features))
        self.vars = np.zeros((self.n_classes, self.n_features))
        self.priors = np.zeros(self.n_classes)
        
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.means[i] = X_c.mean(axis=0)
            self.vars[i] = X_c.var(axis=0) + 1e-9  # Add small value for stability
            self.priors[i] = len(X_c) / len(X)
        
        return self
    
    def _log_gaussian(self, x, mean, var):
        """Log of Gaussian PDF."""
        return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for i in range(self.n_classes):
                log_prior = np.log(self.priors[i])
                log_likelihood = np.sum(self._log_gaussian(x, self.means[i], self.vars[i]))
                posteriors.append(log_prior + log_likelihood)
            predictions.append(self.classes[np.argmax(posteriors)])
        return np.array(predictions)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# Test
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

nb = GaussianNBFromScratch()
nb.fit(X_train, y_train)
print(f"From-scratch accuracy: {nb.score(X_test, y_test):.2%}")

# Compare with sklearn
from sklearn.naive_bayes import GaussianNB
sklearn_nb = GaussianNB()
sklearn_nb.fit(X_train, y_train)
print(f"sklearn accuracy:      {sklearn_nb.score(X_test, y_test):.2%}")
```

---

## 6. scikit-learn Implementation

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Gaussian NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"Gaussian NB accuracy: {gnb.score(X_test, y_test):.4f}")
print(classification_report(y_test, gnb.predict(X_test), target_names=data.target_names))
```

---

## 7. Text Classification with Naive Bayes

Naive Bayes is the **gold standard** for text classification:

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

# Sample text data
texts = [
    "free money now click here", "earn cash fast easy money",
    "buy cheap products discount", "limited offer free gift",
    "meeting tomorrow at 3pm", "project deadline next week",
    "lunch with team today", "quarterly report due friday",
    "budget review scheduled", "conference call at noon",
    "win big prize lottery", "cheap viagra pills now",
    "team building event", "code review complete",
]
labels = [1,1,1,1, 0,0,0,0, 0,0, 1,1, 0,0]  # 1=spam, 0=not spam

# Build pipeline
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', MultinomialNB(alpha=1.0))
])

pipeline.fit(texts, labels)

# Test on new emails
test_emails = [
    "free money offer today",
    "team meeting at 2pm",
    "win lottery cash prize"
]

for email in test_emails:
    pred = pipeline.predict([email])[0]
    proba = pipeline.predict_proba([email])[0]
    label = 'SPAM' if pred == 1 else 'HAM'
    print(f'"{email}" → {label} (confidence: {max(proba):.1%})')
```

---

## 8. Practical Guide

### ✅ When to Use Naive Bayes
- Text classification (spam, sentiment, topic)
- Real-time prediction (very fast)
- Multi-class problems
- Small training datasets
- As a baseline model

### ❌ When NOT to Use
- Feature correlations are important
- Need high accuracy (use ensembles instead)
- Complex decision boundaries

### Advantages
- Extremely fast training and prediction
- Works well with small data
- Handles high-dimensional data
- Easy to implement and understand

---

## 9. Project Ideas & What's Next

### Project Ideas
- 🟢 **News Category Classifier** — Classify news articles into categories
- 🟡 **Sentiment Analyzer** — Build a Twitter/review sentiment classifier
- 🔴 **Email Spam Filter** — Complete end-to-end spam detection system

### What's Next
| Next | Why |
|------|-----|
| [Bagging & Random Forest](../03-Ensemble-Methods/01-Bagging-And-Random-Forest.md) | Combine multiple models |
| [NLP](../08-Specialized-Domains/01-NLP.md) | Deep dive into text processing |

---

[← SVM](./06-SVM.md) | [Back to Index](../README.md) | [Next: Bagging And Random Forest →](../03-Ensemble-Methods/01-Bagging-And-Random-Forest.md)
