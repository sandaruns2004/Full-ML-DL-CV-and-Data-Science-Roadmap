# 🎲 Naive Bayes

> **Prerequisites:** Probability Theory (Bayes' Theorem)
>
> **Difficulty:** ⭐⭐⭐☆☆
>
> **Estimated Reading Time:** 20 minutes

---

## 📋 Table of Contents
1. [What Problem Does This Solve?](#1-what-problem-does-this-solve)
2. [Intuition](#2-intuition)
3. [Mathematics](#3-mathematics)
4. [Visual Explanation](#4-visual-explanation)
5. [Algorithm Workflow](#5-algorithm-workflow)
6. [From Scratch Implementation](#6-from-scratch-implementation)
7. [NumPy Implementation](#7-numpy-implementation)
8. [Scikit-Learn Implementation](#8-scikit-learn-implementation)
9. [Hyperparameter Deep Dive](#9-hyperparameter-deep-dive)
10. [Visualization Lab](#10-visualization-lab)
11. [Failure Cases](#11-failure-cases)
12. [Industry Applications](#12-industry-applications)
13. [Interview Preparation](#13-interview-preparation)
14. [Exercises](#14-exercises)
15. [Further Reading](#15-further-reading)

---

# 1. What Problem Does This Solve?

### 🟢 Beginner
You receive an email that says: *"WINNER! Click here for free money!"* How does your email provider know it's spam? It looks at history. It says, "In the past, the word 'WINNER' appeared in spam emails 99% of the time, and 'free money' appeared 95% of the time." By combining these probabilities, it calculates the overall chance that this specific email is spam. This is Naive Bayes.

### 🟡 Intermediate
Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features. It is incredibly fast, highly scalable, and requires relatively little training data to estimate the necessary parameters. It is the gold standard baseline for Text Classification and NLP baseline models.

### 🔴 Advanced
While discriminative models (like Logistic Regression) attempt to learn the boundary between classes $P(Y|X)$ directly, Naive Bayes is a **Generative** model. It learns the underlying distribution of the data $P(X|Y)$ and the prior probability $P(Y)$, and uses them to calculate the posterior probability. The "naive" assumption—that all features are mathematically independent—is almost always false in the real world, yet the algorithm performs phenomenally well anyway.

---

# 2. Intuition

### The Medical Test
Imagine a disease affects 1% of the population. There is a test for it that is 99% accurate. You take the test, and it comes back positive.
Your intuition says you have a 99% chance of having the disease.
Bayes' Theorem proves your intuition wrong. Because the disease is so rare (1%), the vast majority of positive tests are actually false positives from the 99% of healthy people. Your actual chance of having the disease is only about 50%!

Naive Bayes extends this logic to multiple features. It calculates the probability of a class given the evidence, by multiplying the individual probabilities of every single piece of evidence.

---

# 3. Mathematics

### 3.1 Bayes' Theorem
$$ P(y | X) = \frac{P(X | y) \cdot P(y)}{P(X)} $$
- $P(y | X)$: **Posterior** (Probability that the email is Spam, given the words).
- $P(X | y)$: **Likelihood** (Probability of seeing these words, if we already know it's Spam).
- $P(y)$: **Prior** (Overall probability of any email being Spam).
- $P(X)$: **Evidence** (Overall probability of seeing these words).

### 3.2 The "Naive" Assumption
If $X$ is a vector of $n$ features (words), calculating the exact joint probability $P(x_1, x_2, \dots, x_n | y)$ is computationally impossible.
So, we "naively" assume that the presence of every word is completely independent of every other word. We just multiply them:
$$ P(x_1, x_2, \dots, x_n | y) \approx P(x_1|y) \times P(x_2|y) \times \dots \times P(x_n|y) $$

### 3.3 The Final Equation
Because $P(X)$ is the same for all classes, we can drop the denominator and just find the class that maximizes the numerator:
$$ \hat{y} = \arg\max_y P(y) \prod_{i=1}^n P(x_i | y) $$

---

# 4. Visual Explanation

```mermaid
flowchart TD
    subgraph Training Phase
        Data[Training Emails] --> Prior[Calculate P_Spam and P_Ham]
        Data --> Freq[Count Word Frequencies per Class]
        Freq --> Like[Calculate P_word | Spam and P_word | Ham]
    end
    
    subgraph Prediction Phase
        New["New Email: 'Free Money'"] --> M1["P(Spam) * P('Free'|Spam) * P('Money'|Spam)"]
        New --> M2["P(Ham) * P('Free'|Ham) * P('Money'|Ham)"]
        
        M1 --> Compare{"Which is larger?"}
        M2 --> Compare
        
        Compare --> Final[🏆 Output Class]
    end
```

---

# 5. Algorithm Workflow

1. **Calculate Priors**: Count what percentage of the training data belongs to each class.
2. **Calculate Likelihoods**: For every feature, calculate its probability distribution within each class. (e.g., How often does "Free" appear in Spam vs Ham?)
3. **Log Trick**: When predicting, multiplying dozens of tiny probabilities (like $0.001 \times 0.001$) will cause computer memory to round down to $0$ (Underflow). Instead, we take the `log()` of the probabilities and **add** them together!
4. **Predict**: For a new sample, sum the log-probabilities for each class. The class with the highest sum wins.

---

# 6. From Scratch Implementation

*Implementing Gaussian Naive Bayes for continuous numerical features.*

```python
import numpy as np

class GaussianNaiveBayesScratch:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Calculate mean, variance, and prior for each class
        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(n_samples)
            
    def _gaussian_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        # Gaussian PDF formula
        numerator = np.exp(-((x - mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
        
    def _predict_one(self, x):
        posteriors = []
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            
            # Sum of logs instead of product of probabilities
            posterior = np.sum(np.log(self._gaussian_density(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)
            
        return self.classes[np.argmax(posteriors)]
        
    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])
```

---

# 7. NumPy Implementation

*(See section 6. The above implementation relies heavily on NumPy's vectorized mathematical operations).*

---

# 8. Scikit-Learn Implementation

*Using MultinomialNB, which is the industry standard for Text/Word-Count data.*

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 1. Text Data
texts = [
    "free money now", "win a free prize", "urgent reply needed", # Spam (1)
    "meeting at noon", "lunch tomorrow?", "please review this document" # Ham (0)
]
labels = [1, 1, 1, 0, 0, 0]

# 2. Convert words to numerical counts (Bag of Words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# 3. Train Model
model = MultinomialNB(alpha=1.0) # alpha is Laplace smoothing
model.fit(X, labels)

# 4. Predict new email
new_email = vectorizer.transform(["urgent meeting tomorrow"])
pred = model.predict(new_email)
print("Prediction:", "Spam" if pred[0] == 1 else "Ham")
```

---

# 9. Hyperparameter Deep Dive

- **`alpha` (Laplace Smoothing)**: The most critical hyperparameter. 
  - *The Problem*: If a new email contains the word "Bitcoin", and "Bitcoin" never appeared in your training data, $P(\text{Bitcoin}|\text{Spam}) = 0$. Because Naive Bayes multiplies everything, the entire probability becomes $0$.
  - *The Solution*: Laplace smoothing adds a small imaginary count (usually `alpha=1.0`) to every word in the dictionary so that the probability is never exactly $0$.
- **Model Type**:
  - `MultinomialNB`: Used for discrete counts (e.g., word frequencies).
  - `GaussianNB`: Used for continuous, normally distributed data (e.g., height, weight).
  - `BernoulliNB`: Used for binary/boolean features (e.g., word is present vs absent).

---

# 10. Visualization Lab

*Visualizing Gaussian Naive Bayes decision boundaries on continuous data.*

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from mlxtend.plotting import plot_decision_regions

# Generate distinct blobs of data
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

model = GaussianNB().fit(X, y)

plt.figure(figsize=(8, 6))
plot_decision_regions(X, y, clf=model)
plt.title("Gaussian Naive Bayes Decision Boundary")
plt.show()
```

---

# 11. Failure Cases

### Correlated Features
The "Naive" assumption is that features are independent. If you have two highly correlated features (like "City" and "Zip Code"), Naive Bayes will double-count their evidence. If it strongly points to Class A, the model will become irrationally overconfident in Class A.

### Continuous Data not Normally Distributed
`GaussianNB` strictly assumes that continuous features follow a perfect bell curve (Normal Distribution). If your data is heavily skewed or bimodal, the probability calculations will be wildly inaccurate.

---

# 12. Industry Applications

- **Spam Filtering**: The absolute classic use-case.
- **Sentiment Analysis**: Classifying Tweets or Reviews as Positive or Negative. Naive Bayes is often used as a fast, highly scalable baseline before deploying heavy Transformers like BERT.
- **Recommendation Systems**: Used in collaborative filtering to calculate the probability that a user will like an item.

---

# 13. Interview Preparation

### Beginner Questions
**Q: Why is it called "Naive"?**
> A: Because it makes the mathematically "naive" assumption that every single feature is completely independent of every other feature. In reality, words like "San" and "Francisco" are highly dependent.

### Intermediate Questions
**Q: What is the Zero-Frequency problem and how is it solved?**
> A: If a feature value did not occur in the training data for a specific class, its probability is 0, which wipes out the entire calculation. It is solved using Laplace Smoothing (adding 1 to all counts).

### Advanced Questions
**Q: Why do we use the sum of Log Probabilities instead of multiplying the raw probabilities?**
> A: Multiplying many probabilities (numbers between 0 and 1) results in extremely small numbers. A computer's floating-point architecture will quickly round these to exactly $0.0$ (Underflow). By taking the log, we transform tiny decimals into negative numbers and sum them, preserving the math perfectly.

---

# 14. Exercises

### Easy
Read the `MultinomialNB` Scikit-Learn code block above. Run it yourself. What happens if you change `alpha=0.0`?

### Medium
Implement Laplace smoothing in the from-scratch algorithm. 

### Hard
Write a script that scrapes 100 Wikipedia articles from two different categories (e.g., "Machine Learning" and "Ancient History"). Use `CountVectorizer` and `MultinomialNB` to build a classifier that predicts the category of a newly pasted paragraph.

---

# 15. Further Reading

- *Speech and Language Processing (Jurafsky & Martin)* - Chapter 4 (Naive Bayes and Sentiment Classification)
- Scikit-Learn Documentation: `sklearn.naive_bayes`
