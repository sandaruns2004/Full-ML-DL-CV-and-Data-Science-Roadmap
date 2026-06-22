# 🎲 Probability & Statistics — The Mathematical Engine of Data Science

> **Prerequisites**: Mathematical Foundations (Linear Algebra, Calculus) | **Difficulty**: ⭐⭐☆☆☆ Beginner-Intermediate

---

## 📋 Table of Contents

1. [Why Probability Matters for ML/DS](#1-why-probability-matters-for-mlds)
2. [Sample Spaces, Events & Axioms](#2-sample-spaces-events--axioms)
3. [Conditional Probability & Independence](#3-conditional-probability--independence)
4. [Bayes' Theorem — The Foundation](#4-bayes-theorem--the-foundation)
5. [Random Variables & Expectation](#5-random-variables--expectation)
6. [Variance, Covariance & Moments](#6-variance-covariance--moments)
7. [Common Probability Distributions](#7-common-probability-distributions)
8. [Joint, Marginal & Conditional Distributions](#8-joint-marginal--conditional-distributions)
9. [Law of Large Numbers (LLN)](#9-law-of-large-numbers-lln)
10. [Central Limit Theorem (CLT)](#10-central-limit-theorem-clt)
11. [Maximum Likelihood Estimation (MLE)](#11-maximum-likelihood-estimation-mle)
12. [Maximum A Posteriori (MAP) Estimation](#12-maximum-a-posteriori-map-estimation)
13. [Information Theory Essentials](#13-information-theory-essentials)
14. [Project Ideas & What's Next](#14-project-ideas--whats-next)

---

## 1. Why Probability Matters for ML/DS

Probability is not an optional add-on — it is the **language** in which machine learning is written.

| ML Concept | Underlying Probability |
|------------|----------------------|
| Logistic Regression | Models $P(Y=1 \mid X)$ directly |
| Naive Bayes | Direct application of Bayes' theorem |
| Neural Network outputs | Softmax produces a probability distribution |
| Loss functions | Cross-entropy = negative log-likelihood |
| GANs | Minimax game over probability distributions |
| Bayesian Optimization | Models uncertainty via posterior distributions |
| Dropout | Bernoulli random variable at each neuron |
| Data Augmentation | Sampling from transformed distributions |

**Everything in ML is probability.** If you don't understand probability, you're memorizing formulas instead of understanding *why* they work.

---

## 2. Sample Spaces, Events & Axioms

### The Setup

- **Experiment**: Any process with uncertain outcomes (rolling a die, training a model)
- **Sample Space** ($\Omega$): The set of ALL possible outcomes
  - Rolling a die: $\Omega = \{1, 2, 3, 4, 5, 6\}$
  - Flipping 2 coins: $\Omega = \{HH, HT, TH, TT\}$
- **Event** ($A$): A subset of $\Omega$ — any collection of outcomes
  - "Rolling an even number": $A = \{2, 4, 6\}$

### Kolmogorov's Axioms of Probability

Every probability function $P$ must satisfy three axioms:

$$\text{Axiom 1 (Non-negativity):} \quad P(A) \geq 0 \quad \forall A$$

$$\text{Axiom 2 (Normalization):} \quad P(\Omega) = 1$$

$$\text{Axiom 3 (Additivity):} \quad P(A \cup B) = P(A) + P(B) \quad \text{if } A \cap B = \emptyset$$

**Everything else** in probability is derived from these three axioms. For example:

$$P(A^c) = 1 - P(A) \quad \text{(complement rule — derived from Axioms 2 \& 3)}$$

$$P(A \cup B) = P(A) + P(B) - P(A \cap B) \quad \text{(inclusion-exclusion — derived from Axiom 3)}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2  # pip install matplotlib-venn

# Visualize set operations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Union
v1 = venn2(subsets=(30, 30, 10), set_labels=('A', 'B'), ax=axes[0])
axes[0].set_title('P(A ∪ B) = P(A) + P(B) - P(A∩B)', fontweight='bold')

# Intersection
v2 = venn2(subsets=(30, 30, 10), set_labels=('A', 'B'), ax=axes[1])
v2.get_patch_by_id('10').set_color('white')
v2.get_patch_by_id('01').set_color('white')
v2.get_patch_by_id('11').set_color('#FF6384')
axes[1].set_title('P(A ∩ B) — Intersection', fontweight='bold')

# Complement
v3 = venn2(subsets=(70, 0, 0), set_labels=('A', 'Aᶜ'), ax=axes[2])
axes[2].set_title('P(Aᶜ) = 1 - P(A)', fontweight='bold')

plt.tight_layout()
plt.savefig('probability_sets.png', dpi=150)
plt.show()
```

---

## 3. Conditional Probability & Independence

### Conditional Probability

The probability of $A$ **given that** $B$ has already occurred:

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Intuition**: We "shrink" our sample space from $\Omega$ to just $B$. Then we ask: what fraction of $B$ does $A$ cover?

**Example**: A deck of 52 cards. What's the probability of drawing a King, given that we drew a face card?
- $P(\text{King}) = \frac{4}{52}$
- $P(\text{Face card}) = \frac{12}{52}$ (J, Q, K × 4 suits)
- $P(\text{King} \mid \text{Face card}) = \frac{P(\text{King} \cap \text{Face})}{P(\text{Face})} = \frac{4/52}{12/52} = \frac{4}{12} = \frac{1}{3}$

### The Chain Rule (Product Rule)

From the definition of conditional probability:

$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

For $n$ events:

$$P(A_1 \cap A_2 \cap \dots \cap A_n) = P(A_1) \cdot P(A_2 \mid A_1) \cdot P(A_3 \mid A_1, A_2) \cdots$$

### Independence

Events $A$ and $B$ are **independent** if knowing $B$ occurred gives NO information about $A$:

$$P(A \mid B) = P(A) \quad \Longleftrightarrow \quad P(A \cap B) = P(A) \cdot P(B)$$

**Critical ML application**: Naive Bayes assumes all features are **conditionally independent** given the class label:
$$P(x_1, x_2, \dots, x_n \mid y) = \prod_{i=1}^{n} P(x_i \mid y)$$

This is almost never true in reality, but it works shockingly well in practice!

### Law of Total Probability

If $B_1, B_2, \dots, B_n$ form a **partition** of $\Omega$ (mutually exclusive, collectively exhaustive):

$$P(A) = \sum_{i=1}^{n} P(A \mid B_i) \cdot P(B_i)$$

**ML example**: The total error of a classifier equals the weighted sum of errors within each class.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate conditional probability with dice
np.random.seed(42)
n_rolls = 100_000

die1 = np.random.randint(1, 7, n_rolls)
die2 = np.random.randint(1, 7, n_rolls)
total = die1 + die2

# P(sum >= 10 | die1 = 6)
mask_die1_is_6 = die1 == 6
conditional_prob = np.mean(total[mask_die1_is_6] >= 10)
theoretical = 3 / 6  # die2 must be 4, 5, or 6

print(f"P(sum ≥ 10 | die1 = 6)")
print(f"  Simulated:   {conditional_prob:.4f}")
print(f"  Theoretical: {theoretical:.4f}")

# Visualize: P(sum = k | die1 = 6) vs P(sum = k)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Unconditional distribution of sum
sums = range(2, 13)
unconditional = [np.mean(total == s) for s in sums]
axes[0].bar(sums, unconditional, color='#36A2EB', edgecolor='white')
axes[0].set_title('P(Sum = k) — Unconditional', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Sum of two dice')
axes[0].set_ylabel('Probability')

# Conditional distribution given die1 = 6
conditional = [np.mean(total[mask_die1_is_6] == s) for s in sums]
axes[1].bar(sums, conditional, color='#FF6384', edgecolor='white')
axes[1].set_title('P(Sum = k | Die1 = 6) — Conditional', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Sum of two dice')
axes[1].set_ylabel('Probability')

plt.suptitle('Conditional vs Unconditional Probability', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('conditional_probability.png', dpi=150)
plt.show()
```

---

## 4. Bayes' Theorem — The Foundation

### The Formula

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

**In ML language**:

$$\underbrace{P(\theta \mid \text{data})}_{\text{Posterior}} = \frac{\overbrace{P(\text{data} \mid \theta)}^{\text{Likelihood}} \cdot \overbrace{P(\theta)}^{\text{Prior}}}{\underbrace{P(\text{data})}_{\text{Evidence}}}$$

- **Prior** $P(\theta)$: What you believed before seeing data
- **Likelihood** $P(\text{data} \mid \theta)$: How likely is this data if $\theta$ is true?
- **Posterior** $P(\theta \mid \text{data})$: Updated belief after seeing data
- **Evidence** $P(\text{data})$: Normalizing constant (often intractable)

### Derivation

From the product rule:
$$P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)$$

Dividing both sides by $P(B)$:
$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

### Classic Example: Medical Testing

A disease affects 1% of the population. A test has:
- **Sensitivity** (true positive rate): $P(\text{positive} \mid \text{disease}) = 0.99$
- **Specificity** (true negative rate): $P(\text{negative} \mid \text{no disease}) = 0.95$

You test positive. What's the probability you actually have the disease?

$$P(\text{disease} \mid +) = \frac{P(+ \mid \text{disease}) \cdot P(\text{disease})}{P(+)}$$

Using the Law of Total Probability for the denominator:
$$P(+) = P(+ \mid \text{disease}) \cdot P(\text{disease}) + P(+ \mid \text{no disease}) \cdot P(\text{no disease})$$
$$= 0.99 \times 0.01 + 0.05 \times 0.99 = 0.0099 + 0.0495 = 0.0594$$

$$P(\text{disease} \mid +) = \frac{0.99 \times 0.01}{0.0594} \approx 0.167$$

**Only 16.7%!** Even with a 99% accurate test, a positive result means there's only a ~17% chance you're actually sick. This is the **base rate fallacy** — and it's why Bayesian thinking is essential.

```python
import numpy as np
import matplotlib.pyplot as plt

def bayes_medical_test(prevalence, sensitivity, specificity):
    """Calculate posterior probability of disease given positive test."""
    p_pos_given_disease = sensitivity
    p_pos_given_no_disease = 1 - specificity
    p_disease = prevalence
    
    # Law of Total Probability
    p_positive = (p_pos_given_disease * p_disease + 
                  p_pos_given_no_disease * (1 - p_disease))
    
    # Bayes' Theorem
    p_disease_given_pos = (p_pos_given_disease * p_disease) / p_positive
    return p_disease_given_pos

# How posterior changes with disease prevalence
prevalences = np.linspace(0.001, 0.5, 200)
posteriors_95 = [bayes_medical_test(p, 0.99, 0.95) for p in prevalences]
posteriors_99 = [bayes_medical_test(p, 0.99, 0.99) for p in prevalences]
posteriors_999 = [bayes_medical_test(p, 0.99, 0.999) for p in prevalences]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Posterior vs Prevalence
axes[0].plot(prevalences * 100, posteriors_95, 'r-', lw=2.5, label='Specificity = 95%')
axes[0].plot(prevalences * 100, posteriors_99, 'b-', lw=2.5, label='Specificity = 99%')
axes[0].plot(prevalences * 100, posteriors_999, 'g-', lw=2.5, label='Specificity = 99.9%')
axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
axes[0].set_xlabel('Disease Prevalence (%)', fontsize=12)
axes[0].set_ylabel('P(Disease | Positive Test)', fontsize=12)
axes[0].set_title('Bayes\' Theorem: Medical Testing', fontweight='bold', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 50])

# Plot 2: Sequential Bayesian updating
# Start with prior, update with each positive test
prior = 0.01
posteriors = [prior]
for i in range(5):
    posterior = bayes_medical_test(posteriors[-1], 0.99, 0.95)
    posteriors.append(posterior)

axes[1].bar(range(len(posteriors)), posteriors, color=['#36A2EB'] + ['#FF6384']*5, 
            edgecolor='white', alpha=0.8)
axes[1].set_xlabel('Number of Positive Tests', fontsize=12)
axes[1].set_ylabel('P(Disease)', fontsize=12)
axes[1].set_title('Sequential Bayesian Updating\n(Each positive test updates the prior)', 
                   fontweight='bold', fontsize=13)
for i, v in enumerate(posteriors):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold', fontsize=10)
axes[1].set_xticks(range(len(posteriors)))
axes[1].set_xticklabels(['Prior'] + [f'Test {i+1}' for i in range(5)])

plt.tight_layout()
plt.savefig('bayes_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 5. Random Variables & Expectation

### Random Variables

A **random variable** $X$ is a function that maps outcomes in $\Omega$ to real numbers.

- **Discrete**: $X$ takes countable values (die roll, coin flip, word count)
  - Described by a **Probability Mass Function (PMF)**: $P(X = x)$
  
- **Continuous**: $X$ takes values in an interval (height, temperature, stock price)
  - Described by a **Probability Density Function (PDF)**: $f(x)$
  - $P(a \leq X \leq b) = \int_a^b f(x) \, dx$
  - ⚠️ For continuous variables, $P(X = x) = 0$ for any specific $x$!

### Expected Value (Mean)

The "center of mass" of a probability distribution — the **long-run average**.

**Discrete**: 
$$E[X] = \sum_{x} x \cdot P(X = x)$$

**Continuous**: 
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

### Properties of Expectation

$$E[aX + b] = aE[X] + b \quad \text{(linearity)}$$

$$E[X + Y] = E[X] + E[Y] \quad \text{(always true, even if dependent!)}$$

$$E[XY] = E[X] \cdot E[Y] \quad \text{(only if } X, Y \text{ are independent)}$$

$$E[g(X)] = \sum_{x} g(x) \cdot P(X = x) \quad \text{(Law of the Unconscious Statistician — LOTUS)}$$

### Example: Expected Value of a Fair Die

$$E[X] = \sum_{x=1}^{6} x \cdot \frac{1}{6} = \frac{1+2+3+4+5+6}{6} = 3.5$$

You can **never** roll 3.5, but over thousands of rolls, the average converges to 3.5.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Simulate the convergence of sample mean to expected value
n_rolls = 10000
rolls = np.random.randint(1, 7, n_rolls)
running_mean = np.cumsum(rolls) / np.arange(1, n_rolls + 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: Running average converging to E[X]
axes[0].plot(running_mean, color='#36A2EB', alpha=0.8, linewidth=1)
axes[0].axhline(y=3.5, color='red', linestyle='--', linewidth=2, label='E[X] = 3.5')
axes[0].fill_between(range(n_rolls), 3.5 - 0.1, 3.5 + 0.1, alpha=0.1, color='red')
axes[0].set_xlabel('Number of Rolls', fontsize=12)
axes[0].set_ylabel('Running Average', fontsize=12)
axes[0].set_title('Law of Large Numbers: Die Rolls\nRunning average → E[X] = 3.5', 
                   fontweight='bold', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log')

# Right: PMF of a fair die
x = np.arange(1, 7)
pmf = np.ones(6) / 6
axes[1].bar(x, pmf, color='#4CAF50', edgecolor='white', alpha=0.8, width=0.6)
axes[1].axhline(y=1/6, color='red', linestyle='--', alpha=0.5)
axes[1].set_xlabel('Outcome', fontsize=12)
axes[1].set_ylabel('P(X = x)', fontsize=12)
axes[1].set_title('PMF of Fair Die\nUniform Distribution', fontweight='bold', fontsize=13)
axes[1].set_xticks(x)
axes[1].set_ylim([0, 0.25])
for xi, pi in zip(x, pmf):
    axes[1].text(xi, pi + 0.005, f'{pi:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('expected_value.png', dpi=150)
plt.show()
```

---

## 6. Variance, Covariance & Moments

### Variance

Measures how spread out the distribution is — the **average squared deviation** from the mean.

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

**Proof of the shortcut formula**:
$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2$$

**Standard Deviation**: $\sigma = \sqrt{\text{Var}(X)}$ — in the same units as $X$.

### Properties of Variance

$$\text{Var}(aX + b) = a^2 \cdot \text{Var}(X) \quad \text{(scaling: } b \text{ disappears, } a \text{ squares)}$$

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

If $X, Y$ are **independent**: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$

### Covariance

Measures how two variables move **together**.

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

- $\text{Cov}(X, Y) > 0$: $X$ and $Y$ tend to increase together
- $\text{Cov}(X, Y) < 0$: When $X$ increases, $Y$ tends to decrease
- $\text{Cov}(X, Y) = 0$: No linear relationship (NOT necessarily independent!)

### Correlation (Pearson's $r$)

Normalized covariance — scale-invariant, $r \in [-1, 1]$:

$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

### Moments

The $k$-th moment of $X$ about the origin:
$$\mu_k' = E[X^k]$$

The $k$-th **central** moment:
$$\mu_k = E[(X - \mu)^k]$$

| Moment | Name | Interpretation |
|--------|------|---------------|
| $\mu_1'$ | Mean | Center of distribution |
| $\mu_2$ | Variance | Spread / dispersion |
| $\mu_3 / \sigma^3$ | Skewness | Asymmetry (left/right tail) |
| $\mu_4 / \sigma^4$ | Kurtosis | Tail heaviness / peakedness |

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Demonstrate covariance visually
n = 500
x = np.random.randn(n)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Positive covariance
y_pos = 2 * x + np.random.randn(n) * 0.5
cov_pos = np.cov(x, y_pos)[0, 1]
corr_pos = np.corrcoef(x, y_pos)[0, 1]
axes[0].scatter(x, y_pos, alpha=0.4, s=20, color='#4CAF50')
axes[0].set_title(f'Positive Covariance\nCov={cov_pos:.2f}, r={corr_pos:.2f}', 
                   fontweight='bold', fontsize=13)

# Near-zero covariance
y_zero = np.random.randn(n)
cov_zero = np.cov(x, y_zero)[0, 1]
corr_zero = np.corrcoef(x, y_zero)[0, 1]
axes[1].scatter(x, y_zero, alpha=0.4, s=20, color='#FFCE56')
axes[1].set_title(f'Near-Zero Covariance\nCov={cov_zero:.2f}, r={corr_zero:.2f}', 
                   fontweight='bold', fontsize=13)

# Negative covariance
y_neg = -1.5 * x + np.random.randn(n) * 0.5
cov_neg = np.cov(x, y_neg)[0, 1]
corr_neg = np.corrcoef(x, y_neg)[0, 1]
axes[2].scatter(x, y_neg, alpha=0.4, s=20, color='#FF6384')
axes[2].set_title(f'Negative Covariance\nCov={cov_neg:.2f}, r={corr_neg:.2f}', 
                   fontweight='bold', fontsize=13)

for ax in axes:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)

plt.suptitle('Covariance & Correlation Visualized', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('covariance.png', dpi=150)
plt.show()
```

---

## 7. Common Probability Distributions

### Discrete Distributions

#### Bernoulli Distribution
A single trial with probability $p$ of success.
$$P(X = 1) = p, \quad P(X = 0) = 1-p$$
$$E[X] = p, \quad \text{Var}(X) = p(1-p)$$

#### Binomial Distribution
Number of successes in $n$ independent Bernoulli trials.
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \dots, n$$
$$E[X] = np, \quad \text{Var}(X) = np(1-p)$$

#### Poisson Distribution
Number of events in a fixed interval when events occur at a constant average rate $\lambda$.
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \dots$$
$$E[X] = \lambda, \quad \text{Var}(X) = \lambda$$

### Continuous Distributions

#### Uniform Distribution
Equal probability over an interval $[a, b]$.
$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$
$$E[X] = \frac{a+b}{2}, \quad \text{Var}(X) = \frac{(b-a)^2}{12}$$

#### Normal (Gaussian) Distribution — The King of Distributions
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
$$E[X] = \mu, \quad \text{Var}(X) = \sigma^2$$

**The 68-95-99.7 Rule**: $68\%$ of data falls within $1\sigma$ of $\mu$, $95\%$ within $2\sigma$, $99.7\%$ within $3\sigma$.

#### Exponential Distribution
Time between events in a Poisson process. Memoryless!
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$
$$E[X] = \frac{1}{\lambda}, \quad \text{Var}(X) = \frac{1}{\lambda^2}$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Bernoulli
p = 0.7
x_bern = [0, 1]
pmf_bern = [1-p, p]
axes[0, 0].bar(x_bern, pmf_bern, color=['#FF6384', '#4CAF50'], edgecolor='white', width=0.4)
axes[0, 0].set_title(f'Bernoulli(p={p})\nE[X]={p}, Var={p*(1-p):.2f}', fontweight='bold', fontsize=12)
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['Failure (0)', 'Success (1)'])
axes[0, 0].set_ylabel('P(X=x)')

# 2. Binomial
n_binom, p_binom = 20, 0.3
x_binom = np.arange(0, n_binom + 1)
pmf_binom = stats.binom.pmf(x_binom, n_binom, p_binom)
axes[0, 1].bar(x_binom, pmf_binom, color='#36A2EB', edgecolor='white', alpha=0.8)
axes[0, 1].axvline(n_binom * p_binom, color='red', linestyle='--', lw=2, label=f'E[X]={n_binom*p_binom}')
axes[0, 1].set_title(f'Binomial(n={n_binom}, p={p_binom})', fontweight='bold', fontsize=12)
axes[0, 1].legend()
axes[0, 1].set_xlabel('k (successes)')

# 3. Poisson
lambdas = [1, 4, 10]
x_pois = np.arange(0, 20)
colors = ['#FF6384', '#36A2EB', '#4CAF50']
for lam, color in zip(lambdas, colors):
    pmf_pois = stats.poisson.pmf(x_pois, lam)
    axes[0, 2].plot(x_pois, pmf_pois, 'o-', color=color, lw=2, markersize=5, label=f'λ={lam}')
axes[0, 2].set_title('Poisson Distribution', fontweight='bold', fontsize=12)
axes[0, 2].legend(fontsize=10)
axes[0, 2].set_xlabel('k (events)')

# 4. Uniform
a, b = 2, 8
x_unif = np.linspace(0, 10, 200)
pdf_unif = stats.uniform.pdf(x_unif, loc=a, scale=b-a)
axes[1, 0].fill_between(x_unif, pdf_unif, alpha=0.3, color='#9B59B6')
axes[1, 0].plot(x_unif, pdf_unif, color='#9B59B6', lw=2.5)
axes[1, 0].set_title(f'Uniform[{a}, {b}]\nE[X]={(a+b)/2}, Var={(b-a)**2/12:.2f}', fontweight='bold', fontsize=12)
axes[1, 0].set_ylabel('f(x)')

# 5. Normal (Gaussian) — The King
x_norm = np.linspace(-5, 5, 300)
for mu, sigma, color, label in [(0, 1, '#36A2EB', 'μ=0, σ=1'),
                                  (0, 2, '#FF6384', 'μ=0, σ=2'),
                                  (2, 0.5, '#4CAF50', 'μ=2, σ=0.5')]:
    pdf_norm = stats.norm.pdf(x_norm, mu, sigma)
    axes[1, 1].plot(x_norm, pdf_norm, lw=2.5, color=color, label=label)
    axes[1, 1].fill_between(x_norm, pdf_norm, alpha=0.1, color=color)
axes[1, 1].set_title('Normal (Gaussian) Distribution', fontweight='bold', fontsize=12)
axes[1, 1].legend(fontsize=9)
axes[1, 1].set_ylabel('f(x)')

# 6. Exponential
x_exp = np.linspace(0, 5, 200)
for lam, color in [(0.5, '#FF6384'), (1, '#36A2EB'), (2, '#4CAF50')]:
    pdf_exp = stats.expon.pdf(x_exp, scale=1/lam)
    axes[1, 2].plot(x_exp, pdf_exp, lw=2.5, color=color, label=f'λ={lam}')
    axes[1, 2].fill_between(x_exp, pdf_exp, alpha=0.1, color=color)
axes[1, 2].set_title('Exponential Distribution', fontweight='bold', fontsize=12)
axes[1, 2].legend(fontsize=10)
axes[1, 2].set_ylabel('f(x)')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

plt.suptitle('Common Probability Distributions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('distributions_gallery.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. Joint, Marginal & Conditional Distributions

### Joint Distribution

For two random variables $X$ and $Y$, the **joint distribution** describes their simultaneous behavior.

**Discrete**: $P(X = x, Y = y) = P(X = x \text{ AND } Y = y)$

**Continuous**: $f_{X,Y}(x, y)$ where $P(a \leq X \leq b, c \leq Y \leq d) = \int_a^b \int_c^d f_{X,Y}(x,y) \, dy \, dx$

### Marginal Distribution

"Forget" about one variable by summing/integrating it out:

$$P(X = x) = \sum_y P(X = x, Y = y) \quad \text{(discrete)}$$
$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dy \quad \text{(continuous)}$$

### Conditional Distribution

$$f_{Y|X}(y \mid x) = \frac{f_{X,Y}(x, y)}{f_X(x)}$$

**This is just Bayes' theorem for distributions!**

### Multivariate Gaussian

The most important joint distribution in ML. For a $d$-dimensional vector $\mathbf{x}$:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

Where:
- $\boldsymbol{\mu}$ is the $d$-dimensional mean vector
- $\boldsymbol{\Sigma}$ is the $d \times d$ covariance matrix
- $|\boldsymbol{\Sigma}|$ is the determinant of the covariance matrix

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Joint distribution as contour plot
mean = [0, 0]
cov_matrices = [
    [[1, 0], [0, 1]],       # Independent
    [[1, 0.8], [0.8, 1]],   # Positive correlation
    [[1, -0.8], [-0.8, 1]]  # Negative correlation
]
titles = ['Independent (ρ=0)', 'Positive Corr (ρ=0.8)', 'Negative Corr (ρ=-0.8)']
colors = ['#36A2EB', '#4CAF50', '#FF6384']

for ax, cov, title, color in zip(axes, cov_matrices, titles, colors):
    # Generate samples
    samples = np.random.multivariate_normal(mean, cov, 500)
    
    # Create contour
    x_grid = np.linspace(-4, 4, 100)
    y_grid = np.linspace(-4, 4, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mean, cov)
    Z = rv.pdf(pos)
    
    ax.contourf(X, Y, Z, levels=15, cmap='Blues', alpha=0.6)
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=8, color=color)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_xlabel('X₁')
    ax.set_ylabel('X₂')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('Bivariate Gaussian Distributions', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('joint_distributions.png', dpi=150)
plt.show()
```

---

## 9. Law of Large Numbers (LLN)

### Statement

As the number of independent, identically distributed (i.i.d.) samples increases, their sample mean converges to the true population mean.

**Weak LLN**: For any $\epsilon > 0$:
$$\lim_{n \to \infty} P\left(\left|\bar{X}_n - \mu\right| > \epsilon\right) = 0$$

**Strong LLN**:
$$P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$$

### Why It Matters for ML

- **Training with mini-batches**: The gradient computed on a mini-batch is a sample mean of individual gradients. LLN guarantees that larger batches give more accurate gradient estimates.
- **Cross-validation**: Averaging across $k$ folds gives a more reliable estimate of model performance.
- **Monte Carlo methods**: We can approximate expectations by averaging samples.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
distributions = [
    ('Normal(μ=5, σ=2)', lambda n: np.random.normal(5, 2, n), 5),
    ('Exponential(λ=0.5)', lambda n: np.random.exponential(2, n), 2),
    ('Bernoulli(p=0.3)', lambda n: np.random.binomial(1, 0.3, n), 0.3)
]

for ax, (name, sampler, true_mean) in zip(axes, distributions):
    for trial in range(10):  # 10 independent experiments
        n_max = 5000
        samples = sampler(n_max)
        running_mean = np.cumsum(samples) / np.arange(1, n_max + 1)
        ax.plot(running_mean, alpha=0.4, linewidth=0.8)
    
    ax.axhline(y=true_mean, color='red', linestyle='--', linewidth=2.5, label=f'True μ = {true_mean}')
    ax.set_title(f'LLN: {name}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Number of Samples (n)')
    ax.set_ylabel('Sample Mean')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

plt.suptitle('Law of Large Numbers — Sample Mean → True Mean', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('law_of_large_numbers.png', dpi=150)
plt.show()
```

---

## 10. Central Limit Theorem (CLT)

### Statement

The most important theorem in statistics. Regardless of the original distribution's shape, the **distribution of sample means** approaches a Normal distribution as $n$ increases.

If $X_1, X_2, \dots, X_n$ are i.i.d. with mean $\mu$ and variance $\sigma^2$, then:

$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{as } n \to \infty$$

Or equivalently, after standardizing:

$$Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

### Why It's Revolutionary

- The original data can follow **ANY** distribution (exponential, uniform, Poisson, bimodal — literally anything with finite variance)
- The distribution of sample means will still be approximately Normal for large enough $n$
- This is why confidence intervals and hypothesis tests work!

### Visualization: CLT in Action

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Original distributions (intentionally non-normal)
distributions = {
    'Exponential': lambda n: np.random.exponential(2, n),
    'Uniform': lambda n: np.random.uniform(0, 10, n),
    'Bimodal': lambda n: np.concatenate([np.random.normal(-3, 0.8, n//2), 
                                          np.random.normal(3, 0.8, n - n//2)]),
}

sample_sizes = [1, 2, 5, 30]
n_simulations = 10000

fig, axes = plt.subplots(len(distributions), len(sample_sizes), figsize=(20, 14))

for row, (dist_name, sampler) in enumerate(distributions.items()):
    for col, n in enumerate(sample_sizes):
        # Generate n_simulations sample means, each from n observations
        sample_means = []
        for _ in range(n_simulations):
            sample = sampler(n)
            sample_means.append(np.mean(sample))
        sample_means = np.array(sample_means)
        
        ax = axes[row, col]
        ax.hist(sample_means, bins=50, density=True, alpha=0.7, color='#36A2EB', edgecolor='white')
        
        # Overlay the theoretical normal distribution
        x_range = np.linspace(sample_means.min(), sample_means.max(), 200)
        normal_pdf = stats.norm.pdf(x_range, np.mean(sample_means), np.std(sample_means))
        ax.plot(x_range, normal_pdf, 'r-', linewidth=2.5, label='Normal fit')
        
        if row == 0:
            ax.set_title(f'n = {n}', fontsize=14, fontweight='bold')
        if col == 0:
            ax.set_ylabel(dist_name, fontsize=13, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

plt.suptitle('Central Limit Theorem\nDistribution of Sample Means Becomes Normal as n Increases', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('central_limit_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 11. Maximum Likelihood Estimation (MLE)

### The Core Idea

Given observed data $\mathbf{x} = (x_1, x_2, \dots, x_n)$ and a parametric model $f(x; \theta)$, find the parameter $\hat{\theta}$ that makes the observed data **most likely**.

### The Likelihood Function

$$L(\theta) = P(\text{data} \mid \theta) = \prod_{i=1}^{n} f(x_i; \theta)$$

In practice, we maximize the **log-likelihood** (products become sums, numerically stable):

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^{n} \log f(x_i; \theta)$$

### MLE for the Normal Distribution (Derivation)

Given data $x_1, \dots, x_n$ from $\mathcal{N}(\mu, \sigma^2)$:

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**Finding $\hat{\mu}$**: Take derivative w.r.t. $\mu$, set to zero:

$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$$
$$\Rightarrow \hat{\mu}_{MLE} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}$$

**Finding $\hat{\sigma}^2$**: Take derivative w.r.t. $\sigma^2$, set to zero:

$$\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i - \bar{x})^2 = 0$$
$$\Rightarrow \hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

Note: The MLE for variance divides by $n$ (biased!), not $n-1$ (unbiased).

### Connection to Cross-Entropy Loss

In classification, we model $P(y \mid x; \theta)$ with a neural network. Maximizing the log-likelihood is **identical** to minimizing the cross-entropy loss:

$$\hat{\theta}_{MLE} = \arg\max_\theta \sum_{i=1}^n \log P(y_i \mid x_i; \theta)$$
$$= \arg\min_\theta \left[-\sum_{i=1}^n \log P(y_i \mid x_i; \theta)\right] = \arg\min_\theta \text{CrossEntropy}$$

**Every time you train a neural network with cross-entropy loss, you are performing MLE!**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Generate data from a known Normal distribution
true_mu, true_sigma = 5.0, 2.0
data = np.random.normal(true_mu, true_sigma, 50)

# Compute log-likelihood for different values of mu
mu_range = np.linspace(2, 8, 200)
log_likelihoods = []
for mu in mu_range:
    ll = np.sum(stats.norm.logpdf(data, loc=mu, scale=true_sigma))
    log_likelihoods.append(ll)

mle_mu = mu_range[np.argmax(log_likelihoods)]
sample_mean = np.mean(data)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Log-Likelihood surface
axes[0].plot(mu_range, log_likelihoods, 'b-', linewidth=2.5)
axes[0].axvline(mle_mu, color='red', linestyle='--', linewidth=2, label=f'MLE μ̂ = {mle_mu:.2f}')
axes[0].axvline(sample_mean, color='green', linestyle=':', linewidth=2, label=f'Sample mean = {sample_mean:.2f}')
axes[0].axvline(true_mu, color='orange', linestyle='-', linewidth=2, alpha=0.5, label=f'True μ = {true_mu}')
axes[0].set_xlabel('μ', fontsize=13)
axes[0].set_ylabel('Log-Likelihood ℓ(μ)', fontsize=13)
axes[0].set_title('MLE: Finding μ that Maximizes Log-Likelihood', fontweight='bold', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot 2: Data with MLE fit
x_plot = np.linspace(data.min() - 2, data.max() + 2, 200)
axes[1].hist(data, bins=15, density=True, alpha=0.5, color='#36A2EB', edgecolor='white', label='Data')
axes[1].plot(x_plot, stats.norm.pdf(x_plot, mle_mu, true_sigma), 'r-', lw=2.5, label=f'MLE fit: N({mle_mu:.2f}, {true_sigma}²)')
axes[1].plot(x_plot, stats.norm.pdf(x_plot, true_mu, true_sigma), 'g--', lw=2, alpha=0.7, label=f'True: N({true_mu}, {true_sigma}²)')
axes[1].set_title('MLE Fit vs True Distribution', fontweight='bold', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('Maximum Likelihood Estimation', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('mle.png', dpi=150)
plt.show()
```

---

## 12. Maximum A Posteriori (MAP) Estimation

### MLE vs MAP

MLE treats $\theta$ as a fixed unknown constant and finds the most likely value.
MAP treats $\theta$ as a **random variable** with a prior distribution $P(\theta)$ and finds the mode of the posterior.

$$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta \mid \text{data}) = \arg\max_\theta \left[ \log P(\text{data} \mid \theta) + \log P(\theta) \right]$$

The prior acts as a **regularizer**:
- **Gaussian prior** $P(\theta) \sim \mathcal{N}(0, \sigma^2)$ → $\log P(\theta) = -\frac{\theta^2}{2\sigma^2} + C$ → **L2 Regularization** (Ridge)
- **Laplace prior** $P(\theta) \sim \text{Laplace}(0, b)$ → $\log P(\theta) = -\frac{|\theta|}{b} + C$ → **L1 Regularization** (Lasso)

**This is the Bayesian interpretation of regularization!** L2 regularization is identical to performing MAP estimation with a Gaussian prior on the weights.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Show that MAP with Gaussian prior = MLE + L2 regularization
theta_range = np.linspace(-5, 10, 300)

# Observed data (few points → prior matters a lot)
data = np.array([3.5, 4.2, 5.1])

# Log-likelihood (Normal with known sigma=1)
log_lik = np.array([np.sum(stats.norm.logpdf(data, loc=t, scale=1)) for t in theta_range])

# Priors
gaussian_prior = stats.norm.logpdf(theta_range, loc=0, scale=2)  # N(0, 4)
laplace_prior = stats.laplace.logpdf(theta_range, loc=0, scale=1)

# Posteriors (unnormalized log)
log_post_gaussian = log_lik + gaussian_prior
log_post_laplace = log_lik + laplace_prior

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Normalize for visualization
def normalize(arr):
    arr = arr - arr.max()
    return np.exp(arr) / np.exp(arr).sum()

# Plot 1: Likelihood
axes[0].plot(theta_range, normalize(log_lik), 'b-', lw=2.5, label='Likelihood')
axes[0].axvline(np.mean(data), color='blue', linestyle='--', alpha=0.5)
axes[0].set_title('Likelihood P(data|θ)\nMLE = sample mean', fontweight='bold', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].set_xlabel('θ')

# Plot 2: MAP with Gaussian prior (= L2 regularization)
axes[1].plot(theta_range, normalize(log_lik), 'b-', lw=1.5, alpha=0.5, label='Likelihood')
axes[1].plot(theta_range, normalize(gaussian_prior), 'g-', lw=1.5, alpha=0.5, label='Gaussian Prior N(0,4)')
axes[1].plot(theta_range, normalize(log_post_gaussian), 'r-', lw=2.5, label='Posterior (MAP)')
mle = theta_range[np.argmax(log_lik)]
map_gauss = theta_range[np.argmax(log_post_gaussian)]
axes[1].axvline(mle, color='blue', linestyle='--', alpha=0.5, label=f'MLE={mle:.1f}')
axes[1].axvline(map_gauss, color='red', linestyle='--', alpha=0.5, label=f'MAP={map_gauss:.1f}')
axes[1].set_title('MAP with Gaussian Prior\n= L2 Regularization (Ridge)', fontweight='bold', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].set_xlabel('θ')

# Plot 3: MAP with Laplace prior (= L1 regularization)
axes[2].plot(theta_range, normalize(log_lik), 'b-', lw=1.5, alpha=0.5, label='Likelihood')
axes[2].plot(theta_range, normalize(laplace_prior), 'g-', lw=1.5, alpha=0.5, label='Laplace Prior')
axes[2].plot(theta_range, normalize(log_post_laplace), 'r-', lw=2.5, label='Posterior (MAP)')
map_lap = theta_range[np.argmax(log_post_laplace)]
axes[2].axvline(map_lap, color='red', linestyle='--', alpha=0.5, label=f'MAP={map_lap:.1f}')
axes[2].set_title('MAP with Laplace Prior\n= L1 Regularization (Lasso)', fontweight='bold', fontsize=12)
axes[2].legend(fontsize=9)
axes[2].set_xlabel('θ')

for ax in axes:
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Density (normalized)')

plt.suptitle('MLE vs MAP Estimation — The Bayesian View of Regularization', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mle_vs_map.png', dpi=150)
plt.show()
```

---

## 13. Information Theory Essentials

### Entropy

Measures the **uncertainty** or **information content** of a random variable.

$$H(X) = -\sum_{x} P(x) \log_2 P(x) \quad \text{(bits)}$$

- **Fair coin**: $H = -0.5\log_2(0.5) - 0.5\log_2(0.5) = 1$ bit → maximum uncertainty
- **Biased coin** ($p=0.99$): $H \approx 0.08$ bits → almost no uncertainty
- **Certain event** ($p=1$): $H = 0$ bits → zero uncertainty

### Cross-Entropy

Measures the expected number of bits needed to encode data from distribution $P$ using a code optimized for distribution $Q$:

$$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$

In ML, $P$ is the true label distribution (one-hot), and $Q$ is the model's predicted probabilities. **Minimizing cross-entropy = maximizing log-likelihood**.

### KL Divergence

Measures how different $Q$ is from $P$:

$$D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = H(P, Q) - H(P)$$

- $D_{KL} \geq 0$ (Gibbs' inequality)
- $D_{KL} = 0$ if and only if $P = Q$
- **Not symmetric**: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
- Used in VAEs to measure how far the learned distribution is from the prior

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(probs):
    """Compute entropy in bits."""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros to avoid log(0)
    return -np.sum(probs * np.log2(probs))

def cross_entropy(p, q):
    """Cross-entropy H(P, Q)."""
    p, q = np.array(p), np.array(q)
    mask = p > 0
    return -np.sum(p[mask] * np.log2(q[mask]))

def kl_divergence(p, q):
    """KL divergence D_KL(P || Q)."""
    return cross_entropy(p, q) - entropy(p)

# Plot: Entropy of a coin as p varies
p_range = np.linspace(0.001, 0.999, 200)
entropies = [entropy([p, 1-p]) for p in p_range]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(p_range, entropies, 'b-', linewidth=2.5)
axes[0].fill_between(p_range, entropies, alpha=0.1, color='blue')
axes[0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Maximum = 1 bit')
axes[0].axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='p = 0.5 (fair coin)')
axes[0].set_xlabel('P(Heads)', fontsize=12)
axes[0].set_ylabel('Entropy H(X) [bits]', fontsize=12)
axes[0].set_title('Binary Entropy Function\nMaximum uncertainty at p=0.5', fontweight='bold', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Plot: Cross-entropy in classification
# True distribution: [0.7, 0.2, 0.1] (3-class)
p_true = [0.7, 0.2, 0.1]

# Compare different model predictions
predictions = [
    ([0.7, 0.2, 0.1], 'Perfect model'),
    ([0.6, 0.25, 0.15], 'Good model'),
    ([0.4, 0.35, 0.25], 'Mediocre model'),
    ([0.33, 0.33, 0.34], 'Random guessing'),
    ([0.1, 0.2, 0.7], 'Terrible model'),
]

names = [n for _, n in predictions]
ce_values = [cross_entropy(p_true, q) for q, _ in predictions]
kl_values = [kl_divergence(p_true, q) for q, _ in predictions]
h_p = entropy(p_true)

colors = ['#4CAF50', '#8BC34A', '#FFCE56', '#FF9800', '#FF6384']
bars = axes[1].bar(names, ce_values, color=colors, edgecolor='white', alpha=0.8)
axes[1].axhline(y=h_p, color='blue', linestyle='--', linewidth=2, label=f'H(P) = {h_p:.3f} (lower bound)')
axes[1].set_ylabel('Cross-Entropy H(P, Q) [bits]', fontsize=12)
axes[1].set_title('Cross-Entropy: True dist P = [0.7, 0.2, 0.1]\nCloser to H(P) = better model', 
                   fontweight='bold', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].tick_params(axis='x', rotation=20)
for bar, val in zip(bars, ce_values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02, 
                 f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('information_theory.png', dpi=150)
plt.show()
```

---

## 14. Project Ideas & What's Next

### Project Ideas

#### 🟢 Project 1: Monte Carlo π Estimator (Beginner)
Estimate $\pi$ by randomly sampling points in a unit square and checking if they fall inside a unit circle. Uses: LLN, random variables, simulation.

#### 🟡 Project 2: Bayesian A/B Testing (Intermediate)
Build a Bayesian framework for comparing two website designs:
- Model click-through rates as Beta distributions
- Update with observed data
- Compute $P(\text{variant A} > \text{variant B})$

#### 🔴 Project 3: MLE Parameter Estimation Engine (Advanced)
Build a general-purpose MLE engine that:
- Accepts any parametric distribution (Normal, Exponential, Poisson, etc.)
- Computes the log-likelihood surface
- Finds $\hat{\theta}_{MLE}$ via gradient ascent
- Computes confidence intervals using the Fisher Information Matrix

### What's Next

| Next Topic | Why |
|------------|-----|
| [SQL for Data Science](./10-SQL-For-Data-Science.md) | Apply math to query databases and prepare datasets |

---

[← Mathematical Foundations for Machine Learning](08-Mathematical-Foundations.md) | [Back to Index](./README.md) | [Next: 10. SQL for Data Science →](10-SQL-For-Data-Science.md)
