# 📈 Probability Distributions — The Complete Reference

> **Prerequisites**: Probability & Statistics | **Difficulty**: ⭐⭐☆☆☆ Beginner-Intermediate

---

## 📋 Table of Contents

1. [Why Distributions Matter in ML](#1-why-distributions-matter-in-ml)
2. [Discrete Distributions](#2-discrete-distributions)
3. [Continuous Distributions](#3-continuous-distributions)
4. [Distribution Relationships](#4-distribution-relationships)
5. [Choosing the Right Distribution](#5-choosing-the-right-distribution)
6. [Fitting Distributions to Data](#6-fitting-distributions-to-data)
7. [Distribution Gallery (Visual Reference)](#7-distribution-gallery-visual-reference)
8. [Project Ideas & What's Next](#8-project-ideas--whats-next)

---

## 1. Why Distributions Matter in ML

| ML Concept | Distribution Used |
|------------|------------------|
| Logistic Regression output | Bernoulli |
| Word count features | Multinomial / Poisson |
| Error terms, noise | Normal (Gaussian) |
| Time between events | Exponential |
| Bayesian priors on probabilities | Beta |
| Bayesian priors on positive reals | Gamma |
| Topic models (LDA) | Dirichlet |
| Dropout mask | Bernoulli |
| Weight initialization | Normal / Uniform |
| Softmax output | Categorical |
| VAE latent space | Normal |
| Anomaly detection thresholds | Chi-squared |

---

## 2. Discrete Distributions

### 2.1 Bernoulli Distribution

A single binary trial: success (1) with probability $p$ or failure (0) with probability $1-p$.

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

| Property | Value |
|----------|-------|
| Mean | $p$ |
| Variance | $p(1-p)$ |
| Support | $\{0, 1\}$ |
| ML Use | Single coin flip, binary classification, dropout |

### 2.2 Binomial Distribution

Number of successes in $n$ independent Bernoulli trials.

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \dots, n$$

**Derivation of the mean**:
$$E[X] = E\left[\sum_{i=1}^n X_i\right] = \sum_{i=1}^n E[X_i] = np$$

| Property | Value |
|----------|-------|
| Mean | $np$ |
| Variance | $np(1-p)$ |
| Support | $\{0, 1, \dots, n\}$ |
| ML Use | Number of correct predictions in $n$ samples, A/B test outcomes |

### 2.3 Poisson Distribution

Number of events occurring in a fixed interval when events happen at a constant average rate $\lambda$.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \dots$$

**Derivation**: The Poisson is the limit of the Binomial as $n \to \infty$, $p \to 0$, with $np = \lambda$ constant.

| Property | Value |
|----------|-------|
| Mean | $\lambda$ |
| Variance | $\lambda$ (mean equals variance!) |
| Support | $\{0, 1, 2, \dots\}$ |
| ML Use | Count data, rare event modeling, text (word counts) |

### 2.4 Geometric Distribution

Number of Bernoulli trials until the **first** success.

$$P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \dots$$

| Property | Value |
|----------|-------|
| Mean | $1/p$ |
| Variance | $(1-p)/p^2$ |
| Key property | **Memoryless**: $P(X > s + t \mid X > s) = P(X > t)$ |
| ML Use | Modeling number of attempts until success, waiting times |

### 2.5 Negative Binomial Distribution

Number of trials until $r$ successes (generalizes Geometric with $r = 1$).

$$P(X = k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k = r, r+1, \dots$$

| Property | Value |
|----------|-------|
| Mean | $r/p$ |
| Variance | $r(1-p)/p^2$ |
| ML Use | Overdispersed count data (when variance > mean, unlike Poisson) |

### 2.6 Categorical & Multinomial

**Categorical**: Single trial with $K$ possible outcomes (generalized Bernoulli).
$$P(X = k) = p_k, \quad k \in \{1, \dots, K\}, \quad \sum p_k = 1$$

**Multinomial**: $n$ independent Categorical trials (generalized Binomial).
$$P(X_1 = n_1, \dots, X_K = n_K) = \frac{n!}{n_1! \cdots n_K!} p_1^{n_1} \cdots p_K^{n_K}$$

| ML Use | Softmax output, multi-class classification, topic models |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Bernoulli
p = 0.7
axes[0, 0].bar([0, 1], [1-p, p], color=['#FF6384', '#4CAF50'], width=0.4, edgecolor='white')
axes[0, 0].set_title(f'Bernoulli(p={p})\nE={p}, Var={p*(1-p):.2f}', fontweight='bold')
axes[0, 0].set_xticks([0, 1])
axes[0, 0].set_xticklabels(['Failure', 'Success'])

# Binomial for different n, p
for n, p, c in [(10, 0.5, '#36A2EB'), (20, 0.3, '#FF6384'), (50, 0.7, '#4CAF50')]:
    x = np.arange(0, 60)
    axes[0, 1].plot(x, stats.binom.pmf(x, n, p), 'o-', color=c, ms=3, lw=1.5, label=f'n={n}, p={p}')
axes[0, 1].set_title('Binomial Distribution', fontweight='bold')
axes[0, 1].legend(fontsize=9)

# Poisson
for lam, c in [(1, '#FF6384'), (5, '#36A2EB'), (10, '#4CAF50'), (20, '#FFCE56')]:
    x = np.arange(0, 35)
    axes[0, 2].plot(x, stats.poisson.pmf(x, lam), 'o-', color=c, ms=3, lw=1.5, label=f'λ={lam}')
axes[0, 2].set_title('Poisson Distribution', fontweight='bold')
axes[0, 2].legend(fontsize=9)

# Geometric
for p, c in [(0.1, '#FF6384'), (0.3, '#36A2EB'), (0.7, '#4CAF50')]:
    x = np.arange(1, 25)
    axes[1, 0].plot(x, stats.geom.pmf(x, p), 'o-', color=c, ms=4, lw=1.5, label=f'p={p}')
axes[1, 0].set_title('Geometric Distribution', fontweight='bold')
axes[1, 0].legend(fontsize=9)

# Negative Binomial
for r, p, c in [(3, 0.3, '#FF6384'), (5, 0.5, '#36A2EB'), (10, 0.7, '#4CAF50')]:
    x = np.arange(r, 40)
    axes[1, 1].plot(x, stats.nbinom.pmf(x - r, r, p), 'o-', color=c, ms=3, lw=1.5, label=f'r={r}, p={p}')
axes[1, 1].set_title('Negative Binomial', fontweight='bold')
axes[1, 1].legend(fontsize=9)

# Categorical (Multinomial single trial)
categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']
probs = [0.35, 0.25, 0.2, 0.12, 0.08]
colors_cat = ['#FF6384', '#36A2EB', '#4CAF50', '#FFCE56', '#9B59B6']
axes[1, 2].bar(categories, probs, color=colors_cat, edgecolor='white')
axes[1, 2].set_title('Categorical Distribution\n(Softmax output)', fontweight='bold')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('P(X=k)')

plt.suptitle('Discrete Probability Distributions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('discrete_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3. Continuous Distributions

### 3.1 Uniform Distribution

Equal probability over interval $[a, b]$.

$$f(x) = \frac{1}{b - a}, \quad a \leq x \leq b$$

| Property | Value |
|----------|-------|
| Mean | $(a+b)/2$ |
| Variance | $(b-a)^2/12$ |
| CDF | $F(x) = (x-a)/(b-a)$ |
| ML Use | Random initialization, uniform noise, random number generation |

### 3.2 Normal (Gaussian) Distribution

The most important distribution in all of statistics and ML.

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Why is it everywhere?**
1. **CLT**: Average of many independent variables → Normal
2. **Maximum entropy**: Among all distributions with a given mean and variance, the Normal has maximum entropy (least assumptions)
3. **Computational convenience**: Conjugate to itself, closed-form MLE

| Property | Value |
|----------|-------|
| Mean | $\mu$ |
| Variance | $\sigma^2$ |
| 68-95-99.7 rule | $\mu \pm 1\sigma$ (68%), $\mu \pm 2\sigma$ (95%), $\mu \pm 3\sigma$ (99.7%) |
| ML Use | Error modeling, weight initialization, GANs latent space, noise |

**Standard Normal** $Z \sim \mathcal{N}(0, 1)$:
$$Z = \frac{X - \mu}{\sigma} \quad \text{(standardization / z-score)}$$

### 3.3 Exponential Distribution

Time between events in a Poisson process. The continuous analog of the Geometric.

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

| Property | Value |
|----------|-------|
| Mean | $1/\lambda$ |
| Variance | $1/\lambda^2$ |
| Key property | **Memoryless**: $P(X > s + t \mid X > s) = P(X > t)$ |
| ML Use | Modeling wait times, survival analysis, inter-arrival times |

### 3.4 Beta Distribution

Defined on $[0, 1]$ — perfect for modeling **probabilities**.

$$f(x; \alpha, \beta) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the Beta function.

| Property | Value |
|----------|-------|
| Mean | $\alpha / (\alpha + \beta)$ |
| Variance | $\alpha\beta / [(\alpha+\beta)^2(\alpha+\beta+1)]$ |
| Special cases | $\text{Beta}(1,1) = \text{Uniform}[0,1]$, $\text{Beta}(\alpha,\alpha)$ = symmetric |
| ML Use | Bayesian prior for probabilities, A/B testing, Thompson sampling |

### 3.5 Gamma Distribution

Generalization of the Exponential. Models the time until $\alpha$ events in a Poisson process.

$$f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0$$

| Property | Value |
|----------|-------|
| Mean | $\alpha/\beta$ |
| Variance | $\alpha/\beta^2$ |
| Special case | $\text{Gamma}(1, \beta) = \text{Exponential}(\beta)$ |
| ML Use | Bayesian prior for positive-valued parameters (rates, precisions) |

### 3.6 Student's t-Distribution

The distribution of the t-statistic. Heavier tails than the Normal.

$$f(x; \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

| Property | Value |
|----------|-------|
| Mean | 0 (for $\nu > 1$) |
| Variance | $\nu/(\nu-2)$ (for $\nu > 2$) |
| Limit | As $\nu \to \infty$, $t \to \mathcal{N}(0,1)$ |
| ML Use | Hypothesis testing with small samples, robust regression |

### 3.7 Chi-Squared Distribution

Sum of $k$ squared standard Normal variables: $\chi^2_k = \sum_{i=1}^k Z_i^2$.

$$f(x; k) = \frac{1}{2^{k/2}\Gamma(k/2)} x^{k/2-1} e^{-x/2}, \quad x > 0$$

| Property | Value |
|----------|-------|
| Mean | $k$ |
| Variance | $2k$ |
| ML Use | Chi-squared tests, goodness-of-fit, feature selection |

### 3.8 Log-Normal Distribution

If $\ln(X) \sim \mathcal{N}(\mu, \sigma^2)$, then $X$ follows a Log-Normal distribution.

$$f(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0$$

| Property | Value |
|----------|-------|
| Mean | $e^{\mu + \sigma^2/2}$ |
| Variance | $(e^{\sigma^2} - 1)e^{2\mu + \sigma^2}$ |
| ML Use | Financial returns, response times, file sizes |

### 3.9 Dirichlet Distribution

Generalization of the Beta to $K$ dimensions. A distribution over **probability vectors**.

$$f(\mathbf{x}; \boldsymbol{\alpha}) = \frac{\Gamma(\sum \alpha_i)}{\prod \Gamma(\alpha_i)} \prod_{i=1}^K x_i^{\alpha_i - 1}$$

| ML Use | Prior for multinomial parameters, topic models (LDA), Bayesian classification |

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(3, 3, figsize=(20, 16))

x = np.linspace(-5, 5, 500)
x_pos = np.linspace(0.01, 10, 500)
x_01 = np.linspace(0.01, 0.99, 500)

# 1. Uniform
axes[0, 0].fill_between(np.linspace(0, 1, 100), 1, alpha=0.3, color='#9B59B6')
axes[0, 0].plot([0, 0, 1, 1], [0, 1, 1, 0], color='#9B59B6', lw=2.5)
axes[0, 0].set_title('Uniform[0,1]', fontweight='bold')
axes[0, 0].set_xlim([-0.5, 1.5])

# 2. Normal
for mu, sig, c in [(0, 0.5, '#FF6384'), (0, 1, '#36A2EB'), (0, 2, '#4CAF50'), (-2, 0.7, '#FFCE56')]:
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sig), lw=2, color=c, label=f'μ={mu},σ={sig}')
axes[0, 1].set_title('Normal (Gaussian)', fontweight='bold')
axes[0, 1].legend(fontsize=8)

# 3. Exponential
for lam, c in [(0.5, '#FF6384'), (1, '#36A2EB'), (2, '#4CAF50')]:
    axes[0, 2].plot(x_pos, stats.expon.pdf(x_pos, scale=1/lam), lw=2, color=c, label=f'λ={lam}')
axes[0, 2].set_title('Exponential', fontweight='bold')
axes[0, 2].legend(fontsize=9)

# 4. Beta
betas = [(0.5, 0.5, '#FF6384'), (1, 1, '#36A2EB'), (2, 5, '#4CAF50'), 
         (5, 2, '#FFCE56'), (5, 5, '#9B59B6'), (2, 2, 'gray')]
for a, b, c in betas:
    axes[1, 0].plot(x_01, stats.beta.pdf(x_01, a, b), lw=2, color=c, label=f'α={a},β={b}')
axes[1, 0].set_title('Beta Distribution', fontweight='bold')
axes[1, 0].legend(fontsize=8)
axes[1, 0].set_ylim([0, 4])

# 5. Gamma
for a, b, c in [(1, 1, '#FF6384'), (2, 1, '#36A2EB'), (3, 1, '#4CAF50'), (5, 1, '#FFCE56'), (2, 2, '#9B59B6')]:
    axes[1, 1].plot(x_pos, stats.gamma.pdf(x_pos, a, scale=1/b), lw=2, color=c, label=f'α={a},β={b}')
axes[1, 1].set_title('Gamma Distribution', fontweight='bold')
axes[1, 1].legend(fontsize=8)

# 6. Student's t
for nu, c in [(1, '#FF6384'), (3, '#FFCE56'), (5, '#36A2EB'), (30, '#4CAF50')]:
    axes[1, 2].plot(x, stats.t.pdf(x, nu), lw=2, color=c, label=f'ν={nu}')
axes[1, 2].plot(x, stats.norm.pdf(x), 'k--', lw=1.5, alpha=0.5, label='Normal')
axes[1, 2].set_title("Student's t Distribution", fontweight='bold')
axes[1, 2].legend(fontsize=9)

# 7. Chi-Squared
for k, c in [(1, '#FF6384'), (2, '#36A2EB'), (5, '#4CAF50'), (10, '#FFCE56')]:
    axes[2, 0].plot(x_pos, stats.chi2.pdf(x_pos, k), lw=2, color=c, label=f'k={k}')
axes[2, 0].set_title('Chi-Squared Distribution', fontweight='bold')
axes[2, 0].legend(fontsize=9)
axes[2, 0].set_xlim([0, 10])

# 8. Log-Normal
for mu, sig, c in [(0, 0.25, '#36A2EB'), (0, 0.5, '#4CAF50'), (0, 1, '#FF6384'), (1, 0.5, '#FFCE56')]:
    axes[2, 1].plot(x_pos, stats.lognorm.pdf(x_pos, sig, scale=np.exp(mu)), lw=2, color=c, 
                    label=f'μ={mu},σ={sig}')
axes[2, 1].set_title('Log-Normal Distribution', fontweight='bold')
axes[2, 1].legend(fontsize=8)
axes[2, 1].set_xlim([0, 8])

# 9. Distribution Decision Tree
axes[2, 2].axis('off')
decision = """
DISTRIBUTION CHEAT SHEET:

Binary outcome?        → Bernoulli
Count of successes?    → Binomial
Count (rare events)?   → Poisson
Wait time to event?    → Exponential
Sum of wait times?     → Gamma
Probability parameter? → Beta
Real-valued, symmetric? → Normal
Real-valued, skewed?   → Log-Normal
Small sample t-test?   → Student's t
Sum of squared normals? → Chi-squared
Prob vector parameter? → Dirichlet
"""
axes[2, 2].text(0.05, 0.5, decision, fontsize=11, fontfamily='monospace', 
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='#E8F5E9', alpha=0.8))
axes[2, 2].set_title('Quick Reference', fontweight='bold')

for ax in axes.flat[:-1]:
    ax.grid(True, alpha=0.3)

plt.suptitle('Continuous Probability Distributions Gallery', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('continuous_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 4. Distribution Relationships

```
                    Bernoulli(p)
                    ↓ (n trials)
                Binomial(n, p)
               ↙              ↘
    n→∞, p→0            n→∞
    np=λ const         CLT applies
              ↓                ↓
        Poisson(λ)      Normal(μ, σ²)
              ↓                ↓
    (continuous       (square & sum)
     analog)          
              ↓                ↓
      Exponential(λ)    Chi-Squared(k)
              ↓                ↓
    (sum of k)         (ratio with Normal)
              ↓                ↓
       Gamma(k, λ)     Student's t(ν)
              ↓
    (k=1)→ Exponential
    (special case)→ Chi-Squared(2k, 1/2)
    
    Beta(α, β) ← conjugate prior for → Bernoulli/Binomial
    Gamma(α, β) ← conjugate prior for → Poisson/Exponential
    Dirichlet(α) ← conjugate prior for → Categorical/Multinomial
```

### Key Relationships (Mathematical)

1. $\text{Binomial}(n, p) \xrightarrow{n \to \infty} \mathcal{N}(np, np(1-p))$ — CLT approximation
2. $\text{Binomial}(n, p) \xrightarrow{n \to \infty, p \to 0} \text{Poisson}(np)$ — Rare events
3. $\text{Exponential}(\lambda) = \text{Gamma}(1, \lambda)$ — Special case
4. $\chi^2(k) = \text{Gamma}(k/2, 1/2)$ — Chi-squared is a Gamma
5. $\text{Beta}(1, 1) = \text{Uniform}(0, 1)$ — Flat prior
6. If $X_1, \dots, X_k \sim \mathcal{N}(0,1)$ i.i.d., then $\sum X_i^2 \sim \chi^2(k)$

---

## 5. Choosing the Right Distribution

| Data Type | Shape | Distribution | Example |
|-----------|-------|-------------|---------|
| Binary (yes/no) | — | Bernoulli | Click / no click |
| Count (bounded) | — | Binomial | Correct answers out of 20 |
| Count (unbounded) | — | Poisson | Website visits per hour |
| Continuous, symmetric | Bell-shaped | Normal | Test scores, errors |
| Continuous, positive, skewed | Right-skewed | Log-Normal | Income, stock returns |
| Continuous, positive | Exponential decay | Exponential | Time between clicks |
| Continuous, positive | Flexible | Gamma | Insurance claims |
| Probability (0 to 1) | Flexible | Beta | Conversion rate |
| Duration until event | Various | Weibull | Component lifetime |

### Diagnostic Tests for Choosing Distributions

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate real-world-like data
data = np.random.lognormal(mean=3, sigma=0.8, size=500)  # e.g., income data

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histogram with candidate distributions
axes[0, 0].hist(data, bins=40, density=True, alpha=0.5, color='#36A2EB', edgecolor='white')
x = np.linspace(data.min(), data.max(), 200)

# Fit different distributions
distributions = [
    ('Normal', stats.norm, 'r-'),
    ('Log-Normal', stats.lognorm, 'g-'),
    ('Gamma', stats.gamma, 'b-'),
    ('Exponential', stats.expon, 'm-'),
]

for name, dist, style in distributions:
    params = dist.fit(data)
    pdf = dist.pdf(x, *params)
    ks_stat, ks_p = stats.kstest(data, dist.cdf, args=params)
    axes[0, 0].plot(x, pdf, style, lw=2, label=f'{name} (KS p={ks_p:.4f})')

axes[0, 0].set_title('Histogram + Fitted Distributions', fontweight='bold', fontsize=13)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 2. Q-Q Plot (Normal)
stats.probplot(data, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (Normal)', fontweight='bold', fontsize=13)
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q Plot (Log-Normal)
stats.probplot(np.log(data), dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot of log(data) → Normal\n(Testing Log-Normal fit)', fontweight='bold', fontsize=13)
axes[1, 0].grid(True, alpha=0.3)

# 4. CDF comparison
axes[1, 1].step(np.sort(data), np.arange(1, len(data)+1)/len(data), 'k-', lw=1.5, label='Empirical CDF')
lognorm_params = stats.lognorm.fit(data)
axes[1, 1].plot(np.sort(data), stats.lognorm.cdf(np.sort(data), *lognorm_params), 
                'g-', lw=2.5, label='Log-Normal CDF')
axes[1, 1].set_title('Empirical vs Fitted CDF', fontweight='bold', fontsize=13)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Distribution Fitting: Finding the Best Model for Data', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('distribution_fitting.png', dpi=150)
plt.show()
```

---

## 6. Fitting Distributions to Data

### Method 1: Maximum Likelihood Estimation (MLE)

Find parameters that maximize $L(\theta) = \prod f(x_i; \theta)$.

```python
from scipy import stats
import numpy as np

np.random.seed(42)
data = np.random.gamma(shape=3, scale=2, size=1000)

# Fit Gamma distribution via MLE
shape_mle, loc_mle, scale_mle = stats.gamma.fit(data, floc=0)
print(f"True: shape=3.0, scale=2.0")
print(f"MLE:  shape={shape_mle:.3f}, scale={scale_mle:.3f}")

# Fit Normal for comparison
mu_mle, sigma_mle = stats.norm.fit(data)
print(f"Normal fit: μ={mu_mle:.3f}, σ={sigma_mle:.3f}")

# Compare via AIC (lower is better)
log_lik_gamma = np.sum(stats.gamma.logpdf(data, shape_mle, scale=scale_mle))
log_lik_norm = np.sum(stats.norm.logpdf(data, mu_mle, sigma_mle))

aic_gamma = 2 * 2 - 2 * log_lik_gamma   # 2 parameters
aic_norm = 2 * 2 - 2 * log_lik_norm      # 2 parameters

print(f"\nAIC (Gamma):  {aic_gamma:.1f}")
print(f"AIC (Normal): {aic_norm:.1f}")
print(f"Winner: {'Gamma' if aic_gamma < aic_norm else 'Normal'} (lower AIC is better)")
```

### Method 2: Method of Moments

Set sample moments equal to theoretical moments and solve.

For the Normal distribution:
$$\hat{\mu} = \bar{x}, \quad \hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$$

For the Gamma($\alpha$, $\beta$):
$$E[X] = \frac{\alpha}{\beta} = \bar{x}, \quad \text{Var}(X) = \frac{\alpha}{\beta^2} = s^2$$
$$\Rightarrow \hat{\beta} = \frac{\bar{x}}{s^2}, \quad \hat{\alpha} = \frac{\bar{x}^2}{s^2}$$

### Method 3: Kolmogorov-Smirnov Test

Tests if data follows a specific distribution by measuring the maximum distance between the empirical and theoretical CDFs:

$$D = \max_x |F_n(x) - F(x)|$$

```python
from scipy import stats
import numpy as np

np.random.seed(42)
data = np.random.exponential(2, 200)

# Test against different distributions
tests = {
    'Exponential': stats.expon.fit(data),
    'Normal': stats.norm.fit(data),
    'Gamma': stats.gamma.fit(data),
}

print("Kolmogorov-Smirnov Tests:")
print("-" * 50)
for name, params in tests.items():
    dist = getattr(stats, name.lower()) if name != 'Gamma' else stats.gamma
    ks_stat, p_value = stats.kstest(data, dist.cdf, args=params)
    verdict = "✅ Good fit" if p_value > 0.05 else "❌ Poor fit"
    print(f"{name:15s} KS={ks_stat:.4f}, p={p_value:.4f} {verdict}")
```

---

## 7. Distribution Gallery (Visual Reference)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Comprehensive visual reference card
fig, axes = plt.subplots(4, 4, figsize=(24, 20))

x = np.linspace(-4, 4, 300)
x_pos = np.linspace(0.01, 8, 300)
x_01 = np.linspace(0.005, 0.995, 300)

configs = [
    # (row, col, title, x_data, distributions)
    (0, 0, 'Bernoulli(p=0.3)', [0, 1], [('bar', [0.7, 0.3], '#4CAF50')]),
    (0, 1, 'Binomial(n=20, p=0.4)', np.arange(0, 21), [('stem', stats.binom.pmf(np.arange(0,21), 20, 0.4), '#36A2EB')]),
    (0, 2, 'Poisson(λ=5)', np.arange(0, 18), [('stem', stats.poisson.pmf(np.arange(0,18), 5), '#FF6384')]),
    (0, 3, 'Geometric(p=0.3)', np.arange(1, 16), [('stem', stats.geom.pmf(np.arange(1,16), 0.3), '#FFCE56')]),
    (1, 0, 'Uniform[0,1]', x, [('plot', stats.uniform.pdf(x, 0, 1), '#9B59B6')]),
    (1, 1, 'Normal(0,1)', x, [('plot', stats.norm.pdf(x), '#36A2EB')]),
    (1, 2, 'Exponential(λ=1)', x_pos, [('plot', stats.expon.pdf(x_pos), '#FF6384')]),
    (1, 3, 'Log-Normal(0,0.5)', x_pos, [('plot', stats.lognorm.pdf(x_pos, 0.5), '#4CAF50')]),
    (2, 0, 'Beta(2,5)', x_01, [('plot', stats.beta.pdf(x_01, 2, 5), '#FF6384')]),
    (2, 1, 'Beta(5,2)', x_01, [('plot', stats.beta.pdf(x_01, 5, 2), '#36A2EB')]),
    (2, 2, 'Gamma(3,1)', x_pos, [('plot', stats.gamma.pdf(x_pos, 3), '#4CAF50')]),
    (2, 3, 'Chi-squared(5)', x_pos, [('plot', stats.chi2.pdf(x_pos, 5), '#FFCE56')]),
    (3, 0, "Student's t(3)", x, [('plot', stats.t.pdf(x, 3), '#FF6384')]),
    (3, 1, 'Weibull(k=2)', x_pos, [('plot', stats.weibull_min.pdf(x_pos, 2), '#36A2EB')]),
    (3, 2, 'F(5,10)', x_pos, [('plot', stats.f.pdf(x_pos, 5, 10), '#4CAF50')]),
    (3, 3, 'Laplace(0,1)', x, [('plot', stats.laplace.pdf(x), '#9B59B6')]),
]

for r, c, title, xd, dists in configs:
    ax = axes[r, c]
    for dtype, ydata, color in dists:
        if dtype == 'bar':
            ax.bar([0, 1], ydata, color=color, edgecolor='white', width=0.4)
        elif dtype == 'stem':
            ax.stem(xd, ydata, basefmt=' ', linefmt=color, markerfmt='o')
        else:
            ax.plot(xd, ydata, color=color, lw=2.5)
            ax.fill_between(xd, ydata, alpha=0.15, color=color)
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3)

plt.suptitle('📊 Complete Distribution Gallery — Visual Reference', fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('distribution_gallery.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 8. Project Ideas & What's Next

### Project Ideas

#### 🟢 Project 1: Distribution Identifier (Beginner)
Build a tool that takes a dataset and automatically:
- Fits multiple candidate distributions via MLE
- Runs KS tests for each
- Ranks candidates by AIC/BIC
- Produces Q-Q plots and CDF comparisons

#### 🟡 Project 2: Monte Carlo Simulator (Intermediate)
Build a Monte Carlo simulation framework for risk analysis:
- Model uncertain inputs with appropriate distributions
- Propagate uncertainty through a financial model
- Compute Value at Risk (VaR) using simulation

#### 🔴 Project 3: Custom Distribution Library (Advanced)
Implement 10 distributions from scratch in Python/NumPy:
- PDF, CDF, quantile function, random sampling
- MLE parameter estimation
- Visualization and comparison tools

### What's Next

| Next Topic | Why |
|------------|-----|
| [Data Visualization Mastery](./08-Data-Visualization-Mastery.md) | Visualize distributions and statistical results |
| [Feature Engineering](./07-Feature-Engineering.md) | Transform data using distribution knowledge |
| [Statistical Inference](./03-Statistical-Inference.md) | Use distributions for hypothesis testing |

---

[← Bayesian Statistics](./04-Bayesian-Statistics.md) | [Back to Index](../README.md) | [Next: Data Preprocessing →](./06-Data-Preprocessing.md)
