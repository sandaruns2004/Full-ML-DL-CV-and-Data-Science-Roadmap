# 🎯 Bayesian Statistics — Thinking with Uncertainty

> **Prerequisites**: Probability & Statistics, Statistical Inference | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents

1. [Frequentist vs Bayesian: A Philosophical Shift](#1-frequentist-vs-bayesian-a-philosophical-shift)
2. [Bayes' Theorem for Parameters](#2-bayes-theorem-for-parameters)
3. [Choosing Priors](#3-choosing-priors)
4. [Conjugate Priors](#4-conjugate-priors)
5. [Beta-Binomial Model (Coin Flip)](#5-beta-binomial-model-coin-flip)
6. [Normal-Normal Model](#6-normal-normal-model)
7. [Bayesian Linear Regression](#7-bayesian-linear-regression)
8. [Markov Chain Monte Carlo (MCMC)](#8-markov-chain-monte-carlo-mcmc)
9. [Bayesian vs Frequentist: Comparison](#9-bayesian-vs-frequentist-comparison)
10. [Bayesian Methods in ML](#10-bayesian-methods-in-ml)
11. [Real-World Example: Bayesian A/B Testing](#11-real-world-example-bayesian-ab-testing)
12. [Project Ideas & What's Next](#12-project-ideas--whats-next)

---

## 1. Frequentist vs Bayesian: A Philosophical Shift

### Frequentist View
- Probability = **long-run frequency** of events
- Parameters are **fixed** but unknown constants
- Data is random (we got one sample from many possible samples)
- Uses p-values, confidence intervals

### Bayesian View
- Probability = **degree of belief** or uncertainty
- Parameters are **random variables** with distributions
- Data is fixed (we observe what we observe)
- Uses posterior distributions, credible intervals

| Concept | Frequentist | Bayesian |
|---------|------------|---------|
| Parameter $\theta$ | Fixed unknown | Random variable |
| Probability | Long-run frequency | Degree of belief |
| Inference | $P(\text{data} \mid \theta)$ | $P(\theta \mid \text{data})$ |
| Interval | Confidence interval | Credible interval |
| Prior information | Not used | Encoded as $P(\theta)$ |
| Prediction | Point estimate | Full predictive distribution |

### Why Bayesian for ML?

1. **Quantified uncertainty**: Instead of a single prediction, get a full distribution
2. **Small data**: Priors provide information when data is scarce
3. **Regularization**: Priors = regularization (Gaussian prior = L2, Laplace prior = L1)
4. **Online learning**: Posterior from today becomes the prior for tomorrow
5. **Model comparison**: Bayesian model evidence naturally penalizes complexity

---

## 2. Bayes' Theorem for Parameters

$$\underbrace{P(\theta \mid \mathcal{D})}_{\text{Posterior}} = \frac{\overbrace{P(\mathcal{D} \mid \theta)}^{\text{Likelihood}} \cdot \overbrace{P(\theta)}^{\text{Prior}}}{\underbrace{P(\mathcal{D})}_{\text{Evidence / Marginal Likelihood}}}$$

where:
$$P(\mathcal{D}) = \int P(\mathcal{D} \mid \theta) P(\theta) \, d\theta$$

Since $P(\mathcal{D})$ is just a normalizing constant, we often write:

$$P(\theta \mid \mathcal{D}) \propto P(\mathcal{D} \mid \theta) \cdot P(\theta)$$

$$\text{Posterior} \propto \text{Likelihood} \times \text{Prior}$$

**The flow of Bayesian inference**:
```
Prior belief P(θ) → Observe data D → Update via Bayes → Posterior P(θ|D) → Make decisions
     ↑                                                        ↓
     └──────── Posterior becomes tomorrow's prior ←────────────┘
```

---

## 3. Choosing Priors

### Types of Priors

| Prior Type | Description | When to Use |
|-----------|-------------|-------------|
| **Informative** | Strong prior reflecting expert knowledge | You have genuine domain knowledge |
| **Weakly informative** | Broad but bounded | You know the rough scale but not exact values |
| **Non-informative** (flat/uniform) | Uniform over all values | You want data to dominate (but be careful!) |
| **Conjugate** | Chosen so posterior has the same family | Mathematical convenience |
| **Empirical Bayes** | Prior estimated from data | Hierarchical models |

### Sensitivity Analysis

Always check: **does the prior matter?** If the posterior changes dramatically with different priors, you need more data.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Show how different priors lead to different posteriors with the SAME data
# Data: 7 heads out of 10 coin flips

n_flips = 10
n_heads = 7

theta = np.linspace(0, 1, 500)

# Likelihood: Binomial
likelihood = stats.binom.pmf(n_heads, n_flips, theta)

# Different priors
priors = {
    'Uniform (non-informative)': stats.beta.pdf(theta, 1, 1),
    'Weakly informative Beta(2,2)': stats.beta.pdf(theta, 2, 2),
    'Informative: "fair coin" Beta(50,50)': stats.beta.pdf(theta, 50, 50),
    'Informative: "biased" Beta(2,8)': stats.beta.pdf(theta, 2, 8),
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for ax, (prior_name, prior) in zip(axes.flat, priors.items()):
    # Posterior ∝ Likelihood × Prior
    posterior_unnorm = likelihood * prior
    posterior = posterior_unnorm / np.trapz(posterior_unnorm, theta)
    
    # Normalize each for comparison
    lik_norm = likelihood / np.trapz(likelihood, theta)
    prior_norm = prior / np.trapz(prior, theta)
    
    ax.plot(theta, prior_norm, 'g--', linewidth=2, label='Prior', alpha=0.8)
    ax.plot(theta, lik_norm, 'b:', linewidth=2, label='Likelihood', alpha=0.8)
    ax.plot(theta, posterior, 'r-', linewidth=3, label='Posterior')
    ax.fill_between(theta, posterior, alpha=0.15, color='red')
    
    # MAP estimate
    map_idx = np.argmax(posterior)
    ax.axvline(theta[map_idx], color='red', linestyle=':', alpha=0.5)
    ax.text(theta[map_idx] + 0.02, ax.get_ylim()[1]*0.9, f'MAP={theta[map_idx]:.2f}', 
            color='red', fontweight='bold', fontsize=10)
    
    ax.set_title(f'Prior: {prior_name}', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlabel('θ (probability of heads)')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

plt.suptitle(f'Effect of Different Priors on Posterior\n(Data: {n_heads}/{n_flips} heads)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('prior_sensitivity.png', dpi=150)
plt.show()
```

---

## 4. Conjugate Priors

A prior $P(\theta)$ is **conjugate** to a likelihood $P(\mathcal{D} \mid \theta)$ if the posterior $P(\theta \mid \mathcal{D})$ is in the **same family** as the prior.

This gives us **closed-form** posteriors — no integration required!

| Likelihood | Conjugate Prior | Posterior | Parameter Updated |
|-----------|----------------|-----------|-------------------|
| Binomial (coin flips) | Beta($\alpha, \beta$) | Beta($\alpha + k, \beta + n - k$) | Count of successes |
| Poisson (event counts) | Gamma($\alpha, \beta$) | Gamma($\alpha + \sum x_i, \beta + n$) | Sum of counts |
| Normal (known $\sigma$) | Normal($\mu_0, \sigma_0$) | Normal($\mu_n, \sigma_n$) | Weighted average of prior and data |
| Normal (known $\mu$) | Inverse-Gamma | Inverse-Gamma | Sum of squared deviations |
| Exponential | Gamma | Gamma | Sum of observations |
| Multinomial | Dirichlet | Dirichlet | Category counts |

### Why Conjugate Priors Are Beautiful

For the Beta-Binomial model:
- **Prior**: $\theta \sim \text{Beta}(\alpha, \beta)$ — "I pretend I've already seen $\alpha$ heads and $\beta$ tails"
- **Data**: $k$ heads in $n$ flips
- **Posterior**: $\theta \mid k \sim \text{Beta}(\alpha + k, \beta + n - k)$ — "Now I've seen $\alpha + k$ heads and $\beta + n - k$ tails"

The posterior just **adds the data counts to the prior pseudo-counts**!

---

## 5. Beta-Binomial Model (Coin Flip)

### The Complete Derivation

**Prior**: $\theta \sim \text{Beta}(\alpha, \beta)$:
$$P(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$

**Likelihood**: $k$ heads in $n$ flips:
$$P(k \mid \theta) = \binom{n}{k} \theta^k (1-\theta)^{n-k}$$

**Posterior** $\propto$ Likelihood $\times$ Prior:
$$P(\theta \mid k) \propto \theta^k(1-\theta)^{n-k} \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1} = \theta^{(\alpha+k)-1}(1-\theta)^{(\beta+n-k)-1}$$

This is a Beta distribution! So:
$$\theta \mid k \sim \text{Beta}(\alpha + k, \beta + n - k)$$

**Posterior Mean** (a weighted average of prior and data):
$$E[\theta \mid k] = \frac{\alpha + k}{\alpha + \beta + n} = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \underbrace{\frac{\alpha}{\alpha + \beta}}_{\text{prior mean}} + \frac{n}{\alpha + \beta + n} \cdot \underbrace{\frac{k}{n}}_{\text{MLE}}$$

As $n \to \infty$, the posterior mean → MLE. **The data overwhelms the prior.**

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Sequential Bayesian updating: watch the posterior evolve as data arrives
np.random.seed(42)

true_theta = 0.65  # True probability of heads
n_total = 100
flips = np.random.binomial(1, true_theta, n_total)

# Start with a uniform prior: Beta(1, 1)
alpha_0, beta_0 = 1, 1

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
theta = np.linspace(0, 1, 300)

update_points = [0, 1, 3, 5, 10, 25, 50, 100]

for i, (ax, n) in enumerate(zip(axes.flat, update_points)):
    if n == 0:
        alpha_n, beta_n = alpha_0, beta_0
        subtitle = 'Prior (no data)'
    else:
        data = flips[:n]
        k = data.sum()
        alpha_n = alpha_0 + k
        beta_n = beta_0 + n - k
        subtitle = f'n={n}, k={k} heads'
    
    posterior = stats.beta.pdf(theta, alpha_n, beta_n)
    
    ax.plot(theta, posterior, 'r-', linewidth=2.5)
    ax.fill_between(theta, posterior, alpha=0.2, color='red')
    ax.axvline(true_theta, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'True θ = {true_theta}')
    
    # 95% Credible Interval
    ci = stats.beta.ppf([0.025, 0.975], alpha_n, beta_n)
    ax.axvspan(ci[0], ci[1], alpha=0.1, color='green')
    
    post_mean = alpha_n / (alpha_n + beta_n)
    ax.axvline(post_mean, color='red', linestyle=':', alpha=0.7, label=f'Post. mean = {post_mean:.3f}')
    
    ax.set_title(f'{subtitle}\nBeta({alpha_n},{beta_n})', fontweight='bold', fontsize=11)
    ax.set_xlabel('θ')
    ax.set_xlim([0, 1])
    if i < 2:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Sequential Bayesian Updating — Beta-Binomial Model\n'
             'Posterior sharpens around true θ as data accumulates', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('bayesian_updating.png', dpi=150)
plt.show()
```

---

## 6. Normal-Normal Model

For estimating a Normal mean $\mu$ with known variance $\sigma^2$:

**Prior**: $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$

**Likelihood**: $\bar{x} \mid \mu \sim \mathcal{N}(\mu, \sigma^2/n)$

**Posterior**: $\mu \mid \mathcal{D} \sim \mathcal{N}(\mu_n, \sigma_n^2)$

$$\mu_n = \frac{\frac{\mu_0}{\sigma_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}} \qquad \sigma_n^2 = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}}$$

**In precision form** (precision = $1/\text{variance}$):
$$\tau_n = \tau_0 + n\tau \qquad \mu_n = \frac{\tau_0 \mu_0 + n\tau \bar{x}}{\tau_n}$$

The posterior precision is the **sum** of the prior precision and the data precision!

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Normal-Normal conjugate model
# Estimating the average height of a population

# Prior: Based on world average height ≈ 170cm, but uncertain
mu_0 = 170       # Prior mean
sigma_0 = 10     # Prior std (quite uncertain)

# True population
true_mu = 175
sigma = 6  # Known population std

# Observe data
np.random.seed(42)
sample_sizes = [0, 1, 5, 10, 30, 100]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
x = np.linspace(145, 200, 300)

for ax, n in zip(axes.flat, sample_sizes):
    if n == 0:
        post_mu = mu_0
        post_sigma = sigma_0
        subtitle = 'Prior (no data)'
    else:
        data = np.random.normal(true_mu, sigma, n)
        x_bar = data.mean()
        
        # Posterior parameters
        tau_0 = 1 / sigma_0**2
        tau_data = n / sigma**2
        tau_n = tau_0 + tau_data
        
        post_sigma = np.sqrt(1 / tau_n)
        post_mu = (tau_0 * mu_0 + tau_data * x_bar) / tau_n
        subtitle = f'n = {n}, x̄ = {x_bar:.1f}'
    
    # Plot
    prior_pdf = stats.norm.pdf(x, mu_0, sigma_0)
    post_pdf = stats.norm.pdf(x, post_mu, post_sigma)
    
    ax.plot(x, prior_pdf, 'g--', linewidth=2, label='Prior', alpha=0.7)
    ax.plot(x, post_pdf, 'r-', linewidth=2.5, label='Posterior')
    ax.fill_between(x, post_pdf, alpha=0.2, color='red')
    ax.axvline(true_mu, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'True μ = {true_mu}')
    
    # 95% Credible Interval
    ci = stats.norm.ppf([0.025, 0.975], post_mu, post_sigma)
    ax.axvspan(ci[0], ci[1], alpha=0.08, color='green')
    
    ax.set_title(f'{subtitle}\nPost: μ={post_mu:.1f}, σ={post_sigma:.2f}\n'
                 f'95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]', fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Height (cm)')

plt.suptitle('Normal-Normal Conjugate Model: Estimating Average Height', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('normal_normal_model.png', dpi=150)
plt.show()
```

---

## 7. Bayesian Linear Regression

### Standard (Frequentist) Linear Regression
Finds a single best-fit line: $\hat{w} = (X^TX)^{-1}X^Ty$

### Bayesian Linear Regression
Instead of a single weight vector $\hat{w}$, we compute a **distribution** over possible weight vectors.

**Prior on weights**: $w \sim \mathcal{N}(0, \sigma_w^2 I)$ (Gaussian prior = L2 regularization!)

**Likelihood**: $y \mid X, w \sim \mathcal{N}(Xw, \sigma^2 I)$

**Posterior**: $w \mid X, y \sim \mathcal{N}(\mu_w, \Sigma_w)$

$$\Sigma_w = (\sigma^{-2} X^TX + \sigma_w^{-2} I)^{-1}$$
$$\mu_w = \sigma^{-2} \Sigma_w X^T y$$

### Predictive Distribution

For a new input $x_*$, the predictive distribution is:

$$y_* \mid x_*, \mathcal{D} \sim \mathcal{N}(x_*^T \mu_w, \, x_*^T \Sigma_w x_* + \sigma^2)$$

The prediction includes **two sources of uncertainty**:
1. **Epistemic** (model uncertainty): $x_*^T \Sigma_w x_*$ — uncertainty in the weights
2. **Aleatoric** (data noise): $\sigma^2$ — irreducible noise

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Generate synthetic data
n = 20
X = np.sort(np.random.uniform(-3, 3, n))
y = 1.5 * X + 0.5 + np.random.normal(0, 0.8, n)

# Bayesian Linear Regression (closed-form)
sigma_noise = 0.8         # Known noise std
sigma_prior = 2.0         # Prior std on weights
alpha = 1 / sigma_prior**2  # Prior precision
beta = 1 / sigma_noise**2    # Noise precision

# Design matrix with bias term
Phi = np.column_stack([np.ones(n), X])  # [1, x]

# Posterior covariance and mean
S_inv = alpha * np.eye(2) + beta * Phi.T @ Phi
S = np.linalg.inv(S_inv)  # Posterior covariance
m = beta * S @ Phi.T @ y   # Posterior mean

print(f"Posterior weight mean: w₀ = {m[0]:.3f}, w₁ = {m[1]:.3f}")
print(f"Posterior covariance:\n{S}")

# Predictions
x_test = np.linspace(-5, 5, 200)
Phi_test = np.column_stack([np.ones(len(x_test)), x_test])

y_pred = Phi_test @ m  # Mean prediction
y_var = np.array([phi @ S @ phi + sigma_noise**2 for phi in Phi_test])  # Predictive variance
y_std = np.sqrt(y_var)

# Sample weight vectors from the posterior
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Sample lines from posterior
ax = axes[0]
ax.scatter(X, y, color='red', s=40, zorder=5, label='Observed data')
for _ in range(30):
    w_sample = np.random.multivariate_normal(m, S)
    y_sample = Phi_test @ w_sample
    ax.plot(x_test, y_sample, 'b-', alpha=0.15, linewidth=1)
ax.plot(x_test, y_pred, 'b-', linewidth=2.5, label='Posterior mean')
ax.set_title('Bayesian Linear Regression\n30 samples from posterior', fontweight='bold', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-5, 5])
ax.set_ylim([-8, 10])

# Plot 2: Predictive distribution with uncertainty
ax = axes[1]
ax.scatter(X, y, color='red', s=40, zorder=5, label='Observed data')
ax.plot(x_test, y_pred, 'b-', linewidth=2.5, label='Posterior mean')
ax.fill_between(x_test, y_pred - 1.96*y_std, y_pred + 1.96*y_std, 
                alpha=0.15, color='blue', label='95% predictive interval')
ax.fill_between(x_test, y_pred - y_std, y_pred + y_std, 
                alpha=0.25, color='blue', label='68% predictive interval')
ax.set_title('Predictive Distribution\nUncertainty grows away from data', fontweight='bold', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim([-5, 5])
ax.set_ylim([-8, 10])

plt.tight_layout()
plt.savefig('bayesian_regression.png', dpi=150)
plt.show()
```

---

## 8. Markov Chain Monte Carlo (MCMC)

### The Problem

For most real-world models, the posterior $P(\theta \mid \mathcal{D})$ has no closed form. The integral in the denominator is intractable:

$$P(\mathcal{D}) = \int P(\mathcal{D} \mid \theta) P(\theta) \, d\theta$$

### The Solution: MCMC

Instead of computing the posterior analytically, **sample from it**. Generate a sequence (chain) of samples $\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(T)}$ that converge to the posterior distribution.

### Metropolis-Hastings Algorithm

1. Start at some initial $\theta^{(0)}$
2. For $t = 1, 2, \dots, T$:
   a. **Propose** a new value: $\theta^* \sim q(\theta^* \mid \theta^{(t-1)})$ (e.g., $\theta^* = \theta^{(t-1)} + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$)
   b. Compute **acceptance ratio**: $\alpha = \min\left(1, \frac{P(\theta^*) P(\mathcal{D} \mid \theta^*)}{P(\theta^{(t-1)}) P(\mathcal{D} \mid \theta^{(t-1)})}\right)$
   c. Accept with probability $\alpha$: if accepted, $\theta^{(t)} = \theta^*$; otherwise $\theta^{(t)} = \theta^{(t-1)}$

Notice: we only need the **ratio** of posteriors — the intractable normalizing constant cancels out!

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def metropolis_hastings(log_posterior, theta_init, n_samples, proposal_std=0.5):
    """
    Metropolis-Hastings MCMC sampler.
    
    Parameters:
        log_posterior: function(theta) → log P(theta | data)
        theta_init: starting value
        n_samples: number of samples to generate
        proposal_std: standard deviation of the Gaussian proposal
    
    Returns:
        samples: array of MCMC samples
        acceptance_rate: fraction of proposals accepted
    """
    samples = np.zeros(n_samples)
    samples[0] = theta_init
    accepted = 0
    
    for t in range(1, n_samples):
        # Propose new value
        theta_star = samples[t-1] + np.random.normal(0, proposal_std)
        
        # Log acceptance ratio
        log_alpha = log_posterior(theta_star) - log_posterior(samples[t-1])
        
        # Accept or reject
        if np.log(np.random.uniform()) < log_alpha:
            samples[t] = theta_star
            accepted += 1
        else:
            samples[t] = samples[t-1]
    
    return samples, accepted / n_samples

# Example: Estimate mean of a Normal distribution
np.random.seed(42)
data = np.random.normal(5.0, 2.0, 30)  # True mean = 5.0

# Log-posterior (Normal likelihood × Normal prior)
def log_posterior(mu):
    # Likelihood: data ~ N(mu, 2^2)
    log_lik = np.sum(stats.norm.logpdf(data, loc=mu, scale=2.0))
    # Prior: mu ~ N(0, 10^2)
    log_prior = stats.norm.logpdf(mu, loc=0, scale=10)
    return log_lik + log_prior

# Run MCMC
samples, acc_rate = metropolis_hastings(log_posterior, theta_init=0, n_samples=10000, proposal_std=0.5)
burn_in = 2000
samples_burned = samples[burn_in:]

print(f"Acceptance rate: {acc_rate:.2%}")
print(f"Posterior mean: {samples_burned.mean():.3f}")
print(f"Posterior std: {samples_burned.std():.3f}")
print(f"95% Credible interval: [{np.percentile(samples_burned, 2.5):.3f}, {np.percentile(samples_burned, 97.5):.3f}]")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Trace plot
axes[0, 0].plot(samples, linewidth=0.5, alpha=0.7, color='#36A2EB')
axes[0, 0].axhline(5.0, color='red', linestyle='--', linewidth=2, label='True μ = 5.0')
axes[0, 0].axvline(burn_in, color='orange', linestyle='--', linewidth=2, label=f'Burn-in = {burn_in}')
axes[0, 0].set_title('MCMC Trace Plot', fontweight='bold', fontsize=13)
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('θ (estimated μ)')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Posterior histogram
axes[0, 1].hist(samples_burned, bins=50, density=True, alpha=0.7, color='#FF6384', edgecolor='white')
axes[0, 1].axvline(samples_burned.mean(), color='red', linestyle='-', linewidth=2.5, label=f'Mean = {samples_burned.mean():.3f}')
axes[0, 1].axvline(5.0, color='blue', linestyle='--', linewidth=2, label='True μ = 5.0')
ci = np.percentile(samples_burned, [2.5, 97.5])
axes[0, 1].axvspan(ci[0], ci[1], alpha=0.1, color='green', label=f'95% CI [{ci[0]:.2f}, {ci[1]:.2f}]')
axes[0, 1].set_title('Posterior Distribution (after burn-in)', fontweight='bold', fontsize=13)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Autocorrelation
max_lag = 100
autocorr = [np.corrcoef(samples_burned[:-lag], samples_burned[lag:])[0, 1] if lag > 0 else 1.0 
            for lag in range(max_lag)]
axes[1, 0].bar(range(max_lag), autocorr, color='#4CAF50', alpha=0.7)
axes[1, 0].axhline(0, color='black', linewidth=0.5)
axes[1, 0].set_title('Autocorrelation (should decay quickly)', fontweight='bold', fontsize=13)
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('Autocorrelation')
axes[1, 0].grid(True, alpha=0.3)

# Running mean
running_mean = np.cumsum(samples_burned) / np.arange(1, len(samples_burned) + 1)
axes[1, 1].plot(running_mean, linewidth=1.5, color='#36A2EB')
axes[1, 1].axhline(5.0, color='red', linestyle='--', linewidth=2, label='True μ = 5.0')
axes[1, 1].set_title('Running Mean (convergence diagnostic)', fontweight='bold', fontsize=13)
axes[1, 1].set_xlabel('Samples after burn-in')
axes[1, 1].set_ylabel('Running mean of θ')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Metropolis-Hastings MCMC — Bayesian Estimation', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('mcmc.png', dpi=150)
plt.show()
```

---

## 9. Bayesian vs Frequentist: Comparison

### Confidence vs Credible Intervals

| Feature | Confidence Interval (Frequentist) | Credible Interval (Bayesian) |
|---------|----------------------------------|------------------------------|
| **Interpretation** | 95% of such intervals contain θ | 95% probability θ is in this interval |
| **Intuitive?** | No (the interval is random, not θ) | Yes (directly about θ) |
| **Uses prior?** | No | Yes |
| **Computation** | Exact formulas | May need MCMC |

### When to Go Bayesian

| Use Bayesian | Use Frequentist |
|--------------|----------------|
| Small data (priors help) | Large data (prior doesn't matter) |
| Need uncertainty quantification | Need a quick p-value |
| Sequential updating needed | One-shot analysis |
| Expert knowledge available | Pure data-driven analysis |
| Model comparison needed | Simple hypothesis testing |

---

## 10. Bayesian Methods in ML

### Bayesian Neural Networks
Instead of single weight values, each weight has a distribution → uncertainty in predictions. Useful for safety-critical applications (medical diagnosis, autonomous driving).

### Gaussian Processes
A "Bayesian" approach to regression that places a prior over **functions** rather than parameters. Provides elegant uncertainty estimates without specifying a parametric model.

### Bayesian Optimization
Used for hyperparameter tuning (Optuna, BO). Models the objective function as a Gaussian Process, then uses the posterior to decide which hyperparameters to try next. Far more efficient than grid search!

### Variational Inference
When MCMC is too slow, approximate the posterior $P(\theta \mid \mathcal{D})$ with a simpler distribution $q(\theta)$ by minimizing $D_{KL}(q \| P)$. Used in VAEs and Bayesian deep learning.

---

## 11. Real-World Example: Bayesian A/B Testing

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Scenario: Testing two website designs
# Control A: 120 conversions out of 3000 visitors
# Treatment B: 150 conversions out of 3000 visitors

n_A, conv_A = 3000, 120
n_B, conv_B = 3000, 150

# Uninformative prior: Beta(1, 1)
alpha_prior, beta_prior = 1, 1

# Posterior distributions
alpha_A = alpha_prior + conv_A
beta_A = beta_prior + n_A - conv_A
alpha_B = alpha_prior + conv_B
beta_B = beta_prior + n_B - conv_B

# Sample from posteriors
n_samples = 100_000
samples_A = np.random.beta(alpha_A, beta_A, n_samples)
samples_B = np.random.beta(alpha_B, beta_B, n_samples)

# P(B > A)
prob_B_better = np.mean(samples_B > samples_A)

# Expected lift
lift = (samples_B - samples_A) / samples_A * 100
expected_lift = np.mean(lift)

# Expected loss (risk of choosing B when A is better)
loss_B = np.mean(np.maximum(samples_A - samples_B, 0))
loss_A = np.mean(np.maximum(samples_B - samples_A, 0))

print("=" * 60)
print("BAYESIAN A/B TEST RESULTS")
print("=" * 60)
print(f"Control A:   {conv_A}/{n_A} = {conv_A/n_A:.4f} conversion rate")
print(f"Treatment B: {conv_B}/{n_B} = {conv_B/n_B:.4f} conversion rate")
print(f"\nP(B > A) = {prob_B_better:.4f} ({prob_B_better*100:.1f}%)")
print(f"Expected relative lift: {expected_lift:.2f}%")
print(f"95% CI for lift: [{np.percentile(lift, 2.5):.2f}%, {np.percentile(lift, 97.5):.2f}%]")
print(f"Expected loss if we choose B: {loss_B:.6f}")
print(f"Expected loss if we choose A: {loss_A:.6f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Posterior distributions
theta = np.linspace(0.02, 0.08, 300)
axes[0].plot(theta, stats.beta.pdf(theta, alpha_A, beta_A), 'b-', lw=2.5, label=f'A: Beta({alpha_A},{beta_A})')
axes[0].plot(theta, stats.beta.pdf(theta, alpha_B, beta_B), 'r-', lw=2.5, label=f'B: Beta({alpha_B},{beta_B})')
axes[0].fill_between(theta, stats.beta.pdf(theta, alpha_A, beta_A), alpha=0.15, color='blue')
axes[0].fill_between(theta, stats.beta.pdf(theta, alpha_B, beta_B), alpha=0.15, color='red')
axes[0].set_title('Posterior Conversion Rate Distributions', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Conversion Rate')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Lift distribution
axes[1].hist(lift, bins=80, density=True, alpha=0.7, color='#4CAF50', edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--', lw=2, label='Zero lift')
axes[1].axvline(expected_lift, color='blue', linestyle='-', lw=2.5, label=f'Expected = {expected_lift:.1f}%')
ci_lift = np.percentile(lift, [2.5, 97.5])
axes[1].axvspan(ci_lift[0], ci_lift[1], alpha=0.1, color='green')
pct_positive = np.mean(lift > 0) * 100
axes[1].set_title(f'Lift Distribution\n{pct_positive:.1f}% probability lift > 0', fontweight='bold', fontsize=13)
axes[1].set_xlabel('Relative Lift (%)')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

# Decision gauge
labels = ['Choose A\n(lower risk)', 'Choose B\n(higher expected value)']
values = [loss_A * 10000, loss_B * 10000]
colors = ['#FF6384', '#4CAF50']
bars = axes[2].barh(labels, values, color=colors, edgecolor='white', alpha=0.8)
axes[2].set_title('Expected Loss (per 10,000 visitors)', fontweight='bold', fontsize=13)
axes[2].set_xlabel('Expected lost conversions')
for bar, val in zip(bars, values):
    axes[2].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}', va='center', fontweight='bold', fontsize=12)
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('bayesian_ab_test.png', dpi=150)
plt.show()
```

---

## 12. Project Ideas & What's Next

### Project Ideas

#### 🟢 Project 1: Bayesian Coin Estimator (Beginner)
Build an interactive tool that lets you flip a coin and watch the posterior update in real time. Start with different priors and see how fast each converges.

#### 🟡 Project 2: Bayesian A/B Testing Dashboard (Intermediate)
Build a complete Bayesian A/B testing framework with Streamlit:
- Input conversion data for two variants
- Compute posteriors, P(B > A), expected lift, risk metrics
- Interactive visualization of posterior evolution

#### 🔴 Project 3: MCMC Sampler Library (Advanced)
Implement three MCMC algorithms from scratch:
- Metropolis-Hastings
- Gibbs Sampling
- Hamiltonian Monte Carlo (HMC)
Compare convergence speed and efficiency on a multimodal posterior.

### What's Next

| Next Topic | Why |
|------------|-----|
| [Probability Distributions](./05-Probability-Distributions.md) | Deep dive into every distribution |
| [Data Visualization](./08-Data-Visualization-Mastery.md) | Communicate Bayesian results visually |
| [Feature Engineering](./07-Feature-Engineering.md) | Apply statistical thinking to features |

---

[← Statistical Inference](./03-Statistical-Inference.md) | [Back to Index](../README.md) | [Next: Probability Distributions →](./05-Probability-Distributions.md)
