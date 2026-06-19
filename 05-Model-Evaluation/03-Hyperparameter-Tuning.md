# 🎛️ Hyperparameter Tuning & Optimization

> **Prerequisites**: Cross-Validation, Calculus | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The Mathematical Objective](#1-the-mathematical-objective)
2. [Grid Search (Brute Force)](#2-grid-search-brute-force)
3. [Random Search (The Empirical Winner)](#3-random-search-the-empirical-winner)
4. [Successive Halving & Hyperband](#4-successive-halving--hyperband)
5. [Bayesian Optimization (Gaussian Processes)](#5-bayesian-optimization-gaussian-processes)
6. [TPE & Optuna Implementation](#6-tpe--optuna-implementation)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Mathematical Objective

Let $\mathcal{A}$ be a machine learning algorithm and $\boldsymbol{\lambda} \in \boldsymbol{\Lambda}$ be its hyperparameters. 
Let $V(L, \mathcal{A}_{\boldsymbol{\lambda}}, D_{train}, D_{valid})$ be the validation loss of the algorithm trained on $D_{train}$ with hyperparameters $\boldsymbol{\lambda}$.

The goal of hyperparameter tuning is to find:
$$\boldsymbol{\lambda}^* = \arg\min_{\boldsymbol{\lambda} \in \boldsymbol{\Lambda}} V(L, \mathcal{A}_{\boldsymbol{\lambda}}, D_{train}, D_{valid})$$

Evaluating $V$ is highly expensive because it requires a full model training and cross-validation cycle.

---

## 2. Grid Search (Brute Force)

Evaluates the Cartesian product of a finite set of hyperparameter values.
If you have 4 parameters, each with 5 values, and 5-fold CV, that requires $5^4 \times 5 = 3,125$ model trainings.

**The Curse of Dimensionality:** Grid search complexity grows exponentially with the number of hyperparameters $O(N^d)$.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=0, cv=5)
# grid.fit(X_train, y_train)
```

---

## 3. Random Search (The Empirical Winner)

Instead of a strict grid, Random Search samples from a statistical distribution. 

**Mathematical Justification (Bergstra & Bengio, 2012):**
If the optimal hyperparameter combination occupies at least 5% of the total hyperparameter space (volume), then randomly sampling 60 points gives a $1 - (1-0.05)^{60} \approx 95\%$ chance of finding a combination within the optimal region.

Random search is mathematically proven to be vastly more efficient than Grid Search when only a few hyperparameters actually matter (which is almost always true in Deep Learning).

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

param_dist = {
    'C': reciprocal(0.1, 1000), # Log-uniform distribution for scales
    'gamma': expon(scale=1.0),
    'kernel': ['rbf', 'poly', 'sigmoid']
}

random_search = RandomizedSearchCV(SVC(), param_distributions=param_dist, n_iter=60, cv=5)
# random_search.fit(X_train, y_train)
```

---

## 4. Successive Halving & Hyperband

To speed up Random Search, we shouldn't waste time training bad configurations for the full number of epochs (or on the full dataset).

**Successive Halving**:
1. Sample $N$ random hyperparameter configurations.
2. Train all $N$ models on a small subset of the data (or for a few epochs).
3. Throw away the worst-performing 50%.
4. Train the remaining models on twice as much data (or more epochs).
5. Repeat until 1 model remains on the full dataset.

```python
# Available in scikit-learn!
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

halving_search = HalvingRandomSearchCV(SVC(), param_dist, resource='n_samples', max_resources='auto')
```

---

## 5. Bayesian Optimization (Gaussian Processes)

Grid and Random search are **uninformed** — they don't use past results to decide the next guess.

**Bayesian Optimization** builds a probability model (Surrogate Model) of the objective function and uses it to select the most promising hyperparameters to evaluate next.

1. **Surrogate Model**: Usually a Gaussian Process (GP). It models the objective function $f(\boldsymbol{\lambda})$ as a multivariate normal distribution. It provides both a prediction $\mu$ and an uncertainty $\sigma$ for any hyperparameter combination.
2. **Acquisition Function**: Decides where to sample next by balancing **Exploitation** (picking points where $\mu$ is known to be good) and **Exploration** (picking points where $\sigma$ is high, meaning high uncertainty). Expected Improvement (EI) is standard.

Because the surrogate model is cheap to evaluate, we can search it millions of times to find the optimal next point to test the real, expensive model.

---

## 6. TPE & Optuna Implementation

Gaussian Processes scale poorly to high dimensions. The modern standard is **Tree-structured Parzen Estimator (TPE)**, which models $P(\boldsymbol{\lambda} | \text{Score})$ instead of $P(\text{Score} | \boldsymbol{\lambda})$.

**Optuna** uses TPE by default and allows dynamic, branchable search spaces (define loops, if-statements in the search).

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import logging

# Suppress massive output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    # Dynamic search space definition
    classifier_name = trial.suggest_categorical('classifier', ['RandomForest', 'SVC'])
    
    if classifier_name == 'RandomForest':
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32, log=True)
        model = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=100)
    else:
        svc_c = trial.suggest_float('svc_c', 1e-5, 1e5, log=True)
        model = SVC(C=svc_c, gamma='auto')
        
    # Standard CV evaluation
    # score = cross_val_score(model, X, y, cv=3).mean()
    score = 0.95 # Mock score
    return score

# TPE is the default sampler
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=50)

print(f"Best Trial: {study.best_trial.value}")
print(f"Best Params: {study.best_trial.params}")
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Implement Bayesian Optimization with Scikit-Optimize**: Use `skopt.BayesSearchCV` (which uses Gaussian Processes under the hood) to optimize an XGBoost model. Plot the GP surrogate model's predicted surface.
- 🔴 **Custom Acquisition Function**: Write a small Python script that implements the Expected Improvement math manually to guide a Random Forest surrogate model.

### What's Next
| Next | Why |
|------|-----|
| [Perceptron & MLP](../06-Neural-Networks-Foundations/01-Perceptron-And-MLP.md) | We leave classical Machine Learning behind. It's time for the math of Deep Learning. |

---

[← Cross Validation](./02-Cross-Validation.md) | [Back to Index](../README.md) | [Next: Bias Variance Tradeoff →](./04-Bias-Variance-Tradeoff.md)
