# 🎛️ Hyperparameter Evaluation

> **Prerequisites**: Validation Curves | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 📋 Table of Contents
1. [Grid Search vs Random Search](#1-grid-search-vs-random-search)
2. [Bayesian Optimization](#2-bayesian-optimization)

---

## 1. Grid Search vs Random Search

### 🟢 Beginner
**Simple Explanation**: 
You have a model with 5 different dials (hyperparameters) you can turn. 
- **Grid Search**: You systematically try every single possible combination of the dials. It takes forever.
- **Random Search**: You just spin the dials randomly 100 times and pick the best one. Surprisingly, this usually works better and faster!

### 🟡 Intermediate
**Workflow and Implementation**: 

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Grid Search (3x3x3 = 27 combinations * 5 folds = 135 model fits)
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best Params:", grid_search.best_params_)

# Random Search (Tests 10 random combinations * 5 folds = 50 model fits)
from scipy.stats import randint
param_dist = {'n_estimators': randint(50, 200), 'max_depth': randint(5, 20)}
random_search = RandomizedSearchCV(model, param_dist, n_iter=10, cv=5)
```

### 🔴 Advanced
**Bayesian Optimization**:
Grid and Random search are "dumb"—they do not learn from their previous attempts. Bayesian Optimization (e.g., using Hyperopt or Optuna) uses a probabilistic model (usually a Gaussian Process) to model the mapping from hyperparameters to the evaluation score. 
It balances **Exploration** (trying entirely new areas of the hyperparameter space) with **Exploitation** (fine-tuning near the best parameters found so far). In industry scale deep-learning models where training takes weeks, Bayesian Optimization is mandatory.
