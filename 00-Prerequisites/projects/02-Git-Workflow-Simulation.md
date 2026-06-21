# 🐙 Git Workflow Simulation Project

> **Difficulty**: ⭐⭐☆☆☆ Beginner

In this mini-project, you will simulate a real-world Git collaborative workflow used by Machine Learning engineering teams. You will create a local repository, build a feature branch for an ML model, simulate a realistic merge conflict (where a coworker modifies the same file), and resolve it manually.

## The Scenario

You are working on a team building a Python pipeline for predicting housing prices. 
- The `main` branch has the base code.
- You are tasked with creating a feature branch to add a **Random Forest** model.
- However, someone else on the team (simulated by you making a direct commit on `main`) changes the exact same line of code to add an **XGBoost** model while you were working.
- You must merge your branch, hit a conflict, and resolve it so both models exist in the final pipeline.

---

## Step 1: Initialize the Repository

Open your terminal and create a fresh directory for this simulation:

```bash
mkdir ml-git-simulation
cd ml-git-simulation
git init
```

Create a new file called `model_pipeline.py` and add some base code:

```bash
# Windows (PowerShell)
echo "def load_data(): return 'Dataset loaded'" > model_pipeline.py

# Linux/Mac
echo -e "def load_data():\n    return 'Dataset loaded'" > model_pipeline.py
```

Commit this initial code to the `main` branch:

```bash
git add model_pipeline.py
git commit -m "Initial commit: Add data loading function"
```

---

## Step 2: Create Your Feature Branch

You want to add your Random Forest model, so you create a branch.

**CRITICAL RULE**: Never code directly on `main`!

```bash
git checkout -b feature/random-forest
```

Open `model_pipeline.py` in your text editor (like VS Code) and add your model at the bottom:

```python
def load_data():
    return 'Dataset loaded'

def train_model():
    print("Training Random Forest model...")
    return "RandomForestRegressor"
```

Save the file and commit your changes to your feature branch:

```bash
git add model_pipeline.py
git commit -m "feat: Add Random Forest training logic"
```

---

## Step 3: Simulate Coworker Changes (Creating a Conflict)

Now, switch back to the `main` branch. 
*Imagine this is your coworker, Alice, pushing her code while you were working on your branch.*

```bash
git checkout main
```

Open `model_pipeline.py` again. Your Random Forest code is gone (because you are back on the stable `main` branch). 

Alice was working on an XGBoost model. Add her function to the bottom of the file:

```python
def load_data():
    return 'Dataset loaded'

def train_model():
    print("Training XGBoost model...")
    return "XGBRegressor"
```

Save and commit Alice's work to `main`:

```bash
git add model_pipeline.py
git commit -m "feat: Add XGBoost training logic (Alice)"
```

---

## Step 4: Merge and Resolve the Conflict

You are finished with your Random Forest branch and want to merge it into `main`.

```bash
# Make sure you are on main
git checkout main

# Attempt to merge your feature branch
git merge feature/random-forest
```

You will see a scary message like this:
```text
Auto-merging model_pipeline.py
CONFLICT (content): Merge conflict in model_pipeline.py
Automatic merge failed; fix conflicts and then commit the result.
```

**DON'T PANIC.** This is completely normal. 

Open `model_pipeline.py` in your editor. You will see Git's conflict markers:

```python
def load_data():
    return 'Dataset loaded'

<<<<<<< HEAD
def train_model():
    print("Training XGBoost model...")
    return "XGBRegressor"
=======
def train_model():
    print("Training Random Forest model...")
    return "RandomForestRegressor"
>>>>>>> feature/random-forest
```

**Resolution:**
Your team decides they want to train *both* models and compare them.
Manually delete the markers (`<<<<<<<`, `=======`, `>>>>>>>`) and edit the code so both models are present:

```python
def load_data():
    return 'Dataset loaded'

def train_rf_model():
    print("Training Random Forest model...")
    return "RandomForestRegressor"

def train_xgb_model():
    print("Training XGBoost model...")
    return "XGBRegressor"
```

---

## Step 5: Finalize the Merge

Now that the file is fixed, tell Git that the conflict is resolved and finish the merge:

```bash
git add model_pipeline.py
git commit -m "fix: Resolve merge conflict, keep both RF and XGBoost models"
```

You can now safely delete your feature branch to keep your repository clean:

```bash
git branch -d feature/random-forest
```

## 🎉 Congratulations!

You just executed a real-world Git workflow! This exact scenario happens daily in the ML industry. Knowing how to read and resolve conflict markers without panicking is a mandatory skill for any collaborative engineering role.
