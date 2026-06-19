# 🔄 MLOps & CI/CD Pipelines

> **Prerequisites**: Git, ML Pipeline, Docker | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [What is MLOps?](#1-what-is-mlops)
2. [The CI/CD/CT Lifecycle](#2-the-cicdct-lifecycle)
3. [Continuous Integration (CI) for ML](#3-continuous-integration-ci-for-ml)
4. [Continuous Delivery/Deployment (CD)](#4-continuous-deliverydeployment-cd)
5. [Continuous Training (CT)](#5-continuous-training-ct)
6. [Library Implementation (GitHub Actions)](#6-library-implementation-github-actions)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. What is MLOps?

In traditional software engineering, **DevOps** is the practice of combining Development (writing code) and IT Operations (managing servers) to shorten the software development lifecycle and provide continuous delivery with high software quality.

**MLOps (Machine Learning Operations)** is DevOps applied to Machine Learning. 

Machine Learning systems are significantly more complex than standard software because standard software only depends on **Code**. Machine Learning depends on **Code, Data, and Models**. 
If the underlying data distribution changes in the real world (e.g., a global pandemic changes shopping habits), your model breaks silently, even if the code is perfectly bug-free.

MLOps is the set of automated practices that ensure ML models can be reliably built, deployed, monitored, and retrained in production.

---

## 2. The CI/CD/CT Lifecycle

A mature MLOps pipeline implements three continuous processes:

1. **CI (Continuous Integration)**: Testing and validating code and data.
2. **CD (Continuous Deployment)**: Automatically packaging the model and pushing it to production servers.
3. **CT (Continuous Training)**: Automatically triggering a model retraining loop when new data arrives or model performance decays.

---

## 3. Continuous Integration (CI) for ML

Imagine a team of 5 Data Scientists working on the same recommendation engine. One scientist pushes a change to the `feature_engineering.py` script to GitHub.

Before that code is allowed to be merged into the `main` branch, a **CI Server** (like GitHub Actions or Jenkins) automatically spins up a virtual machine and runs automated tests:
1. **Unit Tests**: Checks if `feature_engineering.py` runs without crashing on a dummy dataset.
2. **Data Validation Tests**: Checks if the new data schema matches the expected schema (e.g., using Great Expectations).
3. **Model Tests**: The CI server trains a small version of the model for 1 epoch just to ensure the loss goes down and backpropagation works.

If any of these tests fail, the GitHub pull request is blocked, preventing broken code from entering the main codebase.

---

## 4. Continuous Delivery/Deployment (CD)

Once the code passes the CI tests and is merged into the `main` branch, the **CD Pipeline** kicks in automatically.

1. **Full Training**: The CD server downloads the entire production dataset (using DVC), trains the massive model, and saves `model.pkl`.
2. **Model Registry Validation**: The new model is tested against a static holdout test set. If its accuracy is *worse* than the currently deployed model, the CD pipeline aborts.
3. **Containerization**: If the model is better, the CD server runs `docker build` to package the model, the FastAPI code, and the Python dependencies into a Docker Image.
4. **Deployment**: The CD server automatically pushes the Docker Image to the cloud (e.g., AWS Elastic Container Registry) and updates the Kubernetes cluster to start routing user traffic to the new model container.

*(All of this happens without a human touching a single terminal command!)*

---

## 5. Continuous Training (CT)

Models degrade over time. A fraud detection model trained in 2021 knows nothing about crypto scams invented in 2024. This is called **Concept Drift**.

Instead of waiting for an engineer to notice the model is failing and manually trigger a new CD pipeline, mature organizations use **Continuous Training**.

A CT trigger could be:
- **Time-based**: Retrain the model every Sunday night at 2:00 AM using the past 7 days of new user data.
- **Performance-based**: The Model Monitoring system (discussed in the next file) detects that accuracy dropped below 85%, which fires a webhook that automatically starts the training pipeline.

---

## 6. Library Implementation (GitHub Actions)

**GitHub Actions** is the easiest way to start with CI/CD. You write a YAML file in the `.github/workflows/` directory of your repository. 

Here is a simple CI pipeline that runs every time someone pushes code to the `main` branch. It sets up Python, installs dependencies, and runs `pytest`.

```yaml
# File: .github/workflows/ml_ci.yml
name: Machine Learning CI Pipeline

# Trigger the workflow on push to the main branch
on:
  push:
    branches:
      - main

jobs:
  test-model-code:
    runs-on: ubuntu-latest
    
    steps:
      # 1. Download the repository code onto the runner
      - name: Checkout Code
        uses: actions/checkout@v4

      # 2. Setup Python environment
      - name: Set up Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      # 3. Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest

      # 4. Run automated unit tests
      # E.g., verifying that the model's output shape is correct
      - name: Run PyTest
        run: |
          pytest tests/test_model_architecture.py
          
      # 5. Run a tiny test training loop (1 epoch, 10 samples)
      - name: Verify Training Loop
        run: |
          python train.py --epochs 1 --test_mode True
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Automated Model Testing**: Create a simple GitHub repository. Write a PyTorch model and a script `test_model.py` that uses the `unittest` framework to assert that passing a tensor of shape `(32, 3, 224, 224)` returns a tensor of shape `(32, 10)`. Set up a GitHub Action to automatically run this test on every push.
- 🟡 **CML (Continuous Machine Learning)**: Use the Iterative.ai `cml` tool inside GitHub Actions. Write a workflow that actually trains a Random Forest model on the cloud runner, generates an AUC-ROC curve image using `matplotlib`, and posts the image automatically as a comment on your Pull Request so your team can review the model's performance before merging!

### What's Next
| Next | Why |
|------|-----|
| [Model Monitoring](./04-Model-Monitoring.md) | We deployed the model successfully via CI/CD. Now it is live in production. But how do we know if it's silently failing? We need to set up Dashboards, Data Drift detection, and Model Monitoring. |

---

[← Model Deployment](./02-Model-Deployment.md) | [Back to Index](../README.md) | [Next: Model Monitoring →](./04-Model-Monitoring.md)
