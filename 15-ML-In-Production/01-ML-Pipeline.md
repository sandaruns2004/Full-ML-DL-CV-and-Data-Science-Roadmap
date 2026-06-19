# ⚙️ ML Pipelines & Data Versioning

> **Prerequisites**: Python, Git | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The "Notebook" Problem](#1-the-notebook-problem)
2. [What is an ML Pipeline?](#2-what-is-an-ml-pipeline)
3. [Data & Model Versioning (DVC)](#3-data--model-versioning-dvc)
4. [Feature Stores](#4-feature-stores)
5. [Orchestration (Airflow / Prefect)](#5-orchestration-airflow--prefect)
6. [Library Implementation (DVC Basics)](#6-library-implementation-dvc-basics)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The "Notebook" Problem

A standard beginner workflow:
1. Open `analysis_final_v3.ipynb`.
2. Run cells out of order to clean data.
3. Train a Random Forest.
4. Save it as `model_good.pkl`.
5. Email it to a software engineer to deploy.

**This is a nightmare for production software.** 
Six months later, when the model starts failing, no one remembers which specific CSV file was used, which preprocessing steps were skipped, or what `v3` of the notebook actually did. The model is completely unreproducible.

In software engineering, if code breaks, you revert the Git commit. In ML, you must version the Code, the Data, AND the Model weights.

---

## 2. What is an ML Pipeline?

An ML Pipeline transforms a messy Jupyter Notebook into a rigid, automated, step-by-step DAG (Directed Acyclic Graph) of Python scripts.

A standard pipeline has 4 isolated stages:
1. `extract.py`: Connects to the database and dumps raw data.
2. `preprocess.py`: Cleans missing values, scales features, and saves `train.csv`.
3. `train.py`: Loads `train.csv`, trains the model, and saves `model.pkl`.
4. `evaluate.py`: Loads the model, tests it against a holdout set, and outputs `metrics.json`.

If any step fails, the pipeline halts. Because the steps are isolated scripts, they can be run perfectly on any machine, triggered automatically on a schedule.

---

## 3. Data & Model Versioning (DVC)

Git is built for text files (code). If you try to `git commit` a 10GB CSV file or a 2GB `model.pt` file, GitHub will instantly reject it.

**Data Version Control (DVC)** solves this. It acts exactly like Git, but for large files.
1. You run `dvc add data.csv`.
2. DVC moves the actual 10GB file to a cloud storage bucket (AWS S3, Google Drive).
3. DVC creates a tiny text file called `data.csv.dvc` containing an MD5 hash pointer to the cloud file.
4. You commit `data.csv.dvc` to GitHub.

Now, your Git repository tracks exactly which version of the dataset was used to train the code in a specific commit. If you checkout an old commit, DVC automatically downloads the correct old dataset!

---

## 4. Feature Stores

In a large company, the Data Science team might write a massive SQL query to calculate "User's Average Purchase Value over 30 days" to use as a feature.
A month later, the Recommender Systems team writes a *slightly different* query to calculate the exact same thing. 
When the model is deployed, the backend engineers write a *third* query in Java to fetch that feature in real-time.

This causes **Training-Serving Skew** (the feature looks different during training vs production), causing catastrophic silent failures.

A **Feature Store** (like Feast or Hopsworks) is a centralized database of pre-computed ML features.
- Data Scientists fetch features from the "Offline Store" (data warehouse) for training.
- Production systems fetch the exact same features from the "Online Store" (Redis/low-latency DB) for real-time inference.
- Everyone uses the exact same definition!

---

## 5. Orchestration (Airflow / Prefect)

How do you automate the pipeline? You don't want to run `python train.py` manually every Tuesday.

**Orchestrators** (like Apache Airflow, Prefect, or Dagster) are the central nervous systems of data engineering.
You define your pipeline using Python code:

```python
# Airflow DAG definition
extract_task >> preprocess_task >> train_task >> evaluate_task
```

The Orchestrator ensures:
- The pipeline runs every night at 2:00 AM.
- If `preprocess_task` fails, it automatically retries 3 times.
- If it fails 3 times, it sends an alert to the Slack channel.
- `train_task` will absolutely not start until `preprocess_task` is successfully completed.

---

## 6. Library Implementation (DVC Basics)

Here is a typical terminal workflow showing how Git and DVC work together seamlessly.

```bash
# 1. Initialize Git and DVC in your project folder
git init
dvc init

# 2. Download your massive dataset
wget https://example.com/massive_dataset.csv -O data/raw_data.csv

# 3. Add the massive data to DVC (NOT Git)
dvc add data/raw_data.csv
# This creates a file: data/raw_data.csv.dvc

# 4. Tell DVC where to store the actual heavy files (e.g., an AWS S3 bucket)
dvc remote add -d myremote s3://my-company-ml-bucket/project-data

# 5. Push the heavy data to the cloud
dvc push

# 6. Commit the tiny pointer file to Git
git add data/raw_data.csv.dvc .gitignore
git commit -m "Added version 1 of the raw dataset"
git push origin main

# --- 6 Months Later on a New Computer ---
git clone https://github.com/my-repo.git
# The raw_data.csv file is NOT there. Only the .dvc pointer is there.
# To get the data:
dvc pull
# DVC reads the pointer and downloads the exact right dataset from S3!
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **DVC Pipeline Construction**: DVC isn't just for data; it's a pipeline runner. Use `dvc stage add` to connect 3 python scripts (`process.py`, `train.py`, `eval.py`). Define their dependencies (`-d data.csv`) and outputs (`-o model.pkl`). Run `dvc repro` to watch DVC automatically figure out which stages need to be re-run based on what files changed!
- 🟡 **Airflow Locally**: Install Apache Airflow locally using Docker. Write a simple DAG that downloads the daily weather data from an API, parses it, and saves it to a local SQLite database. Schedule it to run every 5 minutes and watch the Airflow UI light up green as tasks succeed.

### What's Next
| Next | Why |
|------|-----|
| [Model Deployment](./02-Model-Deployment.md) | The pipeline successfully trained the model. Now, how do we put it on the internet so an iOS app can actually talk to it and get predictions? We need to wrap it in a REST API. |

---

[← Data Ethics Fairness](../14-DS-Advanced/06-Data-Ethics-Fairness.md) | [Back to Index](../README.md) | [Next: Model Deployment →](./02-Model-Deployment.md)
