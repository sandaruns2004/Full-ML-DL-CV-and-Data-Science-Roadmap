# 💾 Data Versioning (DVC)

> **Prerequisites**: Git, MLOps | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Problem with Git and Large Files](#1-the-problem-with-git-and-large-files)
2. [Introducing DVC (Data Version Control)](#2-introducing-dvc-data-version-control)
3. [Core DVC Workflow](#3-core-dvc-workflow)
4. [DVC Pipelines for Reproducibility](#4-dvc-pipelines-for-reproducibility)
5. [Project Ideas & What's Next](#5-project-ideas--whats-next)

---

## 1. The Problem with Git and Large Files

Git is the undisputed king of version control for source code. However, Machine Learning projects have a fatal flaw when it comes to Git: **Data and Models are huge**.

If you commit a 5GB CSV dataset or a 2GB PyTorch `.pth` model to Git, your repository will bloat, cloning will take hours, and GitHub will flat-out reject your push (GitHub has a strict 100MB file limit).

**How do most people handle this?**
They ignore the data (`.gitignore`) and manually upload it to Google Drive or AWS S3. 
*The Result*: Six months later, you check out a specific Git commit of your code, but you have absolutely no idea which version of the dataset on S3 was used to train it. Reproducibility is dead.

---

## 2. Introducing DVC (Data Version Control)

**DVC** solves this by acting as a bridge between Git (for code) and Cloud Storage (for data).

**How it works**:
1. You tell DVC to track your massive `dataset.csv`.
2. DVC calculates a hash of the file and moves the actual 5GB file into a hidden cache folder (which is pushed to S3).
3. DVC generates a tiny text file called `dataset.csv.dvc` (containing just the hash).
4. You commit this tiny `.dvc` text file to Git!

Now, Git tracks the *pointers* to the data, and DVC handles the heavy lifting of moving the actual bytes to AWS S3, Google Cloud, or Azure.

---

## 3. Core DVC Workflow

Installing DVC: `pip install dvc`

### Step 1: Initialization
Inside your existing Git repository:
```bash
dvc init
git commit -m "Initialize DVC"
```

### Step 2: Configure Remote Storage
Tell DVC where to actually store the heavy files (e.g., an AWS S3 bucket).
```bash
dvc remote add -d myremote s3://mybucket/dvcstore
git add .dvc/config
git commit -m "Configure remote storage"
```

### Step 3: Track Data
Instead of `git add dataset.csv`, you use DVC:
```bash
dvc add data/dataset.csv
```
This generates `data/dataset.csv.dvc` and automatically adds `data/dataset.csv` to your `.gitignore`.

Now, commit the pointer to Git and push the data to S3!
```bash
git add data/dataset.csv.dvc data/.gitignore
git commit -m "Add dataset"

# Push code to GitHub
git push origin main

# Push heavy data to S3
dvc push
```

### Step 4: Retrieve Data (Time Travel)
If a coworker clones your Git repository, they won't have the data. They just run:
```bash
dvc pull
```
DVC reads the `.dvc` files in the current Git commit and downloads the exact matching dataset from S3!

---

## 4. DVC Pipelines for Reproducibility

DVC does more than just track data; it tracks **computation graphs** (Pipelines).

If you have a pipeline: `extract.py -> train.py -> evaluate.py`, you can define it in a `dvc.yaml` file.
You specify:
- **Dependencies**: The input data and python script.
- **Outputs**: The model file.
- **Command**: `python train.py`

When you run `dvc repro` (reproduce), DVC checks the hashes. If the input data and the code haven't changed, it instantly skips the training step. If they have changed, it reruns the necessary steps. This guarantees your models are perfectly reproducible from scratch.

---

## 5. Project Ideas & What's Next

### Project Ideas
- 🟢 **Local DVC Tracking**: Create a new git repository. Generate a fake 50MB CSV file. Initialize DVC and configure a "local remote" (just a different folder on your hard drive). Track the CSV, change some rows, track it again, and practice checking out old Git commits and using `dvc pull` to watch the CSV file revert to its older state!

### What's Next
| Next | Why |
|------|-----|
| [ML Projects Phase](../16-Projects/01-End-To-End-Projects.md) | You've learned the math, the architectures, and the deployment tools. It's time to build full end-to-end projects. |

---

[← Distributed Training](07-Distributed-Training.md) | [Back to Index](../README.md) | [Next: Beginner Projects →](../16-Projects/01-Beginner-Projects.md)
