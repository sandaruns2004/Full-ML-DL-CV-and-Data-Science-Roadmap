# 🚀 The Ultimate Machine Learning, DL & Data Science Roadmap
> **A world-class, comprehensive, math-heavy, and code-rich learning path taking you from absolute beginner to production-ready AI Engineer.**

Welcome to the definitive guide for mastering Machine Learning, Deep Learning, Computer Vision, NLP, LLMs, and MLOps. This repository is meticulously designed to bridge the gap between academic theory and industry-grade production AI.

### 🎯 Who is this for?
- **Self-Taught Learners** seeking a structured, step-by-step curriculum.
- **Students & Undergraduates** looking for deep mathematical intuition alongside practical code.
- **Data Scientists & ML Engineers** transitioning into Deep Learning, LLMs, and MLOps.
- **AI Researchers** who want a quick reference for foundational models and math.

### 🌟 Learning Outcomes
By the end of this roadmap, you will be able to:
- Build, train, and deploy robust ML models from scratch.
- Architect Deep Learning and Computer Vision systems using PyTorch.
- Develop advanced NLP applications using Transformers and LLMs.
- Construct production-ready RAG (Retrieval-Augmented Generation) pipelines.
- Deploy scalable AI systems utilizing modern MLOps practices.

---

## 🗺️ Visual Overview & Architecture

```mermaid
flowchart TD
    A[00 - Prerequisites: Python & Math] --> B[01 - Data Science & EDA]
    B --> C[02-05 Classical Machine Learning]
    C --> D[06 - Neural Networks Foundations]
    
    D --> E[07/11 - Computer Vision & CNNs]
    D --> F[12 - NLP]
    
    F --> G[08/09 - Sequence Models & Transformers]
    E --> G
    
    G --> H[10 - Generative AI & Diffusion]
    G --> I[11/12 - LLMs & RAG Pipelines]
    
    H --> J[13/14 - Advanced Topics & RL]
    I --> J
    
    J --> K[15 - MLOps & Production AI Systems]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style K fill:#bbf,stroke:#333,stroke-width:2px
```

---

## 📅 The Learning Roadmap

A structured progression estimating **~500 Hours** of total learning.

| Stage | Topic | Difficulty | Estimated Hours | Description |
| ----- | ----- | ---------- | --------------- | ----------- |
| `00` | Prerequisites | ⭐☆☆☆☆ | 20 | Python Essentials, Linear Algebra, Probability, Calculus |
| `01` | Data Science Foundations | ⭐☆☆☆☆ | 30 | EDA, Visualization, Statistical Inference, Preprocessing |
| `02-05` | Classical ML | ⭐⭐☆☆☆ | 60 | Supervised Learning, **Unsupervised Learning (Clustering, PCA, Apriori)**, Ensembles, Evaluation |
| `06` | Neural Network Foundations | ⭐⭐⭐☆☆ | 40 | Backprop, Optimizers, PyTorch Fundamentals |
| `07` & `11` | Computer Vision | ⭐⭐⭐☆☆ | 50 | CNNs, Object Detection, Segmentation, OpenCV |
| `08` & `12` | NLP Basics | ⭐⭐⭐☆☆ | 40 | Embeddings, Text Classification, RNNs/LSTMs |
| `09` | Transformers | ⭐⭐⭐⭐☆ | 50 | Attention Mechanisms, BERT, GPT, ViT |
| `10` | Generative AI | ⭐⭐⭐⭐☆ | 40 | GANs, VAEs, Diffusion Models |
| `11` | Modern LLMs & RAG | ⭐⭐⭐⭐⭐ | 60 | Fine-Tuning, RLHF, Vector Databases |
| `13` & `14` | Advanced & Distributed | ⭐⭐⭐⭐⭐ | 50 | Reinforcement Learning, Graph Neural Networks, Big Data |
| `15` | MLOps & Production | ⭐⭐⭐⭐⭐ | 60 | Pipelines, CI/CD, Deployment, Docker, Edge ML |
| `16` | Projects | 🎓 | On-going | Applying all skills to end-to-end applications |

---

## 📂 Full Directory Structure & Module Deep Dives

Here is an in-depth look at our newly upgraded modules, such as the Data Science Foundations and Unsupervised Learning sections:

### 00-Prerequisites (Absolute Beginner)
A world-class beginner-to-professional bridge course covering all core foundations before writing machine learning algorithms:
*   `01-Introduction-To-AI-Ecosystem.md` (AI vs ML vs DL vs DS, MLOps)
*   `02-Computer-Fundamentals.md` (CPU, GPU, RAM, VRAM, Filesystems, OS)
*   `03-Programming-Fundamentals.md` (Variables, Loops, Functions, OOP)
*   `04-Development-Environment-Setup.md` (Miniconda, VS Code, Jupyter, Git)
*   `05-Python-Essentials.md` (NumPy, Pandas, Matplotlib, Error Handling)
*   `06-Git-And-GitHub.md` (Version control, Branches, Merging, `.gitignore`)
*   `07-Linux-Fundamentals.md` (Terminal, File Manipulation, Process Management)
*   `08-Mathematical-Foundations.md` (Linear Algebra, Calculus, Matrices)
*   `09-Probability-And-Statistics.md` (Bayes Theorem, Probability Distributions)
*   `10-SQL-For-Data-Science.md` (Basic queries, Aggregations)
*   `projects/` (Data Analysis Starter Notebook, Git Workflow Simulation, Env Validation Script, Linux Command Toolkit)

### 01-Data-Science-Foundations (Beginner → Advanced)
A complete, 16-part curriculum covering the entire data science lifecycle, from data collection to ethics:
*   `01-Introduction-To-Data-Science.md` (Lifecycle, CRISP-DM, Roles)
*   `02-Python-For-Data-Science.md` (Environments, Generators, Vectorization)
*   `03-NumPy.md` (Arrays, Broadcasting, Vectorization)
*   `04-Pandas.md` (DataFrames, GroupBy, Merging)
*   `05-Data-Collection.md` (APIs, Web Scraping, SQL connections)
*   `06-Data-Cleaning.md` (Missing values, Duplicates, Outlier Detection)
*   `07-Exploratory-Data-Analysis.md` (Univariate, Bivariate, Automated Profiling)
*   `08-Descriptive-Statistics.md` (Dispersion, Skewness, Kurtosis)
*   `09-Probability-And-Distributions.md` (Normal, Binomial, Poisson, CLT)
*   `10-Inferential-Statistics.md` (Hypothesis Testing, p-values, A/B Testing)
*   `11-Bayesian-Statistics.md` (Priors, Posteriors, Naive Bayes, MCMC)
*   `12-Data-Visualization.md` (Matplotlib, Seaborn, Plotly, Storytelling)
*   `13-Data-Preprocessing.md` (Encoding, Scaling, ML Pipelines)
*   `14-Feature-Engineering.md` (Polynomials, Binning, TF-IDF)
*   `15-SQL-For-Data-Science.md` (Aggregations, JOINs, Window Functions)
*   `16-Data-Ethics.md` (Algorithmic Bias, GDPR, Responsible AI)
*   `17-Feature-Selection.md` (Filter, Wrapper, Embedded methods)
*   `18-Imbalanced-Data.md` (Accuracy Paradox, SMOTE, Class Weights)
*   `notebooks/` (Interactive Jupyter notebooks for hands-on theory application)
*   `projects/` (5 mandatory hands-on projects including Sales Data Analysis and End-to-End EDA)

### 02-Supervised-Learning (Beginner → Advanced)
A complete, math-heavy, and code-rich breakdown of Supervised Machine Learning algorithms:
*   `01-Introduction-To-Supervised-Learning.md` (Workflow, Regression vs Classification)
*   `02-Linear-Regression.md` (Cost Function, Gradient Descent)
*   `03-Polynomial-Regression.md` (Non-linear Relationships, Degree Selection)
*   `04-Logistic-Regression.md` (Sigmoid, Binary vs Multiclass)
*   `05-KNN.md` (Distance Metrics, K Selection)
*   `06-Decision-Trees.md` (Entropy, Information Gain, Gini)
*   `07-Support-Vector-Machines.md` (Linear, Kernel Trick)
*   `08-Naive-Bayes.md` (Gaussian, Multinomial, Bernoulli)
*   `09-Feature-Engineering-For-Supervised-Learning.md` (Encoding, Scaling, Missing Values)
*   `10-Regularization.md` (Ridge, Lasso, ElasticNet, Overfitting)
*   `11-Model-Building-Pipeline.md` (Scikit-Learn Pipelines, End-to-End Workflows)
*   `12-Model-Selection-Guide.md` (How to choose an algorithm)
*   `notebooks/` (Interactive Jupyter notebooks for hands-on application)
*   `projects/` (6 mandatory projects including House Price and Customer Churn Prediction)

### 03-Ensemble-Methods (Beginner → Advanced)
A complete, math-heavy, and code-rich breakdown of Ensemble Learning algorithms:
*   `01-Introduction-To-Ensemble-Learning.md` (Wisdom of Crowds, Bias-Variance)
*   `02-Bagging.md` (Bootstrap Sampling, Parallel Learning)
*   `03-Random-Forest.md` (Feature Randomness, Feature Importance)
*   `04-Extra-Trees.md` (Extremely Randomized Trees)
*   `05-Voting-Classifiers.md` (Hard vs Soft Voting)
*   `06-Boosting-Introduction.md` (Sequential Improvement)
*   `07-AdaBoost.md` (Sample Weights, Weak Learners)
*   `08-Gradient-Boosting.md` (Residual Learning)
*   `09-XGBoost-Concepts.md` (Regularization, Tree Pruning)
*   `10-LightGBM-Concepts.md` (Histogram-Based, Leaf-Wise Growth)
*   `11-CatBoost-Concepts.md` (Categorical Features, Ordered Boosting)
*   `12-Stacking.md` (Meta Learners, Cross-Validation Strategy)
*   `13-Blending.md` (Holdout-Based Ensembling)
*   `14-Model-Selection-For-Ensembles.md` (Diversity vs Performance Tradeoffs)
*   `notebooks/` (Interactive Jupyter notebooks for hands-on application)
*   `projects/` (5 mandatory projects including Customer Churn and House Price Prediction)

### 04-Unsupervised-Learning (Beginner → Advanced)
A complete, math-heavy, and code-rich breakdown of Unsupervised techniques:
*   `01-Introduction-To-Unsupervised-Learning.md` (Hidden Patterns, Dimensionality)
*   `02-K-Means-Clustering.md` (Centroids, Elbow Method, Silhouette Score)
*   `03-Hierarchical-Clustering.md` (Agglomerative, Divisive, Dendrograms)
*   `04-DBSCAN.md` (Density-based, Epsilon, Noise Detection)
*   `05-Mean-Shift.md` (Kernel Density Estimation, Mode Seeking)
*   `06-Gaussian-Mixture-Models.md` (Soft Clustering, Expectation Maximization)
*   `07-Principal-Component-Analysis.md` (Variance, Eigenvectors, Derivation)
*   `08-tSNE.md` (High-Dimensional Visualization, Similarity Preservation)
*   `09-UMAP-Introduction.md` (Manifold Learning, Speed comparison)
*   `10-Association-Rule-Mining.md` (Market Basket, Support, Confidence, Lift)
*   `11-Apriori-Algorithm.md` (Frequent Itemsets, Rule Generation)
*   `12-Anomaly-Detection.md` (Statistical, Distance, Density Methods)
*   `13-Isolation-Forest.md` (Tree-Based Isolation)
*   `14-Local-Outlier-Factor.md` (Local Density Analysis)
*   `notebooks/` (Interactive Jupyter notebooks for clustering and embeddings)
*   `projects/` (6 mandatory projects including Customer Segmentation and Market Basket Analysis)

### 05-Model-Evaluation (Beginner → Advanced)
A complete, math-heavy, and code-rich breakdown of Model Evaluation and MLOps metrics:
*   `01-Introduction-To-Model-Evaluation.md` (Why evaluate, Data Leakage)
*   `02-Train-Test-Validation-Split.md` (Holdout sets, Statistical Independence)
*   `03-Bias-Variance-Tradeoff.md` (Overfitting vs Underfitting, Math decomposition)
*   `04-Regression-Metrics.md` (MAE, MSE, RMSE, R², Adjusted R²)
*   `05-Classification-Metrics.md` (Accuracy, Precision, Recall, F1-Score)
*   `06-Confusion-Matrix.md` (TP, FP, TN, FN, Multi-class)
*   `07-ROC-And-AUC.md` (Threshold Analysis, Probabilities)
*   `08-Precision-Recall-Curves.md` (Handling extreme imbalances)
*   `09-Cross-Validation.md` (K-Fold, Stratified, LOOCV)
*   `10-Learning-Curves.md` (Diagnosing model health via data volume)
*   `11-Validation-Curves.md` (Finding optimal hyperparameters visually)
*   `12-Hyperparameter-Tuning-Evaluation.md` (Grid Search, Random Search, Bayesian Optimization)
*   `13-Imbalanced-Classification.md` (Accuracy Paradox, Balanced Accuracy)
*   `14-Model-Comparison.md` (Statistical Significance, Student's t-test)
*   `15-Production-Monitoring.md` (Data Drift, Concept Drift, MLOps)
*   `16-Interpretability-Explainability.md` (SHAP, LIME, local/global explainability)
*   `notebooks/` (Interactive Jupyter notebooks with runnable evaluation metrics)
*   `projects/` (5 mandatory projects including Hyperparameter Labs and Production Monitoring Simulators)

*(More directory structures will be mapped out as they are upgraded!)*

---

## ✨ Why This Repository is Different

- 🧱 **Structured Progression**: No more jumping between random tutorials. A clear A-to-Z learning path.
- 📐 **Math meets Code**: We don't skip the math. We explain it visually and then implement it from scratch in NumPy.
- 💼 **Industry-Focused**: Focuses heavily on what actually matters in modern tech (MLOps, Deployments, LLMs).
- 🛠️ **Hands-On Projects**: Move from theory to practice with carefully scoped real-world projects.
- 📝 **Cheat Sheets & Notes**: Quick reference guides for algorithms, metrics, and deep learning architectures.
- 🤝 **Open-Source Friendly**: Built by the community, for the community.

---

## 🚀 Skills You Will Gain

After completing this repository you will be able to:
- ✅ **Build ML models**: From Linear Regression to Gradient Boosting Trees and Unsupervised clustering.
- ✅ **Train Deep Learning systems**: Architect custom Neural Networks in PyTorch.
- ✅ **Build Computer Vision applications**: Implement Real-time Object Tracking and Segmentation.
- ✅ **Build NLP applications**: Sentiment Analysis, Named Entity Recognition, and Translation.
- ✅ **Use Transformers**: Master Self-Attention, BERT, and GPT architectures.
- ✅ **Build RAG systems**: Combine Vector Databases with LLMs for intelligent retrieval.
- ✅ **Deploy AI systems**: Dockerize applications and expose REST APIs using FastAPI.
- ✅ **Create production ML pipelines**: Version data with DVC and track experiments.

---

## 🛠️ Project Showcase

Applying knowledge is the fastest way to learn. Each concept is paired with an end-of-module mini-project.

| Project | Difficulty | Skills Learned |
| ------- | ---------- | -------------- |
| **Real Estate Price Predictor** | Beginner | Pandas, EDA, Feature Engineering, Random Forest |
| **Customer Segmentation Engine** | Beginner | Unsupervised Learning, K-Means |
| **Market Basket Analysis** | Beginner | Apriori, Association Rules |
| **Image Compression Pipeline** | Intermediate | PCA, Eigen-decomposition |
| **Credit Card Fraud Detection** | Intermediate | Isolation Forest, LOF, Anomaly Detection |
| **Sensor Anomaly Detection** | Intermediate | Time Series, Z-Score |
| **Dimensionality Reduction Pipeline** | Intermediate | PCA + t-SNE Pipeline |
| **Pneumonia Detection from X-Rays** | Intermediate | PyTorch, CNNs, Transfer Learning (ResNet), Medical AI |
| **Real-Time Object Tracking with YOLO** | Intermediate | Computer Vision, OpenCV, YOLOv8 |
| **FastAPI Credit Scoring Engine** | Intermediate | XGBoost, Model Deployment, REST APIs, Docker |

*(Recommended Future Projects: Multi-Modal Chatbots, Distributed Reinforcement Learning Agents, Real-time Streaming Recommendation Engines).*

---

## 📓 Interactive Notebook Showcase

Hands-on Jupyter Notebooks to explore concepts interactively:

| Notebook | Topic | Status |
| -------- | ----- | ------ |
| `Deep-Learning-From-Scratch.ipynb` | Neural Networks | ✅ Available |
| `PyTorch-Transformer-Attention.ipynb` | Transformers | ✅ Available |
| `RAG-Pipeline-Project.ipynb` | LLMs & Vector DBs | ✅ Available |

*(Planned Interactive Notebooks: Diffusion Models from Scratch, Multi-GPU Distributed Training, MLOps CI/CD pipelines).*

---

## ⚡ Quick Start & Setup Instructions

Get your environment set up in less than 2 minutes so you can start running the code examples immediately!

### 1. Clone the repository
```bash
git clone https://github.com/sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap.git
cd Full-ML-DL-CV-and-Data-Science-Roadmap
```

### 2. Create a virtual environment
It is highly recommended to use a virtual environment to isolate dependencies.
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install requirements
Our `requirements.txt` is categorized by module (ML, DL, NLP, etc.). Ensure you are in the root directory:
```bash
pip install -r requirements.txt
```
*(Note: If you run into issues installing PyTorch, please visit the [PyTorch website](https://pytorch.org/get-started/locally/) for specific commands for your OS/CUDA version).*

### 4. How to Run Examples
All Python blocks inside the markdown files are completely standalone and runnable! 
You can copy-paste them into a Python script (`test.py`) or into a Jupyter Notebook cell and they will execute, generate synthetic data, train the model, and save visualizations to your local directory.

Navigation Guide: Start chronologically in the `00-Prerequisites` folder, and read through the beautifully formatted markdown files. Run any associated Jupyter notebooks in the `Jupyter Notebooks` folder when referenced.

---

## 🤝 Contributor Experience

We welcome contributions from everyone! Whether you're fixing a typo, adding a new project, or expanding a mathematical proof.

- **Contribution Guidelines**: Please read our `CONTRIBUTING.md` (coming soon) before submitting a PR.
- **Repository Standards**: Follow consistent directory structures and descriptive file naming.
- **Documentation Standards**: Use clear, concise language. Include LaTeX for mathematical formulas.
- **Pull Request Process**: Fork the repo, create a feature branch (`feat/new-notebook`), and submit a PR with a detailed description.
- **Issue Reporting**: Found a bug or have a suggestion? Open an issue using our provided templates, detailing the problem and expected behavior.

---

## ❓ FAQ

<details>
<summary><b>1. Do I need mathematics first?</b></summary>
While you can run code without math, understanding the "why" requires it. Phase 00 covers all the necessary Linear Algebra, Calculus, and Probability you will need.
</details>

<details>
<summary><b>2. Can beginners use this roadmap?</b></summary>
Absolutely. Phase 00 and 01 are specifically designed for absolute beginners. We build intuition from the ground up.
</details>

<details>
<summary><b>3. How long does it take?</b></summary>
If you study 10-15 hours a week, expect it to take around 6 to 9 months to complete thoroughly.
</details>

<details>
<summary><b>4. Which projects should I build first?</b></summary>
Start with the "Real Estate Price Predictor" in the Beginner Projects folder to master Pandas, EDA, and Scikit-Learn.
</details>

<details>
<summary><b>5. Is this enough for a job?</b></summary>
Yes. This curriculum covers more ground than most Master's degree programs, especially in MLOps and Production AI (Phase 15), which recruiters highly value.
</details>

<details>
<summary><b>6. Do I need a powerful GPU?</b></summary>
For Phases 00-05, a standard laptop is fine. For Deep Learning (Phases 06+), you can use free cloud GPUs like Google Colab or Kaggle Notebooks.
</details>

<details>
<summary><b>7. Is this curriculum updated for modern architectures?</b></summary>
Yes! The curriculum includes modern paradigms like Vision Transformers (ViT), Large Language Models (LLMs), RAG pipelines, and Diffusion models.
</details>

<details>
<summary><b>8. What programming language is used?</b></summary>
Python is the exclusive language used throughout the roadmap.
</details>

<details>
<summary><b>9. Should I learn TensorFlow or PyTorch?</b></summary>
We focus primarily on PyTorch due to its dominance in research and modern industry applications, though concepts apply to both.
</details>

<details>
<summary><b>10. How is this different from generic online courses?</b></summary>
This roadmap provides a broader, more modern scope (including MLOps and LLMs) while remaining entirely text-based and open-source.
</details>

<details>
<summary><b>11. Where do I find the datasets?</b></summary>
All datasets used are open-source and linked directly within the specific module or project file.
</details>

<details>
<summary><b>12. How do I track my progress?</b></summary>
Fork this repository and use GitHub checkmarks (`[x]`) in your own README to track your progress as you complete modules.
</details>

<details>
<summary><b>13. Do I need to memorize the code?</b></summary>
No. Focus on understanding the concepts and architecture. You will always have access to documentation in the real world.
</details>

<details>
<summary><b>14. Are there any video lectures?</b></summary>
This is a text and code-based curriculum. We recommend supplementary YouTube videos when you need a different explanation perspective.
</details>

<details>
<summary><b>15. What is MLOps and why is it included?</b></summary>
MLOps (Machine Learning Operations) is how you put models into production reliably. It's the most requested skill by employers today.
</details>

<details>
<summary><b>16. Can I contribute a new project?</b></summary>
Yes! Please review the Contributor Experience section and submit a Pull Request.
</details>

<details>
<summary><b>17. I found a typo, what should I do?</b></summary>
Please open an issue or directly submit a pull request fixing the typo. Community help is greatly appreciated!
</details>

<details>
<summary><b>18. Is there a Discord or community?</b></summary>
Currently, discussions are held in the GitHub Discussions tab of this repository.
</details>

<details>
<summary><b>19. Can I use this to teach my own class?</b></summary>
Yes, this repository is open-source. Please just provide appropriate attribution to the original repository.
</details>

<details>
<summary><b>20. What do I do after finishing the roadmap?</b></summary>
Read research papers, contribute to major open-source ML libraries (like PyTorch or HuggingFace), and build complex capstone projects.
</details>

---

## 📈 Repository Statistics

![GitHub Repo stars](https://img.shields.io/github/stars/sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap?style=for-the-badge&color=yellow)
![GitHub forks](https://img.shields.io/github/forks/sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap?style=for-the-badge&color=orange)
![GitHub issues](https://img.shields.io/github/issues/sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap?style=for-the-badge&color=red)
![GitHub license](https://img.shields.io/github/license/sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap?style=for-the-badge&color=blue)

### Star History
*(Placeholder for Star History Graph)*
```html
<a href="https://star-history.com/#sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=sandaruns2004/Full-ML-DL-CV-and-Data-Science-Roadmap&type=Date" />
 </picture>
</a>
```

### Contribution Graph
*(Placeholder for Contribution Graph - you can use tools like GitHub Profile Summary Cards or similar open source tools to render this)*

---
> **Keywords for SEO**: Machine Learning Roadmap, Deep Learning Roadmap, Data Science Roadmap, Computer Vision Roadmap, NLP Roadmap, AI Engineer Roadmap, LLM Roadmap, MLOps Roadmap, Python for AI, Artificial Intelligence Learning Path.
