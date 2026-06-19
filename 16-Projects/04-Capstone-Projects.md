# 🎓 Capstone Projects (Portfolio Builders)

> **Target**: Prove your ability to architect, train, and deploy massive Machine Learning systems.

These projects are designed to be the crown jewels of your GitHub portfolio. They combine everything you've learned: Advanced Deep Learning, Big Data, MLOps, CI/CD, and Cloud Deployment.

If you can build one of these, you have the skills of a practicing Machine Learning Engineer.

---

## 1. Enterprise E-Commerce Recommender System

**The Scenario**: An e-commerce company wants to increase sales by showing users personalized products on the homepage. They have a massive dataset of past clicks, purchases, and product metadata.

**The Architecture**:
1. **Data Processing (PySpark)**: Use PySpark to process 5 million rows of simulated user-item interactions. Create a user-item interaction matrix.
2. **The Two-Tower Model (PyTorch)**: Build a Neural Collaborative Filtering Two-Tower model. The User Tower processes user age and history. The Item Tower processes product descriptions (using an NLP embedding) and product categories.
3. **The Vector Database (FAISS)**: After training, extract the 128-dimensional embeddings for all 50,000 products. Load them into a FAISS (Facebook AI Similarity Search) index for lightning-fast retrieval.
4. **The API (FastAPI)**: Build an API with an endpoint `/recommend?user_id=123`. The API should quickly pass the user through the User Tower, get their embedding, query the FAISS database, and return the top 10 recommended product IDs.
5. **MLOps**: Write a `Dockerfile` for the API. Set up a GitHub Action that automatically runs unit tests on your FastAPI code whenever you push to the `main` branch.

**The "Wow" Factor**: Implement a basic UI using `Streamlit`. Show a mock homepage where selecting different "Users" from a dropdown instantly changes the displayed grid of recommended products.

---

## 2. End-to-End MLOps Pipeline for Fraud Detection

**The Scenario**: A bank processes thousands of credit card transactions per second. They need a model that detects fraud, but fraud tactics change every week. They need an automated pipeline that continuously retrains the model on new data.

**The Architecture**:
1. **The Model**: Train a highly optimized XGBoost or LightGBM model on an imbalanced credit card fraud dataset. Optimize for the F1-Score or Area Under the Precision-Recall Curve (PR-AUC).
2. **Data Versioning (DVC)**: Use DVC to version the massive fraud dataset. Store the actual data in an AWS S3 bucket, and track the `.dvc` pointers in Git.
3. **Experiment Tracking (MLflow)**: Integrate MLflow into your training script. Automatically log the hyperparameters (`max_depth`, `learning_rate`), the PR-AUC score, and the trained model artifact to a local MLflow tracking server.
4. **Orchestration (Airflow)**: Write an Apache Airflow DAG with 3 tasks:
   - `fetch_new_data`: Simulates downloading the last 7 days of transactions.
   - `train_model`: Runs the training script, comparing the new model to the currently deployed model in MLflow.
   - `deploy_model`: If the new model beats the old model, promote it to "Production" in the MLflow Model Registry.
5. **Monitoring (Evidently AI)**: Simulate "Data Drift" by artificially multiplying the transaction amounts of the incoming data by 3. Use Evidently AI to generate a drift report proving the model's environment has changed.

**The "Wow" Factor**: Fully containerize the entire stack. Provide a `docker-compose.yml` file that spins up the Airflow scheduler, the MLflow server, the Postgres database, and the FastAPI inference endpoint all with one command (`docker-compose up`).

---

## 3. Large Vision-Language Model (VLM) for Medical Diagnostics

**The Scenario**: A hospital wants an AI assistant that can look at a patient's MRI/X-Ray and provide a textual diagnostic report to assist doctors.

**The Architecture**:
1. **The Architecture (ViT + GPT)**: You will build an Encoder-Decoder VLM from scratch. 
   - **Encoder**: A pre-trained Vision Transformer (ViT) that takes the medical image and outputs a sequence of image patches (e.g., $196 \times 768$ matrix).
   - **Cross-Attention**: You must write a custom Transformer block that allows a Language Model to "attend" to the image patches.
   - **Decoder**: A pre-trained causal language model (like GPT-2 or a tiny LLaMA) that generates text conditioned on the cross-attention.
2. **The Dataset**: Find a dataset like `MIMIC-CXR` (Chest X-Rays paired with actual radiology text reports) or `IU X-Ray`.
3. **Training (Distributed)**: Because this model is massive, use PyTorch `DistributedDataParallel` (DDP) or `HuggingFace Accelerate` to train it. Use mixed-precision training (FP16) to fit it into GPU memory.
4. **Evaluation (BLEU/ROUGE)**: Write an evaluation script that compares the AI's generated diagnostic report against the true doctor's report using NLP metrics like BLEU or ROUGE.

**The "Wow" Factor**: Implement Grad-CAM or Attention Rollout. When the model generates the word "Tumor", visually highlight the specific part of the X-Ray image that the Cross-Attention mechanism was looking at when it generated that word. This provides crucial "Explainable AI" for the doctors.

---

[← Advanced Projects](./03-Advanced-Projects.md) | [Back to Index](../README.md) | [Next: Research Paper Reading Guide →](./05-Research-Paper-Reading-Guide.md)
