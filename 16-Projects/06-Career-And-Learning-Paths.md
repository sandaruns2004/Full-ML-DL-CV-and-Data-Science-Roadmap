# 👔 Career & Learning Paths

> **Target**: Help you navigate the complex landscape of AI job titles and interviews.

"Machine Learning" is a massive umbrella term. A Data Scientist at Facebook has a completely different day-to-day job than an ML Engineer at OpenAI. This guide breaks down the primary career paths and how to tailor your resume for them.

---

## 1. The Data Scientist (The Analyst)

**The Role**: Data Scientists solve business problems. They spend 80% of their time writing SQL, analyzing A/B test results, building dashboards, and presenting insights to the CEO. They use ML (like XGBoost or Prophet) to forecast sales or predict churn, but they usually don't put neural networks into production C++ servers.

**Key Skills to Emphasize**:
- SQL (Window functions, CTEs)
- A/B Testing & Causal Inference
- Scikit-Learn, Pandas, Tableau/Looker
- Excellent communication and presentation skills

**Interview Focus**: Product Sense (e.g., "How would you measure the success of the new Facebook Like button?"), SQL whiteboard tests, and explaining ML algorithms simply.

---

## 2. The Machine Learning Engineer (The Builder)

**The Role**: ML Engineers are Software Engineers first, and Data Scientists second. They take the Python prototypes built by the Research team and rewrite them to be highly optimized, scalable, and deployed via Kubernetes. They build the data pipelines, the Docker containers, and the REST APIs.

**Key Skills to Emphasize**:
- Python (Advanced Object-Oriented, Async), C++, or Go
- PyTorch / TensorFlow
- Docker, Kubernetes, AWS/GCP
- CI/CD, Apache Airflow, MLOps

**Interview Focus**: LeetCode (Data Structures & Algorithms), System Design (e.g., "Design the Netflix Recommender System architecture"), and Deep Learning fundamentals.

---

## 3. The AI Researcher / Applied Scientist

**The Role**: This is usually reserved for PhDs, though not strictly required anymore. They work at places like DeepMind or FAIR. They do not build web APIs. They read arXiv papers, invent new mathematical loss functions, and train massive foundational models on 10,000 GPUs.

**Key Skills to Emphasize**:
- Deep mathematical understanding of Calculus and Linear Algebra
- PyTorch internals (custom CUDA kernels)
- Published papers at NeurIPS, ICML, or CVPR
- Extreme expertise in a niche (e.g., Diffusion Models or Graph Neural Networks)

**Interview Focus**: Whiteboarding complex mathematical proofs, explaining the exact gradients of a custom attention mechanism, and discussing the absolute bleeding-edge SOTA papers.

---

## 4. The Data Engineer (The Plumber)

**The Role**: Without Data Engineers, ML is impossible. They build the massive distributed data lakes that the ML models train on. They write the ETL (Extract, Transform, Load) scripts that move Terabytes of data from a raw database into a pristine Feature Store.

**Key Skills to Emphasize**:
- Apache Spark, Hadoop, Kafka
- Distributed Systems
- Snowflake, BigQuery, Databricks
- SQL and Python

**Interview Focus**: Database design, distributed system architecture, and writing highly optimized MapReduce or Spark code.

---

## How to Get Hired (Without a Degree)

1. **Stop doing Titanic and MNIST**: Every recruiter has seen 1,000 resumes with the Titanic dataset. It proves nothing.
2. **Build End-to-End**: A project that scrapes its own unique data, trains a decent model, and is deployed live on a public URL is 100x more impressive than an amazing model trapped in a local Jupyter Notebook.
3. **Contribute to Open Source**: Find an issue on the `scikit-learn` or `pytorch` GitHub repo. Submit a Pull Request. Putting "Contributor to PyTorch" on your resume will get you an interview anywhere.
4. **Write Technical Blogs**: Write a Medium or Substack article explaining a complex paper (like DDPMs) in simple terms. Senior engineers love hiring people who can explain complex math clearly.

---

[← Research Paper Reading Guide](05-Research-Paper-Reading-Guide.md) | [Back to Index](../README.md) | [Next: Future Versions →](07-Future-Versions.md)
