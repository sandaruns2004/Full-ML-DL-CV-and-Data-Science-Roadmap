# 🐘 Big Data & Distributed Machine Learning

> **Prerequisites**: Python, Scikit-Learn | **Difficulty**: ⭐⭐⭐⭐☆ Advanced

---

## 📋 Table of Contents
1. [The "Out of Memory" Problem](#1-the-out-of-memory-problem)
2. [Hadoop & The MapReduce Paradigm](#2-hadoop--the-mapreduce-paradigm)
3. [Apache Spark (Resilient Distributed Datasets)](#3-apache-spark-resilient-distributed-datasets)
4. [Dask: Pandas for Big Data](#4-dask-pandas-for-big-data)
5. [Distributed Deep Learning (Horovod & DDP)](#5-distributed-deep-learning-horovod--ddp)
6. [Library Implementation (PySpark MLlib)](#6-library-implementation-pyspark-mllib)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The "Out of Memory" Problem

Standard Data Science tools (Pandas, Scikit-Learn) operate completely **In-Memory**. 
If you load a 10GB CSV file into Pandas, it will consume about 20-30GB of RAM (due to Python overhead). If your laptop only has 16GB of RAM, the kernel crashes instantly.

When datasets grow to Terabytes or Petabytes, you cannot buy a single computer large enough to hold it. 

The solution is **Distributed Computing**: Instead of 1 supercomputer with 1,000GB of RAM, we rent 100 cheap servers with 10GB of RAM each, and network them together into a **Cluster**. The challenge is writing code that seamlessly executes across all 100 machines simultaneously.

---

## 2. Hadoop & The MapReduce Paradigm

In the 2000s, Google and Yahoo invented the foundational Big Data architecture: **Hadoop** and **MapReduce**.

**HDFS (Hadoop Distributed File System)**: Takes a 1TB file, chops it into 128MB blocks, and scatters those blocks across 100 different computers.

**MapReduce**: A programming paradigm to process that scattered data.
1. **Map**: Send the Python script to the 100 computers. Each computer runs the script *only on its local 128MB chunk of data* (e.g., counting the words in its chunk).
2. **Shuffle**: The computers exchange data over the network to group similar keys together.
3. **Reduce**: The computers aggregate the final results (e.g., summing up the total word counts).

*Problem*: Hadoop MapReduce writes intermediate results to the hard drive after every step. It is incredibly slow and completely unsuitable for Machine Learning (which requires hundreds of iterative passes over the data).

---

## 3. Apache Spark (Resilient Distributed Datasets)

In 2014, **Apache Spark** destroyed Hadoop MapReduce by doing one simple thing: **In-Memory Processing**.

Instead of writing to the hard drive, Spark keeps the scattered data chunks in the RAM of the 100 computers. This makes Spark up to 100x faster than Hadoop.

### RDDs and DataFrames
Spark's core data structure is the RDD (Resilient Distributed Dataset). Today, we use the **Spark DataFrame**, which looks exactly like a Pandas DataFrame, but is actually chopped up and spread across a cluster.

### Lazy Evaluation
If you tell Spark to filter a dataset and multiply a column by 10, **it does absolutely nothing**. It just adds the instructions to a DAG (Directed Acyclic Graph) Execution Plan. 
It only executes the code when you call an "Action" (like `.show()` or `.count()`). This allows Spark's optimizer to mathematically combine and streamline your code before running it.

---

## 4. Dask: Pandas for Big Data

Spark is written in Scala (Java ecosystem). While it has a Python wrapper (`PySpark`), it feels "un-Pythonic" and error messages can be terrifying Java stack traces.

**Dask** is a modern, pure-Python alternative. 
Dask DataFrames perfectly mimic the Pandas API. Under the hood, a Dask DataFrame is just 1,000 small Pandas DataFrames coordinated by a central scheduler. 

If you already know Pandas, you can learn Dask in 5 minutes. It is the preferred tool for Python-heavy teams handling data in the 50GB to 1TB range.

---

## 5. Distributed Deep Learning (Horovod & DDP)

What about Neural Networks? Training a 70 Billion parameter LLM takes 300 years on a single GPU. We need to train it on a cluster of 1,024 GPUs.

**Data Parallelism**:
The standard approach. 
1. Copy the exact same Neural Network to all 1,024 GPUs.
2. Chop the massive training dataset into 1,024 chunks.
3. Every GPU does a forward/backward pass on its own chunk of data to calculate gradients.
4. **All-Reduce**: The GPUs pause, communicate over the network to average all their gradients together, and update their weights identically.

**Frameworks**:
- **PyTorch DDP (DistributedDataParallel)**: The native standard for multi-GPU training.
- **Horovod**: Built by Uber, it uses MPI (Message Passing Interface) and ring-allreduce algorithms to scale TensorFlow and PyTorch across thousands of machines with 90% efficiency.

---

## 6. Library Implementation (PySpark MLlib)

Let's build a Distributed Machine Learning model using Spark's `MLlib`. 
*(Note: Spark MLlib requires strict Vector assembly, unlike Scikit-Learn).*

```python
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# 1. Initialize Spark Session (Connect to the Cluster)
# In a real environment, this connects to a cluster of 100 machines.
# Here, it runs locally using all your laptop's CPU cores.
spark = SparkSession.builder.appName("HousePricePrediction").getOrCreate()

# 2. Load Distributed Data
# This could be reading 10,000 CSV files from an AWS S3 bucket simultaneously
data = spark.createDataFrame([
    (2000, 3, 10, 500000),
    (1500, 2, 5, 300000),
    (3000, 4, 1, 800000),
    (1200, 1, 15, 200000)
], ["SqFt", "Bedrooms", "Age", "Price"])

# 3. Vector Assembler
# Spark MLlib requires ALL features to be combined into a single Vector column
assembler = VectorAssembler(
    inputCols=["SqFt", "Bedrooms", "Age"],
    outputCol="features"
)
prepared_data = assembler.transform(data)

# 4. Train-Test Split
train_data, test_data = prepared_data.randomSplit([0.8, 0.2], seed=42)

# 5. Distributed Random Forest
# Spark will automatically parallelize the building of 100 trees across the cluster
rf = RandomForestRegressor(featuresCol="features", labelCol="Price", numTrees=100)
model = rf.fit(train_data)

# 6. Predict and Evaluate
predictions = model.transform(test_data)
evaluator = RegressionEvaluator(labelCol="Price", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"Root Mean Squared Error (RMSE): {rmse}")
spark.stop()
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Dask Out-of-Core Processing**: Download the NYC Taxi dataset (a massive multi-gigabyte CSV). Try opening it in Pandas (it might crash or lag). Then, use `dask.dataframe.read_csv()` to open it instantly. Compute the average fare price grouped by passenger count to see how Dask processes the file in chunks without filling up your RAM.
- 🟡 **PyTorch DDP on Colab**: Open a Google Colab notebook with 2 GPUs (if available, or simulate it using Kaggle notebooks). Write a PyTorch image classifier and wrap your model in `torch.nn.parallel.DistributedDataParallel`. Train it on CIFAR-10 and measure the speedup compared to a single GPU.

### What's Next
| Next | Why |
|------|-----|
| [Data Ethics & Fairness](./06-Data-Ethics-Fairness.md) | As we handle massive datasets, we must also ensure our models are fair, unbiased, and ethical before deploying them to the real world. |

---

[← Geospatial Analysis](./04-Geospatial-Analysis.md) | [Back to Index](../README.md) | [Next: Data Ethics Fairness →](./06-Data-Ethics-Fairness.md)
