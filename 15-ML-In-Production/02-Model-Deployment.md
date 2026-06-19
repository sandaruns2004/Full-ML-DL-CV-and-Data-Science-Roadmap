# 🚀 Model Deployment (FastAPI & Docker)

> **Prerequisites**: ML Pipeline, Python Basics | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [The Goal of Deployment](#1-the-goal-of-deployment)
2. [Serialization (Saving the Model)](#2-serialization-saving-the-model)
3. [Building a REST API with FastAPI](#3-building-a-rest-api-with-fastapi)
4. [Containerization with Docker](#4-containerization-with-docker)
5. [Cloud Deployment (AWS / GCP)](#5-cloud-deployment-aws--gcp)
6. [Library Implementation (End-to-End Code)](#6-library-implementation-end-to-end-code)
7. [Project Ideas & What's Next](#7-project-ideas--whats-next)

---

## 1. The Goal of Deployment

A Machine Learning model is useless if it only exists in a Jupyter Notebook on your laptop. 
**Deployment** is the process of putting your model on a server so that other applications (like an iOS app, a React website, or a backend microservice) can send it data and receive predictions in real-time.

There are three main deployment paradigms:
1. **Batch/Offline**: Run the model once a night on a massive database (e.g., generating Netflix recommendations while users sleep).
2. **Real-time (Online)**: Wrap the model in an API. The model responds to user actions in milliseconds (e.g., detecting credit card fraud the moment a user swipes their card).
3. **Edge Deployment**: Shrink the model and run it directly on the user's phone or IoT device (e.g., Apple FaceID).

This file focuses on the industry standard for Real-time inference: **REST APIs and Docker**.

---

## 2. Serialization (Saving the Model)

Before you deploy a model, you must save it to disk. This is called Serialization.

- **Scikit-Learn (Pickle/Joblib)**: Converts a Python object into a byte stream. `joblib.dump(model, 'model.pkl')`.
- **PyTorch**: Saves the neural network weights (the `state_dict`). `torch.save(model.state_dict(), 'weights.pt')`.
- **ONNX (Open Neural Network Exchange)**: A universal format. You can train a model in PyTorch, export it to ONNX, and run it in a C++ or Node.js environment without needing PyTorch installed!

*Warning: Never unpickle a `.pkl` file from an untrusted source. Pickles can execute arbitrary malicious code upon loading.*

---

## 3. Building a REST API with FastAPI

To make the model accessible over the internet, we wrap it in a REST API. 

For years, Python developers used Flask. Today, the undisputed king of ML APIs is **FastAPI**.
FastAPI is significantly faster than Flask, natively supports asynchronous code (`async/await`), and most importantly, it uses **Pydantic** for automatic data validation.

If your model expects an `age` (integer) and `salary` (float), and a user sends a JSON payload with `age: "twenty"`, FastAPI will automatically intercept the bad request and return an HTTP 422 Error before it ever touches your ML model.

---

## 4. Containerization with Docker

You built the FastAPI app. It works perfectly on your laptop. You upload the code to an AWS server, and it instantly crashes. 
Why? The server has Python 3.8, but your laptop has Python 3.11. The server is missing the exact C++ drivers needed for pandas. 

**Docker solves the "It works on my machine" problem.**

A Docker **Container** is a lightweight, standalone, executable package of software that includes everything needed to run an application: code, runtime, system tools, libraries, and settings.

You write a `Dockerfile` that says:
> "Start with an Ubuntu Linux computer. Install Python 3.10. `pip install fastapi scikit-learn`. Copy my `model.pkl` and `main.py` into the computer. Run the API."

Docker builds this into an **Image**. You can drop this Image onto an AWS server, a Google Cloud server, or a Mac laptop, and it is mathematically guaranteed to run exactly the same way.

---

## 5. Cloud Deployment (AWS / GCP)

Once you have a Docker Image, you can host it on the cloud.

1. **Serverless (Easy)**: AWS App Runner, Google Cloud Run. You just give them your Docker container, and they handle everything. If 10,000 users hit your API simultaneously, Google automatically spins up 100 containers to handle the traffic, and scales back down to 0 when they leave.
2. **Virtual Machines (Medium)**: AWS EC2. You rent a Linux server, SSH into it, install Docker, and run your container 24/7.
3. **Kubernetes (Hard)**: AWS EKS / Google GKE. The industry standard for massive enterprise deployments. A Kubernetes cluster manages thousands of interacting Docker containers, automatically healing them if they crash.

---

## 6. Library Implementation (End-to-End Code)

Let's write a complete FastAPI application and the Dockerfile to deploy it.

**1. `main.py` (The API)**
```python
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# 1. Initialize the App
app = FastAPI(title="House Price Predictor API")

# 2. Load the model (runs once when server starts)
model = joblib.load("random_forest_model.pkl")

# 3. Define the expected Input schema using Pydantic
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft: int
    year_built: int

# 4. Define the Prediction Endpoint
@app.post("/predict")
async def predict_price(features: HouseFeatures):
    # Convert Pydantic object to a NumPy array
    input_data = np.array([[
        features.bedrooms, 
        features.bathrooms, 
        features.sqft, 
        features.year_built
    ]])
    
    # Run the model
    prediction = model.predict(input_data)[0]
    
    # Return JSON response
    return {"predicted_price_usd": round(prediction, 2)}
```

**2. `Dockerfile`**
```dockerfile
# Start from an official Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the trained model file
COPY main.py .
COPY random_forest_model.pkl .

# Expose port 8000 for the API
EXPOSE 8000

# Command to run the application using Uvicorn (an ASGI server)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 7. Project Ideas & What's Next

### Project Ideas
- 🟢 **Local Docker API**: Train a Scikit-Learn `LogisticRegression` model on the Titanic dataset and save it as `model.pkl`. Write a FastAPI script to load it. Write a `Dockerfile`, run `docker build -t titanic-api .`, and run the container on your laptop using `docker run -p 8000:8000 titanic-api`. Send a POST request to `localhost:8000/predict` using Postman or cURL!
- 🟡 **Deploy to Google Cloud Run**: Take the exact Docker container you just built and push it to Google Cloud Registry. Use the Google Cloud Console to deploy it to Cloud Run. It will give you a public `https://...` URL that anyone in the world can use to query your model.

### What's Next
| Next | Why |
|------|-----|
| [MLOps (CI/CD)](./03-MLOps.md) | Building the Docker container manually is fine for a side project. But how do professional teams automatically train, build, and deploy new Docker containers every time a Github commit is merged? We need MLOps and CI/CD pipelines. |

---

[← ML Pipeline](./01-ML-Pipeline.md) | [Back to Index](../README.md) | [Next: MLOps →](./03-MLOps.md)
