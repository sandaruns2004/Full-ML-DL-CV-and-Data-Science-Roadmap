# 🛠️ Intermediate Projects

> **Target**: Master Deep Learning, PyTorch, Computer Vision, and basic deployment.

These projects push you beyond Scikit-Learn. You will build Custom Neural Networks from scratch, work with massive unstructured data (images), and deploy your models.

---

## 1. Pneumonia Detection from X-Rays

**Goal**: Build a Convolutional Neural Network (CNN) in PyTorch to classify Chest X-Ray images as "Normal" or "Pneumonia".

**The Dataset**: The "Chest X-Ray Images (Pneumonia)" dataset on Kaggle (over 5,000 images).

**Requirements**:
1. **PyTorch Dataset**: Write a custom subclass of `torch.utils.data.Dataset`. It should use PIL to load the JPEG images and resize them all to exactly $224 \times 224$.
2. **Data Augmentation**: Use `torchvision.transforms` to apply Random Horizontal Flips and Random Rotations (up to 15 degrees) to the training set to prevent overfitting.
3. **Transfer Learning**: Do not build a CNN from scratch. Load a pre-trained `ResNet18` from `torchvision.models`. Freeze all the convolutional layers. Replace the final `fc` (fully connected) layer with a new linear layer that outputs 2 classes.
4. **Training Loop**: Write a manual PyTorch training loop utilizing `CrossEntropyLoss` and the `Adam` optimizer. Ensure the code runs on a GPU (`.to('cuda')`) if available.
5. **Evaluation**: Because the dataset is highly imbalanced (many more Pneumonia images than Normal), calculate the **F1-Score** and plot a Confusion Matrix.

**Bonus Challenge**: Implement **Grad-CAM**. Write a script that takes a Pneumonia X-Ray, passes it through the model, and generates a heatmap over the image showing *exactly which part of the lungs* caused the CNN to predict Pneumonia.

---

## 2. Real-Time Object Tracking with YOLO

**Goal**: Use a state-of-the-art Object Detection model to process a video file frame-by-frame and track moving vehicles.

**The Tools**: The `ultralytics` YOLOv8 library and `OpenCV` (`cv2`).

**Requirements**:
1. **Load YOLO**: Download the pre-trained `yolov8n.pt` (Nano) model. 
2. **Video Processing**: Use `cv2.VideoCapture` to load a `.mp4` video of highway traffic.
3. **Inference**: For every frame in the video, pass the frame to the YOLO model to get the bounding boxes of all detected cars and trucks.
4. **Drawing**: Use `cv2.rectangle` to draw the bounding boxes onto the frame, and `cv2.putText` to write the class name and confidence score above the box.
5. **Output**: Use `cv2.VideoWriter` to save the modified frames back into a brand new `.mp4` video file showing the AI detection in action.

**Bonus Challenge**: Write logic that counts how many cars cross a specific horizontal line on the screen, creating an automated traffic counter!

---

## 3. The FastAPI Credit Scoring Engine

**Goal**: Train a tabular XGBoost model and deploy it as a production-ready REST API using FastAPI and Docker.

**The Dataset**: The "Give Me Some Credit" dataset on Kaggle (predicting if a borrower will experience financial distress).

**Requirements**:
1. **Modeling**: Train an `xgboost.XGBClassifier` on the dataset. Save the model to disk using `joblib`.
2. **FastAPI**: Create a `main.py` file. Define a `Pydantic` BaseModel representing the expected inputs (Age, DebtRatio, MonthlyIncome, etc.).
3. **Endpoint**: Create a `POST /score` endpoint that takes the JSON input, converts it to a Numpy array, passes it to the loaded XGBoost model, and returns a JSON response: `{"approved": true, "probability_of_default": 0.12}`.
4. **Dockerization**: Write a `Dockerfile` that packages your FastAPI app and the XGBoost model. Build the image and run it locally.
5. **Testing**: Write a `test_api.py` script using the `requests` library to send 10 dummy applicants to your Dockerized API and print the responses.

**Bonus Challenge**: Add a `/metrics` endpoint to your FastAPI app that returns a Prometheus-compatible string showing exactly how many total predictions the API has made since it booted up.

---

[← Beginner Projects](./01-Beginner-Projects.md) | [Back to Index](../README.md) | [Next: Advanced Projects →](./03-Advanced-Projects.md)
