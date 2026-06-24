# 10 - Edge Detection Visualizer

## 🎯 Objective
Explore classic Computer Vision techniques by applying mathematical filters to isolate edges and gradients in an image without using Neural Networks.

## 🧠 Concepts Covered
- **Canny Edge Detector**: A multi-stage algorithm that detects a wide range of edges in images.
- **Sobel Operator**: Computes an approximation of the gradient of the image intensity function to find edges.
- **Laplacian Filter**: Computes the second derivative of an image to highlight regions of rapid intensity change.
- **OpenCV Integration**: Using `cv2` to perform image matrix operations dynamically in a web app.

## 🚀 Getting Started

Launch the interactive Streamlit dashboard to upload images and play with the edge detection filters:
```bash
streamlit run app.py
```

## 📂 Project Structure
```
10-Edge-Detection-Visualizer/
│
├── app.py                # Streamlit Web App using OpenCV
└── README.md
```
