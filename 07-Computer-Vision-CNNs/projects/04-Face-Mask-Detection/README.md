# 04 - Face Mask Detection

## 🎯 Objective
Combine classic Computer Vision (OpenCV) with Deep Learning (PyTorch CNNs) to localize human faces and classify whether they are wearing a mask.

## 🧠 Concepts Covered
- **Haar Cascades**: Using OpenCV (`cv2.CascadeClassifier`) for rapid object localization (face detection).
- **Regions of Interest (ROI)**: Cropping the localized faces from the image.
- **Inference Pipeline**: Passing the ROI into a Convolutional Neural Network for binary classification (Mask vs No Mask).
- **Bounding Boxes**: Drawing rectangles and labels on images dynamically.

## 🚀 Getting Started

### Running the Web App
This project is fully deployed as a Streamlit dashboard. 
```bash
streamlit run app.py
```
*Note: If you have not trained and saved a custom `mask_detector.pth` model in a `src/` directory, the app will fall back to using uninitialized weights. The face localization (bounding boxes) will still work perfectly via OpenCV, but the mask classification will be random until trained weights are provided.*

## 📂 Project Structure
```
04-Face-Mask-Detection/
│
├── app.py                # Streamlit Web App with OpenCV & PyTorch
└── README.md
```
