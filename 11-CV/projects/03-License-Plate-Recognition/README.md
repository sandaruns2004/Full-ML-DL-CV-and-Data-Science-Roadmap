# 🪪 Project 03: License Plate Recognition

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Goal**: Detection + OCR Pipeline.

Combine a bounding box detector with an Optical Character Recognition (OCR) engine to build an Automated License Plate Recognition (ALPR) system.

---

## The Implementation Roadmap

### 🟢 Beginner: The Detection Stage
1. **Dataset**: Find a dataset of cars with license plates on Kaggle or Roboflow.
2. **Model Training**: Train a YOLO model to detect a single class: `License_Plate`.
3. **Cropping**: Write a Python script that takes an image, predicts the bounding box for the plate, and saves the cropped image of *just* the plate to your disk.

### 🟡 Intermediate: The OCR Stage
4. **Preprocessing**: The cropped plate might be blurry or have low contrast. Use OpenCV to convert the cropped plate to grayscale and apply adaptive thresholding.
5. **Text Extraction**: Pass the preprocessed crop into `EasyOCR` or `PaddleOCR` to extract the raw string of text.

### 🔴 Advanced: Pipeline Integration
6. **End-to-End**: Combine the two stages into a single seamless function. The function takes an image, detects the plate, crops it in memory, reads the text, and returns the final string.
7. **Display**: Use OpenCV to display the original image with the extracted text printed clearly in large font above the car.
8. **Regex**: Ensure the OCR output is formatted correctly by using Regular Expressions to strip out any logos or random characters that aren't part of a standard license plate format.
