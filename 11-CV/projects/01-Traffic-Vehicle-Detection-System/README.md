# 🚗 Project 01: Traffic Vehicle Detection System

> **Difficulty**: ⭐⭐☆☆☆ Intermediate | **Goal**: Real-time YOLO tracking.

Build a real-world YOLO-based vehicle detector capable of tracking cars on a highway.

---

## The Implementation Roadmap

### 🟢 Beginner: Detection
1. **Dataset**: Download the COCO subset for vehicles, or use Roboflow to find a traffic dataset.
2. **Inference**: Use a pre-trained `YOLOv8n` (nano) model out of the box. Run it on a `.mp4` video of a highway using the Ultralytics CLI or Python API to draw bounding boxes on cars.

### 🟡 Intermediate: Fine-tuning & Filtering
3. **Training**: The pre-trained model might struggle with tiny cars far away. Fine-tune the YOLO model specifically on your downloaded traffic dataset for 10 epochs.
4. **Filtering**: Modify your Python script to only draw bounding boxes if the class is `Car`, `Truck`, `Bus`, or `Motorcycle`. Ignore pedestrians or stop signs.

### 🔴 Advanced: Real-time Speed & Optimization
5. **Optimization**: Ensure the script runs at a minimum of 30 FPS. If it doesn't, export the model to ONNX format and run inference using ONNX Runtime.
6. **Bonus Math**: Draw two horizontal lines on the video frame. If a car's bounding box crosses Line A, start a timer. When it crosses Line B, calculate its estimated MPH based on the pixel distance!
