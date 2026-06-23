# Project 04: Face Mask Detection

> [!NOTE]  
> Computer Vision models aren't just meant to run on static images. In this project, you will integrate your trained CNN with OpenCV to perform real-time inference on a live webcam feed.

## The Goal
Train a binary classifier to detect whether a person is wearing a face mask or not, and deploy it to a live video stream.

## Requirements

1. **Dataset**: Find a Face Mask dataset (Kaggle has several). It should have two classes: `With_Mask` and `Without_Mask`.
2. **Model Training**: Train a lightweight CNN (like MobileNet) using Transfer Learning. Since it needs to run in real-time on video frames, inference speed is critical.
3. **OpenCV Integration**: 
   - Write a script that opens the default webcam using `cv2.VideoCapture(0)`.
   - Read the frames in a loop.
   - Use an OpenCV Haar Cascade face detector to find the bounding box of the face.
   - Crop the face, resize it to your CNN's required input size (e.g., $224 \times 224$), and pass it through your model.
4. **Visual Feedback**: Draw a bounding box around the face. Make the box **Green** if the mask is on, and **Red** if the mask is off. Display the probability score above the box.

## Bonus Challenge
Optimize the inference loop so the webcam feed runs smoothly at >20 FPS.
