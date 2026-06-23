# 🏋️ Project 06: Fitness Pose Analyzer

> **Difficulty**: ⭐⭐⭐☆☆ Intermediate | **Goal**: Tracking joints for exercise form.

Build an application that can track human joints in real-time and provide feedback on a user's exercise form.

---

## The Implementation Roadmap

### 🟢 Beginner: Find the Joints
1. **Framework**: Install and initialize `mediapipe.solutions.pose`.
2. **Camera Feed**: Read a webcam or a pre-recorded video of someone doing bicep curls.
3. **Extraction**: Extract the $X, Y$ pixel coordinates of the 3 key joints: Shoulder, Elbow, and Wrist. Print them to the console.

### 🟡 Intermediate: The Trigonometry
4. **Math**: Write a Python function that takes the $X, Y$ coordinates of the three joints and calculates the exact angle of the elbow using Trigonometry (`math.atan2`).
5. **Logic**:
   - If the angle is $> 160$ degrees, the arm is straight (State = Down).
   - If the angle is $< 30$ degrees, the curl is complete (State = Up).

### 🔴 Advanced: The Real-Time Dashboard
6. **Rep Counter**: Implement state-tracking logic to count how many full repetitions the user has completed.
7. **Dashboard**: Output the video feed with the MediaPipe pose skeleton drawn directly over the user. Display the current elbow angle dynamically near the elbow joint, and display a large "Rep Count" in the top corner of the screen.
