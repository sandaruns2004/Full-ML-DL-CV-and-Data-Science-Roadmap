# 🛂 Project 05: Face Attendance System

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Goal**: Real-time embeddings & verification.

Build an end-to-end Face Recognition pipeline capable of verifying employees as they walk into an office.

---

## The Implementation Roadmap

### 🟢 Beginner: The Database
1. **Data Collection**: Create a folder of images containing faces of "authorized" people (e.g., your friends or family members).
2. **Extraction**: Use the `face_recognition` Python library (or DeepFace) to extract the 128-dimensional embeddings for everyone in your database and store them in a Python dictionary (Name $\rightarrow$ Vector).

### 🟡 Intermediate: Real-Time Matching
3. **Webcam Loop**: Open your webcam using OpenCV.
4. **Detect & Extract**: For every frame, detect faces in the live feed and extract their live embeddings.
5. **Verification Math**: Calculate the Euclidean distance (or Cosine Similarity) between the live face vector and all the vectors in your dictionary.

### 🔴 Advanced: The UI and Security
6. **Visual Output**: Draw a box around the face on the live feed. If a match is found (distance is below the threshold), display the person's name in Green. If no match is found, display "UNKNOWN" in Red.
7. **Liveness Detection (Bonus)**: Simple face recognition can be fooled by holding a photo up to the webcam. Integrate a simple liveness check, such as requiring the user to blink (by measuring the Eye Aspect Ratio using facial landmarks) before granting access.
