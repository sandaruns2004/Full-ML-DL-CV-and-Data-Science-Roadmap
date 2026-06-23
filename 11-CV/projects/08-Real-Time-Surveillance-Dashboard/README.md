# 🚨 Project 08: Real-Time Surveillance Dashboard

> **Difficulty**: ⭐⭐⭐⭐⭐ Advanced | **Goal**: Multi-threading & Anomaly Detection.

Simulate a security desk where multiple video streams are analyzed simultaneously, triggering alerts when specific conditions are met.

---

## The Implementation Roadmap

### 🟢 Beginner: Anomaly Logic
1. **Detection Model**: Use a highly optimized YOLO-Nano model.
2. **Alert Logic**: Define a "Restricted Zone" (a polygon) on a single camera feed. If a `Person` bounding box intersects the Restricted Zone, trigger an alert (e.g., print "INTRUDER DETECTED").

### 🟡 Intermediate: Multi-Camera Processing
3. **The Bottleneck**: Reading two video streams sequentially in a `while` loop will cut your FPS in half. 
4. **Multi-Threading**: Write a Python script using the `threading` or `multiprocessing` library capable of reading and processing at least two video streams concurrently without freezing the main program.

### 🔴 Advanced: The Unified Dashboard
5. **Stitching**: Use OpenCV (or a UI framework like Streamlit/Tkinter) to stitch the two camera feeds side-by-side into a single unified window.
6. **Visual Alerts**: If the Restricted Zone is breached on Camera 2, make the border of Camera 2's feed flash red on the unified dashboard and log a timestamp of the event to a text file.
