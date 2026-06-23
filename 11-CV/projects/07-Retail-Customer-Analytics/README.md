# 🛒 Project 07: Retail Customer Analytics

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Goal**: ByteTrack logic and tripwires.

Combine Object Detection with Object Tracking to solve a classic business problem: Foot traffic analytics.

---

## The Implementation Roadmap

### 🟢 Beginner: Detection and Tracking
1. **Detection**: Use YOLOv8 to detect the `Person` class in a video of a store entrance.
2. **Tracking**: Integrate **ByteTrack** or **DeepSORT** (via the Ultralytics API). Verify that the system assigns a unique ID to every person detected and maintains that ID as they walk.

### 🟡 Intermediate: The Tripwire Logic
3. **The Line**: Draw an invisible horizontal "tripwire" line in the middle of the screen.
4. **State Management**: Maintain a dictionary of each tracked ID and its previous center $Y$ coordinate.
5. **Counting**: 
   - If a person's center moves from *above* the line in frame $T-1$ to *below* the line in frame $T$, increment the `Entered` counter.
   - If a person moves from *below* to *above*, increment the `Exited` counter.

### 🔴 Advanced: Robustness and Rendering
6. **Robustness**: What happens if a person stands exactly on the line and shifts their weight back and forth? Implement a "buffer zone" (hysteresis) around the tripwire to prevent double-counting.
7. **Output**: Render the video with bounding boxes, unique IDs floating above the heads, the drawn tripwire line (which flashes red when crossed), and the total counters clearly displayed at the top.
