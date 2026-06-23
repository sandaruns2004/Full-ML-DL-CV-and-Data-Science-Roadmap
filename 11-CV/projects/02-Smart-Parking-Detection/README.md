# 🅿️ Project 02: Smart Parking Detection

> **Difficulty**: ⭐⭐☆☆☆ Intermediate | **Goal**: Detection + Spatial Logic.

Combine Object Detection with spatial mathematics to build an automated parking lot management system.

---

## The Implementation Roadmap

### 🟢 Beginner: Find the Cars
1. **Detection**: Use a pre-trained YOLO model to detect all cars in a static image of a parking lot.
2. **Output**: Print the total count of cars detected to the console.

### 🟡 Intermediate: Map the Lot
3. **Region Mapping**: Open the image in an editor (or write a quick OpenCV script) to manually define a set of polygons representing the valid, designated parking spots. Store these coordinates in a JSON file.
4. **Logic Integration**: For each defined parking spot polygon, check if a detected car's bounding box intersects it (using IoU or point-in-polygon math). If it does, mark the spot as "Taken". If not, mark it as "Empty".

### 🔴 Advanced: The Dashboard
5. **Dashboard Rendering**: Output a clean version of the image with:
   - **Red boxes** drawn over taken spots.
   - **Green boxes** drawn over empty spots.
   - A text overlay in the top left: "Available Spots: X".
6. **Video Stream**: Apply this logic to a video feed of a parking lot, updating the "Available Spots" counter in real-time as cars pull in and out.
