# 🧾 Project 09: OCR Document Scanner

> **Difficulty**: ⭐⭐⭐⭐☆ Advanced | **Goal**: Un-warping images and Regex extraction.

Build an end-to-end pipeline that takes a messy, angled photo of a receipt and extracts structured financial data.

---

## The Implementation Roadmap

### 🟢 Beginner: Basic OCR
1. **Raw Inference**: Pass a raw, messy photo of a receipt directly into `EasyOCR` or `PaddleOCR`. Notice how many words it gets wrong because of shadows and weird angles.

### 🟡 Intermediate: Preprocessing (The Secret Sauce)
2. **Grayscale & Blur**: Convert the image to Grayscale and apply a Gaussian Blur.
3. **Edge Detection**: Use OpenCV Canny Edge detection to find the edges of the paper.
4. **Perspective Transform**: Write a script to find the four corners of the receipt and apply a Perspective Transform (`cv2.warpPerspective`) to flatten the angled receipt into a perfect 2D top-down view (like a flatbed scanner).
5. **Re-run OCR**: Run OCR on the flattened image and observe the massive improvement in accuracy.

### 🔴 Advanced: Data Parsing
6. **Post-Processing (Regex)**: The OCR engine just outputs a massive string. Write a Regular Expression to scan the OCR output text and extract any number formatted as currency (e.g., `$14.99` or `Total: 14.99`).
7. **Output**: Print the final structured total to the console, ignoring all the fluff (like the store name or the date).
