# 🏔️ Mean Shift Clustering

> **Prerequisites**: DBSCAN | **Difficulty**: ⭐⭐⭐☆☆ Advanced

---

## 1. Mode Seeking and KDE

### 🟢 Beginner
**Simple Explanation**: Imagine dropping dozens of people on a foggy, hilly landscape. They are told to blindly walk uphill until they reach the highest peak. People who end up at the exact same peak belong to the same cluster. 

### 🟡 Intermediate
**Workflow**: 
Mean Shift is a sliding-window algorithm that attempts to find dense areas of data points. It is a centroid-based algorithm, which works by updating candidates for centroids to be the mean of the points within the sliding-window.
**Applications**: Computer Vision (Object tracking, Image segmentation).

### 🔴 Advanced
**Mathematics (Kernel Density Estimation)**: 
Mean Shift builds upon KDE. Let $K(x)$ be a kernel function (like a Gaussian). The multivariate kernel density estimate is:
$f(x) = \frac{1}{n h^d} \sum_{i=1}^n K\left(\frac{x - x_i}{h}\right)$
The algorithm computes the gradient of this density and shifts the points towards the mode (peak) via gradient ascent.
