# 🏔️ Mean Shift Clustering

> **Prerequisites**: Probability Density Functions, K-Means | **Difficulty**: ⭐⭐⭐☆☆ Intermediate

---

## 📋 Table of Contents
1. [Learning Progression: Mean Shift](#1-learning-progression-mean-shift)
2. [Step-by-Step Algorithm Working](#2-step-by-step-algorithm-working)
3. [The Mathematics: Kernel Density Estimation](#3-the-mathematics-kernel-density-estimation)
4. [Python Implementation (From Scratch)](#4-python-implementation-from-scratch)
5. [Scikit-Learn Implementation](#5-scikit-learn-implementation)
6. [Real-World Applications](#6-real-world-applications)
7. [Advantages & Limitations](#7-advantages--limitations)

---

## 1. Learning Progression: Mean Shift

### 🟢 Beginner (Intuition & Analogy)
Imagine you are blindfolded in a hilly landscape, and your goal is to find the highest peak. 
1. You feel the ground around your feet. If it slopes upward to the right, you take a step to the right. 
2. You check again, find the upward slope, and take another step.
3. Eventually, the ground around you slopes downward in every direction. You are standing on a peak!

Now imagine thousands of people are dropped randomly across this landscape, all doing the exact same thing. Eventually, everyone will gather at the various peaks of the hills. People who end up at the exact same peak belong to the same "cluster".
This is exactly how **Mean Shift** works! It moves data points towards the highest density of data (the peaks).

### 🟡 Intermediate (Concept & Reasoning)
Mean Shift is a **sliding-window-based** algorithm that attempts to find dense areas of data points. It is a centroid-based algorithm, meaning that the goal is to locate the center points of each group/class.
However, unlike K-Means, you **do not need to choose the number of clusters**. The algorithm discovers them automatically. 
Instead of choosing $K$, you must choose the **Bandwidth**, which dictates the size of the "window" (or the radius of the circle) used to search for the highest density.

### 🔴 Advanced (Algorithmic Depth)
Mean Shift builds upon the concept of **Kernel Density Estimation (KDE)**. Imagine placing a probability distribution (a Kernel, usually Gaussian) over every single data point in your dataset, and adding them all up. This creates a topological map (a surface) where areas with lots of points have high "mountains" of probability, and sparse areas have "valleys".
Mean Shift mathematically computes the gradient of this KDE surface and shifts the window in the direction of the maximum increase in density (the mode) until it reaches a local maximum.

---

## 2. Step-by-Step Algorithm Working

1. **Initialize Windows**: Start by placing a circular window (of radius `bandwidth`) centered on every single data point.
2. **Calculate the Mean**: For a given window, find the center of mass (the mean) of all data points inside that window.
3. **Shift**: Move the center of the window to that newly calculated mean.
4. **Repeat**: Repeat steps 2 and 3 until the window stops moving (the mean of the points inside the window is equal to the center of the window). The window has reached a "peak" (a local density maximum).
5. **Merge**: Once all windows have stopped moving, many windows will have converged on the exact same peak. Merge all windows that end up at the same peak into a single cluster.

---

## 3. The Mathematics: Kernel Density Estimation

Given $n$ data points $x_i$ in a $d$-dimensional space, the multivariate kernel density estimator with a kernel $K(x)$ and a bandwidth $h$ is:

$$ f(x) = \frac{1}{nh^d} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right) $$

To find the peaks (modes), we want to find where the gradient of the density function is zero ($\nabla f(x) = 0$). 
Using a typical Gaussian Kernel, the gradient yields the **mean shift vector**:

$$ m(x) = \frac{\sum_{x_i \in N(x)} K(x_i - x) x_i}{\sum_{x_i \in N(x)} K(x_i - x)} - x $$

Where $N(x)$ is the neighborhood of points within the bandwidth $h$.
The algorithm simply updates $x \leftarrow x + m(x)$ iteratively. The first term is simply the weighted mean of the data points in the neighborhood!

---

## 4. Python Implementation (From Scratch)

To truly understand it, let's implement a simplified, non-optimized version.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class MeanShiftFromScratch:
    def __init__(self, bandwidth=2.0):
        self.bandwidth = bandwidth
        self.centroids = {}

    def fit(self, data):
        # 1. Initialize a window for every single data point
        centroids = {i: list(point) for i, point in enumerate(data)}
        
        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                
                # 2. Find all points within the bandwidth
                for feature_set in data:
                    if np.linalg.norm(np.array(feature_set) - np.array(centroid)) < self.bandwidth:
                        in_bandwidth.append(feature_set)
                
                # 3. Calculate new mean and shift
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))
            
            # 4. Filter out duplicate centroids (merge windows that hit the same peak)
            uniques = sorted(list(set(new_centroids)))
            
            prev_centroids = dict(centroids)
            centroids = {i: list(point) for i, point in enumerate(uniques)}
            
            # 5. Check for convergence (if centroids stopped moving)
            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids.get(i, [])):
                    optimized = False
                    break
                    
            if optimized:
                break
                
        self.centroids = centroids

# Test the algorithm
X, _ = make_blobs(n_samples=100, centers=3, cluster_std=0.8, random_state=42)
ms = MeanShiftFromScratch(bandwidth=3)
ms.fit(X)

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, label='Data points')
for c in ms.centroids:
    plt.scatter(ms.centroids[c][0], ms.centroids[c][1], color='red', marker='X', s=200)
plt.title("Mean Shift From Scratch (Red X = Discovered Peaks)")
plt.legend()
# plt.show()
```

---

## 5. Scikit-Learn Implementation

In practice, we use `scikit-learn`'s optimized version, which includes a function `estimate_bandwidth` to automatically find a good window size!

```python
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# 1. Generate Data
X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.6, random_state=42)

# 2. Estimate Bandwidth automatically! (This is a huge advantage)
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=100)
print(f"Estimated Bandwidth: {bandwidth:.2f}")

# 3. Fit Mean Shift
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)

labels = ms.labels_
cluster_centers = ms.cluster_centers_
n_clusters_ = len(np.unique(labels))
print(f"Number of estimated clusters: {n_clusters_}")

# 4. Visualization
plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40, alpha=0.6)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Centroids')
plt.title(f"Mean Shift Clustering (Discovered {n_clusters_} Clusters)", fontsize=14, fontweight='bold')
plt.legend()
plt.savefig('mean_shift.png', dpi=150)
# plt.show()
```

---

## 6. Real-World Applications

- 🎥 **Object Tracking in Video (CamShift)**: If you identify a red car in frame 1, Mean Shift can be used to track the "highest density of red pixels" in subsequent frames, effectively tracking the moving car!
- 🖼️ **Image Segmentation**: Smoothing an image while preserving edges by clustering pixels that are close in both location and color space.

---

## 7. Advantages & Limitations

### ✅ Advantages
- **No need to specify K**: Discovers the number of clusters automatically.
- **Handles complex shapes**: Since it relies on density, it can handle non-spherical clusters.
- **Robust to Outliers**: Outliers generally do not impact the peaks (modes) of the dense regions.

### ❌ Limitations
- **Computationally Expensive**: The naive approach is $O(T \cdot N^2)$, where $T$ is the number of iterations and $N$ is the number of data points. This makes it very slow for large datasets compared to K-Means.
- **Highly Sensitive to Bandwidth**: The entire algorithm hinges on the `bandwidth` parameter. A window too large merges distinct clusters; a window too small fragments single clusters into many.
- **Curse of Dimensionality**: Kernel density estimation struggles in high dimensions, making Mean Shift less effective for data with many features.

---

[← DBSCAN Clustering](./03-DBSCAN-Clustering.md) | [Back to Index](../README.md) | [Next: Gaussian Mixture Models →](./05-Gaussian-Mixture-Models.md)
