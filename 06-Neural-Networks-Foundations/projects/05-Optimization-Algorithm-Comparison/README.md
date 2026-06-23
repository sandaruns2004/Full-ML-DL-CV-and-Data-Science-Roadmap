# 🏎️ Project 05: Optimization Algorithm Visualizer

## 🎯 Goal
Compare how different optimization algorithms navigate complex 3D mathematical surfaces.

## 📝 Description
If you read the theory, you know Adam is "better" than SGD. But what does that actually look like? In this project, you will define complex 2D cost functions (like the Beale function or Ackley function) that are famous for having "false valleys" and local minima. You will then drop different optimizers onto the surface and plot their paths.

## ✅ Requirements
- Implement standard SGD, SGD with Momentum, RMSProp, and Adam.
- Define a complex mathematical surface with a known global minimum.
- Plot a 3D surface or 2D contour map using `matplotlib`.
- Animate or plot the trajectory of each algorithm as it attempts to find the bottom of the valley.
- Write a brief summary comparing which algorithms got stuck, which were fastest, and which took the smoothest paths.

## 📁 Files
- `optimizer_race.ipynb`
- `surfaces.py`
