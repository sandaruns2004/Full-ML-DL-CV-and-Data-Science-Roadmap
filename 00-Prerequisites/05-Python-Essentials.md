# 🐍 Python Essentials for Machine Learning

> **Prerequisites**: Basic programming knowledge | **Difficulty**: ⭐☆☆☆☆ Beginner

---

## 📋 Table of Contents

1. [Why Python for ML?](#1-why-python-for-ml)
2. [Python Data Structures Review](#2-python-data-structures-review)
3. [File & Error Handling](#3-file--error-handling)
4. [NumPy — Numerical Computing](#4-numpy--numerical-computing)
5. [Pandas — Data Manipulation](#5-pandas--data-manipulation)
6. [Matplotlib & Seaborn — Visualization](#6-matplotlib--seaborn--visualization)
7. [SciPy — Scientific Computing](#7-scipy--scientific-computing)
8. [Python Best Practices for ML](#8-python-best-practices-for-ml)
9. [Project Ideas](#9-project-ideas)
10. [What's Next](#10-whats-next)

---

## 1. Why Python for ML?

Python dominates Machine Learning for several reasons:

- **Rich ecosystem**: NumPy, Pandas, scikit-learn, TensorFlow, PyTorch
- **Readable syntax**: Code reads like pseudocode from research papers
- **Community**: Largest ML community, most tutorials and resources
- **Interoperability**: C/C++ extensions for performance-critical code
- **Jupyter Notebooks**: Interactive exploration and visualization

### The ML Python Stack

```
┌─────────────────────────────────────────┐
│         Applications Layer              │
│   scikit-learn │ TensorFlow │ PyTorch   │
├─────────────────────────────────────────┤
│         Data Layer                      │
│   Pandas │ Polars │ SQL Alchemy         │
├─────────────────────────────────────────┤
│         Computation Layer               │
│   NumPy │ SciPy │ CuPy (GPU)           │
├─────────────────────────────────────────┤
│         Visualization Layer             │
│   Matplotlib │ Seaborn │ Plotly         │
├─────────────────────────────────────────┤
│         Python Core                     │
└─────────────────────────────────────────┘
```

---

## 2. Python Data Structures Review

Before diving into ML libraries, let's review the Python data structures you'll use constantly.

### 2.1 Lists, Tuples, and Dictionaries

```python
# Lists — mutable, ordered collections
features = [1.5, 2.3, 0.7, 4.1]
features.append(3.2)
print(f"Features: {features}")  # [1.5, 2.3, 0.7, 4.1, 3.2]

# List comprehensions — the Pythonic way
squared = [x**2 for x in range(10)]
even_squared = [x**2 for x in range(10) if x % 2 == 0]
print(f"Squared: {squared}")
print(f"Even squared: {even_squared}")

# Tuples — immutable, used for fixed data
image_shape = (224, 224, 3)  # height, width, channels
batch_size, num_classes = 32, 10  # tuple unpacking

# Dictionaries — key-value pairs (used everywhere in ML configs)
hyperparameters = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'optimizer': 'adam',
    'layers': [128, 64, 32]
}

# Accessing and iterating
for param, value in hyperparameters.items():
    print(f"  {param}: {value}")
```

### 2.2 Lambda Functions and Map/Filter

```python
# Lambda functions — anonymous, one-liner functions
square = lambda x: x ** 2
print(f"Square of 5: {square(5)}")  # 25

# Map — apply function to every element
temperatures_f = [32, 68, 100, 212]
temperatures_c = list(map(lambda f: (f - 32) * 5/9, temperatures_f))
print(f"Celsius: {temperatures_c}")  # [0.0, 20.0, 37.78, 100.0]

# Filter — keep elements that satisfy condition
scores = [85, 42, 91, 67, 73, 55, 88]
passing = list(filter(lambda x: x >= 70, scores))
print(f"Passing scores: {passing}")  # [85, 91, 73, 88]

# Zip — combine iterables
names = ['Alice', 'Bob', 'Charlie']
scores = [92, 85, 78]
student_scores = dict(zip(names, scores))
print(student_scores)  # {'Alice': 92, 'Bob': 85, 'Charlie': 78}
```

### 2.3 Generators (Memory Efficient for Large Datasets)

```python
# Generator for large datasets — doesn't load everything into memory
def data_generator(file_path, batch_size=32):
    """Yield batches of data from a file."""
    batch = []
    with open(file_path, 'r') as f:
        for line in f:
            batch.append(line.strip().split(','))
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:  # Don't forget the last partial batch
            yield batch

# Usage (conceptual — no actual file needed)
# for batch in data_generator('large_dataset.csv', batch_size=64):
#     process(batch)

# Generator expression — like list comprehension but lazy
squares_gen = (x**2 for x in range(1_000_000))  # No memory allocation!
first_10 = [next(squares_gen) for _ in range(10)]
print(f"First 10 squares: {first_10}")
```

---

## 3. File & Error Handling

Handling files properly and catching errors is crucial when dealing with real-world, messy datasets.

### 3.1 File Handling

Python provides built-in functions for reading and writing files. The `with` statement ensures files are automatically closed.

```python
# Writing to a file
with open('config.txt', 'w') as file:
    file.write("learning_rate=0.01\n")
    file.write("epochs=100\n")

# Reading from a file
with open('config.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        print(line.strip())  # strip() removes the newline character

# Appending to a file
with open('config.txt', 'a') as file:
    file.write("batch_size=32\n")
```

### 3.2 Error Handling (Exceptions)

If a file is missing or data is corrupted, you don't want your entire 48-hour training loop to crash. Use `try...except` blocks.

```python
import os

def load_dataset(file_path):
    try:
        # Code that might raise an error
        with open(file_path, 'r') as file:
            data = file.read()
            return data
    except FileNotFoundError:
        # What to do if the file isn't found
        print(f"Error: The dataset at {file_path} was not found.")
        return None
    except Exception as e:
        # Catch any other unexpected error
        print(f"An unexpected error occurred: {e}")
        return None
    finally:
        # This block always executes
        print("Finished data loading attempt.")

data = load_dataset("non_existent_file.csv")
```

---

## 4. NumPy — Numerical Computing

NumPy is the **foundation of all numerical computing in Python**. Every ML library builds on top of NumPy arrays.

### 3.1 Why NumPy? (Speed Comparison)

```python
import numpy as np
import time

# Pure Python vs NumPy speed comparison
size = 1_000_000

# Python list operation
python_list = list(range(size))
start = time.time()
python_result = [x * 2 for x in python_list]
python_time = time.time() - start

# NumPy operation
numpy_array = np.arange(size)
start = time.time()
numpy_result = numpy_array * 2
numpy_time = time.time() - start

print(f"Python list: {python_time:.4f}s")
print(f"NumPy array: {numpy_time:.4f}s")
print(f"NumPy is {python_time/numpy_time:.1f}x faster!")
# Typically NumPy is 50-100x faster
```

**Why is NumPy faster?**
- **Vectorization**: Operations on entire arrays at once (no Python loops)
- **Contiguous memory**: Arrays stored in continuous memory blocks
- **C backend**: Core operations written in C/Fortran
- **SIMD**: Uses CPU vector instructions for parallel arithmetic

### 3.2 Array Creation

```python
import numpy as np

# From Python lists
a = np.array([1, 2, 3, 4, 5])
print(f"1D array: {a}, shape: {a.shape}, dtype: {a.dtype}")

# 2D array (matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
print(f"2D array shape: {matrix.shape}")  # (3, 3)

# Common creation functions
zeros = np.zeros((3, 4))          # 3x4 matrix of zeros
ones = np.ones((2, 3))            # 2x3 matrix of ones
identity = np.eye(4)              # 4x4 identity matrix
range_arr = np.arange(0, 10, 2)   # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)   # [0, 0.25, 0.5, 0.75, 1.0]

# Random arrays (crucial for ML)
np.random.seed(42)  # For reproducibility
uniform = np.random.rand(3, 3)           # Uniform [0, 1)
normal = np.random.randn(3, 3)           # Standard normal
integers = np.random.randint(0, 100, (3, 3))  # Random integers
print(f"Random normal:\n{normal}")

# Special: Create arrays like another array
template = np.array([[1, 2], [3, 4]])
zeros_like = np.zeros_like(template)
ones_like = np.ones_like(template)
```

### 3.3 Array Operations (Vectorized)

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Element-wise operations (these are VECTORIZED — no loops!)
print(f"Add:      {a + b}")        # [11, 22, 33, 44, 55]
print(f"Multiply: {a * b}")        # [10, 40, 90, 160, 250]
print(f"Power:    {a ** 2}")        # [1, 4, 9, 16, 25]
print(f"Sqrt:     {np.sqrt(a)}")    # [1.0, 1.41, 1.73, 2.0, 2.24]
print(f"Exp:      {np.exp(a)}")     # [2.72, 7.39, 20.09, 54.60, 148.41]
print(f"Log:      {np.log(a)}")     # [0.0, 0.69, 1.10, 1.39, 1.61]

# Comparison operations (return boolean arrays)
print(f"a > 3:    {a > 3}")         # [False, False, False, True, True]
print(f"a == 3:   {a == 3}")        # [False, False, True, False, False]

# Statistical operations
data = np.array([23, 45, 12, 67, 34, 89, 56, 78])
print(f"Mean:   {np.mean(data):.2f}")       # 50.50
print(f"Median: {np.median(data):.2f}")     # 50.50
print(f"Std:    {np.std(data):.2f}")        # 24.27
print(f"Var:    {np.var(data):.2f}")        # 589.00
print(f"Min:    {np.min(data)}")            # 12
print(f"Max:    {np.max(data)}")            # 89
print(f"ArgMax: {np.argmax(data)}")         # 5 (index of max)
print(f"Sum:    {np.sum(data)}")            # 404
```

### 3.4 Indexing and Slicing

```python
import numpy as np

# 1D Indexing
a = np.array([10, 20, 30, 40, 50, 60, 70, 80])
print(f"a[0] = {a[0]}")      # 10
print(f"a[-1] = {a[-1]}")    # 80
print(f"a[2:5] = {a[2:5]}")  # [30, 40, 50]
print(f"a[::2] = {a[::2]}")  # [10, 30, 50, 70] (every 2nd)

# 2D Indexing
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(f"Row 0:      {matrix[0]}")         # [1, 2, 3, 4]
print(f"Col 1:      {matrix[:, 1]}")      # [2, 6, 10]
print(f"Element:    {matrix[1, 2]}")      # 7
print(f"Submatrix:\n{matrix[0:2, 1:3]}")  # [[2, 3], [6, 7]]

# Boolean indexing (CRITICAL for ML data filtering)
data = np.array([15, 22, 8, 45, 11, 67, 3, 89])
mask = data > 20
print(f"Mask: {mask}")                    # [F, T, F, T, F, T, F, T]
print(f"Filtered: {data[mask]}")          # [22, 45, 67, 89]

# Fancy indexing
indices = np.array([0, 3, 5])
print(f"Fancy: {data[indices]}")          # [15, 45, 67]
```

### 3.5 Broadcasting

Broadcasting allows NumPy to perform operations on arrays of different shapes. This is **essential** for ML computations.

```python
import numpy as np

# Scalar broadcast
a = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"a + 10:\n{a + 10}")
# [[11, 12, 13],
#  [14, 15, 16]]

# Column vector broadcast
col = np.array([[100], [200]])  # Shape: (2, 1)
print(f"a + col:\n{a + col}")
# [[101, 102, 103],
#  [204, 205, 206]]

# Row vector broadcast
row = np.array([10, 20, 30])  # Shape: (3,)
print(f"a + row:\n{a + row}")
# [[11, 22, 33],
#  [14, 25, 36]]

# Broadcasting rules:
# 1. Arrays with fewer dims get 1s prepended to their shape
# 2. Arrays with size 1 along a dimension are stretched
# 3. If sizes don't match and neither is 1 → ERROR
#
# Example: (2, 3) + (3,) → (2, 3) + (1, 3) → (2, 3) + (2, 3) ✓
# Example: (2, 3) + (2,) → (2, 3) + (1, 2) → ERROR (3 ≠ 2)
```

### 3.6 Reshaping and Manipulation

```python
import numpy as np

a = np.arange(12)
print(f"Original: {a}")  # [0, 1, 2, ..., 11]

# Reshape
b = a.reshape(3, 4)      # 3 rows, 4 columns
print(f"Reshaped (3,4):\n{b}")

c = a.reshape(2, 2, 3)   # 3D: 2 blocks of 2x3
print(f"Reshaped (2,2,3):\n{c}")

# Use -1 to auto-calculate one dimension
d = a.reshape(-1, 3)     # Auto-calculate rows: 12/3 = 4 rows
print(f"Auto-reshape:\n{d}")

# Flatten (multi-dim → 1D)
flat = b.flatten()        # Returns a copy
ravel = b.ravel()         # Returns a view (faster, no copy)

# Transpose
print(f"Transpose:\n{b.T}")  # (4, 3)

# Concatenation
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])
vertical = np.vstack([x, y])      # Stack vertically → (4, 2)
horizontal = np.hstack([x, y])    # Stack horizontally → (2, 4)
print(f"vstack:\n{vertical}")
print(f"hstack:\n{horizontal}")

# Adding dimensions (critical for DL)
vec = np.array([1, 2, 3])          # Shape: (3,)
row_vec = vec[np.newaxis, :]       # Shape: (1, 3)
col_vec = vec[:, np.newaxis]       # Shape: (3, 1)
print(f"row shape: {row_vec.shape}, col shape: {col_vec.shape}")
```

### 3.7 Linear Algebra with NumPy

```python
import numpy as np

# Matrix multiplication (THE most common ML operation)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Three equivalent ways:
result1 = np.dot(A, B)
result2 = A @ B          # Preferred syntax (Python 3.5+)
result3 = np.matmul(A, B)
print(f"A @ B:\n{result1}")
# [[19, 22],
#  [43, 50]]

# Key difference: * is element-wise, @ is matrix multiply
print(f"A * B (element-wise):\n{A * B}")  # [[5, 12], [21, 32]]
print(f"A @ B (matrix):\n{A @ B}")        # [[19, 22], [43, 50]]

# Determinant
det = np.linalg.det(A)
print(f"det(A) = {det:.2f}")  # -2.0

# Inverse
A_inv = np.linalg.inv(A)
print(f"A inverse:\n{A_inv}")
print(f"A @ A_inv:\n{A @ A_inv}")  # ≈ Identity

# Eigenvalues and eigenvectors (used in PCA!)
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Solving linear systems: Ax = b
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print(f"Solution x: {x}")

# Singular Value Decomposition (used in recommendation systems)
U, S, Vt = np.linalg.svd(A)
print(f"U:\n{U}\nS: {S}\nVt:\n{Vt}")

# Norms (measuring vector/matrix size)
v = np.array([3, 4])
print(f"L2 norm: {np.linalg.norm(v)}")      # 5.0 (Euclidean distance)
print(f"L1 norm: {np.linalg.norm(v, 1)}")   # 7.0 (Manhattan distance)
print(f"Linf norm: {np.linalg.norm(v, np.inf)}")  # 4.0 (max absolute)
```

---

## 5. Pandas — Data Manipulation

Pandas is the **standard tool for data manipulation** in ML. You'll use it to load, clean, explore, and transform datasets.

### 4.1 Series and DataFrames

```python
import pandas as pd
import numpy as np

# Series — 1D labeled array
scores = pd.Series([85, 92, 78, 95, 88], 
                    index=['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
                    name='test_score')
print(scores)
print(f"\nMean: {scores.mean():.1f}")
print(f"Above 85: {scores[scores > 85]}")

# DataFrame — 2D labeled table (THIS is what you'll use most)
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'salary': [50000, 60000, 75000, 55000, 65000],
    'department': ['Engineering', 'Marketing', 'Engineering', 'HR', 'Marketing'],
    'experience_years': [3, 5, 10, 4, 7]
}
df = pd.DataFrame(data)
print(df)
print(f"\nShape: {df.shape}")    # (5, 5)
print(f"Columns: {list(df.columns)}")
print(f"Types:\n{df.dtypes}")
```

### 4.2 Loading Data

```python
import pandas as pd

# CSV — most common for ML datasets
# df = pd.read_csv('data.csv')
# df = pd.read_csv('data.csv', sep=';', header=0, na_values=['?', 'NA'])

# Excel
# df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# JSON
# df = pd.read_json('data.json')

# From URL
# df = pd.read_csv('https://raw.githubusercontent.com/.../data.csv')

# From sklearn datasets (great for practice!)
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(df.head())
print(f"\nShape: {df.shape}")
print(f"\n{df.describe()}")
```

### 4.3 Data Exploration (EDA Essentials)

```python
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Quick overview
print(df.head(10))           # First 10 rows
print(df.tail(5))            # Last 5 rows
print(df.info())             # Data types, non-null counts
print(df.describe())         # Statistical summary

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"Total missing: {df.isnull().sum().sum()}")

# Value counts (for categorical columns)
print(f"\nTarget distribution:\n{df['target'].value_counts()}")

# Correlation matrix (for numerical columns)
print(f"\nCorrelation:\n{df.corr(numeric_only=True).round(2)}")

# Unique values
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
```

### 4.4 Selecting, Filtering, and Sorting

```python
import pandas as pd
import numpy as np

# Create sample dataset
np.random.seed(42)
df = pd.DataFrame({
    'student': [f'Student_{i}' for i in range(1, 11)],
    'math': np.random.randint(50, 100, 10),
    'science': np.random.randint(50, 100, 10),
    'english': np.random.randint(50, 100, 10),
    'grade': np.random.choice(['A', 'B', 'C'], 10)
})

# Column selection
print(df['math'])                    # Single column (Series)
print(df[['math', 'science']])       # Multiple columns (DataFrame)

# Row selection
print(df.iloc[0])        # First row by position
print(df.iloc[0:3])      # First 3 rows
print(df.loc[0])          # First row by label
print(df.loc[0:2, 'math':'science'])  # Rows 0-2, columns math to science

# Filtering
high_math = df[df['math'] > 80]
print(f"High math scores:\n{high_math}")

# Multiple conditions (use & for AND, | for OR, ~ for NOT)
smart_students = df[(df['math'] > 70) & (df['science'] > 70)]
print(f"\nGood at both:\n{smart_students}")

# Sorting
sorted_df = df.sort_values('math', ascending=False)
print(f"\nSorted by math (desc):\n{sorted_df}")

# Multi-column sort
sorted_df = df.sort_values(['grade', 'math'], ascending=[True, False])
```

### 4.5 Data Cleaning and Transformation

```python
import pandas as pd
import numpy as np

# Create messy data (realistic!)
messy_data = pd.DataFrame({
    'name': ['Alice', 'BOB', 'charlie', None, 'Eve'],
    'age': [25, -5, 30, 28, 200],
    'salary': [50000, np.nan, 75000, np.nan, 65000],
    'city': ['New York', 'new york', 'NEW YORK', 'London', 'London ']
})

print("Before cleaning:")
print(messy_data)

# Handle missing values
messy_data['salary'] = messy_data['salary'].fillna(messy_data['salary'].median())
messy_data['name'] = messy_data['name'].fillna('Unknown')

# Fix inconsistent text
messy_data['name'] = messy_data['name'].str.title()
messy_data['city'] = messy_data['city'].str.strip().str.title()

# Fix invalid values
messy_data.loc[messy_data['age'] < 0, 'age'] = np.nan
messy_data.loc[messy_data['age'] > 120, 'age'] = np.nan
messy_data['age'] = messy_data['age'].fillna(messy_data['age'].median())

print("\nAfter cleaning:")
print(messy_data)

# Apply custom function
messy_data['salary_category'] = messy_data['salary'].apply(
    lambda x: 'High' if x > 60000 else 'Low'
)

# Group by and aggregate
summary = messy_data.groupby('city').agg({
    'salary': ['mean', 'count'],
    'age': 'mean'
}).round(2)
print(f"\nGrouped summary:\n{summary}")
```

### 4.6 Merging and Joining

```python
import pandas as pd

# Two related tables
students = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'major': ['CS', 'Math', 'CS', 'Physics', 'Math']
})

grades = pd.DataFrame({
    'student_id': [1, 2, 3, 1, 2, 3, 4],
    'course': ['ML', 'ML', 'ML', 'Stats', 'Stats', 'Stats', 'ML'],
    'grade': [95, 88, 92, 85, 90, 78, 96]
})

# Inner join (only matching rows)
merged = pd.merge(students, grades, left_on='id', right_on='student_id')
print(f"Inner merge:\n{merged}")

# Left join (keep all students)
merged_left = pd.merge(students, grades, left_on='id', right_on='student_id', how='left')
print(f"\nLeft merge:\n{merged_left}")

# Pivot table (super useful for EDA)
pivot = grades.pivot_table(values='grade', index='student_id', columns='course', aggfunc='mean')
print(f"\nPivot table:\n{pivot}")
```

---

## 6. Matplotlib & Seaborn — Visualization

Visualization is **critical** for understanding data and communicating results.

### 5.1 Matplotlib Basics

```python
import matplotlib.pyplot as plt
import numpy as np

# Basic line plot
x = np.linspace(0, 2 * np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y_sin, label='sin(x)', color='#2196F3', linewidth=2)
ax.plot(x, y_cos, label='cos(x)', color='#FF5722', linewidth=2, linestyle='--')
ax.set_xlabel('x (radians)', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Sine and Cosine Functions', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('trig_functions.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.2 Common ML Plots

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Scatter plot (relationship between features)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5
axes[0, 0].scatter(x, y, c='#2196F3', alpha=0.6, edgecolors='white')
axes[0, 0].set_title('Scatter Plot (Feature Relationship)')
axes[0, 0].set_xlabel('Feature 1')
axes[0, 0].set_ylabel('Feature 2')

# 2. Histogram (feature distribution)
data = np.random.randn(1000)
axes[0, 1].hist(data, bins=30, color='#4CAF50', edgecolor='white', alpha=0.8)
axes[0, 1].axvline(np.mean(data), color='red', linestyle='--', label=f'Mean: {np.mean(data):.2f}')
axes[0, 1].set_title('Histogram (Feature Distribution)')
axes[0, 1].legend()

# 3. Box plot (outlier detection)
data_groups = [np.random.randn(100) + i for i in range(4)]
bp = axes[0, 2].boxplot(data_groups, labels=['A', 'B', 'C', 'D'], patch_artist=True)
colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[0, 2].set_title('Box Plot (Outlier Detection)')

# 4. Bar chart (class distribution)
categories = ['Cat', 'Dog', 'Bird', 'Fish']
counts = [45, 38, 22, 15]
bars = axes[1, 0].bar(categories, counts, color=['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'])
axes[1, 0].set_title('Bar Chart (Class Distribution)')
for bar, count in zip(bars, counts):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     str(count), ha='center', fontweight='bold')

# 5. Heatmap (correlation matrix)
corr = np.random.rand(5, 5)
corr = (corr + corr.T) / 2  # Make symmetric
np.fill_diagonal(corr, 1)
im = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 1].set_title('Heatmap (Correlation Matrix)')
plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

# 6. Line plot (training curves)
epochs = range(1, 51)
train_loss = 2.0 * np.exp(-np.array(epochs) * 0.05) + np.random.randn(50) * 0.05
val_loss = 2.0 * np.exp(-np.array(epochs) * 0.04) + np.random.randn(50) * 0.08 + 0.1
axes[1, 2].plot(epochs, train_loss, label='Train Loss', color='#2196F3')
axes[1, 2].plot(epochs, val_loss, label='Val Loss', color='#FF5722')
axes[1, 2].set_title('Training Curves')
axes[1, 2].legend()
axes[1, 2].set_xlabel('Epoch')

plt.suptitle('Common ML Visualizations', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('ml_plots.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.3 Seaborn (Statistical Visualization)

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Set style
sns.set_style('whitegrid')
sns.set_palette('husl')

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Pair plot (relationships between ALL features)
# sns.pairplot(df, hue='species')  # Use this separately — creates its own figure

# 2. Distribution plot
sns.histplot(data=df, x='sepal length (cm)', hue='species', kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Distribution by Species')

# 3. Violin plot
sns.violinplot(data=df, x='species', y='petal length (cm)', ax=axes[0, 1])
axes[0, 1].set_title('Violin Plot')

# 4. Correlation heatmap
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0],
            square=True, linewidths=0.5)
axes[1, 0].set_title('Correlation Heatmap')

# 5. Joint plot (density + scatter)
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)',
                hue='species', style='species', s=80, ax=axes[1, 1])
axes[1, 1].set_title('Scatter with Species')

plt.tight_layout()
plt.savefig('seaborn_plots.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 7. SciPy — Scientific Computing

SciPy extends NumPy with advanced mathematical functions used in ML.

### 6.1 Key SciPy Modules for ML

```python
from scipy import stats
from scipy import optimize
from scipy.spatial import distance
import numpy as np

# --- Statistical Functions ---
# Normal distribution
mu, sigma = 0, 1
x = np.linspace(-4, 4, 100)
pdf = stats.norm.pdf(x, mu, sigma)  # Probability density function
cdf = stats.norm.cdf(x, mu, sigma)  # Cumulative distribution function

# Hypothesis testing
data1 = np.random.normal(5, 1, 100)
data2 = np.random.normal(5.5, 1, 100)
t_stat, p_value = stats.ttest_ind(data1, data2)
print(f"T-test: t={t_stat:.4f}, p={p_value:.4f}")
if p_value < 0.05:
    print("  → Statistically significant difference!")

# --- Distance Metrics ---
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(f"Euclidean: {distance.euclidean(a, b):.4f}")
print(f"Manhattan: {distance.cityblock(a, b):.4f}")
print(f"Cosine:    {distance.cosine(a, b):.4f}")

# --- Optimization (used in ML training!) ---
# Find minimum of a function
def loss_function(x):
    return (x - 3)**2 + 2 * np.sin(x)

result = optimize.minimize_scalar(loss_function, bounds=(0, 10), method='bounded')
print(f"Minimum at x = {result.x:.4f}, f(x) = {result.fun:.4f}")
```

---

## 8. Python Best Practices for ML

### 7.1 Project Structure

```
ml_project/
├── data/
│   ├── raw/              # Original, immutable data
│   ├── processed/        # Cleaned, transformed data
│   └── external/         # Data from third-party sources
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── visualization/
│       └── plots.py
├── models/               # Saved model files
├── results/              # Output files, figures
├── requirements.txt
├── README.md
└── config.yaml           # Hyperparameters, paths
```

### 7.2 Reproducibility

```python
import numpy as np
import random
import os

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # If using TensorFlow
    # import tensorflow as tf
    # tf.random.set_seed(seed)
    
    # If using PyTorch
    # import torch
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

set_seed(42)
print("All seeds set for reproducibility!")
```

### 7.3 Configuration Management

```python
# config.py — Centralized configuration
from dataclasses import dataclass

@dataclass
class ModelConfig:
    # Data
    data_path: str = 'data/raw/dataset.csv'
    test_size: float = 0.2
    random_state: int = 42
    
    # Model
    model_type: str = 'random_forest'
    n_estimators: int = 100
    max_depth: int = 10
    
    # Training
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    # Output
    model_save_path: str = 'models/best_model.pkl'
    results_path: str = 'results/'

config = ModelConfig()
print(f"Model type: {config.model_type}")
print(f"Learning rate: {config.learning_rate}")
```

---

## 9. Project Ideas

### 🟢 Project 1: Data Analysis Dashboard (Beginner)
**Goal**: Load a real dataset, clean it, explore it, and create insightful visualizations.
- **Dataset**: Titanic dataset (from Kaggle)
- **Skills**: Pandas, Matplotlib, Seaborn
- **Output**: A set of 10+ visualizations revealing survival patterns

### 🟡 Project 2: NumPy Linear Algebra Library (Intermediate)
**Goal**: Build your own mini linear algebra library using only NumPy.
- Implement matrix operations, decompositions, and solvers
- **Skills**: NumPy, linear algebra, testing
- **Output**: Reusable library with unit tests

### 🔴 Project 3: Custom Data Pipeline (Advanced)
**Goal**: Build an ETL pipeline that loads data from multiple sources, cleans, transforms, and exports.
- Handle CSV, JSON, and API data sources
- Implement data validation and error handling
- **Skills**: Pandas, generators, OOP
- **Output**: Production-ready data pipeline class

---

## 10. What's Next

After mastering these Python tools, you're ready to learn how to version control your code:

| Next Topic | Why |
|------------|-----|
| [Git and GitHub](./06-Git-And-GitHub.md) | Version control is essential for collaborating and saving your ML projects |

### 🛤️ Learning Pathway

```
You are here: Python Essentials
                    │
                    ▼
          Git and GitHub
                    │
                    ▼
          Linux Fundamentals
```

---

> **Key Takeaway**: NumPy arrays and Pandas DataFrames are the two data structures you'll use in 90% of your ML work. Master them, and everything else becomes easier.

---

[Back to Index](./README.md) | [Next: Git and GitHub →](./06-Git-And-GitHub.md)
