# 🐍 Python for Data Science

> **Prerequisites**: [Python Essentials](../00-Prerequisites/01-Python-Essentials.md) | **Difficulty**: ⭐☆☆☆☆ Beginner

---

## 📋 Table of Contents

1. [The Data Science Environment](#1-the-data-science-environment)
2. [Python Refresher for DS](#2-python-refresher-for-ds)
3. [Working with Modules & Files](#3-working-with-modules--files)
4. [Introduction to Vectorization (NumPy Core)](#4-introduction-to-vectorization-numpy-core)
5. [Data Tabulation (Pandas Core)](#5-data-tabulation-pandas-core)
6. [What's Next](#6-whats-next)

---

## 1. The Data Science Environment

### 🟢 Beginner

In Data Science, we rarely write scripts in Notepad and run them in the terminal. We use environments that allow us to interact with our data, visualize it immediately, and document our process.

**The Holy Trinity of DS Environments:**

1. **Jupyter Notebooks (`.ipynb`)**
   - The industry standard for exploration
   - Mixes Python code, text (Markdown), and visual outputs
   - Great for sharing results and storytelling

2. **Virtual Environments**
   - Data Science projects use many libraries. They often conflict!
   - A virtual environment is an isolated sandbox for your project's dependencies
   - Tools: `venv`, `conda`, `pipenv`

3. **IDEs (Integrated Development Environments)**
   - VS Code, PyCharm
   - Used when moving from exploration (Jupyter) to production code (Scripts/APIs)

### 🟡 Intermediate

**Setting up a robust project:**

Always define your dependencies clearly. In this module, we use `requirements.txt`:

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate it (Windows)
venv\Scripts\activate
# Or macOS/Linux: source venv/bin/activate

# 3. Install packages
pip install -r requirements.txt

# 4. Start Jupyter
jupyter notebook
```

**Why `conda` vs `pip`?**
- `pip` installs Python packages.
- `conda` installs packages and their non-Python dependencies (like C++ libraries). In Data Science, `conda` is often preferred because scientific libraries rely heavily on compiled C/C++/Fortran code.

### 🔴 Advanced

**Managing reproducible environments:**

In production, `requirements.txt` is often not enough because it doesn't freeze sub-dependencies securely.

Tools for production ML environments:
- **Poetry / Pipenv**: Better dependency resolution and locking.
- **Docker**: Containerizes the entire OS, Python runtime, and libraries. This guarantees that if it works on your machine, it works on the deployment server.

---

## 2. Python Refresher for DS

> 💡 *For a full Python basics tutorial, see [00-Prerequisites/01-Python-Essentials.md](../00-Prerequisites/01-Python-Essentials.md).*

### 🟢 Beginner

In Data Science, we process large collections of items. You must be comfortable with:

**List Comprehensions** (The Pythonic way to transform data):
```python
# The old way (slow and verbose)
prices = [10.99, 25.50, 5.00]
taxed_prices = []
for p in prices:
    taxed_prices.append(p * 1.08)

# The DS way (fast and clean)
taxed_prices = [p * 1.08 for p in prices]
```

**Dictionaries** (Used constantly for configurations and JSON data):
```python
user_data = {
    "user_101": {"age": 25, "purchases": [10, 50, 100]},
    "user_102": {"age": 30, "purchases": [5, 20]}
}

# Accessing nested data
user_101_total = sum(user_data["user_101"]["purchases"])
```

### 🟡 Intermediate

**Lambda Functions, Map, and Filter**

When cleaning data in Pandas, you will frequently apply functions to entire columns. Lambda functions are perfect for this.

```python
# Lambda: a small, anonymous function
# Syntax: lambda arguments: expression

# Normal function
def is_expensive(price):
    return price > 20

# Lambda equivalent
is_expensive_lambda = lambda p: p > 20

# Using map (apply to all)
prices = [10, 25, 5, 40]
discounted = list(map(lambda p: p * 0.9, prices))  # [9.0, 22.5, 4.5, 36.0]

# Using filter (keep matching)
expensive = list(filter(lambda p: p > 20, prices)) # [25, 40]
```

### 🔴 Advanced

**Generators for Big Data**

If you load a 10GB CSV file into a Python list, your computer will crash (Out of Memory). Generators yield one item at a time, keeping memory usage near zero.

```python
# List comprehension (Memory intensive)
# Loads all 10 million squares into RAM at once
squares_list = [x**2 for x in range(10_000_000)] 

# Generator expression (Memory efficient)
# Calculates each square ONLY when requested
squares_gen = (x**2 for x in range(10_000_000))

# Get the next value
print(next(squares_gen)) # 0
print(next(squares_gen)) # 1
```

---

## 3. Working with Modules & Files

### 🟢 Beginner

Data comes in files. Python has built-in modules to handle them.

```python
import os
import csv
import json

# Reading a simple text file
with open('data.txt', 'r') as file:
    content = file.read()
    
# 'with' statement ensures the file closes automatically!
```

### 🟡 Intermediate

**Parsing CSV and JSON** (The two most common data formats)

```python
import csv
import json

# Reading CSV manually
with open('users.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row['name'], row['age'])

# Reading JSON (APIs always return JSON)
json_string = '{"name": "Alice", "score": 95}'
data = json.loads(json_string)  # String to Python Dict

# Writing JSON
with open('output.json', 'w') as f:
    json.dump(data, f, indent=4) # Python Dict to File
```

### 🔴 Advanced

**File Paths and the OS Module**

Hardcoding paths (`C:/Users/name/data.csv`) breaks your code when someone else runs it on a Mac or Linux machine. Always use `os.path` or `pathlib`.

```python
import os
from pathlib import Path

# The modern, robust way to handle paths (Pathlib)
# This works identically on Windows, Mac, and Linux
current_dir = Path.cwd()
data_file = current_dir / "data" / "dataset.csv"

# Check if file exists before trying to read
if data_file.exists():
    print(f"Ready to load {data_file.name}")
else:
    print("Data file missing!")
```

---

## 4. Introduction to Vectorization (NumPy Core)

> 💡 *For deep math and matrix operations, see [00-Prerequisites/01-Python-Essentials.md](../00-Prerequisites/01-Python-Essentials.md).*

### 🟢 Beginner

Python `for` loops are extremely slow for math. NumPy solves this using **Vectorization** — applying an operation to an entire array at once using optimized C code.

```python
import numpy as np
import time

size = 1_000_000

# SLOW: Python List
py_list = list(range(size))
start = time.time()
py_result = [x * 5 for x in py_list]
print(f"Python Loop: {time.time() - start:.4f} seconds")

# FAST: NumPy Array
np_array = np.arange(size)
start = time.time()
np_result = np_array * 5  # <--- VECTORIZATION
print(f"NumPy Vectorization: {time.time() - start:.4f} seconds")

# NumPy is typically 50-100x faster!
```

### 🟡 Intermediate

**Array Operations without Loops**

```python
import numpy as np

# Creating arrays
data = np.array([10, 20, 30, 40, 50])

# Math operations apply to EVERY element instantly
print(data + 5)    # [15, 25, 35, 45, 55]
print(data * 2)    # [20, 40, 60, 80, 100]

# Boolean Masking (Filtering without loops)
# This is how we filter data in Data Science!
mask = data > 25
print(mask)        # [False, False, True, True, True]
print(data[mask])  # [30, 40, 50]
```

---

## 5. Data Tabulation (Pandas Core)

> 💡 *For advanced data manipulation, merges, and groupings, see [00-Prerequisites/01-Python-Essentials.md](../00-Prerequisites/01-Python-Essentials.md).*

### 🟢 Beginner

If NumPy is for arrays of numbers, **Pandas** is for tables (like Excel). It is built on top of NumPy.

```python
import pandas as pd

# Creating a DataFrame (a table)
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 75000]
}
df = pd.DataFrame(data)

print(df)
#       Name  Age  Salary
# 0    Alice   25   50000
# 1      Bob   30   60000
# 2  Charlie   35   75000
```

### 🟡 Intermediate

**Loading and inspecting real data**

Instead of Python `csv` modules, Pandas can read a file in one line of code.

```python
import pandas as pd

# Load from file or URL
# df = pd.read_csv('customers.csv')

# Quick inspection methods you will use every day:
# df.head(5)        # View first 5 rows
# df.info()         # View data types and missing values count
# df.describe()     # View statistical summary (mean, min, max)

# Selecting data
# ages = df['Age']                 # Select one column (Returns a Series)
# subset = df[['Name', 'Salary']]  # Select multiple columns (Returns a DataFrame)
```

### 🔴 Advanced

**The Power of Pandas Vectorization**

Just like NumPy, never use `for` loops in Pandas. Use vectorized string methods and math.

```python
import pandas as pd

df = pd.DataFrame({
    'Full_Name': [' alice smith ', 'BOB JONES', 'charlie Brown'],
    'Price': [10.99, 25.50, 5.00]
})

# SLOW way (Don't do this!)
# for i in range(len(df)):
#     df.loc[i, 'Name'] = df.loc[i, 'Name'].strip().title()

# FAST way (Vectorized String Methods)
df['Clean_Name'] = df['Full_Name'].str.strip().str.title()

# Applying custom logic fast
df['Category'] = df['Price'].apply(lambda x: 'Expensive' if x > 20 else 'Cheap')

print(df[['Clean_Name', 'Category']])
```

---

## 6. What's Next

Now that you understand the environment and basic Python data tools, you need to get some data!

| Next Topic | Why |
|------------|-----|
| [Data Collection](./03-Data-Collection.md) | Learn to scrape the web, hit APIs, and query databases to build datasets |
| [Data Cleaning](./04-Data-Cleaning.md) | Real data is messy. Learn how to handle missing values and outliers |

---

[← Previous: Introduction to Data Science](./01-Introduction-to-Data-Science.md) | [Back to Main Index](../README.md) | [Next: Data Collection →](./03-Data-Collection.md)
