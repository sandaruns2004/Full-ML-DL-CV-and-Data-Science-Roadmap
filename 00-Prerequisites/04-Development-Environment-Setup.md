# ⚙️ 04. Development Environment Setup

> **Prerequisites**: Module 02 | **Difficulty**: ⭐☆☆☆☆ Beginner

To write and run AI code efficiently, you need a professional, scalable setup. Writing code in a basic text editor or running a single global Python installation will quickly lead to disaster. We will configure an industry-standard environment identical to what ML Engineers use at top tech companies.

---

## 📋 Table of Contents
1. [The "Dependency Hell" Problem](#1-the-dependency-hell-problem)
2. [Step 1: Install Miniconda](#2-step-1-install-miniconda)
3. [Step 2: Virtual Environments (The Solution)](#3-step-2-virtual-environments-the-solution)
4. [Step 3: Visual Studio Code (The IDE)](#4-step-3-visual-studio-code-the-ide)
5. [Step 4: Jupyter Notebooks (The Sandbox)](#5-step-4-jupyter-notebooks-the-sandbox)
6. [Step 5: Git (Version Control)](#6-step-5-git-version-control)

---

## 1. The "Dependency Hell" Problem

Why shouldn't you just go to `python.org`, download Python, and start `pip install`ing everything globally on your computer?

Imagine this scenario:
*   **Project A** (an older Computer Vision project) requires `TensorFlow 2.8` and `Numpy 1.20`.
*   **Project B** (a brand new LLM project) requires `PyTorch 2.1` and `Numpy 1.24`.

If you only have one global Python installation on your laptop, installing Project B's requirements will forcefully upgrade Numpy, instantly **breaking** Project A. This state of broken, conflicting packages is affectionately known as "Dependency Hell."

**The Solution:** Virtual Environments. We create isolated, walled-off "boxes" for every single project.

---

## 2. Step 1: Install Miniconda

Anaconda is a distribution of Python built specifically for Data Science. It excels at handling complex C/C++ backend dependencies (like GPU CUDA drivers) that standard `pip` struggles with.

We highly recommend **Miniconda** (a lightweight version of Anaconda that doesn't bloat your system with a massive GUI and 500 pre-installed packages).

1.  Navigate to the [Miniconda Download Page](https://docs.conda.io/en/latest/miniconda.html).
2.  Download the installer for your specific Operating System.
3.  Run the installer. 
    *   *Windows Users*: When asked, do **not** check "Add Miniconda3 to my PATH" unless you are advanced. Instead, you will use the "Anaconda Prompt" application installed on your start menu.
    *   *Mac/Linux Users*: Allow the installer to run `conda init` at the end.

**Verification**:
Open your terminal (or Anaconda Prompt on Windows) and type:
```bash
conda --version
# Expected Output: conda 23.x.x
```

---

## 3. Step 2: Virtual Environments (The Solution)

Let's create an isolated environment specifically for this roadmap repository.

1.  Open your terminal/Anaconda Prompt.
2.  Create an environment named `ml-roadmap` using Python 3.10:
    ```bash
    conda create --name ml-roadmap python=3.10 -y
    ```
3.  **Activate** the environment. You must do this *every single time* you open a new terminal to work on this project:
    ```bash
    conda activate ml-roadmap
    ```
    *(Notice how your terminal prompt changes to show `(ml-roadmap)` at the beginning).*
4.  Install the foundational Data Science libraries using `pip`:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn jupyter
    ```

> [!TIP]
> **Deactivation**: When you are finished working, you can type `conda deactivate` to return to your system's base environment.

---

## 4. Step 3: Visual Studio Code (The IDE)

Visual Studio Code (VS Code) is arguably the most popular Integrated Development Environment (IDE) in the world. It is lightweight, highly customizable, and free.

1.  Download [VS Code](https://code.visualstudio.com/).
2.  Install it with default settings.
3.  Open VS Code. Go to the **Extensions** tab (the square icon on the left sidebar, or `Ctrl+Shift+X`).
4.  Search for and install the following essential extensions:
    *   **Python** (by Microsoft)
    *   **Jupyter** (by Microsoft)
    *   *(Optional but Recommended)*: **Pylance**, **Prettier**, **GitLens**.

---

## 5. Step 4: Jupyter Notebooks (The Sandbox)

Jupyter Notebooks (`.ipynb` files) are the standard playground for Data Scientists. They allow you to write code in distinct "cells", execute them one by one, and render visual outputs (graphs, tables) immediately below the code.

To use your newly created `ml-roadmap` conda environment inside a Jupyter Notebook within VS Code:

1.  Ensure your environment is activated in the terminal: `conda activate ml-roadmap`
2.  Register your environment with Jupyter by installing the `ipykernel`:
    ```bash
    python -m ipykernel install --user --name=ml-roadmap
    ```
3.  Open VS Code, create a new file named `test.ipynb`.
4.  In the top right corner of the notebook interface, click **Select Kernel** (or it might say "Python 3.x.x").
5.  Select `Jupyter Kernel` -> `ml-roadmap`.
6.  Type `print("Hello AI")` in a cell and press `Shift + Enter` to run it!

---

## 6. Step 5: Git (Version Control)

Git is a version control system. It is like the "Save As" feature on steroids. It allows you to save snapshots of your codebase, travel back in time to previous versions, and collaborate seamlessly with a team.

1.  **Windows**: Download and install [Git for Windows](https://git-scm.com/download/win).
2.  **macOS**: Open the terminal and type `git --version`. It will prompt you to install the Apple Command Line Tools if you don't already have them.
3.  **Linux (Ubuntu)**: Run `sudo apt update && sudo apt install git`

**Verification**:
```bash
git --version
# Expected Output: git version 2.x.x
```

**Global Configuration**:
You must tell Git who you are so it can attribute your code changes to you. Run these commands in your terminal:
```bash
git config --global user.name "Your Name Here"
git config --global user.email "your.email@example.com"
```

---

## 🎯 Summary Checklist

- [ ] I understand why Virtual Environments are necessary to prevent Dependency Hell.
- [ ] I have successfully installed Miniconda.
- [ ] I can create, activate, and deactivate a conda virtual environment.
- [ ] I have installed VS Code alongside the Python and Jupyter extensions.
- [ ] I know how to select my conda environment as the Kernel inside a Jupyter notebook.
- [ ] I have installed Git and configured my global user name and email.

You are fully equipped! In the next module, **[05-Python-Essentials.md](./05-Python-Essentials.md)**, we will dive deep into Python syntax, data structures, and the core scientific libraries.

---

[← 03. Programming Fundamentals](03-Programming-Fundamentals.md) | [Back to Index](./README.md) | [Next: Python Essentials for Machine Learning →](05-Python-Essentials.md)
