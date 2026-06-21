# 🐧 07. Linux Fundamentals for ML

> **Prerequisites**: Module 04 | **Difficulty**: ⭐⭐☆☆☆ Beginner

If you want to train Deep Learning models at scale, you will not be doing it on a Windows laptop. 99% of AI training happens in the cloud (AWS EC2, GCP Compute Engine, Azure VMs) on **headless Linux servers** (servers with no graphical user interface or mouse). 

You must be comfortable navigating, moving data, and managing long-running training processes using *only* the terminal.

---

## 📋 Table of Contents
1. [The Shell and Navigation](#1-the-shell-and-navigation)
2. [File & Directory Manipulation](#2-file--directory-manipulation)
3. [Viewing & Filtering Data](#3-viewing--filtering-data)
4. [Permissions & Ownership](#4-permissions--ownership)
5. [Process Management (Crucial for ML)](#5-process-management-crucial-for-ml)
6. [Secure Shell (SSH) and SCP](#6-secure-shell-ssh-and-scp)
7. [Environment Variables](#7-environment-variables)

---

## 1. The Shell and Navigation

The "Shell" (usually `bash` or `zsh`) is the program that interprets your typed commands and talks to the Linux kernel.

### Where am I?
```bash
pwd              # Print Working Directory. Outputs your current absolute path.
                 # Example output: /home/ubuntu/ml_project
```

### What is here?
```bash
ls               # List files in the current directory
ls -l            # List files in a 'long' format (shows size, permissions, dates)
ls -la           # List ALL files (including hidden files that start with a dot, like .env or .git)
ls -lh           # List files with Human-readable sizes (e.g., 5.2G instead of 5200000000 bytes)
```

### How do I move?
```bash
cd data_folder   # Change Directory into 'data_folder'
cd ..            # Move UP one directory to the parent folder
cd ~             # Go to your user's home directory (e.g., /home/ubuntu/)
cd /var/log      # Go to an absolute path
```

---

## 2. File & Directory Manipulation

```bash
# Creating
mkdir dataset             # Make a new directory called 'dataset'
mkdir -p data/raw/images  # Make nested directories all at once
touch script.py           # Create an empty file called 'script.py'

# Moving and Copying
cp data.csv backup.csv    # Copy a file
cp -r old_dir new_dir     # Copy a directory Recursively (everything inside it)
mv old_name.py new.py     # Move (or rename) a file

# Deleting (WARNING: Linux has no Recycle Bin. Deletion is permanent!)
rm old.py                 # Remove a file
rm -rf temp_folder        # Forcefully and Recursively Remove a directory and all its contents
```

---

## 3. Viewing & Filtering Data

As a Data Scientist, you will often need to inspect massive CSV files. Opening a 10GB CSV in Excel will crash your computer. Linux can inspect it instantly.

```bash
# Viewing whole files
cat config.yaml        # Print the entire file to the terminal

# Peeking at large files
head -n 10 data.csv    # View the first 10 lines (perfect for checking CSV column headers)
tail -n 10 data.csv    # View the last 10 lines (perfect for seeing the end of a log file)

# Searching within files
grep "Error" train.log # Search for the word "Error" inside the file train.log
grep -i "error" train.log # Case-insensitive search
```

---

## 4. Permissions & Ownership

Sometimes you'll get a `Permission Denied` error when trying to run a script or install a library.

In Linux, files have permissions for the User, Group, and Others: Read (`r`), Write (`w`), and Execute (`x`).

```bash
# Check permissions
ls -l script.sh
# Output: -rw-r--r-- (User can read/write, others can only read. NO ONE can execute)

# Make a python script or bash script executable
chmod +x train.sh

# Run a command as the "Super User" (Administrator / Root)
sudo apt update
sudo apt install htop
```

---

## 5. Process Management (Crucial for ML)

When you start training a Deep Learning model, it might take 48 hours. If you just type `python train.py` and then close your laptop or lose your SSH internet connection, **the training process will immediately die**.

### Viewing Resource Usage
```bash
# See what is consuming your CPU and RAM in real-time
htop

# See what is consuming your GPU (NVIDIA only)
# EXTREMELY important for checking VRAM usage and avoiding Out-Of-Memory errors!
watch -n 1 nvidia-smi  # Updates every 1 second
```

### Running Long ML Jobs
To keep a process running after you disconnect from the server, use `nohup` (No Hangup) or `tmux`/`screen`.

```bash
# Run training in the background. The '&' sends it to the background.
nohup python train.py > training_logs.txt 2>&1 &

# You can safely close your terminal now. To check progress later:
tail -f training_logs.txt  # The '-f' means "follow", it will update live!

# How to kill a runaway ML process?
ps aux | grep python       # Find the Process ID (PID) of your python script (e.g., 14532)
kill -9 14532              # Forcefully kill process 14532
```

---

## 6. Secure Shell (SSH) and SCP

To connect to your cloud GPU server, you use SSH.

```bash
# Connect to a remote server
ssh username@192.168.1.50

# Connect using a specific identity key file (e.g., an AWS .pem file)
ssh -i my_key.pem ubuntu@ec2-ip-address.compute.amazonaws.com
```

To transfer your dataset from your laptop to the cloud server, use SCP (Secure Copy):
```bash
# Copy local dataset.zip to the remote server's home directory
scp dataset.zip ubuntu@ec2-ip-address.compute.amazonaws.com:~/ 
```

---

## 7. Environment Variables

Often used to store sensitive API keys securely without hardcoding them into your Python scripts.

```bash
# Set an environment variable
export OPENAI_API_KEY="sk-12345abcdef"

# Print it
echo $OPENAI_API_KEY

# In Python, you access it like this:
# import os
# api_key = os.environ.get("OPENAI_API_KEY")
```

---

## 🎯 Summary Checklist

- [ ] I can navigate the Linux file system using `pwd`, `cd`, and `ls`.
- [ ] I can create, copy, move, and delete files safely.
- [ ] I can inspect the headers of a massive CSV using `head` without crashing my machine.
- [ ] I know how to check GPU VRAM usage using `nvidia-smi`.
- [ ] I understand how to use `nohup` or `tmux` to keep an ML model training after I close my laptop.
- [ ] I know how to transfer files to a cloud server using `scp`.

Next up, we return to theory to cover the mathematical engine of AI: **[08-Mathematical-Foundations.md](./08-Mathematical-Foundations.md)**!
