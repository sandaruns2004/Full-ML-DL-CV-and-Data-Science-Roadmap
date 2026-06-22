# ⚡ Distributed Training

> **Prerequisites**: PyTorch Basics | **Difficulty**: ⭐⭐⭐⭐⭐ Professional

---

## 📋 Table of Contents
1. [The Compute Crisis](#1-the-compute-crisis)
2. [Data Parallelism (DP vs. DDP)](#2-data-parallelism-dp-vs-ddp)
3. [Model Parallelism & Pipeline Parallelism](#3-model-parallelism--pipeline-parallelism)
4. [Fully Sharded Data Parallel (FSDP) and DeepSpeed](#4-fully-sharded-data-parallel-fsdp-and-deepspeed)
5. [Project Ideas & What's Next](#5-project-ideas--whats-next)

---

## 1. The Compute Crisis

Modern neural networks (like GPT-3 with 175 Billion parameters) cannot fit inside a single GPU. Even a massive NVIDIA A100 has 80GB of VRAM. A 175B parameter model in 16-bit precision requires 350GB of VRAM just to store the weights! Adding the optimizer states and gradients easily pushes the requirement past 1 Terabyte.

To train these models, we must distribute the training process across multiple GPUs, and often across multiple physical servers (nodes).

---

## 2. Data Parallelism (DP vs. DDP)

If your model **can** fit on a single GPU, but you want to train it faster by processing more data at once, you use Data Parallelism.

### DataParallel (DP) - The Old Way
- The model is copied to all available GPUs.
- The data batch (e.g., 64 images) is split evenly (e.g., 4 GPUs get 16 images each).
- Each GPU calculates gradients.
- **The Bottleneck**: GPU 0 acts as a "master." All GPUs send their gradients to GPU 0, which averages them, updates the model weights, and then broadcasts the new weights back to all GPUs. This creates massive communication overhead and under-utilizes the GPUs.

### DistributedDataParallel (DDP) - The Modern Way
PyTorch's `DistributedDataParallel` solves the bottleneck.
- There is no "master" GPU.
- Each GPU runs in its own independent process.
- During the backward pass, gradients are synchronized across all GPUs simultaneously using an optimized mathematical operation called **Ring-AllReduce**.
- Every GPU independently updates its own weights (which remain perfectly in sync).

```python
# Simplified DDP setup in PyTorch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 1. Initialize the process group (handles communication between GPUs)
dist.init_process_group(backend='nccl')

# 2. Assign the specific GPU to this process
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)

# 3. Create model and move to the specific GPU
model = MyModel().cuda(local_rank)

# 4. Wrap the model in DDP
model = DDP(model, device_ids=[local_rank])

# 5. Use a DistributedSampler so each GPU gets different data
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

---

## 3. Model Parallelism & Pipeline Parallelism

If your model **cannot** fit on a single GPU, Data Parallelism will fail with an Out Of Memory (OOM) error. You must split the model itself.

### Tensor / Model Parallelism (Megatron-LM)
You slice individual matrix multiplications across multiple GPUs. 
- GPU 1 computes the left half of the matrix multiplication.
- GPU 2 computes the right half.
- They communicate the results instantly. 
- *Requires ultra-fast NVLink connections between GPUs. Rarely used across different physical machines.*

### Pipeline Parallelism
You split the network layer-by-layer.
- GPU 1 holds Layers 1-10.
- GPU 2 holds Layers 11-20.
- GPU 1 computes its forward pass and sends the hidden state to GPU 2.
- *The Problem*: GPU 2 sits idle waiting for GPU 1 to finish. This is called the "Pipeline Bubble." Modern frameworks use micro-batching (GPipe) to keep all GPUs busy.

---

## 4. Fully Sharded Data Parallel (FSDP) and DeepSpeed

The ultimate state-of-the-art solution for massive models is **ZeRO (Zero Redundancy Optimizer)**, developed by Microsoft DeepSpeed, and adopted by PyTorch as **FSDP**.

In standard DDP, every single GPU holds a 100% complete copy of the model weights, optimizer states, and gradients. This is incredibly redundant.

**How FSDP / ZeRO works:**
Instead of copying the model, it **shards** (slices) the model weights across all GPUs.
- If you have a 4GB model and 4 GPUs, each GPU only stores 1GB of the model.
- During the forward pass, when GPU 1 needs Layer 2 (which is stored on GPU 2), it rapidly requests those specific weights over the network, computes the activation, and immediately deletes the weights from memory to save space.

This allows you to train massive models just by throwing more GPUs at the problem!

---

## 5. Project Ideas & What's Next

### Project Ideas
- 🟢 **Launch a DDP Script**: Take a basic PyTorch training loop and modify it to use `DistributedDataParallel`. Run it on your local machine using PyTorch's `torchrun` command, simulating 2 GPUs on your CPU (e.g., `torchrun --nproc_per_node=2 train.py`).

### What's Next
| Next | Why |
|------|-----|
| [Data Versioning (DVC)](./08-Data-Versioning-DVC.md) | We track code with Git, but how do we track the gigabytes of data and models trained on these distributed clusters? |

---

[← Experiment Tracking](06-Experiment-Tracking.md) | [Back to Index](../README.md) | [Next: Data Versioning (DVC) →](08-Data-Versioning-DVC.md)
