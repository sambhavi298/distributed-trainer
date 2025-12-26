# Distributed Trainer — Minimal but Real Distributed Training System

## Overview

This project implements a **minimal, explicit distributed training system** from scratch using PyTorch.  
The goal is not peak performance, but **architectural clarity**, **correctness**, and **honest evaluation**.

Unlike typical projects that rely entirely on `torch.nn.parallel.DistributedDataParallel`, this system exposes the core mechanics of distributed training:
- Worker orchestration
- Rank / world-size semantics
- Gradient synchronization (all-reduce)
- Checkpointing and fault recovery
- Platform-aware limitations

This project is intended as a **systems-level ML engineering project**, not a benchmark race.

---

## Project Goals

- Build a **data-parallel training system** where each worker holds a full model replica
- Implement **explicit gradient synchronization** (custom all-reduce)
- Support **checkpointing and resume**
- Demonstrate **fault-awareness**
- Clearly document **design tradeoffs and platform constraints**

---

## Architecture Summary

### Data Parallelism

- Each worker:
  - Loads the same model architecture
  - Processes a different shard of the dataset
  - Computes local gradients
- Gradients are synchronized across workers using a custom all-reduce mechanism
- Model weights remain consistent across workers

### Worker Orchestration

- Workers are launched using `torch.multiprocessing.spawn`
- Each worker receives:
  - `rank`
  - `world_size`
- Rank is used to:
  - Control logging
  - Coordinate synchronization
  - Manage checkpoints

---

## Project Structure

```
distributed-trainer/
├── data/
│   ├── cifar.py          # CIFAR-10 dataset loading
│   └── __init__.py
├── models/
│   ├── resnet.py         # ResNet-18 adapted for CIFAR-10
│   └── __init__.py
├── trainer/
│   ├── worker.py         # Training loop per worker
│   ├── allreduce.py     # Custom gradient all-reduce
│   ├── checkpoint.py    # Save / load checkpoints
│   ├── comms.py         # Synchronization helpers
│   ├── coordinator.py   # Worker coordination logic
│   ├── compression.py   # (Reserved for future work)
│   └── __init__.py
├── scripts/
│   ├── launch_local.py  # Entry point for local training
│   └── __init__.py
├── docs/
├── README.md
└── requirements.txt
```

---

## Training Modes

### Single-Worker Training (Baseline)

- `--num_workers` maps directly to `world_size`.
- Fully supported and validated
- Used to establish correctness and performance baseline
- Includes checkpointing and resume

Example:
```bash
python -m scripts.launch_local \
  --epochs 5 \
  --batch_size 128 \
  --lr 0.1 \
  --num_workers 1 \
  --device cpu
```

---

### Multi-Worker Distributed Training (Design-Complete)

* Multiple workers launched locally
* Explicit rank/world-size handling
* Custom all-reduce used for gradient synchronization

Example:

```bash
python -m scripts.launch_local \
  --epochs 2 \
  --batch_size 128 \
  --lr 0.1 \
  --num_workers 2 \
  --device cpu
```

---

## Checkpointing and Fault Recovery

* Each worker periodically saves training state
* On restart:
  * Latest checkpoint is loaded
  * Training resumes from last completed epoch
* This enables recovery from:
  * Process termination
  * Manual interruption

---

## Platform Note (Important)

### Windows Limitation

On **Windows**, Python multiprocessing uses **spawn semantics**, which significantly increases memory usage and serialization overhead.

As a result:

* Single-worker training runs reliably
* Multi-worker training **may terminate or hang** due to OS-level memory and process management limitations
* Errors such as `MemoryError`, process termination, or hanging joins are expected behavior

This is **not a correctness issue in the distributed training design**.

### Industry Context

Production distributed training systems are deployed on **Linux-based environments**, where:

* Fork-based multiprocessing is available
* Shared memory handling is more efficient
* NCCL and MPI backends are supported

The architecture implemented here directly maps to such environments.

---

## Evaluation Philosophy

This project prioritizes:

* Correctness over speed
* Explicit mechanisms over hidden abstractions
* Honest reporting over inflated claims

Metrics considered:

* Loss convergence
* Correct resume from checkpoint
* Worker startup and coordination behavior
* Failure modes and recovery behavior

No artificial performance claims are made.

---

## What This Project Demonstrates

* Understanding of **distributed ML systems**
* Ability to design **data-parallel training pipelines**
* Awareness of **OS-level constraints**
* Clear separation between **design correctness** and **runtime environment**
* Real-world engineering judgment

---

## What This Project Does NOT Claim

* Production-ready performance
* Optimized communication kernels
* NCCL-level efficiency
* Large-scale (>8 nodes) scalability

These are deliberate design exclusions.

---

## Project Milestones

* [x] Step 1: Project structure and system contracts
* [x] Step 2: Dataset and model abstraction
* [x] Step 3: Single-worker training baseline
* [x] Step 4: Checkpointing and resume
* [x] Step 5: Distributed training architecture (data parallel + all-reduce)

---

## Future Work

* Gradient compression (top-k / quantization)
* Linux-based multi-node execution
* Parameter-server variant
* Kubernetes-based orchestration
* Communication benchmarking

---

## Final Note

This project is intentionally minimal.
Every line exists to expose **how distributed training actually works**, not to hide complexity behind libraries.

That is its value.
