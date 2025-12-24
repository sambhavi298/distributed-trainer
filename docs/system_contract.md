# System Contract: Minimal but Real Distributed Training System

## 1. Purpose

This document defines the **core system contract** for a minimal, fault-tolerant distributed training system based on data parallelism.  
The goal is to ensure architectural clarity, prevent scope creep, and establish clear responsibilities for each component before implementation.

This system prioritizes:
- Correctness over performance
- Explicit communication over hidden abstractions
- Honest evaluation over inflated claims

---

## 2. High-Level Architecture

The system consists of **three logical components**:

1. **Worker**
2. **Coordinator**
3. **Communication Layer**

No additional components are introduced in the base system.

---

## 3. Worker Contract

Each worker is a long-running process responsible for model training on a shard of data.

### Responsibilities

A worker must:

- Load the **model architecture** (identical across all workers)
- Load a **unique shard of the dataset**
- Execute the local training loop:
  - Forward pass
  - Backward pass
  - Gradient computation
- Participate in **synchronous gradient aggregation**
- Apply the **averaged gradients** to update model parameters
- Periodically **save checkpoints**
- Reload from the latest checkpoint upon restart

### Training Loop (Conceptual)

for each training step:
batch = get_local_data()
loss = forward(model, batch)
loss.backward()
local_gradients = extract_gradients(model)
global_gradients = all_reduce(local_gradients)
model.update(global_gradients)


Workers do not communicate directly with each other except through the defined communication primitives.

---

## 4. Coordinator Contract

The coordinator is a lightweight control process.

### Responsibilities

The coordinator:

- Assigns **worker ranks**
- Maintains **world size**
- Bootstraps communication topology
- Monitors worker liveness (heartbeat / timeout)
- Detects worker failure
- Triggers worker restart and checkpoint reload

### Non-Responsibilities

The coordinator does **not**:
- Perform gradient aggregation
- Store model parameters
- Participate in training computation
- Act as a parameter server

The coordinator exists to support correctness and fault tolerance, not performance optimization.

---

## 5. Communication Contract

The communication layer exposes **exactly one collective primitive**:

### `all_reduce(gradients) → averaged_gradients`

#### Properties

- Synchronous
- Deterministic
- Blocks until all active workers participate
- Returns the same averaged gradients to all workers

#### Implementation Notes

- Implemented using a **ring all-reduce topology**
- Built using **explicit message passing** (e.g., gRPC or MPI)
- Treated as a black-box function by the worker training loop

No other collective operations (broadcast, scatter, gather) are required in the base system.

---

## 6. Fault Tolerance Model

### Failure Assumptions

- Workers may crash or become unreachable
- Network is reliable but may experience delays
- No Byzantine (malicious) failures are considered

### Recovery Strategy

- Coordinator detects failure via heartbeat or RPC timeout
- Failed worker is restarted
- Worker reloads the **latest available checkpoint**
- Training resumes without restarting the entire job

Fault tolerance guarantees **progress**, not zero-loss recovery.

---

## 7. Checkpointing Contract

Checkpoints must include:
- Model parameters
- Optimizer state
- Training step / epoch counter

Checkpointing is:
- Periodic
- Local or shared storage (implementation-dependent)
- Required for fault recovery

---

## 8. Assumptions

The system assumes:

- Homogeneous workers (same model, similar compute)
- Synchronous training
- Small cluster size (≤ 8 workers)
- Single training job at a time
- One model per training run

These assumptions are **explicit design constraints**, not limitations.

---

## 9. Out of Scope

The following are intentionally excluded:

- Model parallelism
- Pipeline parallelism
- Custom NCCL / CUDA kernels
- Asynchronous training in the base system
- Large-scale (>8 nodes) performance tuning
- Multi-tenant job scheduling

---

## 10. Definition of Success

The system is considered correct if:

- Training converges similarly to a single-worker baseline
- Adding workers produces measurable speedup
- Worker failure does not crash the system
- Training resumes from the last checkpoint after failure
- Communication behavior is understandable and reproducible

Performance optimization is secondary to correctness and clarity.

---

## 11. Design Philosophy

This system is built to:
- Expose distributed training mechanics explicitly
- Favor debuggability over abstraction
- Encourage reasoning about tradeoffs and failure modes

It is a learning and demonstration system, not a production framework.
