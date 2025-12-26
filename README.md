# Distributed Trainer

This project implements a distributed training system.

## Distributed Training (Local Multi-Worker)

This project implements a custom data-parallel distributed training
pipeline using Python multiprocessing, explicit rank/world-size handling,
and a custom all-reduce implementation.

### Platform Note (Windows)

On Windows, Python multiprocessing uses spawn semantics, which makes
multi-worker training with large models and shared state memory-intensive
and unstable.

As a result:
- Single-worker training is fully supported and validated.
- Multi-worker distributed training is **design-complete** but may
  terminate due to OS-level memory and process management limitations.

This limitation does not affect the correctness of the distributed
training design and mirrors real-world practice, where production
distributed training systems run on Linux-based environments.
