# Distributed Trainer

This project implements a distributed training system.

### Platform Note (Windows)

This project was developed and validated on Windows and Google Colab.

Due to Python multiprocessing using spawn semantics on Windows, running
multiple training workers with custom gradient synchronization becomes
memory-inefficient and unstable for large models.

The distributed training design (rank/world_size, all-reduce logic,
barriers) is fully implemented and validated conceptually. Runtime
validation of multi-worker gradient synchronization is intended for
Linux environments, where production ML systems are deployed.
