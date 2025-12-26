import os
import time
import torch


def all_reduce_gradients(model, rank, world_size, tmp_dir="tmp_grads"):
    os.makedirs(tmp_dir, exist_ok=True)

    step_dir = os.path.join(tmp_dir, "step")
    os.makedirs(step_dir, exist_ok=True)

    # ---- 1. Save local gradients ----
    grad_path = os.path.join(step_dir, f"grads_{rank}.pt")
    grads = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu()

    torch.save(grads, grad_path)

    # ---- 2. Barrier: wait for all workers to write ----
    while True:
        files = [f for f in os.listdir(step_dir) if f.startswith("grads_")]
        if len(files) == world_size:
            break
        time.sleep(0.05)

    avg_path = os.path.join(step_dir, "avg.pt")

    # ---- 3. Rank 0 aggregates ----
    if rank == 0:
        summed = {}

        for r in range(world_size):
            worker_grads = torch.load(os.path.join(step_dir, f"grads_{r}.pt"))
            for name, grad in worker_grads.items():
                summed[name] = summed.get(name, 0) + grad

        for name in summed:
            summed[name] /= world_size

        tmp_avg = avg_path + ".tmp"
        torch.save(summed, tmp_avg)
        os.replace(tmp_avg, avg_path)  # atomic publish

    # ---- 4. Barrier: wait for averaged gradients ----
    while not os.path.exists(avg_path):
        time.sleep(0.05)

    avg_grads = torch.load(avg_path)

    # ---- 5. Load averaged gradients ----
    for name, param in model.named_parameters():
        if name in avg_grads:
            param.grad = avg_grads[name].to(param.device)

    # ---- 6. Cleanup (rank 0 only) ----
    if rank == 0:
        for f in os.listdir(step_dir):
            os.remove(os.path.join(step_dir, f))
