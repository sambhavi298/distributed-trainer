import os
import time
import torch


def all_reduce_gradients(model, rank, world_size, tmp_dir="tmp_grads"):
    os.makedirs(tmp_dir, exist_ok=True)

    # 1. Save local gradients
    grad_path = os.path.join(tmp_dir, f"grads_rank_{rank}.pt")
    grads = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu()

    torch.save(grads, grad_path)

    # 2. Wait for all workers
    while True:
        files = os.listdir(tmp_dir)
        if len(files) >= world_size:
            break
        time.sleep(0.1)

    # 3. Rank 0 aggregates
    if rank == 0:
        summed_grads = {}

        for r in range(world_size):
            path = os.path.join(tmp_dir, f"grads_rank_{r}.pt")
            worker_grads = torch.load(path)

            for name, grad in worker_grads.items():
                if name not in summed_grads:
                    summed_grads[name] = grad.clone()
                else:
                    summed_grads[name] += grad

        # Average
        for name in summed_grads:
            summed_grads[name] /= world_size

        tmp_avg_path = os.path.join(tmp_dir, "avg_grads.tmp")
        final_avg_path = os.path.join(tmp_dir, "avg_grads.pt")

        torch.save(summed_grads, tmp_avg_path)
        os.replace(tmp_avg_path, final_avg_path)

    # 4. Wait for averaged gradients
    avg_path = os.path.join(tmp_dir, "avg_grads.pt")
    while not os.path.exists(avg_path):
        time.sleep(0.1)

    avg_grads = torch.load(avg_path)

    # 5. Load averaged gradients back
    for name, param in model.named_parameters():
        if name in avg_grads:
            param.grad = avg_grads[name].to(param.device)

    # 6. Cleanup (only rank 0)
    if rank == 0:
        for f in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, f))
