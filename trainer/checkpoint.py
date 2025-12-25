import torch
import os


def save_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
    epoch: int,
    global_step: int,
):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str,
    model,
    optimizer,
):
    if not os.path.exists(checkpoint_path):
        return 0, 0  # epoch, global_step

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint["epoch"], checkpoint["global_step"]
