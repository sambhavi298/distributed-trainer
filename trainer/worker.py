import torch
import torch.nn as nn
import torch.optim as optim

from models.resnet import get_model
from data.cifar import get_train_loader
from trainer.checkpoint import save_checkpoint, load_checkpoint
from trainer.allreduce import all_reduce_gradients


def train_worker(
    rank: int,
    world_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    checkpoint_dir: str,
    device: str = "cpu",
):
    device = torch.device(device)
    
    checkpoint_path = f"{checkpoint_dir}/worker_{rank}.pt"
    
    print(f"[Worker {rank}/{world_size}] starting")

    # Model
    model = get_model()
    model.to(device)

    # Optimizer & loss
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint if exists
    start_epoch, global_step = load_checkpoint(
        checkpoint_path, model, optimizer
    )

    # Data
    train_loader = get_train_loader(batch_size)

    print(f"Starting training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            all_reduce_gradients(
                model,
                rank=rank,
                world_size=world_size,
                tmp_dir="tmp_grads",
            )
            optimizer.step()

            global_step += 1

            if global_step % 100 == 0:
                print(
                    f"[Worker {rank}] "
                    f"Epoch [{epoch}] Step [{global_step}] "
                    f"Loss: {loss.item():.4f}"
                )

        # Save checkpoint after every epoch
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch + 1,
            global_step,
        )
