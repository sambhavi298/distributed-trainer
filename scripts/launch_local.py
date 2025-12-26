import argparse
import torch.multiprocessing as mp
from trainer.worker import train_worker


def run_worker(rank, world_size, args):
    train_worker(
        rank=rank,
        world_size=world_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir="checkpoints",
        device=args.device,
        k_ratio=args.k_ratio,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/worker.pt",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--k_ratio", type=float, default=0.1)

    args = parser.parse_args()

    world_size = args.num_workers

    mp.spawn(
        run_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
