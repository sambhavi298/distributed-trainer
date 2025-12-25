import argparse
from trainer.worker import train_worker


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

    args = parser.parse_args()

    train_worker(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
    )


if __name__ == "__main__":
    main()
