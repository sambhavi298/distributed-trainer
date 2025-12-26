import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_train_loader(batch_size: int, num_workers: int = 2):
    """
    Returns a DataLoader for CIFAR-10 training data.
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,   # IMPORTANT
    transform=transform,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader
