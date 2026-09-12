"""
CIFAR-10 dataloader helper.

Output:
    images: torch.Tensor [B, 3, 32, 32]
    labels: torch.Tensor [B]
"""

from typing import Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_cifar10_dataloaders(
    data_dir: str = "./datasets",
    batch_size: int = 64,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns:
        train_loader, val_loader, test_loader
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform,
    )

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform,
    )

    val_size = int(val_ratio * len(full_train))
    train_size = len(full_train) - val_size

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar10_dataloaders()

    images, labels = next(iter(train_loader))

    print("images:", images.shape)
    print("labels:", labels.shape)