import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def get_emnist_dataloaders(
    data_dir: str = "./datasets",
    batch_size: int = 64,
    val_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
):
    """
    Canonical EMNIST (Balanced split) loader for this project -- the main
    benchmark pipeline (experiments/run_phase_grid.py) and the standalone
    utility scripts (experiments/sanity_check.py,
    experiments/calibrate_threshold.py) all call this function rather than
    each defining their own copy, so there is exactly one place that knows
    how EMNIST is loaded, split, and normalized.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    full_train = torchvision.datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=True,
        download=True,
        transform=transform,
    )

    test_set = torchvision.datasets.EMNIST(
        root=data_dir,
        split="balanced",
        train=False,
        download=True,
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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_emnist_dataloaders()

    images, labels = next(iter(train_loader))

    print("images:", images.shape)
    print("labels:", labels.shape)