import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_emnist_loaders(batch_size=64, val_ratio=0.15, seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    full_train = torchvision.datasets.EMNIST(
        root="./data", split="balanced", train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.EMNIST(
        root="./data", split="balanced", train=False, download=True, transform=transform
    )

    val_size = int(val_ratio * len(full_train))
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
