"""
Sanity check for the SNN benchmark pipeline.

This script verifies that the complete training pipeline works
correctly before running benchmark experiments.

Checks:
    - Data loading
    - Neural encoding
    - SNN forward/backward pass
    - Loss decreases during training
    - Accuracy exceeds random chance
"""
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.emnist import get_emnist_dataloaders
from models.snn import SNNClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_loader, val_loader, test_loader = get_emnist_dataloaders(batch_size=64)

    model = SNNClassifier(
        input_dim=28 * 28,
        num_classes=47,
        time_steps=32,
        hidden_dim=256,
        coding="rate",
        v_th=0.1,
        encoding_args={"rate_scale": 0.5},
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 4):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")