import torch
import torch.nn as nn
import torch.optim as optim

from datasets.emnist import get_emnist_dataloaders
from models.snn import SNNClassifier


def run_calibration(v_th, coding="rate", num_batches=20):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_loader, _, _ = get_emnist_dataloaders(batch_size=64)

    model = SNNClassifier(
        input_dim=28 * 28,
        num_classes=47,
        time_steps=32,
        hidden_dim=256,
        coding=coding,
        v_th=v_th,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_spikes = 0.0

    model.train()

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, output_seq, spike_count = model(x, return_seq=True)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)
        total_spikes += spike_count.item()

    avg_loss = total_loss / num_batches
    acc = total_correct / total_samples
    avg_spikes = total_spikes / total_samples

    return avg_loss, acc, avg_spikes


if __name__ == "__main__":
    thresholds = [0.05, 0.1, 0.2, 0.5, 1.0]

    for v_th in thresholds:
        loss, acc, spikes = run_calibration(v_th=v_th, coding="rate")
        print(
            f"v_th={v_th:<4} | "
            f"loss={loss:.4f} | "
            f"acc={acc:.4f} | "
            f"avg_hidden_spikes/sample={spikes:.2f}"
        )