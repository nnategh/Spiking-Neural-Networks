from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn

from readout.aggregation import apply_readout

def evaluate_dvs(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    coding: str = "rate",
    coding_args: Optional[Dict] = None,
) -> Tuple[float, float]:
    model.eval()
    coding_args = coding_args or {}

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if not torch.is_floating_point(x):
                x = x.float()
            y = y.to(device)

            spk = model(x)  # [T, B, C]
            logits = apply_readout(spk, coding=coding, coding_args=coding_args)

            loss = criterion(logits, y)

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_n += bs

    return total_loss / total_n, total_correct / total_n