from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
from readout.aggregation import apply_readout


@torch.no_grad()
def _to_float_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_floating_point(x):
        return x.float()
    return x


def train_one_epoch_dvs(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    coding: str = "rate",
    coding_args: Optional[Dict] = None,
) -> Tuple[float, float]:
    model.train()
    coding_args = coding_args or {}

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = _to_float_if_needed(x).to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        spk = model(x)  # [T,B,C]
        assert spk.ndim == 3, f"Expected [T,B,C], got {spk.shape}"
        assert torch.isfinite(spk).all(), "NaN/Inf in spk"

        # Quick TTFS sanity
        if coding == "ttfs":
            t0 = spk[0].sum().item()
            t_last = spk[-1].sum().item()
            print("spikes t=0 vs t=T-1:", t0, t_last)


        logits = apply_readout(spk, coding=coding, coding_args=coding_args)
        assert logits.ndim == 2, f"Expected [B,C], got {logits.shape}"
        assert logits.shape[0] == y.size(0), "Batch mismatch"
        assert torch.isfinite(logits).all(), "NaN/Inf in logits"

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += bs

    return total_loss / total_n, total_correct / total_n