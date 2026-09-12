from typing import Optional, Tuple
import torch


def rate_readout(spk: torch.Tensor) -> torch.Tensor:
    """
    spk: [T, B, C] output spikes
    return logits: [B, C]  (bigger = more evidence)
    """
    return spk.float().sum(dim=0)


def ttfs_readout(spk: torch.Tensor, t_max: Optional[int] = None, alpha: float = 1.0) -> torch.Tensor:
    """
    Time-to-first-spike readout.
    Earlier spikes get larger weights.

    spk: [T, B, C]
    return logits: [B, C]
    """
    T = spk.shape[0]
    if t_max is None:
        t_max = T
    
    spk = spk.float()
    t = torch.arange(T, device = spk.device, dtype = spk.dtype).view(T, 1, 1)
    weights = torch.exp(-alpha * t) # earlier time -> larger weights
    logits = (spk * weights).sum(dim=0)
    
    return logits


def burst_readout(spk: torch.Tensor, win: int = 3) -> torch.Tensor:
    """
    Simple burst evidence: maximum spike count in any short window.
    spk: [T, B, C]
    """
    T = spk.shape[0]
    if win <= 1:
        return spk.float().sum(dim=0)

    # sliding window sum
    sums = []
    for t0 in range(0, T - win + 1):
        sums.append(spk[t0:t0 + win].float().sum(dim=0))  # [B, C]
    stacked = torch.stack(sums, dim=0)  # [T-win+1, B, C]
    return stacked.max(dim=0).values


def phase_readout(spk: torch.Tensor, period: int = 8) -> torch.Tensor:
    """
    Phase evidence: spikes aligned to a reference oscillator.
    Very simple: weight spikes by cos(phase) (you can refine later).
    spk: [T, B, C]
    """
    T = spk.shape[0]
    period = max(int(period), 1)
    t = torch.arange(T, device=spk.device).float()
    phase = (2.0 * torch.pi) * ((t % period) / float(period))  # [T]
    w = torch.cos(phase).view(T, 1, 1)  # [T,1,1]
    return (spk.float() * w).sum(dim=0)  # [B,C]


def apply_readout(
    spk: torch.Tensor,
    coding: str,
    coding_args: Optional[dict] = None,
) -> torch.Tensor:
    coding_args = coding_args or {}
    coding = coding.lower()

    if coding == "rate":
        return rate_readout(spk)
    if coding == "ttfs":
        return ttfs_readout(spk, t_max=coding_args.get("t_max", None), alpha = float(coding_args.get("alpha",1.0)))
    if coding == "burst":
        return burst_readout(spk, win=int(coding_args.get("win", 3)))
    if coding == "phase":
        return phase_readout(spk, period=int(coding_args.get("period", 8)))

    raise ValueError(f"Unknown coding for readout: {coding}")