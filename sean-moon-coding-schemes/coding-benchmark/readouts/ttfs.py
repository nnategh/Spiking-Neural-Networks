from typing import Optional
import torch


def ttfs_readout(
    spk: torch.Tensor,
    alpha: Optional[float] = None,
    no_spike_penalty: float = 0.0,
) -> torch.Tensor:
    """
    Time-to-first-spike readout.

    Args:
        spk: output spikes, shape [T, B, C]
        alpha: decay strength for time weighting. If None (default),
            scaled to the actual window length as `4.0 / T` so the decay
            spans the whole sequence regardless of `time_steps`. A fixed
            alpha (e.g. 1.0) decays to ~0 well before T for any T bigger
            than ~10-15, making the readout effectively blind to any
            spike after that -- silently discarding most of a
            TTFS-encoded image's information whenever pixel intensities
            aren't concentrated near the very brightest values (as they
            are for EMNIST strokes, but not for CIFAR-10's natural
            images).
        no_spike_penalty: value added when a class never spikes

    Returns:
        logits: [B, C]
    """
    if spk.dim() != 3:
        raise ValueError(f"Expected spk shape [T, B, C], but got {spk.shape}")

    T = spk.shape[0]
    spk = spk.float()

    if alpha is None:
        alpha = 4.0 / T

    t = torch.arange(T, device=spk.device, dtype=spk.dtype).view(T, 1, 1)
    weights = torch.exp(-alpha * t)

    logits = (spk * weights).sum(dim=0)

    if no_spike_penalty != 0.0:
        has_spike = spk.sum(dim=0) > 0
        logits = torch.where(
            has_spike,
            logits,
            torch.full_like(logits, no_spike_penalty),
        )

    return logits