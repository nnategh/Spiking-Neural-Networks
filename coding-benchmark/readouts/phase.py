import torch

def phase_readout(
    spk: torch.Tensor,
    period: int = 8,
    window: int = None,
) -> torch.Tensor:
    """
    Phase-based readout.

    Each spike is weighted according to its phase within a periodic cycle.

    Args:
        spk: output spikes with shape [T, B, C]
        period: oscillation period
        window: if set, timesteps >= window get zero weight instead of
            continuing the periodic cosine pattern. Diagnostic option for
            repeat_cycles=False encodings, where no new input arrives after
            t=cycle_steps: without this, the leftover "coasting" tail
            (t >= cycle_steps) still gets cosine-weighted and summed,
            injecting a period-dependent residual unrelated to the actual
            encoded information whenever (T - period) isn't an exact
            multiple of period.

    Returns:
        logits: [B, C]
    """
    if spk.dim() != 3:
        raise ValueError(f"Expected spk shape [T, B, C], but got {spk.shape}")

    T = spk.shape[0]

    period = max(int(period), 1)

    t = torch.arange(
        T,
        device=spk.device,
        dtype=torch.float32,
    )

    phase = 2.0 * torch.pi * ((t % period) / period)

    # cosine weighting
    weights = torch.cos(phase)

    if window is not None:
        weights = weights * (t < window).to(weights.dtype)

    weights = weights.view(T, 1, 1)

    logits = (spk.float() * weights).sum(dim=0)

    return logits