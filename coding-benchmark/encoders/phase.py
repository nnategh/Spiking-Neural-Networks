import torch


def denorm_to_unit(x: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized image tensor from [-1,1] to [0,1].
    """
    x01 = 0.5 * x + 0.5
    return torch.clamp(x01, 0.0, 1.0)


def phase_encode(
    x: torch.Tensor,
    time_steps: int,
    num_bins: int = 8,
    cycle_steps: int = 8,
    threshold: float = 0.15,
    repeat_cycles: bool = False,
):
    """
    Phase-of-firing encoding for static frame-based images.

    Expected input:
        x: [B, C, H, W] or [B, N]

    Returns:
        spikes: [T, B, C*H*W]
        phase_index: [T]

    Encoding principle:
        1. Normalize pixel intensity to [0,1].
        2. Map intensity to phase bin:
              brighter pixel -> earlier phase
              darker pixel -> later phase
        3. Map phase bin to offset within each oscillation cycle.
        4. Emit spikes periodically at: t_k = offset + k * cycle_steps

    DoF:
        num_bins:
            Phase resolution.

        cycle_steps:
            Oscillation cycle length.
            Smaller cycle_steps = higher relative frequency.
            Larger cycle_steps = lower relative frequency.

        repeat_cycles:
            False (default): each pixel fires at most once, in its first
            cycle only (sparse, spike-budget-comparable to TTFS).
            True: each pixel re-fires every cycle for the whole duration
            (redundancy ablation).
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be > 0")

    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")

    if cycle_steps <= 0:
        raise ValueError("cycle_steps must be > 0")

    if cycle_steps > time_steps:
        raise ValueError("cycle_steps must be <= time_steps")

    B = x.size(0)
    x_flat = x.view(B, -1)
    x01 = denorm_to_unit(x_flat)

    fire_mask = x01 > threshold

    # intensity -> phase bin
    phase_float = (1.0 - x01) * (num_bins - 1)
    phase_bin = torch.floor(phase_float).long().clamp(0, num_bins - 1)

    # phase bin -> offset within cycle
    if num_bins == 1:
        offset = torch.zeros_like(phase_bin)
    else:
        offset_float = phase_bin.float() / float(num_bins - 1)
        offset = torch.round(offset_float * (cycle_steps - 1)).long()
        offset = offset.clamp(0, cycle_steps - 1)

    T = time_steps
    N = x_flat.size(1)

    spikes = torch.zeros(T, B, N, device=x.device, dtype=torch.float32)

    b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, N)
    n_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)

    max_k = ((T - 1) // cycle_steps) if repeat_cycles else 0

    for k in range(max_k + 1):
        t_k = offset + k * cycle_steps
        valid = fire_mask & (t_k < T)

        if valid.any():
            spikes[t_k[valid], b_idx[valid], n_idx[valid]] = 1.0

    # phase label of each timestep
    phase_index = torch.arange(T, device=x.device) % cycle_steps

    return spikes, phase_index