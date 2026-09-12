import torch


def denorm_to_unit(x: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized image tensor from [-1, 1] to [0, 1].
    """
    x01 = 0.5 * x + 0.5
    return torch.clamp(x01, 0.0, 1.0)


def ttfs_encode(
    x: torch.Tensor,
    time_steps: int,
    gamma: float = 1.0,
    t_min: int = 0,
    t_max=None,
    threshold: float = 0.15,
) -> torch.Tensor:
    """
    Time-to-First-Spike (TTFS) encoding for static frame-based images.

    Expected input:
        x: [B, C, H, W] or [B, N]

    Returns:
        spikes: [T, B, C*H*W]

    Encoding principle:
        - Each pixel emits at most one spike.
        - Higher intensity pixels fire earlier.
        - Lower intensity pixels fire later.
        - Pixels below threshold remain silent.

    Latency mapping:
        t = t_min + (t_max - t_min) * (1 - x)^gamma
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be > 0")

    if t_max is None:
        t_max = time_steps - 1

    t_min = int(t_min)
    t_max = int(t_max)

    if not (0 <= t_min <= t_max <= time_steps - 1):
        raise ValueError("Require 0 <= t_min <= t_max <= time_steps - 1")

    B = x.size(0)
    x_flat = x.view(B, -1)

    x01 = denorm_to_unit(x_flat)

    fire_mask = x01 > threshold

    x_clamped = torch.clamp(x01, 1e-6, 1.0)

    t_float = t_min + (t_max - t_min) * ((1.0 - x_clamped) ** float(gamma))
    spike_times = torch.round(t_float).long().clamp(0, time_steps - 1)

    spikes = torch.zeros(
        time_steps,
        B,
        x_flat.size(1),
        device=x.device,
        dtype=torch.float32,
    )

    b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, x_flat.size(1))
    n_idx = torch.arange(x_flat.size(1), device=x.device).unsqueeze(0).expand(B, x_flat.size(1))

    spikes[spike_times[fire_mask], b_idx[fire_mask], n_idx[fire_mask]] = 1.0

    return spikes