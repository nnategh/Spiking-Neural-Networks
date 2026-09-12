import torch


def ttfs_encode_flat(
        x: torch.Tensor,
        time_steps: int,
        gamma: float = 1.0,
        t_min: int = 0,
        t_max=None,
) -> torch.Tensor:
    """
    Time-to-First-Spike (TTFS) encoding with nonlinear intensity -> latency mapping.

    Encoding principle:
        - Information is encoded in the timing of the first spike.
        - Brighter (higher intensity) pixels fire earlier and darker pixels fire later.
        - Each neuron emits at most one spike.

    Latency mapping:
        t = t_min + (t_max - t_min) * (1 - x)^gamma

        - x in [0,1] is interpreted as stimulus intensity.
        - gamma controls the nonlinearity of the intensity -> latency mapping.

    Role of gamma (scheme-specific DoF):
        gamma = 1.0  -> linear mapping (baseline)
        gamma > 1.0  -> increases contrast: bright pixels fire much earlier
        gamma < 1.0  -> compresses contrast: spike times more evenly distributed

    Experimental objective:
        gamma is introduced as the primary encoding-specific degree of freedom (DoF)
        for TTFS. This allows sensitivity analysis of how nonlinear latency mapping
        affects performance, independent of training hyperparameters.

    Args:
        x: Input tensor [B,1,H,W] or flattened [B,N] (typically normalized)
        time_steps: Total simulation timesteps (T)
        gamma: Nonlinearity exponent controlling latency contrast
        t_min: Minimum allowable spike time
        t_max: Maximum allowable spike time (default: T-1)

    Returns:
        spikes: Binary spike tensor of shape [T, B, N],
                containing exactly one spike per neuron.
    """
    # Default upper latency bound is last timestep
    if t_max is None:
        t_max = time_steps - 1

    t_min = int(t_min)
    t_max = int(t_max)

    # Ensure valid latency range
    if not (0 <= t_min <= t_max <= time_steps - 1):
        raise ValueError("Require 0 <= t_min <= t_max <= time_steps-1")

    B = x.size(0)

    # Flatten spatial dimensions to [B,N]
    x_flat = x.view(B, -1)

    # Recover intensity values in [0,1]
    # EMNIST inputs were normalized to approx [-1,1]
    x_denorm = torch.clamp(0.5 * x_flat + 0.5, 0.0, 1.0)

    # Avoid exact zeros 
    eps = 1e-3
    x_clamped = torch.clamp(x_denorm, eps, 1.0)

    # Nonlinear intensity -> latency transformation
    # Higher intensity -> smaller (earlier) spike time
    t_float = t_min + (t_max - t_min) * ((1.0 - x_clamped) ** float(gamma))

    # Discretize to integer timestep
    spike_times = torch.round(t_float).long().clamp(0, time_steps - 1)

    # Allocate spike tensor
    spikes = torch.zeros(time_steps, B, x_flat.size(1), device=x.device)

    # Indexing to place exactly one spike per neuron
    idx = torch.arange(x_flat.size(1), device=x.device).unsqueeze(0).expand(B, -1)
    spikes[spike_times, torch.arange(B, device=x.device).unsqueeze(1), idx] = 1.0

    return spikes