import math
import torch
from typing import Optional, Tuple, Union


def _to_unit_interval(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Ensure x is in [0, 1].
    If input is normalized to [-1, 1], convert it back to [0, 1].
    """
    if x.min() < -eps:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def phase_encode_flat(
    x: torch.Tensor,
    time_steps: int,
    flatten: bool = True,
    invert: bool = False,
    # B: number of phase bins
    num_bins: int = 8,      
    # L: cycle length in timesteps    
    cycle_steps: int = 8,      
    threshold: float = 0.0,
    device=None,
    dtype: torch.dtype = torch.float32,
    return_phase: bool = False,
    dt: Optional[float] = None,
    return_time: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float],
]:
    """
    Phase-of-firing encoding with decoupled phase resolution and cycle length.

    Definitions:
        - num_bins (B):
            Number of discrete phase bins in one cycle.
            Controls phase resolution.

        - cycle_steps (L):
            Number of timesteps in one oscillation cycle.
            Controls cycle duration and frequency.

        - phase bin b ∈ {0, ..., B-1}: Discrete phase index.

        - phase angle θ ∈ [0, 2π): θ = 2π * (b / B)

    Physical-time interpretation:
        - dt: seconds per timestep
        - cycle duration τ = L * dt
        - oscillation frequency f = 1 / (L * dt)

    Encoding mechanism:
        1. Map intensity x ∈ [0,1] to a phase bin: b = floor((1 - x) * (B - 1))

        2. Map phase bin to a timestep offset within the cycle: offset = round( b / (B - 1) * (L - 1))

        3. Emit spikes at: t_k = offset + kL

    Notes:
        - num_bins controls phase resolution.
        - cycle_steps controls oscillation speed / frequency.
        - These two are now independent parameters.
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be > 0")

    num_bins = int(num_bins)
    cycle_steps = int(cycle_steps)

    if num_bins <= 0:
        raise ValueError("num_bins must be > 0")
    if cycle_steps <= 0:
        raise ValueError("cycle_steps must be > 0")
    if cycle_steps > time_steps:
        raise ValueError(
            f"cycle_steps (L={cycle_steps}) must be <= time_steps (T={time_steps})."
        )

    if return_time and dt is None:
        raise ValueError("return_time=True requires dt.")
    if dt is not None and dt <= 0:
        raise ValueError("dt must be > 0 if provided.")

    if device is None:
        device = x.device

    x = x.to(device=device)
    x_unit = _to_unit_interval(x)

    if invert:
        x_unit = 1.0 - x_unit

    fire_mask = x_unit > threshold

    # Flatten spatial dimensions
    if x_unit.ndim == 2:
        x_flat = x_unit
        mask_flat = fire_mask
        Bsz, N = x_flat.shape
        out_flat = True
    elif x_unit.ndim == 4:
        Bsz = x_unit.shape[0]
        if flatten:
            x_flat = x_unit.view(Bsz, -1)
            mask_flat = fire_mask.view(Bsz, -1)
            N = x_flat.shape[1]
            out_flat = True
        else:
            out_flat = False
    else:
        raise ValueError(f"Unsupported input shape {x_unit.shape}")

    if not out_flat:
        raise ValueError("For now, use flatten=True.")

    spikes = torch.zeros((time_steps, Bsz, N), device=device, dtype=dtype)

    # 1) intensity -> phase bin in [0, num_bins-1]
    phase_float = (1.0 - x_flat) * (num_bins - 1)
    phase_bin = torch.floor(phase_float).long().clamp(0, num_bins - 1)

    # 2) phase bin -> phase angle
    theta = (2.0 * math.pi) * (phase_bin.to(torch.float32) / float(num_bins))

    # 3) phase bin -> timestep offset in [0, cycle_steps-1]
    if num_bins == 1:
        offset = torch.zeros_like(phase_bin)
    else:
        offset_float = phase_bin.to(torch.float32) / float(num_bins - 1)
        offset = torch.round(offset_float * (cycle_steps - 1)).long().clamp(0, cycle_steps - 1)

    b_idx = torch.arange(Bsz, device=device).unsqueeze(1).expand(Bsz, N)
    n_idx = torch.arange(N, device=device).unsqueeze(0).expand(Bsz, N)

    valid0 = mask_flat
    if not valid0.any():
        if not return_phase and not return_time:
            return spikes
        if return_time:
            t_ms = torch.arange(time_steps, device=device, dtype=torch.float32) * float(dt) * 1e3
            tau_ms = float(cycle_steps) * float(dt) * 1e3
            f_hz = 1.0 / (float(cycle_steps) * float(dt))
            return spikes, phase_bin, theta, t_ms, tau_ms, f_hz
        return spikes, phase_bin, theta

    # repeat once per cycle
    max_k = (time_steps - 1) // cycle_steps
    for k in range(max_k + 1):
        t_k = offset + k * cycle_steps
        valid = valid0 & (t_k >= 0) & (t_k < time_steps)
        if valid.any():
            spikes[t_k[valid], b_idx[valid], n_idx[valid]] = 1.0

    if not return_phase and not return_time:
        return spikes

    if return_time:
        t_ms = torch.arange(time_steps, device=device, dtype=torch.float32) * float(dt) * 1e3
        tau_ms = float(cycle_steps) * float(dt) * 1e3
        f_hz = 1.0 / (float(cycle_steps) * float(dt))
        return spikes, phase_bin, theta, t_ms, tau_ms, f_hz

    return spikes, phase_bin, theta