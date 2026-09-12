import torch

def _to_unit_interval(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """ 
    Ensure x is in [0, 1].
    If x appears to be normalized to [-1, 1] (common with Normalize(mean=0.5, std=0.5)),
    convert back to [0, 1] via (x + 1) / 2.
    """
    if x.min() < -eps:
        x = (x+1.0) / 2.0
    return x.clamp(0.0,1.0)

def burst_encode_flat( 
    x: torch.Tensor,
    time_steps: int,
    max_spikes = None,
    isi: int =1,
    threshold: float = 0.0,
    early: bool = True,
    device = None,
    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
    """
    Burst encoding (flat) producing spike trains of shape [T, B, N].

    Core idea:
        Encode pixel intensity using a *burst*: a short packet of K spikes emitted
        by each neuron, with fixed inter-spike interval (ISI).

    Input:
        x is interpreted as pixel intensity in [0,1] (after denormalization if needed).

    Burst size (spike count):
        For each neuron/pixel, burst count K is computed as:
            K = floor(x * max_spikes)
        then thresholded:
            if x <= threshold, set K = 0

        So higher intensity -> more spikes (larger burst), capped by max_spikes.

    Burst timing (ISI):
        Given K spikes, spike times are placed at:
            t_j = start + ISI * j,  for j = 0,1,...,K-1

        ISI is measured in *timesteps*:
            ISI = 1  -> spikes are consecutive (tight burst)
            ISI > 1  -> spikes are spaced out (longer burst window)

    Burst placement:
        - early=True: start = 0 (bursts begin immediately)
        - early=False: burst is approximately centered around T/2:
            start = mid - (ISI*(K-1))/2

        when centered, bursts may be clipped at the boundaries; out-of-range
        spike times are discarded via the (t_j < T) validity check.

    Scheme-specific degrees of freedom (DoF):

        - max_spikes:
            Controls burst magnitude scaling. It sets the maximum number of
            spikes a neuron can emit within the simulation window.

            Larger max_spikes -> stronger redundancy for high-intensity pixels.

        - isi:
            Inter-spike interval inside the burst (measured in timesteps).

            isi = 1  -> tightly packed burst
            isi > 1  -> temporally spread burst

    These DoF modify temporal spike structure without changing network architecture
    or training hyperparameters.
    """

    if time_steps <= 0:
        raise ValueError("time_steps must be > 0")

    if device is None:
        device = x.device

    if max_spikes is None:
        max_spikes = time_steps
    max_spikes = int(max_spikes)
    if not (1 <= max_spikes <= time_steps):
        raise ValueError("max_spikes must be in [1,time_steps]")
    
    isi = int(isi)
    if isi <= 0:
        raise ValueError("isi must be >= 1")
  
    x = x.to(device=device)
    x_unit = _to_unit_interval(x)

    # Flatten to [B, N]
    if x_unit.ndim == 4:
        B = x_unit.shape[0]
        x_flat = x_unit.view(B, -1)
    elif x_unit.ndim == 2:
        x_flat = x_unit
        B = x_flat.shape[0]
    else:
        raise ValueError(f"Unsupported input shape {x_unit.shape}")

    B, N = x_flat.shape
    spikes = torch.zeros((time_steps, B, N), device=device, dtype=dtype)

    # Determine burst count K per pixel: K in [0, max_spikes]
    # Use floor(x * max_spikes). x=1 -> max_spikes spikes, x=0 -> 0 spikes.
    K = torch.floor(x_flat * max_spikes).long().clamp(0,max_spikes)
    # Apply threshold: values <= threshold -> K=0
    K = torch.where(x_flat > threshold, K, torch.zeros_like(K))

    if not (K > 0).any():
        return spikes

    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    n_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)


    if early:
        start = torch.zeros_like(K)
    else:
        # center burst: start = mid - (isi*(K-1))/2
        mid = time_steps // 2
        start = mid - ((isi * (K - 1)) // 2)
    # place spikes at start + isi*j
    for j in range(max_spikes):
        active = K > j
        if not active.any():
            continue
        t_j = start + isi * j
        valid = active & (t_j >= 0) & (t_j < time_steps)
        if valid.any():
            spikes[t_j[valid], b_idx[valid], n_idx[valid]] = 1.0
    return spikes
 