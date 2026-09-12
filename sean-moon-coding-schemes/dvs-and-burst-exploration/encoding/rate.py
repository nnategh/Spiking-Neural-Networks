import torch

def denorm_to_unit(x: torch.Tensor) -> torch.Tensor:
    """
    Since raw EMNIST input is normalized to approximately [-1,1],
    To make pixel intensity, convert normalized tensor back to [0,1].
    """
    x01 = 0.5 * x + 0.5
    return torch.clamp(x01, 0.0, 1.0)

def rate_encode_flat(
        x: torch.Tensor, 
        time_steps: int,
        rate_scale:float = 0.5,
        poisson: bool = False,
        ) -> torch.Tensor:
    """
    Rate-based spike encoding with an explicit rate scaling parameter.
    This implementation extends the previous "direct injection" style rate coding into a stochastic firing model.

    Encoding principle:
        - Pixel intensity in [0,1] is interpreted as the grayscale brighteness of a pixel.
        - rate_scale controls the global spike firing probability. (scheme-specific DoF).
        - Spikes are generated independently at each timestep.

    Rate_scale:
        rate_scale is introduced as the primary encoding-specific degree of freedom (DoF),
        allowing sensitivity analysis of true rate modulation independent of training
        hyperparameters.

    Args:
        x: Input tensor [B, 1, 28, 28] or flattened [B, N]
        time_steps: Number of simulation timesteps (T)
        rate_scale: Global scaling factor applied to intensity -> firing probability
        poisson: If True, use Poisson sampling; otherwise Bernoulli sampling

    Returns:
        spikes: Binary spike tensor [T, B, N]
    """

    if time_steps <= 0:
        raise ValueError("time_steps must be > 0")

    B = x.size(0)

    # Flatten spatial dimensions to [B,N]
    x_flat = x.view(B, -1)
    # Recover intensity values in [0,1]
    x01 = denorm_to_unit(x_flat)

    # Apply scheme specific rate scaling 
    # lam represents expected firing intensity per timestep
    lam = torch.clamp(rate_scale * x01, min=0.0)

    if poisson:
        # Poisson sampling:
        # Each timestep samples spike counts from Poisson(lam).
        # Any positive count is treated as a spike (binary conversion).
        spikes = torch.poisson(lam.unsqueeze(0).repeat(time_steps, 1, 1))
        spikes = (spikes > 0).to(torch.float32)
    else:
        # Bernoulli sampling:
        # Each timestep independently samples spike ~ Bernoulli(p)
        p = torch.clamp(lam, 0.0, 1.0)
        # Generate uniform random numbers for comparison
        u = torch.rand((time_steps, B, x01.size(1)), device=x.device)
        # Spike if u < p
        spikes = (u < p.unsqueeze(0)).to(torch.float32)

    return spikes


