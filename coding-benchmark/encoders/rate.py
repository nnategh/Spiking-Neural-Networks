import torch


def denorm_to_unit(x: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized image tensor from [-1,1] to [0,1].
    """
    x01 = 0.5 * x + 0.5
    return torch.clamp(x01, 0.0, 1.0)


def rate_encode(
    x: torch.Tensor,
    time_steps: int,
    rate_scale: float = 0.5,
    poisson: bool = False,
) -> torch.Tensor:
    """
    Rate encoding for static frame-based images.

    Expected input:
        x: [B, C, H, W] or [B, N]

    Returns:
        spikes: [T, B, C*H*W]

    Encoding principle:
        - Pixel intensity is interpreted as firing probability.
        - Higher pixel intensity produces higher firing probability.
        - Spikes are generated independently at each timestep.
    """
    if time_steps <= 0:
        raise ValueError("time_steps must be > 0")

    B = x.size(0)
    x_flat = x.view(B, -1)

    # EMNIST/CIFAR loaders use Normalize(mean=0.5, std=0.5), so values are in [-1,1].
    x01 = denorm_to_unit(x_flat)

    p = torch.clamp(rate_scale * x01, 0.0, 1.0)

    if poisson:
        spikes = torch.poisson(p.unsqueeze(0).repeat(time_steps, 1, 1))
        spikes = (spikes > 0).to(torch.float32)
    else:
        u = torch.rand((time_steps, B, x01.size(1)), device=x.device)
        spikes = (u < p.unsqueeze(0)).to(torch.float32)

    return spikes