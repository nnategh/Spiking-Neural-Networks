from typing import Optional
import torch

def rate_readout(spk: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Rate-based readout.

    Args:
        spk: output spikes
             expected shape: [T,B,C]
             T = timesteps, B = batch size, C = num classes
        normalize: if True, return average firing rate instead of spike count

    Returns:
        logits: [B, C]
    """
    if spk.dim() != 3:
        raise ValueError(f"Expected spk shape [T, B, C], but got {spk.shape}")

    logits = spk.float().sum(dim=0)  # [B, C]

    if normalize:
        logits = logits / spk.shape[0]

    return logits


def apply_readout(
    spk: torch.Tensor,
    readout: str = "rate",
    readout_args: Optional[dict] = None,
) -> torch.Tensor:
    readout_args = readout_args or {}
    readout = readout.lower()

    if readout == "rate":
        return rate_readout(
            spk,
            normalize=bool(readout_args.get("normalize", False)),
        )

    raise ValueError(f"Unknown readout type: {readout}")