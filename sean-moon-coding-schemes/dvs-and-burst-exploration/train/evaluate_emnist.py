import time
import torch

def decision_latency_from_seq(
    logits_seq: torch.Tensor,
    margin: float = 0.5,
    stable_k: int = 2,
) -> torch.Tensor:
    """
    Coding-scheme - agnostic decision latency.

    Args:
        logits_seq: [T, B, C] tensor of logits over time.
        margin: require (top1 - top2) >= margin to consider a decision.
        stable_k: require the predicted class AND margin to remain stable for k steps.

    Returns:
        lat: [B] tensor of decision timesteps in [0, T-1].
             If never decided, returns T-1.
    """
    if logits_seq.ndim != 3:
        raise ValueError(f"logits_seq must have shape [T,B,C], got {tuple(logits_seq.shape)}")

    T, B, C = logits_seq.shape
    if T == 0:
        raise ValueError("logits_seq has T=0")

    preds = logits_seq.argmax(dim=-1)  # [T, B]
    top2 = torch.topk(logits_seq, k=2, dim=-1).values  # [T, B, 2]
    margins = top2[..., 0] - top2[..., 1]              # [T, B]

    lat = torch.full((B,), T - 1, device=logits_seq.device, dtype=torch.long)

    # Note: This double loop is simple/robust.
    for b in range(B):
        for t in range(T):
            if margins[t, b] < margin:
                continue
            end = min(T, t + stable_k)
            if (preds[t:end, b] == preds[t, b]).all() and (margins[t:end, b] >= margin).all():
                lat[b] = t
                break

    return lat


# Robustness perturbations

def add_gaussian_noise(
    x: torch.Tensor,
    sigma: float,
    clamp_min: float = -1.0,
    clamp_max: float = 1.0,
) -> torch.Tensor:
    """
    Add Gaussian noise to inputs.

    Use clamp_min/max = (-1,1) if your inputs are normalized to [-1,1].
    Use clamp_min/max = (0,1) if your inputs are in [0,1].
    """
    return (x + sigma * torch.randn_like(x)).clamp(clamp_min, clamp_max)


def add_salt_pepper(
    x: torch.Tensor,
    p: float,
    *,
    low: float = -1.0,
    high: float = 1.0,
) -> torch.Tensor:
    """
    Salt-and-pepper noise for robustness evaluation.
    Models sparse, high-magnitude corruption such as dead or stuck pixels,
    which strongly affects temporally sparse coding schemes.
    """
    rnd = torch.rand_like(x)
    x2 = x.clone()
    x2[rnd < (p / 2)] = low
    x2[(rnd >= (p / 2)) & (rnd < p)] = high
    return x2


# Main evaluation

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    *,
    latency_margin: float = 0.5,
    stable_k: int = 2,
    noise_fn = None,
    return_latency_dist: bool = False,
):
    """
    Evaluate SNN model with SNN-relevant metrics.

    Requires model.forward(x, return_seq=True) returning:
        out: [B, C]
        out_seq: [T, B, C]
        spike_count: scalar (spike total over batch and all timesteps)

    Returns:
        avg_loss, acc, metrics dict
        metrics includes:
            - mean_latency_t
            - spikes_per_sample
            - throughput_sps
        optionally:
            - median_latency_t
            - early_decision_rate
    """
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    latency_sum = 0.0
    spike_sum = 0.0
    latency_all = [] if return_latency_dist else None

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if noise_fn is not None:
            x = noise_fn(x)

        out, out_seq, spike_count = model(x, return_seq=True)
        loss = criterion(out, y)

        bs = x.size(0)
        loss_sum += loss.item() * bs
        correct += (out.argmax(dim=1) == y).sum().item()
        total += bs

        lat = decision_latency_from_seq(out_seq, margin=latency_margin, stable_k=stable_k)
        latency_sum += lat.sum().item()

        if return_latency_dist:
            latency_all.append(lat.detach().cpu())

        spike_sum += float(spike_count)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.time()

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)

    metrics = {
        "mean_latency_t": latency_sum / max(total, 1),
        "spikes_per_sample": spike_sum / max(total, 1),
        "throughput_sps": total / max((t1 - t0), 1e-9),
    }

    if return_latency_dist and latency_all:
        lat_all = torch.cat(latency_all, dim=0)
        metrics["median_latency_t"] = lat_all.median().item()
        # fraction of samples that decided before halfway through the window
        metrics["early_decision_rate"] = (lat_all < (out_seq.size(0) // 2)).float().mean().item()

    return avg_loss, acc, metrics
