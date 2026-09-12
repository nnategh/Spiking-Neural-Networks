"""
Minimal proof-of-concept for Experiment 4 (learnable oscillation frequency).

Question: can `cycle_steps` in phase coding be learned via backprop?

This script demonstrates, with actual gradients, why the answer is "not
without a differentiable relaxation": the discrete (hard) phase encoder
gives `cycle_steps` a gradient of None/0 no matter what, while a soft
relaxation of the same encoder gives it a real, usable gradient and lets
it actually move under an optimizer.
"""

import argparse
import csv
import os

import torch
import torch.nn as nn

from encoders.ttfs import denorm_to_unit


def hard_phase_encode_param(x, time_steps, num_bins, cycle_steps_param, threshold):
    """
    Same discrete indexing logic as encoders/phase.py, but with
    cycle_steps passed in as a tensor to show what happens if you
    naively try to make it learnable: the round()+int() cast required
    for indexing severs the autograd graph completely.
    """
    B = x.size(0)
    x_flat = x.view(B, -1)
    x01 = denorm_to_unit(x_flat)
    fire_mask = x01 > threshold

    phase_float = (1.0 - x01) * (num_bins - 1)
    phase_bin = torch.floor(phase_float).long().clamp(0, num_bins - 1)

    cycle_steps_int = int(torch.round(cycle_steps_param).item())
    cycle_steps_int = max(cycle_steps_int, num_bins)

    if num_bins == 1:
        offset = torch.zeros_like(phase_bin)
    else:
        offset_float = phase_bin.float() / float(num_bins - 1)
        offset = torch.round(offset_float * (cycle_steps_int - 1)).long()
        offset = offset.clamp(0, cycle_steps_int - 1)

    T = time_steps
    N = x_flat.size(1)
    spikes = torch.zeros(T, B, N, device=x.device, dtype=torch.float32)

    b_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, N)
    n_idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, N)

    spikes[offset[fire_mask], b_idx[fire_mask], n_idx[fire_mask]] = 1.0

    return spikes


def soft_phase_encode_param(x, time_steps, num_bins, cycle_steps_param, threshold, sigma=1.0):
    """
    Differentiable relaxation: each pixel's spike is a Gaussian kernel
    centered at a continuous position that depends smoothly on
    cycle_steps_param, so gradient can flow back to it.
    """
    B = x.size(0)
    x_flat = x.view(B, -1)
    x01 = denorm_to_unit(x_flat)
    fire_mask = (x01 > threshold).float()

    phase_float = (1.0 - x01) * (num_bins - 1)
    phase_bin = torch.floor(phase_float)

    if num_bins == 1:
        offset_frac = torch.zeros_like(phase_bin)
    else:
        offset_frac = phase_bin / (num_bins - 1)

    pos = offset_frac * (cycle_steps_param - 1)  # [B, N], differentiable wrt cycle_steps_param

    T = time_steps
    time_grid = torch.arange(T, device=x.device, dtype=x.dtype).view(T, 1, 1)

    weights = torch.exp(-0.5 * ((time_grid - pos.unsqueeze(0)) / sigma) ** 2)
    weights = weights / (weights.sum(dim=0, keepdim=True) + 1e-8)

    spikes = weights * fire_mask.unsqueeze(0)
    return spikes


def timing_sensitive_feature(spikes, period=8.0):
    """
    A stand-in for the project's real phase_readout: weights each
    timestep by cos(2*pi*t/period) before summing. Unlike a plain
    sum-over-time (which is nearly blind to *when* a spike lands, only
    to whether it landed at all), this is actually sensitive to spike
    timing -- so it's the right kind of readout to test whether shifting
    cycle_steps changes the loss at all.
    """
    T = spikes.size(0)
    t = torch.arange(T, dtype=spikes.dtype, device=spikes.device)
    w = torch.cos(2 * torch.pi * t / period).view(T, 1, 1)
    return (spikes * w).sum(dim=0)


def save_records(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = ["variant", "step", "cycle_steps", "grad"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_diagnostic(output_path="results/learnable_frequency_diagnostic.csv"):
    torch.manual_seed(0)

    B, N, T = 32, 64, 16
    num_bins = 8
    threshold = 0.2
    num_classes = 5
    sigma = 1.5
    # Deliberately not an exact integer: at cycle_steps == num_bins (8.0)
    # every pixel's continuous position lands exactly on an integer grid
    # point, which is a symmetric special case where the *local* gradient
    # happens to cancel. 8.3 avoids that coincidence.
    init_cycle_steps = 8.3

    x = torch.rand(B, N) * 2 - 1  # mimic denorm_to_unit's expected [-1,1] input
    y = torch.randint(0, num_classes, (B,))
    criterion = nn.CrossEntropyLoss()

    print("=" * 60)
    print("1) HARD phase_encode with cycle_steps as a parameter")
    print("=" * 60)

    cycle_steps_hard = nn.Parameter(torch.tensor(init_cycle_steps))
    classifier_hard = nn.Linear(N, num_classes)

    spikes = hard_phase_encode_param(x, T, num_bins, cycle_steps_hard, threshold)
    feat = timing_sensitive_feature(spikes)
    logits = classifier_hard(feat)
    loss = criterion(logits, y)
    loss.backward()

    print(f"cycle_steps value: {cycle_steps_hard.item():.4f}")
    print(f"cycle_steps.grad : {cycle_steps_hard.grad}")

    print()
    print("=" * 60)
    print("2) SOFT (relaxed) phase_encode with cycle_steps as a parameter")
    print("=" * 60)

    cycle_steps_soft = nn.Parameter(torch.tensor(init_cycle_steps))
    classifier_soft = nn.Linear(N, num_classes)

    spikes = soft_phase_encode_param(x, T, num_bins, cycle_steps_soft, threshold, sigma=sigma)
    feat = timing_sensitive_feature(spikes)
    logits = classifier_soft(feat)
    loss = criterion(logits, y)
    loss.backward()

    print(f"cycle_steps value: {cycle_steps_soft.item():.4f}")
    print(f"cycle_steps.grad : {cycle_steps_soft.grad.item():.6e}")

    print()
    print("=" * 60)
    print("3) A few optimizer steps: does cycle_steps actually move?")
    print("=" * 60)

    records = []

    cycle_steps_hard2 = nn.Parameter(torch.tensor(init_cycle_steps))
    classifier_hard2 = nn.Linear(N, num_classes)
    opt_hard = torch.optim.SGD([cycle_steps_hard2] + list(classifier_hard2.parameters()), lr=0.5)

    records.append({"variant": "hard", "step": 0, "cycle_steps": cycle_steps_hard2.item(), "grad": None})
    for step in range(1, 21):
        opt_hard.zero_grad()
        spikes = hard_phase_encode_param(x, T, num_bins, cycle_steps_hard2, threshold)
        feat = timing_sensitive_feature(spikes)
        logits = classifier_hard2(feat)
        loss = criterion(logits, y)
        loss.backward()
        grad = cycle_steps_hard2.grad.item() if cycle_steps_hard2.grad is not None else None
        opt_hard.step()
        records.append({"variant": "hard", "step": step, "cycle_steps": cycle_steps_hard2.item(), "grad": grad})

    cycle_steps_soft2 = nn.Parameter(torch.tensor(init_cycle_steps))
    classifier_soft2 = nn.Linear(N, num_classes)
    opt_soft = torch.optim.SGD([cycle_steps_soft2] + list(classifier_soft2.parameters()), lr=0.5)

    records.append({"variant": "soft", "step": 0, "cycle_steps": cycle_steps_soft2.item(), "grad": None})
    for step in range(1, 21):
        opt_soft.zero_grad()
        spikes = soft_phase_encode_param(x, T, num_bins, cycle_steps_soft2, threshold, sigma=sigma)
        feat = timing_sensitive_feature(spikes)
        logits = classifier_soft2(feat)
        loss = criterion(logits, y)
        loss.backward()
        grad = cycle_steps_soft2.grad.item() if cycle_steps_soft2.grad is not None else None
        opt_soft.step()
        records.append({"variant": "soft", "step": step, "cycle_steps": cycle_steps_soft2.item(), "grad": grad})

    hard_trace = [r["cycle_steps"] for r in records if r["variant"] == "hard"]
    soft_trace = [r["cycle_steps"] for r in records if r["variant"] == "soft"]

    print("HARD cycle_steps trace (expected: frozen at init value):")
    print([f"{v:.4f}" for v in hard_trace])
    print()
    print("SOFT cycle_steps trace (expected: moves under gradient descent):")
    print([f"{v:.4f}" for v in soft_trace])

    save_records(records, output_path)
    print()
    print(f"Saved step-by-step trace to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/learnable_frequency_diagnostic.csv")
    args = parser.parse_args()

    run_diagnostic(output_path=args.output)


if __name__ == "__main__":
    main()
