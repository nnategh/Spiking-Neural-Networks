import argparse
import time
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from datasets.DVS_GC_loader import get_dvs_gesture_dataloaders
from models.snn_dvs import SNN_DVS
from readout.aggregation import apply_readout
from torch.utils.data import random_split, DataLoader


def run_one_epoch(model, loader, opt, crit, device, coding: str, coding_args: Optional[Dict] = None):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for X, y in loader:
        X = X.to(device)
        if not torch.is_floating_point(X):
            X = X.float()
        y = y.to(device)

        opt.zero_grad(set_to_none=True)
        spk_out = model(X)
        logits = apply_readout(spk_out, coding=coding, coding_args=coding_args)
        loss = crit(logits, y)

        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())

    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model, loader, crit, device, coding: str, coding_args: Optional[Dict] = None):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    for X, y in loader:
        X = X.to(device)
        if not torch.is_floating_point(X):
            X = X.float()
        y = y.to(device)

        spk_out = model(X)
        logits = apply_readout(spk_out, coding=coding, coding_args=coding_args)
        loss = crit(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())

    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate_with_metrics(model, loader, crit, device, coding: str, coding_args: Optional[Dict] = None):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    total_spikes = 0.0
    total_latency = 0.0

    t_start = time.time()

    for X, y in loader:
        X = X.to(device)
        if not torch.is_floating_point(X):
            X = X.float()
        y = y.to(device)

        spk_out = model(X)
        logits = apply_readout(spk_out, coding=coding, coding_args=coding_args)
        loss = crit(logits, y)

        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())

        total_spikes += float(spk_out.sum().item())

        T = spk_out.shape[0]
        fired = (spk_out > 0)
        t_idx = torch.arange(T, device=spk_out.device).view(T, 1, 1).expand_as(spk_out)
        t_first = torch.where(fired, t_idx, torch.full_like(t_idx, fill_value=T))
        t_first = t_first.min(dim=0).values.float()   # [B, C]
        sample_latency = t_first.min(dim=1).values    # [B]
        sample_latency = torch.where(
            sample_latency == T,
            torch.full_like(sample_latency, float(T)),
            sample_latency
        )
        total_latency += float(sample_latency.sum().item())

    elapsed = time.time() - t_start

    mean_loss = total_loss / max(total, 1)
    mean_acc = correct / max(total, 1)
    mean_spikes_per_sample = total_spikes / max(total, 1)
    mean_latency_t = total_latency / max(total, 1)
    throughput_sps = total / max(elapsed, 1e-8)

    return {
        "test_loss": mean_loss,
        "test_acc": mean_acc,
        "test_acc_pct": 100.0 * mean_acc,
        "mean_spikes_per_sample": mean_spikes_per_sample,
        "mean_latency_t": mean_latency_t,
        "throughput_sps": throughput_sps,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--coding", type=str, default="rate", choices=["rate", "ttfs", "phase", "burst"])
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_classes", type=int, default=11)  # DVSGesture commonly 11 classes
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--tau_out", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    # coding-specific args
    p.add_argument("--rate_scale", type = float, default = 1.0)
    p.add_argument("--ttfs_alpha", type = float, default= 1.0)
    p.add_argument("--phase_period", type = int, default = 8)
    p.add_argument("--burst_win", type = int, default= 3)
    args = p.parse_args()

    results = []

    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Device:", device)

    train_loader, test_loader = get_dvs_gesture_dataloaders(
        data_dir=args.data_root,
        batch_size=args.batch_size,
        n_time_bins= args.T,
        num_workers=4,
        pin_memory=(device.type != "mps"),  # mps pin_memory warning
    )

    full_train = train_loader.dataset
    n = len(full_train)
    n_val = max(1, int(0.1 * n))
    n_train = n - n_val

    train_set, val_set = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type != "mps"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type != "mps"),
        drop_last=False,
    )

    # infer input_dim from one batch
    X0, _ = next(iter(train_loader))
    _, T, C, H, W = X0.shape
    input_dim = int(C * H * W)

    print("Example batch:", X0.shape)  # [B, T, C, H, W]
    print("input_dim:", input_dim)

    model = SNN_DVS(
        time_steps=args.T,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        tau=args.tau,
        tau_out=args.tau_out,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    # coding-specific args
    coding_args = {}

    if args.coding == "rate":
        coding_args["rate_scale"] = args.rate_scale
    elif args.coding == "ttfs":
        coding_args["t_max"] = args.T
        coding_args["alpha"] = args.ttfs_alpha
    elif args.coding == "phase":
        coding_args["period"] = args.phase_period
    elif args.coding == "burst":
        coding_args["win"] = args.burst_win

    print("Coding args:", coding_args)

    best_val = -1.0
    best_state = None

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, opt, crit, device, args.coding, coding_args)
        va_loss, va_acc = evaluate(model, val_loader,crit, device, args.coding, coding_args)
        print (
            f"Epoch {ep:02d} | train_acc = {tr_acc:.3f} loss = {tr_loss:.4f} |"
            f"val_acc = {va_acc:.3f} loss = {va_loss:.4f}"
        )

        results.append({
        "coding": args.coding,
        "epoch": ep,
        "train_acc":tr_acc,
        "train_loss": tr_loss,
        "val_acc": va_acc,
        "val_loss": va_loss,
        "T": args.T,
        "hidden_dim": args.hidden_dim,
        "tau": args.tau,
        "tau_out": args.tau_out,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,

        "rate_scale": args.rate_scale if args.coding == "rate" else None,
        "ttfs_alpha": args.ttfs_alpha if args.coding == "ttfs" else None,
        "ttfs_t_max": args.T if args.coding == "ttfs" else None,
        "phase_period": args.phase_period if args.coding == "phase" else None,
        "burst_win": args.burst_win if args.coding == "burst" else None,
        })

        if va_acc > best_val:
                    best_val = va_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

    test_metrics = evaluate_with_metrics(model, test_loader, crit, device, args.coding, coding_args)
    minutes = (time.time() - t0) / 60.0

    print(f"TEST | acc={test_metrics['test_acc']:.3f} loss={test_metrics['test_loss']:.4f}")
    print("Minutes:", minutes)
    print("Mean latency (t):", test_metrics["mean_latency_t"])
    print("Mean spikes/sample:", test_metrics["mean_spikes_per_sample"])
    print("Throughput (samples/s):", test_metrics["throughput_sps"])

    for r in results:
        r["test_acc"] = test_metrics["test_acc"]
        r["test_loss"] = test_metrics["test_loss"]
        r["test_acc_pct"] = test_metrics["test_acc_pct"]
        r["mean_latency_t"] = test_metrics["mean_latency_t"]
        r["mean_spikes_per_sample"] = test_metrics["mean_spikes_per_sample"]
        r["throughput_sps"] = test_metrics["throughput_sps"]
        r["minutes"] = minutes

    df = pd.DataFrame(results)

    out_dir = Path("results/dvs_results")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / "dvs_results.csv"

    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_csv(csv_path, index=False)
    print("Saved results to:", csv_path)

if __name__ == "__main__":
    main()