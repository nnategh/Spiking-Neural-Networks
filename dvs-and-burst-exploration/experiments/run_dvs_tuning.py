import argparse
import itertools
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from datasets.DVS_GC_loader import get_dvs_gesture_dataloaders
from models.snn_dvs import SNN_DVS
from experiments.run_dvs import run_one_epoch, evaluate, evaluate_with_metrics


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def make_coding_args(args: argparse.Namespace) -> Dict:
    coding_args: Dict = {}

    if args.coding == "rate":
        coding_args["rate_scale"] = args.rate_scale
    elif args.coding == "ttfs":
        coding_args["t_max"] = args.T
        coding_args["alpha"] = args.ttfs_alpha
    elif args.coding == "phase":
        coding_args["period"] = args.phase_period
    elif args.coding == "burst":
        coding_args["win"] = args.burst_win

    return coding_args


def build_model(
    time_steps: int,
    input_dim: int,
    hidden_dim: int,
    num_classes: int,
    tau: float,
    tau_out: float,
    device: torch.device,
) -> SNN_DVS:
    model = SNN_DVS(
        time_steps=time_steps,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        tau=tau,
        tau_out=tau_out,
    ).to(device)
    return model


def main() -> None:
    p = argparse.ArgumentParser()

    # Data / model basics
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--coding", type=str, required=True, choices=["rate", "ttfs", "phase", "burst"])
    p.add_argument("--num_classes", type=int, default=11)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)

    # Search grids
    p.add_argument("--T_list", type=str, default="10,20,30")
    p.add_argument("--lr_list", type=str, default="5e-4,1e-3,2e-3")
    p.add_argument("--tau_list", type=str, default="2.0")
    p.add_argument("--tau_out_list", type=str, default="2.0")
    p.add_argument("--hidden_dim_list", type=str, default="256")

    # Fixed coding parameters during tuning
    p.add_argument("--rate_scale", type=float, default=1.0)
    p.add_argument("--ttfs_alpha", type=float, default=1.0)
    p.add_argument("--phase_period", type=int, default=8)
    p.add_argument("--burst_win", type=int, default=4)

    # Output
    p.add_argument("--out_csv", type=str, default=None)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Device:", device)

    T_list = parse_int_list(args.T_list)
    lr_list = parse_float_list(args.lr_list)
    tau_list = parse_float_list(args.tau_list)
    tau_out_list = parse_float_list(args.tau_out_list)
    hidden_dim_list = parse_int_list(args.hidden_dim_list)

    project_root = Path(__file__).resolve().parents[1]

    if args.out_csv is None:
        out_csv = project_root / "results" / "dvs_results" / f"dvs_tuning_{args.coding}.csv"
    else:
        out_csv = project_root / args.out_csv

    epoch_csv = project_root / "results" / "dvs_results" / f"dvs_epochlog_{args.coding}.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    epoch_rows = []

    grid = list(itertools.product(T_list, lr_list, tau_list, tau_out_list, hidden_dim_list))
    print(f"Total runs: {len(grid)}")

    for run_idx, (T, lr, tau, tau_out, hidden_dim) in enumerate(grid, start=1):
        print("=" * 80)
        print(
            f"Run {run_idx}/{len(grid)} | coding={args.coding} | "
            f"T={T}, lr={lr}, tau={tau}, tau_out={tau_out}, hidden_dim={hidden_dim}"
        )

        # Make a shallow args-like object for coding args
        run_args = argparse.Namespace(**vars(args))
        run_args.T = T
        run_args.lr = lr
        run_args.tau = tau
        run_args.tau_out = tau_out
        run_args.hidden_dim = hidden_dim

        coding_args = make_coding_args(run_args)
        print("Coding args:", coding_args)

        train_loader_full, test_loader = get_dvs_gesture_dataloaders(
            data_dir=args.data_root,
            batch_size=args.batch_size,
            n_time_bins=T,
            num_workers=args.num_workers,
            pin_memory=(device.type != "mps"),
        )

        full_train = train_loader_full.dataset
        n = len(full_train)
        n_val = max(1, int(0.1 * n))
        n_train = n - n_val

        train_set, val_set = random_split(
            full_train,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device.type != "mps"),
            drop_last=False,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type != "mps"),
            drop_last=False,
        )

        X0, _ = next(iter(train_loader))
        _, T_check, C, H, W = X0.shape
        assert T_check == T, f"Expected T={T}, got {T_check}"
        input_dim = int(C * H * W)

        model = build_model(
            time_steps=T,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=args.num_classes,
            tau=tau,
            tau_out=tau_out,
            device=device,
        )

        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()

        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_epoch = -1
        best_state: Optional[Dict[str, torch.Tensor]] = None

        t0 = time.time()

        for ep in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_one_epoch(
                model, train_loader, opt, crit, device, args.coding, coding_args
            )
            va_loss, va_acc = evaluate(
                model, val_loader, crit, device, args.coding, coding_args
            )

            print(
                f"Epoch {ep:02d} | train_acc={tr_acc:.3f} loss={tr_loss:.4f} | "
                f"val_acc={va_acc:.3f} loss={va_loss:.4f}"
            )
            epoch_rows.append({
                "coding": args.coding,
                "epoch": ep,
                "train_acc": tr_acc,
                "train_loss": tr_loss,
                "val_acc": va_acc,
                "val_loss": va_loss,
                "T": T,
                "lr": lr,
                "tau": tau,
                "tau_out": tau_out,
                "hidden_dim": hidden_dim,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "run_id": f"T{T}_lr{lr}_tau{tau}",
                "rate_scale": args.rate_scale if args.coding == "rate" else None,
                "ttfs_alpha": args.ttfs_alpha if args.coding == "ttfs" else None,
                "phase_period": args.phase_period if args.coding == "phase" else None,
                "burst_win": args.burst_win if args.coding == "burst" else None,
            })

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_val_loss = va_loss
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)

        test_metrics = evaluate_with_metrics(
            model, test_loader, crit, device, args.coding, coding_args
        )
        minutes = (time.time() - t0) / 60.0

        row = {
            "coding": args.coding,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "test_acc": test_metrics["test_acc"],
            "test_loss": test_metrics["test_loss"],
            "test_acc_pct": test_metrics["test_acc_pct"],
            "mean_latency_t": test_metrics["mean_latency_t"],
            "mean_spikes_per_sample": test_metrics["mean_spikes_per_sample"],
            "throughput_sps": test_metrics["throughput_sps"],
            "minutes": minutes,
            "T": T,
            "lr": lr,
            "tau": tau,
            "tau_out": tau_out,
            "hidden_dim": hidden_dim,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "rate_scale": args.rate_scale if args.coding == "rate" else None,
            "ttfs_alpha": args.ttfs_alpha if args.coding == "ttfs" else None,
            "ttfs_t_max": T if args.coding == "ttfs" else None,
            "phase_period": args.phase_period if args.coding == "phase" else None,
            "burst_win": args.burst_win if args.coding == "burst" else None,
        }
        all_rows.append(row)

    final_df = pd.DataFrame(all_rows)

    # Append existing results only once at the end
    if out_csv.exists():
        old_df = pd.read_csv(out_csv)
        final_df = pd.concat([old_df, final_df], ignore_index=True)

    final_df = final_df.sort_values(["coding", "best_val_acc"], ascending=[True, False])
    final_df.to_csv(out_csv, index=False)

    print("Saved tuning results to:", out_csv)
    print("\nTop results:")
    print(final_df.head(10).to_string(index=False))

    epoch_df = pd.DataFrame(epoch_rows)

    if epoch_csv.exists():
        old_epoch_df = pd.read_csv(epoch_csv)
        epoch_df = pd.concat([old_epoch_df, epoch_df], ignore_index=True)

    epoch_df.to_csv(epoch_csv, index=False)

    print("Saved epoch log to:", epoch_csv)

if __name__ == "__main__":
    main()