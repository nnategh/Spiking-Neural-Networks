import argparse
import time
import pandas as pd
import torch

from models.snn_emnist import SNN_EMNIST
from train.train_emnist import train_one_epoch
from train.evaluate_emnist import evaluate
from datasets.EMNIST_loader import get_emnist_loaders


def train_and_test(
    n_max,
    isi,
    args,
    device
):
    encoding_args = {
        "max_spikes": n_max,
        "isi": isi,
    }
    if args.dt is not None:
        encoding_args["dt"] = args.dt

    model = SNN_EMNIST(
        time_steps=args.T,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        coding="burst",
        tau_out=args.tau_out,
        encoding_args=encoding_args,
    ).to(device)

    train_loader, val_loader, test_loader = get_emnist_loaders(
        batch_size=args.batch_size,
        seed=args.seed
    )

    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = -1
    best_state = None

    for _ in range(args.epochs):
        train_one_epoch(model, train_loader, opt, crit, device)
        _, val_acc, _ = evaluate(model, val_loader, crit, device)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_m = evaluate(
        model,
        test_loader,
        crit,
        device,
        return_latency_dist=False
    )

    return {
        "test_acc_pct": float(test_acc * 100),
        "mean_latency_t": float(test_m["mean_latency_t"]),
        "spikes_per_sample": float(test_m["spikes_per_sample"]),
        "throughput_sps": float(test_m["throughput_sps"]),
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--out", default="burst_trend_results.csv")

    p.add_argument("--n_max_list", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--isi_list", nargs="*", type=int, default=[1, 2, 3, 4])

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--T", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--tau_out", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_classes", type=int, default=47)

    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    run_id = 0
    t0 = time.time()

    for n_max in args.n_max_list:
        for isi in args.isi_list:
            run_id += 1
            print(f"\n[{run_id}] n_max={n_max} isi={isi}")

            metrics = train_and_test(n_max, isi, args, device)
            rows.append({
                "n_max": n_max,
                "isi": isi,
                **metrics
            })

            pd.DataFrame(rows).to_csv(args.out, index=False)

    print("\nSaved:", args.out)
    print("Total minutes:", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()