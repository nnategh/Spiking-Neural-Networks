import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


METRICS = [
    ("test_acc_pct", "Accuracy (%)"),
    ("mean_latency_t", "Mean Latency (timesteps)"),
    ("spikes_per_sample", "Spikes per Sample"),
    ("throughput_sps", "Throughput (samples/s)"),
]

def plot_vs_isi(df, metric, ylabel, out_dir):
    plt.figure(figsize=(6, 4))

    for n_max, sub in df.groupby("n_max"):
        sub = sub.sort_values("isi")
        plt.plot(
            sub["isi"],
            sub[metric],
            marker="o",
            label=f"n_max={n_max}",
        )

    plt.xlabel("isi")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs isi")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_vs_isi.png", dpi=200)
    plt.close()


def plot_vs_nmax(df, metric, ylabel, out_dir):
    plt.figure(figsize=(6, 4))

    for isi, sub in df.groupby("isi"):
        sub = sub.sort_values("n_max")
        plt.plot(
            sub["n_max"],
            sub[metric],
            marker="o",
            label=f"isi={isi}",
        )

    plt.xlabel("n_max")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs n_max")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"{metric}_vs_n_max.png", dpi=200)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_csv", type=str, default="results/emnist_trend_results/burst_trend_results.csv")
    p.add_argument("--out_dir", type=str, default="figures/emnist/burst")
    args = p.parse_args()

    in_path = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found. Run burst trend experiment first.")

    df = pd.read_csv(in_path).sort_values(["n_max", "isi"])

    for metric, ylabel in METRICS:
        plot_vs_isi(df, metric, ylabel, out_dir)
        plot_vs_nmax(df, metric, ylabel, out_dir)

    print("Saved line plots to:", out_dir.resolve())

if __name__ == "__main__":
    main()