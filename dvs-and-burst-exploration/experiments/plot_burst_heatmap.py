import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


METRICS = [
    ("test_acc_pct", "Accuracy (%)", "Burst Coding: Accuracy"),
    ("mean_latency_t", "Mean Latency (timesteps)", "Burst Coding: Latency"),
    ("spikes_per_sample", "Spikes per Sample", "Burst Coding: Spike Count"),
    ("throughput_sps", "Throughput (samples/s)", "Burst Coding: Throughput"),
]


def save_heatmap(pivot: pd.DataFrame, out_path: Path, title: str, cbar_label: str):
    fig = plt.figure(figsize=(6.5, 5.0))
    ax = plt.gca()

    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(r) for r in pivot.index])

    ax.set_xlabel("isi")
    ax.set_ylabel("n_max")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close(fig)


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

    df = pd.read_csv(in_path)

    required_cols = {
        "n_max",
        "isi",
        "test_acc_pct",
        "mean_latency_t",
        "spikes_per_sample",
        "throughput_sps",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values(["n_max", "isi"])

    for metric, cbar_label, title in METRICS:
        pivot = df.pivot(index="n_max", columns="isi", values=metric)
        out_path = out_dir / f"burst_heatmap_{metric}.png"
        save_heatmap(
            pivot=pivot,
            out_path=out_path,
            title=title,
            cbar_label=cbar_label,
        )

    print("Saved heatmaps to:", out_dir.resolve())


if __name__ == "__main__":
    main()