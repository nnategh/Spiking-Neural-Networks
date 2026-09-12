import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_bar(df, col, ylabel, title, outpath):
    plt.figure(figsize=(6, 4))
    plt.bar(df.index.str.upper(), df[col])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    csv_path = Path("results/dvs_results/dvs_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            "results/dvs_results/dvs_results.csv not found. Run the DVS experiments first."
        )

    df = pd.read_csv(csv_path)

    best_idx = df.groupby("coding")["test_acc_pct"].idxmax()
    summary = df.loc[best_idx].copy().set_index("coding").sort_index()

    out_dir = Path("figures/dvs")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(out_dir / "dvs_results_summary.csv")

    plot_bar(
        summary, "test_acc_pct", "Test Accuracy (%)",
        "DVS Accuracy Comparison Across Coding Schemes",
        out_dir / "fig_accuracy.png"
    )

    plot_bar(
        summary, "mean_latency_t", "Mean Latency (time step)",
        "DVS Latency Comparison Across Coding Schemes",
        out_dir / "fig_latency.png"
    )

    plot_bar(
        summary, "mean_spikes_per_sample", "Mean Spikes per Sample",
        "DVS Spike Count Comparison Across Coding Schemes",
        out_dir / "fig_spikes.png"
    )

    plot_bar(
        summary, "throughput_sps", "Throughput (samples/s)",
        "DVS Throughput Comparison Across Coding Schemes",
        out_dir / "fig_throughput.png"
    )

    print("Saved summary to:", out_dir / "dvs_results_summary.csv")
    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()