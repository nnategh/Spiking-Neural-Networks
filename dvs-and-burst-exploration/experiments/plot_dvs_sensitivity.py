import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

def main():
    project_root = Path(__file__).resolve().parents[1]

    in_path = project_root / "results" / "dvs_results" / "dvs_sensitivity.csv"
    out_dir = project_root / "figures" / "dvs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    metrics = [
        ("test_acc_pct", "Accuracy Sensitivity"),
        ("mean_latency_t", "Latency Sensitivity"),
        ("mean_spikes_per_sample", "Spikes Sensitivity"),
        ("throughput_sps", "Throughput Sensitivity"),
    ]

    # pivot: rows=coding, cols=metric
    piv = df.pivot(index="coding", columns="metric", values="sensitivity")

    # enforce order
    piv = piv.reindex(["rate", "ttfs", "phase", "burst"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    for ax, (m, title) in zip(axes, metrics):

        if m not in piv.columns:
            ax.set_visible(False)
            continue

        y = piv[m].fillna(0.0)

        ax.bar(y.index.str.upper(), y.values, color="#4C72B0")
        y_max = max(1.0, y.max())
        ax.set_ylim(0, y_max * 1.2)

        ax.set_title(title, pad=10)
        ax.set_ylabel("Sensitivity S = (max − min) / mean")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("DVS Readout DoF Sensitivity (Normalized Range)", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_file = out_dir / "fig_dvs_sensitivity.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", out_file.resolve())

if __name__ == "__main__":
    main()