import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_trend(df, x_col, y_col, title, outpath):
    if x_col not in df.columns or y_col not in df.columns:
        return
    if df.empty:
        return

    df = df.sort_values(x_col)

    plt.figure(figsize=(6, 4))
    plt.plot(df[x_col], df[y_col], marker="o")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def best_per_param(df, param_col):
    if param_col not in df.columns or df.empty:
        return pd.DataFrame()

    # choose the epoch row with best val_acc for each parameter value
    idx = df.groupby(param_col)["val_acc"].idxmax()
    return df.loc[idx].copy().sort_values(param_col)


def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "results" / "dvs_results" / "dvs_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)

    print("Loaded:", csv_path)
    print("Columns:", list(df.columns))

    out_dir = project_root / "figures" / "dvs_trends"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "test_acc_pct",
        "mean_latency_t",
        "mean_spikes_per_sample",
        "throughput_sps",
    ]

    # Rate 
    rate_df = df[df["coding"] == "rate"].copy()
    rate_summary = best_per_param(rate_df, "rate_scale")

    for m in metrics:
        plot_trend(
            rate_summary,
            "rate_scale",
            m,
            f"Rate Coding: {m} vs rate_scale",
            out_dir / f"rate_{m}.png",
        )

    # TTFS 
    ttfs_df = df[df["coding"] == "ttfs"].copy()
    ttfs_summary = best_per_param(ttfs_df, "ttfs_alpha")

    for m in metrics:
        plot_trend(
            ttfs_summary,
            "ttfs_alpha",
            m,
            f"TTFS: {m} vs ttfs_alpha",
            out_dir / f"ttfs_{m}.png",
        )

    # Phase 
    phase_df = df[df["coding"] == "phase"].copy()
    phase_summary = best_per_param(phase_df, "phase_period")

    for m in metrics:
        plot_trend(
            phase_summary,
            "phase_period",
            m,
            f"Phase: {m} vs phase_period",
            out_dir / f"phase_{m}.png",
        )

    # Burst 
    burst_df = df[df["coding"] == "burst"].copy()
    burst_summary = best_per_param(burst_df, "burst_win")

    for m in metrics:
        plot_trend(
            burst_summary,
            "burst_win",
            m,
            f"Burst: {m} vs burst_win",
            out_dir / f"burst_{m}.png",
        )

    print("Saved trend plots to:", out_dir)


if __name__ == "__main__":
    main()