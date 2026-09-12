import pandas as pd
from pathlib import Path


def best_per_param(df, param_col):
    if param_col not in df.columns or df.empty:
        return pd.DataFrame()

    idx = df.groupby(param_col)["val_acc"].idxmax()
    return df.loc[idx].copy().sort_values(param_col)


def compute_sensitivity(values):
    if len(values) == 0:
        return None

    v_max = values.max()
    v_min = values.min()
    v_mean = values.mean()

    if v_mean == 0:
        return None

    return (v_max - v_min) / v_mean


def analyze_coding(df, coding, param_col, metrics):
    sub_df = df[df["coding"] == coding].copy()
    summary = best_per_param(sub_df, param_col)

    results = []

    for m in metrics:
        if m not in summary.columns:
            continue

        S = compute_sensitivity(summary[m])

        results.append({
            "coding": coding,
            "parameter": param_col,
            "metric": m,
            "sensitivity": S,
            "max": summary[m].max(),
            "min": summary[m].min(),
            "mean": summary[m].mean(),
        })

    return results


def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "results" / "dvs_results" / "dvs_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found.")

    df = pd.read_csv(csv_path)

    metrics = [
        "test_acc_pct",
        "mean_latency_t",
        "mean_spikes_per_sample",
        "throughput_sps",
    ]

    all_results = []

    # Rate
    all_results += analyze_coding(df, "rate", "rate_scale", metrics)

    # TTFS
    all_results += analyze_coding(df, "ttfs", "ttfs_alpha", metrics)

    # Phase
    all_results += analyze_coding(df, "phase", "phase_period", metrics)

    # Burst
    all_results += analyze_coding(df, "burst", "burst_win", metrics)

    result_df = pd.DataFrame(all_results)

    out_path = project_root / "results" / "dvs_results" / "dvs_sensitivity.csv"
    result_df.to_csv(out_path, index=False)

    print("Saved sensitivity results to:", out_path)
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()