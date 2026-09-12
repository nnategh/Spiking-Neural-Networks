import pandas as pd
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results" / "dvs_results"

    codings = ["rate", "ttfs", "phase", "burst"]
    best_configs = []

    for coding in codings:
        path = results_dir / f"dvs_tuning_{coding}.csv"

        if not path.exists():
            print(f"[WARN] Missing file: {path}")
            continue

        df = pd.read_csv(path)

        if df.empty:
            print(f"[WARN] Empty file: {path}")
            continue

        best = df.loc[df["best_val_acc"].idxmax()]

        best_configs.append({
            "coding": coding,
            "best_val_acc": best["best_val_acc"],
            "best_epoch": best["best_epoch"],
            "T": best["T"],
            "lr": best["lr"],
            "tau": best["tau"],
            "tau_out": best["tau_out"],
            "hidden_dim": best["hidden_dim"],
            "test_acc_pct": best["test_acc_pct"],
            "mean_latency_t": best["mean_latency_t"],
            "mean_spikes_per_sample": best["mean_spikes_per_sample"],
            "throughput_sps": best["throughput_sps"],
            "rate_scale": best["rate_scale"] if "rate_scale" in best.index else None,
            "ttfs_alpha": best["ttfs_alpha"] if "ttfs_alpha" in best.index else None,
            "phase_period": best["phase_period"] if "phase_period" in best.index else None,
            "burst_win": best["burst_win"] if "burst_win" in best.index else None,
        })

    summary = pd.DataFrame(best_configs)

    if summary.empty:
        print("[ERROR] No valid tuning files found.")
        return

    out_path = results_dir / "dvs_best_configs.csv"
    summary.to_csv(out_path, index=False)

    print("Best configs by coding:")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()