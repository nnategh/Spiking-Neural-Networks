import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        for key, value in row.items():
            if value is None or value == "":
                row[key] = None
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass  # leave strings (coding, readout, experiment, ...) as-is

    return rows


def group_mean_std(rows, group_keys, metric_keys):
    """
    Groups rows by `group_keys` (e.g. ["coding"]) and returns, for each
    group, the mean/std across rows (typically across seeds) for every
    key in `metric_keys`.
    """
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    summary = {}
    for key, group_rows in groups.items():
        summary[key] = {}
        for metric in metric_keys:
            values = [r[metric] for r in group_rows if r[metric] is not None]
            summary[key][metric] = (np.mean(values), np.std(values))

    return summary


def plot_baseline(csv_path, output_dir):
    rows = load_rows(csv_path)

    metrics = [
        ("test_acc", "Accuracy"),
        ("avg_latency", "Latency (timesteps to decision)"),
        ("avg_input_spikes", "Input spike count"),
        ("avg_hidden_spikes", "Hidden spike count"),
        ("avg_energy_cost", "Energy cost (synaptic ops proxy)"),
    ]

    summary = group_mean_std(rows, ["coding"], [m for m, _ in metrics])
    codings = sorted(summary.keys(), key=lambda k: k[0])
    labels = [c[0] for c in codings]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))

    for ax, (metric, title) in zip(axes, metrics):
        means = [summary[c][metric][0] for c in codings]
        stds = [summary[c][metric][1] for c in codings]

        ax.bar(labels, means, yerr=stds, capsize=5, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_title(title)
        ax.set_ylabel(metric)

    fig.suptitle("Baseline comparison: rate vs ttfs vs phase (mean +/- std across seeds)")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "baseline_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_epochs(csv_path, output_dir):
    rows = load_rows(csv_path)

    # One curve set per (coding, num_bins, cycle_steps, threshold, repeat_cycles)
    # combo, averaged across seeds.
    config_keys = ["coding", "num_bins", "cycle_steps", "threshold", "repeat_cycles"]
    configs = {}
    for row in rows:
        key = tuple(row[k] for k in config_keys)
        configs.setdefault(key, []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for key, group_rows in sorted(configs.items(), key=lambda kv: str(kv[0])):
        epochs = sorted(set(r["epoch"] for r in group_rows))

        train_acc_by_epoch = []
        val_acc_by_epoch = []
        for e in epochs:
            epoch_rows = [r for r in group_rows if r["epoch"] == e]
            train_acc_by_epoch.append(np.mean([r["train_acc"] for r in epoch_rows]))
            val_acc_by_epoch.append(np.mean([r["val_acc"] for r in epoch_rows]))

        label = key[0]  # coding name is enough to distinguish in the baseline case
        axes[0].plot(epochs, train_acc_by_epoch, label=label)
        axes[1].plot(epochs, val_acc_by_epoch, label=label)

    axes[0].set_title("Train accuracy")
    axes[1].set_title("Validation accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()

    fig.suptitle("Training curves (mean across seeds)")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "epoch_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


GRID_METRICS = [
    # (csv column, display title, colormap, value format, higher_is_better)
    ("test_acc", "Mean test accuracy", "viridis", "{:.3f}", True),
    ("avg_latency", "Mean latency (timesteps to decision)", "viridis_r", "{:.1f}", False),
    ("avg_energy_cost", "Mean energy cost (synaptic ops proxy)", "viridis_r", "{:.0f}", False),
]


def plot_grid_heatmap(csv_path, output_dir, metrics=None):
    rows = load_rows(csv_path)

    metrics = metrics or GRID_METRICS
    metric_names = [m[0] for m in metrics]
    summary = group_mean_std(rows, ["num_bins", "cycle_steps"], metric_names)

    num_bins_values = sorted(set(k[0] for k in summary))
    cycle_steps_values = sorted(set(k[1] for k in summary))

    for metric, title, cmap, fmt, _ in metrics:
        grid = np.full((len(num_bins_values), len(cycle_steps_values)), np.nan)
        for (nb, cs), stats in summary.items():
            i = num_bins_values.index(nb)
            j = cycle_steps_values.index(cs)
            grid[i, j] = stats[metric][0]

        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(grid, cmap=cmap, aspect="auto")

        ax.set_xticks(range(len(cycle_steps_values)))
        ax.set_xticklabels([int(c) for c in cycle_steps_values])
        ax.set_yticks(range(len(num_bins_values)))
        ax.set_yticklabels([int(b) for b in num_bins_values])
        ax.set_xlabel("cycle_steps")
        ax.set_ylabel("num_bins")
        ax.set_title(f"{title} across (num_bins, cycle_steps)", fontsize=11, wrap=True)

        for i, nb in enumerate(num_bins_values):
            for j, cs in enumerate(cycle_steps_values):
                if np.isnan(grid[i, j]):
                    continue
                ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center", color="white", fontsize=9)
                if cs < nb:
                    # cycle_steps < num_bins: phase bins collide onto the
                    # same offset slot (resolution collapses). Flagged,
                    # not excluded, since the full grid was requested.
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, edgecolor="red", linewidth=2,
                    ))

        fig.colorbar(im, ax=ax, label=metric)
        fig.text(0.5, 0.01, "Red outline: cycle_steps < num_bins (phase-bin collision, resolution collapses)",
                  ha="center", fontsize=8, color="#a33")
        fig.tight_layout(rect=[0, 0.04, 1, 1])

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"frequency_resolution_grid_heatmap_{metric}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=str, default=None, help="baseline results CSV")
    parser.add_argument("--epoch", type=str, default=None, help="*_epoch.csv from run_phase_grid")
    parser.add_argument("--grid", type=str, default=None, help="frequency_resolution_grid results CSV")
    parser.add_argument("--output-dir", type=str, default="results/plots")

    args = parser.parse_args()

    if args.baseline:
        plot_baseline(args.baseline, args.output_dir)

    if args.epoch:
        plot_epochs(args.epoch, args.output_dir)

    if args.grid:
        plot_grid_heatmap(args.grid, args.output_dir)

    if not (args.baseline or args.epoch or args.grid):
        parser.print_help()


if __name__ == "__main__":
    main()
