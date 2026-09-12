import argparse
import subprocess
import sys


def run_command(cmd):
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--coding",
        type=str,
        required=True,
        choices=["rate", "ttfs", "phase", "burst"]
    )

    p.add_argument("--T", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--tau_out", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    # Coding-specific sweep values

    sweep_dict = {
        "rate": {
            "arg_name": "--rate_scale",
            "values": [0.25, 0.5, 1.0, 2.0, 4.0],
        },

        "ttfs": {
            "arg_name": "--ttfs_alpha",
            "values": [0.25, 0.5, 0.7, 1.0, 2.0, 4.0],
        },

        "phase": {
            "arg_name": "--phase_period",
            "values": [4, 6, 8, 10, 12, 14, 16],
        },

        "burst": {
            "arg_name": "--burst_win",
            "values": [2, 3, 4, 5, 6, 7, 8],
        },
    }

    cfg = sweep_dict[args.coding]

    print(f"\nCoding: {args.coding}")
    print("Sweep values:", cfg["values"])

    # Run sweep

    for value in cfg["values"]:

        cmd = [
            sys.executable,
            "-m",
            "experiments.run_dvs",

            "--coding", args.coding,

            "--T", str(args.T),
            "--lr", str(args.lr),

            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),

            "--hidden_dim", str(args.hidden_dim),
            "--tau", str(args.tau),
            "--tau_out", str(args.tau_out),

            "--seed", str(args.seed),

            cfg["arg_name"], str(value),
        ]

        run_command(cmd)

    print("\nDone.")
    print("Trend results appended to results/dvs_results/dvs_results.csv")


if __name__ == "__main__":
    main()