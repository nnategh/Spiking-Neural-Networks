import argparse
import csv
import os
import copy
import torch
import torch.nn as nn

from config.phase_grid import BASE_CONFIG, EXPERIMENTS
from readouts import rate_readout, ttfs_readout, phase_readout
from models.snn import SNNClassifier
from datasets.cifar10 import get_cifar10_dataloaders
from datasets.emnist import get_emnist_dataloaders

# (input_dim, num_classes) per dataset -- static frame-based, flattened input.
DATASET_DIMS = {
    "emnist": (28 * 28, 47),
    "cifar10": (3 * 32 * 32, 10),
}


def apply_readout(spk, readout, cycle_steps=None, repeat_cycles=None):
    if readout == "rate":
        return rate_readout(spk)

    if readout == "ttfs":
        return ttfs_readout(spk)

    if readout == "phase" or readout == "phase_windowed":
        # With repeat_cycles=False, no new input arrives after
        # t=cycle_steps, so the post-cycle "coasting" tail must be
        # excluded -- otherwise the cosine pattern keeps going over it
        # and injects a period-dependent residual unrelated to the
        # actual encoded information (see readouts/phase.py's `window`
        # docstring). With repeat_cycles=True, spikes recur throughout
        # the whole window, so no windowing is needed.
        window = cycle_steps if repeat_cycles is not True else None
        return phase_readout(spk, period=cycle_steps, window=window)

    raise ValueError(f"Unknown readout: {readout}")


def get_loaders(config, seed=None):
    if seed is None:
        seed = config["seed"]

    dataset_name = config.get("dataset", "emnist")

    if dataset_name == "cifar10":
        return get_cifar10_dataloaders(batch_size=config["batch_size"], seed=seed)

    if dataset_name == "emnist":
        return get_emnist_dataloaders(
            batch_size=config["batch_size"],
            val_ratio=0.1,
            seed=seed,
            num_workers=2,
        )

    raise ValueError(f"Unknown dataset: {dataset_name}")


def unpack_model_outputs(outputs):
    if len(outputs) == 4:
        _, output_seq, hidden_spike_count, _ = outputs
    else:
        _, output_seq, hidden_spike_count = outputs

    return output_seq, hidden_spike_count


def decision_latency(output_seq):
    """
    Per-sample time-to-decision: the earliest timestep from which the
    running argmax stays equal to the final-timestep prediction all the
    way to the end. Works on any coding scheme's output_seq [T,B,C],
    since it only looks at the accumulated readout, not the encoding.
    """
    T = output_seq.size(0)

    preds = output_seq.argmax(dim=-1)  # [T, B]
    final_pred = preds[-1]  # [B]

    matches_final = (preds == final_pred.unsqueeze(0)).float()  # [T, B]

    # Leading run length of "still matches final" when scanned backwards.
    stable_run = torch.cumprod(matches_final.flip(0), dim=0).sum(dim=0)  # [B]

    return T - stable_run  # [B], 0-indexed earliest stable timestep


def train_one_epoch(model, loader, optimizer, criterion, setting, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0
    total_hidden_spikes = 0.0
    total_input_spikes = 0.0
    total_decision_latency = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x, return_seq=True)
        output_seq, hidden_spike_count = unpack_model_outputs(outputs)

        logits = apply_readout(
            output_seq,
            setting["readout"],
            setting["cycle_steps"],
            setting.get("repeat_cycles"),
        )

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)
        total_hidden_spikes += hidden_spike_count.item()
        total_input_spikes += model.last_input_spike_count.item()
        total_decision_latency += decision_latency(output_seq).sum().item()

    avg_loss = total_loss / total
    acc = correct / total
    avg_hidden_spikes = total_hidden_spikes / total
    avg_input_spikes = total_input_spikes / total
    avg_decision_latency = total_decision_latency / total

    return avg_loss, acc, avg_hidden_spikes, avg_input_spikes, avg_decision_latency


@torch.no_grad()
def evaluate(model, loader, criterion, setting, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    total_hidden_spikes = 0.0
    total_input_spikes = 0.0
    total_decision_latency = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x, return_seq=True)
        output_seq, hidden_spike_count = unpack_model_outputs(outputs)

        logits = apply_readout(
            output_seq,
            setting["readout"],
            setting["cycle_steps"],
            setting.get("repeat_cycles"),
        )

        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)
        total_hidden_spikes += hidden_spike_count.item()
        total_input_spikes += model.last_input_spike_count.item()
        total_decision_latency += decision_latency(output_seq).sum().item()

    avg_loss = total_loss / total
    acc = correct / total
    avg_hidden_spikes = total_hidden_spikes / total
    avg_input_spikes = total_input_spikes / total
    avg_decision_latency = total_decision_latency / total

    return avg_loss, acc, avg_hidden_spikes, avg_input_spikes, avg_decision_latency


def build_model(setting, config, device):
    encoding_args = {}

    if setting["coding"] == "rate":
        if setting.get("rate_scale") is not None:
            encoding_args["rate_scale"] = setting["rate_scale"]

    if setting["coding"] == "ttfs":
        if setting.get("threshold") is not None:
            encoding_args["threshold"] = setting["threshold"]

    if setting["coding"] == "phase":
        encoding_args = {
            "num_bins": setting["num_bins"],
            "cycle_steps": setting["cycle_steps"],
        }
        if setting.get("threshold") is not None:
            encoding_args["threshold"] = setting["threshold"]
        if setting.get("repeat_cycles") is not None:
            encoding_args["repeat_cycles"] = setting["repeat_cycles"]

    input_dim, num_classes = DATASET_DIMS[config.get("dataset", "emnist")]

    model = SNNClassifier(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        num_classes=num_classes,
        time_steps=config["time_steps"],
        coding=setting["coding"],
        encoding_args=encoding_args,
    ).to(device)

    return model


def run_one_setting(setting, config):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    seed = setting.get("seed")
    if seed is None:
        seed = config.get("seed", 0)

    print("=" * 60)
    print(f"Experiment: {setting['experiment']}")
    print(f"Device: {device}")
    print(f"Coding: {setting['coding']}")
    print(f"Readout: {setting['readout']}")
    print(f"num_bins: {setting['num_bins']}")
    print(f"cycle_steps: {setting['cycle_steps']}")
    print(f"threshold: {setting.get('threshold')}")
    print(f"repeat_cycles: {setting.get('repeat_cycles')}")
    print(f"rate_scale: {setting.get('rate_scale')}")
    print(f"seed: {seed}")
    print("=" * 60)

    torch.manual_seed(seed)

    train_loader, val_loader, test_loader = get_loaders(config, seed=seed)

    model = build_model(setting, config, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    epoch_logs = []

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc, train_hidden_spikes, train_input_spikes, train_latency = train_one_epoch(
            model, train_loader, optimizer, criterion, setting, device
        )

        val_loss, val_acc, val_hidden_spikes, val_input_spikes, val_latency = evaluate(
            model, val_loader, criterion, setting, device
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"train_hidden_spikes={train_hidden_spikes:.2f}, train_input_spikes={train_input_spikes:.2f}, "
            f"train_latency={train_latency:.2f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
            f"val_hidden_spikes={val_hidden_spikes:.2f}, val_input_spikes={val_input_spikes:.2f}, "
            f"val_latency={val_latency:.2f}"
        )

        epoch_logs.append({
            "experiment": setting["experiment"],
            "coding": setting["coding"],
            "num_bins": setting["num_bins"],
            "cycle_steps": setting["cycle_steps"],
            "threshold": setting.get("threshold"),
            "repeat_cycles": setting.get("repeat_cycles"),
            "rate_scale": setting.get("rate_scale"),
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_hidden_spikes, test_input_spikes, test_latency = evaluate(
        model, test_loader, criterion, setting, device
    )

    # Synaptic-operations proxy: each spike fans out to every downstream
    # neuron in the next fully-connected layer (input->hidden->output).
    # Higher = more energy spent, i.e. this is a *cost*, not a score.
    avg_energy_cost = (
        test_input_spikes * model.hidden_dim
        + test_hidden_spikes * model.num_classes
    )

    result = {
        "experiment": setting["experiment"],
        "coding": setting["coding"],
        "readout": setting["readout"],
        "num_bins": setting["num_bins"],
        "cycle_steps": setting["cycle_steps"],
        "threshold": setting.get("threshold"),
        "repeat_cycles": setting.get("repeat_cycles"),
        "rate_scale": setting.get("rate_scale"),
        "seed": seed,
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "loss": test_loss,
        "avg_hidden_spikes": test_hidden_spikes,
        "avg_input_spikes": test_input_spikes,
        "avg_energy_cost": avg_energy_cost,
        "avg_latency": test_latency,
    }

    return result, epoch_logs


def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "experiment",
        "coding",
        "readout",
        "num_bins",
        "cycle_steps",
        "threshold",
        "repeat_cycles",
        "rate_scale",
        "seed",
        "test_acc",
        "best_val_acc",
        "loss",
        "avg_hidden_spikes",
        "avg_input_spikes",
        "avg_energy_cost",
        "avg_latency",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_epoch_logs(epoch_logs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "experiment",
        "coding",
        "num_bins",
        "cycle_steps",
        "threshold",
        "repeat_cycles",
        "rate_scale",
        "seed",
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_logs)


def merge_rows(existing_path, new_rows, filter_key, filter_value):
    """
    Loads rows already saved at `existing_path` (if any), drops rows
    where `filter_key == filter_value` (the ones we just recomputed),
    and returns those kept rows followed by `new_rows`. Lets --coding
    replace just one scheme's rows in a CSV without recomputing (and
    overwriting) the others.
    """
    if not os.path.exists(existing_path):
        return new_rows

    with open(existing_path, newline="") as f:
        existing_rows = list(csv.DictReader(f))

    kept = [r for r in existing_rows if r.get(filter_key) != filter_value]
    return kept + new_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=[
            "baseline",
            "frequency",
            "resolution",
            "threshold_ttfs",
            "threshold_phase",
            "repeat_cycles",
            "frequency_resolution_grid",
            "rate_scale",
            "phase_neighborhood",
            "phase_neighborhood_windowed",
            "cifar10_baseline",
            "cifar10_threshold_ttfs",
            "cifar10_threshold_phase",
            "cifar10_frequency_resolution_grid",
        ],
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override BASE_CONFIG['epochs'] for this run only.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["emnist", "cifar10"],
        help="Override the dataset for this run only (auto-detected from --exp name otherwise).",
    )
    parser.add_argument(
        "--coding",
        type=str,
        default=None,
        choices=["rate", "ttfs", "phase"],
        help="Only run settings for this coding scheme, and merge the "
             "results into the existing output CSV (replacing prior rows "
             "for this coding, keeping others untouched) instead of "
             "overwriting the whole file.",
    )

    args = parser.parse_args()

    config = BASE_CONFIG.copy()
    if args.exp.startswith("cifar10_"):
        config["dataset"] = "cifar10"
    if args.dataset is not None:
        config["dataset"] = args.dataset
    if args.epochs is not None:
        config["epochs"] = args.epochs

    settings = EXPERIMENTS[args.exp]
    if args.coding is not None:
        settings = [s for s in settings if s["coding"] == args.coding]

    results = []
    all_epoch_logs = []

    for setting in settings:
        result, epoch_logs = run_one_setting(setting, config)
        results.append(result)
        all_epoch_logs.extend(epoch_logs)

    output_path = args.output
    if output_path is None:
        output_path = f"results/{args.exp}_results.csv"

    epoch_output_path = output_path.rsplit(".csv", 1)[0] + "_epoch.csv"

    if args.coding is not None:
        results = merge_rows(output_path, results, "coding", args.coding)
        all_epoch_logs = merge_rows(epoch_output_path, all_epoch_logs, "coding", args.coding)

    save_results(results, output_path)
    print(f"Saved results to {output_path}")

    save_epoch_logs(all_epoch_logs, epoch_output_path)
    print(f"Saved epoch logs to {epoch_output_path}")


if __name__ == "__main__":
    main()