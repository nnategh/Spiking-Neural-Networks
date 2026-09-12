# SNN Coding-Scheme Benchmark

A benchmark comparing Rate, Time-to-First-Spike (TTFS), and Phase coding
for spiking neural networks (SNNs) on EMNIST and CIFAR-10.

> Full methodology, literature review, results, and future work: see
> [Technical Report](#technical-report) below.

## Overview

This repository benchmarks three neural coding schemes — **Rate**,
**Time-to-First-Spike (TTFS)**, and **Phase** — under a single-hidden-layer
LIF architecture, applying the same coding principle at both the input
encoding stage and the output readout stage for each scheme. Each scheme
is compared on **accuracy**, **synaptic-operation (SynOps) cost**, **spike
activity**, and **decision latency**, on both **EMNIST (Balanced)** and
**CIFAR-10**.

Fairness setup: the network architecture, training protocol (epochs,
learning rate, seeds), and time budget (`time_steps`) are held identical
across all three schemes. Each scheme's own encoding hyperparameters
(e.g. TTFS/Phase threshold, Phase `num_bins`/`cycle_steps`) are tuned
separately per scheme and per dataset — they are not shared or forced to
be identical, since they are internal to each coding scheme rather than
architectural constants.

The project investigates how these schemes trade off against each other,
how Phase coding's two temporal parameters (`num_bins`, `cycle_steps`)
interact, and whether coding-scheme behavior transfers between datasets.

## Key Findings

- On EMNIST, **Rate** achieved the highest accuracy (~81.7%).
- **TTFS** had the lowest SynOps (energy-proxy) cost.
- **Phase** achieved the lowest decision latency.
- No single coding scheme dominated across all evaluation metrics.
- The optimal Phase configuration was dataset-dependent:
  - EMNIST: `num_bins=4, cycle_steps=8`
  - CIFAR-10: `num_bins=4, cycle_steps=4`
- Overall, coding-scheme performance depends on both the dataset and which
  evaluation objective (accuracy, energy, or latency) is prioritized — no
  scheme is universally "best."

## Coding Schemes

- **Rate** — pixel intensity is used as a per-timestep firing probability.
- **Time-to-First-Spike (TTFS)** — each pixel emits at most one spike; higher
  intensity fires earlier.
- **Phase** — pixel intensity is mapped to a phase offset within a periodic
  oscillation cycle.

## Datasets

- **EMNIST (Balanced split, 47 classes)** — downloaded automatically on
  first run (`download=True` in `datasets/emnist.py`).
- **CIFAR-10 (10 classes)** — not downloaded automatically; place the
  extracted dataset at `datasets/cifar-10-batches-py/` before running
  CIFAR-10 experiments (`download=False` in `datasets/cifar10.py`).

## Evaluation Metrics

- **Accuracy** (`test_acc`) — selected on validation accuracy, reported on
  test accuracy.
- **Spike activity** (`avg_input_spikes`, `avg_hidden_spikes`).
- **Energy proxy** (`avg_energy_cost` / `avg_synops`) — a fan-out-weighted
  synaptic-operations count, not a raw spike count (input spikes fan out to
  `hidden_dim` synapses; hidden spikes fan out to only `num_classes`).
- **Decision latency** (`avg_latency`) — mean number of timesteps until the
  network's prediction stabilizes.

## Repository Structure

```
config/phase_grid.py     # BASE_CONFIG + every EXPERIMENTS entry (source of truth for what was run)
encoders/                # rate.py, ttfs.py, phase.py — image -> spike train
readouts/                # rate.py, ttfs.py, phase.py — hidden spikes -> class scores
models/                  # lif.py (surrogate-gradient LIF), snn.py (encoder + LIF + readout)
datasets/                # cifar10.py, emnist.py — the one place each dataset is loaded/split
experiments/
  run_phase_grid.py      # main driver — `python -m experiments.run_phase_grid --exp <name>`
  plot_results.py        # turns results/*.csv into results/plots/*.png
  calibrate_threshold.py # one-off threshold-search helper (uses datasets/emnist.py)
  learnable_frequency_diagnostic.py  # learnable-cycle_steps proof-of-concept
  sanity_check.py        # smoke test for the pipeline
results/*.csv            # one row per (setting, seed) run; *_epoch.csv has per-epoch train/val curves
results/plots/*.png      # rendered figures (results/plots/cifar10/ for CIFAR-10-specific plots)
requirements.txt
report/Coding_Scheme_Benchmark_Report.pdf  # full technical report (see Technical Report section)
```

## Running Experiments

```bash
pip install -r requirements.txt
python -m experiments.run_phase_grid --exp baseline
python -m experiments.plot_results
```

Pick any experiment name from `EXPERIMENTS` in
[config/phase_grid.py](config/phase_grid.py) for `--exp`. `--coding
rate|ttfs|phase` reruns just one scheme and merges it back into the
existing results CSV instead of overwriting the whole file. `--epochs N`
overrides the epoch budget for a quick smoke run.

## Experiments

1. **Baseline coding comparison** — Rate, TTFS, and Phase on EMNIST, each
   with its own matched readout and tuned threshold/resolution, 3 seeds.
2. **Phase frequency sweep** — `cycle_steps` varied at fixed `num_bins`.
3. **Phase resolution sweep** — `num_bins` varied at fixed `cycle_steps`.
4. **Joint frequency × resolution grid** — every valid `(num_bins,
   cycle_steps)` pair; reveals a strong interaction effect at `(4, 8)` not
   visible from either single-axis sweep.
5. **Phase neighborhood / bug check** — densely samples around the `(4,
   8)` optimum to confirm it is a genuine local maximum, not an artifact.
6. **TTFS and Phase threshold sweeps** — selects each scheme's operating
   threshold on EMNIST and (separately) CIFAR-10.
7. **Rate-scale sweep** — accuracy/energy trade-off knob for Rate coding.
8. **`repeat_cycles` ablation** — Phase pixels firing once per encoding
   window vs. every cycle.
9. **CIFAR-10 experiments** — baseline comparison, threshold sweeps, and
   joint frequency × resolution grid repeated on CIFAR-10.
10. **Learnable oscillation frequency (proof-of-concept)** — tests whether
    `cycle_steps` can be learned via a differentiable relaxation instead of
    swept; confirms feasibility, not yet integrated into the main pipeline.

Full methodology and per-seed numeric results for all of the above are in
the Technical Report.

## Future Work

- Full integration of learnable oscillation frequency into the main
  training pipeline (currently a standalone proof-of-concept).
- Matched vs. mismatched encoding/readout comparison (open question — no
  experiment here yet pairs a mismatched encoding/readout combination).
- Extension to event-based datasets (e.g. DVS-Gesture).
- A stronger, convolutional SNN backbone, particularly for CIFAR-10.

## Technical Report

[report/Coding_Scheme_Benchmark_Report.pdf](report/Coding_Scheme_Benchmark_Report.pdf)
is the full technical report: literature review, detailed methodology,
per-seed results and interpretation, contributions, limitations, and the
complete future-work discussion. This README is a summary; the report is
the source of truth for methodology and results detail.
