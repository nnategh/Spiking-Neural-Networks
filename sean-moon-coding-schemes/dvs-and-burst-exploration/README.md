# DVS-Gesture and Burst-Coding Exploration

An earlier senior-project exploration of neural coding schemes (Rate,
TTFS, Phase, and Burst) for spiking neural networks, preserved separately
from the newer [`../coding-benchmark/`](../coding-benchmark/) project.

## Why this is preserved

This project's Rate/TTFS/Phase results on EMNIST are superseded by
`../coding-benchmark/`'s more rigorous version of the same comparison
(matched encoding/readout, a fan-out-weighted SynOps energy metric) and
are not reproduced here. What this project has that `../coding-benchmark/`
does not is **Burst coding** and an **event-based DVS-Gesture dataset**
pipeline, neither of which has yet been reproduced under the newer
methodology. It is kept here as the reference implementation for both,
pending that work.

## What's included

- **Burst coding on EMNIST** — the burst encoder, its parameter-trend
  sweep, and results/figures. Shares its EMNIST model/training code with
  the rate/ttfs/phase encoders (the model class supports all four coding
  schemes; it isn't split by scheme).
- **DVS-Gesture (event-based data)** — the full pipeline: dataset loading,
  hyperparameter tuning, training, a parameter-trend sweep, sensitivity
  analysis, and plotting, covering all four coding schemes on this
  dataset.

Not included (present in the original project if needed): EMNIST
rate/ttfs/phase trend sweeps and cross-scheme sensitivity analysis
(superseded by `../coding-benchmark/`, see above), and a
robustness-under-noise (Gaussian / salt-and-pepper) comparison, which is
not superseded by anything but was out of scope for this snapshot.

## Project Structure

```
encoding/       # rate.py, ttfs.py, phase.py, burst.py -- image -> spike train
models/         # lif.py; snn_emnist.py (EMNIST model, all 4 coding schemes); snn_dvs.py (DVS-Gesture model)
readout/        # aggregation.py -- hidden spikes -> class scores
train/          # train/evaluate loops, EMNIST and DVS-Gesture variants
datasets/       # EMNIST_loader.py; DVS_GC_loader.py (loads the DVS-Gesture
                #   dataset via tonic.datasets.DVSGesture -- filename predates
                #   this README's terminology, kept as-is)
experiments/
  run_burst_trend.py, plot_burst_lines.py,
  plot_burst_heatmap.py                    # EMNIST burst-coding parameter sweep
  run_dvs_tuning.py, select_best_dvs_hyperparam.py,
  run_dvs.py, run_dvs_trend.py,
  compute_dvs_sensitivity.py, plot_dvs_*.py  # DVS-Gesture pipeline
results/
  emnist_trend_results/burst_trend_results.csv  # burst only (rate/ttfs/phase superseded, see above)
  dvs_results/*.csv                             # DVS-Gesture results, all four schemes
figures/
  emnist/burst/                # burst parameter-sweep figures
  dvs/, dvs_trends/             # DVS-Gesture result and trend figures
requirements.txt
```

## Methodological Note

This project predates `../coding-benchmark/`'s fairness conventions:
matched encoding/readout per scheme, each scheme's hyperparameters tuned
separately but architecture/training/time-budget held identical, and a
SynOps energy proxy rather than raw spike counts. Its results are **not
directly comparable, numerically, to `../coding-benchmark/`'s**. It should
be read as an earlier feasibility/exploratory study, and as the starting
point for porting Burst coding and the DVS-Gesture pipeline into
`../coding-benchmark/`'s methodology (see the Technical Report's Future
Work section).

## Running

```bash
pip install -r requirements.txt
```

```bash
# EMNIST, burst-coding parameter sweep
python3 -m experiments.run_burst_trend
python3 -m experiments.plot_burst_lines
python3 -m experiments.plot_burst_heatmap

# DVS-Gesture: tune -> select best -> final run -> trend sweep -> sensitivity
python3 -m experiments.run_dvs_tuning --coding rate   # repeat per scheme: ttfs, phase, burst
python3 -m experiments.select_best_dvs_hyperparam
python3 -m experiments.run_dvs --coding rate          # repeat per scheme
python3 -m experiments.run_dvs_trend --coding rate --T 20 --lr 1e-3   # repeat per scheme
python3 -m experiments.compute_dvs_sensitivity
python3 -m experiments.plot_dvs_results
python3 -m experiments.plot_dvs_trends
python3 -m experiments.plot_dvs_sensitivity
```

All commands run from this folder's root. `run_dvs_tuning.py` and
`run_dvs_trend.py` require `--coding` (and `run_dvs_trend.py` also
requires `--T` and `--lr`); the other scripts take no required arguments.
`run_dvs_tuning.py` writes its tuning and epoch-log CSVs to
`results/dvs_results/`, and `select_best_dvs_hyperparam.py` reads from
and writes to that same directory, consistent with the DVS results
already committed there.

Every import listed above was verified in a fresh virtual environment
against `requirements.txt`. Full end-to-end training (which needs the
actual EMNIST and DVS-Gesture data on disk) was not re-run as part of this
snapshot.

## Technical Report

Full methodology, results, and the future-work discussion referenced
above are in
[`../coding-benchmark/report/Coding_Scheme_Benchmark_Report.pdf`](../coding-benchmark/report/Coding_Scheme_Benchmark_Report.pdf).
That report covers the `coding-benchmark/` project directly; this
project's own results (Burst coding, DVS-Gesture) are not part of it yet,
per the Methodological Note above.
