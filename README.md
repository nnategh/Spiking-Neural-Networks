# SNN Coding-Scheme Research

This repository contains two related projects studying neural coding
schemes (Rate, Time-to-First-Spike, Phase, and Burst) in spiking neural
networks.

## [`coding-benchmark`](coding-benchmark)

The primary, current benchmark. Compares Rate, TTFS, and Phase coding on
EMNIST and CIFAR-10, using a matched encoding/readout methodology with
scheme-specific tuning under a consistent evaluation protocol. This is
the primary reference for current Rate/TTFS/Phase results. Full
methodology and results are in its technical report:
[`coding-benchmark/report/Coding_Scheme_Benchmark_Report.pdf`](coding-benchmark/report/Coding_Scheme_Benchmark_Report.pdf).

## [`dvs-and-burst-exploration`](dvs-and-burst-exploration)

Earlier senior-project work, preserved because it contains **Burst
coding** and event-based **DVS-Gesture** experiments not yet covered by
`coding-benchmark/`. It predates that project's methodology and is
exploratory/feasibility work rather than a finalized benchmark.

## Relationship between the two projects

Results from the two projects should not be compared numerically: they
use different evaluation methodologies. `coding-benchmark/` is the
reference for Rate, TTFS, and Phase coding; `dvs-and-burst-exploration/`
is the reference for Burst coding and DVS-Gesture, pending integration of
that work into `coding-benchmark/`'s methodology.
