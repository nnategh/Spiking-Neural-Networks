BASE_CONFIG = {
    "dataset": "emnist",
    "time_steps": 32,
    "batch_size": 128,
    "epochs": 35,
    "lr": 1e-3,
    "hidden_dim": 512,
    "seed": 0,
}

EXPERIMENTS = {
    "baseline": [
        {
            "experiment": "baseline",
            "coding": "rate",
            "readout": "rate",
            "num_bins": None,
            "cycle_steps": None,
            "seed": seed,
        }
        for seed in [0, 1, 2]
    ] + [
        {
            "experiment": "baseline",
            "coding": "ttfs",
            "readout": "ttfs",
            "num_bins": None,
            "cycle_steps": None,
            "threshold": 0.2,
            "seed": seed,
        }
        for seed in [0, 1, 2]
    ] + [
        {
            "experiment": "baseline",
            "coding": "phase",
            "readout": "phase",
            "num_bins": 4,
            "cycle_steps": 8,
            "threshold": 0.2,
            "repeat_cycles": False,
            "seed": seed,
        }
        for seed in [0, 1, 2]
    ],

    "frequency": [
        {
            "experiment": "frequency",
            "coding": "phase",
            "readout": "phase",
            "num_bins": 8,
            "cycle_steps": c,
            "threshold": 0.2,
            "repeat_cycles": False,
        }
        for c in [4, 6, 8, 12, 16]
    ],

    "resolution": [
        {
            "experiment": "resolution",
            "coding": "phase",
            "readout": "phase",
            "num_bins": b,
            "cycle_steps": 16,
            "threshold": 0.2,
            "repeat_cycles": False,
            "seed": seed,
        }
        for b in [4, 8, 12, 16]
        for seed in [0, 1, 2]
    ],

    "rate_scale": [
        {
            "experiment": "rate_scale",
            "coding": "rate",
            "readout": "rate",
            "num_bins": None,
            "cycle_steps": None,
            "rate_scale": rs,
            "seed": seed,
        }
        for rs in [0.3, 0.5, 0.7, 0.9, 1.0]
        for seed in [0, 1, 2]
    ],

    "threshold_ttfs": [
        {
            "experiment": "threshold_ttfs",
            "coding": "ttfs",
            "readout": "ttfs",
            "num_bins": None,
            "cycle_steps": None,
            "threshold": t,
        }
        for t in [0.05, 0.1, 0.15, 0.2, 0.3]
    ],

    "threshold_phase": [
        {
            "experiment": "threshold_phase",
            "coding": "phase",
            "readout": "phase",
            "num_bins": 8,
            "cycle_steps": 8,
            "threshold": t,
            "repeat_cycles": False,
        }
        for t in [0.1, 0.15, 0.2, 0.25, 0.3]
    ],

    "repeat_cycles": [
        {
            "experiment": "repeat_cycles",
            "coding": "phase",
            "readout": "phase",
            "num_bins": 8,
            "cycle_steps": 8,
            "threshold": 0.15,
            "repeat_cycles": rc,
            "seed": seed,
        }
        for rc in [False, True]
        for seed in [0, 1, 2]
    ],

    # Full (num_bins, cycle_steps) grid: all 4x5=20 combinations,
    # including cycle_steps < num_bins cells (phase bins collide onto
    # the same offset slot there -- expected resolution collapse, not a
    # bug, but requested so the full picture is visible rather than
    # leaving those cells blank).
    "frequency_resolution_grid": [
        {
            "experiment": "frequency_resolution_grid",
            "coding": "phase",
            "readout": "phase",
            "num_bins": nb,
            "cycle_steps": cs,
            "threshold": 0.2,
            "repeat_cycles": False,
            "seed": seed,
        }
        for nb in [4, 8, 12, 16]
        for cs in [4, 6, 8, 12, 16]
        for seed in [0, 1, 2]
    ],

    # Fine-grained neighborhood around the (num_bins=4, cycle_steps=8)
    # outlier from frequency_resolution_grid. Purpose: bug-check --
    # if there's a real dynamical effect, accuracy should form a smooth
    # local hill around (4,8); if (4,8) is an isolated spike with no
    # support from its neighbors, that points to a bug rather than a
    # genuine phase-resolution/frequency interaction.
    "phase_neighborhood": [
        {
            "experiment": "phase_neighborhood",
            "coding": "phase",
            "readout": "phase",
            "num_bins": nb,
            "cycle_steps": cs,
            "threshold": 0.2,
            "repeat_cycles": False,
            "seed": seed,
        }
        for nb in [3, 4, 5]
        for cs in [6, 7, 8, 9, 10]
        for seed in [0, 1, 2]
    ],

    # Same grid as `phase_neighborhood`, but with the coasting-tail
    # cosine weighting zeroed out (readout="phase_windowed"). If the
    # jagged (nb, cs) pattern smooths out here, the "clean divisor of T"
    # readout artifact was the cause; if it's still jagged, look
    # elsewhere (e.g. hidden-layer dynamics).
    "phase_neighborhood_windowed": [
        {
            "experiment": "phase_neighborhood_windowed",
            "coding": "phase",
            "readout": "phase_windowed",
            "num_bins": nb,
            "cycle_steps": cs,
            "threshold": 0.2,
            "repeat_cycles": False,
            "seed": seed,
        }
        for nb in [3, 4, 5]
        for cs in [6, 7, 8, 9, 10]
        for seed in [0, 1, 2]
    ],

    # CIFAR-10 baseline comparison, mirroring `baseline` (EMNIST).
    # Reuses the EMNIST-tuned threshold/num_bins/cycle_steps values as a
    # starting point -- CIFAR-10's pixel-intensity distribution differs
    # (3-channel natural images vs. 1-channel handwritten digits), so
    # these may need their own tuning pass once this first comparison is
    # in.
    "cifar10_baseline": [
        {
            "experiment": "cifar10_baseline",
            "coding": "rate",
            "readout": "rate",
            "num_bins": None,
            "cycle_steps": None,
            "seed": seed,
        }
        for seed in [0, 1, 2]
    ] + [
        {
            "experiment": "cifar10_baseline",
            "coding": "ttfs",
            "readout": "ttfs",
            "num_bins": None,
            "cycle_steps": None,
            "threshold": 0.3,  # CIFAR-10-tuned: accuracy plateau 0.0-0.3, cheapest in that range
            "seed": seed,
        }
        for seed in [0, 1, 2]
    ] + [
        {
            "experiment": "cifar10_baseline",
            "coding": "phase",
            "readout": "phase",
            "num_bins": 4,
            "cycle_steps": 4,  # CIFAR-10-tuned: joint grid winner (was 8 for EMNIST)
            "threshold": 0.0,  # CIFAR-10-tuned: best val_acc, no background to filter
            "repeat_cycles": False,
            "seed": seed,
        }
        for seed in [0, 1, 2]
    ],

    # CIFAR-10's pixel intensities are spread broadly across [0,1]
    # (no clear background-vs-stroke bimodality like EMNIST), so the
    # threshold sweep range is wider than EMNIST's (0.05-0.3).
    "cifar10_threshold_ttfs": [
        {
            "experiment": "cifar10_threshold_ttfs",
            "coding": "ttfs",
            "readout": "ttfs",
            "num_bins": None,
            "cycle_steps": None,
            "threshold": t,
            "seed": seed,
        }
        for t in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        for seed in [0, 1, 2]
    ],

    "cifar10_threshold_phase": [
        {
            "experiment": "cifar10_threshold_phase",
            "coding": "phase",
            "readout": "phase",
            "num_bins": 4,
            "cycle_steps": 8,
            "threshold": t,
            "repeat_cycles": False,
            "seed": seed,
        }
        for t in [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        for seed in [0, 1, 2]
    ],

    # Full (num_bins, cycle_steps) grid for CIFAR-10, mirroring the
    # EMNIST `frequency_resolution_grid` (all 4x5=20 combos, including
    # cycle_steps < num_bins collision cells). Uses threshold=0.0, the
    # CIFAR-10-tuned phase optimum -- not EMNIST's 0.2.
    "cifar10_frequency_resolution_grid": [
        {
            "experiment": "cifar10_frequency_resolution_grid",
            "coding": "phase",
            "readout": "phase",
            "num_bins": nb,
            "cycle_steps": cs,
            "threshold": 0.0,
            "repeat_cycles": False,
            "seed": seed,
        }
        for nb in [4, 8, 12, 16]
        for cs in [4, 6, 8, 12, 16]
        for seed in [0, 1, 2]
    ],
}