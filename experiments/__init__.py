"""Additional experiment groups for experiment_windows.py.

Each module here is one experiment group, selected with
`python experiment_windows.py --experiment <name>`:

    entropy        experiments/entropy.py      -- alternative entropy definitions
    mlep_degradation  experiments/degradation.py -- blur / JPEG degradation heads

They import the harness (data loading, training loop, BN recalibration,
evaluation, atomic report writing) from experiment_windows rather than
re-implementing it, and write the same kind of results/<mode>_<stamp>.txt file.
"""
