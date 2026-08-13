"""Additional experiment groups for mlep/experiments/windows.py.

Each module here is one experiment group, selected with
`python -m mlep.experiments.windows --experiment <name>`:

    entropy        mlep/experiments/entropy.py      -- alternative entropy definitions
    mlep_degradation  mlep/experiments/degradation.py -- blur / JPEG degradation heads

They import the harness (data loading, training loop, BN recalibration,
evaluation, atomic report writing) from experiment_windows rather than
re-implementing it, and write the same kind of results/<mode>_<stamp>.txt file.
"""
