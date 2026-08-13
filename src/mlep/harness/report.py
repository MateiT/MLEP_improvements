"""Atomic report writing, shared by every experiment.

The renderer itself is NOT here: each experiment's build_*_report indexes its own
config registry, so putting one of them in the shared layer would make harness/
depend on experiments/ -- exactly the cycle this split removed. Callers pass
their builder in."""
# Split out of the original mlep/experiments/windows.py. The sweep's CONFIGS and
# main() live in mlep.experiments.windows; everything reusable lives here, so
# consumers no longer need in-function imports to dodge a circular dependency.

import os

def write_report(results, args, device, gpu_name, use_amp, total, builder):
    """Atomically replace --out with the current report. Written via a temp file
    and os.replace so a kill mid-write cannot leave a truncated results file.

    `builder` renders the text: mlep.experiments.windows.build_report for the
    sweep, build_entropy_report / build_degradation_report for the other two, so
    every experiment writes its results the same way."""
    report = builder(results, args, device, gpu_name, use_amp, total)
    tmp = args.out + '.tmp'
    with open(tmp, 'w') as f:
        f.write(report + "\n")
    os.replace(tmp, args.out)
    return report


