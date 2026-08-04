"""Aggregate several experiment_windows.py result files into mean +/- std tables.

A single short run is noise-dominated: at 200 steps the smoke test produced a
0.95 vs 0.73 clean-AP gap between configs that differ only in training-time
augmentation, which is not a real effect. Running the sweep over several seeds
and reading the spread here tells you which gaps survive the noise.

Usage:
    python scripts/aggregate_results.py results/sweep_20260725_101112_seed*.txt
    python scripts/aggregate_results.py results/*.txt --metric acc
"""
import argparse
import math
import re
import sys
from collections import defaultdict

CONFIG_RE = re.compile(r"^\[([^\]]+)\]")
SCEN_RE = re.compile(
    r"^\s+test/(\S+)\s+acc=([\d.]+|nan)\s+ap=([\d.]+|nan)"
)
TIME_RE = re.compile(r"train_time=(\d+)s")


def parse(path):
    """-> {(config, scenario): value}, {config: train_time}, in_progress. Later
    duplicate configs in one file overwrite earlier ones, matching the report
    layout. experiment_windows.py rewrites its --out after every config, so a
    file from a sweep that is still running (or was interrupted) parses fine --
    it just holds fewer configs. `in_progress` flags that case."""
    scores, times, current = {}, {}, None
    in_progress = False
    with open(path) as fh:
        for line in fh:
            if line.startswith('STATUS:'):
                in_progress = 'IN PROGRESS' in line
                continue
            m = CONFIG_RE.match(line)
            if m:
                current = m.group(1)
                continue
            if current is None:
                continue
            m = TIME_RE.search(line)
            if m:
                times[current] = int(m.group(1))
            m = SCEN_RE.match(line)
            if m:
                scen, acc, ap = m.group(1), m.group(2), m.group(3)
                scores[(current, scen, "acc")] = float(acc)
                scores[(current, scen, "ap")] = float(ap)
    return scores, times, in_progress


def mean_std(vals):
    vals = [v for v in vals if not math.isnan(v)]
    if not vals:
        return float("nan"), float("nan"), 0
    mu = sum(vals) / len(vals)
    if len(vals) < 2:
        return mu, 0.0, len(vals)
    var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
    return mu, math.sqrt(var), len(vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="result .txt files (one per seed)")
    p.add_argument("--metric", default="ap", choices=["ap", "acc"])
    args = p.parse_args()

    per_run = [parse(f) for f in args.files]
    if not per_run:
        sys.exit("no files parsed")
    partial = [f for f, (_, _, ip) in zip(args.files, per_run) if ip]

    collected = defaultdict(list)
    times = defaultdict(list)
    configs, scenarios = [], []
    for scores, tmap, _ in per_run:
        for (cfg, scen, metric), val in scores.items():
            if metric != args.metric:
                continue
            collected[(cfg, scen)].append(val)
            if cfg not in configs:
                configs.append(cfg)
            if scen not in scenarios:
                scenarios.append(scen)
        for cfg, t in tmap.items():
            times[cfg].append(t)

    if not configs:
        sys.exit("no results found -- are these experiment_windows.py output files?")

    n_runs = len(args.files)
    # Rank by clean score when present, else by the first scenario.
    key_scen = "clean" if "clean" in scenarios else scenarios[0]
    configs.sort(key=lambda c: mean_std(collected.get((c, key_scen), []))[0],
                 reverse=True)

    name_w = max(len(c) for c in configs) + 1
    col_w = 16
    width = name_w + col_w * len(scenarios) + 9
    print(f"{args.metric.upper()} across {n_runs} run(s), mean +/- std  "
          f"[sorted by {key_scen}]")
    print("=" * width)
    print(f"{'config':{name_w}s}" + "".join(f"{s:>{col_w}s}" for s in scenarios)
          + f"{'time(s)':>9s}")
    print("-" * width)
    for cfg in configs:
        row = f"{cfg:{name_w}s}"
        for scen in scenarios:
            mu, sd, n = mean_std(collected.get((cfg, scen), []))
            cell = "n/a" if n == 0 else f"{mu:.3f}+/-{sd:.3f}"
            row += f"{cell:>{col_w}s}"
        t = times.get(cfg, [])
        row += f"{sum(t)/len(t):9.0f}" if t else f"{'-':>9s}"
        print(row)
    print("=" * width)

    missing = [c for c in configs
               if any(len(collected.get((c, s), [])) != n_runs for s in scenarios)]
    if missing:
        print(f"note: incomplete across runs (fewer than {n_runs} values): "
              + ", ".join(missing))
    if n_runs < 2:
        print("note: single run -- std is 0 by construction and means nothing. "
              "Pass several seed files to get a real spread.")
    if partial:
        print(f"note: {len(partial)} file(s) still in progress, so their configs "
              "are only partly represented: " + ", ".join(partial))


if __name__ == "__main__":
    main()
