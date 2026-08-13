"""Rendering and writing the shared report format."""
# Split out of the original mlep/experiments/windows.py. The sweep's CONFIGS and
# main() live in mlep.experiments.windows; everything reusable lives here, so
# consumers no longer need in-function imports to dodge a circular dependency.

import os
import time

def build_report(results, args, device, gpu_name, use_amp, total):
    """Render the results collected so far as the report text.

    Called after EVERY config, not just at the end, so --out always reflects all
    completed work. A long sweep that is interrupted -- the shell it was launched
    from going away, the kernel OOM killer, a Ctrl-C -- then still leaves usable
    results on disk instead of losing hours of GPU time. `total` is the number of
    configs the run intends to do, so the report can say whether it is finished."""
    scen_names = list(EVAL_SCENARIOS.keys())

    def clean_ap(r):
        v = r['scenarios']['clean']['ap']
        return v if v == v else -1        # NaN -> sort last

    # sorted(), not .sort(): the caller's list stays in run order.
    results = sorted(results, key=clean_ap, reverse=True)
    done = len(results)

    lines = []
    lines.append("MLEP window / robustness sweep")
    lines.append(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if done < total:
        lines.append(f"STATUS: IN PROGRESS -- {done}/{total} configs done. This file "
                     f"is rewritten after each one, so it is safe to read now.")
    else:
        lines.append(f"STATUS: complete -- {done}/{total} configs.")
    lines.append(f"device={device}{f' ({gpu_name})' if gpu_name else ''}  "
                 f"arch={args.arch}  amp={use_amp}  workers={args.num_threads}  "
                 f"steps={args.max_train_steps}  "
                 f"crop={args.cropSize}  batch={args.batch_size}  lr={args.lr}  "
                 f"seed={args.seed}  "
                 f"classes={args.classes or 'ALL'}  val_batches={args.max_val_batches}  "
                 f"bn_recal={args.bn_recal_batches or 'off'}")
    lines.append("")

    # Per-model detail: config kwargs + acc/ap/n for every test scenario.
    lines.append("Per-model results")
    lines.append("-" * 60)
    for r in results:
        lines.append(f"[{r['name']}]  {CONFIGS[r['name']]}")
        lines.append(f"    in_channels={r['in_ch']}  params={r['params']/1e6:.2f}M  "
                     f"steps={r.get('steps', '?')}  train_time={r['time']:.0f}s")
        for s in scen_names:
            sc = r['scenarios'][s]
            # acc_best / logit_mean go AFTER (n=...): aggregate_results.py's SCEN_RE
            # matches a prefix of the line, so appending fields keeps old and new
            # result files parsing identically.
            lines.append(f"    test/{s:10s}  acc={sc['acc']:.4f}  "
                         f"ap={sc['ap']:.4f}  (n={sc['n']})  "
                         f"acc_best={sc['acc_best']:.4f}  "
                         f"logit_mean={sc['logit_mean']:+.2f}")
        lines.append("")

    # Compact AP table (rows = configs, cols = scenarios).
    name_w = max(len(r['name']) for r in results) + 1
    width = name_w + 11 * len(scen_names) + 10
    lines.append("Average Precision (AP) per test scenario  [higher = more robust]")
    lines.append("=" * width)
    lines.append(f"{'config':{name_w}s}" + "".join(f"{s:>11s}" for s in scen_names)
                 + f"{'time(s)':>10s}")
    lines.append("-" * width)
    for r in results:
        row = f"{r['name']:{name_w}s}"
        for s in scen_names:
            row += f"{r['scenarios'][s]['ap']:11.4f}"
        row += f"{r['time']:10.0f}"
        lines.append(row)
    lines.append("=" * width)
    lines.append("Cols: clean vs corrupted test sets. Compare a config's clean AP "
                 "with its blur / jpeg / webp / blur+jpeg AP to read off robustness.")
    lines.append("acc is at the fixed 0.5 threshold; acc_best is the best any "
                 "threshold could do. acc << acc_best means the scores separate the "
                 "classes but the decision boundary is misplaced -- check bn_recal.")
    lines.append("Note: short training -> ranking is indicative. Re-run the top "
                 "configs with more --max_train_steps to confirm.")
    return "\n".join(lines)



def write_report(results, args, device, gpu_name, use_amp, total, builder=build_report):
    """Atomically replace --out with the current report. Written via a temp file
    and os.replace so a kill mid-write cannot leave a truncated results file.

    `builder` is the function that renders the text; it defaults to this file's
    build_report (the window sweep), and the entropy / degradation experiments
    pass their own so every experiment writes its results the same way."""
    report = builder(results, args, device, gpu_name, use_amp, total)
    tmp = args.out + '.tmp'
    with open(tmp, 'w') as f:
        f.write(report + "\n")
    os.replace(tmp, args.out)
    return report


