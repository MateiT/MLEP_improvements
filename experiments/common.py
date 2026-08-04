"""Shared helpers for the entropy and degradation experiments.

Nothing here changes how the window sweep reports: the header / STATUS /
per-model / ASCII-table layout below is the same one build_report() uses, and
the files are written through experiment_windows.write_report(), so
scripts/aggregate_results.py keeps parsing them.

The CSVs the two experiments produce are flat siblings of --out with the same
stamp (results/entropy_<stamp>_metrics.csv next to results/entropy_<stamp>.txt),
so no new directory layout is introduced.
"""
import os
import time

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def equal_error_rate(y_true, prob):
    """Rate at which FPR == FNR, read off the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, prob)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def expected_calibration_error(y_true, prob, bins=10):
    """Standard 10-bin ECE: mean |accuracy - confidence| weighted by bin size."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        # conf = mean predicted p(positive) in the bin, acc = observed fraction
        # of positives in it. A calibrated model has the two equal everywhere.
        conf = prob[m].mean()
        acc = y_true[m].mean()
        ece += m.mean() * abs(acc - conf)
    return float(ece)


def binary_metrics(y_true, prob):
    """Every binary metric the two experiments report, from labels + p(positive).

    `prob` must be a probability (threshold 0.5); pass sigmoid(logit) for a
    network head. Ranking metrics are threshold-free so the sigmoid does not
    move them."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    both = len(np.unique(y_true)) > 1
    pred = prob > 0.5
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_true, pred, average='binary', zero_division=0)
    return dict(
        n=int(len(y_true)),
        pos=int(y_true.sum()),
        roc_auc=float(roc_auc_score(y_true, prob)) if both else float('nan'),
        pr_auc=float(average_precision_score(y_true, prob)) if both else float('nan'),
        acc=float(accuracy_score(y_true, pred)),
        precision=float(pr), recall=float(rc), f1=float(f1),
        eer=equal_error_rate(y_true, prob) if both else float('nan'),
        brier=float(np.mean((prob - y_true) ** 2)),
        ece=expected_calibration_error(y_true, prob) if both else float('nan'),
    )


def bootstrap_ci(y_true, prob, key='roc_auc', n_boot=200, seed=0, alpha=0.05):
    """Percentile bootstrap CI for one key of binary_metrics."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    if len(np.unique(y_true)) < 2 or len(y_true) < 10:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(binary_metrics(y_true[idx], prob[idx])[key])
    if not vals:
        return float('nan'), float('nan')
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def confusion(y_true, y_pred, n_classes):
    """n x n confusion counts (rows = true, cols = predicted)."""
    m = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        m[t, p] += 1
    return m


def spearman(a, b):
    """Spearman rank correlation without pulling in scipy.stats."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float('nan')
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float('nan')


def _rank(x):
    """Average ranks, ties shared (what Spearman needs)."""
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average the ranks of equal values
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return ranks


# --------------------------------------------------------------------------- #
# text / file helpers -- same conventions as build_report / write_report
# --------------------------------------------------------------------------- #
def report_header(title, args, device, gpu_name, use_amp, done, total, extra=''):
    """The header block every result file in this project starts with."""
    lines = [title, f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
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
                 f"bn_recal={args.bn_recal_batches or 'off'}"
                 + (f"  {extra}" if extra else ""))
    lines.append("")
    return lines


def table(lines, header, rows, note=None):
    """Append an ASCII table in the same style as the sweep's AP table."""
    widths = [max(len(str(header[c])), *(len(str(r[c])) for r in rows)) + 2
              if rows else len(str(header[c])) + 2 for c in range(len(header))]
    widths[0] = max(widths[0], 10)
    width = sum(widths)
    lines.append("=" * width)
    lines.append("".join(f"{h:>{w}s}" for h, w in zip(header, widths)))
    lines.append("-" * width)
    for r in rows:
        lines.append("".join(f"{str(v):>{w}s}" for v, w in zip(r, widths)))
    lines.append("=" * width)
    if note:
        lines.append(note)
    return lines


def confusion_text(lines, mat, labels, indent='    '):
    """Confusion matrix as text, inside the result file (no new artefacts)."""
    w = max(9, max(len(str(l)) for l in labels) + 1)
    corner = 'true\\pred'
    lines.append(indent + f"{corner:>{w}s}" + "".join(f"{str(l):>{w}s}" for l in labels))
    for i, l in enumerate(labels):
        lines.append(indent + f"{str(l):>{w}s}"
                     + "".join(f"{v:>{w}d}" for v in mat[i]))
    return lines


def check_split_disjoint(dataroot, train_split, val_split, classes=None,
                         sample_per_dir=2000):
    """Lightweight leakage check between two split folders.

    The pipeline splits by SOURCE IMAGE (the train/ and val/ folders) and only
    then crops / resizes / corrupts, so leakage would have to come from the same
    source file existing in both folders. Compare (basename, size) pairs, which
    catches a dataset accidentally staged into both, and sample at most
    `sample_per_dir` files per directory so the check costs a listing rather
    than a walk of 288k files. Returns the report lines (also printed)."""
    overlap, checked = 0, 0
    troot = os.path.join(dataroot, train_split)
    vroot = os.path.join(dataroot, val_split)
    if not (os.path.isdir(troot) and os.path.isdir(vroot)):
        return [f"leakage check: skipped ({train_split}/ or {val_split}/ missing)"]
    names = classes or sorted(d for d in os.listdir(troot)
                              if os.path.isdir(os.path.join(troot, d)))
    for cls in names:
        for lab in ('0_real', '1_fake'):
            td = os.path.join(troot, cls, lab)
            vd = os.path.join(vroot, cls, lab)
            if not (os.path.isdir(td) and os.path.isdir(vd)):
                continue
            tf = sorted(os.listdir(td))[:sample_per_dir]
            vf = set(sorted(os.listdir(vd))[:sample_per_dir])
            sizes = {}
            for f in tf:
                if f in vf:
                    sizes[f] = os.path.getsize(os.path.join(td, f))
            for f, sz in sizes.items():
                if os.path.getsize(os.path.join(vd, f)) == sz:
                    overlap += 1
            checked += len(tf)
    line = (f"leakage check: {overlap} identical (name, size) source file(s) shared "
            f"between {train_split}/ and {val_split}/ over {checked} sampled files"
            + ("  <-- INVESTIGATE" if overlap else "  -- ok"))
    return [line]


def csv_path(out, suffix):
    """results/entropy_<stamp>.txt + 'metrics' -> results/entropy_<stamp>_metrics.csv"""
    stem = out[:-4] if out.endswith('.txt') else out
    return f"{stem}_{suffix}.csv"


def write_csv(path, header, rows):
    """Atomic CSV write, same tmp + os.replace discipline as write_report."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(','.join(str(h) for h in header) + '\n')
        for r in rows:
            f.write(','.join(_cell(v) for v in r) + '\n')
    os.replace(tmp, path)
    return path


def _cell(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    s = str(v)
    return f'"{s}"' if (',' in s or '"' in s) else s


def fmt(v, nd=4):
    """nan-safe fixed-point, so tables line up even when a metric is undefined."""
    try:
        if v != v:
            return 'nan'
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return str(v)
