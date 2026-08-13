"""Stress the four most interesting generators with increasing blur / JPEG / WebP.

    python python -m mlep.evaluation.perturbations                     # picks from the newest clean run
    python python -m mlep.evaluation.perturbations --clean_csv <path>   # or an explicit one

Reads the clean run's per-image predictions (python -m mlep.evaluation.trained_model),
ranks generators by AP, and takes the best AND worst GAN plus the best AND worst
diffusion generator -- four in total. Those four are then re-scored under every
condition in trained_eval_common.PERTURBATIONS:

    blur    sigma 0.5 -> 1.5 -> 3.0     (jpeg off)
    jpeg    quality 75 -> 45 -> 15      (blur off)
    webp    quality 80 -> 50 -> 20      (blur off)  <- Google's format

plus a clean row recomputed under the SAME image cap, so every delta is
apples-to-apples rather than being read against the full-corpus clean number.

Writes results/trained_model/perturbation_sweep_<stamp>.txt and _metrics.csv.
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np


from mlep.experiments import common as C                             # noqa: E402
from mlep.harness.device import get_device, setup_cuda_perf
from mlep.evaluation import common as T                         # noqa: E402
from mlep.evaluation.trained_model import header, write_text


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # 10 conditions per generator, so the default cap is tighter than the clean
    # pass's -- override with 0 to use every image.
    T.add_common_args(ap, default_max_per_label=500)
    ap.add_argument('--clean_csv', default='',
                    help='clean predictions CSV; default = newest in results/trained_model')
    ap.add_argument('--select_by', default='ap', choices=['ap', 'acc'],
                    help='metric used to rank best/worst (default ap)')
    ap.add_argument('--tag', default='perturbation')
    return ap.parse_args()


def newest_clean_csv():
    hits = sorted(glob.glob(os.path.join(T.RESULTS_DIR, '*_predictions_*.csv')),
                  key=os.path.getmtime, reverse=True)
    if not hits:
        raise SystemExit("No *_predictions_*.csv in results/trained_model -- run "
                         "python -m mlep.evaluation.trained_model first, or pass --clean_csv.")
    return hits[0]


def clean_ranking(path, ai_col='p_ai'):
    """-> {(set, generator): {'n','acc','ap'}} pooled from the clean predictions.

    Pooling the per-image probabilities is the only correct way to get a
    generator's AP: average precision is not an average of its parts, so the
    per-category rows cannot be combined arithmetically.
    """
    by_gen = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if ai_col not in row:
                raise SystemExit(f"{path} has no {ai_col} column "
                                 f"(columns: {list(row)})")
            key = (row['set'], row['generator'])
            y, p = by_gen.setdefault(key, ([], []))
            y.append(int(row['label']))
            p.append(float(row[ai_col]))
    out = {}
    for key, (y, p) in by_gen.items():
        m = C.binary_metrics(np.array(y), np.array(p))
        out[key] = dict(n=m['n'], acc=m['acc'], ap=m['pr_auc'])
    return out


def pick_four(ranking, select_by, warn=print):
    """best + worst GAN, best + worst diffusion. -> [(set, gen, role, metric)]

    A family with a single generator yields one entry, not a duplicated pair; a
    family with none is reported and skipped rather than failing the run, since
    GAN-set-2 / Diffusion-set are not guaranteed to be present.
    """
    picks = []
    for kind, sets in (('gan', T.GAN_SETS), ('diffusion', T.DIFFUSION_SETS)):
        fam = {k: v for k, v in ranking.items() if k[0] in sets}
        if not fam:
            warn(f"[warn] no {kind} generators in the clean run -- skipping that "
                 f"half of the selection")
            continue
        # secondary key = acc, so a tie on the primary metric is broken
        # deterministically rather than by dict order
        order = sorted(fam.items(),
                       key=lambda kv: (kv[1][select_by], kv[1]['acc'], kv[0]))
        worst, best = order[0], order[-1]
        picks.append((*worst[0], f'worst_{kind}', worst[1][select_by]))
        if best[0] != worst[0]:
            picks.append((*best[0], f'best_{kind}', best[1][select_by]))
        else:
            warn(f"[warn] only one {kind} generator -- it is both best and worst")
    return picks


def sweep_report(res, meta, args, device, gpu, use_amp, done, total, picks, sel):
    lines = header(meta, args, device, gpu, use_amp, done, total,
                   "Trained-model perturbation sweep -- blur / JPEG / WebP",
                   unit='generator x condition runs')
    lines += [
        f"Selection: best and worst GAN + best and worst diffusion generator by "
        f"{args.select_by.upper()} on the clean run",
        f"  source: {os.path.relpath(args.clean_csv, T.ROOT)}",
    ]
    for s, g, role, v in picks:
        lines.append(f"  {role:16s} {f'{s}/{g}':40s} clean "
                     f"{args.select_by}={100 * v:.1f}%")
    lines += [
        "",
        "Each condition is applied with prob=1.0, i.e. to EVERY image, so the rows "
        "are deterministic. Blur runs before JPEG, the order data_augment uses.",
        "Blur sigmas and JPEG qualities are the checkpoint's own training levels, "
        "so the blur/jpeg head columns are interpretable against what it was",
        "taught. WebP was NOT in the training grid -- those three rows are an "
        "out-of-distribution probe, and the jpeg head's response to them is the",
        "interesting part. d_acc / d_ap are percentage-point changes from this "
        "generator's clean row at the same image cap.",
        "",
    ]

    diag = [h for h in meta['heads'] if h != 'ai']
    hrow = ['generator', 'condition', 'n', 'acc', 'ap', 'd_acc', 'd_ap'] \
        + [f'{h}_mean' for h in diag]
    for s, g, role, _ in picks:
        rows = [r for r in res if r['set'] == s and r['generator'] == g]
        if not rows:
            continue
        base = next((r for r in rows if r['condition'] == 'clean'), None)
        body = []
        for r in rows:
            d_acc = (100 * (r['acc'] - base['acc'])) if base else float('nan')
            d_ap = (100 * (r['ap'] - base['ap'])) if base else float('nan')
            body.append([r['condition'], str(r['n']),
                         C.fmt(100 * r['acc'], 1), C.fmt(100 * r['ap'], 1),
                         C.fmt(d_acc, 1) if r['condition'] != 'clean' else '-',
                         C.fmt(d_ap, 1) if r['condition'] != 'clean' else '-']
                        + [C.fmt(r['heads'][h]['mean'], 3) for h in diag])
        lines.append(f"{role}: {s}/{g}")
        C.table(lines, hrow[1:], body)
        lines.append("")

    # Does the model actually degrade monotonically as each corruption worsens?
    lines.append("Monotonicity of AP within each ladder "
                 "(expected: non-increasing as severity rises)")
    for s, g, role, _ in picks:
        for grp, conds in T.PERTURBATION_GROUPS.items():
            aps = [next((r['ap'] for r in res
                         if r['set'] == s and r['generator'] == g
                         and r['condition'] == c), None) for c in conds]
            if any(a is None for a in aps):
                continue
            ok = all(aps[i] >= aps[i + 1] - 1e-9 for i in range(len(aps) - 1))
            arrow = ' -> '.join(f"{100 * a:.1f}" for a in aps)
            lines.append(f"  {g:24s} {grp:5s}: {arrow}   "
                         f"{'monotone' if ok else 'NOT monotone'}")
    lines.append("")
    lines.append("A non-monotone ladder is a real result, not noise to smooth over: "
                 "it means a heavier corruption helped, which is worth explaining.")

    # Mean over the four generators, per condition -- the compact summary.
    lines.append("")
    lines.append("Mean over the selected generators, per condition")
    srows = []
    for cond in T.PERTURBATIONS:
        sub = [r for r in res if r['condition'] == cond]
        if not sub:
            continue
        srows.append([cond, str(len(sub)),
                      C.fmt(100 * float(np.mean([r['acc'] for r in sub])), 1),
                      C.fmt(100 * float(np.mean([r['ap'] for r in sub])), 1)]
                     + [C.fmt(float(np.mean([r['heads'][h]['mean'] for r in sub])), 3)
                        for h in diag])
    C.table(lines, ['condition', 'generators', 'mean_acc', 'mean_ap']
            + [f'{h}_mean' for h in diag], srows)
    return lines


def main():
    args = parse_args()
    args.clean_csv = args.clean_csv or newest_clean_csv()
    T.seed_everything(args.seed)
    device = get_device(args.device)
    gpu = setup_cuda_perf(device)
    use_amp = (device.type == 'cuda') and not args.no_amp

    model, meta = T.load_trained_model(args.model_path, device)
    ranking = clean_ranking(args.clean_csv)
    picks = pick_four(ranking, args.select_by)
    if not picks:
        raise SystemExit("Nothing selected -- the clean CSV has no GAN or "
                         "diffusion generators.")

    wanted = {(s, g) for s, g, _, _ in picks}
    gens = {(g.set_name, g.name): g for g in T.discover_generators(
        args.dataroot, only_sets={s for s, _ in wanted},
        only_generators={g for _, g in wanted})}
    missing = wanted - set(gens)
    if missing:
        raise SystemExit(f"selected but not found on disk: {sorted(missing)}")

    outdir = T.ensure_results_dir()
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_txt = os.path.join(outdir, f'{args.tag}_sweep_{stamp}.txt')

    print(f"checkpoint : {args.model_path}")
    print(f"clean run  : {os.path.relpath(args.clean_csv, T.ROOT)}")
    print(f"selected   : " + ', '.join(f"{r}={s}/{g}" for s, g, r, _ in picks))
    print(f"conditions : {list(T.PERTURBATIONS)}")
    print(f"cap        : {args.max_per_label or 'ALL'} images per label dir\n")

    total = len(picks) * len(T.PERTURBATIONS)
    res, csv_rows, done, t0 = [], [], 0, time.time()
    for s, g, role, _ in picks:
        gen = gens[(s, g)]
        for cond, aug in T.PERTURBATIONS.items():
            opt = T.EvalOpt(aug=aug, loadSize=meta['loadSize'],
                            cropSize=meta['cropSize'], no_crop=args.no_crop)
            t1 = time.time()
            r = T.score_generator(model, gen, opt, device, args, use_amp=use_amp)
            done += 1
            if r is None:
                print(f"({done}/{total}) {gen.key} {cond}: no images")
                continue
            m = T.ai_metrics(r['y'], r['p'], meta['ai_index'])
            hs = T.head_stats(r['y'], r['p'], meta['heads'])
            res.append(dict(set=s, generator=g, role=role, condition=cond,
                            heads=hs, **m))
            csv_rows.append([s, g, role, cond, m['n'], m['acc'], m['ap'],
                             m['roc_auc'], m['real_acc'], m['fake_acc']]
                            + [v for h in meta['heads']
                               for v in (hs[h]['mean'], hs[h]['rate'])])
            el = time.time() - t1
            print(f"({done}/{total}) {gen.key:30s} {cond:10s} n={m['n']:5d}  "
                  f"acc={100 * m['acc']:5.1f}  ap={100 * m['ap']:5.1f}  "
                  f"[{el:.0f}s]  "
                  + '  '.join(f"{h}={hs[h]['mean']:.3f}" for h in meta['heads']))

            write_text(out_txt, sweep_report(res, meta, args, device, gpu, use_amp,
                                             done, total, picks, args.select_by))
            C.write_csv(C.csv_path(out_txt, 'metrics'),
                        ['set', 'generator', 'role', 'condition', 'n', 'acc', 'ap',
                         'roc_auc', 'real_acc', 'fake_acc']
                        + [f'{h}_{k}' for h in meta['heads']
                           for k in ('mean', 'rate')],
                        csv_rows)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    for p in (out_txt, C.csv_path(out_txt, 'metrics')):
        print(f"  {os.path.relpath(p, T.ROOT)}")


if __name__ == '__main__':
    main()
