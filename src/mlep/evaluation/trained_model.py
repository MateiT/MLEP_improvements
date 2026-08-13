"""Score a trained MLEP checkpoint on every test-set generator, on CLEAN images.

    python python -m mlep.evaluation.trained_model                      # all generators, all images
    python python -m mlep.evaluation.trained_model --max_per_label 50    # plumbing smoke test

Clean means blur_prob = jpg_prob = webp_prob = 0, so mlep.data.datasets.data_augment
short-circuits and returns the PIL image untouched. Nothing is blurred or
recompressed by us on this pass -- which is the point: the checkpoint has 'blur'
and 'jpeg' heads alongside 'ai', and on undegraded input those two should stay
quiet. The second report below is what turns that expectation into numbers.

Writes into results/trained_model/:
    clean_acc_ap_<stamp>.txt             acc + AP per generator (the headline)
    clean_acc_ap_<stamp>_metrics.csv     same, plus per-category rows
    clean_head_activation_<stamp>.txt    blur / jpeg head response on clean input
    clean_predictions_<stamp>.csv        per-image probabilities, all heads

The reports are rewritten after every generator and carry a STATUS line, so a
long run is safe to read while it is still going -- the same discipline
experiments/common.report_header uses.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch


from mlep.experiments import common as C                 # noqa: E402
from mlep.harness.device import get_device, setup_cuda_perf
from mlep.evaluation import common as T            # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    T.add_common_args(ap, default_max_per_label=0)
    ap.add_argument('--tag', default='clean', help='output filename prefix')
    return ap.parse_args()


def write_text(path, lines):
    """Atomic text write, same tmp + os.replace discipline as C.write_csv."""
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(tmp, path)
    return path


def header(meta, args, device, gpu, use_amp, done, total, title, unit='generators'):
    geom = (f"resize {meta['loadSize']} (no crop, full frame)" if args.no_crop
            else f"resize {meta['loadSize']} -> center crop {meta['cropSize']}")
    lines = [title, f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    lines.append(f"STATUS: {'complete' if done >= total else 'IN PROGRESS'} -- "
                 f"{done}/{total} {unit}. This file is rewritten after each "
                 f"one, so it is safe to read now.")
    lines += [
        f"checkpoint: {os.path.relpath(meta['path'], T.ROOT)}",
        f"  experiment={meta['experiment']}  steps={meta['total_steps']}  "
        f"heads={meta['heads']}  params={meta['params'] / 1e6:.2f}M  "
        f"bn_recalibrated={meta['bn_recalibrated']}",
        f"  cfg={meta['cfg']}",
        f"dataroot: {os.path.relpath(args.dataroot, T.ROOT)}",
        f"geometry: {geom}   (checkpoint was trained at "
        f"loadSize={meta['loadSize']} cropSize={meta['cropSize']})",
        f"device={device}{f' ({gpu})' if gpu else ''}  amp={use_amp}  "
        f"batch={args.batch_size}  workers={args.num_threads}  seed={args.seed}  "
        f"images_per_label={args.max_per_label or 'ALL'}",
        "",
    ]
    return lines


def acc_ap_report(rows, means, meta, args, device, gpu, use_amp, done, total):
    lines = header(meta, args, device, gpu, use_amp, done, total,
                   "Trained-model clean evaluation -- accuracy / average precision")
    lines.append("AI-vs-real head only (logit index "
                 f"{meta['ai_index']} of {meta['heads']}). acc/ap in %, "
                 "threshold 0.5 for acc; ap is average_precision_score.")
    lines.append("")
    header_row = ['generator', 'set', 'kind', 'n', 'acc', 'ap', 'roc_auc',
                  'real_acc', 'fake_acc']
    body = [[r['name'], r['set'], r['kind'], str(r['n']),
             C.fmt(100 * r['acc'], 1), C.fmt(100 * r['ap'], 1),
             C.fmt(100 * r['roc_auc'], 1), C.fmt(100 * r['real_acc'], 1),
             C.fmt(100 * r['fake_acc'], 1)] for r in rows]
    C.table(lines, header_row, body)
    lines.append("")
    lines.append("Per-set means (unweighted over generators, as test.py reports them)")
    mrows = [[k, str(v['n_gen']), str(v['n_img']),
              C.fmt(100 * v['acc'], 1), C.fmt(100 * v['ap'], 1)]
             for k, v in means.items()]
    C.table(lines, ['set', 'generators', 'images', 'mean_acc', 'mean_ap'], mrows)
    return lines


def head_report(rows, meta, args, device, gpu, use_amp, done, total):
    lines = header(meta, args, device, gpu, use_amp, done, total,
                   "Trained-model clean evaluation -- blur / JPEG head response")
    lines += [
        "Nothing on this pass was blurred or recompressed by the pipeline "
        "(blur_prob = jpg_prob = webp_prob = 0, so data_augment returns the image",
        "untouched). The blur and jpeg heads should therefore sit LOW. 'rate' is "
        "the fraction of images the head puts above 0.5; _real / _fake are the",
        "head's mean probability on the 0_real and 1_fake halves separately.",
        "",
        "READ disk_fmt AND THE _real/_fake SPLIT BEFORE CALLING A HIGH NUMBER A "
        "BUG. 'Clean' here means clean with respect to OUR pipeline only -- it says",
        "nothing about how the files were stored. Several sets ship 0_real as JPEG "
        "and 1_fake as PNG, and there the jpeg head is reading the container",
        "format, correctly, which happens to be the label. That is flagged as "
        "'<head>~label' in the verdict column: it is a property of the BENCHMARK,",
        "not a fault of the model, but it means an AI-vs-real score on those sets "
        "can be earned from a compression cue rather than from generator",
        "artefacts, and should be quoted with that caveat.",
        "",
        "The cue only survives when the image is not downscaled on the way in: a "
        "256x256 JPEG real keeps its 8x8 blocks and reads ~0.9, while a 500x375 or",
        "1024x1024 JPEG resized down to 256 reads ~0.02-0.08 because the resize "
        "destroys them.",
        "",
    ]
    diag = [h for h in meta['heads'] if h != 'ai']
    if not diag:
        lines.append("This checkpoint has only an 'ai' head -- no blur/jpeg "
                     "diagnostics to report.")
        return lines

    hrow = ['generator', 'set', 'n', 'disk_fmt']
    for h in diag:
        hrow += [f'{h}_mean', f'{h}_real', f'{h}_fake', f'{h}_rate']
    hrow.append('verdict')
    body = []
    for r in rows:
        cells = [r['name'], r['set'], str(r['n']), r['disk_fmt']]
        hot, split = [], []
        for h in diag:
            s = r['heads'][h]
            cells += [C.fmt(s['mean'], 3), C.fmt(s['real_mean'], 3),
                      C.fmt(s['fake_mean'], 3), C.fmt(s['rate'], 3)]
            if s['rate'] > 0.5:
                hot.append(h)
            # A head that separates the two classes this hard is tracking the
            # label, not the degradation -- on a CLEAN pass that can only come
            # from the sets differing by storage format.
            gap = abs(s['real_mean'] - s['fake_mean'])
            if gap > 0.5:
                split.append(f'{h}~label')
        verdict = ' '.join(split) if split else ('HOT: ' + '+'.join(hot) if hot
                                                else 'quiet')
        cells.append(verdict)
        body.append(cells)
    C.table(lines, hrow, body)

    lines.append("")
    lines.append("Overall, pooled over every generator scored so far:")
    for h in diag:
        w = np.array([r['n'] for r in rows], dtype=float)
        mean = float(np.average([r['heads'][h]['mean'] for r in rows], weights=w))
        rate = float(np.average([r['heads'][h]['rate'] for r in rows], weights=w))
        n_hot = sum(1 for r in rows if r['heads'][h]['rate'] > 0.5)
        n_split = sum(1 for r in rows
                      if abs(r['heads'][h]['real_mean']
                             - r['heads'][h]['fake_mean']) > 0.5)
        lines.append(f"  {h:5s}: mean p={mean:.3f}  rate(>0.5)={rate:.3f}  "
                     f"{n_hot}/{len(rows)} generators above 0.5 rate  "
                     f"-> {'as expected (quiet)' if rate <= 0.5 else 'ELEVATED'}")
        if n_split:
            lines.append(f"         {n_split}/{len(rows)} generators have the {h} "
                         f"head separating real from fake by >0.5 -- see the "
                         f"'{h}~label' rows above")
    return lines


def main():
    args = parse_args()
    T.seed_everything(args.seed)
    device = get_device(args.device)
    gpu = setup_cuda_perf(device)
    use_amp = (device.type == 'cuda') and not args.no_amp

    model, meta = T.load_trained_model(args.model_path, device)
    gens = T.discover_generators(args.dataroot, T.csv_list(args.sets),
                                 T.csv_list(args.generators))
    outdir = T.ensure_results_dir()
    stamp = time.strftime('%Y%m%d_%H%M%S')
    acc_path = os.path.join(outdir, f'{args.tag}_acc_ap_{stamp}.txt')
    head_path = os.path.join(outdir, f'{args.tag}_head_activation_{stamp}.txt')
    pred_path = os.path.join(outdir, f'{args.tag}_predictions_{stamp}.csv')
    csv_rows = []

    print(f"checkpoint : {args.model_path}")
    print(f"heads      : {meta['heads']}  cfg={meta['cfg']}")
    print(f"device     : {device} ({gpu})  amp={use_amp}")
    print(f"generators : {len(gens)} over "
          f"{sorted(set(g.set_name for g in gens))}")
    geom = 'no crop' if args.no_crop else f"crop {meta['cropSize']}"
    print(f"geometry   : resize {meta['loadSize']}, {geom}"
          f"  images/label={args.max_per_label or 'ALL'}")
    print()

    opt = T.EvalOpt(loadSize=meta['loadSize'], cropSize=meta['cropSize'],
                    no_crop=args.no_crop)
    # header once; appended per generator so a long run keeps its results on disk
    with open(pred_path, 'w') as f:
        f.write('set,generator,leaf,label,' +
                ','.join(f'p_{h}' for h in meta['heads']) + ',path\n')

    rows, t0 = [], time.time()
    for i, gen in enumerate(gens, 1):
        t1 = time.time()
        res = T.score_generator(model, gen, opt, device, args, use_amp=use_amp)
        if res is None:
            print(f"({i}/{len(gens)}) {gen.key}: no images, skipped")
            continue
        m = T.ai_metrics(res['y'], res['p'], meta['ai_index'])
        hs = T.head_stats(res['y'], res['p'], meta['heads'])
        rows.append(dict(name=gen.name, set=gen.set_name, kind=gen.kind,
                         disk_fmt=gen.disk_formats(), heads=hs, **m))

        with open(pred_path, 'a') as f:
            for j, path in enumerate(res['paths']):
                probs = ','.join(f"{v:.6f}" for v in res['p'][j])
                f.write(f"{gen.set_name},{gen.name},{res['leaf'][j]},"
                        f"{res['y'][j]},{probs},"
                        f"{os.path.relpath(path, args.dataroot)}\n")
        # leaf='ALL' is the pooled generator-level row, so the CSV answers
        # "acc/ap per generator" without re-deriving it from the leaf rows (which
        # cannot be averaged for AP anyway).
        csv_rows.append([gen.set_name, gen.name, 'ALL', gen.kind,
                         m['n'], m['acc'], m['ap'], m['roc_auc'],
                         m['real_acc'], m['fake_acc']]
                        + [v for h in meta['heads']
                           for v in (hs[h]['mean'], hs[h]['rate'])])
        if len(res['per_leaf']) > 1:
            for pl in res['per_leaf']:
                lm = T.ai_metrics(pl['y'], pl['p'], meta['ai_index'])
                lh = T.head_stats(pl['y'], pl['p'], meta['heads'])
                csv_rows.append([gen.set_name, gen.name, pl['leaf'], gen.kind,
                                 lm['n'], lm['acc'], lm['ap'], lm['roc_auc'],
                                 lm['real_acc'], lm['fake_acc']]
                                + [v for h in meta['heads']
                                   for v in (lh[h]['mean'], lh[h]['rate'])])

        el = time.time() - t1
        print(f"({i}/{len(gens)}) {gen.key:34s} n={m['n']:6d}  "
              f"acc={100 * m['acc']:5.1f}  ap={100 * m['ap']:5.1f}  "
              f"[{el:.0f}s, {m['n'] / max(el, 1e-9):.0f} img/s]  "
              + '  '.join(f"{h}={hs[h]['mean']:.3f}" for h in meta['heads']))

        means = {}
        for s in sorted(set(r['set'] for r in rows)):
            sub = [r for r in rows if r['set'] == s]
            means[s] = dict(n_gen=len(sub), n_img=sum(r['n'] for r in sub),
                            acc=float(np.mean([r['acc'] for r in sub])),
                            ap=float(np.mean([r['ap'] for r in sub])))
        write_text(acc_path, acc_ap_report(rows, means, meta, args, device, gpu,
                                           use_amp, i, len(gens)))
        write_text(head_path, head_report(rows, meta, args, device, gpu, use_amp,
                                          i, len(gens)))
        C.write_csv(C.csv_path(acc_path, 'metrics'),
                    ['set', 'generator', 'leaf', 'kind', 'n', 'acc', 'ap',
                     'roc_auc', 'real_acc', 'fake_acc']
                    + [f'{h}_{k}' for h in meta['heads'] for k in ('mean', 'rate')],
                    csv_rows)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    for p in (acc_path, C.csv_path(acc_path, 'metrics'), head_path, pred_path):
        print(f"  {os.path.relpath(p, T.ROOT)}")


if __name__ == '__main__':
    main()
