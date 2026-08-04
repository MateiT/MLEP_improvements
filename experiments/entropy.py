"""Experiment group 1: does another entropy definition detect AI images better?

Run it with
    python experiment_windows.py --experiment entropy --dataroot ... --out results/entropy_<stamp>.txt
or  ./run.sh entropy

"Better" here means a higher score on the held-out split and a smaller drop
under the corruption scenarios -- never a larger entropy VALUE. Renyi and
Tsallis are monotone re-weightings of the same value distribution, so they can
only help by changing how much a rare vs a common pixel value counts; the whole
point of the ablation is to find out whether that re-weighting is worth anything
once a classifier has seen it.

Two stages, both reported into the same results file (--entropy_stage picks):

  features  Compute entropy features per image (whole-image, non-overlapping
            8x8 / 16x16 / 32x32 windows, and the 2x2 MLEP window convention;
            grayscale, R, G, B and the joint RGB-triple view; summary stats
            mean/std/min/max/median/IQR/p10/p25/p75/p90 for every local map) and
            fit logistic regression / random forest / SVM / gradient-boosted
            trees on each ablation subset. This is the stage that can afford
            whole-image and 32x32 windows, which are far too expensive as a
            stride-1 front-end inside the network.
  deep      Train the MLEP network itself with each entropy front-end, through
            exactly the same run_config() the window sweep uses -- same steps,
            same BN recalibration, same corruption scenarios.

Ablation (both stages):
  A shannon (the existing baseline)   E shannon + renyi
  B renyi only (a = 0.5, 2, 4)        F shannon + tsallis
  C tsallis only (q = 0.5, 2, 4)      G shannon + permutation
  D permutation only (3x3, 4x4)       H all entropies

Leakage: the splits are the on-disk train/ and val/ folders, i.e. different
source images. Crops, resizes and corruptions are applied by the existing data
pipeline AFTER that split, never before it, so no transformed copy of a training
image can reach the test side. check_split_disjoint() verifies it each run.
"""
import os
import time

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import get_dataset
from experiments import common as C


# --------------------------------------------------------------------------- #
# Deep-stage configs: the same kwargs dict CONFIGS uses, so build_model and
# run_config take them unchanged. entropy_mode may be a list -> one map per mode
# is stacked, with mode 0's channels bit-identical to the single-mode config.
#
# Every config keeps the 3-scale pyramid, so the axis under test is the entropy
# functional and nothing else. ent_A_shannon is a verbatim copy of the sweep's
# baseline_2x2 and is the row everything else is read against.
#
# Permutation entropy is order-3, so it needs a window >= 3; the perm configs
# therefore run at 3x3 / 4x4 and the mixed ones (G, H) at 3x3, where every
# stacked map shares the same grid. Their honest comparison partner is
# w3x3_multiscale in CONFIGS, not baseline_2x2.
# --------------------------------------------------------------------------- #
_PYR = [1.0, 0.5, 0.25]
RENYI = ['renyi_0.5', 'renyi_2', 'renyi_4']
TSALLIS = ['tsallis_0.5', 'tsallis_2', 'tsallis_4']

ENTROPY_CONFIGS = {
    # A -- baseline
    'ent_A_shannon':        dict(entropy_mode='shannon', window_sizes=[2], scales=_PYR),
    # B -- renyi only
    'ent_B_renyi0.5':       dict(entropy_mode='renyi_0.5', window_sizes=[2], scales=_PYR),
    'ent_B_renyi2':         dict(entropy_mode='renyi_2', window_sizes=[2], scales=_PYR),
    'ent_B_renyi4':         dict(entropy_mode='renyi_4', window_sizes=[2], scales=_PYR),
    # C -- tsallis only
    'ent_C_tsallis0.5':     dict(entropy_mode='tsallis_0.5', window_sizes=[2], scales=_PYR),
    'ent_C_tsallis2':       dict(entropy_mode='tsallis_2', window_sizes=[2], scales=_PYR),
    'ent_C_tsallis4':       dict(entropy_mode='tsallis_4', window_sizes=[2], scales=_PYR),
    # D -- permutation only (3x3 and 4x4 neighbourhoods)
    'ent_D_perm3':          dict(entropy_mode='perm', window_sizes=[3], scales=_PYR),
    'ent_D_perm4':          dict(entropy_mode='perm', window_sizes=[4], scales=_PYR),
    # E/F/G -- shannon plus one family
    'ent_E_shannon_renyi':  dict(entropy_mode=['shannon'] + RENYI,
                                 window_sizes=[2], scales=_PYR),
    'ent_F_shannon_tsallis': dict(entropy_mode=['shannon'] + TSALLIS,
                                  window_sizes=[2], scales=_PYR),
    'ent_G_shannon_perm':   dict(entropy_mode=['shannon', 'perm'],
                                 window_sizes=[3], scales=_PYR),
    # H -- everything
    'ent_H_all':            dict(entropy_mode=['shannon'] + RENYI + TSALLIS + ['perm'],
                                 window_sizes=[3], scales=_PYR),
    # H, with every map divided by its own maximum. Mixing functionals mixes
    # value ranges much harder than mixing window sizes does (tsallis_4 tops out
    # at ~0.94 where shannon at 3x3 reaches 3.17), and conv1 has no bias and is
    # followed by a BN over its OUTPUT channels, so nothing downstream undoes it.
    'ent_H_all_normalized': dict(entropy_mode=['shannon'] + RENYI + TSALLIS + ['perm'],
                                 window_sizes=[3], scales=_PYR, normalize_entropy=True),
}

# Ablation groups -> which entropy families their feature columns / deep configs
# come from. Same letters in both stages so the two are read side by side.
GROUPS = {
    'A_shannon':         ('shannon',),
    'B_renyi':           ('renyi',),
    'C_tsallis':         ('tsallis',),
    'D_perm':            ('perm',),
    'E_shannon_renyi':   ('shannon', 'renyi'),
    'F_shannon_tsallis': ('shannon', 'tsallis'),
    'G_shannon_perm':    ('shannon', 'perm'),
    'H_all':             ('shannon', 'renyi', 'tsallis', 'perm'),
}


# --------------------------------------------------------------------------- #
# Feature stage: entropy summary statistics + classical classifiers
# --------------------------------------------------------------------------- #
VIEWS = ('gray', 'r', 'g', 'b', 'joint')
LOCAL_WINDOWS = (2, 8, 16, 32)          # 2 = the existing MLEP window convention
PERM_WINDOWS = (3, 4)
COUNT_VARIANTS = ('shannon', 'renyi_0.5', 'renyi_2', 'renyi_4',
                  'tsallis_0.5', 'tsallis_2', 'tsallis_4')
STATS = ('mean', 'std', 'min', 'max', 'median', 'iqr', 'p10', 'p25', 'p75', 'p90')

# ImageNet statistics the dataset normalises with; inverted to recover the 8-bit
# pixel values, because every entropy here is a statistic of DISCRETE symbols and
# a float image has no repeated values to count.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def feature_names():
    names = []
    for view in VIEWS:
        for v in COUNT_VARIANTS:
            names.append(f"{view}_whole_{v}")
        for w in LOCAL_WINDOWS:
            for v in COUNT_VARIANTS:
                names.extend(f"{view}_w{w}_{v}_{s}" for s in STATS)
        if view != 'joint':
            # A permutation is an ordering of scalars; an RGB triple has none.
            names.append(f"{view}_whole_perm")
            for pw in PERM_WINDOWS:
                names.extend(f"{view}_perm{pw}_{s}" for s in STATS)
    return names


FEATURE_NAMES = feature_names()


def group_columns(group):
    """Indices of the feature columns belonging to an ablation group."""
    fams = GROUPS[group]
    return np.array([i for i, n in enumerate(FEATURE_NAMES)
                     if any(f in n for f in fams)], dtype=int)


def to_uint8(batch):
    """Undo the dataset's Normalize and return (B, 3, H, W) uint8."""
    x = batch.detach().cpu().numpy() * _STD + _MEAN
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def group_counts(vals):
    """vals: (n, K) integer windows -> (n, K) size of each element's value group.

    Same sorted-run trick the network uses (ResNet._patch_entropy), so the
    feature stage and the deep stage compute the same quantity."""
    n, K = vals.shape
    s = np.sort(vals, axis=1)
    change = np.ones_like(s, dtype=bool)
    change[:, 1:] = s[:, 1:] != s[:, :-1]
    idx = np.broadcast_to(np.arange(K), (n, K))
    start = np.maximum.accumulate(np.where(change, idx, 0), axis=1)
    cand = np.where(change, idx, K)
    nxt = np.minimum.accumulate(cand[:, ::-1], axis=1)[:, ::-1]
    nxt = np.concatenate([nxt[:, 1:], np.full((n, 1), K, dtype=nxt.dtype)], axis=1)
    return (nxt - start).astype(np.float64)


def entropy_from_counts(count, K, variant):
    """Shannon / Renyi / Tsallis from per-element group sizes -- see
    ResNet._entropy_from_group_counts for why one count tensor serves all three."""
    p = count / K
    if variant == 'shannon':
        return -np.log2(p).mean(axis=1)
    kind, param = variant.split('_')
    a = float(param)
    s = (p ** a / count).sum(axis=1)
    if kind == 'renyi':
        return np.log2(s) / (1.0 - a)
    return (1.0 - s) / (a - 1.0)


def to_blocks(plane, w):
    """Non-overlapping w x w blocks of a 2-D plane -> (n, w*w). Any partial row
    or column at the right/bottom edge is dropped."""
    H, W = plane.shape
    h, ww = (H // w) * w, (W // w) * w
    b = plane[:h, :ww].reshape(h // w, w, ww // w, w).transpose(0, 2, 1, 3)
    return b.reshape(-1, w * w)


def _ord_code(a, b, c):
    """3-bit ordinal-pattern code of a triplet, as in ResNet._perm_entropy_map."""
    return (a > b).astype(np.int64) + 2 * (a > c).astype(np.int64) \
        + 4 * (b > c).astype(np.int64)


def _perm_entropy_from_hist(hist):
    total = hist.sum(axis=-1, keepdims=True)
    p = hist / np.maximum(total, 1)
    return -(p * np.log2(np.maximum(p, 1e-12))).sum(axis=-1)


def perm_entropy_whole(plane):
    """Permutation entropy over every horizontal/vertical triplet of the image."""
    h = _ord_code(plane[:, :-2], plane[:, 1:-1], plane[:, 2:]).ravel()
    v = _ord_code(plane[:-2, :], plane[1:-1, :], plane[2:, :]).ravel()
    hist = np.bincount(np.concatenate([h, v]), minlength=8).astype(np.float64)
    return float(_perm_entropy_from_hist(hist))


def perm_entropy_blocks(plane, w):
    """Per-neighbourhood permutation entropy -> one value per w x w block."""
    b = to_blocks(plane, w).reshape(-1, w, w)
    codes = []
    for i in range(w - 2):
        codes.append(_ord_code(b[:, i, :], b[:, i + 1, :], b[:, i + 2, :]))
        codes.append(_ord_code(b[:, :, i], b[:, :, i + 1], b[:, :, i + 2]))
    codes = np.concatenate(codes, axis=1)
    hist = (codes[:, :, None] == np.arange(8)).sum(axis=1).astype(np.float64)
    return _perm_entropy_from_hist(hist)


def summarise(vals):
    """The compact summary of one local-entropy map (order matches STATS)."""
    q10, q25, q50, q75, q90 = np.percentile(vals, [10, 25, 50, 75, 90])
    return [float(vals.mean()), float(vals.std()), float(vals.min()),
            float(vals.max()), float(q50), float(q75 - q25),
            float(q10), float(q25), float(q75), float(q90)]


def image_features(img):
    """img: (3, H, W) uint8 -> 1-D float32 vector aligned with FEATURE_NAMES."""
    r, g, b = (img[0].astype(np.int32), img[1].astype(np.int32),
               img[2].astype(np.int32))
    gray = np.rint(0.299 * r + 0.587 * g + 0.114 * b).astype(np.int32)
    # One symbol per RGB triple: two pixels match only if all three channels do.
    joint = (r << 16) | (g << 8) | b
    planes = dict(gray=gray, r=r, g=g, b=b, joint=joint)

    out = []
    for view in VIEWS:
        plane = planes[view]
        whole = plane.reshape(1, -1)
        cnt = group_counts(whole)
        for v in COUNT_VARIANTS:
            out.append(float(entropy_from_counts(cnt, whole.shape[1], v)[0]))
        for w in LOCAL_WINDOWS:
            blocks = to_blocks(plane, w)
            cnt = group_counts(blocks)
            for v in COUNT_VARIANTS:
                out.extend(summarise(entropy_from_counts(cnt, w * w, v)))
        if view != 'joint':
            out.append(perm_entropy_whole(plane))
            for pw in PERM_WINDOWS:
                out.extend(summarise(perm_entropy_blocks(plane, pw)))
    return np.asarray(out, dtype=np.float32)


class FeatureView(torch.utils.data.Dataset):
    """(image, label) -> (feature vector, label), computed IN THE DATALOADER.

    image_features() costs ~0.13 s per 224x224 image; doing it in the main
    process leaves the run apparently frozen for tens of minutes with the 8
    loader workers idle. Wrapping the dataset moves the work into those workers,
    so the cost is divided by --num_threads and the loop below is just I/O."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, lab = self.base[i]
        return torch.from_numpy(image_features(to_uint8(img[None])[0])), lab


def extract_features(args, split, is_train, aug, batches, label):
    """Entropy features for `batches` batches of one split, class by class.

    Looping over classes is what gives every sample its class tag, which the
    per-class ("per-generator", see the report note) breakdown needs -- the
    ImageFolder loader alone only yields (image, real/fake)."""
    from experiment_windows import make_loader, make_opt

    root = os.path.join(args.dataroot, split)
    classes = (args.classes.split(',') if args.classes
               else sorted(d for d in os.listdir(root)
                           if os.path.isdir(os.path.join(root, d))))
    per_class = max(1, batches // max(1, len(classes)))
    planned = per_class * len(classes) * args.batch_size
    every = max(1, per_class // 2)
    X, y, tags = [], [], []
    t0, seen = time.time(), 0
    for ci, cls in enumerate(classes, 1):
        opt = make_opt(args, split, is_train, aug=aug)
        opt.classes = [cls]
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader = make_loader(FeatureView(get_dataset(opt)), args, shuffle=True,
                             generator=g, persistent=False)
        for bi, (feat, lab) in enumerate(loader):
            if bi >= per_class:
                break
            X.append(feat.numpy())
            y.append(lab.numpy().astype(int))
            tags.extend([cls] * len(lab))
            seen += len(lab)
            # Same cadence idea as the training loop's step lines: this stage can
            # run for minutes, so it has to say something while it does.
            if (bi + 1) % every == 0:
                print(f"  [features] {label}: {seen}/{planned} images  "
                      f"({ci}/{len(classes)} {cls})  ({time.time() - t0:.0f}s)")
        del loader
    X = np.concatenate(X) if X else np.zeros((0, len(FEATURE_NAMES)), np.float32)
    print(f"  [features] {label}: {X.shape[0]} images x {X.shape[1]} features "
          f"({time.time() - t0:.0f}s)")
    return X, np.concatenate(y), np.asarray(tags)


def classifiers(seed):
    """The four classifiers the comparison uses. Scaled where it matters
    (logreg, SVM); the tree models are scale-invariant."""
    return {
        'logreg': lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
        'rf': lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                             random_state=seed),
        'svm': lambda: make_pipeline(
            StandardScaler(), SVC(probability=True, random_state=seed)),
        'gbt': lambda: HistGradientBoostingClassifier(random_state=seed),
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_entropy_report(results, args, device, gpu_name, use_amp, total):
    """Same layout as experiment_windows.build_report: header, STATUS, one block
    per model, then compact tables. Written by write_report() after every model."""
    def clean_key(r):
        v = r['scenarios'].get('clean', {}).get('roc_auc', float('nan'))
        return v if v == v else -1

    results = sorted(results, key=clean_key, reverse=True)
    done = len(results)
    scen_names = list(dict.fromkeys(
        s for r in results for s in r['scenarios']))

    lines = C.report_header("MLEP entropy comparison", args, device, gpu_name,
                            use_amp, done, total,
                            extra=f"stage={args.entropy_stage}  "
                                  f"feat_batches={args.entropy_feat_train_batches}/"
                                  f"{args.entropy_feat_val_batches}")
    setup = next((r['setup'] for r in results if r.get('setup')), [])
    if setup:
        lines.extend(setup)
        lines.append("")

    lines.append("Per-model results")
    lines.append("-" * 60)
    for r in results:
        lines.append(f"[{r['name']}]  {r['desc']}")
        lines.append(f"    {r['sub']}")
        for s in scen_names:
            sc = r['scenarios'].get(s)
            if sc is None:
                continue
            # Prefix identical to the sweep's, so scripts/aggregate_results.py
            # parses these files too; the wider metrics are appended after it.
            lines.append(
                f"    test/{s:10s}  acc={C.fmt(sc['acc'])}  "
                f"ap={C.fmt(sc['pr_auc'])}  (n={sc['n']})  "
                f"roc_auc={C.fmt(sc['roc_auc'])}  f1={C.fmt(sc['f1'])}  "
                f"prec={C.fmt(sc['precision'])}  rec={C.fmt(sc['recall'])}  "
                f"eer={C.fmt(sc['eer'])}  brier={C.fmt(sc['brier'])}  "
                f"ece={C.fmt(sc['ece'])}")
        if r.get('ci'):
            lo, hi = r['ci']
            lines.append(f"    clean roc_auc 95% CI (bootstrap): "
                         f"[{C.fmt(lo)}, {C.fmt(hi)}]")
        if r.get('per_class'):
            per = "  ".join(f"{k}={C.fmt(v)}" for k, v in sorted(r['per_class'].items()))
            lines.append(f"    per-class clean roc_auc: {per}")
        lines.append("")

    if results:
        rows = [[r['name']] + [C.fmt(r['scenarios'][s]['roc_auc']) if s in r['scenarios']
                               else '-' for s in scen_names] for r in results]
        lines.append("ROC-AUC per test scenario  [higher = more robust]")
        C.table(lines, ['config'] + scen_names, rows,
                note="Cols: clean vs corrupted test sets. Read a row's clean cell "
                     "against its blur / jpeg / webp cells for robustness, and every "
                     "row against the A_shannon row of the same stage/classifier.")
        lines.append("")

        # Ablation table: the best classifier per group, and the delta vs A.
        feats = [r for r in results if r['stage'] == 'features']
        if feats:
            best = {}
            for r in feats:
                g = r['group']
                if g not in best or clean_key(r) > clean_key(best[g]):
                    best[g] = r
            base = best.get('A_shannon')
            rows = []
            for g in GROUPS:
                r = best.get(g)
                if r is None:
                    continue
                d = (clean_key(r) - clean_key(base)) if base is not None else float('nan')
                mean_corr = np.nanmean([r['scenarios'][s]['roc_auc']
                                        for s in scen_names if s != 'clean'
                                        and s in r['scenarios']])
                rows.append([g, r['clf'], C.fmt(clean_key(r)), C.fmt(d),
                             C.fmt(mean_corr)])
            lines.append("Ablation (feature stage): best classifier per group")
            C.table(lines, ['group', 'clf', 'clean', 'delta_vs_A', 'mean_corrupt'], rows,
                    note="delta_vs_A is clean ROC-AUC minus the A_shannon row. "
                         "'Better' = positive delta AND a mean_corrupt at least as "
                         "high, not a larger entropy value.")
            lines.append("")

    lines.append("Notes")
    lines.append("- Feature stage: entropy summary statistics + sklearn; deep stage: "
                 "the MLEP network trained by run_config(), identical to the sweep.")
    lines.append("- Splits are the on-disk train/ and val/ folders; crops, resizes "
                 "and corruptions are applied after that split, never before.")
    lines.append("- Short training -> the deep ranking is indicative. Re-run the top "
                 "configs with more --max_train_steps to confirm.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run_entropy_experiment(args, device, gpu_name, use_amp):
    from experiment_windows import EVAL_SCENARIOS, run_config, write_report

    args.keep_scores = True           # evaluate() then returns labels + logits
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    deep_names = [n for n in (args.configs.split(',') if args.configs
                              else list(ENTROPY_CONFIGS))]
    unknown = [n for n in deep_names if n not in ENTROPY_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}. "
                         f"Available: {list(ENTROPY_CONFIGS)}")
    do_feat = args.entropy_stage in ('both', 'features')
    do_deep = args.entropy_stage in ('both', 'deep')
    n_feat = len(GROUPS) * len(classifiers(seed)) if do_feat else 0
    total = n_feat + (len(deep_names) if do_deep else 0)

    setup = C.check_split_disjoint(args.dataroot, args.train_split, args.val_split,
                                   args.classes.split(',') if args.classes else None)
    print(f"Device: {device}" + (f" ({gpu_name})" if gpu_name else "") +
          f" | arch: {args.arch} | steps: {args.max_train_steps} "
          f"| crop: {args.cropSize} | batch: {args.batch_size} "
          f"| workers: {args.num_threads} | amp: {use_amp}")
    print(f"Entropy experiment: stage={args.entropy_stage}  "
          f"{n_feat} feature model(s) + {len(deep_names) if do_deep else 0} deep config(s)")
    print(f"  {setup[0]}\n")

    results, done = [], 0
    metric_rows = []

    if do_feat:
        n_sets = 1 + len(EVAL_SCENARIOS) + (1 if args.unseen_split else 0)
        print(f"  feature stage: {n_sets} sets, "
              f"{args.entropy_feat_train_batches * args.batch_size} train + "
              f"{args.entropy_feat_val_batches * args.batch_size} val images each, "
              f"{len(FEATURE_NAMES)} features per image")
        Xtr, ytr, ctr = extract_features(args, args.train_split, True, {},
                                         args.entropy_feat_train_batches, 'train/clean')
        val = {}
        for sname, saug in EVAL_SCENARIOS.items():
            val[sname] = extract_features(args, args.val_split, False, saug,
                                          args.entropy_feat_val_batches,
                                          f'val/{sname}')
        if args.unseen_split:
            val['unseen'] = extract_features(args, args.unseen_split, False, {},
                                             args.entropy_feat_val_batches,
                                             'unseen/clean')

        # features.csv: the clean train + val matrices, one row per image.
        path = C.csv_path(args.out, 'features')
        header = ['split', 'class', 'label'] + FEATURE_NAMES
        rows = [['train', c, int(l)] + list(x) for x, l, c in zip(Xtr, ytr, ctr)]
        Xv, yv, cv = val['clean']
        rows += [['val', c, int(l)] + list(x) for x, l, c in zip(Xv, yv, cv)]
        C.write_csv(path, header, rows)
        print(f"  [features] wrote {path}\n")

        for gi, group in enumerate(GROUPS):
            cols = group_columns(group)
            for cname, factory in classifiers(seed).items():
                name = f"{group}+{cname}"
                done += 1
                print(f"=== [{done}/{total}] {name} : {len(cols)} features ===")
                t0 = time.time()
                clf = factory()
                clf.fit(Xtr[:, cols], ytr)
                print(f"  [{name}] fit {Xtr.shape[0]} samples "
                      f"({time.time() - t0:.0f}s)")
                scenarios, ci, per_class = {}, None, {}
                for sname, (Xs, ys, cs) in val.items():
                    prob = clf.predict_proba(Xs[:, cols])[:, 1]
                    m = C.binary_metrics(ys, prob)
                    scenarios[sname] = m
                    print(f"  [{name}] test/{sname:9s}: acc={m['acc']:.4f} "
                          f"ap={m['pr_auc']:.4f} (n={m['n']}) "
                          f"[roc_auc={m['roc_auc']:.4f} f1={m['f1']:.4f} "
                          f"eer={m['eer']:.4f}]")
                    metric_rows.append(['features', group, cname, sname]
                                       + [m[k] for k in _METRIC_KEYS])
                    if sname == 'clean':
                        ci = C.bootstrap_ci(ys, prob, 'roc_auc', seed=seed)
                        for cls in np.unique(cs):
                            sel = cs == cls
                            if len(np.unique(ys[sel])) > 1:
                                per_class[cls] = C.binary_metrics(
                                    ys[sel], prob[sel])['roc_auc']
                results.append(dict(
                    name=name, stage='features', group=group, clf=cname,
                    desc=f"stage=features  group={group}  families={GROUPS[group]}",
                    sub=f"features={len(cols)}  train_n={Xtr.shape[0]}  "
                        f"fit_time={time.time() - t0:.0f}s",
                    scenarios=scenarios, ci=ci, per_class=per_class,
                    setup=setup if done == 1 else []))
                write_report(results, args, device, gpu_name, use_amp, total,
                             builder=build_entropy_report)
                print(f"  [{name}] done -- {done}/{total} written to {args.out}\n")

    if do_deep:
        for name in deep_names:
            cfg = ENTROPY_CONFIGS[name]
            done += 1
            print(f"=== [{done}/{total}] {name} : {cfg} ===")
            r = run_config(name, cfg, args, device)
            scenarios = {}
            for sname, sc in r['scenarios'].items():
                prob = 1.0 / (1.0 + np.exp(-np.clip(sc['scores'], -30, 30)))
                m = C.binary_metrics(sc['y_true'], prob)
                m['acc_best'] = sc['acc_best']
                scenarios[sname] = m
                metric_rows.append(['deep', name, 'mlep', sname]
                                   + [m[k] for k in _METRIC_KEYS])
            clean = r['scenarios'].get('clean', {})
            ci = C.bootstrap_ci(clean.get('y_true', []),
                                1.0 / (1.0 + np.exp(-np.clip(clean.get('scores', []),
                                                             -30, 30))),
                                'roc_auc', seed=seed) if 'scores' in clean else None
            results.append(dict(
                name=name, stage='deep', group='', clf='mlep', desc=str(cfg),
                sub=f"in_channels={r['in_ch']}  params={r['params'] / 1e6:.2f}M  "
                    f"steps={r['steps']}  train_time={r['time']:.0f}s",
                scenarios=scenarios, ci=ci, per_class={},
                setup=setup if done == 1 else []))
            write_report(results, args, device, gpu_name, use_amp, total,
                         builder=build_entropy_report)
            print(f"  [{name}] done -- {done}/{total} written to {args.out}\n")

    C.write_csv(C.csv_path(args.out, 'metrics'),
                ['stage', 'config', 'classifier', 'scenario'] + _METRIC_KEYS,
                metric_rows)
    C.write_csv(C.csv_path(args.out, 'ablation'),
                ['group', 'stage', 'config', 'classifier', 'clean_roc_auc',
                 'delta_vs_A', 'mean_corrupt_roc_auc'],
                _ablation_rows(results))

    report = write_report(results, args, device, gpu_name, use_amp, total,
                          builder=build_entropy_report)
    print("\n" + report)
    print(f"\nResults written to {args.out}")
    print(f"CSVs: {C.csv_path(args.out, 'metrics')}, "
          f"{C.csv_path(args.out, 'ablation')}"
          + (f", {C.csv_path(args.out, 'features')}" if do_feat else ""))


_METRIC_KEYS = ['n', 'pos', 'roc_auc', 'pr_auc', 'acc', 'precision', 'recall',
                'f1', 'eer', 'brier', 'ece']


def _group_of(r):
    """Ablation letter of a result row, for both stages ('ent_B_renyi2' -> B)."""
    if r['stage'] == 'features':
        return r['group']
    letter = r['name'].split('_')[1] if r['name'].startswith('ent_') else '?'
    for g in GROUPS:
        if g.startswith(letter + '_'):
            return g
    return '?'


def _ablation_rows(results):
    rows = []
    base = {}
    for r in results:
        g = _group_of(r)
        clean = r['scenarios'].get('clean', {}).get('roc_auc', float('nan'))
        key = (r['stage'], r['clf'])
        if g == 'A_shannon' and (key not in base or clean > base[key]):
            base[key] = clean
    for r in results:
        g = _group_of(r)
        clean = r['scenarios'].get('clean', {}).get('roc_auc', float('nan'))
        corr = [m['roc_auc'] for s, m in r['scenarios'].items() if s != 'clean']
        b = base.get((r['stage'], r['clf']), float('nan'))
        rows.append([g, r['stage'], r['name'], r['clf'], clean, clean - b,
                     float(np.nanmean(corr)) if corr else float('nan')])
    return rows
