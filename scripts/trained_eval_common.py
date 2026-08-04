"""Shared plumbing for scoring a TRAINED checkpoint on the test-set generators.

WHY THIS EXISTS
---------------
test.py scores the released MLEP weights: a single logit, loaded into
resnet50(num_classes=1). The checkpoints ./run.sh degradation writes are not that
network. results/degradation_*_deg_baseline_2x2.pt carries

    heads       ['ai', 'blur', 'jpeg']      num_classes 3      in_channels 9
    cfg         {'window_sizes': [2], 'scales': [1.0, 0.5, 0.25]}

so it emits three probabilities per image (experiments/degradation.py's SL_BIN:
logit 0 = AI-generated, 1 = blurred, 2 = JPEG-compressed) and has to be rebuilt
from its own cfg. Pointing test.py at one of these fails on a shape mismatch in
conv1 and fc1, which is what this module exists to get right in one place.

Everything else is reused rather than reimplemented:
  data.datasets.binary_dataset   the exact resize -> degrade -> crop -> flip ->
                                 ToTensor -> Normalize chain used everywhere else,
                                 so the images scored here are the kind the model
                                 was trained on
  experiments.common             binary_metrics (acc + pr_auc, i.e. AP), table,
                                 write_csv, report_header, fmt
  experiment_windows             get_device, setup_cuda_perf, amp_autocast
"""
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.datasets import binary_dataset            # noqa: E402
from networks.resnet import resnet50                # noqa: E402

LABELS = ('0_real', '1_fake')
DEFAULT_DATAROOT = os.path.join(ROOT, 'datasets', 'TestDatasets')
RESULTS_DIR = os.path.join(ROOT, 'TrainedModelResults')

# Which sets count as GAN vs diffusion for the best/worst split. Keyed on the set
# directory name so adding GAN-set-2 later needs no code change.
GAN_SETS = ('GAN-set-1', 'GAN-set-2')
DIFFUSION_SETS = ('Diffusion-set',)


# --------------------------------------------------------------------------- #
# perturbation ladder
# --------------------------------------------------------------------------- #
# prob=1.0 everywhere so each condition is deterministic (ALWAYS applied), the
# convention test.py's CORRUPTIONS and experiment_windows.EVAL_SCENARIOS use. With
# single-element blur_sig / jpg_qual / webp_qual lists, data.datasets'
# sample_continuous / sample_discrete return that element outright, so there is no
# RNG in the corruption at all.
#
# The blur sigmas and JPEG qualities are the checkpoint's OWN training levels
# (ck['blur_levels'], ck['jpeg_levels']), which is what makes the blur / jpeg head
# outputs interpretable against what the model was taught.
#
# WebP is Google's format and is the "google compression" arm. Note it was NOT in
# the degradation training grid -- experiments/degradation.py degrades on blur x
# jpeg only -- so these rows are an out-of-distribution probe, and how the *jpeg*
# head answers them is the interesting part.
PERTURBATIONS = {
    'clean':       dict(),
    'blur_0.5':    dict(blur_prob=1.0, blur_sig=[0.5]),
    'blur_1.5':    dict(blur_prob=1.0, blur_sig=[1.5]),
    'blur_3.0':    dict(blur_prob=1.0, blur_sig=[3.0]),
    'jpeg_q75':    dict(jpg_prob=1.0, jpg_qual=[75], jpg_method=['pil']),
    'jpeg_q45':    dict(jpg_prob=1.0, jpg_qual=[45], jpg_method=['pil']),
    'jpeg_q15':    dict(jpg_prob=1.0, jpg_qual=[15], jpg_method=['pil']),
    'webp_q80':    dict(webp_prob=1.0, webp_qual=[80]),
    'webp_q50':    dict(webp_prob=1.0, webp_qual=[50]),
    'webp_q20':    dict(webp_prob=1.0, webp_qual=[20]),
}
PERTURBATION_GROUPS = {
    'blur': ['blur_0.5', 'blur_1.5', 'blur_3.0'],
    'jpeg': ['jpeg_q75', 'jpeg_q45', 'jpeg_q15'],
    'webp': ['webp_q80', 'webp_q50', 'webp_q20'],
}


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_trained_model(path, device):
    """-> (model.eval() on device, meta dict).

    Handles both checkpoint shapes this repo produces:
      * a degradation/sweep checkpoint -- dict with 'model' + 'cfg', rebuilt through
        resnet50(**cfg) so window_sizes / scales / entropy_mode / num_classes all
        come from the file rather than from an assumption here;
      * a bare state_dict, e.g. pretrained/model_epoch_best.pth, which is the
        released 1-logit MLEP network.
    strict=True in both cases: a silently partial load would show up as plausible
    but meaningless accuracy, which is the one failure mode worth crashing on.
    """
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ck, dict) and 'model' in ck and 'cfg' in ck:
        cfg = dict(ck['cfg'])
        cfg.setdefault('num_classes', ck.get('num_classes', 1))
        model = resnet50(**cfg)
        model.load_state_dict(ck['model'], strict=True)
        heads = list(ck.get('heads') or ['ai'])
        meta = dict(path=path, cfg=cfg, heads=heads,
                    loadSize=ck.get('loadSize', 256), cropSize=ck.get('cropSize', 224),
                    total_steps=ck.get('total_steps'), experiment=ck.get('experiment'),
                    bn_recalibrated=ck.get('bn_recalibrated'),
                    blur_levels=ck.get('blur_levels'), jpeg_levels=ck.get('jpeg_levels'))
    else:
        state = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
        model = resnet50(num_classes=1)
        model.load_state_dict(state, strict=True)
        meta = dict(path=path, cfg=dict(num_classes=1), heads=['ai'],
                    loadSize=256, cropSize=224, total_steps=None,
                    experiment='released', bn_recalibrated=None,
                    blur_levels=None, jpeg_levels=None)

    if 'ai' not in meta['heads']:
        raise SystemExit(f"{path}: heads {meta['heads']} has no 'ai' head to score.")
    meta['ai_index'] = meta['heads'].index('ai')
    meta['params'] = sum(p.numel() for p in model.parameters())
    return model.to(device).eval(), meta


# --------------------------------------------------------------------------- #
# generator discovery
# --------------------------------------------------------------------------- #
class Generator:
    """One scored unit: a generator directory, plus the leaf dirs under it that
    actually hold 0_real/1_fake.

    Two layouts occur and both are already normal for this project (see
    data/__init__.py's get_dataset): flat -- biggan/{0_real,1_fake} -- and one
    category level -- progan/car/{0_real,1_fake}, ddpm/google-ddpm-cat-256/... .
    Scoring each leaf and pooling the predictions gives exactly the generator-level
    acc/AP that concatenating them would, and yields per-category rows for free.
    """

    def __init__(self, set_name, name, gen_dir, leaves):
        self.set_name = set_name
        self.name = name
        self.dir = gen_dir
        self.leaves = leaves

    def leaf_label(self, leaf):
        """'-' for a flat generator, else the category dir name."""
        rel = os.path.relpath(leaf, self.dir)
        return '-' if rel == '.' else rel

    @property
    def key(self):
        return f"{self.set_name}/{self.name}"

    @property
    def kind(self):
        if self.set_name in DIFFUSION_SETS:
            return 'diffusion'
        return 'gan' if self.set_name in GAN_SETS else 'other'

    def n_images(self):
        return sum(len(os.listdir(os.path.join(l, lab)))
                   for l in self.leaves for lab in LABELS
                   if os.path.isdir(os.path.join(l, lab)))

    def disk_formats(self):
        """Extensions present on disk, commonest first.

        Reported because a 'clean' run is only clean with respect to OUR pipeline:
        whichfaceisreal ships as JPEG and san holds 19 BMPs, so an elevated jpeg
        head there is the model being right, not a bug.
        """
        counts = {}
        for leaf in self.leaves:
            for lab in LABELS:
                d = os.path.join(leaf, lab)
                if not os.path.isdir(d):
                    continue
                for f in os.listdir(d):
                    ext = os.path.splitext(f)[1].lower().lstrip('.')
                    if ext:
                        counts[ext] = counts.get(ext, 0) + 1
        return ','.join(k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1]))


def _is_leaf(d):
    return all(os.path.isdir(os.path.join(d, lab)) for lab in LABELS)


def _subdirs(d):
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


def discover_generators(dataroot, only_sets=None, only_generators=None, warn=print):
    """-> list of Generator, sorted by (set, name).

    Absent or empty sets are skipped with a warning instead of raising: GAN-set-2
    is not downloaded on every box, and this is the same failure that kills
    test.py at its os.listdir.
    """
    out = []
    if not os.path.isdir(dataroot):
        raise SystemExit(f"dataroot {dataroot} does not exist")
    for set_name in _subdirs(dataroot):
        if only_sets and set_name not in only_sets:
            continue
        set_dir = os.path.join(dataroot, set_name)
        gens = _subdirs(set_dir)
        if not gens:
            warn(f"[skip] {set_name}: no generator directories")
            continue
        for gen in gens:
            if only_generators and gen not in only_generators:
                continue
            gdir = os.path.join(set_dir, gen)
            leaves = [gdir] if _is_leaf(gdir) else [
                os.path.join(gdir, s) for s in _subdirs(gdir)
                if _is_leaf(os.path.join(gdir, s))]
            if not leaves:
                warn(f"[skip] {set_name}/{gen}: no 0_real/1_fake under it")
                continue
            out.append(Generator(set_name, gen, gdir, leaves))
    if not out:
        raise SystemExit(f"No populated generators under {dataroot}")
    return out


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class EvalOpt:
    """The attribute bag data.datasets.binary_dataset reads.

    isTrain=False fixes the chain to resize -> data_augment -> crop -> ToTensor ->
    Normalize with no flip, which is what DegradationDataset mirrors for its own
    val sets.
    """

    def __init__(self, aug=None, loadSize=256, cropSize=224,
                 no_crop=False, no_resize=False):
        self.isTrain = False
        self.mode = 'binary'
        self.no_flip = True
        self.no_crop = no_crop
        self.no_resize = no_resize
        self.loadSize = loadSize
        self.cropSize = cropSize
        # clean defaults; data_augment short-circuits when all three probs are 0
        self.blur_prob, self.blur_sig = 0.0, [0.5]
        self.jpg_prob, self.jpg_qual, self.jpg_method = 0.0, [75], ['pil']
        self.webp_prob, self.webp_qual = 0.0, [80]
        for k, v in (aug or {}).items():
            setattr(self, k, v)


def leaf_loader(leaf, opt, max_per_label=0, seed=0, batch_size=32, workers=8,
                pin_memory=True):
    """-> (DataLoader, paths) with paths aligned to iteration order.

    max_per_label>0 takes a SEEDED RANDOM sample of that many images per class
    rather than the first N: filenames encode generation order in several of these
    sets, so a prefix could correlate with something systematic. Same discipline,
    and the same reason, as scripts/stage_dataset.py's sampling.
    """
    ds = binary_dataset(opt, leaf)
    idx = list(range(len(ds.samples)))
    if max_per_label:
        by_class = {}
        for i, (_, cls) in enumerate(ds.samples):
            by_class.setdefault(cls, []).append(i)
        idx = []
        for cls in sorted(by_class):
            pool = by_class[cls]
            take = (sorted(random.Random(seed + cls).sample(pool, max_per_label))
                    if len(pool) > max_per_label else pool)
            idx.extend(take)
        ds = Subset(ds, idx)
        paths = [ds.dataset.samples[i][0] for i in idx]
    else:
        paths = [p for p, _ in ds.samples]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=pin_memory)
    return loader, paths


@torch.no_grad()
def predict(model, loader, device, use_amp=False):
    """-> (y_true[N], probs[N, n_heads]) as sigmoid probabilities.

    Only the binary logits are read (out[:, :n_heads]); a checkpoint trained with
    --deg_sev_weight>0 also carries 11 severity logits after them, which are not
    scored here.
    """
    from experiment_windows import amp_autocast
    n_heads = model.fc1.out_features if hasattr(model, 'fc1') else 1
    ys, ps = [], []
    for img, label in loader:
        with amp_autocast(use_amp):
            out = model(img.to(device, non_blocking=True)).float()
        ps.append(torch.sigmoid(out[:, :n_heads]).cpu().numpy())
        ys.append(label.numpy())
    if not ys:
        return np.zeros(0, dtype=int), np.zeros((0, n_heads))
    return np.concatenate(ys).astype(int), np.concatenate(ps)


def score_generator(model, gen, opt, device, args, use_amp=False):
    """Score every leaf of one generator and pool. -> dict with pooled arrays and
    the per-leaf breakdown."""
    y_all, p_all, path_all, leaf_all = [], [], [], []
    per_leaf = []
    for leaf in gen.leaves:
        loader, paths = leaf_loader(
            leaf, opt, max_per_label=args.max_per_label, seed=args.seed,
            batch_size=args.batch_size, workers=args.num_threads,
            pin_memory=(device.type == 'cuda'))
        y, p = predict(model, loader, device, use_amp=use_amp)
        if len(y) == 0:
            continue
        label = gen.leaf_label(leaf)
        per_leaf.append(dict(leaf=label, n=len(y), y=y, p=p))
        y_all.append(y)
        p_all.append(p)
        path_all.extend(paths)
        leaf_all.extend([label] * len(y))
    if not y_all:
        return None
    return dict(y=np.concatenate(y_all), p=np.concatenate(p_all),
                paths=path_all, leaf=leaf_all, per_leaf=per_leaf)


# --------------------------------------------------------------------------- #
# metrics / formatting
# --------------------------------------------------------------------------- #
def ai_metrics(y, p, ai_index):
    """acc + AP for the AI-vs-real head, via experiments.common.binary_metrics
    (whose pr_auc IS average_precision_score, the same AP test.py reports)."""
    from experiments import common as C
    m = C.binary_metrics(y, p[:, ai_index])
    return dict(n=m['n'], acc=m['acc'], ap=m['pr_auc'], roc_auc=m['roc_auc'],
                real_acc=float((p[y == 0, ai_index] <= 0.5).mean()) if (y == 0).any() else float('nan'),
                fake_acc=float((p[y == 1, ai_index] > 0.5).mean()) if (y == 1).any() else float('nan'))


def head_stats(y, p, heads):
    """Mean probability and >0.5 rate for every head, overall and split by class.

    This is what makes the clean run's blur/jpeg claim checkable: on undegraded
    input those two heads should sit low, and 'low' has to be a number in a file
    rather than an impression.
    """
    out = {}
    for i, h in enumerate(heads):
        col = p[:, i]
        out[h] = dict(
            mean=float(col.mean()), rate=float((col > 0.5).mean()),
            real_mean=float(col[y == 0].mean()) if (y == 0).any() else float('nan'),
            fake_mean=float(col[y == 1].mean()) if (y == 1).any() else float('nan'),
            p90=float(np.percentile(col, 90)),
        )
    return out


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_common_args(ap, default_max_per_label=0):
    ck_default = os.path.join(
        ROOT, 'results',
        'degradation_20260803_151210_seed100_deg_baseline_2x2.pt')
    ap.add_argument('--model_path', default=ck_default,
                    help='trained checkpoint (default: the deg_baseline_2x2 run)')
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--sets', default='', help='comma-separated set names to limit to')
    ap.add_argument('--generators', default='', help='comma-separated generators to limit to')
    ap.add_argument('--max_per_label', type=int, default=default_max_per_label,
                    help='0 = every image; N = seeded random N per 0_real/1_fake dir')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_threads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=100)
    # experiment_windows.get_device takes a device STRING ('' = auto-detect
    # cuda -> mps -> cpu), not test.py's gpu_ids list.
    ap.add_argument('--device', default='', help="'' = auto, else cuda / cuda:1 / cpu")
    ap.add_argument('--no_amp', action='store_true')
    ap.add_argument('--no_crop', action='store_true',
                    help="skip the center crop and score the full resized image, "
                         "as test.py does; default matches the checkpoint's own "
                         "loadSize->cropSize geometry")
    return ap


def csv_list(s):
    return [x.strip() for x in s.split(',') if x.strip()] or None


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR
