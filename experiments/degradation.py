"""Experiment group 2: can MLEP also say HOW an image was degraded?

Run it with
    python experiment_windows.py --experiment mlep_degradation --dataroot ... \
        --out results/degradation_<stamp>.txt
or  ./run.sh degradation

The AI-vs-real detector is not rewritten. It is MLEP's own front-end at its own
settings -- 2x2 entropy windows on the 3-scale pyramid, i.e. exactly the
released baseline_2x2 -- trained by the same run_config-style
train / BN-recalibrate / evaluate loop. The ONLY difference from the stock model
is that the final layer emits three logits instead of one (num_classes is
already a ResNet argument, so there is no network change at all):

    logit 0      ai_generated_probability
    logit 1      blur_probability
    logit 2      jpeg_compression_probability

Severity is optional and off by default. With --deg_sev_weight > 0 the layer
emits 11 more logits -- a 7-way predicted_jpeg_quality (lossless, 15..100) and a
4-way predicted_blur_level (none, weak, medium, strong) -- and the report gains
the severity metrics. At the default weight 0 none of that is built, so the
model is the released one plus two extra output units.

Degradations
------------
JPEG quality 15 / 30 / 45 / 60 / 75 / 100 plus an untouched lossless condition.
Quality 100 is a JPEG (jpeg_compression_probability target 1) and stays its own
severity level -- only the untouched image counts as "not compressed".

Blur reuses data.datasets.gaussian_blur, i.e. scipy.ndimage.gaussian_filter with
its default truncate=4.0, so the kernel is 2*int(4*sigma + 0.5) + 1 pixels:

    none    sigma 0.0   (no filter applied at all)
    weak    sigma 0.5   kernel  5x5
    medium  sigma 1.5   kernel 13x13
    strong  sigma 3.0   kernel 25x25

Blur is applied before JPEG, the same order data_augment uses.

Leakage
-------
Splits are the on-disk train/ and val/ folders, i.e. disjoint SOURCE images, and
every variant is generated from a source inside its own split -- a source image
and its blurred/compressed copies can never straddle the split. Which cell of
the (quality x blur) grid a variant gets is a deterministic function of the
source path (crc32) and the variant index, so the assignment is reproducible and
verifiable; tests/test_degradation_dataset.py checks both properties.

Four grid cells are held out of training entirely and only appear in the
'unseen' evaluation, which is the unseen-combination test.

Training a keeper
-----------------
The defaults here are the short fixed-step comparison recipe this file was built
for: constant LR, plain Adam, one evaluation at the end, weights discarded. Six
flags, all off by default and all leaving the default run byte-identical, turn it
into a real training run whose output is a model rather than a table:

    --init_from PATH          load everything but fc1.* from a checkpoint;
                              pretrained/model_epoch_best.pth is this exact
                              network, so released MLEP drops straight in
    --deg_save_ckpt           write <out-stem>_<config>.pt after BN recalibration
    --deg_clean_oversample N  repeat the untouched cell N times in the TRAINING
                              cell list only -- 1 of 24 training cells is clean,
                              which is what starves the ai head and skews the
                              blur / jpeg heads to 71% / 83% positives
    --lr_schedule cosine      with --warmup_steps and --lr_min
    --weight_decay W          Adam -> AdamW
    --deg_val_every N         score one condition every N steps and keep the best

Evaluation is deliberately untouched by all of them: CONDITIONS and the val sets
keep the plain 24-cell grid, and the mid-training scores are console-only, so
every reported number keeps its meaning and stays comparable to earlier runs.
"""
import math
import os
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from data.datasets import gaussian_blur, pil_jpg
from experiments import common as C


# --------------------------------------------------------------------------- #
# degradation grid
# --------------------------------------------------------------------------- #
JPEG_LEVELS = ['lossless', 15, 30, 45, 60, 75, 100]
BLUR_LEVELS = [('none', 0.0), ('weak', 0.5), ('medium', 1.5), ('strong', 3.0)]
BLUR_METHOD = 'gaussian(scipy.ndimage.gaussian_filter, truncate=4.0)'
# Held out of training; the only cells the 'unseen' condition evaluates on.
HELDOUT_CELLS = [(30, 'medium'), (75, 'weak'), (15, 'strong'), (100, 'medium')]

N_BIN, N_JQ, N_BL = 3, len(JPEG_LEVELS), len(BLUR_LEVELS)
N_OUT_SEV = N_BIN + N_JQ + N_BL
SL_BIN, SL_JQ, SL_BL = slice(0, N_BIN), slice(N_BIN, N_BIN + N_JQ), \
    slice(N_BIN + N_JQ, N_OUT_SEV)


def n_out(sev_weight):
    """How many logits the model gets: 3 (the three asked-for probabilities) or,
    with the optional severity heads on, 3 + 7 + 4."""
    return N_OUT_SEV if sev_weight > 0 else N_BIN

# Severity rank of a quality level (0 = untouched, 6 = worst), used for the
# ordinal correlation. JPEG_LEVELS' own order is not monotone in severity.
_SEVERITY_RANK = {0: 0, 6: 1, 5: 2, 4: 3, 3: 4, 2: 5, 1: 6}
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')

# The default config is MLEP's own 2x2 windows over the 3-scale pyramid, i.e. the
# released front-end (experiment_windows.CONFIGS['baseline_2x2'], the hardcoded
# 2x2 Shannon fast path, 9 input channels). This experiment is about what that
# model can predict, not about changing it, so nothing here varies the window size
# or the channel layout -- every config is 2x2 over the same pyramid.
#
# deg_renyi4_2x2 is the one deliberate exception and is NOT run by default (pass
# --configs to select it). The entropy comparison found Renyi a=4 worth +0.018
# blur ROC-AUC over Shannon, in all three seeds and monotone in a; blur is a
# side-effect there but an explicit target here, so this is where that reading
# gets tested. Same windows, same 9 channels, same everything else -- only the
# functional applied to the window's value distribution differs, and since the
# 2x2 lookup table (ResNet._entropy_2x2_table) it costs the same to train.
_PYR = [1.0, 0.5, 0.25]
DEGRADATION_CONFIGS = {
    'deg_baseline_2x2': dict(window_sizes=[2], scales=_PYR),
    'deg_renyi4_2x2':   dict(window_sizes=[2], scales=_PYR, entropy_mode='renyi_4'),
}
# Only this one runs unless --configs says otherwise, so the default path is
# unchanged and stays a single training run.
DEFAULT_DEGRADATION_CONFIGS = ['deg_baseline_2x2']


def all_cells():
    return [(q, b) for q in JPEG_LEVELS for b, _ in BLUR_LEVELS]


def train_cells():
    return [c for c in all_cells() if c not in HELDOUT_CELLS]


CLEAN_CELL = ('lossless', 'none')


def train_cells_for(args):
    """The training cell list, with the untouched cell optionally repeated.

    train_cells() is 24 cells of which exactly one is untouched, so a source
    lands on a clean image 1/24 of the time and the blur / jpeg heads train on
    71% / 83% positives -- which is what the blur head's false positives on the
    'clean' condition and both heads' all-positive collapse on 'unseen' are made
    of. cell_of() picks with crc32(path) % len(cells), so repeating an entry
    simply raises its frequency; the hashing tests/test_degradation_dataset.py
    pins is untouched.

    Only the TRAINING sets use this. CONDITIONS and every val_set keep the plain
    grid, so no reported number changes meaning and results stay comparable to
    earlier runs."""
    extra = max(0, getattr(args, 'deg_clean_oversample', 0))
    return train_cells() + [CLEAN_CELL] * extra


CONDITIONS = {
    # name -> cells evaluated. 'seen' / 'unseen' are the headline pair.
    'clean':     [('lossless', 'none')],
    'jpeg_only': [(q, 'none') for q in JPEG_LEVELS if q != 'lossless'],
    'blur_only': [('lossless', b) for b, _ in BLUR_LEVELS if b != 'none'],
    'both':      [c for c in train_cells()
                  if c[0] != 'lossless' and c[1] != 'none'],
    'seen':      train_cells(),
    'unseen':    HELDOUT_CELLS,
}


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class DegradationDataset(torch.utils.data.Dataset):
    """ImageFolder-style scan + deterministic degradation, with metadata.

    Mirrors data.datasets.binary_dataset's transform chain (resize -> degrade ->
    crop -> flip -> ToTensor -> Normalize) so the images the network sees here
    are the same kind it sees everywhere else in the project; the difference is
    that the degradation is chosen per sample from a fixed grid instead of
    sampled at random, and is reported alongside the image."""

    def __init__(self, root, classes, cells, variants, args, is_train, seed=0):
        self.samples = []
        for cls in sorted(classes):
            for lab, ai in (('0_real', 0), ('1_fake', 1)):
                d = os.path.join(root, cls, lab)
                if not os.path.isdir(d):
                    continue
                for f in sorted(os.listdir(d)):
                    if f.lower().endswith(IMG_EXT):
                        self.samples.append((os.path.join(d, f), ai, cls))
        if not self.samples:
            raise SystemExit(f"no images found under {root}")
        self.cells = list(cells)
        self.variants = max(1, min(variants, len(self.cells)))
        self.seed = seed
        self.is_train = is_train
        self.split = os.path.basename(os.path.normpath(root))
        self.resize = transforms.Resize((args.loadSize, args.loadSize))
        self.crop = (transforms.RandomCrop(args.cropSize) if is_train
                     else transforms.CenterCrop(args.cropSize))
        self.flip = (transforms.RandomHorizontalFlip() if is_train
                     else transforms.Lambda(lambda im: im))
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.samples) * self.variants

    def cell_of(self, index):
        """Deterministic (quality, blur) for a flat index. crc32 rather than
        hash(): hash() is salted per process, so the same index would land on a
        different cell in every run and the metadata would not reproduce."""
        src, v = divmod(index, self.variants)
        path = self.samples[src][0]
        k = (zlib.crc32(path.encode()) + v * 7919 + self.seed) % len(self.cells)
        return src, v, self.cells[k]

    def meta(self, index):
        src, v, (qual, blur) = self.cell_of(index)
        path, ai, cls = self.samples[src]
        sigma = dict(BLUR_LEVELS)[blur]
        radius = int(4.0 * sigma + 0.5)
        return dict(
            source_id=path, source_index=src, variant=v, split=self.split,
            label='fake' if ai else 'real', ai_label=ai, cls=cls,
            generator=('progan' if ai else 'real'),
            jpeg_quality=qual, blur_level=blur, blur_method=BLUR_METHOD,
            blur_kernel=(2 * radius + 1) if sigma > 0 else 0, blur_sigma=sigma,
            both=int(qual != 'lossless' and blur != 'none'), seed=self.seed)

    def targets(self, index):
        _, _, (qual, blur) = self.cell_of(index)
        ai = self.samples[index // self.variants][1]
        jq = JPEG_LEVELS.index(qual)
        bl = [b for b, _ in BLUR_LEVELS].index(blur)
        return [float(ai), float(bl > 0), float(qual != 'lossless'),
                float(jq), float(bl), float(index)]

    def __getitem__(self, index):
        src, _, (qual, blur) = self.cell_of(index)
        path = self.samples[src][0]
        with Image.open(path) as im:
            img = self.resize(im.convert('RGB'))
        sigma = dict(BLUR_LEVELS)[blur]
        if sigma > 0 or qual != 'lossless':
            arr = np.array(img)
            if sigma > 0:
                gaussian_blur(arr, sigma)          # in-place, per channel
            if qual != 'lossless':
                arr = pil_jpg(arr, qual)
            img = Image.fromarray(arr)
        img = self.flip(self.crop(img))
        return self.norm(self.to_tensor(img)), torch.tensor(self.targets(index))


# --------------------------------------------------------------------------- #
# train / evaluate
# --------------------------------------------------------------------------- #
def multihead_loss(out, tgt, sev_weight):
    bce = nn.functional.binary_cross_entropy_with_logits(out[:, SL_BIN], tgt[:, :3])
    if sev_weight <= 0 or out.shape[1] == N_BIN:
        return bce                      # three logits: BCE only, as in run_config
    ce_q = nn.functional.cross_entropy(out[:, SL_JQ], tgt[:, 3].long())
    ce_b = nn.functional.cross_entropy(out[:, SL_BL], tgt[:, 4].long())
    return bce + sev_weight * (ce_q + ce_b)


@torch.no_grad()
def evaluate_heads(model, loader, device, max_batches, use_amp=False):
    """Collect the three probabilities (and the severity argmaxes if the model
    has those heads) for up to max_batches batches."""
    from experiment_windows import amp_autocast
    model.eval()
    probs, jq, bl, tgts = [], [], [], []
    for b, (img, t) in enumerate(loader):
        if b >= max_batches:
            break
        with amp_autocast(use_amp):
            out = model(img.to(device, non_blocking=True)).float()
        probs.append(torch.sigmoid(out[:, SL_BIN]).cpu().numpy())
        if out.shape[1] > N_BIN:
            jq.append(out[:, SL_JQ].argmax(1).cpu().numpy())
            bl.append(out[:, SL_BL].argmax(1).cpu().numpy())
        tgts.append(t.numpy())
    if not probs:
        return None
    return dict(prob=np.concatenate(probs), tgt=np.concatenate(tgts),
                jq=np.concatenate(jq) if jq else None,
                bl=np.concatenate(bl) if bl else None)


HEADS = ('ai', 'blur', 'jpeg')


def condition_metrics(ev):
    """Binary metrics per head, plus the severity metrics when the model has the
    severity heads. Without them the returned dict simply has no 'jq_*'/'bl_*'
    keys, and every consumer below checks before printing."""
    out = {'n': int(len(ev['tgt']))}
    for i, h in enumerate(HEADS):
        out[h] = C.binary_metrics(ev['tgt'][:, i], ev['prob'][:, i])
        out[h + '_cm'] = C.confusion(ev['tgt'][:, i].astype(int),
                                     (ev['prob'][:, i] > 0.5).astype(int), 2)
    if ev.get('jq') is None:
        return out

    jq_true, bl_true = ev['tgt'][:, 3].astype(int), ev['tgt'][:, 4].astype(int)
    jq_pred, bl_pred = ev['jq'].astype(int), ev['bl'].astype(int)
    out['jq_cm'] = C.confusion(jq_true, jq_pred, N_JQ)
    out['bl_cm'] = C.confusion(bl_true, bl_pred, N_BL)

    # MAE / +-15 accuracy are in QUALITY POINTS, so only samples that actually
    # have a quality (i.e. not the untouched lossless condition) can contribute.
    comp = jq_true > 0
    qv = np.array([0] + JPEG_LEVELS[1:], dtype=float)
    out['jq_exact'] = float((jq_pred == jq_true).mean())
    if comp.any():
        err = np.abs(qv[jq_pred[comp]] - qv[jq_true[comp]])
        out['jq_mae'] = float(err.mean())
        out['jq_within15'] = float((err <= 15).mean())
    else:
        out['jq_mae'] = out['jq_within15'] = float('nan')
    out['bl_acc'] = float((bl_pred == bl_true).mean())
    out['jq_spearman'] = C.spearman([_SEVERITY_RANK[v] for v in jq_true],
                                    [_SEVERITY_RANK[v] for v in jq_pred])
    out['bl_spearman'] = C.spearman(bl_true, bl_pred)
    return out


def warm_start(model, path, name):
    """Load every tensor except the fc1.* head from a checkpoint.

    pretrained/model_epoch_best.pth is this exact network -- conv1 (64, 9, 3, 3),
    layer1/layer2 only, fc1 (1, 512) -- so the released MLEP detector drops
    straight in and only the head has to change shape. A 1-output fc1 seeds
    output 0 (ai), which is the head it was trained to be."""
    sd = torch.load(path, map_location='cpu', weights_only=True)
    sd = sd.get('model', sd) if isinstance(sd, dict) and 'model' in sd else sd
    head = {k: v for k, v in sd.items() if k.startswith('fc1.')}
    body = {k: v for k, v in sd.items() if not k.startswith('fc1.')}
    missing, unexpected = model.load_state_dict(body, strict=False)
    if unexpected:
        raise SystemExit(f"--init_from {path}: {len(unexpected)} tensor(s) the model "
                         f"has no home for, e.g. {unexpected[:4]}. Wrong front-end "
                         f"config for this checkpoint?")
    stray = [k for k in missing if not k.startswith('fc1.')]
    if stray:
        raise SystemExit(f"--init_from {path}: {len(stray)} tensor(s) missing from the "
                         f"checkpoint, e.g. {stray[:4]}")
    seeded = ''
    w = head.get('fc1.weight')
    if w is not None and w.shape[0] == 1 and w.shape[1] == model.fc1.weight.shape[1]:
        with torch.no_grad():
            model.fc1.weight[0].copy_(w[0].to(model.fc1.weight.device))
            model.fc1.bias[0].copy_(head['fc1.bias'][0].to(model.fc1.bias.device))
        seeded = ', ai logit seeded from its 1-output head'
    print(f"  [{name}] warm-started from {path}: {len(body)} tensor(s) loaded, "
          f"fc1 re-initialised{seeded}")


def ckpt_path(out, name, tag=''):
    """results/degradation_<stamp>_seed100.txt -> ..._<config><tag>.pt

    Same <stem>_<suffix>.<ext> shape as C.csv_path, so checkpoints join the
    existing results family instead of starting a new layout."""
    stem = out[:-4] if out.endswith('.txt') else out
    return f"{stem}_{name}{tag}.pt"


def save_degradation_ckpt(path, model, cfg, args, step, n_in):
    """Everything build_model needs to rebuild this model, plus the preprocessing
    it was trained under. Written with the tmp + os.replace discipline the report
    and the CSVs already use, and never over an existing file."""
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite the existing checkpoint {path}")
    payload = dict(
        # 'model' / 'total_steps' are the key names networks/base_model.py's
        # load_networks already expects of a dict payload.
        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
        total_steps=step,
        experiment='mlep_degradation',
        arch=args.arch,
        cfg=cfg,                       # resolved, includes num_classes
        num_classes=cfg['num_classes'],
        in_channels=n_in,
        heads=list(HEADS),
        jpeg_levels=list(JPEG_LEVELS), blur_levels=list(BLUR_LEVELS),
        bn_recalibrated=bool(args.bn_recal_batches > 0),
        loadSize=args.loadSize, cropSize=args.cropSize,
        norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225],
        args=vars(args),
    )
    torch.save(payload, path + '.tmp')
    os.replace(path + '.tmp', path)
    return os.path.getsize(path)


def load_degradation_model(path, device):
    """Round trip of save_degradation_ckpt. Returns (model, checkpoint dict).

    The weights were saved after BN recalibration, so the model is ready to
    evaluate as-is. Preprocessing is Resize(loadSize) -> CenterCrop(cropSize) ->
    ToTensor -> Normalize(norm_mean, norm_std), all recorded in the payload."""
    from experiment_windows import build_model
    ck = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(ck['arch'], ck['cfg'], device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    return model, ck


def lr_lambda(args):
    """Multiplier on --lr at a given step, or None when the LR is constant.

    Returning None rather than a constant-1.0 lambda is deliberate: LambdaLR's
    constructor calls step() and rewrites param_group['lr'] even at factor 1.0,
    so the default path must not build one at all."""
    if args.lr_schedule == 'none' and args.warmup_steps <= 0:
        return None
    warm, total = max(0, args.warmup_steps), max(1, args.max_train_steps)
    floor = (args.lr_min / args.lr) if args.lr > 0 else 0.0

    def f(step):
        if step < warm:
            return (step + 1) / warm
        if args.lr_schedule != 'cosine':
            return 1.0
        frac = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac))
    return f


def _mid_train_eval(model, name, step, args, device, use_amp, val_set, recal_loader):
    """One condition, scored the only way that means anything mid-training.

    recalibrate_bn's docstring measures what skipping this does: the same weights
    read acc 0.4938 / AP 0.8057 on training-EMA BN stats and 0.9762 / 0.9985 after
    recalibration. Ranking metrics move too, so a best-checkpoint choice made
    without it is noise. The buffers are snapshotted and put back afterwards, so
    training resumes from the true EMA and only the eval sees the recal stats."""
    from experiment_windows import make_loader, recalibrate_bn
    snap = {k: v.detach().clone() for k, v in model.state_dict().items()
            if 'running_' in k or 'num_batches_tracked' in k}
    if recal_loader is not None:
        recalibrate_bn(model, recal_loader, args.deg_val_recal, device, use_amp=use_amp)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = make_loader(val_set, args, shuffle=True, generator=g, persistent=False)
    ev = evaluate_heads(model, loader, device, args.max_val_batches, use_amp)
    del loader
    model.load_state_dict(snap, strict=False)
    model.train()
    if ev is None:
        return None, float('nan')
    m = condition_metrics(ev)
    aucs = [m[h]['roc_auc'] for h in HEADS]
    ok = [v for v in aucs if v == v]
    return m, (sum(ok) / len(ok) if ok else float('nan'))


def run_degradation_config(name, cfg, args, device, use_amp, val_sets):
    """One config: train the multi-head model, then evaluate every condition.

    Deliberately the same shape as experiment_windows.run_config -- same fixed
    step count, same log lines, same BN recalibration before eval -- with the
    multi-head loss as the only difference."""
    from experiment_windows import build_model, make_loader, recalibrate_bn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_full = dict(cfg, num_classes=n_out(args.deg_sev_weight))
    model = build_model(args.arch, cfg_full, device)
    n_in = model.conv1.in_channels
    n_params = sum(p.numel() for p in model.parameters())
    if args.init_from:
        warm_start(model, args.init_from, name)

    cells = train_cells_for(args)
    train_set = DegradationDataset(
        os.path.join(args.dataroot, args.train_split), _classes(args, args.train_split),
        cells, args.deg_variants, args, True, seed=args.seed)
    train_loader = make_loader(train_set, args, shuffle=True, drop_last=True)
    print(f"  [{name}] train set: {len(train_set)} samples "
          f"({len(train_set.samples)} sources x {train_set.variants} variants "
          f"over {len(cells)} cells)  in_ch={n_in}")

    if args.weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(0.9, 0.999),
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    lam = lr_lambda(args)
    sched = (None if lam is None
             else torch.optim.lr_scheduler.LambdaLR(optimizer, lam))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    from experiment_windows import amp_autocast

    # Mid-training evaluation needs a train loader of its own for the throwaway
    # BN recalibration; built once and reused so the cost is per-run, not per-eval.
    mid_recal_loader = None
    if args.deg_val_every > 0 and args.deg_val_recal > 0:
        mid_recal_loader = make_loader(
            DegradationDataset(
                os.path.join(args.dataroot, args.train_split),
                _classes(args, args.train_split), cells, args.deg_variants,
                args, True, seed=args.seed),
            args, shuffle=True, drop_last=True, persistent=False)
    if args.deg_val_every > 0 and args.deg_val_cond not in val_sets:
        raise SystemExit(f"--deg_val_cond {args.deg_val_cond!r} is not one of "
                         f"{list(val_sets)}")
    best = dict(score=float('-inf'), step=0, state=None)

    model.train()
    step, t0, done = 0, time.time(), False
    for epoch in range(10 ** 6):
        for img, tgt in train_loader:
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True).float()
            with amp_autocast(use_amp):
                loss = multihead_loss(model(img), tgt, args.deg_sev_weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if sched is not None:
                sched.step()
            step += 1
            if step % args.log_freq == 0:
                print(f"  [{name}] step {step}/{args.max_train_steps}  "
                      f"loss={loss.item():.4f}  ({time.time() - t0:.0f}s)")
            if (args.deg_val_every > 0 and step % args.deg_val_every == 0
                    and step < args.max_train_steps):
                m, score = _mid_train_eval(model, name, step, args, device, use_amp,
                                           val_sets[args.deg_val_cond], mid_recal_loader)
                if m is not None:
                    # same fields as the end-of-run per-condition line, under a
                    # val/ prefix -- console only, the report is untouched.
                    print(f"  [{name}] step {step}/{args.max_train_steps}  "
                          f"val/{args.deg_val_cond:9s}: "
                          f"ai_auc={m['ai']['roc_auc']:.4f} "
                          f"blur_auc={m['blur']['roc_auc']:.4f} "
                          f"jpeg_auc={m['jpeg']['roc_auc']:.4f} (n={m['n']})  "
                          f"mean_auc={score:.4f}")
                if score == score and score > best['score']:
                    best = dict(score=score, step=step, state=(
                        {k: v.detach().clone() for k, v in model.state_dict().items()}
                        if args.deg_save_ckpt else None))
            if step >= args.max_train_steps:
                done = True
                break
        if done:
            break
    train_time = time.time() - t0
    print(f"  [{name}] trained {step} steps in {train_time:.0f}s "
          f"(in_ch={n_in}, params={n_params / 1e6:.2f}M)")
    del train_loader, mid_recal_loader

    if args.bn_recal_batches > 0:
        recal_set = DegradationDataset(
            os.path.join(args.dataroot, args.train_split),
            _classes(args, args.train_split), cells, args.deg_variants,
            args, True, seed=args.seed)
        recal_loader = make_loader(recal_set, args, shuffle=True, drop_last=True,
                                   persistent=False)
        t_recal = time.time()
        seen = recalibrate_bn(model, recal_loader, args.bn_recal_batches, device,
                              use_amp=use_amp)
        print(f"  [{name}] BN recalibrated on {seen} train batches "
              f"({time.time() - t_recal:.0f}s)")

    # Saved after BN recalibration on purpose: the file then carries usable
    # running stats and loads ready to evaluate.
    if args.deg_save_ckpt:
        p = ckpt_path(args.out, name)
        size = save_degradation_ckpt(p, model, cfg_full, args, step, n_in)
        print(f"  [{name}] checkpoint written to {p} ({size / 1e6:.1f} MB)")
        if best['state'] is not None:
            # Swap in the best mid-training weights just long enough to give them
            # their own recalibration and file, then put the final model back so
            # the evaluation below -- and therefore the report -- is unchanged.
            final_state = {k: v.detach().clone()
                           for k, v in model.state_dict().items()}
            model.load_state_dict(best['state'])
            if args.bn_recal_batches > 0:
                recalibrate_bn(model, recal_loader, args.bn_recal_batches, device,
                               use_amp=use_amp)
            pb = ckpt_path(args.out, name, '_best')
            size = save_degradation_ckpt(pb, model, cfg_full, args, best['step'], n_in)
            print(f"  [{name}] best checkpoint (step {best['step']}, "
                  f"mean_auc={best['score']:.4f}) written to {pb} "
                  f"({size / 1e6:.1f} MB)")
            model.load_state_dict(final_state)

    if args.bn_recal_batches > 0:
        del recal_loader, recal_set

    conditions, preds = {}, []
    for cname, dset in val_sets.items():
        g = torch.Generator()
        g.manual_seed(args.seed)
        loader = make_loader(dset, args, shuffle=True, generator=g, persistent=False)
        ev = evaluate_heads(model, loader, device, args.max_val_batches, use_amp)
        del loader
        if ev is None:
            continue
        m = condition_metrics(ev)
        conditions[cname] = m
        sev = ('' if 'jq_exact' not in m else
               f" [jq_exact={m['jq_exact']:.4f} jq_mae={m['jq_mae']:.1f} "
               f"bl_acc={m['bl_acc']:.4f}]")
        print(f"  [{name}] test/{cname:9s}: "
              f"ai_auc={m['ai']['roc_auc']:.4f} blur_auc={m['blur']['roc_auc']:.4f} "
              f"jpeg_auc={m['jpeg']['roc_auc']:.4f} (n={m['n']})" + sev)
        preds.extend(_prediction_rows(name, cfg, cname, dset, ev))

    return dict(name=name, cfg=cfg, in_ch=n_in, params=n_params, time=train_time,
                steps=step, conditions=conditions, preds=preds)


def _prediction_rows(name, cfg, cond, dset, ev):
    rows = []
    for k in range(len(ev['tgt'])):
        md = dset.meta(int(ev['tgt'][k, 5]))
        rows.append([
            name, str(cfg), cond, md['source_id'], md['split'], md['label'],
            md['generator'], md['cls'], md['jpeg_quality'], md['blur_level'],
            md['blur_method'], md['blur_kernel'], md['blur_sigma'], md['both'],
            md['seed'],
            float(ev['prob'][k, 0]), float(ev['prob'][k, 1]), float(ev['prob'][k, 2]),
            # empty unless the optional severity heads are on
            '' if ev['jq'] is None else JPEG_LEVELS[int(ev['jq'][k])],
            '' if ev['bl'] is None else BLUR_LEVELS[int(ev['bl'][k])][0],
        ])
    return rows


PRED_HEADER = ['config', 'mlep_config', 'condition', 'source_id', 'split', 'label',
               'generator', 'class', 'jpeg_quality', 'blur_level', 'blur_method',
               'blur_kernel', 'blur_sigma', 'both_applied', 'seed',
               'ai_generated_probability', 'blur_probability',
               'jpeg_compression_probability', 'predicted_jpeg_quality',
               'predicted_blur_level']


def _classes(args, split):
    root = os.path.join(args.dataroot, split)
    return (args.classes.split(',') if args.classes
            else sorted(d for d in os.listdir(root)
                        if os.path.isdir(os.path.join(root, d))))


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_degradation_report(results, args, device, gpu_name, use_amp, total):
    def key(r):
        v = r['conditions'].get('seen', {}).get('ai', {}).get('roc_auc', float('nan'))
        return v if v == v else -1

    results = sorted(results, key=key, reverse=True)
    done = len(results)
    conds = list(dict.fromkeys(c for r in results for c in r['conditions']))
    # severity heads are optional; with them off the model emits exactly the three
    # asked-for probabilities and the severity sections are simply absent.
    sev = any('jq_exact' in m for r in results for m in r['conditions'].values())

    lines = C.report_header("MLEP degradation prediction", args, device, gpu_name,
                            use_amp, done, total,
                            extra=f"variants={args.deg_variants}  "
                                  f"sev_weight={args.deg_sev_weight}  "
                                  f"outputs={n_out(args.deg_sev_weight)}")
    setup = next((r['setup'] for r in results if r.get('setup')), [])
    if setup:
        lines.extend(setup)
    lines.append(f"grid: jpeg={JPEG_LEVELS} blur={[b for b, _ in BLUR_LEVELS]} "
                 f"sigma={[s for _, s in BLUR_LEVELS]}")
    lines.append(f"blur: {BLUR_METHOD}; kernel = 2*int(4*sigma+0.5)+1")
    lines.append(f"held out of training (the 'unseen' condition): {HELDOUT_CELLS}")
    lines.append("model: MLEP's released 2x2 entropy front-end; the only change is "
                 + (f"{n_out(args.deg_sev_weight)} output logits "
                    "(3 probabilities + the optional severity heads)" if sev else
                    "3 output logits instead of 1 (ai / blur / jpeg)"))
    # Only present when the run deviates from the stock recipe, so the reports of
    # every earlier run keep their exact shape. A long run's checkpoint is a
    # deliverable, though, so the recipe that produced it has to be on the record.
    recipe = []
    if args.init_from:
        recipe.append(f"warm-started from {args.init_from} (fc1 re-initialised)")
    if args.lr_schedule != 'none' or args.warmup_steps > 0:
        recipe.append(f"lr {args.lr:g} {args.lr_schedule}"
                      + (f" after {args.warmup_steps} warmup steps"
                         if args.warmup_steps > 0 else "")
                      + (f", floor {args.lr_min:g}"
                         if args.lr_schedule == 'cosine' else ""))
    if args.weight_decay > 0:
        recipe.append(f"AdamW weight_decay={args.weight_decay:g}")
    if args.deg_clean_oversample > 0:
        n_cells = len(train_cells()) + args.deg_clean_oversample
        recipe.append(f"training cell list is the {len(train_cells())}-cell grid with "
                      f"{CLEAN_CELL} repeated {args.deg_clean_oversample}x "
                      f"({n_cells} entries, {1 + args.deg_clean_oversample} of them "
                      f"untouched); evaluation cells unchanged")
    if recipe:
        lines.append("training: " + "; ".join(recipe))
    lines.append("")

    lines.append("Per-model results")
    lines.append("-" * 60)
    for r in results:
        lines.append(f"[{r['name']}]  {r['cfg']}")
        lines.append(f"    in_channels={r['in_ch']}  params={r['params'] / 1e6:.2f}M  "
                     f"steps={r['steps']}  train_time={r['time']:.0f}s")
        for c in conds:
            m = r['conditions'].get(c)
            if m is None:
                continue
            for h in HEADS:
                b = m[h]
                # 'acc=' / 'ap=' / '(n=)' prefix kept so aggregate_results.py can
                # read these files too; the head name rides in the scenario field.
                lines.append(
                    f"    test/{c + '/' + h:14s}  acc={C.fmt(b['acc'])}  "
                    f"ap={C.fmt(b['pr_auc'])}  (n={b['n']})  "
                    f"roc_auc={C.fmt(b['roc_auc'])}  prec={C.fmt(b['precision'])}  "
                    f"rec={C.fmt(b['recall'])}  f1={C.fmt(b['f1'])}  "
                    f"eer={C.fmt(b['eer'])}  brier={C.fmt(b['brier'])}  "
                    f"ece={C.fmt(b['ece'])}")
            if 'jq_exact' in m:
                lines.append(
                    f"    severity/{c:10s}  jq_exact={C.fmt(m['jq_exact'])}  "
                    f"jq_mae={C.fmt(m['jq_mae'], 2)}  "
                    f"jq_within15={C.fmt(m['jq_within15'])}  "
                    f"bl_acc={C.fmt(m['bl_acc'])}  "
                    f"jq_rho={C.fmt(m['jq_spearman'])}  "
                    f"bl_rho={C.fmt(m['bl_spearman'])}")
        # Confusion matrices for the headline condition only, to keep the file
        # readable; every condition's matrices are in the *_confusions.csv.
        for c in ('seen', 'unseen'):
            m = r['conditions'].get(c)
            if m is None:
                continue
            if 'jq_cm' in m:
                lines.append(f"    confusion [{c}] jpeg quality (rows=true):")
                C.confusion_text(lines, m['jq_cm'], JPEG_LEVELS, indent='      ')
                lines.append(f"    confusion [{c}] blur level (rows=true):")
                C.confusion_text(lines, m['bl_cm'], [b for b, _ in BLUR_LEVELS],
                                 indent='      ')
            for h in HEADS:
                lines.append(f"    confusion [{c}] {h} (rows=true):")
                C.confusion_text(lines, m[h + '_cm'], ['no', 'yes'], indent='      ')
        lines.append("")

    if results:
        for h in HEADS:
            rows = [[r['name']] + [C.fmt(r['conditions'][c][h]['roc_auc'])
                                   if c in r['conditions'] else '-' for c in conds]
                    for r in results]
            lines.append(f"ROC-AUC of the {h} head per condition")
            C.table(lines, ['config'] + conds, rows)
            lines.append("")
        if sev:
            rows = [[r['name']] + [C.fmt(r['conditions'][c]['jq_exact'])
                                   if c in r['conditions'] else '-' for c in conds]
                    + [C.fmt(r['conditions'].get('seen', {})
                             .get('jq_mae', float('nan')), 2)]
                    for r in results]
            lines.append("Exact JPEG-quality accuracy per condition (+ seen MAE)")
            C.table(lines, ['config'] + conds + ['seen_mae'], rows,
                    note="'unseen' holds the four (quality, blur) cells never "
                         "trained on, so its columns are the unseen-combination "
                         "result.")
            lines.append("")

    lines.append("Notes")
    lines.append("- Untouched images are the only 'not compressed' condition; "
                 "quality 100 is a compressed image at its own severity level.")
    if sev:
        lines.append("- MAE and +-15 accuracy are over samples that have a quality, "
                     "i.e. excluding the untouched condition.")
    else:
        lines.append("- Severity heads are off (--deg_sev_weight 0), so the model "
                     "predicts only the three probabilities.")
    lines.append("- Every variant of a source image stays inside that source's "
                 "split; cells are assigned by crc32(path), so they reproduce.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run_degradation_experiment(args, device, gpu_name, use_amp):
    from experiment_windows import write_report

    names = (args.configs.split(',') if args.configs
             else list(DEFAULT_DEGRADATION_CONFIGS))
    unknown = [n for n in names if n not in DEGRADATION_CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}. "
                         f"Available: {list(DEGRADATION_CONFIGS)}")

    # Everything below this point scans the split folders (tens of thousands of
    # files) before a single step is trained, so the run identifies itself first
    # rather than sitting silent through the scans.
    print(f"Device: {device}" + (f" ({gpu_name})" if gpu_name else "") +
          f" | arch: {args.arch} | steps: {args.max_train_steps} "
          f"| crop: {args.cropSize} | batch: {args.batch_size} "
          f"| workers: {args.num_threads} | amp: {use_amp}")
    print(f"Degradation experiment: {len(names)} config(s): {names}")
    print(f"  grid={len(all_cells())} cells ({len(train_cells())} trained, "
          f"{len(HELDOUT_CELLS)} held out)  variants/image={args.deg_variants}"
          + (f"  clean cell repeated {args.deg_clean_oversample}x in training"
             if args.deg_clean_oversample > 0 else ""))

    t_scan = time.time()
    setup = C.check_split_disjoint(args.dataroot, args.train_split, args.val_split,
                                   args.classes.split(',') if args.classes else None)
    val_root = os.path.join(args.dataroot, args.val_split)
    val_classes = _classes(args, args.val_split)
    val_sets = {c: DegradationDataset(val_root, val_classes, cells, args.deg_variants,
                                      args, False, seed=args.seed)
                for c, cells in CONDITIONS.items()}
    if args.unseen_split:
        u_root = os.path.join(args.dataroot, args.unseen_split)
        if os.path.isdir(u_root):
            val_sets['unseen_gen'] = DegradationDataset(
                u_root, _classes(args, args.unseen_split), CONDITIONS['seen'],
                args.deg_variants, args, False, seed=args.seed)

    # Leakage assertion, not just a report line: no source file may appear in
    # both the training set and any evaluation set.
    train_srcs = {p for p, _, _ in DegradationDataset(
        os.path.join(args.dataroot, args.train_split), _classes(args, args.train_split),
        train_cells_for(args), 1, args, True, seed=args.seed).samples}
    for cname, ds in val_sets.items():
        clash = train_srcs.intersection(p for p, _, _ in ds.samples)
        if clash:
            raise SystemExit(f"source-image leakage: {len(clash)} image(s) are in "
                             f"both train and the '{cname}' evaluation set")
    setup.append(f"leakage check: {len(train_srcs)} train sources, none present in "
                 f"any of the {len(val_sets)} evaluation sets -- ok")

    for line in setup:
        print(f"  {line}")
    print(f"  scanned {len(train_srcs)} train sources and "
          f"{len(val_sets)} evaluation sets "
          f"({', '.join(f'{c}:{len(d)}' for c, d in val_sets.items())}) "
          f"in {time.time() - t_scan:.0f}s")
    print()

    results, metric_rows, robust_rows, conf_rows, pred_rows = [], [], [], [], []
    for i, name in enumerate(names, 1):
        cfg = DEGRADATION_CONFIGS[name]
        print(f"=== [{i}/{len(names)}] {name} : {cfg} ===")
        r = run_degradation_config(name, cfg, args, device, use_amp, val_sets)
        r['setup'] = setup if i == 1 else []
        pred_rows.extend(r.pop('preds'))
        results.append(r)
        for cname, m in r['conditions'].items():
            for h in HEADS:
                metric_rows.append([name, cname, h] + [m[h][k] for k in _MK])
            # severity metrics are not binary, so they live in *_robustness.csv
            robust_rows.extend(_robust_rows(name, cname, m))
            conf_rows.extend(_conf_rows(name, cname, m))
        write_report(results, args, device, gpu_name, use_amp, len(names),
                     builder=build_degradation_report)
        print(f"  [{name}] done -- {i}/{len(names)} written to {args.out}\n")

    C.write_csv(C.csv_path(args.out, 'predictions'), PRED_HEADER, pred_rows)
    C.write_csv(C.csv_path(args.out, 'metrics'),
                ['config', 'condition', 'head'] + _MK, metric_rows)
    C.write_csv(C.csv_path(args.out, 'robustness'),
                ['config', 'condition', 'metric', 'value'], robust_rows)
    C.write_csv(C.csv_path(args.out, 'confusions'),
                ['config', 'condition', 'matrix', 'true', 'pred', 'count'], conf_rows)

    report = write_report(results, args, device, gpu_name, use_amp, len(names),
                          builder=build_degradation_report)
    print("\n" + report)
    print(f"\nResults written to {args.out}")
    print(f"CSVs: {C.csv_path(args.out, 'predictions')}, "
          f"{C.csv_path(args.out, 'metrics')}, "
          f"{C.csv_path(args.out, 'robustness')}, "
          f"{C.csv_path(args.out, 'confusions')}")


_MK = ['n', 'pos', 'roc_auc', 'pr_auc', 'acc', 'precision', 'recall', 'f1',
       'eer', 'brier', 'ece']


def _robust_rows(name, cond, m):
    rows = [[name, cond, k, m[k]] for k in
            ('jq_exact', 'jq_mae', 'jq_within15', 'bl_acc', 'jq_spearman',
             'bl_spearman') if k in m]
    rows += [[name, cond, f"{h}_roc_auc", m[h]['roc_auc']] for h in HEADS]
    return rows


def _conf_rows(name, cond, m):
    rows = []
    mats = [(m[h + '_cm'], ['no', 'yes'], h) for h in HEADS]
    if 'jq_cm' in m:
        mats = [(m['jq_cm'], JPEG_LEVELS, 'jpeg_quality'),
                (m['bl_cm'], [b for b, _ in BLUR_LEVELS], 'blur_level')] + mats
    for mat, labels, tag in mats:
        for i, t in enumerate(labels):
            for j, p in enumerate(labels):
                rows.append([name, cond, tag, t, p, int(mat[i][j])])
    return rows
