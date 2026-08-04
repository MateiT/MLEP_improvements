"""
Quick experiment harness to compare different local-entropy window settings.

It trains the MLEP network for a *small* number of steps under several
configurations and reports validation accuracy / average precision so you can
see which setting looks most promising before committing to a full training run.
Every config sits on the same 3-scale pyramid, so the axes actually under test
are: sliding-window size (2x2 .. 8x8), one window vs three combined, shannon vs
unique entropy, per-window entropy normalisation, multi-window registration,
rich/poor texture-patch separation, and train-time augmentation.

This is intentionally NOT a full training script -- it uses a small crop size,
a capped number of training steps and only a few validation batches so a whole
sweep finishes in minutes on a laptop (CPU / Apple MPS / single GPU).

Example (Ubuntu + RTX 4090)
---------------------------
    python experiment_windows.py \
        --dataroot /Data/MLEP/datasets/TrainDatasets \
        --classes car,cat,chair,horse \
        --arch resnet50 --batch_size 32 --cropSize 128 \
        --max_train_steps 500 --max_val_batches 20

Every config trains for the SAME number of steps (--max_train_steps) so the
comparison is fair; wall-clock differs only slightly with each config's cost
(more windows/scales -> a bit slower). Keep --max_train_steps modest so a whole
sweep stays quick rather than running a full training.

On CUDA the harness auto-enables TF32, cuDNN autotuning, mixed precision (AMP),
pinned memory and multi-worker data loading, so the 4090 stays fed.

Pick a subset of configs with --configs:
    python experiment_windows.py ... --configs baseline_2x2,multiwindow_multiscale
List available configs with --list_configs.

Two further experiment groups live in experiments/ and are selected with
--experiment (default 'windows' = this sweep, unchanged):

    --experiment entropy            experiments/entropy.py
        Do other entropy definitions (Renyi, Tsallis, permutation, and
        combinations) detect AI images better than Shannon?
    --experiment mlep_degradation   experiments/degradation.py
        Predict AI-vs-real, blurred-or-not and JPEG-compressed-or-not (plus both
        severities) from the same front-end.

They reuse the machinery below -- make_opt/make_loader, build_model, run_config,
recalibrate_bn, evaluate, write_report -- and write the same kind of report file,
so --list_configs, the console output and scripts/aggregate_results.py all work
the same way for them. See experiments/*.py for their configs and metrics.
"""

import os
import sys
import time
import argparse
import contextlib
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score

from data import get_dataset
from networks.resnet import resnet18, resnet50


# --------------------------------------------------------------------------- #
# Configurations to compare. Each is a set of kwargs forwarded to the network.
# Add / remove entries freely -- this is where you experiment.
# --------------------------------------------------------------------------- #
CONFIGS = {
    # The original paper setting (2x2 window, 3-scale pyramid) -> 9 channels.
    'baseline_2x2': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25]),
    # Baseline window/pyramid, but the 'unique' (set-cardinality) feature instead
    # of shannon -> 9 channels. Note 'unique' leaves the fast hardcoded 2x2 path
    # in resnet.py and takes the general sort-based route, so it is slower than
    # baseline_2x2 despite having the same shape.
    'w2x2_unique':         dict(window_sizes=[2],    scales=[1.0, 0.5, 0.25],
                                entropy_mode='unique'),
    # Single 3x3 window, same pyramid.
    'w3x3_multiscale': dict(window_sizes=[3], scales=[1.0, 0.5, 0.25]),
    'w4x4_multiscale': dict(window_sizes=[4], scales=[1.0, 0.5, 0.25]),
    'w5x5_multiscale': dict(window_sizes=[5], scales=[1.0, 0.5, 0.25]),
    'w6x6_multiscale': dict(window_sizes=[6], scales=[1.0, 0.5, 0.25]),
    'w7x7_multiscale': dict(window_sizes=[7], scales=[1.0, 0.5, 0.25]),
    'w8x8_multiscale': dict(window_sizes=[8], scales=[1.0, 0.5, 0.25]),

    # Combine 2x2, 4x4 AND 6x6 windows (the multi-window idea) over the 3-scale
    # pyramid -> 3 windows * 3 scales * 3 RGB = 27 channels.
    'multiwindow_multiscale':     dict(window_sizes=[2, 4, 6],    scales=[1.0, 0.5, 0.25]),

    # The next two are exact twins of multiwindow_multiscale, each changing one
    # thing and nothing else, so the difference is attributable.

    # --- does normalising entropy to [0,1] per window matter? ---
    # Mixing window sizes mixes value ranges: shannon entropy is bounded by
    # log2(w*w), i.e. 2.0 bits for 2x2 but ~5.17 for 6x6, so the wide-window
    # channels enter conv1 with ~2.5x the dynamic range of the narrow ones.
    # conv1 has bias=False and bn1 normalises the 64 OUTPUT channels after the
    # convolution has already summed the inputs, so nothing downstream undoes
    # this. normalize_entropy divides each map by its own log2(K) instead.
    'multiwindow_multiscale_normalized':  dict(window_sizes=[2, 4, 6],
                                               scales=[1.0, 0.5, 0.25],
                                               normalize_entropy=True),

    # --- does exact window registration matter? ---
    # A w-window map is H-w+1 wide, so the three windows disagree on size and the
    # default 'resize' path bilinear-resamples them onto the 2x2 grid. That is
    # doubly lossy. It aligns the maps' outer extents rather than the pixels they
    # actually describe, which STRETCHES them rather than shifting them: measured
    # error is ~0 at the image centre and grows outward to ~1.6px for the 6x6
    # window near the border. A convnet can absorb a constant shift into learned
    # filter positions but not a position-dependent warp, since the weights are
    # shared. On top of that the resampling blurs the high-frequency structure
    # that is the entire signal here. window_align='pad' replicate-pads instead,
    # which is exact and interpolation-free (see the note in ResNet.forward) and
    # measured free: +1 MiB and no extra time vs 'resize'.
    # Covered by tests/test_window_align.py.
    'multiwindow_multiscale_aligned':  dict(window_sizes=[2, 4, 6],
                                            scales=[1.0, 0.5, 0.25],
                                            window_align='pad'),

    # --- train WITH augmentation (does it help robustness to corruptions?) ---
    # Same model as baseline_2x2, but the training images are randomly blurred /
    # JPEG-compressed. The 'train_aug' key is consumed by the harness (not the model).
    'baseline_train_blur': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                train_aug=dict(blur_prob=0.5, blur_sig=[0.0, 3.0])),
    'baseline_train_jpeg': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                train_aug=dict(jpg_prob=0.5, jpg_qual=[30, 100],
                                               jpg_method=['cv2', 'pil'])),

    # --- prob 1.0: train on ONE domain instead of a contradictory mixture ------ #
    # results/texsplit_trainaug_seed100.txt has every corrupted AP cell BELOW 0.5
    # (0.415-0.494, 12/12 across 4 configs) while acc ~ 0.52. Below-chance AP is an
    # INVERTED ranking, not weak performance: under corruption these models score
    # real above fake. Suspected cause is the 0.5 probability itself -- it trains on
    # a 50/50 mix of clean and corrupted images whose class ordering is opposite,
    # and the clean domain is ~0.99 separable vs ~0.5 corrupted, so it dominates the
    # gradient and its ordering gets applied backwards to corrupted input.
    #
    # These two change ONLY the probability, 0.5 -> 1.0. jpeg keeps
    # baseline_train_jpeg's quality/method verbatim so the pair isolates the
    # probability; do not also "fix" jpg_qual here or the comparison confounds two
    # changes at once. webp has no 0.5 predecessor -- it is absent from training
    # altogether today, despite being an eval scenario.
    'baseline_train_jpeg_p1': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                   train_aug=dict(jpg_prob=1.0, jpg_qual=[30, 100],
                                                  jpg_method=['cv2', 'pil'])),
    'baseline_train_webp_p1': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                   train_aug=dict(webp_prob=1.0, webp_qual=[80])),

    # --- joint-colour entropy (cross-channel statistics) --------------------- #
    # First, a correction to the premise this came from: the front-end is NOT
    # grayscale. _entropy_2x2_shannon keeps the channel dim, so baseline_2x2's 9
    # channels are already (R,G,B) x 3 scales, and random_rearrange_blocks applies
    # ONE permutation across all channels (`blocks[i, :, perm]`), not an independent
    # one per channel. The only grayscale step in the repo is _texture_diversity's
    # x.mean(dim=1), which ranks patches for the texture split.
    #
    # What IS genuinely missing is any CROSS-channel statistic: the three maps are
    # independent marginals. A 2x2 window can be maximally entropic in R, in G and
    # in B and still contain only two distinct COLOURS. color_entropy adds the
    # entropy of the joint RGB-triple distribution (two pixels are one symbol only
    # if they agree in every channel), which no combination of the marginals can
    # recover.
    #
    # Why it is worth a run on the robustness problem specifically: JPEG and WebP
    # both convert to YCbCr, subsample chroma 4:2:0 and quantise the chroma planes
    # much harder than luma. So recompression damages cross-channel structure and
    # per-channel structure by DIFFERENT amounts -- a difference a marginal-only
    # front-end has no channel to express. Every corrupted AP in
    # results/prob1_seed100.txt sits at 0.42-0.60, so there is room.
    #
    # 'joint' appends the map -> 4 per (scale, window) = 12 channels, with channels
    # 0..2 of each group bit-identical to baseline_2x2, so it is a clean ADDITIVE
    # test (strictly more input, nothing removed). 'joint_only' feeds the joint map
    # alone -> 3 channels, FEWER than the baseline, which is the ablation that says
    # whether the joint statistic carries signal itself or merely rides along.
    # Read color_joint_2x2 vs baseline_2x2, and the two train_* rows against
    # baseline_train_jpeg / baseline_train_webp_p1 -- augmented training is where the
    # robustness actually lives, so that is the cell where a front-end change has to
    # show up. train_aug dicts are copied verbatim from those configs; keep them
    # identical or the pairs stop being controlled comparisons.
    # Covered by tests/test_color_entropy.py.
    'color_joint_2x2':      dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                 color_entropy='joint'),
    'color_jointonly_2x2':  dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                 color_entropy='joint_only'),
    'color_joint_train_jpeg': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                   color_entropy='joint',
                                   train_aug=dict(jpg_prob=0.5, jpg_qual=[30, 100],
                                                  jpg_method=['cv2', 'pil'])),
    'color_joint_train_webp_p1': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                      color_entropy='joint',
                                      train_aug=dict(webp_prob=1.0, webp_qual=[80])),

    # --- PatchCraft-style rich/poor texture separation ------------------------ #
    # The hypothesis. baseline_2x2 scores clean AP 0.9935 but jpeg 0.4671 /
    # webp 0.4747 -- BELOW CHANCE (results/sweep_20260725_183342_seed100.txt).
    # The front-end counts distinct pixel values, and recompression flattens
    # those counts everywhere at once, so a single whole-image entropy map cannot
    # tell "smooth because the scene is smooth" from "smooth because the codec
    # smoothed it". Sorting patches by texture diversity and handing the network
    # the flat regions and the busy regions as two SEPARATE images lets it read
    # the two populations independently: the poor canvas isolates exactly what a
    # codec destroys, the rich canvas what mostly survives.
    # Adapted from Zhong et al., "Exploring Texture Patch for Efficient
    # AI-generated Image Detection" (PatchCraft).
    #
    # texsplit_2x2 is the head-to-head against baseline_2x2 (18 channels, both
    # canvases stacked). texsplit_2x2_diff feeds (rich - poor) instead, which is
    # what PatchCraft actually uses and keeps 9 channels -- capacity-matched to
    # baseline_2x2, so a win there cannot be explained by conv1 being wider.
    #
    # p=16 is the primary patch size: the patch grid has an even axis at cropSize
    # 128, 224 AND 256, so the split discards no pixels. p=32 (PatchCraft's own
    # size) is lossless at 128/256 but leaves a 7x7 grid at cropSize 224, forcing
    # a one-column trim that throws away 14.3% of the crop -- a useful second data
    # point on seam count, NOT a clean comparison against baseline_2x2 at CROP=224.
    # Covered by tests/test_texture_split.py.
    'texsplit_2x2':       dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                               texture_split=True, texture_patch_size=16),
    'texsplit_2x2_diff':  dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                               texture_split=True, texture_patch_size=16,
                               texture_split_mode='diff'),
    'texsplit_2x2_p32':   dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                               texture_split=True, texture_patch_size=32),

    # The split, trained UNDER corruption. Without these two the split is only
    # ever measured on clean-trained models, which is the wrong cell to look in:
    # baseline_train_jpeg is the only config whose jpeg AP clears chance, and the
    # only one whose logit_mean stays near zero across every scenario. Augmented
    # training is doing the robustness work, so if rich/poor separation helps at
    # all it should show ON TOP of it. Read these as two PAIRED deltas against
    # baseline_train_blur / baseline_train_jpeg, not as four standalone numbers.
    #
    # train_aug is consumed by the harness, texture_split by build_model, so the
    # two compose with no extra plumbing. The dicts below are copied verbatim from
    # the baseline_train_* configs above -- keep them identical or the pairs stop
    # being controlled comparisons.
    #
    # Note jpg_qual=[30, 100] means quality 30 OR 100, not the range between:
    # data_augment picks it with sample_discrete, and make_opt bypasses the
    # base_options range expansion. blur_sig=[0.0, 3.0] IS a true range (it goes
    # through sample_continuous). The two read as parallel but are not.
    'texsplit_2x2_train_blur': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                    texture_split=True, texture_patch_size=16,
                                    train_aug=dict(blur_prob=0.5, blur_sig=[0.0, 3.0])),
    'texsplit_2x2_train_jpeg': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                    texture_split=True, texture_patch_size=16,
                                    train_aug=dict(jpg_prob=0.5, jpg_qual=[30, 100],
                                                   jpg_method=['cv2', 'pil'])),

    # Control for the above: the texture split reassembles patches, which is a
    # spatial scramble in its own right, and so is random_rearrange_blocks. This
    # config isolates the shuffle's own contribution so the two are not confused.
    'baseline_2x2_norearr': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                 use_rearrange=False),
}


# --------------------------------------------------------------------------- #
# Test-time corruption scenarios. Every trained model is evaluated on each of
# these, so we can see how well it holds up on blurred / compressed images.
# prob=1.0 -> the corruption is ALWAYS applied (deterministic robustness test).
# --------------------------------------------------------------------------- #
EVAL_SCENARIOS = {
    'clean':      dict(),
    'blur':       dict(blur_prob=1.0, blur_sig=[2.0]),
    'jpeg':       dict(jpg_prob=1.0, jpg_qual=[75], jpg_method=['pil']),
    'webp':       dict(webp_prob=1.0, webp_qual=[80]),
    'blur+jpeg':  dict(blur_prob=1.0, blur_sig=[2.0],
                       jpg_prob=1.0, jpg_qual=[75], jpg_method=['pil']),
}


def get_device(pref):
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def setup_cuda_perf(device):
    """Turn on the standard 'go fast on an NVIDIA GPU' switches.

    - TF32 for matmul/conv: big throughput win on Ampere/Ada (RTX 4090) at no
      meaningful accuracy cost for this task.
    - cuDNN autotuner: picks the fastest conv kernels. Worth it because within a
      run only a couple of input shapes occur (train crop, val size).
    Returns the GPU name (or None) for the report header."""
    if device.type != 'cuda':
        return None
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    return torch.cuda.get_device_name(device)


def amp_autocast(use_amp):
    """autocast(fp16) on CUDA when AMP is on, else a no-op context. The entropy
    front-end (unfold / sort / equality tests) is NOT an autocast-eligible op, so
    it stays in fp32; only conv1 + the resnet backbone run in fp16."""
    if use_amp:
        return torch.amp.autocast('cuda')
    return contextlib.nullcontext()


def resolve_num_threads(requested):
    """Auto-pick dataloader workers when the user leaves it at -1.

    The dataset transforms use lambdas, which can't be pickled to workers under
    the 'spawn' start method (macOS / Windows) -- there we must stay at 0. On
    Linux (fork) workers are safe and essential to keep the 4090 fed during the
    Python-side blur/JPEG augmentation."""
    if requested >= 0:
        return requested
    if sys.platform.startswith('linux'):
        return min(8, (os.cpu_count() or 1))
    return 0


def make_opt(args, split, is_train, aug=None):
    """Build a minimal options object that data.create_dataloader understands,
    without going through the heavy argparse-based TrainOptions.

    `aug` optionally overrides the blur / JPEG augmentation params (used both for
    training-time augmentation and for building corrupted test sets)."""
    aug = aug or {}
    return SimpleNamespace(
        mode='binary',
        isTrain=is_train,
        dataroot=os.path.join(args.dataroot, split),
        classes=args.classes.split(',') if args.classes else [],
        class_bal=False,
        serial_batches=not is_train,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        loadSize=args.loadSize,
        cropSize=args.cropSize,
        no_flip=False,
        # Random-crop for train, CENTER-crop for val -- both to cropSize. no_crop
        # would hand the val loader the full 256x256 image while training saw 224
        # crops; the BN stats we recalibrate on 224 crops should not then be applied
        # to a different input size. (Measured on its own this is worth ~0.06 AP,
        # far less than the BN issue, but it is free to get right.)
        no_crop=False,
        no_resize=False,
        rz_interp=['bilinear'],
        # image corruptions (0 prob -> no-op, i.e. clean images)
        blur_prob=aug.get('blur_prob', 0.0),
        blur_sig=aug.get('blur_sig', [0.5]),
        jpg_prob=aug.get('jpg_prob', 0.0),
        jpg_method=aug.get('jpg_method', ['pil']),
        jpg_qual=aug.get('jpg_qual', [75]),
        webp_prob=aug.get('webp_prob', 0.0),
        webp_qual=aug.get('webp_qual', [80]),
    )


def build_model(arch, cfg, device):
    # This is a whitelist: a CONFIGS key with no line below is dropped silently,
    # so the config trains as a clone of the baseline and "measures" a null
    # effect that is really a plumbing bug. Every model kwarg needs an entry here.
    # Guarded by tests/test_texture_split.py::test_config_kwargs_reach_the_model.
    kwargs = dict(
        # 1 logit = the AI-vs-real head, i.e. every existing config. The
        # degradation experiment asks for more heads and passes num_classes;
        # nothing else in this file sets it, so the default is unchanged.
        num_classes=cfg.get('num_classes', 1),
        window_sizes=cfg.get('window_sizes', [2]),
        scales=cfg.get('scales', [1.0, 0.5, 0.25]),
        entropy_mode=cfg.get('entropy_mode', 'shannon'),
        use_rearrange=cfg.get('use_rearrange', True),
        rearrange_block_size=cfg.get('rearrange_block_size', 2),
        normalize_entropy=cfg.get('normalize_entropy', False),
        window_align=cfg.get('window_align', 'resize'),
        texture_split=cfg.get('texture_split', False),
        texture_patch_size=cfg.get('texture_patch_size', 16),
        texture_split_mode=cfg.get('texture_split_mode', 'concat'),
        color_entropy=cfg.get('color_entropy', False),
    )
    factory = {'resnet18': resnet18, 'resnet50': resnet50}[arch]
    model = factory(pretrained=False, **kwargs)   # pretrained=False: conv1 is reshaped
    return model.to(device)


@torch.no_grad()
def recalibrate_bn(model, loader, n_batches, device, use_amp=False):
    """Re-estimate every BatchNorm's running stats over the training distribution.

    WHY THIS EXISTS
    ---------------
    BatchNorm's default momentum=0.1 makes running_mean/running_var an EMA over
    roughly the last ~20 batches of training -- ~320 images at batch 16, collected
    while the weights were still moving. Per layer the error looks harmless (at
    bn1 the stored variance sits ~10% below the true batch variance), but it
    compounds through the nine BN layers of layer1+layer2, and the head is a
    single Linear(128, 1) on a global average pool, so nothing downstream absorbs
    the drift.

    Measured on baseline_2x2 at the sweep's settings (3000 steps, 8 classes,
    batch 16), same weights in both rows:

        model.eval() as-is            logit mean +24.16   acc 0.4938   AP 0.8057
        after this recalibration      logit mean  -0.25   acc 0.9762   AP 0.9985

    Every logit was displaced by ~24, so every prediction collapsed to one class
    and accuracy pinned at the class prior. The sign is arbitrary -- a 600-step
    run shifted by -13.85 instead -- which is why some configs used to report
    all-real and others all-fake. Note AP moves too: this was never just a
    threshold problem, it corrupted the ranking the sweep exists to produce.

    momentum=None switches BN to a cumulative average, so the pass below replaces
    the EMA with an honest mean over n_batches. 25 batches (~2 s) already
    saturates the fix; the default is 50 for margin.

    Estimated on TRAIN data on purpose. Using val here would be transductive --
    the reported numbers would have seen the test set.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns or n_batches <= 0:
        return 0
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # cumulative moving average
    was_training = model.training
    model.train()                  # train() is what makes BN update its stats
    seen = 0
    try:
        for img, _ in loader:
            if seen >= n_batches:
                break
            with amp_autocast(use_amp):
                model(img.to(device, non_blocking=True))
            seen += 1
    finally:
        for m, mom in zip(bns, saved):
            m.momentum = mom
        model.train(was_training)
    return seen


@torch.no_grad()
def evaluate(model, loader, device, max_batches, use_amp=False, keep_scores=False):
    model.eval()
    y_true, logits = [], []
    for b, (img, label) in enumerate(loader):
        if b >= max_batches:
            break
        with amp_autocast(use_amp):
            out = model(img.to(device, non_blocking=True))
        logits.extend(out.float().flatten().cpu().tolist())
        y_true.extend(label.flatten().tolist())
    y_true, logits = np.array(y_true), np.array(logits)
    # Score on the logits directly rather than sigmoid(logits). sigmoid is strictly
    # monotonic, so the rank-based AP is identical and 'probability > 0.5' is
    # exactly 'logit > 0' -- but np.exp(-logit) overflows once the logits get large,
    # which is precisely the regime this whole file's BN bug produces.
    acc = accuracy_score(y_true, logits > 0)
    both = len(set(y_true.tolist())) > 1
    ap = average_precision_score(y_true, logits) if both else float('nan')
    # acc at the best achievable threshold, alongside acc at the fixed 0.5.
    # A row where acc is ~0.5 but acc_best is ~0.95 says "the scores separate the
    # classes, the decision boundary is in the wrong place" -- i.e. exactly the BN
    # failure recalibrate_bn() addresses. Without it that row is indistinguishable
    # from a model that genuinely cannot classify, which is what hid the bug.
    acc_best = best_threshold_acc(y_true, logits) if both else float('nan')
    out = dict(acc=acc, ap=ap, n=len(y_true), acc_best=acc_best,
               logit_mean=float(logits.mean()) if len(logits) else float('nan'))
    # keep_scores is off for the window sweep, so its result dicts (and the
    # report built from them) are exactly what they always were. The entropy
    # experiment turns it on to compute the wider metric set it reports.
    if keep_scores:
        out['y_true'], out['scores'] = y_true, logits
    return out


def best_threshold_acc(y_true, scores):
    """Highest accuracy reachable over all thresholds on `scores`.

    Sort once and sweep: at the k-th cut the predictions are the k highest scores
    positive and the rest negative, so the running count of positives among them
    gives the accuracy in O(n log n) rather than O(n^2)."""
    order = np.argsort(-scores, kind='stable')
    y = y_true[order]
    pos_total = y.sum()
    n = len(y)
    # correct(k) = (positives in the top k) + (negatives in the bottom n-k)
    tp = np.concatenate(([0.0], np.cumsum(y)))
    fp = np.arange(n + 1) - tp
    correct = tp + ((n - pos_total) - fp)
    return float(correct.max() / n)


def make_loader(dataset, args, shuffle, generator=None, drop_last=False,
                persistent=True):
    """DataLoader tuned for a CUDA host: pinned memory + persistent workers so the
    GPU is not stalled on the Python-side augmentation between batches.

    prefetch_factor is raised above the default 2 because the queue depth is what
    hides read latency: at 8 workers the default keeps only 16 batches in flight,
    which is nothing if the images are coming off a slow disk rather than out of
    the page cache. (The real fix for that is scripts/stage_dataset.py -- this
    just stops a cold cache from stalling the GPU outright.)

    persistent=False for loaders the caller abandons early: evaluate() stops after
    max_val_batches, and workers left alive past that keep reading images nobody
    will ever look at, competing for I/O with the loader that is actually in use."""
    pin = (args.device_type == 'cuda')
    workers = args.num_threads
    kwargs = {}
    if workers > 0:
        # Both of these are rejected outright when num_workers == 0.
        kwargs.update(persistent_workers=persistent, prefetch_factor=4)
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, generator=generator,
        num_workers=workers, pin_memory=pin, drop_last=drop_last, **kwargs)


def run_config(name, cfg, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args.arch, cfg, device)
    n_in = model.conv1.in_channels
    n_params = sum(p.numel() for p in model.parameters())

    # Optional training-time augmentation (blur / JPEG) for this config.
    train_aug = cfg.get('train_aug', {})
    train_dataset = get_dataset(make_opt(args, args.train_split, True, aug=train_aug))
    train_loader = make_loader(train_dataset, args, shuffle=True, drop_last=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # Mixed precision: ~2x throughput on a 4090, negligible effect on this task.
    use_amp = (device.type == 'cuda' and not args.no_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Every config trains for exactly args.max_train_steps steps -> a fair
    # comparison. Wall-clock varies only with each config's per-step cost.
    model.train()
    step, t0 = 0, time.time()
    done = False
    for epoch in range(10 ** 6):
        for img, label in train_loader:
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).float()
            with amp_autocast(use_amp):
                out = model(img).squeeze(1)
                loss = criterion(out, label)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            if step % args.log_freq == 0:
                print(f"  [{name}] step {step}/{args.max_train_steps}  "
                      f"loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")
            if step >= args.max_train_steps:
                done = True
                break
        if done:
            break

    train_time = time.time() - t0
    print(f"  [{name}] trained {step} steps in {train_time:.0f}s "
          f"(in_ch={n_in}, params={n_params/1e6:.2f}M)")

    # Drop the training loader before evaluating. We break out of its iterator
    # mid-epoch, and persistent_workers keeps all 8 workers alive and prefetching
    # training batches that will never be consumed -- through the whole eval
    # phase below, stealing I/O from the loaders that are actually being read.
    del train_loader, train_dataset

    # Fix up the BatchNorm running stats before switching to eval mode -- see
    # recalibrate_bn()'s docstring for why the EMA left by training is not usable.
    # Its loader is built here (after the training loader is gone) and dropped
    # immediately after, for the same I/O reason as above.
    if args.bn_recal_batches > 0:
        recal_dataset = get_dataset(make_opt(args, args.train_split, True, aug=train_aug))
        recal_loader = make_loader(recal_dataset, args, shuffle=True, drop_last=True,
                                   persistent=False)
        t_recal = time.time()
        seen = recalibrate_bn(model, recal_loader, args.bn_recal_batches, device,
                              use_amp=use_amp)
        print(f"  [{name}] BN recalibrated on {seen} train batches "
              f"({time.time() - t_recal:.0f}s)")
        del recal_loader, recal_dataset

    # Evaluate on every corruption scenario (clean / blur / jpeg / blur+jpeg).
    # A fixed generator per loader guarantees each scenario sees the SAME images,
    # so differences come only from the corruption, not from sampling.
    scenarios = {}
    for sname, saug in EVAL_SCENARIOS.items():
        val_dataset = get_dataset(make_opt(args, args.val_split, False, aug=saug))
        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = make_loader(val_dataset, args, shuffle=True, generator=g,
                                 persistent=False)
        sc = evaluate(model, val_loader, device, args.max_val_batches, use_amp=use_amp,
                      keep_scores=getattr(args, 'keep_scores', False))
        scenarios[sname] = sc
        print(f"  [{name}] test/{sname:9s}: acc={sc['acc']:.4f} ap={sc['ap']:.4f} "
              f"(n={sc['n']}) [acc_best={sc['acc_best']:.4f} "
              f"logit_mean={sc['logit_mean']:+.2f}]")

    return dict(name=name, in_ch=n_in, params=n_params, time=train_time,
                steps=step, scenarios=scenarios)


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


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataroot',
                   help='root that contains the train/ and val/ split folders '
                        '(required unless --list_configs)')
    p.add_argument('--classes', default='',
                   help='comma-separated class folders (e.g. car,cat,chair,horse). '
                        'Empty -> use every subfolder.')
    p.add_argument('--train_split', default='train')
    p.add_argument('--val_split', default='val')
    p.add_argument('--arch', default='resnet18', choices=['resnet18', 'resnet50'],
                   help='resnet18 is much faster and fine for a quick sweep')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--loadSize', type=int, default=256)
    p.add_argument('--cropSize', type=int, default=128,
                   help='smaller crop = far cheaper entropy computation')
    p.add_argument('--num_threads', type=int, default=-1,
                   help='dataloader workers. -1 = auto (up to 8 on Linux, 0 on '
                        'macOS/Windows where spawn cannot pickle the dataset '
                        "lambdas). Set explicitly to override.")
    p.add_argument('--no_amp', action='store_true',
                   help='disable CUDA mixed precision (AMP is on by default on GPU)')
    p.add_argument('--lr', type=float, default=2e-4)
    # A constant LR is right for the short fixed-step comparison runs this file
    # was built for, so it stays the default. These only matter for a long run,
    # where the last steps otherwise land wherever the noise leaves them.
    p.add_argument('--lr_schedule', default='none', choices=['none', 'cosine'],
                   help="LR schedule over --max_train_steps ('none' = the constant "
                        'LR this sweep has always used, and the default)')
    p.add_argument('--warmup_steps', type=int, default=0,
                   help='linear LR warmup steps before --lr_schedule takes over '
                        '(0 = no warmup, the default)')
    p.add_argument('--lr_min', type=float, default=0.0,
                   help='LR floor for --lr_schedule cosine')
    p.add_argument('--weight_decay', type=float, default=0.0,
                   help='>0 switches Adam to AdamW with this decoupled weight decay. '
                        '0 (default) keeps plain Adam.')
    p.add_argument('--init_from', default='',
                   help='warm-start from a checkpoint: every tensor except the fc1.* '
                        'head is loaded, and a 1-output fc1 seeds output 0. '
                        "'' (default) = the usual random init. Use "
                        'pretrained/model_epoch_best.pth to start from released MLEP.')
    p.add_argument('--max_train_steps', type=int, default=300,
                   help='fixed step count trained for EVERY config (fair comparison). '
                        'Keep modest so the sweep stays quick.')
    p.add_argument('--max_val_batches', type=int, default=20)
    p.add_argument('--bn_recal_batches', type=int, default=50,
                   help='train batches used to re-estimate BatchNorm running stats '
                        'before evaluating (0 = off). Training leaves BN with a '
                        'momentum-0.1 EMA over its last ~20 batches, which displaces '
                        'every eval logit by 10-25 and pins accuracy at the class '
                        'prior (~0.50). 25 batches already fixes it; 50 is margin.')
    p.add_argument('--log_freq', type=int, default=50)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--device', default='', help="'cuda', 'mps', 'cpu' (auto if empty)")
    p.add_argument('--configs', default='',
                   help='comma-separated subset of config names (default: all)')
    p.add_argument('--out', default='experiment_results.txt',
                   help='text file to write the per-model results to')
    p.add_argument('--list_configs', action='store_true')

    # --- additional experiments ------------------------------------------- #
    # 'windows' is this file's original sweep and stays the default, so every
    # existing command line behaves exactly as before. The other two are the
    # separate experiment groups; each owns its config registry, its report
    # builder and its own --out file, and reuses the train / BN-recal / eval
    # machinery above rather than duplicating it.
    p.add_argument('--experiment', default='windows',
                   choices=['windows', 'entropy', 'mlep_degradation'],
                   help='which experiment group to run (default: the window sweep)')
    # entropy experiment
    p.add_argument('--entropy_stage', default='both',
                   choices=['both', 'features', 'deep'],
                   help="[entropy] 'features' = summary-statistic entropy features "
                        "+ sklearn classifiers; 'deep' = train the MLEP network with "
                        "each entropy front-end; 'both' runs the two in that order.")
    p.add_argument('--entropy_feat_train_batches', type=int, default=60,
                   help='[entropy] train batches used to fit the feature classifiers')
    p.add_argument('--entropy_feat_val_batches', type=int, default=30,
                   help='[entropy] val batches used to score the feature classifiers')
    p.add_argument('--unseen_split', default='',
                   help='[entropy/mlep_degradation] extra split folder holding unseen '
                        'generators (e.g. test). Empty = skip that evaluation.')
    # degradation experiment
    p.add_argument('--deg_variants', type=int, default=4,
                   help='[mlep_degradation] degraded variants generated per source image')
    p.add_argument('--deg_sev_weight', type=float, default=0.0,
                   help='[mlep_degradation] loss weight of the OPTIONAL severity '
                        'heads. 0 (default) = the model predicts exactly the three '
                        'asked-for probabilities; >0 adds the quality/blur-level '
                        'heads and the predicted_* columns')
    p.add_argument('--deg_clean_oversample', type=int, default=0,
                   help='[mlep_degradation] extra copies of the untouched '
                        "('lossless','none') cell in the TRAINING cell list. Only 1 "
                        'of the 24 training cells is untouched, so the ai head sees '
                        'clean images ~4%% of the time and the blur/jpeg heads train '
                        'on 71%%/83%% positives. 0 (default) = the plain 24-cell grid. '
                        'Evaluation cells are never affected.')
    p.add_argument('--deg_save_ckpt', action='store_true',
                   help='[mlep_degradation] save the trained weights next to --out as '
                        '<stem>_<config>.pt. Off by default: the comparison runs only '
                        'want the report, so nothing is written unless asked.')
    p.add_argument('--deg_val_every', type=int, default=0,
                   help='[mlep_degradation] run a mid-training evaluation every N '
                        'steps and (with --deg_save_ckpt) keep the best one. '
                        '0 (default) = off, training is one uninterrupted loop.')
    p.add_argument('--deg_val_cond', default='seen',
                   help='[mlep_degradation] which CONDITIONS entry --deg_val_every '
                        'scores. One condition only -- all six costs 6x for little.')
    p.add_argument('--deg_val_recal', type=int, default=10,
                   help='[mlep_degradation] train batches for the throwaway BN '
                        'recalibration each mid-training evaluation needs (see '
                        'recalibrate_bn: eval on training-EMA stats is meaningless)')
    args = p.parse_args()

    registry = CONFIGS
    if args.experiment == 'entropy':
        from experiments.entropy import ENTROPY_CONFIGS as registry
    elif args.experiment == 'mlep_degradation':
        from experiments.degradation import DEGRADATION_CONFIGS as registry

    if args.list_configs:
        for k, v in registry.items():
            print(f"{k:26s} {v}")
        return

    if not args.dataroot:
        raise SystemExit("--dataroot is required (unless --list_configs)")

    device = get_device(args.device)
    args.device_type = device.type
    args.num_threads = resolve_num_threads(args.num_threads)
    gpu_name = setup_cuda_perf(device)
    use_amp = (device.type == 'cuda' and not args.no_amp)

    if args.experiment != 'windows':
        # Never write over the window sweep's default file by accident.
        if args.out == 'experiment_results.txt':
            stamp = time.strftime('%Y%m%d_%H%M%S')
            short = 'entropy' if args.experiment == 'entropy' else 'degradation'
            args.out = os.path.join('results', f'{short}_{stamp}.txt')
            os.makedirs('results', exist_ok=True)
        if args.experiment == 'entropy':
            from experiments.entropy import run_entropy_experiment as run_experiment
        else:
            from experiments.degradation import run_degradation_experiment as run_experiment
        run_experiment(args, device, gpu_name, use_amp)
        return

    names = args.configs.split(',') if args.configs else list(CONFIGS.keys())
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown configs: {unknown}. Available: {list(CONFIGS)}")

    print(f"Device: {device}" + (f" ({gpu_name})" if gpu_name else "") +
          f" | arch: {args.arch} | steps: {args.max_train_steps} "
          f"| crop: {args.cropSize} | batch: {args.batch_size} "
          f"| workers: {args.num_threads} | amp: {use_amp}")
    print(f"Running {len(names)} config(s): {names}\n")

    results = []
    for i, name in enumerate(names, 1):
        print(f"=== [{i}/{len(names)}] {name} : {CONFIGS[name]} ===")
        results.append(run_config(name, CONFIGS[name], args, device))
        # Checkpoint after every config so an interrupted sweep keeps its work.
        write_report(results, args, device, gpu_name, use_amp, len(names))
        print(f"  [{name}] done -- {i}/{len(names)} written to {args.out}\n")

    report = write_report(results, args, device, gpu_name, use_amp, len(names))
    print("\n" + report)
    print(f"\nResults written to {args.out}")


if __name__ == '__main__':
    main()
