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
    python -m mlep.experiments.windows \
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
    python -m mlep.experiments.windows ... --configs baseline_2x2,multiwindow_multiscale
List available configs with --list_configs.

Two further experiment groups live in experiments/ and are selected with
--experiment (default 'windows' = this sweep, unchanged):

    --experiment entropy            mlep/experiments/entropy.py
        Do other entropy definitions (Renyi, Tsallis, permutation, and
        combinations) detect AI images better than Shannon?
    --experiment mlep_degradation   mlep/experiments/degradation.py
        Predict AI-vs-real, blurred-or-not and JPEG-compressed-or-not (plus both
        severities) from the same front-end.

They reuse the machinery below -- make_opt/make_loader, build_model, run_config,
recalibrate_bn, evaluate, write_report -- and write the same kind of report file,
so --list_configs, the console output and scripts/aggregate_results.py all work
the same way for them. See mlep/experiments/*.py for their configs and metrics.
"""
import os
import sys
import time
import argparse

import numpy as np
import torch

from mlep.harness.data import EVAL_SCENARIOS, make_loader, make_opt
from mlep.harness.device import (amp_autocast, get_device, resolve_num_threads,
                                 setup_cuda_perf)
from mlep.harness.evaluate import best_threshold_acc, evaluate, run_config
from mlep.harness.model import build_model, recalibrate_bn
from mlep.harness.report import build_report, write_report

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
        from mlep.experiments.entropy import ENTROPY_CONFIGS as registry
    elif args.experiment == 'mlep_degradation':
        from mlep.experiments.degradation import DEGRADATION_CONFIGS as registry

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
            from mlep.experiments.entropy import run_entropy_experiment as run_experiment
        else:
            from mlep.experiments.degradation import run_degradation_experiment as run_experiment
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
