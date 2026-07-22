"""
Quick experiment harness to compare different local-entropy window settings.

It trains the MLEP network for a *small* number of steps under several
configurations (different sliding-window sizes, single- vs multi-scale, shannon
vs unique entropy) and reports validation accuracy / average precision so you can
see which setting looks most promising before committing to a full training run.

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
    # 2x2 only, single scale -> is the pyramid even helping? (3 channels)
    'w2x2_singlescale':    dict(window_sizes=[2],       scales=[1.0]),
    # Multi-window with the 'unique' (set-cardinality) feature instead of shannon.
    'w2x2_unique':         dict(window_sizes=[2],    scales=[1.0, 0.5, 0.25],
                                entropy_mode='unique'),
    # Single 3x3 window, same pyramid.
    'w3x3_multiscale': dict(window_sizes=[3], scales=[1.0, 0.5, 0.25]),
    'w4x4_multiscale': dict(window_sizes=[4], scales=[1.0, 0.5, 0.25]),
    'w5x5_multiscale': dict(window_sizes=[5], scales=[1.0, 0.5, 0.25]),
    'w6x6_multiscale': dict(window_sizes=[6], scales=[1.0, 0.5, 0.25]),
    'w7x7_multiscale': dict(window_sizes=[7], scales=[1.0, 0.5, 0.25]),
    'w8x8_multiscale': dict(window_sizes=[8], scales=[1.0, 0.5, 0.25]),

    # Combine 2x2 AND 3x3 windows (the multi-window idea) -> 18 channels.
    'multiwindow_multiscale':     dict(window_sizes=[2, 4, 6],    scales=[1.0, 0.5, 0.25]),
    # Three window sizes at a single scale -> 9 channels (fair vs baseline).
    'multiwindow_singlescale':  dict(window_sizes=[2, 4, 6], scales=[1.0]),
    'multiwindow_singlescale_normalized': dict(window_sizes=[2, 4, 6], scales=[1.0, 0.5, 0.25],
                                normalize_entropy=True),

    # --- train WITH augmentation (does it help robustness to corruptions?) ---
    # Same model as baseline_2x2, but the training images are randomly blurred /
    # JPEG-compressed. The 'train_aug' key is consumed by the harness (not the model).
    'baseline_train_blur': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                train_aug=dict(blur_prob=0.5, blur_sig=[0.0, 3.0])),
    'baseline_train_jpeg': dict(window_sizes=[2], scales=[1.0, 0.5, 0.25],
                                train_aug=dict(jpg_prob=0.5, jpg_qual=[30, 100],
                                               jpg_method=['cv2', 'pil'])),
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
        return torch.cuda.amp.autocast()
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
        no_crop=not is_train,     # center-crop for val, random-crop for train
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
    kwargs = dict(
        num_classes=1,
        window_sizes=cfg.get('window_sizes', [2]),
        scales=cfg.get('scales', [1.0, 0.5, 0.25]),
        entropy_mode=cfg.get('entropy_mode', 'shannon'),
        use_rearrange=cfg.get('use_rearrange', True),
        rearrange_block_size=cfg.get('rearrange_block_size', 2),
        normalize_entropy=cfg.get('normalize_entropy', False),
    )
    factory = {'resnet18': resnet18, 'resnet50': resnet50}[arch]
    model = factory(pretrained=False, **kwargs)   # pretrained=False: conv1 is reshaped
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device, max_batches, use_amp=False):
    model.eval()
    y_true, y_pred = [], []
    for b, (img, label) in enumerate(loader):
        if b >= max_batches:
            break
        with amp_autocast(use_amp):
            out = model(img.to(device, non_blocking=True))
        out = out.float().sigmoid().flatten().cpu()
        y_pred.extend(out.tolist())
        y_true.extend(label.flatten().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred) if len(set(y_true.tolist())) > 1 else float('nan')
    return acc, ap, len(y_true)


def make_loader(dataset, args, shuffle, generator=None, drop_last=False):
    """DataLoader tuned for a CUDA host: pinned memory + persistent workers so the
    GPU is not stalled on the Python-side augmentation between batches."""
    pin = (args.device_type == 'cuda')
    workers = args.num_threads
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=shuffle, generator=generator,
        num_workers=workers, pin_memory=pin, drop_last=drop_last,
        persistent_workers=(workers > 0))


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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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

    # Evaluate on every corruption scenario (clean / blur / jpeg / blur+jpeg).
    # A fixed generator per loader guarantees each scenario sees the SAME images,
    # so differences come only from the corruption, not from sampling.
    scenarios = {}
    for sname, saug in EVAL_SCENARIOS.items():
        val_dataset = get_dataset(make_opt(args, args.val_split, False, aug=saug))
        g = torch.Generator()
        g.manual_seed(args.seed)
        val_loader = make_loader(val_dataset, args, shuffle=True, generator=g)
        acc, ap, n_val = evaluate(model, val_loader, device, args.max_val_batches,
                                  use_amp=use_amp)
        scenarios[sname] = dict(acc=acc, ap=ap, n=n_val)
        print(f"  [{name}] test/{sname:9s}: acc={acc:.4f} ap={ap:.4f} (n={n_val})")

    return dict(name=name, in_ch=n_in, params=n_params, time=train_time,
                steps=step, scenarios=scenarios)


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
    p.add_argument('--max_train_steps', type=int, default=300,
                   help='fixed step count trained for EVERY config (fair comparison). '
                        'Keep modest so the sweep stays quick.')
    p.add_argument('--max_val_batches', type=int, default=20)
    p.add_argument('--log_freq', type=int, default=50)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--device', default='', help="'cuda', 'mps', 'cpu' (auto if empty)")
    p.add_argument('--configs', default='',
                   help='comma-separated subset of config names (default: all)')
    p.add_argument('--out', default='experiment_results.txt',
                   help='text file to write the per-model results to')
    p.add_argument('--list_configs', action='store_true')
    args = p.parse_args()

    if args.list_configs:
        for k, v in CONFIGS.items():
            print(f"{k:22s} {v}")
        return

    if not args.dataroot:
        raise SystemExit("--dataroot is required (unless --list_configs)")

    device = get_device(args.device)
    args.device_type = device.type
    args.num_threads = resolve_num_threads(args.num_threads)
    gpu_name = setup_cuda_perf(device)
    use_amp = (device.type == 'cuda' and not args.no_amp)

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
    for name in names:
        print(f"=== {name} : {CONFIGS[name]} ===")
        results.append(run_config(name, CONFIGS[name], args, device))
        print()

    # Build the report (best clean-AP first), print it AND save it to --out.
    scen_names = list(EVAL_SCENARIOS.keys())

    def clean_ap(r):
        v = r['scenarios']['clean']['ap']
        return v if v == v else -1        # NaN -> sort last

    results.sort(key=clean_ap, reverse=True)

    lines = []
    lines.append("MLEP window / robustness sweep")
    lines.append(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"device={device}{f' ({gpu_name})' if gpu_name else ''}  "
                 f"arch={args.arch}  amp={use_amp}  workers={args.num_threads}  "
                 f"steps={args.max_train_steps}  "
                 f"crop={args.cropSize}  batch={args.batch_size}  lr={args.lr}  "
                 f"classes={args.classes or 'ALL'}  val_batches={args.max_val_batches}")
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
            lines.append(f"    test/{s:10s}  acc={sc['acc']:.4f}  "
                         f"ap={sc['ap']:.4f}  (n={sc['n']})")
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
    lines.append("Note: short training -> ranking is indicative. Re-run the top "
                 "configs with more --max_train_steps to confirm.")

    report = "\n".join(lines)
    print("\n" + report)
    with open(args.out, 'w') as f:
        f.write(report + "\n")
    print(f"\nResults written to {args.out}")


if __name__ == '__main__':
    main()
