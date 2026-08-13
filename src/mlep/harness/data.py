"""Dataset options, dataloaders and the test-time corruption scenarios."""
# Split out of the original mlep/experiments/windows.py. The sweep's CONFIGS and
# main() live in mlep.experiments.windows; everything reusable lives here, so
# consumers no longer need in-function imports to dodge a circular dependency.

from types import SimpleNamespace

import torch

from mlep.data import get_dataset

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


