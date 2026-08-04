"""Stage a subset of the training corpus onto a fast filesystem (default: RAM).

WHY THIS EXISTS
---------------
This box's root volume is rotational (/sys/block/vda/queue/rotational == 1). It
serves ~124 random reads/s, ~3.8 MB/s, at ~90 ms latency. Training reads 256x256
PNGs (~100 KB each) in shuffled order across 288,048 files spread over 28.8 GB,
which is the worst case for a seeking disk. Measured on a live sweep: the GPU sat
at 0% utilisation and 60 W of 450 W while all 8 dataloader workers blocked in 'D'
state inside folio_wait_bit_common, and training ran at 0.467 s/step -- 34 img/s,
i.e. 3.4 MB/s, matching the device ceiling almost exactly.

The first ~600 steps of a run look fast only because those images happen to still
be in the page cache; once that is exhausted throughput collapses ~30x. That is
the 5s -> 9s -> 57s -> 200s ramp in the sweep logs.

The key observation: a 3000-step run at batch 16 touches 48,000 images, which is
16.7% of ONE epoch. We were paying cold random-read cost for a corpus we barely
sample. Staging a few thousand images per class into tmpfs removes the disk from
the loop entirely and lets the 4090 run at its actual speed.

USAGE
-----
    python scripts/stage_dataset.py --classes car,cat,chair,horse,boat,person,dog,bird
    ./run.sh stage                      # same thing, via run.sh's knobs

Idempotent: re-running copies only what is missing, so an interrupted stage
resumes and a completed one is a fast no-op. /dev/shm does not survive a reboot,
so expect to re-run it after one -- run.sh does that automatically.

The subset is a SEEDED RANDOM sample of each class/label directory, not the first
N files: filenames may encode generation order, and a prefix could correlate with
something systematic. Same seed -> same subset, so runs stay reproducible and a
re-stage does not silently change the data underneath a half-finished sweep.
"""
import argparse
import os
import random
import shutil
import sys

LABELS = ('0_real', '1_fake')
TMP_SUFFIX = '.staging-tmp'


def list_images(d):
    """-> {filename: size}, ignoring dotfiles, subdirectories and our own temp
    files. scandir gives us the size in the same pass as the listing, which is
    what lets plan_dir() do the size comparison for free."""
    if not os.path.isdir(d):
        return {}
    out = {}
    with os.scandir(d) as it:
        for e in it:
            if e.name.startswith('.') or e.name.endswith(TMP_SUFFIX):
                continue
            if e.is_file(follow_symlinks=False):
                out[e.name] = e.stat().st_size
    return out


def plan_dir(src, dest, n, seed):
    """-> (wanted, missing, sizes). n=0 or n>=len means take everything (val).

    A destination file counts as staged only if its size MATCHES the source. A
    name-only check is not enough: an interrupted copy leaves a short or empty
    file, and treating that as done bakes the corruption in permanently -- the
    next run then dies on PIL.UnidentifiedImageError deep inside a dataloader
    worker, thousands of steps into a config. Comparing sizes makes a re-stage
    self-healing, and it is nearly free because both listings come from scandir."""
    src_sizes = list_images(src)
    if not src_sizes:
        return [], [], {}
    names = sorted(src_sizes)
    if n and n < len(names):
        # sorted() first so the sample depends only on the seed, never on the
        # order readdir happens to return.
        names = sorted(random.Random(seed).sample(names, n))
    have = list_images(dest)
    missing = [f for f in names if have.get(f) != src_sizes[f]]
    return names, missing, src_sizes


def copy_atomic(src, dst):
    """Copy via a temp file in the destination directory, then rename.

    rename(2) within one filesystem is atomic, so dst either does not exist or is
    the complete file -- never a half-written one. Without this, a Ctrl-C or an
    OOM kill mid-copy leaves a truncated image that looks staged by name."""
    tmp = dst + TMP_SUFFIX
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        # BaseException, not Exception: KeyboardInterrupt is the likely case and
        # we still want the partial file gone.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataroot', default='datasets/TrainDatasets',
                    help='source root holding train/ and val/')
    ap.add_argument('--dest', default='/dev/shm/mlep_data',
                    help='where to stage (default: tmpfs, i.e. RAM)')
    ap.add_argument('--classes', default='',
                    help='comma-separated; empty = every class folder in train/')
    ap.add_argument('--per-dir', type=int, default=4000,
                    help='train images per class/label dir (0 = all). 4000 gives '
                         '64k images over 8 classes, so a 3000-step run at batch '
                         '16 draws 48k = 0.75 epoch and never repeats an image.')
    ap.add_argument('--val-per-dir', type=int, default=0,
                    help='val images per class/label dir (0 = all; there are only '
                         '200, so all of val is ~0.4 GB for 8 classes)')
    ap.add_argument('--seed', type=int, default=0, help='subset sampling seed')
    ap.add_argument('--force', action='store_true',
                    help='delete dest/<split> first instead of topping it up')
    args = ap.parse_args()

    train_root = os.path.join(args.dataroot, 'train')
    if not os.path.isdir(train_root):
        sys.exit(f"stage_dataset: no such directory: {train_root}")

    classes = ([c for c in args.classes.split(',') if c] or
               sorted(d for d in os.listdir(train_root)
                      if os.path.isdir(os.path.join(train_root, d))))

    if args.force:
        for split in ('train', 'val'):
            shutil.rmtree(os.path.join(args.dest, split), ignore_errors=True)

    # ---- plan every directory up front, so we can size the copy and bail out
    # ---- before writing a single byte if it will not fit.
    jobs, est_bytes, n_wanted, n_missing = [], 0, 0, 0
    for split, per_dir in (('train', args.per_dir), ('val', args.val_per_dir)):
        for cls in classes:
            for label in LABELS:
                src = os.path.join(args.dataroot, split, cls, label)
                dst = os.path.join(args.dest, split, cls, label)
                wanted, missing, sizes = plan_dir(src, dst, per_dir, args.seed)
                if not wanted:
                    continue
                jobs.append((src, dst, missing, f"{split}/{cls}/{label}"))
                n_wanted += len(wanted)
                n_missing += len(missing)
                est_bytes += sum(sizes[f] for f in missing)

    if not jobs:
        sys.exit(f"stage_dataset: found no images under {args.dataroot} "
                 f"for classes {classes}")

    gb = est_bytes / 1e9
    print(f"Staging {n_wanted} images ({len(classes)} classes) -> {args.dest}")
    if n_missing == 0:
        print("Everything already staged. Nothing to do.")
        return
    os.makedirs(args.dest, exist_ok=True)
    free = shutil.disk_usage(args.dest).free
    print(f"  {n_missing} still to copy, ~{gb:.1f} GB; "
          f"{free / 1e9:.1f} GB free on the destination")
    # 1.1x: tmpfs rounds every file up to a page, and we do not want to discover
    # we are 200 MB short after copying 6 GB.
    if est_bytes * 1.1 > free:
        sys.exit(f"stage_dataset: need ~{gb * 1.1:.1f} GB but only "
                 f"{free / 1e9:.1f} GB is free on {args.dest}.\n"
                 f"    Lower --per-dir, drop a class, or pick another --dest.\n"
                 f"    (/dev/shm defaults to half of RAM; raise it with\n"
                 f"     sudo mount -o remount,size=32G /dev/shm)")

    copied = 0
    for src, dst, missing, label in jobs:
        if not missing:
            continue
        os.makedirs(dst, exist_ok=True)
        for name in missing:
            copy_atomic(os.path.join(src, name), os.path.join(dst, name))
            copied += 1
        print(f"  {label:34s} +{len(missing):6d}  "
              f"({copied}/{n_missing})", flush=True)

    print(f"Done: {copied} images copied, {n_wanted} staged in total.")
    print(f"Point training at it with:  --dataroot {args.dest}")


if __name__ == '__main__':
    main()
