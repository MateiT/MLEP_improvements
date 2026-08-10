"""Full retrain of a 4-output MLEP variant on the half-size ProGAN train set.

    python train_perturbation4.py --dataroot datasets/TrainDatasets \
        --name mlep_pert4 --out results/pert4_<stamp>.txt

Every source image contributes exactly FOUR training samples -- clean, blurred,
JPEG-compressed and noisy -- and the network gets four sigmoid outputs:

    logit 0   ai_generated        (the original MLEP head, unchanged)
    logit 1   blurred
    logit 2   jpeg_compressed
    logit 3   noise_added

The degradations are mutually exclusive: a sample carries at most one, so the
last three targets are one-hot for a perturbed variant and all-zero for the
clean one. That makes the three degradation heads exactly 25% positive by
construction, in every category and for both labels.

Severity levels
---------------
    blur   sigma in {1, 3, 5}      scipy.ndimage.gaussian_filter, truncate=4.0
    jpeg   quality in {90, 70, 50} PIL encoder
    noise  sigma in {1, 3, 5}      additive Gaussian, 8-bit units, clipped

The level is picked by ROUND ROBIN over each <category>/<label> group's sorted
file list (level = position % 3) rather than by a random draw. A random draw
gives equal levels only in expectation; round robin gives them exactly, which is
what "all categories have an equal amount of perturbations, and the same for
real/fake" asks for. It is also reproducible without storing a manifest -- see
--audit, which prints the realised counts.

Training recipe
---------------
Follows the released MLEP train.py: ResNet-50 front-end, Adam(lr 1e-4,
betas 0.9/0.999), BCEWithLogitsLoss, batch 64, resize 256 -> RandomCrop 224 ->
RandomHorizontalFlip, lr *= 0.9 every --delr_freq epochs (floor 1e-6),
validation + checkpoint every epoch, early stopping after --earlystop_epoch
epochs without improvement, seed 100.

Three deliberate deviations from that file, all documented and all flagged in
the report header:

  1. BatchNorm statistics are re-estimated before every validation and before
     every save (see experiment_windows.recalibrate_bn). Without it the stored
     EMA displaces every logit and validation accuracy collapses to the class
     prior -- measured +24.16 logit shift, acc 0.4938 vs 0.9762.
  2. AMP + cuDNN autotuning are on. train.py's seed_torch() sets
     cudnn.enabled=False, which costs ~2x throughput for no benefit here.
  3. resnet50(pretrained=True) is NOT used: ImageNet weights do not fit this
     architecture (9-channel conv1, layer3/layer4 removed, fc1 head) and
     load_state_dict raises. Training starts from random init unless
     --init_from is given.
"""
import argparse
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from data.datasets import gaussian_blur, pil_jpg
from experiment_windows import (amp_autocast, get_device, make_loader,
                                recalibrate_bn, setup_cuda_perf)
from networks.resnet import resnet50

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

# variant index -> (name, target column in logits 1..3). 0 = clean, no column.
VARIANTS = ('clean', 'blur', 'jpeg', 'noise')
LEVELS = {'blur': [1.0, 3.0, 5.0], 'jpeg': [90, 70, 50], 'noise': [1.0, 3.0, 5.0]}
HEADS = ('ai', 'blur', 'jpeg', 'noise')


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class Perturbation4Dataset(torch.utils.data.Dataset):
    """ImageFolder-style scan; every source yields 4 variants.

    Index arithmetic is (source, variant) = divmod(index, 4), so variant 0 of
    every source is clean and variants 1..3 are blur / jpeg / noise. The
    severity level is a function of the source's rank inside its own
    <category>/<label> group, which is what makes the level counts exactly
    balanced rather than balanced in expectation.

    Transform order mirrors data.datasets.binary_dataset and
    experiments.degradation.DegradationDataset: resize -> degrade -> crop ->
    flip -> ToTensor -> Normalize. Degrading before the crop matters for JPEG:
    the 8x8 block grid must be laid down on the full frame, not on the crop.
    """

    def __init__(self, root, classes, args, is_train):
        self.samples = []          # (path, ai_label, category, level_index)
        for cls in sorted(classes):
            for lab, ai in (('0_real', 0), ('1_fake', 1)):
                d = os.path.join(root, cls, lab)
                if not os.path.isdir(d):
                    continue
                files = sorted(f for f in os.listdir(d)
                               if f.lower().endswith(IMG_EXT))
                for rank, f in enumerate(files):
                    self.samples.append(
                        (os.path.join(d, f), ai, cls, rank % 3))
        if not self.samples:
            raise SystemExit(f"no images found under {root}")
        self.is_train = is_train
        self.resize = transforms.Resize((args.loadSize, args.loadSize))
        self.crop = (transforms.RandomCrop(args.cropSize) if is_train
                     else transforms.CenterCrop(args.cropSize))
        self.flip = (transforms.RandomHorizontalFlip() if is_train
                     else transforms.Lambda(lambda im: im))
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # Noise must be reproducible for the val set or the metric wobbles from
        # epoch to epoch for reasons that have nothing to do with the model.
        self.fixed_noise = not is_train
        self.seed = args.seed

    def __len__(self):
        return len(self.samples) * 4

    def describe(self, index):
        src, v = divmod(index, 4)
        path, ai, cls, lvl = self.samples[src]
        kind = VARIANTS[v]
        level = None if kind == 'clean' else LEVELS[kind][lvl]
        return dict(path=path, ai=ai, category=cls, variant=kind, level=level)

    def targets(self, index):
        """[ai, blurred, jpeg, noise] -- the last three are one-hot or all-zero."""
        src, v = divmod(index, 4)
        t = [float(self.samples[src][1]), 0.0, 0.0, 0.0]
        if v > 0:
            t[v] = 1.0
        return t

    def __getitem__(self, index):
        src, v = divmod(index, 4)
        path, _, _, lvl = self.samples[src]
        with Image.open(path) as im:
            img = self.resize(im.convert('RGB'))

        kind = VARIANTS[v]
        if kind != 'clean':
            arr = np.array(img)
            if kind == 'blur':
                gaussian_blur(arr, LEVELS['blur'][lvl])       # in-place, per channel
            elif kind == 'jpeg':
                arr = pil_jpg(arr, LEVELS['jpeg'][lvl])
            else:
                sigma = LEVELS['noise'][lvl]
                if self.fixed_noise:
                    rng = np.random.default_rng(
                        (zlib.crc32(path.encode()) + self.seed) & 0xFFFFFFFF)
                    n = rng.normal(0.0, sigma, arr.shape)
                else:
                    n = np.random.normal(0.0, sigma, arr.shape)
                arr = np.clip(arr.astype(np.float32) + n, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        img = self.flip(self.crop(img))
        # Two items only, so the shared recalibrate_bn()/make_loader() plumbing
        # takes this loader unchanged. The variant is recoverable from the
        # target: columns 1..3 are one-hot for a perturbed sample, all-zero for
        # a clean one (see variant_of).
        return self.norm(self.to_tensor(img)), torch.tensor(self.targets(index))


# --------------------------------------------------------------------------- #
# balance audit
# --------------------------------------------------------------------------- #
def audit(ds, title):
    """Realised counts per category x label x variant x level. This is the
    evidence for the balance claim, not an assertion about it."""
    import collections
    per_cat = collections.Counter()
    per_lab = collections.Counter()
    per_lvl = collections.Counter()
    for path, ai, cls, lvl in ds.samples:
        per_cat[cls] += 1
        per_lab[ai] += 1
        per_lvl[lvl] += 1
    lines = [f"\n{title}: {len(ds.samples)} sources x 4 variants = {len(ds)} samples"]
    sizes = sorted(set(per_cat.values()))
    lines.append(f"  categories: {len(per_cat)}  sources/category: "
                 f"{sizes if len(sizes) <= 3 else f'{min(sizes)}..{max(sizes)}'}")
    lines.append(f"  real sources: {per_lab[0]}   fake sources: {per_lab[1]}")
    lines.append(f"  per category/label, each of clean|blur|jpeg|noise gets "
                 f"exactly {per_cat[min(per_cat)] // 2} samples")
    lines.append(f"  severity levels (index 0/1/2) over all sources: "
                 f"{per_lvl[0]} / {per_lvl[1]} / {per_lvl[2]}")
    # the strong statement: level balance holds INSIDE every category/label group
    grp = collections.defaultdict(collections.Counter)
    for path, ai, cls, lvl in ds.samples:
        grp[(cls, ai)][lvl] += 1
    spread = max(max(c.values()) - min(c.values()) for c in grp.values())
    lines.append(f"  max level imbalance within any category/label group: "
                 f"{spread} sample(s)  [{len(grp)} groups]")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def variant_of(T):
    """Recover the variant index from the targets: 0 when columns 1..3 are all
    zero (clean), else 1 + argmax over those one-hot columns."""
    deg = T[:, 1:]
    return np.where(deg.sum(1) > 0, deg.argmax(1) + 1, 0)


@torch.no_grad()
def evaluate(model, loader, device, use_amp, max_batches=0):
    """Per-head metrics, plus the AI head broken down by degradation type."""
    model.eval()
    P, T = [], []
    for b, (img, tgt) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        with amp_autocast(use_amp):
            out = model(img.to(device, non_blocking=True)).float()
        P.append(torch.sigmoid(out).cpu().numpy())
        T.append(tgt.numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    V = variant_of(T)

    def m(p, t):
        if len(np.unique(t)) < 2:
            return dict(acc=float(((p > 0.5) == t).mean()), ap=float('nan'),
                        auc=float('nan'), n=len(t))
        return dict(acc=float(accuracy_score(t, p > 0.5)),
                    ap=float(average_precision_score(t, p)),
                    auc=float(roc_auc_score(t, p)), n=len(t))

    res = {h: m(P[:, i], T[:, i]) for i, h in enumerate(HEADS)}
    # The number that matters: can the AI head still work under each degradation?
    res['ai_by_variant'] = {VARIANTS[k]: m(P[V == k, 0], T[V == k, 0])
                            for k in range(4)}
    return res


def fmt_eval(r):
    s = "  ".join(f"{h}: acc={r[h]['acc']:.4f} ap={r[h]['ap']:.4f}"
                  for h in HEADS)
    b = "  ".join(f"{k}={r['ai_by_variant'][k]['ap']:.4f}"
                  for k in VARIANTS)
    return f"{s}\n      ai AP by variant:  {b}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataroot', default='datasets/TrainDatasets')
    p.add_argument('--classes', default='',
                   help='comma list; empty = every category folder')
    p.add_argument('--train_split', default='train')
    p.add_argument('--val_split', default='val')
    p.add_argument('--name', default='mlep_pert4')
    p.add_argument('--checkpoints_dir', default='checkpoints')
    p.add_argument('--out', default='')
    # --- the released MLEP recipe ---
    p.add_argument('--arch', default='resnet50')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--niter', type=int, default=50, help='max epochs')
    p.add_argument('--delr_freq', type=int, default=20,
                   help='epochs between lr *= 0.9')
    p.add_argument('--earlystop_epoch', type=int, default=15)
    p.add_argument('--loadSize', type=int, default=256)
    p.add_argument('--cropSize', type=int, default=224)
    p.add_argument('--seed', type=int, default=100)
    # --- harness ---
    p.add_argument('--num_threads', type=int, default=8)
    p.add_argument('--bn_recal_batches', type=int, default=50)
    p.add_argument('--max_val_batches', type=int, default=0, help='0 = all')
    p.add_argument('--max_train_steps', type=int, default=0,
                   help='0 = full epochs; >0 caps steps per epoch (smoke test)')
    p.add_argument('--init_from', default='',
                   help='warm-start checkpoint; fc1.* is re-initialised')
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--device', default='')
    p.add_argument('--audit', action='store_true',
                   help='print the balance audit and exit without training')
    return p.parse_args()


def classes_of(args, split):
    if args.classes:
        return args.classes.split(',')
    root = os.path.join(args.dataroot, split)
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


def warm_start(model, path):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    ck = {k: v for k, v in ck.items() if not k.startswith('fc1.')}
    missing, unexpected = model.load_state_dict(ck, strict=False)
    stray = [k for k in missing if not k.startswith('fc1.')]
    if unexpected or stray:
        raise SystemExit(f"--init_from {path}: incompatible checkpoint "
                         f"(unexpected={unexpected[:3]} missing={stray[:3]})")
    return f"warm-started from {path} (fc1 re-initialised)"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    gpu = setup_cuda_perf(device)
    use_amp = (device.type == 'cuda' and not args.no_amp)
    # make_loader reads this off args to decide on pinned memory.
    args.device_type = device.type

    train_set = Perturbation4Dataset(
        os.path.join(args.dataroot, args.train_split),
        classes_of(args, args.train_split), args, is_train=True)
    val_set = Perturbation4Dataset(
        os.path.join(args.dataroot, args.val_split),
        classes_of(args, args.val_split), args, is_train=False)

    report = [audit(train_set, "TRAIN"), audit(val_set, "VAL")]
    print("\n".join(report), flush=True)
    if args.audit:
        return 0

    save_dir = os.path.join(args.checkpoints_dir, args.name)
    os.makedirs(save_dir, exist_ok=True)
    out = args.out or os.path.join(
        'results', f"pert4_{time.strftime('%Y%m%d_%H%M%S')}_{args.name}.txt")
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)

    model = resnet50(pretrained=False, num_classes=4,
                     window_sizes=[2], scales=[1.0, 0.5, 0.25]).to(device)
    init_note = warm_start(model, args.init_from) if args.init_from \
        else "random init (ImageNet weights do not fit this architecture)"
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(args.beta1, 0.999))
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    train_loader = make_loader(train_set, args, shuffle=True, drop_last=True)
    # Shuffled: the val set is stored real-then-fake, so a truncated pass with
    # --max_val_batches would otherwise see a single class and every metric
    # would be nan. Metrics over the full set are order-independent, and the
    # val perturbations are seeded per path, so this costs no reproducibility.
    g = torch.Generator(); g.manual_seed(args.seed)
    val_loader = make_loader(val_set, args, shuffle=True, generator=g,
                             persistent=False)

    header = [
        "MLEP 4-output perturbation retrain",
        f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"device={device.type} ({gpu})  arch={args.arch}  amp={use_amp}  "
        f"workers={args.num_threads}  batch={args.batch_size}  lr={args.lr:g}  "
        f"seed={args.seed}  crop={args.cropSize}  params={n_params/1e6:.2f}M",
        f"dataroot: {args.dataroot}  ({len(train_set)} train / {len(val_set)} val samples)",
        f"heads: {list(HEADS)}   variants: {list(VARIANTS)}",
        f"levels: blur sigma {LEVELS['blur']}, jpeg quality {LEVELS['jpeg']}, "
        f"noise sigma {LEVELS['noise']} (8-bit units)",
        f"init: {init_note}",
        "recipe: released MLEP train.py -- Adam(lr 1e-4, 0.9/0.999), BCEWithLogits, "
        f"batch {args.batch_size}, resize {args.loadSize} -> RandomCrop {args.cropSize} "
        f"-> flip, lr*=0.9 every {args.delr_freq} epochs (floor 1e-6), "
        f"early stop after {args.earlystop_epoch} epochs without val improvement",
        "deviations: BN recalibrated before every validation/save; AMP + cuDNN "
        "autotune on; no ImageNet init (incompatible stem)",
        "".join(report),
        "",
    ]

    def write_report(lines, status):
        with open(out, 'w') as f:
            f.write("\n".join(header[:-1] + [f"STATUS: {status}", ""] + lines) + "\n")

    best = dict(ap=-1.0, epoch=-1)
    bad_epochs = 0
    lr_now = args.lr
    log = []
    t_start = time.time()

    for epoch in range(args.niter):
        model.train()
        t0, run_loss, nb = time.time(), 0.0, 0
        for i, (img, tgt) in enumerate(train_loader):
            if args.max_train_steps and i >= args.max_train_steps:
                break
            img = img.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True).float()
            with amp_autocast(use_amp):
                loss = criterion(model(img), tgt)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item(); nb += 1
            if nb % 200 == 0:
                print(f"  epoch {epoch} step {nb}  loss={run_loss/nb:.4f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        train_time = time.time() - t0

        # BN stats left by training are an EMA over the last ~20 batches and are
        # not usable at eval -- see the module docstring.
        if args.bn_recal_batches > 0:
            recalibrate_bn(model, train_loader, args.bn_recal_batches, device,
                           use_amp=use_amp)

        r = evaluate(model, val_loader, device, use_amp, args.max_val_batches)
        ap = r['ai']['ap']
        line = (f"[epoch {epoch}] loss={run_loss/max(nb,1):.4f} lr={lr_now:.2e} "
                f"train={train_time:.0f}s\n      {fmt_eval(r)}")
        print(line, flush=True)
        log.append(line)

        torch.save(model.state_dict(),
                   os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
        if ap > best['ap']:
            best = dict(ap=ap, epoch=epoch, **{h: r[h] for h in HEADS})
            bad_epochs = 0
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'model_epoch_best.pth'))
            log.append(f"      -> new best ai AP {ap:.4f}, saved model_epoch_best.pth")
        else:
            bad_epochs += 1

        # released recipe: decay at the END of every delr_freq-th epoch
        if epoch % args.delr_freq == 0 and epoch != 0:
            for pg in optimizer.param_groups:
                pg['lr'] = max(pg['lr'] * 0.9, 1e-6)
            lr_now = optimizer.param_groups[0]['lr']
            log.append(f"      -> lr decayed to {lr_now:.2e}")

        write_report(log, f"IN PROGRESS -- epoch {epoch+1}/{args.niter}, "
                          f"best ai AP {best['ap']:.4f} @ epoch {best['epoch']}")
        if bad_epochs >= args.earlystop_epoch:
            log.append(f"early stop: {bad_epochs} epochs without improvement")
            break

    torch.save(model.state_dict(), os.path.join(save_dir, 'model_epoch_last.pth'))
    log.append(f"\nbest ai AP {best['ap']:.4f} at epoch {best['epoch']}; "
               f"total {(time.time()-t_start)/3600:.2f} h")
    write_report(log, f"complete -- best ai AP {best['ap']:.4f} @ epoch {best['epoch']}")
    print(f"\nreport -> {out}\ncheckpoints -> {save_dir}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
