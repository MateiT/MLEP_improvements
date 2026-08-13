"""Test the blur / jpeg / noise heads of a 4-output MLEP model on TestDatasets.

    python scripts/eval_pert4_heads.py                       # 500/generator/label
    python scripts/eval_pert4_heads.py --per_label 100       # quick pass

TestDatasets carries no degradation labels, so we make them: every sampled image
is rendered at ten conditions -- clean plus the nine training levels -- and the
ground truth is what we applied. Rendering all ten from one source makes the
comparison PAIRED (same image clean vs degraded), which removes image-to-image
variance, and costs one file open instead of ten.

    clean | blur sigma 1,3,5 | jpeg quality 90,70,50 | noise sigma 1,3,5

Per generator and level: detection rate = fraction of degraded images whose own
head fires (p > 0.5), and ROC-AUC of those against the SAME images clean. Per
generator and head we also report the clean false-positive rate, without which a
detection rate cannot be read -- a head stuck at "yes" scores 100% on all three
of its levels.

The 26 generators are then averaged UNWEIGHTED: crn (6382 images) and seeingdark
(180) each contribute exactly 1/26.

Two caveats the report repeats, because they change how the numbers read:

  * JPEG-source contamination. Nine generators ship JPEG files -- eight with
    0_real entirely JPEG (dalle, glide x3, guided, ldm x3) and whichfaceisreal
    with both classes. Their "clean" negatives are already compressed, so the
    jpeg head correctly fires on them and the jpeg row is penalised through no
    fault of the model. The jpeg rows are therefore reported twice: over all 26
    generators, and over the clean-source ones only. Blur and noise are
    unaffected.
  * san (219/label) and seeingdark (180/label) cannot reach --per_label 500.
    They still carry a full 1/26 weight, as asked, but their per-generator
    estimates have roughly twice the sampling noise of the others.
"""
import argparse
import os
import random
import sys
import time
import zlib

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import gaussian_blur, pil_jpg                    # noqa: E402
from experiment_windows import amp_autocast, get_device, setup_cuda_perf  # noqa: E402
from networks.resnet import resnet50                                # noqa: E402

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')
JPEG_EXT = ('.jpg', '.jpeg')

# condition index -> (label, head index, kind, level). 0 is the shared clean ref.
CONDITIONS = [
    ('clean',      None, None,    None),
    ('blur_s1',       1, 'blur',   1.0),
    ('blur_s3',       1, 'blur',   3.0),
    ('blur_s5',       1, 'blur',   5.0),
    ('jpeg_q90',      2, 'jpeg',    90),
    ('jpeg_q70',      2, 'jpeg',    70),
    ('jpeg_q50',      2, 'jpeg',    50),
    ('noise_s1',      3, 'noise',  1.0),
    ('noise_s3',      3, 'noise',  3.0),
    ('noise_s5',      3, 'noise',  5.0),
]
NC = len(CONDITIONS)
HEADS = ('ai', 'blur', 'jpeg', 'noise')


def find_generators(root):
    """(set, generator) -> {label: [paths]}. Handles both the flat layout and the
    nested one (progan/stylegan/stylegan2/cyclegan/ddpm keep per-category
    subdirectories); nested generators are pooled so each counts as ONE unit."""
    gens = {}
    for s in sorted(os.listdir(root)):
        sd = os.path.join(root, s)
        if not os.path.isdir(sd):
            continue
        for g in sorted(os.listdir(sd)):
            gd = os.path.join(sd, g)
            if not os.path.isdir(gd):
                continue
            per = {'0_real': [], '1_fake': []}
            for dp, _, fn in os.walk(gd):
                lab = ('0_real' if os.sep + '0_real' in dp + os.sep else
                       '1_fake' if os.sep + '1_fake' in dp + os.sep else None)
                if lab is None:
                    continue
                per[lab] += [os.path.join(dp, f) for f in fn
                             if f.lower().endswith(IMG_EXT)]
            if per['0_real'] or per['1_fake']:
                gens[f"{s}/{g}"] = per
    return gens


class TenWayDataset(torch.utils.data.Dataset):
    """One item = one source image rendered at all ten conditions."""

    def __init__(self, samples, load=256, crop=224, seed=0):
        self.samples = samples                     # (path, gen_idx, ai_label)
        self.resize = transforms.Resize((load, load))
        self.crop = transforms.CenterCrop(crop)
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, gi, ai = self.samples[i]
        with Image.open(path) as im:
            base = self.resize(im.convert('RGB'))
        out = []
        for ci, (_, _, kind, lvl) in enumerate(CONDITIONS):
            if kind is None:
                img = base
            else:
                arr = np.array(base)
                if kind == 'blur':
                    gaussian_blur(arr, lvl)                  # in place, per channel
                elif kind == 'jpeg':
                    arr = pil_jpg(arr, lvl)
                else:
                    rng = np.random.default_rng(
                        (zlib.crc32(path.encode()) + self.seed + ci) & 0xFFFFFFFF)
                    arr = np.clip(arr.astype(np.float32)
                                  + rng.normal(0.0, lvl, arr.shape),
                                  0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
            out.append(self.norm(self.to_tensor(self.crop(img))))
        return torch.stack(out), gi, ai


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt',
                    default='checkpoints/pert4_decay/model_epoch_best.pth')
    ap.add_argument('--dataroot', default='datasets/TestDatasets')
    ap.add_argument('--per_label', type=int, default=500)
    ap.add_argument('--batch_sources', type=int, default=8,
                    help='sources per forward; images per forward is 10x this')
    ap.add_argument('--num_threads', type=int, default=8)
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--out', default='')
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = get_device(''); gpu = setup_cuda_perf(dev); use_amp = dev.type == 'cuda'

    gens = find_generators(a.dataroot)
    names = sorted(gens)
    samples, meta = [], []
    for gi, g in enumerate(names):
        rnd = random.Random(a.seed + gi)
        n_r = n_f = 0
        jpeg_src = 0
        for lab, ai in (('0_real', 0), ('1_fake', 1)):
            files = sorted(gens[g][lab])
            take = files if len(files) <= a.per_label \
                else rnd.sample(files, a.per_label)
            for p in take:
                samples.append((p, gi, ai))
                jpeg_src += p.lower().endswith(JPEG_EXT)
            if ai == 0:
                n_r = len(take)
            else:
                n_f = len(take)
        meta.append(dict(name=g, n_real=n_r, n_fake=n_f, jpeg_src=jpeg_src,
                         short=(n_r < a.per_label or n_f < a.per_label)))

    ds = TenWayDataset(samples, seed=a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch_sources, shuffle=False,
                                     num_workers=a.num_threads,
                                     pin_memory=(dev.type == 'cuda'))

    model = resnet50(num_classes=4, window_sizes=[2],
                     scales=[1.0, 0.5, 0.25]).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=True))
    model.eval()

    P = np.zeros((len(samples), NC, 4), dtype=np.float32)
    G = np.array([s[1] for s in samples])
    Y = np.array([s[2] for s in samples])
    t0, done = time.time(), 0
    with torch.no_grad():
        for x, gi, ai in dl:
            b = x.shape[0]
            x = x.view(b * NC, *x.shape[2:]).to(dev, non_blocking=True)
            with amp_autocast(use_amp):
                o = model(x).float()
            P[done:done + b] = torch.sigmoid(o).view(b, NC, 4).cpu().numpy()
            done += b
            if done % (a.batch_sources * 50) == 0:
                el = time.time() - t0
                print(f"  {done}/{len(samples)} sources  {done/el:.0f} src/s  "
                      f"ETA {(len(samples)-done)/max(done/el,1e-9)/60:.0f} min",
                      flush=True)
    print(f"scored {len(samples)} sources x {NC} conditions in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)

    # ---------------- per generator, per level ---------------- #
    per_gen = {}
    for gi, m in enumerate(meta):
        sel = G == gi
        row = {}
        for ci, (cname, head, _, _) in enumerate(CONDITIONS):
            if head is None:
                continue
            pos, neg = P[sel, ci, head], P[sel, 0, head]
            row[cname] = dict(
                det=float((pos > 0.5).mean()),
                auc=float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))],
                                        np.r_[neg, pos])))
        for h in (1, 2, 3):
            row[f'fpr_{HEADS[h]}'] = float((P[sel, 0, h] > 0.5).mean())
        # ai head per condition, free of charge
        for ci, (cname, _, _, _) in enumerate(CONDITIONS):
            y = Y[sel]
            row[f'ai_ap_{cname}'] = (float(average_precision_score(y, P[sel, ci, 0]))
                                     if len(np.unique(y)) > 1 else float('nan'))
        per_gen[m['name']] = row

    clean_src = [m['name'] for m in meta if m['jpeg_src'] == 0]

    def macro(key, subset=None):
        ns = subset or names
        return float(np.mean([per_gen[n][key] for n in ns]))

    # ---------------- report ---------------- #
    L = []
    L.append("MLEP 4-output model -- blur / jpeg / noise head test on TestDatasets")
    L.append(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"checkpoint: {a.ckpt}")
    L.append(f"device={dev.type} ({gpu})  amp={use_amp}  seed={a.seed}")
    L.append(f"sampling: {a.per_label} images per generator per label -> "
             f"{len(samples)} sources x {NC} conditions = {len(samples)*NC} "
             f"forward passes")
    L.append(f"generators: {len(names)} (each weighted 1/{len(names)}, "
             f"regardless of image count)")
    L.append("geometry: resize 256 -> degrade -> center crop 224 (as in training)")
    L.append("levels: blur sigma 1/3/5, jpeg quality 90/70/50, noise sigma 1/3/5")
    L.append("")
    L.append("detection rate = fraction of degraded images whose own head gives "
             "p > 0.5")
    L.append("roc_auc        = that head, degraded vs the SAME images clean (paired)")
    L.append("")

    L.append("=" * 62)
    L.append("HEADLINE -- macro-average over all %d generators" % len(names))
    L.append("=" * 62)
    L.append(f"{'level':<12}{'detection rate':>16}{'roc_auc':>12}")
    L.append("-" * 62)
    for cname, head, _, _ in CONDITIONS:
        if head is None:
            continue
        L.append(f"{cname:<12}{macro_fmt(per_gen, names, cname, 'det'):>16}"
                 f"{macro_fmt(per_gen, names, cname, 'auc'):>12}")
    L.append("=" * 62)
    L.append("")
    L.append("clean false-positive rate (companion to the above -- a head stuck "
             "at 'yes' would score 100% on every level)")
    for h in (1, 2, 3):
        L.append(f"    {HEADS[h]:<6} FPR on clean images: "
                 f"{macro('fpr_'+HEADS[h])*100:6.2f}%")
    L.append("")

    L.append("-" * 62)
    L.append("JPEG rows recomputed over clean-source generators only")
    L.append(f"  excluded ({len(names)-len(clean_src)}): "
             + ", ".join(m['name'] for m in meta if m['jpeg_src'] > 0))
    L.append("-" * 62)
    L.append(f"{'level':<12}{'detection rate':>16}{'roc_auc':>12}")
    for cname, head, _, _ in CONDITIONS:
        if head != 2:
            continue
        L.append(f"{cname:<12}{macro_fmt(per_gen,clean_src,cname,'det'):>16}"
                 f"{macro_fmt(per_gen,clean_src,cname,'auc'):>12}")
    L.append(f"    jpeg FPR on clean images: "
             f"{macro('fpr_jpeg', clean_src)*100:6.2f}%")
    L.append("")

    L.append("=" * 118)
    L.append("PER GENERATOR -- detection rate (%), and the two short generators")
    L.append("=" * 118)
    hdr = f"{'generator':<34}{'n':>6}" + "".join(
        f"{c[0].replace('_',''):>9}" for c in CONDITIONS if c[1]) + f"{'fprB':>7}{'fprJ':>7}{'fprN':>7}"
    L.append(hdr)
    L.append("-" * 118)
    for m in meta:
        r = per_gen[m['name']]
        cells = "".join(f"{r[c[0]]['det']*100:9.1f}" for c in CONDITIONS if c[1])
        flag = " SHORT" if m['short'] else (" JPEGsrc" if m['jpeg_src'] else "")
        L.append(f"{m['name']:<34}{m['n_real']+m['n_fake']:>6}{cells}"
                 f"{r['fpr_blur']*100:7.1f}{r['fpr_jpeg']*100:7.1f}"
                 f"{r['fpr_noise']*100:7.1f}{flag}")
    L.append("=" * 118)
    L.append("")

    L.append("SECONDARY -- ai head AP per condition (macro-average), for reference")
    L.append("-" * 62)
    for cname, _, _, _ in CONDITIONS:
        vals = [per_gen[n][f'ai_ap_{cname}'] for n in names]
        vals = [v for v in vals if v == v]
        L.append(f"    {cname:<12} ai AP = {np.mean(vals):.4f}  ({len(vals)} generators)")
    L.append("")
    L.append("Notes")
    short = [f"{m['name']} ({min(m['n_real'], m['n_fake'])}/label)"
             for m in meta if m['short']]
    L.append("- Every generator counts 1/%d." % len(names)
             + (" Below the %d/label target, hence noisier: %s."
                % (a.per_label, "; ".join(short)) if short else ""))
    L.append("- Nested generators (ddpm, progan, stylegan, stylegan2, cyclegan) "
             "pool their category subdirectories and count as one generator.")
    L.append("- Degradation is applied AFTER the resize to 256, so a level means "
             "the same thing on a 1024px source as on a 256px one.")

    out = a.out or f"results/pert4_headtest_{time.strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nreport -> {out}")
    return 0


def macro_fmt(per_gen, names, cname, key):
    return f"{np.mean([per_gen[n][cname][key] for n in names])*100:.2f}" \
        if key == 'det' else f"{np.mean([per_gen[n][cname][key] for n in names]):.4f}"


if __name__ == '__main__':
    sys.exit(main())
