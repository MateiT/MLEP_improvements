"""Scoring a model, and the train/recalibrate/evaluate loop every experiment shares."""
# Split out of the original mlep/experiments/windows.py. The sweep's CONFIGS and
# main() live in mlep.experiments.windows; everything reusable lives here, so
# consumers no longer need in-function imports to dodge a circular dependency.

import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score

from mlep.data import get_dataset
from mlep.harness.data import EVAL_SCENARIOS, make_loader, make_opt
from mlep.harness.device import amp_autocast
from mlep.harness.model import build_model, recalibrate_bn

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


