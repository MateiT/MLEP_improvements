"""Regression tests for recalibrate_bn() (the acc=0.50 fix).

Training leaves BatchNorm's running_mean / running_var as a momentum-0.1 EMA over
roughly the last ~20 batches, collected while the weights were still moving. At
batch 16 that estimate is off by enough that switching to model.eval() displaces
every logit by 10-25, so every prediction lands on the same side of zero and
accuracy pins at the class prior (~0.50) no matter how well the model separates
the classes. Measured on baseline_2x2 at the sweep's settings, same weights:

    model.eval() as-is        logit mean +24.16   acc 0.4938   AP 0.8057
    after recalibrate_bn()    logit mean  -0.25   acc 0.9762   AP 0.9985

recalibrate_bn() resets the stats, flips BN to momentum=None (cumulative average)
and runs n_batches of forward passes in train mode, so the stored statistics
become an honest mean over the data instead of a lagging EMA.

These tests use synthetic tensors only, so they run anywhere and need no dataset.

Run directly (no pytest needed):
    python tests/test_bn_recalibration.py
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_windows import recalibrate_bn, best_threshold_acc   # noqa: E402
from networks.resnet import resnet18                                # noqa: E402

import numpy as np                                                  # noqa: E402


def build():
    # use_rearrange=False: the block shuffle is irrelevant here and makes the
    # activations (and therefore the expected BN stats) harder to reason about.
    return resnet18(pretrained=False, num_classes=1, window_sizes=[2],
                    scales=[1.0, 0.5], use_rearrange=False)


def batches(n, batch=4, size=48, seed=0):
    """A fixed synthetic 'dataset' of (image, label) pairs, quantised to 8-bit
    steps because the entropy front-end counts DISTINCT pixel values -- continuous
    noise makes every window maximally entropic and the maps degenerate."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        x = torch.randint(0, 16, (batch, 3, size, size), generator=g).float() / 16.0
        out.append((x, torch.randint(0, 2, (batch,), generator=g).float()))
    return out


def bn_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


def test_stats_match_measured_activations():
    """After recalibration every BN's stored stats should equal the statistics of
    the activations actually flowing into it."""
    model = build()
    data = batches(6)

    # Put something plausible-but-wrong in the running stats, the way a training
    # run does, then verify recalibration replaces it with the true values.
    for m in bn_modules(model):
        m.running_mean.fill_(7.0)
        m.running_var.fill_(0.01)

    seen = recalibrate_bn(model, data, n_batches=len(data), device=torch.device('cpu'))
    assert seen == len(data), f"expected {len(data)} batches consumed, got {seen}"

    # Measure the true per-channel statistics over the same data by hooking every
    # BN's input. The replay has to run in TRAIN mode, the same regime the stats
    # were collected in: in train mode each BN normalises with its own batch
    # statistics, so a layer further down sees different activations than it would
    # in eval mode. Only bn1 -- which has no BN upstream -- is regime-independent.
    # momentum=0 freezes the running stats so the replay cannot disturb what it is
    # measuring (running = (1-0)*running + 0*batch).
    captured = {}

    def make_hook(name):
        def hook(mod, inputs, _out):
            captured.setdefault(name, []).append(inputs[0].detach())
        return hook

    handles = [m.register_forward_hook(make_hook(n))
               for n, m in model.named_modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bn_modules(model):
        m.momentum = 0.0
    model.train()
    with torch.no_grad():
        for x, _ in data:
            model(x)
    for h in handles:
        h.remove()

    checked = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.modules.batchnorm._BatchNorm):
            continue
        a = torch.cat(captured[name], dim=0)
        mean = a.mean(dim=(0, 2, 3))
        var = a.var(dim=(0, 2, 3), unbiased=True)   # BN stores the unbiased variance
        assert torch.allclose(mod.running_mean, mean, atol=1e-4, rtol=1e-3), \
            f"{name}: running_mean does not match the measured activation mean"
        assert torch.allclose(mod.running_var, var, atol=1e-4, rtol=1e-2), \
            f"{name}: running_var does not match the measured activation variance"
        checked += 1
    assert checked >= 9, f"expected to check every BN layer, only saw {checked}"
    print(f"ok: {checked} BN layers match their measured activation statistics")


def test_restores_momentum_and_mode():
    """recalibrate_bn() has to flip momentum and training mode to do its job; it
    must not leave either changed, or the caller's next training step silently
    switches to a cumulative average."""
    model = build()
    for m in bn_modules(model):
        m.momentum = 0.1
    model.eval()
    recalibrate_bn(model, batches(2), n_batches=2, device=torch.device('cpu'))
    assert not model.training, "model was left in train mode"
    assert all(m.momentum == 0.1 for m in bn_modules(model)), "momentum not restored"

    model.train()
    recalibrate_bn(model, batches(2), n_batches=2, device=torch.device('cpu'))
    assert model.training, "model was left in eval mode"
    print("ok: momentum and training mode are restored")


def test_n_batches_caps_the_pass():
    """The batch cap is what keeps this cheap (~2 s), so it has to be honoured
    even when the loader could supply more."""
    model = build()
    assert recalibrate_bn(model, batches(10), 3, torch.device('cpu')) == 3
    # 0 disables it entirely -- the BN_RECAL=0 escape hatch in run.sh.
    before = [m.running_mean.clone() for m in bn_modules(model)]
    assert recalibrate_bn(model, batches(10), 0, torch.device('cpu')) == 0
    for b, m in zip(before, bn_modules(model)):
        assert torch.equal(b, m.running_mean), "n_batches=0 still touched the stats"
    print("ok: n_batches caps the pass and 0 disables it")


def test_fixes_a_displaced_decision_boundary():
    """End-to-end: give the model BN stats that displace every logit to one side,
    then check recalibration brings them back. This is the observed failure in
    miniature -- perfect ranking, chance accuracy."""
    model = build()
    data = batches(8)
    cpu = torch.device('cpu')
    recalibrate_bn(model, data, n_batches=len(data), device=cpu)
    good = [(m.running_mean.clone(), m.running_var.clone()) for m in bn_modules(model)]

    def logits():
        model.eval()
        with torch.no_grad():
            return torch.cat([model(x).flatten() for x, _ in data])

    base = logits()
    # Corrupt the stats the way a lagging EMA does: the means drift and the
    # variances come out too low.
    for m in bn_modules(model):
        m.running_mean.add_(5.0)
        m.running_var.mul_(0.25)
    displaced = logits()
    shift = (displaced.mean() - base.mean()).abs()
    # The failure is defined by the displacement swamping the SPREAD of the scores:
    # that is what leaves the ranking (and so AP) intact while every prediction
    # lands on one side of zero. Absolute logit scale is meaningless on an
    # untrained model, so compare the two.
    assert shift > base.std(), (f"test setup did not displace the logits past their "
                                f"own spread (shift={shift:.3f}, std={base.std():.3f})")

    recalibrate_bn(model, data, n_batches=len(data), device=cpu)
    for (gm, gv), m in zip(good, bn_modules(model)):
        assert torch.allclose(m.running_mean, gm, atol=1e-4), "means not recovered"
        assert torch.allclose(m.running_var, gv, atol=1e-4), "variances not recovered"
    assert torch.allclose(logits(), base, atol=1e-4), "logits not recovered"
    print(f"ok: a {shift:.2f} logit displacement ({shift / base.std():.1f}x the "
          f"score spread) is undone by recalibration")


def test_best_threshold_acc():
    """The acc_best column is what makes this bug visible in a result file, so it
    has to be right -- including the all-one-side case that defines the bug."""
    y = np.array([0, 0, 1, 1])
    # Perfectly ranked but every score below the 0.5 sigmoid threshold: accuracy
    # at a fixed cut is 0.5, yet a threshold exists that gets everything right.
    assert best_threshold_acc(y, np.array([-9.0, -8.0, -7.0, -6.0])) == 1.0
    # Only the higher-score-is-fake direction is swept, so an inverted ranking
    # scores at the class prior rather than 1.0. That matches how AP reads it: a
    # model that ranks backwards is wrong, not secretly right.
    assert best_threshold_acc(y, np.array([1.0, 0.0, -1.0, -2.0])) == 0.5
    # Brute-force cross-check on random data.
    rng = np.random.default_rng(0)
    for _ in range(50):
        yt = rng.integers(0, 2, 40)
        s = rng.normal(size=40)
        # Candidate thresholds: every score, plus one past each end so the
        # all-negative and all-positive cuts are covered too.
        cuts = np.concatenate((s, [s.min() - 1, s.max() + 1]))
        brute = max((yt == (s > t)).mean() for t in cuts)
        assert abs(best_threshold_acc(yt, s) - brute) < 1e-12
    print("ok: best_threshold_acc matches a brute-force sweep")


if __name__ == '__main__':
    torch.manual_seed(0)
    for fn in (test_stats_match_measured_activations,
               test_restores_momentum_and_mode,
               test_n_batches_caps_the_pass,
               test_fixes_a_displaced_decision_boundary,
               test_best_threshold_acc):
        fn()
    print("\nAll BN recalibration tests passed.")
