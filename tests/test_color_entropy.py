"""Regression tests for ResNet.color_entropy (joint RGB-triple entropy maps).

The existing front-end computes local entropy INDEPENDENTLY PER CHANNEL -- it was
never grayscale -- so its three maps are marginal distributions and carry no
cross-channel information. A 2x2 window can be maximally entropic in R, in G and
in B while containing only two distinct COLOURS. `color_entropy` adds the entropy
of the joint RGB-triple distribution, where two pixels are the same symbol only
when they agree in every channel:

  'joint'      -- append it (4 maps per scale/window, 12 channels by default),
                  leaving the 3 marginal channels of each group bit-identical.
  'joint_only' -- feed only the joint map (3 channels by default).

These tests pin the arithmetic, prove the joint map is genuinely NOT recoverable
from the marginals (test_joint_is_not_a_function_of_the_marginals -- if that one
fails the whole feature is a no-op dressed as a change), check the entropy values
against a brute-force reference, and pin `color_entropy=False` as an
unchanged-by-construction default so the pretrained 9-channel weights still load.

Run directly (no pytest needed):
    python tests/test_color_entropy.py
"""
import math
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from mlep.networks.resnet import resnet18                        # noqa: E402
from mlep.experiments.windows import CONFIGS
from mlep.harness.model import build_model


def build(color_entropy=False, window_sizes=(2,), scales=(1.0, 0.5, 0.25), **kw):
    # the block shuffle is random, so it is off unless a test asks for it
    kw.setdefault('use_rearrange', False)
    return resnet18(pretrained=False, num_classes=1, window_sizes=list(window_sizes),
                    scales=list(scales), color_entropy=color_entropy, **kw).eval()


def conv1_input(model, x):
    """The tensor that actually reaches conv1, i.e. the reconciled entropy stack.
    Independent of the (randomly initialised) weights, so two models built with
    different `color_entropy` are directly comparable."""
    got = {}
    h = model.conv1.register_forward_pre_hook(lambda m, i: got.__setitem__('t', i[0]))
    with torch.no_grad():
        model(x)
    h.remove()
    return got['t']


def quantised_image(batch=2, size=64, levels=256, seed=0):
    """8-bit-quantised random image. The front-end counts DISTINCT values, so
    continuous noise makes every window maximally entropic and every map
    degenerates to a constant -- see tests/test_bn_recalibration.py."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, levels, (batch, 3, size, size), generator=g).float() / levels
    return (x - 0.45) / 0.225        # roughly ImageNet Normalize; affine, so exact
                                     # value-equality (all the front-end reads) holds


def window(rgb):
    """A single 2x2 window from three length-4 channel lists, in raster order."""
    t = torch.tensor(rgb, dtype=torch.float)          # (3, 4)
    return t.view(1, 3, 2, 2)


def brute_force(x, w, mode):
    """Reference joint entropy, computed one window at a time with a Counter over
    RGB triples. Deliberately slow and obvious -- it shares no code with the
    tensorised implementation."""
    b, c, H, W = x.shape
    out = torch.zeros(b, 1, H - w + 1, W - w + 1)
    for i in range(b):
        for r in range(H - w + 1):
            for col in range(W - w + 1):
                block = x[i, :, r:r + w, col:col + w]
                colours = Counter(tuple(block[:, a, d].tolist())
                                  for a in range(w) for d in range(w))
                K = w * w
                if mode == 'unique':
                    out[i, 0, r, col] = len(colours)
                else:
                    out[i, 0, r, col] = -sum((n / K) * math.log2(n / K)
                                             for n in colours.values())
    return out


# --------------------------------------------------------------------------- #

def test_default_is_off_and_unchanged():
    """color_entropy defaults to False and the stack is bit-identical to a model
    built without the kwarg at all -- the backward-compatibility gate."""
    old = resnet18(pretrained=False, num_classes=1, window_sizes=[2],
                   scales=[1.0, 0.5, 0.25], use_rearrange=False).eval()
    new = build(False)
    assert old.color_entropy is False
    assert old.conv1.in_channels == 9 and new.conv1.in_channels == 9
    x = quantised_image()
    assert torch.equal(conv1_input(old, x), conv1_input(new, x))


def test_in_channels_arithmetic():
    """(3 marginals | 3+1 | 1) * windows * scales, times 2 under a concat split."""
    assert build(False).conv1.in_channels == 9
    assert build('joint').conv1.in_channels == 12
    assert build('joint_only').conv1.in_channels == 3
    assert build('joint', window_sizes=(2, 4), scales=(1.0,),
                 window_align='pad').conv1.in_channels == 8
    # composes with the texture split: 2 canvases * 1 window * 3 scales * 4
    assert build('joint', texture_split=True,
                 texture_patch_size=16).conv1.in_channels == 24
    assert build('joint', texture_split=True, texture_patch_size=16,
                 texture_split_mode='diff').conv1.in_channels == 12
    for bad in ('joint-only', 'rgb', True, 3):
        try:
            build(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"color_entropy={bad!r} should have raised")
    assert build(None).color_entropy is False


def test_joint_is_not_a_function_of_the_marginals():
    """THE test. Two windows with byte-identical per-channel entropy but different
    joint entropy: no function of the 3 marginal maps can tell them apart, so if
    this passes the new channel is carrying information the baseline cannot see.

      A: R=[0,0,1,1] G=[0,1,0,1] B=[0,0,0,0] -> colours 000 010 100 110 = 4
      B: R=[0,0,1,1] G=[0,0,1,1] B=[0,0,0,0] -> colours 000 x2, 110 x2   = 2

    Each channel is 'two pairs' (entropy 1.0) in both, and B is flat in both.
    """
    m = build('joint')
    a = window([[0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 0]])
    b = window([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]])
    marg_a, marg_b = m._entropy_map(a, 2), m._entropy_map(b, 2)
    assert torch.equal(marg_a, marg_b), "premise broken: marginals should match"
    assert marg_a.flatten().tolist() == [1.0, 1.0, 0.0]
    joint_a = m._joint_entropy_map(a, 2).item()
    joint_b = m._joint_entropy_map(b, 2).item()
    assert abs(joint_a - 2.0) < 1e-6, joint_a          # 4 colours, uniform
    assert abs(joint_b - 1.0) < 1e-6, joint_b          # 2 colours, uniform
    # and the same pair under 'unique'
    u = build('joint', entropy_mode='unique')
    assert u._joint_entropy_map(a, 2).item() == 4.0
    assert u._joint_entropy_map(b, 2).item() == 2.0


def test_joint_entropy_matches_brute_force():
    """Tensorised K x K same-colour matrix vs a per-window Counter, both modes,
    both an even and an odd window size."""
    x = quantised_image(batch=2, size=12, levels=3, seed=1)   # 3 levels -> real ties
    for mode in ('shannon', 'unique'):
        for w in (2, 3):
            m = build('joint', entropy_mode=mode)
            got = m._joint_entropy_map(x, w)
            want = brute_force(x, w, mode)
            assert got.shape == want.shape == (2, 1, 12 - w + 1, 12 - w + 1)
            assert torch.allclose(got, want, atol=1e-6), \
                f"{mode} w={w}: max err {(got - want).abs().max().item()}"


def test_joint_map_is_shared_across_channels():
    """One map for all three channels, not three -- it is a joint statistic."""
    m = build('joint')
    e = m._joint_entropy_map(quantised_image(batch=3, size=16), 2)
    assert e.shape == (3, 1, 15, 15)


def test_channel_layout_and_bit_identical_marginals():
    """Layout is [R, G, B, joint] per (scale, window) group, so channels 0..2 of
    group s are exactly what baseline_2x2 puts at channels 3s..3s+2. Bit-exact,
    which is only reachable if the joint map was appended rather than mixed in."""
    base, joint, only = build(False), build('joint'), build('joint_only')
    x = quantised_image()
    sb, sj, so = conv1_input(base, x), conv1_input(joint, x), conv1_input(only, x)
    assert sb.shape[1] == 9 and sj.shape[1] == 12 and so.shape[1] == 3
    assert sb.shape[2:] == sj.shape[2:] == so.shape[2:]
    for s in range(3):                                   # three scales
        assert torch.equal(sj[:, 4 * s:4 * s + 3], sb[:, 3 * s:3 * s + 3]), s
        # ...and the 4th channel of each group is the joint map, which is exactly
        # what joint_only feeds on its own.
        assert torch.equal(sj[:, 4 * s + 3], so[:, s]), s


def test_normalize_entropy_bounds_the_joint_map():
    """normalize_entropy divides by log2(K) (shannon) / K (unique) here too, so
    the joint channel enters conv1 on the same [0, 1] scale as the marginals."""
    x = quantised_image(size=32)
    for mode, hi in (('shannon', 1.0), ('unique', 1.0)):
        m = build('joint', entropy_mode=mode, normalize_entropy=True)
        e = m._joint_entropy_map(x, 2)
        assert e.min() >= 0.0 and e.max() <= hi + 1e-6, (mode, e.max().item())
    # unnormalised shannon is bounded by log2(4) = 2, unique by 4
    assert build('joint')._joint_entropy_map(x, 2).max() <= 2.0 + 1e-6
    assert build('joint', entropy_mode='unique')._joint_entropy_map(x, 2).max() <= 4.0


def test_full_forward_runs():
    """End to end through conv1 -> layer1/2 -> avgpool -> fc1, including combined
    with the texture split and multi-window padding (the shapes all have to line
    up for real, not just at the stack)."""
    x = quantised_image(batch=2, size=64)
    cases = [dict(color_entropy='joint'),
             dict(color_entropy='joint_only'),
             dict(color_entropy='joint', window_sizes=(2, 4), window_align='pad'),
             dict(color_entropy='joint', texture_split=True, texture_patch_size=16),
             # the default block shuffle ON, since every real config uses it
             dict(color_entropy='joint', use_rearrange=True)]
    for kw in cases:
        m = build(**kw)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 1), (kw, out.shape)
        assert torch.isfinite(out).all(), kw


def test_config_kwargs_reach_the_model():
    """build_model's kwargs are a fixed WHITELIST: omit color_entropy there and the
    new configs train as silent byte-identical clones of baseline_2x2, reporting a
    null effect that is really a plumbing bug. This is the test that catches it."""
    expected = {'color_joint_2x2': ('joint', 12),
                'color_jointonly_2x2': ('joint_only', 3),
                'color_joint_train_jpeg': ('joint', 12),
                'color_joint_train_webp_p1': ('joint', 12)}
    for name, (mode, ch) in expected.items():
        assert name in CONFIGS, f"{name} missing from CONFIGS"
        m = build_model('resnet18', CONFIGS[name], torch.device('cpu'))
        assert m.color_entropy == mode, (name, m.color_entropy)
        assert m.conv1.in_channels == ch, (name, m.conv1.in_channels)
    b = build_model('resnet18', CONFIGS['baseline_2x2'], torch.device('cpu'))
    assert b.color_entropy is False and b.conv1.in_channels == 9


def test_train_aug_pairs_are_controlled():
    """The two train_* colour configs must reuse their baselines' aug dicts
    verbatim, or the paired deltas they exist to measure are confounded."""
    assert CONFIGS['color_joint_train_jpeg']['train_aug'] == \
        CONFIGS['baseline_train_jpeg']['train_aug']
    assert CONFIGS['color_joint_train_webp_p1']['train_aug'] == \
        CONFIGS['baseline_train_webp_p1']['train_aug']


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok    {fn.__name__}")
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
