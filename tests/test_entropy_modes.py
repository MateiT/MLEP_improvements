"""Regression tests for the alternative entropy definitions (experiment group 1).

`entropy_mode` used to be 'shannon' or 'unique'. It now also accepts
'renyi_<alpha>', 'tsallis_<q>', 'perm', and a LIST of any of those (whose maps
are concatenated on the channel dim). The risk in that change is twofold:

  1. the new formulas are wrong -- so every value is checked against a
     brute-force reference computed straight from the definition;
  2. the old path moved -- so the tests below pin that 'shannon' still produces
     bit-identical maps and the same conv1.in_channels, which is what keeps the
     pretrained weights and every existing config valid.

Renyi at alpha -> 1 and Tsallis at q -> 1 both converge to Shannon; that limit is
used as an independent check that the two families are parameterised correctly.

Run directly (no pytest needed):
    python tests/test_entropy_modes.py
"""
import math
import os
import sys
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.resnet import (entropy_max, parse_entropy_mode,        # noqa: E402
                             resnet18)
from experiments.entropy import ENTROPY_CONFIGS                      # noqa: E402


def build(mode='shannon', window_sizes=(2,), scales=(1.0,), **kw):
    kw.setdefault('use_rearrange', False)          # the shuffle is random
    return resnet18(pretrained=False, num_classes=1, entropy_mode=mode,
                    window_sizes=list(window_sizes), scales=list(scales), **kw).eval()


def quantised_image(batch=2, size=32, levels=8, seed=0):
    """Few levels on purpose: with 256 levels every 2x2 window holds 4 distinct
    values and all the entropies collapse to their maximum, which would make the
    reference comparisons below pass vacuously."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, levels, (batch, 3, size, size), generator=g).float() / levels


def conv1_input(model, x):
    got = {}
    h = model.conv1.register_forward_pre_hook(lambda m, i: got.__setitem__('t', i[0]))
    with torch.no_grad():
        model(x)
    h.remove()
    return got['t']


def reference_entropy(vals, kind, param):
    """The textbook definition, from a plain Counter of the window's values."""
    n = len(vals)
    ps = [c / n for c in Counter(vals).values()]
    if kind == 'unique':
        return float(len(ps))
    if kind == 'shannon':
        return -sum(p * math.log2(p) for p in ps)
    if kind == 'renyi':
        return math.log2(sum(p ** param for p in ps)) / (1.0 - param)
    if kind == 'tsallis':
        return (1.0 - sum(p ** param for p in ps)) / (param - 1.0)
    raise AssertionError(kind)


def windows(x, w):
    """Every wxw window of one (H, W) plane, as a list of value tuples."""
    H, W = x.shape
    return [tuple(x[i:i + w, j:j + w].flatten().tolist())
            for i in range(H - w + 1) for j in range(W - w + 1)]


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def test_parse_modes():
    assert parse_entropy_mode('shannon') == ('shannon', None)
    assert parse_entropy_mode('unique') == ('unique', None)
    assert parse_entropy_mode('perm') == ('perm', None)
    assert parse_entropy_mode('renyi_0.5') == ('renyi', 0.5)
    assert parse_entropy_mode('tsallis_2') == ('tsallis', 2.0)


def test_parse_rejects_nonsense():
    for bad in ('renyi', 'renyi_x', 'renyi_1', 'tsallis_1', 'gibbs', ''):
        try:
            parse_entropy_mode(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")


def test_entropy_max_matches_the_definitions():
    assert entropy_max('shannon', None, 4) == 2.0
    assert entropy_max('unique', None, 4) == 4
    assert abs(entropy_max('renyi', 2.0, 4) - 2.0) < 1e-9      # uniform -> log2 K
    assert abs(entropy_max('tsallis', 2.0, 4) - 0.75) < 1e-9   # (1 - K^(1-q))/(q-1)
    assert abs(entropy_max('perm', None, 4) - math.log2(6)) < 1e-9


# --------------------------------------------------------------------------- #
# values
# --------------------------------------------------------------------------- #
def test_maps_match_a_brute_force_reference():
    # w=3, i.e. the GENERAL path. The 2x2 Shannon map deliberately keeps the
    # released lookup table instead (see test_the_2x2_shannon_fast_path_is_intact).
    x = quantised_image(batch=1, size=16, levels=4)
    for mode in ('shannon', 'renyi_0.5', 'renyi_2', 'renyi_4',
                 'tsallis_0.5', 'tsallis_2', 'tsallis_4', 'unique'):
        kind, param = parse_entropy_mode(mode)
        m = build(mode, window_sizes=(3,))
        got = m._entropy_map(x, 3)[0, 0]
        want = windows(x[0, 0], 3)
        assert got.numel() == len(want), mode
        flat = got.flatten().tolist()
        for k in (0, 7, 51, len(want) - 1):
            ref = reference_entropy(want[k], kind, param)
            assert abs(flat[k] - ref) < 1e-4, f"{mode}: {flat[k]} vs {ref}"


def test_renyi_and_tsallis_approach_shannon_as_the_parameter_approaches_one():
    # Renyi is defined here with log2, so it converges to Shannon in bits.
    # Tsallis has no logarithm, so its q -> 1 limit is Shannon in NATS -- a fixed
    # factor ln2 below the bit-valued maps. That is the standard definition and
    # a constant scale is invisible to both BatchNorm and the sklearn pipelines,
    # but it is pinned here so nobody "fixes" one family into the other's units.
    x = quantised_image(batch=1, size=16, levels=4)
    ref = build('shannon', window_sizes=(3,))._entropy_map(x, 3)
    for mode, scale in (('renyi_1.0001', 1.0), ('tsallis_1.0001', math.log(2))):
        got = build(mode, window_sizes=(3,))._entropy_map(x, 3)
        assert (got - scale * ref).abs().max() < 1e-2, mode


def test_permutation_entropy_is_bounded_and_uses_six_patterns():
    x = quantised_image(batch=1, size=16, levels=64)
    e = build('perm', window_sizes=(3,))._entropy_map(x, 3)
    assert e.min() >= 0.0
    # Only 6 of the 8 three-bit codes are realisable orderings, so log2(6) bounds
    # the map -- if a spurious code were ever counted this would exceed it.
    assert e.max() <= math.log2(6) + 1e-5


def test_permutation_entropy_of_a_monotone_ramp_is_zero():
    # Every triplet along a ramp has the same ordering, so H = 0 exactly.
    ramp = torch.arange(64.).view(1, 1, 8, 8).repeat(1, 3, 1, 1) / 64.
    ramp = ramp + torch.arange(8.).view(1, 1, 1, 8) / 64.
    e = build('perm', window_sizes=(3,))._entropy_map(ramp, 3)
    assert e.abs().max() < 1e-5


# --------------------------------------------------------------------------- #
# compatibility: nothing that existed may move
# --------------------------------------------------------------------------- #
def test_the_2x2_shannon_fast_path_is_intact():
    """What the released weights were trained on, including its 0.8 rounding of
    log2(3) - 2/3 = 0.8113. Adding entropy modes must not silently 'fix' that."""
    x = quantised_image(size=24, levels=3)
    got = build('shannon')._entropy_map(x, 2)
    legacy = {0.0, 0.8, 1.0, 1.5, 2.0}
    seen = set(round(v, 4) for v in got.flatten().tolist())
    assert seen <= legacy, sorted(seen - legacy)
    assert 0.8 in seen, "the rounded three-same case never occurred; test is blind"


def test_the_2x2_lookup_equals_the_general_path():
    """The w=2 shortcut must be a speed optimisation only. A 2x2 window has just
    five possible value patterns, so the shortcut classifies the window and reads
    the answer off a table; here that table is checked against the sort-based
    general path, value for value, on real windows."""
    x = quantised_image(batch=1, size=24, levels=3)      # ties are the point
    for mode in ('unique', 'renyi_0.5', 'renyi_2', 'renyi_4',
                 'tsallis_0.5', 'tsallis_2', 'tsallis_4'):
        m = build(mode, window_sizes=(2,))
        fast = m._entropy_map(x, 2)
        patches = x.unfold(2, 2, 1).unfold(3, 2, 1).contiguous()
        slow = m._patch_entropy(patches.view(1, 3, -1, 4), 4, mode)
        assert (fast.flatten() - slow.flatten()).abs().max() < 1e-6, mode


def test_the_2x2_lookup_only_ever_hits_the_five_reachable_slots():
    """Four pixels can be equal in 0, 1, 2, 3 or 6 of their six pairs -- 4 and 5
    are impossible. Those two slots hold nan, so if the counting were ever wrong
    the maps would go nan rather than quietly return a neighbouring value."""
    m = build('renyi_2', window_sizes=(2,))
    table = m.entropy_table_2x2('renyi', 2.0)
    assert len(table) == 7
    assert all(table[i] != table[i] for i in (4, 5))          # nan
    assert all(table[i] == table[i] for i in (0, 1, 2, 3, 6))
    for levels in (2, 3, 8, 256):                              # ties to none
        e = m._entropy_map(quantised_image(batch=2, size=16, levels=levels), 2)
        assert torch.isfinite(e).all(), levels


def test_the_2x2_lookup_did_not_touch_shannon():
    """Shannon still goes through the released categorisation, not the table."""
    x = quantised_image(size=24, levels=3)
    m = build('shannon')
    assert (m._entropy_map(x, 2) - m._entropy_2x2_shannon(x)).abs().max() == 0.0
    assert m.entropy_table_2x2('shannon', None)[3] == 0.8      # the rounded case


def test_single_mode_keeps_the_channel_count():
    for mode in ('shannon', 'unique', 'renyi_2', 'tsallis_0.5'):
        m = build(mode, scales=(1.0, 0.5, 0.25))
        assert m.conv1.in_channels == 9, mode


def test_a_list_of_modes_concatenates_them():
    m = build(['shannon', 'renyi_2'], scales=(1.0, 0.5, 0.25))
    assert m.conv1.in_channels == 18
    x = quantised_image()
    stack = conv1_input(m, x)
    assert stack.shape[1] == 18
    # the first half must be exactly what the single-mode model produces
    single = conv1_input(build('shannon', scales=(1.0, 0.5, 0.25)), x)
    assert (stack[:, :3] - single[:, :3]).abs().max() < 1e-6


def test_perm_requires_a_window_of_at_least_three():
    try:
        build('perm', window_sizes=(2,))
    except ValueError:
        return
    raise AssertionError("perm with a 2x2 window should have been rejected")


def test_every_entropy_config_builds_and_runs():
    from experiment_windows import build_model
    from types import SimpleNamespace                          # noqa: F401
    x = quantised_image(batch=2, size=64)
    for name, cfg in ENTROPY_CONFIGS.items():
        m = build_model('resnet18', dict(cfg, use_rearrange=False),
                        torch.device('cpu')).eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 1), f"{name}: {out.shape}"
        exp = (len(m.entropy_modes) * len(cfg.get('window_sizes', [2]))
               * len(cfg.get('scales', [1.0, 0.5, 0.25])) * 3)
        assert m.conv1.in_channels == exp, f"{name}: {m.conv1.in_channels} != {exp}"


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
