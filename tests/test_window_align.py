"""Regression tests for ResNet.window_align (the multi-window registration fix).

Different window sizes produce different-sized entropy maps (a w-window over an
H-wide input yields H-w+1 columns), so they have to be reconciled before the
channel-dim concat. Two ways:

  'resize' (default) -- bilinear-resample every map onto the first map's grid.
  'pad'              -- replicate-pad every map out to the largest map's size.

'pad' is exact. The maps are stride-1, so every window size samples the SAME
unit lattice and they differ only in extent: cell i of the w map is centred on
input coordinate i + (w-1)/2, which is where cell i + (w-w_min)/2 of the w_min
map sits. Padding by d = (w - w_min)/2 per side therefore fixes the size and the
registration in one step, with no interpolation.

These tests pin that down, and pin 'resize' as the unchanged default so existing
single-window configs and the pretrained weights are unaffected.

Run directly (no pytest needed):
    python tests/test_window_align.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.resnet import resnet18   # noqa: E402


def build(align, window_sizes=(2, 4, 6), scales=(1.0,)):
    return resnet18(pretrained=False, num_classes=1, window_sizes=list(window_sizes),
                    scales=list(scales), window_align=align,
                    use_rearrange=False).eval()   # rearrange shuffles blocks; off for determinism


def conv1_input(model, x):
    """The tensor that actually reaches conv1, i.e. the reconciled entropy stack.
    Independent of the (randomly initialised) weights, so two models with
    different `window_align` are directly comparable."""
    got = {}
    h = model.conv1.register_forward_pre_hook(lambda m, i: got.__setitem__('t', i[0]))
    with torch.no_grad():
        model(x)
    h.remove()
    return got['t']


def spot_image(row=100, col=100, size=10):
    """Flat image with one high-entropy square at a known location."""
    torch.manual_seed(0)
    x = torch.full((1, 3, 224, 224), 0.5)
    x[:, :, row:row + size, col:col + size] = torch.rand(1, 3, size, size)
    return x


def centroid_row(m):
    ys = torch.arange(m.shape[-2], dtype=torch.float)
    return ((m.sum(-1).squeeze() * ys).sum() / m.sum()).item()


def group(stack, idx):
    """The 3 RGB channels belonging to window index `idx`."""
    return stack[:, idx * 3:(idx + 1) * 3]


def test_pad_output_size_matches_smallest_window():
    """Padding grows every map to the w_min map's size -- 223 for w=2 on 224px.
    That also means multiwindow configs keep the same spatial dims as
    baseline_2x2, unlike centre-cropping which would shrink them to 219."""
    stack = conv1_input(build('pad'), spot_image())
    assert stack.shape == (1, 9, 223, 223), stack.shape


def test_pad_leaves_raw_entropy_values_untouched():
    """No interpolation: the interior of each padded map is bit-identical to the
    unpadded map, offset by exactly d = (w - w_min) / 2."""
    model = build('pad')
    x = spot_image()
    stack = conv1_input(model, x)
    with torch.no_grad():
        for idx, (w, d) in enumerate(((2, 0), (4, 1), (6, 2))):
            g = group(stack, idx)
            interior = g[:, :, d:g.shape[-2] - d, d:g.shape[-1] - d] if d else g
            assert torch.equal(interior, model._entropy_map(x, w)), f"w={w}"


def test_pad_registers_windows_exactly():
    """The whole point. A feature at a known location must land at the same
    coordinate in every window group. 'resize' aligns the maps' outer extents
    instead of the pixels they describe, which STRETCHES them: error is ~0 at the
    image centre and grows toward the edges. 'pad' is exact everywhere."""
    stacks = {a: conv1_input(build(a), spot_image(row=200)) for a in ('pad', 'resize')}
    err = {a: [centroid_row(group(s, i).mean(1)) - centroid_row(group(s, 0).mean(1))
               for i in (1, 2)]
           for a, s in stacks.items()}
    # pad: exact to well under a hundredth of a pixel.
    assert max(abs(e) for e in err['pad']) < 0.01, err['pad']
    # resize: off by ~0.8px (w=4) and ~1.6px (w=6) this far from centre. Asserted
    # so the test fails loudly if the default path is ever silently changed.
    assert abs(err['resize'][1]) > 1.0, err['resize']


def test_resize_error_vanishes_at_image_centre():
    """Documents WHY the resize error is easy to miss: it is a stretch, not a
    shift, so probing the middle of the image shows almost nothing."""
    s = conv1_input(build('resize'), spot_image(row=112))
    err = centroid_row(group(s, 2).mean(1)) - centroid_row(group(s, 0).mean(1))
    assert abs(err) < 0.1, err


def test_pad_is_a_noop_for_a_single_window():
    """Single-window configs (baseline_2x2 and the wNxN_multiscale family) have
    nothing to reconcile, so 'pad' must be bit-identical to 'resize' there. This
    is what keeps the change from touching the pretrained-weights path."""
    x = spot_image()
    a = conv1_input(build('pad', window_sizes=(2,), scales=(1.0, 0.5, 0.25)), x)
    b = conv1_input(build('resize', window_sizes=(2,), scales=(1.0, 0.5, 0.25)), x)
    assert torch.equal(a, b)


def test_mixed_parity_is_rejected():
    """d = (w - w_min)/2 must be a whole number of pixels, so all window sizes
    have to share a parity. Note this is about parity, NOT powers of two: map
    size is H-w+1 (additive), so [2,4,6] and [2,4,8] are equally valid."""
    for bad in ([2, 3], [3, 4, 6]):
        try:
            resnet18(pretrained=False, num_classes=1, window_sizes=bad, window_align='pad')
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have been rejected")


def test_unknown_align_mode_is_rejected():
    try:
        resnet18(pretrained=False, num_classes=1, window_align='bilinear')
    except ValueError:
        return
    raise AssertionError("unknown window_align should have been rejected")


def test_default_is_resize():
    """Default must stay 'resize' so nothing already trained is invalidated."""
    m = resnet18(pretrained=False, num_classes=1,
                 window_sizes=[2], scales=[1.0, 0.5, 0.25])
    assert m.window_align == 'resize'
    assert m.conv1.in_channels == 9


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
