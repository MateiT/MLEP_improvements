"""Regression tests for ResNet.texture_split (PatchCraft-style rich/poor split).

The idea under test: partition the image into non-overlapping patches, rank them
by texture diversity, reassemble the busy half into a "rich" canvas and the flat
half into a "poor" canvas, then run the ordinary 2x2 local-entropy front-end on
each. The two stacks are either concatenated ('concat', 2x the channels) or
subtracted ('diff', same channel count -- what PatchCraft itself uses).

Both canvases hold exactly half the patches, so they are the same size and the
concat needs no bilinear resampling. That matters: resampling blurs the
high-frequency entropy structure that is the entire signal here, which is the
same reason window_align='pad' exists.

The first test is the important one -- it pins the DEFAULT front-end as
bit-identical to what it was before this feature existed, so the released
9-channel resnet50 weights stay valid.

Run directly (no pytest needed):
    python tests/test_texture_split.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.resnet import resnet18   # noqa: E402


def build(split=True, p=16, mode='concat', rearrange=False,
          window_sizes=(2,), scales=(1.0, 0.5, 0.25)):
    return resnet18(pretrained=False, num_classes=1, window_sizes=list(window_sizes),
                    scales=list(scales), texture_split=split, texture_patch_size=p,
                    texture_split_mode=mode, use_rearrange=rearrange).eval()


def conv1_input(model, x):
    """The tensor that actually reaches conv1, i.e. the reconciled entropy stack.
    Independent of the (randomly initialised) weights. Same idiom as
    tests/test_window_align.py."""
    got = {}
    h = model.conv1.register_forward_pre_hook(lambda m, i: got.__setitem__('t', i[0]))
    with torch.no_grad():
        model(x)
    h.remove()
    return got['t']


def normalise(x):
    """ImageNet normalisation, exactly as data/datasets.py applies it after
    ToTensor. One copy of the constants, used by every image helper here."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def quantised_image(batch=2, size=224, seed=0):
    """8-bit-quantised then ImageNet-normalised, like the real dataloader.

    The quantisation is load-bearing: the entropy front-end counts DISTINCT pixel
    values, so continuous noise makes every window maximally entropic and the maps
    degenerate to a constant (same reasoning as tests/test_bn_recalibration.py)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, (batch, 3, size, size), generator=g).float() / 255.
    return normalise(x)


def structured_image(batch=2, size=224):
    """Half smooth gradient, half 8-bit noise -- so the diversity ranking is
    non-degenerate and the rich/poor separation is unambiguous."""
    x = torch.linspace(0, 1, size).view(1, 1, 1, size).expand(batch, 3, size, size).clone()
    g = torch.Generator().manual_seed(1)
    noise = torch.randint(0, 256, (batch, 3, size, size // 2), generator=g).float() / 255.
    x[:, :, :, size // 2:] = noise
    return (x * 255).round() / 255.        # quantise to 8-bit steps


def legacy_stack(model, x):
    """The entropy front-end AS IT WAS before texture_split existed.

    Deliberately a separate implementation rather than a call into forward(): it
    is the fixed reference the backward-compatibility test compares against, so
    it must not track future edits to forward()."""
    if model.use_rearrange:
        x = model.random_rearrange_blocks(x, model.rearrange_block_size)
    feats = []
    for s in model.scales:
        if s == 1.0:
            xs = x
        else:
            down = F.interpolate(x, scale_factor=s, mode='bilinear', align_corners=False)
            xs = F.interpolate(down, size=x.shape[2:], mode='bilinear', align_corners=False)
        for w in model.window_sizes:
            feats.append(model._entropy_map(xs, w))
    target = feats[0].shape[-2:]
    feats = [f if f.shape[-2:] == target
             else F.interpolate(f, size=target, mode='bilinear', align_corners=False)
             for f in feats]
    return torch.cat(feats, dim=1)


def patches_of(x, p):
    """(batch, channels, nh*nw, p, p) in raster order."""
    b, c = x.shape[:2]
    n = (x.shape[2] // p) * (x.shape[3] // p)
    return x.unfold(2, p, p).unfold(3, p, p).contiguous().view(b, c, n, p, p)


def test_default_path_is_unchanged():
    """THE backward-compatibility test. With no texture kwargs the front-end must
    be bit-identical to the pre-feature implementation, in eval AND train mode
    (train mode also pins how much global RNG the shuffle consumes, since
    random_rearrange_blocks draws a randperm per image during training)."""
    x = quantised_image(size=128)
    for rearrange in (False, True):
        m = resnet18(pretrained=False, num_classes=1, window_sizes=[2],
                     scales=[1.0, 0.5, 0.25], use_rearrange=rearrange)
        assert m.texture_split is False
        assert m.texture_split_mode == 'concat'
        assert m.conv1.in_channels == 9, m.conv1.in_channels

        m.eval()
        assert torch.equal(conv1_input(m, x), legacy_stack(m, x)), f"eval, rearrange={rearrange}"

        m.train()
        torch.manual_seed(7)
        got = conv1_input(m, x)
        torch.manual_seed(7)
        want = legacy_stack(m, x)
        assert torch.equal(got, want), f"train, rearrange={rearrange}"


def test_in_channels_arithmetic():
    """in_channels = (2 if concat-split else 1) * len(windows) * len(scales) * 3.
    'diff' must NOT double the width -- that is what makes it capacity-matched to
    the un-split config it is compared against."""
    cases = [
        (dict(split=False), 9),
        (dict(split=True, mode='concat'), 18),
        (dict(split=True, mode='diff'), 9),
        (dict(split=True, mode='concat', window_sizes=(2, 4), scales=(1.0,)), 12),
        (dict(split=True, mode='diff', window_sizes=(2, 4), scales=(1.0,)), 6),
        (dict(split=False, window_sizes=(2, 4, 6)), 27),
    ]
    for kw, want in cases:
        got = build(**kw).conv1.in_channels
        assert got == want, f"{kw} -> {got}, want {want}"


def test_unknown_split_mode_is_rejected():
    try:
        resnet18(pretrained=False, num_classes=1, texture_split_mode='average')
    except ValueError:
        return
    raise AssertionError("unknown texture_split_mode should have been rejected")


def test_rich_canvas_is_more_textured():
    """The sort actually separates the two populations.

    A canvas is a reordering of the original patches, so re-running the diversity
    measure on the canvas recovers the original per-patch values. Asserting
    min(rich) >= max(poor) demands a CLEAN separation -- far stronger than
    comparing the two means, which a half-broken sort could still pass."""
    m = build(p=16)
    x = structured_image()
    with torch.no_grad():
        rich, poor = m._split_texture_canvases(x, 16)
        d_rich = m._texture_diversity(rich, 16)
        d_poor = m._texture_diversity(poor, 16)
        d_orig = m._texture_diversity(x, 16)

    assert (d_rich.min(dim=1).values >= d_poor.max(dim=1).values).all(), \
        (d_rich.min(dim=1).values, d_poor.max(dim=1).values)
    # ...and no patch's diversity was invented or lost in the reshuffle.
    both = torch.sort(torch.cat([d_rich, d_poor], dim=1), dim=1).values
    assert torch.allclose(both, torch.sort(d_orig, dim=1).values, atol=1e-4)
    # Sanity: the split is non-degenerate on this image (a constant image would
    # trivially satisfy the assertion above).
    assert d_rich.mean() > 2 * d_poor.mean() + 1e-6, (d_rich.mean(), d_poor.mean())


def test_patches_are_preserved():
    """No pixel duplicated, dropped or scrambled. Two levels:
    (a) the pixel multiset of rich+poor equals the cropped input's;
    (b) every canvas patch is element-for-element one of the input patches, in
        diversity order -- which is what catches a wrong permute in the reassembly.
    """
    m = build(p=16)
    x = structured_image(batch=2, size=224)
    p, nh, nw = 16, 14, 14
    with torch.no_grad():
        rich, poor = m._split_texture_canvases(x, p)
        order = torch.argsort(m._texture_diversity(x, p), dim=1,
                              descending=True, stable=True)

    flat = torch.cat([rich.flatten(1), poor.flatten(1)], dim=1)
    assert torch.equal(torch.sort(flat, dim=1).values,
                       torch.sort(x.flatten(1), dim=1).values)

    src = patches_of(x, p)
    half = (nh * nw) // 2
    for name, canvas, offset in (('rich', rich, 0), ('poor', poor, half)):
        got = patches_of(canvas, p)
        for k in range(half):
            for b in range(x.shape[0]):
                want = src[b, :, order[b, offset + k]]
                assert torch.equal(got[b, :, k], want), f"{name} patch {k}, image {b}"


def test_no_interpolation_at_concat():
    """Why the equal-size canvas rule exists, and the channel-order convention.

    Under 'concat' the first 9 channels must be bit-identical to the un-split
    front-end run on the rich canvas, and the last 9 to the poor canvas. Bit
    equality is only reachable if no bilinear resample ran on the way."""
    m = build(p=16, mode='concat')
    x = quantised_image(size=224)
    with torch.no_grad():
        rich, poor = m._split_texture_canvases(x, 16)
    assert rich.shape == poor.shape == (2, 3, 112, 224), (rich.shape, poor.shape)

    stack = conv1_input(m, x)
    # 2x2 stride-1 windows over a 112x224 canvas -> 111x223.
    assert stack.shape == (2, 18, 111, 223), stack.shape
    with torch.no_grad():
        assert torch.equal(stack[:, :9], legacy_stack(m, rich)), 'rich half'
        assert torch.equal(stack[:, 9:], legacy_stack(m, poor)), 'poor half'


def test_diff_mode_is_the_channelwise_difference():
    """'diff' feeds rich - poor at the un-split channel count."""
    m = build(p=16, mode='diff')
    x = quantised_image(size=224)
    with torch.no_grad():
        rich, poor = m._split_texture_canvases(x, 16)
    stack = conv1_input(m, x)
    assert stack.shape == (2, 9, 111, 223), stack.shape
    with torch.no_grad():
        want = legacy_stack(m, rich) - legacy_stack(m, poor)
    assert torch.equal(stack, want)


def test_ties_are_broken_by_raster_order():
    """Determinism on flat regions. A constant image gives every patch diversity
    0; stable sorting must fall back to raster index rather than an arbitrary
    (and possibly CPU-vs-CUDA-divergent) permutation."""
    m = build(p=16)
    x = torch.full((1, 3, 64, 64), 0.3)
    with torch.no_grad():
        rich, poor = m._split_texture_canvases(x, 16)
        again, _ = m._split_texture_canvases(x, 16)
    assert torch.equal(rich, again)

    # 4x4 patch grid, row split -> the rich canvas is patches 0..7 laid out 2x4.
    src = patches_of(x, 16)
    want = (src[:, :, :8].view(1, 3, 2, 4, 16, 16)
                         .permute(0, 1, 2, 4, 3, 5).contiguous().view(1, 3, 32, 64))
    assert torch.equal(rich, want)
    assert rich.shape == poor.shape == (1, 3, 32, 64)


def test_odd_and_indivisible_sizes():
    """Grid arithmetic at the edges: crop to a whole number of patches, prefer the
    lossless axis, trim one patch column only when both axes are odd, and refuse a
    grid too small to split."""
    m = build(p=16)
    with torch.no_grad():
        # 14x14 grid: even rows -> lossless row split, no trim.
        r, p_ = m._split_texture_canvases(quantised_image(1, 224), 16)
        assert r.shape == p_.shape == (1, 3, 112, 224), r.shape

        # 225px with p=32 -> 7x7 grid: both axes odd, so one patch column is
        # trimmed and the split goes by columns. 224 rows, 3 patch columns.
        r, p_ = m._split_texture_canvases(quantised_image(1, 225), 32)
        assert r.shape == p_.shape == (1, 3, 224, 96), r.shape

        # A 1x1 grid cannot be split.
        try:
            m._split_texture_canvases(torch.zeros(1, 3, 32, 32), 32)
        except ValueError:
            pass
        else:
            raise AssertionError("a 1x1 patch grid should have been rejected")

    # And the odd, non-square stack still survives the whole network.
    model = build(p=32)
    with torch.no_grad():
        out = model(quantised_image(1, 225))
    assert out.shape == (1, 1), out.shape


def test_split_survives_the_block_shuffle():
    """Order of operations: the split runs on the original image, and the block
    shuffle is then applied per canvas. So a shuffled canvas must still hold
    exactly the pixels its unshuffled counterpart held -- no block may migrate
    across the rich/poor boundary."""
    m = build(p=16, rearrange=True)
    x = quantised_image(size=224)
    with torch.no_grad():
        canvases = m._split_texture_canvases(x, 16)
        shuffled = [m.random_rearrange_blocks(c, 2) for c in canvases]
    for name, before, after in zip(('rich', 'poor'), canvases, shuffled):
        assert after.shape == before.shape
        assert torch.equal(torch.sort(after.flatten(1), dim=1).values,
                           torch.sort(before.flatten(1), dim=1).values), name
    # The shuffle must actually have done something, or the test proves nothing.
    assert not torch.equal(shuffled[0], canvases[0])


def split_spy(model):
    """Install a recorder on `model`'s split and return the log it appends to.

    Each entry is (batch_size, rich, poor) for one call. Assigned to the INSTANCE,
    so it needs no teardown and cannot leak into another test."""
    log = []
    real = model._split_texture_canvases

    def spy(t, p):
        rich, poor = real(t, p)
        log.append((t.shape[0], rich, poor))
        return rich, poor

    model._split_texture_canvases = spy
    return log


def test_every_image_in_the_batch_is_split_independently():
    """Coverage, and why the split cannot depend on the real/fake label.

    forward() takes only the image and `if self.texture_split` is a config-level
    flag with no per-sample gate, so every image is split -- there is no label in
    scope to condition on. What that argument still needs is that a canvas depends
    on ITS OWN image and nothing else. The harness feeds shuffled MIXED-class
    batches (experiment_windows.py builds both loaders with shuffle=True), so a
    canvas that varied with batch composition would let one class bleed into
    another's features. Bit-identity between each image split alone and its row of
    the batch split rules that out."""
    m = build(p=16, rearrange=False)
    # Deliberately heterogeneous, including a constant image whose split is decided
    # purely by the raster tie-break and a pure gradient with no texture at all.
    x = torch.cat([
        normalise(structured_image(batch=2, size=224)),
        quantised_image(batch=2, size=224),
        normalise(torch.full((1, 3, 224, 224), 0.3)),
        normalise(torch.linspace(0, 1, 224).view(1, 1, 1, 224)
                       .expand(1, 3, 224, 224).contiguous()),
    ])
    n = x.shape[0]

    log = split_spy(m)
    with torch.no_grad():
        m(x)
    assert len(log) == 1, f"expected one split call per forward, got {len(log)}"
    assert log[0][0] == n, f"split saw {log[0][0]} of {n} images"

    _, rich, poor = log[0]
    for i in range(n):
        log.clear()
        with torch.no_grad():
            m(x[i:i + 1])
        assert log[0][0] == 1
        assert torch.equal(log[0][1], rich[i:i + 1]), f"rich canvas of image {i}"
        assert torch.equal(log[0][2], poor[i:i + 1]), f"poor canvas of image {i}"


def test_corrupted_input_is_still_split():
    """The split must also cover blurred / recompressed images -- that is the whole
    point of the experiment, since the corrupted columns are what it is trying to
    improve.

    Drives the REAL corruption path (data.datasets.data_augment, the same callable
    the dataloader wraps in its Compose) with the harness's own EVAL_SCENARIOS
    dicts, rather than a stand-in blur. Corruption happens in the worker before
    ToTensor, so the tensor reaching the model is already corrupted; this asserts
    the split still runs on it AND that the resulting canvas moves, which would
    catch a future "skip the split when the input looks smooth" optimisation
    reintroducing a data-dependent gate."""
    import numpy as np                                        # noqa: PLC0415
    from PIL import Image                                     # noqa: PLC0415
    from types import SimpleNamespace                          # noqa: PLC0415

    from data.datasets import data_augment                     # noqa: PLC0415
    from experiment_windows import EVAL_SCENARIOS              # noqa: PLC0415

    def to_tensor(pil):
        t = torch.from_numpy(np.array(pil)).float().permute(2, 0, 1)[None] / 255.
        return normalise(t)

    src = Image.fromarray(
        (structured_image(batch=1, size=224)[0].permute(1, 2, 0).numpy() * 255)
        .round().astype('uint8'))

    m = build(p=16, rearrange=False)
    log = split_spy(m)
    with torch.no_grad():
        m(to_tensor(src))
    clean_rich = log[0][1]

    for name, scenario in EVAL_SCENARIOS.items():
        if name == 'clean':
            continue
        # Same defaults make_opt uses, overridden by the scenario. Built as a dict
        # first: passing **scenario alongside the same keywords would be a
        # duplicate-kwarg TypeError.
        params = dict(blur_prob=0.0, blur_sig=[0.5], jpg_prob=0.0,
                      jpg_method=['pil'], jpg_qual=[75],
                      webp_prob=0.0, webp_qual=[80])
        params.update(scenario)
        x = to_tensor(data_augment(src, SimpleNamespace(**params)))
        # Vacuity guard: if the corruption silently did nothing, the canvas
        # assertion below would pass for the wrong reason.
        assert not torch.equal(x, to_tensor(src)), f"{name} left the image untouched"

        log.clear()
        with torch.no_grad():
            m(x)
        assert len(log) == 1 and log[0][0] == 1, f"{name} was not split"
        assert not torch.equal(log[0][1], clean_rich), \
            f"{name} produced the clean canvas"


def test_config_kwargs_reach_the_model():
    """build_model in experiment_windows.py is a fixed whitelist: a CONFIGS key
    with no line there is dropped SILENTLY, so the config would train as a clone
    of baseline_2x2 and report a null effect that is really a plumbing bug."""
    from experiment_windows import CONFIGS, build_model     # noqa: E402

    cpu = torch.device('cpu')
    names = [n for n in CONFIGS if n.startswith('texsplit')]
    assert names, "no texsplit_* configs registered"
    for name in names:
        cfg = CONFIGS[name]
        m = build_model('resnet18', cfg, cpu)
        assert m.texture_split is True, name
        assert m.texture_patch_size == cfg['texture_patch_size'], name
        assert m.texture_split_mode == cfg.get('texture_split_mode', 'concat'), name
        want = 9 if m.texture_split_mode == 'diff' else 18
        assert m.conv1.in_channels == want, (name, m.conv1.in_channels)

    base = build_model('resnet18', CONFIGS['baseline_2x2'], cpu)
    assert base.texture_split is False
    assert base.conv1.in_channels == 9


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
