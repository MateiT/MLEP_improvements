"""Regression tests for the degradation experiment (experiment group 2).

The claims this experiment makes only hold if three things are true, and none of
them is visible from the metrics themselves:

  1. no leakage -- every variant of a source image stays inside that source's
     split, so a blurred copy of a training image can never be an evaluation
     image. The dataset builds its file list from ONE split folder and derives
     the degradation from the file path, which makes this structural; the tests
     below check it rather than trusting it.
  2. the held-out cells really are held out -- 'unseen' is only an
     unseen-combination test if those four (quality, blur) pairs never appear in
     training.
  3. the labels describe the image that was actually produced -- an untouched
     image must carry jpeg=0, quality 100 must carry jpeg=1, and the pixels must
     actually change when a degradation is applied.

Cells are assigned with crc32(path), not hash(path): hash() is salted per
process, so the metadata would not reproduce between runs.

Run directly (no pytest needed):
    python tests/test_degradation_dataset.py
"""
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.degradation import (BLUR_LEVELS, CLEAN_CELL,         # noqa: E402
                                     CONDITIONS,
                                     DEFAULT_DEGRADATION_CONFIGS,
                                     DEGRADATION_CONFIGS, HELDOUT_CELLS,
                                     JPEG_LEVELS, N_BIN, N_BL, N_JQ, N_OUT_SEV,
                                     DegradationDataset, all_cells, ckpt_path,
                                     condition_metrics, load_degradation_model,
                                     lr_lambda, multihead_loss, n_out,
                                     save_degradation_ckpt, train_cells,
                                     train_cells_for, warm_start)


def make_tree(root, splits=('train', 'val'), n=6, size=64):
    """Two splits with disjoint filenames, so any overlap is a real bug."""
    rng = np.random.RandomState(0)
    for s, split in enumerate(splits):
        for cls in ('car', 'cat'):
            for lab in ('0_real', '1_fake'):
                d = os.path.join(root, split, cls, lab)
                os.makedirs(d)
                for i in range(n):
                    a = rng.randint(0, 256, (size, size, 3)).astype(np.uint8)
                    Image.fromarray(a).save(
                        os.path.join(d, f"{split}_{cls}_{lab}_{i}.png"))


def fake_args(root):
    return SimpleNamespace(dataroot=root, classes='', loadSize=64, cropSize=32,
                           batch_size=4, num_threads=0, seed=100, deg_variants=4,
                           device_type='cpu')


def dataset(root, split='train', cells=None, variants=4, is_train=True):
    args = fake_args(root)
    return DegradationDataset(os.path.join(root, split), ('car', 'cat'),
                              cells or train_cells(), variants, args, is_train,
                              seed=args.seed)


def with_tree(fn):
    root = tempfile.mkdtemp(prefix='mlep_deg_')
    try:
        make_tree(root)
        fn(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #
def test_grid_shape():
    assert len(all_cells()) == len(JPEG_LEVELS) * len(BLUR_LEVELS) == 28
    assert len(train_cells()) == 28 - len(HELDOUT_CELLS)
    assert all(c in all_cells() for c in HELDOUT_CELLS)
    assert N_OUT_SEV == N_BIN + N_JQ + N_BL == 14
    # three probabilities by default; the severity heads are opt-in
    assert n_out(0.0) == N_BIN == 3 and n_out(0.5) == N_OUT_SEV


def test_lossless_is_the_only_uncompressed_level():
    assert JPEG_LEVELS[0] == 'lossless'
    assert 100 in JPEG_LEVELS and JPEG_LEVELS.index(100) != 0


def test_conditions_cover_the_four_regimes():
    assert CONDITIONS['clean'] == [('lossless', 'none')]
    assert all(q != 'lossless' and b == 'none' for q, b in CONDITIONS['jpeg_only'])
    assert all(q == 'lossless' and b != 'none' for q, b in CONDITIONS['blur_only'])
    assert all(q != 'lossless' and b != 'none' for q, b in CONDITIONS['both'])
    assert CONDITIONS['unseen'] == HELDOUT_CELLS


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
def test_no_source_image_is_in_two_splits():
    def body(root):
        tr = {os.path.basename(p) for p, _, _ in dataset(root, 'train').samples}
        va = {os.path.basename(p) for p, _, _ in dataset(root, 'val').samples}
        assert tr and va
        assert not (tr & va), sorted(tr & va)
    with_tree(body)


def test_all_variants_of_a_source_stay_in_its_split():
    def body(root):
        ds = dataset(root, 'train')
        for i in range(len(ds)):
            src, v, _ = ds.cell_of(i)
            assert ds.samples[src][0].startswith(os.path.join(root, 'train'))
            assert 0 <= v < ds.variants
    with_tree(body)


def test_training_never_sees_a_held_out_cell():
    def body(root):
        ds = dataset(root, 'train')
        cells = {ds.cell_of(i)[2] for i in range(len(ds))}
        assert not (cells & set(HELDOUT_CELLS)), sorted(cells & set(HELDOUT_CELLS))
    with_tree(body)


def test_cell_assignment_is_reproducible():
    def body(root):
        a, b = dataset(root, 'train'), dataset(root, 'train')
        assert [a.cell_of(i)[2] for i in range(len(a))] == \
               [b.cell_of(i)[2] for i in range(len(b))]
        # and not just constant -- it must actually spread over the grid
        assert len({a.cell_of(i)[2] for i in range(len(a))}) > 5
    with_tree(body)


# --------------------------------------------------------------------------- #
# labels and pixels
# --------------------------------------------------------------------------- #
def test_targets_agree_with_the_assigned_cell():
    def body(root):
        ds = dataset(root, 'train', cells=all_cells())
        for i in range(len(ds)):
            _, _, (q, b) = ds.cell_of(i)
            ai, blur_bin, jpeg_bin, jq, bl, idx = ds.targets(i)
            assert jpeg_bin == float(q != 'lossless'), (q, jpeg_bin)
            assert blur_bin == float(b != 'none'), (b, blur_bin)
            assert JPEG_LEVELS[int(jq)] == q and BLUR_LEVELS[int(bl)][0] == b
            assert int(idx) == i
            assert ai == float('1_fake' in ds.samples[i // ds.variants][0])
    with_tree(body)


def test_metadata_is_complete_and_kernels_are_documented():
    def body(root):
        ds = dataset(root, 'train', cells=all_cells())
        want = {'source_id', 'split', 'label', 'generator', 'jpeg_quality',
                'blur_level', 'blur_method', 'blur_kernel', 'blur_sigma', 'both',
                'seed'}
        kernels = {}
        for i in range(len(ds)):
            md = ds.meta(i)
            assert want <= set(md), sorted(want - set(md))
            kernels[md['blur_level']] = md['blur_kernel']
            assert md['both'] == int(md['jpeg_quality'] != 'lossless'
                                     and md['blur_level'] != 'none')
        # 2*int(4*sigma + 0.5) + 1 for sigma 0.5 / 1.5 / 3.0
        for lvl, k in (('none', 0), ('weak', 5), ('medium', 13), ('strong', 25)):
            if lvl in kernels:
                assert kernels[lvl] == k, (lvl, kernels[lvl])
    with_tree(body)


def test_degradations_actually_change_the_pixels():
    def body(root):
        args = fake_args(root)
        base = DegradationDataset(os.path.join(root, 'train'), ('car',),
                                  [('lossless', 'none')], 1, args, False)
        clean = base[0][0]
        for cell in (('lossless', 'strong'), (15, 'none'), (100, 'none'),
                     (15, 'strong')):
            ds = DegradationDataset(os.path.join(root, 'train'), ('car',), [cell],
                                    1, args, False)
            assert (ds[0][0] - clean).abs().max() > 1e-3, cell
        # ...and harder blur must differ from weaker blur
        weak = DegradationDataset(os.path.join(root, 'train'), ('car',),
                                  [('lossless', 'weak')], 1, args, False)[0][0]
        strong = DegradationDataset(os.path.join(root, 'train'), ('car',),
                                    [('lossless', 'strong')], 1, args, False)[0][0]
        assert (weak - strong).abs().max() > 1e-3
    with_tree(body)


def test_getitem_returns_a_tensor_and_the_six_targets():
    def body(root):
        ds = dataset(root, 'train')
        img, tgt = ds[0]
        assert img.shape == (3, 32, 32)
        assert tgt.shape == (6,)
        # recalibrate_bn iterates `for img, _ in loader`, so the pair shape matters
        loader = torch.utils.data.DataLoader(ds, batch_size=4)
        b_img, b_tgt = next(iter(loader))
        assert b_img.shape == (4, 3, 32, 32) and b_tgt.shape == (4, 6)
    with_tree(body)


# --------------------------------------------------------------------------- #
# model / loss / metrics
# --------------------------------------------------------------------------- #
def test_the_front_end_is_mleps_own_2x2_pyramid():
    """The degradation model must be the released MLEP front-end, unchanged: 2x2
    windows over the 3-scale pyramid, i.e. the same 9 input channels the stock
    detector uses. Only the head count -- and, for the opt-in renyi config, the
    entropy functional -- may differ."""
    from experiment_windows import CONFIGS, build_model
    base = CONFIGS['baseline_2x2']
    for name, cfg in DEGRADATION_CONFIGS.items():
        assert cfg['window_sizes'] == [2], f"{name}: {cfg['window_sizes']}"
        assert cfg['scales'] == base.get('scales', [1.0, 0.5, 0.25]), name
        m = build_model('resnet18', dict(cfg, num_classes=3, use_rearrange=False),
                        torch.device('cpu')).eval()
        assert m.conv1.in_channels == 9, f"{name}: {m.conv1.in_channels}"


def test_the_default_run_is_the_stock_shannon_front_end_only():
    """Adding front-end variants must not change what a plain run does."""
    assert DEFAULT_DEGRADATION_CONFIGS == ['deg_baseline_2x2']
    assert 'entropy_mode' not in DEGRADATION_CONFIGS['deg_baseline_2x2']


def test_the_model_emits_three_logits_by_default_and_fourteen_with_severity():
    from experiment_windows import build_model
    x = torch.rand(2, 3, 64, 64)
    for name, cfg in DEGRADATION_CONFIGS.items():
        for w, want in ((0.0, N_BIN), (0.5, N_OUT_SEV)):
            m = build_model('resnet18',
                            dict(cfg, num_classes=n_out(w), use_rearrange=False),
                            torch.device('cpu')).eval()
            with torch.no_grad():
                out = m(x)
            assert out.shape == (2, want), f"{name} @ sev={w}: {out.shape}"


def _loss_inputs(n_logits):
    torch.manual_seed(0)
    out = torch.randn(8, n_logits, requires_grad=True)
    tgt = torch.zeros(8, 6)
    tgt[:, :3] = (torch.rand(8, 3) > 0.5).float()
    tgt[:, 3] = torch.randint(0, N_JQ, (8,)).float()
    tgt[:, 4] = torch.randint(0, N_BL, (8,)).float()
    return out, tgt


def test_multihead_loss_is_finite_and_uses_every_head():
    out, tgt = _loss_inputs(N_OUT_SEV)
    loss = multihead_loss(out, tgt, 0.5)
    assert torch.isfinite(loss)
    loss.backward()
    g = out.grad.abs().sum(0)
    assert (g[:N_BIN] > 0).all() and (g[N_BIN:] > 0).all()


def test_the_default_loss_is_plain_bce_over_the_three_probabilities():
    out, tgt = _loss_inputs(N_BIN)
    loss = multihead_loss(out, tgt, 0.0)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(out, tgt[:, :3])
    assert abs(float(loss) - float(ref)) < 1e-9
    loss.backward()
    assert (out.grad.abs().sum(0) > 0).all()


def test_severity_metrics_are_computed_in_quality_points():
    n = 40
    jq_true = np.arange(n) % N_JQ
    ev = dict(prob=np.tile(np.array([[0.9, 0.1, 0.8]]), (n, 1)),
              jq=jq_true.copy(), bl=(np.arange(n) % N_BL),
              tgt=np.stack([np.ones(n), np.zeros(n), np.ones(n), jq_true,
                            np.arange(n) % N_BL, np.arange(n)], axis=1))
    m = condition_metrics(ev)
    assert m['jq_exact'] == 1.0 and m['bl_acc'] == 1.0
    assert m['jq_mae'] == 0.0 and m['jq_within15'] == 1.0
    assert abs(m['jq_spearman'] - 1.0) < 1e-9
    assert m['jq_cm'].shape == (N_JQ, N_JQ) and m['bl_cm'].shape == (N_BL, N_BL)
    assert m['ai']['acc'] == 1.0 and m['blur']['acc'] == 1.0

    # one quality level off by exactly one step (15 points) -> MAE 15/n_compressed
    ev2 = dict(ev, jq=np.where(np.arange(n) == 1, jq_true + 1, jq_true))
    m2 = condition_metrics(ev2)
    assert m2['jq_exact'] < 1.0 and m2['jq_within15'] == 1.0 and m2['jq_mae'] > 0

    # ...and with the severity heads off there are no severity metrics at all,
    # only the three binary heads the report then writes.
    m3 = condition_metrics(dict(ev, jq=None, bl=None))
    assert 'jq_exact' not in m3 and 'jq_cm' not in m3
    assert all(h in m3 for h in ('ai', 'blur', 'jpeg')) and m3['n'] == n


# --------------------------------------------------------------------------- #
# the long-run training flags
#
# All six are off by default and none of them may touch the default run. The
# harness is not bit-reproducible (cuDNN benchmark, TF32, AMP and unseeded fork'd
# dataloader workers all predate this), so a numeric before/after diff cannot
# prove that -- these pin the code path instead.
# --------------------------------------------------------------------------- #
def train_args(**kw):
    base = dict(deg_clean_oversample=0, lr=2e-4, lr_schedule='none', warmup_steps=0,
                lr_min=0.0, max_train_steps=1000)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_default_training_cell_list_is_the_plain_grid():
    assert train_cells_for(train_args()) == train_cells()
    # ...and an args object predating the flag behaves as if it were 0
    assert train_cells_for(SimpleNamespace()) == train_cells()


def test_clean_oversampling_only_repeats_the_untouched_cell():
    assert CLEAN_CELL == ('lossless', 'none') and CLEAN_CELL in train_cells()
    cells = train_cells_for(train_args(deg_clean_oversample=12))
    assert len(cells) == len(train_cells()) + 12
    assert cells.count(CLEAN_CELL) == 13
    # every other cell keeps its single entry, and no new cell appears
    assert set(cells) == set(train_cells())
    assert all(cells.count(c) == 1 for c in set(cells) if c != CLEAN_CELL)
    # the held-out cells stay held out no matter how much we oversample
    assert not any(c in cells for c in HELDOUT_CELLS)


def test_oversampling_never_reaches_the_evaluation_cells():
    """The eval sets are built from CONDITIONS, which the flag must not see --
    that is what keeps every reported metric comparable to earlier runs."""
    before = {k: list(v) for k, v in CONDITIONS.items()}
    train_cells_for(train_args(deg_clean_oversample=12))
    assert {k: list(v) for k, v in CONDITIONS.items()} == before
    assert CONDITIONS['seen'] == train_cells()


def test_the_default_lr_is_constant_and_builds_no_scheduler():
    # None, not a lambda returning 1.0: LambdaLR's constructor calls step() and
    # rewrites param_group['lr'], so the default path must not build one.
    assert lr_lambda(train_args()) is None


def test_cosine_lr_starts_at_one_ends_at_the_floor_and_warms_up():
    f = lr_lambda(train_args(lr_schedule='cosine', max_train_steps=1000))
    assert abs(f(0) - 1.0) < 1e-12
    assert abs(f(1000) - 0.0) < 1e-12
    assert abs(f(500) - 0.5) < 1e-9
    assert all(f(s) >= f(s + 1) for s in range(0, 1000, 50))       # monotone down

    # a floor is expressed as a fraction of --lr
    g = lr_lambda(train_args(lr_schedule='cosine', lr_min=2e-5, max_train_steps=100))
    assert abs(g(100) - 0.1) < 1e-9

    # warmup is linear and hands over at exactly full LR
    h = lr_lambda(train_args(lr_schedule='cosine', warmup_steps=10,
                             max_train_steps=1000))
    assert abs(h(0) - 0.1) < 1e-12 and abs(h(9) - 1.0) < 1e-12
    assert abs(h(10) - 1.0) < 1e-12

    # warmup alone, with no schedule, holds at 1.0 afterwards
    w = lr_lambda(train_args(warmup_steps=4))
    assert w is not None and abs(w(3) - 1.0) < 1e-12 and w(500) == 1.0


def test_checkpoint_paths_join_the_existing_results_family():
    out = 'results/degradation_20260803_121054_seed100.txt'
    assert ckpt_path(out, 'deg_baseline_2x2') == \
        'results/degradation_20260803_121054_seed100_deg_baseline_2x2.pt'
    assert ckpt_path(out, 'deg_baseline_2x2', '_best').endswith(
        '_deg_baseline_2x2_best.pt')
    # a name without the .txt extension still gets one stem, not two
    assert ckpt_path('results/run', 'cfg') == 'results/run_cfg.pt'


def _tiny_model_args(out):
    return SimpleNamespace(arch='resnet18', bn_recal_batches=50, loadSize=256,
                           cropSize=224, out=out, seed=100)


def test_checkpoint_round_trips_and_refuses_to_overwrite():
    from experiment_windows import build_model
    d = tempfile.mkdtemp(prefix='mlep_ckpt_')
    try:
        dev = torch.device('cpu')
        cfg = dict(DEGRADATION_CONFIGS['deg_baseline_2x2'], num_classes=N_BIN)
        model = build_model('resnet18', cfg, dev)
        args = _tiny_model_args(os.path.join(d, 'run.txt'))
        p = ckpt_path(args.out, 'deg_baseline_2x2')
        save_degradation_ckpt(p, model, cfg, args, 2000, model.conv1.in_channels)

        loaded, ck = load_degradation_model(p, dev)
        assert ck['total_steps'] == 2000 and ck['num_classes'] == N_BIN
        assert ck['in_channels'] == 9 and ck['arch'] == 'resnet18'
        assert ck['cfg'] == cfg and ck['experiment'] == 'mlep_degradation'
        # the preprocessing the weights were trained under travels with them
        assert ck['cropSize'] == 224 and ck['loadSize'] == 256
        assert len(ck['norm_mean']) == 3 and len(ck['norm_std']) == 3

        x = torch.randn(2, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), loaded(x), atol=0, rtol=0)

        # results are never overwritten, and checkpoints follow the same rule
        try:
            save_degradation_ckpt(p, model, cfg, args, 2000, 9)
            raise AssertionError('overwrote an existing checkpoint')
        except SystemExit:
            pass
        assert not os.path.exists(p + '.tmp')          # atomic write left no debris
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_warm_start_loads_the_body_and_reinitialises_the_head():
    """pretrained/model_epoch_best.pth is this network with a 1-output fc1, so a
    warm start has to keep every other tensor and rebuild only the head."""
    from experiment_windows import build_model
    d = tempfile.mkdtemp(prefix='mlep_warm_')
    try:
        dev = torch.device('cpu')
        cfg1 = dict(DEGRADATION_CONFIGS['deg_baseline_2x2'], num_classes=1)
        donor = build_model('resnet18', cfg1, dev)
        src = os.path.join(d, 'donor.pth')
        torch.save(donor.state_dict(), src)            # bare state_dict, as released

        cfg3 = dict(DEGRADATION_CONFIGS['deg_baseline_2x2'], num_classes=N_BIN)
        model = build_model('resnet18', cfg3, dev)
        assert model.fc1.weight.shape[0] == 3
        warm_start(model, src, 'deg_baseline_2x2')

        dsd = donor.state_dict()
        for k, v in model.state_dict().items():
            if k.startswith('fc1.'):
                continue
            assert torch.equal(v, dsd[k]), f"{k} was not loaded"
        # the ai logit inherits the donor's single head; the other two are fresh
        assert torch.equal(model.fc1.weight[0], dsd['fc1.weight'][0])
        assert torch.equal(model.fc1.bias[0], dsd['fc1.bias'][0])
        assert model.fc1.weight.shape == (3, dsd['fc1.weight'].shape[1])

        # a checkpoint for a different front-end must fail loudly, not silently
        other = build_model('resnet18', dict(window_sizes=[4], scales=[1.0],
                                             num_classes=1), dev)
        bad = os.path.join(d, 'bad.pth')
        torch.save(other.state_dict(), bad)
        try:
            warm_start(build_model('resnet18', cfg3, dev), bad, 'x')
            raise AssertionError('accepted a mismatched checkpoint')
        except (SystemExit, RuntimeError):
            pass
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_the_released_mlep_checkpoint_is_warm_startable():
    """Not a mock: the actual released file, if it is present."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, 'pretrained', 'model_epoch_best.pth')
    if not os.path.exists(src):
        return
    from experiment_windows import build_model
    cfg = dict(DEGRADATION_CONFIGS['deg_baseline_2x2'], num_classes=N_BIN)
    model = build_model('resnet50', cfg, torch.device('cpu'))
    warm_start(model, src, 'deg_baseline_2x2')
    sd = torch.load(src, map_location='cpu', weights_only=True)
    assert torch.equal(model.conv1.weight, sd['conv1.weight'])
    assert torch.equal(model.fc1.weight[0], sd['fc1.weight'][0])


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
