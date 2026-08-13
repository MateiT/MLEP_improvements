"""Model construction and BatchNorm recalibration."""
# Split out of the original mlep/experiments/windows.py. The sweep's CONFIGS and
# main() live in mlep.experiments.windows; everything reusable lives here, so
# consumers no longer need in-function imports to dodge a circular dependency.

import torch
import torch.nn as nn

from mlep.networks.resnet import resnet18, resnet50
from mlep.harness.device import amp_autocast

def build_model(arch, cfg, device):
    # This is a whitelist: a CONFIGS key with no line below is dropped silently,
    # so the config trains as a clone of the baseline and "measures" a null
    # effect that is really a plumbing bug. Every model kwarg needs an entry here.
    # Guarded by tests/test_texture_split.py::test_config_kwargs_reach_the_model.
    kwargs = dict(
        # 1 logit = the AI-vs-real head, i.e. every existing config. The
        # degradation experiment asks for more heads and passes num_classes;
        # nothing else in this file sets it, so the default is unchanged.
        num_classes=cfg.get('num_classes', 1),
        window_sizes=cfg.get('window_sizes', [2]),
        scales=cfg.get('scales', [1.0, 0.5, 0.25]),
        entropy_mode=cfg.get('entropy_mode', 'shannon'),
        use_rearrange=cfg.get('use_rearrange', True),
        rearrange_block_size=cfg.get('rearrange_block_size', 2),
        normalize_entropy=cfg.get('normalize_entropy', False),
        window_align=cfg.get('window_align', 'resize'),
        texture_split=cfg.get('texture_split', False),
        texture_patch_size=cfg.get('texture_patch_size', 16),
        texture_split_mode=cfg.get('texture_split_mode', 'concat'),
        color_entropy=cfg.get('color_entropy', False),
    )
    factory = {'resnet18': resnet18, 'resnet50': resnet50}[arch]
    model = factory(pretrained=False, **kwargs)   # pretrained=False: conv1 is reshaped
    return model.to(device)



@torch.no_grad()
def recalibrate_bn(model, loader, n_batches, device, use_amp=False):
    """Re-estimate every BatchNorm's running stats over the training distribution.

    WHY THIS EXISTS
    ---------------
    BatchNorm's default momentum=0.1 makes running_mean/running_var an EMA over
    roughly the last ~20 batches of training -- ~320 images at batch 16, collected
    while the weights were still moving. Per layer the error looks harmless (at
    bn1 the stored variance sits ~10% below the true batch variance), but it
    compounds through the nine BN layers of layer1+layer2, and the head is a
    single Linear(128, 1) on a global average pool, so nothing downstream absorbs
    the drift.

    Measured on baseline_2x2 at the sweep's settings (3000 steps, 8 classes,
    batch 16), same weights in both rows:

        model.eval() as-is            logit mean +24.16   acc 0.4938   AP 0.8057
        after this recalibration      logit mean  -0.25   acc 0.9762   AP 0.9985

    Every logit was displaced by ~24, so every prediction collapsed to one class
    and accuracy pinned at the class prior. The sign is arbitrary -- a 600-step
    run shifted by -13.85 instead -- which is why some configs used to report
    all-real and others all-fake. Note AP moves too: this was never just a
    threshold problem, it corrupted the ranking the sweep exists to produce.

    momentum=None switches BN to a cumulative average, so the pass below replaces
    the EMA with an honest mean over n_batches. 25 batches (~2 s) already
    saturates the fix; the default is 50 for margin.

    Estimated on TRAIN data on purpose. Using val here would be transductive --
    the reported numbers would have seen the test set.
    """
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns or n_batches <= 0:
        return 0
    saved = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # cumulative moving average
    was_training = model.training
    model.train()                  # train() is what makes BN update its stats
    seen = 0
    try:
        for img, _ in loader:
            if seen >= n_batches:
                break
            with amp_autocast(use_amp):
                model(img.to(device, non_blocking=True))
            seen += 1
    finally:
        for m, mom in zip(bns, saved):
            m.momentum = mom
        model.train(was_training)
    return seen


