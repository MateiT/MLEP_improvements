"""Device selection, CUDA fast paths and AMP."""
# Split out of the original mlep/experiments/windows.py. The sweep's CONFIGS and
# main() live in mlep.experiments.windows; everything reusable lives here, so
# consumers no longer need in-function imports to dodge a circular dependency.

import os
import sys
import contextlib

import torch

def get_device(pref):
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def setup_cuda_perf(device):
    """Turn on the standard 'go fast on an NVIDIA GPU' switches.

    - TF32 for matmul/conv: big throughput win on Ampere/Ada (RTX 4090) at no
      meaningful accuracy cost for this task.
    - cuDNN autotuner: picks the fastest conv kernels. Worth it because within a
      run only a couple of input shapes occur (train crop, val size).
    Returns the GPU name (or None) for the report header."""
    if device.type != 'cuda':
        return None
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    return torch.cuda.get_device_name(device)


def amp_autocast(use_amp):
    """autocast(fp16) on CUDA when AMP is on, else a no-op context. The entropy
    front-end (unfold / sort / equality tests) is NOT an autocast-eligible op, so
    it stays in fp32; only conv1 + the resnet backbone run in fp16."""
    if use_amp:
        return torch.amp.autocast('cuda')
    return contextlib.nullcontext()


def resolve_num_threads(requested):
    """Auto-pick dataloader workers when the user leaves it at -1.

    The dataset transforms use lambdas, which can't be pickled to workers under
    the 'spawn' start method (macOS / Windows) -- there we must stay at 0. On
    Linux (fork) workers are safe and essential to keep the 4090 fed during the
    Python-side blur/JPEG augmentation."""
    if requested >= 0:
        return requested
    if sys.platform.startswith('linux'):
        return min(8, (os.cpu_count() or 1))
    return 0


