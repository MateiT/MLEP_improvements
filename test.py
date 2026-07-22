import sys
import time
import os
import csv
import torch
from util import Logger, printSet, get_device
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch(100)
DATASETS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'TestDatasets')
DetectionTests = {
# GAN-set-1 is BaiduYun-only (ForenSynths) and not downloaded yet; re-enable once present.
#                   'GAN-set-1': { 'dataroot'   : os.path.join(DATASETS_ROOT, 'GAN-set-1'),
#                                  'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
#                                  'no_crop'    : True,
#                                },
                  'GAN-set-2': { 'dataroot'   : os.path.join(DATASETS_ROOT, 'GAN-set-2'),
                                 'no_resize'  : True,
                                 'no_crop'    : True,
                               },
              'Diffusion-set': { 'dataroot'   : os.path.join(DATASETS_ROOT, 'Diffusion-set'),
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },
#                   'Test-set': { 'dataroot'   : '/Data/Test-set',
#                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
#                                 'no_crop'    : True,
#                               },

                 }


# Test-time corruption scenarios. Each trained-once model is evaluated on the
# SAME images under each of these, so we can see how much blur / JPEG
# compression degrades detection. prob=1.0 -> ALWAYS applied (deterministic).
# Mirrors experiment_windows.EVAL_SCENARIOS so the two harnesses are comparable.
CORRUPTIONS = {
    'clean': dict(),
    'blur':  dict(blur_prob=1.0, blur_sig=[2.0]),
    'jpeg':  dict(jpg_prob=1.0, jpg_qual=[75], jpg_method=['pil']),
}


def set_corruption(opt, cfg):
    """Overwrite opt's blur/JPEG fields for the current scenario. Fields absent
    from cfg are reset to their clean (no-op) values so scenarios don't leak."""
    opt.blur_prob  = cfg.get('blur_prob', 0.0)
    opt.blur_sig   = cfg.get('blur_sig', [0.5])
    opt.jpg_prob   = cfg.get('jpg_prob', 0.0)
    opt.jpg_qual   = cfg.get('jpg_qual', [75])
    opt.jpg_method = cfg.get('jpg_method', ['pil'])


opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# get model
device = get_device(opt.gpu_ids)
model = resnet50(num_classes=1)
model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.to(device)
model.eval()

scenarios = [s.strip() for s in opt.corruptions.split(',') if s.strip()]
unknown = [s for s in scenarios if s not in CORRUPTIONS]
if unknown:
    raise SystemExit(f"Unknown corruptions {unknown}. Available: {list(CORRUPTIONS)}")

print(f"Device: {device} | scenarios: {scenarios}")

# results[scenario][testSet] = {'rows': [(model, acc, ap), ...], 'acc': mean, 'ap': mean}
results = {s: {} for s in scenarios}

for scen in scenarios:
    set_corruption(opt, CORRUPTIONS[scen])
    printSet(f"CORRUPTION = {scen}")
    for testSet in DetectionTests.keys():
        dataroot = DetectionTests[testSet]['dataroot']
        printSet(testSet)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        accs = []; aps = []; rows = []
        # sorted() so the model order is stable across scenarios and runs;
        # dirs-only so stray files (.DS_Store, zips) don't crash the loader.
        model_dirs = sorted(d for d in os.listdir(dataroot)
                            if os.path.isdir(os.path.join(dataroot, d)))
        for v_id, val in enumerate(model_dirs):
            opt.dataroot = '{}/{}'.format(dataroot, val)
            opt.classes  = ''
            opt.no_resize = DetectionTests[testSet]['no_resize']
            opt.no_crop   = DetectionTests[testSet]['no_crop']
            acc, ap, _, _, _, _ = validate(model, opt)
            accs.append(acc); aps.append(ap); rows.append((val, acc, ap))
            print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
        m_acc, m_ap = float(np.mean(accs)), float(np.mean(aps))
        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(len(rows), 'Mean', m_acc*100, m_ap*100))
        print('*'*25)
        results[scen][testSet] = dict(rows=rows, acc=m_acc, ap=m_ap)


# --------------------------------------------------------------------------- #
# Build a comparison report (console + --out file): per model, the acc/ap under
# every scenario side by side, so the blur / jpeg drop vs clean is readable.
# --------------------------------------------------------------------------- #
def fmt_pct(x):
    return f"{x*100:6.1f}"

lines = []
lines.append("MLEP pretrained-model corruption test")
lines.append(f"date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"model_path={opt.model_path}  device={device}")
lines.append(f"scenarios={scenarios}  (blur: sig=2.0 always; jpeg: qual=75 pil always)")
lines.append("")

for testSet in DetectionTests.keys():
    models = [r[0] for r in results[scenarios[0]][testSet]['rows']]
    name_w = max([len(m) for m in models] + [len('Mean')]) + 1

    header = f"{testSet}"
    lines.append(header)
    lines.append("=" * (name_w + 9 * len(scenarios) * 2 + 4))
    # column headers: <model>   clean_acc clean_ap  blur_acc blur_ap ...
    col = f"{'model':{name_w}s}"
    for s in scenarios:
        col += f"{s+'_acc':>10s}{s+'_ap':>10s}"
    lines.append(col)
    lines.append("-" * len(col))

    for i, m in enumerate(models):
        row = f"{m:{name_w}s}"
        for s in scenarios:
            acc, ap = results[s][testSet]['rows'][i][1], results[s][testSet]['rows'][i][2]
            row += f"{fmt_pct(acc):>10s}{fmt_pct(ap):>10s}"
        lines.append(row)

    # Mean row
    mrow = f"{'Mean':{name_w}s}"
    for s in scenarios:
        mrow += f"{fmt_pct(results[s][testSet]['acc']):>10s}{fmt_pct(results[s][testSet]['ap']):>10s}"
    lines.append("-" * len(col))
    lines.append(mrow)
    lines.append("")

lines.append("acc/ap in %. Compare each model's clean columns with its blur / jpeg "
             "columns to read off robustness to corruption.")

report = "\n".join(lines)
print("\n" + report)
with open(opt.out, 'w') as f:
    f.write(report + "\n")
print(f"\nResults written to {opt.out}")
