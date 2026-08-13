# TrainDatasets is a HALF subset of ProGAN train — read this before comparing numbers

`train/` here is **not** the full CNNDetection/NPR ProGAN training set. It is a
deterministic half, taken because the full archive is 70 GB (7 x 10 GB volumes)
and this box has ~50 GB free. `val/` is complete.

|            | categories | per category/label | images  | on disk |
|------------|-----------:|-------------------:|--------:|--------:|
| `train/`   |         20 |     9,001 (of 18,003) | 360,040 | ~37 GB |
| `val/`     |         20 |       200 (complete) |   8,000 | ~0.8 GB |

Layout is unchanged: `<split>/<category>/{0_real,1_fake}/NNNNN.png`, so
`--dataroot datasets/TrainDatasets` works exactly as before.

## The selection rule

> Within each `<category>/<label>/`, sort filenames ascending and keep the
> first `n // 2`.

Filenames are zero-padded 5-digit stems, so lexicographic and numeric order
agree and `sorted(os.listdir(d))[:len(...)//2]` reproduces the set exactly —
no dependence on the tool that built it.

The rule lands on the **same boundary in all 40 category/label groups**:

    kept:    00000.png ... 09946.png     (9,001 files)
    dropped: 09947.png ... 17999.png     (9,001 or 9,002 files)

So the subset is equivalently described as: **keep every image whose numeric
stem is <= 9946.** One group, `boat/0_real`, has 18,002 source images rather
than 18,003 (one is missing upstream); its boundary is identical.

Both labels are treated identically, so the real/fake balance of the full set
is preserved exactly: 180,020 real and 180,020 fake.

## Rebuilding it

`scripts/fetch_half_dataset.py` downloads **only** the kept half straight from
the upstream archive — it applies the rule itself, so there is no full download
to prune afterwards. It needs ~37 GB free and no 7z step:

```bash
python scripts/fetch_half_dataset.py --dry-run
python scripts/fetch_half_dataset.py datasets/TrainDatasets/train
python scripts/verify_half_subset.py
```

The val split is small and taken whole:

```bash
bash scripts/datasets/download_train_valset.sh
```

If you instead already have the **full** train set from
`scripts/datasets/download_train_trainset.sh`, the same subset is one command:

```bash
find datasets/TrainDatasets/train -name '*.png' | awk -F/ '$NF+0 > 9946' | xargs rm -f
```

`docs/dataset_subset_manifest.json` records the per-group counts and boundary.

## What this changes

The existing checkpoint (`results/degradation_*_deg_baseline_2x2.pt`, 200k steps
over 206,752 sources) was trained on the **full** set. Anything retrained here
sees half the sources per category, so training-set size is no longer held
constant against those runs — re-run a baseline before comparing.
