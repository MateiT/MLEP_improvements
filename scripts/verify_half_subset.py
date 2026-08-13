"""Check that datasets/TrainDatasets matches the documented half-subset rule.

    python scripts/verify_half_subset.py

The rule (see docs/DATASET_SUBSET.md): within each
<category>/<label>/, sort filenames ascending and keep the first n // 2, which
lands on 00000.png .. 09946.png -- 9,001 files -- in all 40 groups.

Exits non-zero if any group is short, has files above the boundary, or has an
unreadable PNG (checked on a sample, or all of them with --deep).
"""
import argparse
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'datasets', 'TrainDatasets')
BOUNDARY = 9946          # highest numeric stem kept
EXPECT_TRAIN = 9001      # files per category/label in train/
EXPECT_VAL = 200         # val/ is complete


def check(split, expect, boundary, deep, sample):
    root = os.path.join(ROOT, split)
    if not os.path.isdir(root):
        print("MISSING %s" % root)
        return 1
    bad = 0
    cats = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
    print("\n%s/  (%d categories)" % (split, len(cats)))
    for cat in cats:
        for lab in ('0_real', '1_fake'):
            d = os.path.join(root, cat, lab)
            if not os.path.isdir(d):
                print("  MISSING %s/%s" % (cat, lab))
                bad += 1
                continue
            fs = sorted(f for f in os.listdir(d) if f.endswith('.png'))
            stems = [int(f[:-4]) for f in fs]
            msg = []
            if len(fs) != expect:
                msg.append("count %d != %d" % (len(fs), expect))
            if boundary is not None and stems and max(stems) > boundary:
                over = sum(s > boundary for s in stems)
                msg.append("%d file(s) above boundary %d" % (over, boundary))
            if os.path.exists(d + '/.part') or any(
                    f.endswith('.part') for f in os.listdir(d)):
                msg.append("leftover .part file(s)")
            if msg:
                print("  %-24s %s" % (cat + '/' + lab, "; ".join(msg)))
                bad += 1
    return bad


def check_images(split, deep, sample):
    from PIL import Image
    root = os.path.join(ROOT, split)
    fs = []
    for dp, _, fn in os.walk(root):
        fs += [os.path.join(dp, f) for f in fn if f.endswith('.png')]
    if not deep:
        step = max(1, len(fs) // sample)
        fs = fs[::step]
    bad = []
    for f in fs:
        try:
            im = Image.open(f)
            im.load()
            if im.size != (256, 256):
                bad.append((f, str(im.size)))
        except Exception as exc:                       # noqa: BLE001
            bad.append((f, repr(exc)))
    print("  decoded %d image(s), %d bad" % (len(fs), len(bad)))
    for f, why in bad[:10]:
        print("    %s  %s" % (f, why))
    return len(bad)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--deep', action='store_true',
                    help='decode every PNG instead of a sample')
    ap.add_argument('--sample', type=int, default=2000,
                    help='images to decode when not --deep (default 2000)')
    a = ap.parse_args()

    bad = check('train', EXPECT_TRAIN, BOUNDARY, a.deep, a.sample)
    bad += check('val', EXPECT_VAL, None, a.deep, a.sample)
    print("\nimage decode check:")
    bad += check_images('train', a.deep, a.sample)

    print("\n%s" % ("OK -- subset matches the rule" if bad == 0
                    else "FAILED -- %d problem group(s)/image(s)" % bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
