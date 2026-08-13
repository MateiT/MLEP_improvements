"""Rebuild datasets/TrainDatasets/train -- the deterministic half of ProGAN train.

    python scripts/fetch_half_dataset.py datasets/TrainDatasets/train
    python scripts/fetch_half_dataset.py --dry-run          # print the plan only

Why this exists
---------------
scripts/datasets/download_train_trainset.sh cannot run on a machine with less
than ~140 GB free. The upstream archive is seven 10 GiB 7z volumes that 7z must
hold ALL of before it can extract, and what it extracts is a 69.8 GB
progan_train.zip which then has to be unzipped on top of that.

The way around it: the outer 7z is STORE mode. Concatenated, the seven volumes
are exactly

    [32-byte 7z start header][progan_train.zip][98-byte 7z next header]

so the inner ZIP64 archive can be read directly with HTTP range requests. This
script fetches only the ~71 MB central directory, decides which entries to keep,
and then streams only those. Nothing large ever touches the disk, and the
download is roughly half of what a full extract would move.

The subset rule (also in datasets/TrainDatasets/SUBSET.md)
----------------------------------------------------------
    Within each <category>/<label>/, sort filenames ascending and keep the
    first n // 2.

Filenames are zero-padded 5-digit stems, so lexicographic and numeric order
agree; the rule lands on the SAME boundary in all 40 category/label groups:

    kept    00000.png .. 09946.png      (9,001 files)
    dropped 09947.png .. 17999.png

so it is equivalently "keep every image whose numeric stem is <= 9946".
Verify a rebuilt copy with scripts/verify_half_subset.py.
"""
import argparse
import collections
import concurrent.futures as cf
import os
import struct
import sys
import threading
import time
import zlib

import urllib3

BASE = ("https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/"
        "progan_train.7z.%03d")
N_PARTS = 7
ZIP_BASE = 32          # the inner zip starts here in the concatenated stream

_lock = threading.Lock()
stat = collections.Counter()
pool = None


# --------------------------------------------------------------------------- #
# virtual stream over the seven volumes
# --------------------------------------------------------------------------- #
def part_sizes():
    """Byte length of each volume, via HEAD. Not hardcoded: if upstream ever
    re-splits the archive, everything below still lines up."""
    sizes = []
    for i in range(1, N_PARTS + 1):
        r = pool.request('HEAD', BASE % i, redirect=True)
        if r.status != 200:
            raise SystemExit(f"HEAD {BASE % i} -> HTTP {r.status}")
        sizes.append(int(r.headers['content-length']))
    return sizes


def read_range(sizes, starts, vo, length):
    """Read `length` bytes at virtual offset `vo`, spanning volumes as needed."""
    out = []
    while length > 0:
        p = max(i for i in range(N_PARTS) if starts[i] <= vo)
        po = vo - starts[p]
        n = min(length, sizes[p] - po)
        r = pool.request('GET', BASE % (p + 1), redirect=True,
                         headers={'Range': f'bytes={po}-{po + n - 1}'})
        if r.status not in (200, 206):
            raise IOError(f"range read part {p + 1} -> HTTP {r.status}")
        out.append(r.data)
        vo += n
        length -= n
    return b''.join(out)


def central_directory(sizes, starts):
    """Locate and fetch the ZIP64 central directory by reading the archive tail,
    rather than assuming fixed offsets."""
    total = sum(sizes)
    tail = read_range(sizes, starts, total - 4096, 4096)
    i = tail.rfind(b'PK\x06\x06')                       # zip64 end of central dir
    if i < 0:
        raise SystemExit("no ZIP64 end-of-central-directory found in the tail")
    cd_size, cd_off = struct.unpack_from('<QQ', tail, i + 40)
    print(f"  central directory: {cd_size / 1e6:.1f} MB at zip offset {cd_off}")
    return read_range(sizes, starts, ZIP_BASE + cd_off, cd_size)


def parse_central_directory(buf):
    """-> [(name, method, comp_size, uncomp_size, local_header_offset, crc)]."""
    ents, off, n = [], 0, len(buf)
    while off < n - 4 and buf[off:off + 4] == b'PK\x01\x02':
        (_, _, _, _, meth, _, _, crc, csz, usz, nl, el, cl,
         _, _, _, lho) = struct.unpack_from('<IHHHHHHIIIHHHHHII', buf, off)
        name = buf[off + 46:off + 46 + nl].decode('utf-8', 'replace')
        ex = buf[off + 46 + nl:off + 46 + nl + el]
        if 0xFFFFFFFF in (csz, usz, lho):               # zip64 extra field
            p = 0
            while p + 4 <= len(ex):
                hid, hsz = struct.unpack_from('<HH', ex, p)
                dat, q = ex[p + 4:p + 4 + hsz], 0
                if hid == 1:
                    if usz == 0xFFFFFFFF:
                        usz = struct.unpack_from('<Q', dat, q)[0]; q += 8
                    if csz == 0xFFFFFFFF:
                        csz = struct.unpack_from('<Q', dat, q)[0]; q += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack_from('<Q', dat, q)[0]
                    break
                p += 4 + hsz
        if name.lower().endswith('.png'):
            ents.append((name, meth, csz, usz, lho, crc))
        off += 46 + nl + el + cl
    return ents


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def do_chunk(entries, sizes, starts, dest, tries=3):
    """Fetch one contiguous span into memory, then write its entries.

    Buffering the span first matters: this disk sustains far less than the
    network, and a worker that writes straight off the socket holds the response
    open long enough for the CDN to drop it (IncompleteRead)."""
    todo = [e for e in entries
            if not (os.path.exists(os.path.join(dest, e[0]))
                    and os.path.getsize(os.path.join(dest, e[0])) == e[3])]
    if not todo:
        with _lock:
            stat['skipped'] += len(entries)
        return
    start = ZIP_BASE + todo[0][4]
    last = todo[-1]
    end = ZIP_BASE + last[4] + 30 + 512 + last[2] + 4096
    for attempt in range(tries):
        try:
            buf = read_range(sizes, starts, start, min(end, sum(sizes)) - start)
            break
        except Exception as exc:                          # noqa: BLE001
            if attempt == tries - 1:
                with _lock:
                    stat['failed_chunks'] += 1
                print(f"  chunk failed after {tries} tries: {exc}", flush=True)
                return
            time.sleep(2 ** attempt)

    for name, meth, csz, usz, lho, crc in todo:
        o = ZIP_BASE + lho - start
        if buf[o:o + 4] != b'PK\x03\x04':
            with _lock:
                stat['bad_header'] += 1
            continue
        nl, el = struct.unpack_from('<HH', buf, o + 26)
        data = buf[o + 30 + nl + el:o + 30 + nl + el + csz]
        raw = data if meth == 0 else zlib.decompressobj(-15).decompress(data)
        if zlib.crc32(raw) & 0xFFFFFFFF != crc or len(raw) != usz:
            with _lock:
                stat['bad_crc'] += 1
            continue
        p = os.path.join(dest, name)
        with open(p + '.part', 'wb') as f:
            f.write(raw)
        os.replace(p + '.part', p)
        with _lock:
            stat['files'] += 1
            stat['bytes'] += len(raw)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dest', nargs='?', default='datasets/TrainDatasets/train')
    ap.add_argument('--conns', type=int, default=24,
                    help='parallel range requests (default 24)')
    ap.add_argument('--chunks', type=int, default=1024,
                    help='more chunks = shorter-lived HTTP requests')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the selection and exit without downloading')
    a = ap.parse_args()

    global pool
    pool = urllib3.PoolManager(maxsize=a.conns * 2, retries=urllib3.Retry(
        total=5, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504]))

    print("resolving volume sizes...")
    sizes = part_sizes()
    starts = [sum(sizes[:i]) for i in range(N_PARTS)]
    print(f"  {N_PARTS} volumes, {sum(sizes) / 1e9:.1f} GB total")

    ents = parse_central_directory(central_directory(sizes, starts))
    print(f"  {len(ents)} png entries in the archive")

    groups = collections.defaultdict(list)
    for e in ents:
        cat, lab, _ = e[0].split('/', 2)
        groups[f"{cat}/{lab}"].append(e)

    selected = []
    for g in sorted(groups):
        # THE RULE: sort by filename, keep the first half.
        byname = sorted(groups[g], key=lambda e: e[0].split('/')[-1])
        selected += byname[:len(byname) // 2]
        os.makedirs(os.path.join(a.dest, g), exist_ok=True)
    selected.sort(key=lambda e: e[4])            # offset order = sequential reads

    total = sum(e[3] for e in selected)
    print(f"\nselected {len(selected)} of {len(ents)} files "
          f"({total / 1e9:.1f} GB) over {len(groups)} category/label groups")
    if a.dry_run:
        for g in sorted(groups):
            byname = sorted(groups[g], key=lambda e: e[0].split('/')[-1])
            half = byname[:len(byname) // 2]
            print(f"  {g:<22} keep {len(half):>5} of {len(byname):>5}  "
                  f"{half[0][0].split('/')[-1]} .. {half[-1][0].split('/')[-1]}")
        return 0

    k = max(1, len(selected) // a.chunks)
    chunks = [selected[i:i + k] for i in range(0, len(selected), k)]
    print(f"{len(chunks)} chunks over {a.conns} connections -> {a.dest}\n")

    t0, done = time.time(), 0
    with cf.ThreadPoolExecutor(max_workers=a.conns) as ex:
        futs = [ex.submit(do_chunk, c, sizes, starts, a.dest) for c in chunks]
        for f in cf.as_completed(futs):
            f.result()
            done += 1
            if done % 25 == 0 or done == len(chunks):
                el = time.time() - t0
                print(f"  {done}/{len(chunks)} chunks | {stat['files']} files | "
                      f"{stat['bytes'] / 1e9:.1f} GB | {el / 60:.0f} min | "
                      f"ETA {(el / done * (len(chunks) - done)) / 60:.0f} min",
                      flush=True)

    print(f"\nDONE files={stat['files']} skipped={stat['skipped']} "
          f"bad_crc={stat['bad_crc']} failed_chunks={stat['failed_chunks']} "
          f"in {(time.time() - t0) / 60:.0f} min")
    print("verify with:  python scripts/verify_half_subset.py")
    return 1 if (stat['bad_crc'] or stat['failed_chunks']) else 0


if __name__ == '__main__':
    sys.exit(main())
