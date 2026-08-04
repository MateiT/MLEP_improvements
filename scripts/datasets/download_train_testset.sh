#!/usr/bin/env bash
# Download the ForenSynths CNN_synth_testset (~20 GB) and unpack it into the
# layout test.py's 'GAN-set-1' entry expects:
#     datasets/TestDatasets/GAN-set-1/<generator>/{0_real,1_fake}
#     datasets/TestDatasets/GAN-set-1/progan/<category>/{0_real,1_fake}
# The zip has NO top-level folder -- it expands straight to the 13 generator
# dirs -- so it must be extracted INTO GAN-set-1, not next to it.
# 20 GB compressed + 20 GB extracted: make sure ~40 GB is free before starting.
set -euo pipefail
cd "$(dirname "$0")/../.."          # -> repo root

DEST="datasets/TestDatasets/GAN-set-1"
ZIP="datasets/TestDatasets/CNN_synth_testset.zip"
URL="https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip"

if [ -d "$DEST/progan/car/1_fake" ]; then
  echo "[skip] $DEST already populated"; exit 0
fi

mkdir -p "$(dirname "$ZIP")"
# -c resumes a partial download instead of restarting 20 GB from zero.
[ -f "$ZIP" ] && echo "[have] $ZIP" || wget -c "$URL" -O "$ZIP"

mkdir -p "$DEST"
if command -v unzip >/dev/null; then
  unzip -q -o "$ZIP" -d "$DEST"
elif command -v 7z >/dev/null; then
  7z x -y -bd -o"$DEST" "$ZIP"
else
  echo "need unzip or 7z (apt install unzip p7zip-full)"; exit 1
fi

# Keep the zip only if you plan to re-extract; it is another 20 GB.
rm -f "$ZIP"
echo "Done -> $DEST"
