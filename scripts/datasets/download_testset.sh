#!/usr/bin/env bash
# Download the MLEP TestDatasets (per-model .tar.gz archives from Google Drive)
# and unpack them into the layout test.py expects:
#     datasets/TestDatasets/GAN-set-2/<model>/{0_real,1_fake}
#     datasets/TestDatasets/Diffusion-set/<model>/{0_real,1_fake}
#
# File IDs were harvested from the NPR / GANGen / UniversalFakeDetect Drive
# folders. Direct CLI download is currently blocked by Google Drive's per-file
# "download quota exceeded" error on these popular archives. If a file 403s:
#   1. Open its https://drive.google.com/uc?id=<ID> link in a browser and click
#      "Add shortcut to Drive" / "Make a copy" into YOUR OWN Drive, then run
#      `gdown <your-copy-ID>` -- your copy has a fresh quota, OR
#   2. Just wait ~24h for the shared quota to reset and re-run this script.
# Already-present models (AttGAN, and the 11 populated diffusion sets) are skipped.
set -uo pipefail
cd "$(dirname "$0")"
ROOT="datasets/TestDatasets"
command -v gdown >/dev/null || { echo "pip install gdown first"; exit 1; }

# model_name <TAB> google-drive-file-id  (grouped by target set)
fetch () {          # fetch <set-dir> <model> <file-id>
  local set="$1" model="$2" id="$3"
  local dest="$ROOT/$set/$model"
  if [ -n "$(find "$dest" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | head -1)" ]; then
    echo "[skip] $set/$model already populated"; return
  fi
  echo "[get ] $set/$model"
  mkdir -p "$ROOT/$set"
  local tgz="$ROOT/$set/$model.tar.gz"
  gdown "$id" -O "$tgz" || { echo "  !! quota/permission block on $model -- see header notes"; return; }
  tar -xzf "$tgz" -C "$ROOT/$set" && rm -f "$tgz"
  # If the archive unpacked to a differently-named top dir, normalise it to <model>.
  [ -d "$dest" ] || { local top; top=$(tar -tzf "$tgz" 2>/dev/null | head -1 | cut -d/ -f1); [ -n "${top:-}" ] && [ -d "$ROOT/$set/$top" ] && mv "$ROOT/$set/$top" "$dest"; }
}

# ---- GAN-set-2 (GANGen-Detection, folder 11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj) ----
fetch GAN-set-2 BEGAN       1ck_j-056-_L0bc1SJD5qt-ADDSxyGNJc
fetch GAN-set-2 CramerGAN   1_0nCtboh6spvnzT4lRKIMIAcaVfBYbsm
fetch GAN-set-2 InfoMaxGAN  1qSQywIVE9ZeP8bTZjpfgYR6jcDM6BK9w
fetch GAN-set-2 MMDGAN      1Aa7wGY28PJOYXAXMb4Jx6CzbhJm2mapq
fetch GAN-set-2 RelGAN      1md-y6GFb-28t_4-mUrSyRyerFC1pUOeq
fetch GAN-set-2 S3GAN       1THixtJQsTO5Cd-8N4JIxormvV-wsGXuc
fetch GAN-set-2 SNGAN       1L1iS_KdXaShhwNBNf3tGOYhxvRyeypek
fetch GAN-set-2 STGAN       1ChS-8jejuR1i7cWV0FoR9wjD0b1nvj30
fetch GAN-set-2 AttGAN      1k7msxU6dS4NPKzE8jxb6AsDx-Kp7MbFn   # already present -> auto-skipped

# ---- Diffusion-set (UniversalFakeDetect, folder 1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-) ----
fetch Diffusion-set dalle        1fNoJW36iZ5Gla2SCGnXUjCgQ_5hyPJ-c
fetch Diffusion-set glide_50_27  1ivS7QPjX5JJXwUifVP2l1RhGaxKJwIfN
fetch Diffusion-set glide_100_10 1p8TcYuqIX_cSrVlQyrc3H_GEfWZB7QCv
fetch Diffusion-set glide_100_27 1GEfxkGnPYSLSSkvNRQmCDfufqIkFnuY3
fetch Diffusion-set guided       1oiP-Jr8ytA5NoQXD0ey7N7y5sTtmG0-L
fetch Diffusion-set ldm_100      1PzC98nXHawJffPYY_J-pYrCH5h5-VyXd
fetch Diffusion-set ldm_200_cfg  1XTWknMu9mUQYzOaU9EhC_Bo8Y0nU6RbM
fetch Diffusion-set ldm_200      14l_1nBZgvcrJSJaePXs76xgfAmSSZ219

echo
echo "Done. Remove any still-empty model folders so test.py's os.listdir loop"
echo "does not choke on them, e.g.:"
echo "  find $ROOT -type d -empty -delete"
