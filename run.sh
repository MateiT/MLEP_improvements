#!/usr/bin/env bash
# MLEP window / robustness sweep on an NVIDIA RTX 4090.
#
#   ./run.sh list                     show the available configs and exit
#   ./run.sh test                     ~10 s   -- regression tests (no data needed)
#   ./run.sh stage                    ~5 min  -- copy the data subset into RAM
#   ./run.sh smoke                    ~2 min  -- confirms the GPU path end-to-end
#   ./run.sh sweep                    ~6-7 h  -- every config x 3 seeds (see RUNTIME)
#   ./run.sh confirm cfg1,cfg2        long run on whichever configs won the sweep
#   ./run.sh entropy                  experiment group 1 -- entropy definitions
#   ./run.sh degradation              experiment group 2 -- blur / JPEG heads
#
# smoke/sweep/confirm stage the data automatically; 'stage' is only for doing it
# up front. See the I/O section below for why staging is not optional here.
#
# Results go to results/<mode>_<stamp>[_seed<n>].txt, full console log to logs/.
# Nothing is overwritten between runs. Combine the per-seed files with:
#   conda run -n MLEP python scripts/aggregate_results.py results/sweep_<stamp>_seed*.txt
#
# ALWAYS start a sweep detached -- it runs for hours in the foreground, so
# closing the terminal or dropping an SSH connection SIGHUPs it and kills it:
#   setsid nohup ./run.sh sweep </dev/null >/dev/null 2>&1 &
# Nothing is lost by discarding stdout there; run.sh tees everything to logs/.
# The result file is rewritten after every config, so you can read it (and run
# aggregate_results.py on it) while the sweep is still going.
#
# Override any knob from the environment, e.g.
#   STEPS=500 SEEDS=100 CONFIGS=baseline_2x2,w4x4_multiscale ./run.sh sweep
#   BATCH=64 CLASSES=car ./run.sh sweep
#   BN_RECAL=0 ./run.sh sweep         # skip BN recalibration (see below -- don't)
#
# On Linux the dataloader uses 'fork', so num_threads > 0 is safe (the "keep 0"
# note in experiment_windows.py is a macOS-only spawn/pickling caveat).
#
# --------------------------------------------------------------------------- #
# RUNTIME / MEMORY, measured on this box (RTX 4090 24 GB, resnet18, crop 224)
# --------------------------------------------------------------------------- #
# Training seconds per step at batch 32:
#     baseline_2x2 (fast 2x2 path)  0.06     w8x8_multiscale          0.42
#     w4x4_multiscale               0.28     multiwindow_multiscale   0.68
# Anything with a window LARGER than 2x2 leaves the hardcoded fast path in
# resnet.py and takes the general unfold+sort route, which is 5-11x slower.
# At w=2 every entropy family (shannon / unique / renyi / tsallis) now uses a
# five-value lookup instead -- a 2x2 window has only five possible value
# patterns -- so renyi_4 and tsallis_4 cost the same as the baseline (measured:
# 0.63 vs 0.98 ms per batch-16 map, against 41 ms for the general path).
#
# At batch 16, seconds/step and peak GPU:
#     multiwindow_multiscale           0.34    5674 MiB
#     multiwindow_multiscale_aligned   0.34    5675 MiB   <- exact registration
#                                                            costs +1 MiB
#
# Peak GPU memory is the real constraint, not speed: the general path
# materialises a (B, 3, H*W, w*w) patch tensor and torch.sort adds int64 indices
# at 2x the value size on top, so cost grows with w*w. multiwindow_multiscale
# peaks at 21.3 GB of the 24.6 GB card at batch 32. batch 128 -- what the old
# version of this script suggested -- is far out of reach.
#
# --------------------------------------------------------------------------- #
# DISK I/O -- read this before trusting any timing above
# --------------------------------------------------------------------------- #
# This box's root volume is ROTATIONAL (/sys/block/vda/queue/rotational == 1):
# ~124 random reads/s, ~3.8 MB/s, ~90 ms latency. The 8 default classes are
# 28.8 GB over 288,048 PNGs, and shuffle=True reads them in the worst possible
# order for a seeking disk.
#
# Run straight off the disk, EVERY config is pinned at ~0.47 s/step no matter
# what the model does -- the GPU idles at 0% while the dataloader workers block
# in 'D' state. The seconds-per-step figures above are only reachable with the
# data in RAM, and a sweep's time(s) column is otherwise measuring the disk.
#
# So the data modes stage a subset into tmpfs first (scripts/stage_dataset.py).
# A 3000-step run only ever touches 48k of the 288k images -- 16.7% of one epoch
# -- so PER_DIR=4000 (64k images, ~6.4 GB) is a superset of what a run samples.
#   STAGE=0        run straight off DATAROOT, no staging (slow here; correct on
#                  a box with an SSD, or once the page cache is warm)
#   STAGE_DIR=...  where to stage        (default /dev/shm/mlep_data)
#   PER_DIR=n      train images per class/label dir (0 = all)
# /dev/shm is RAM and does not survive a reboot; staging just re-runs, and it
# only copies what is missing.
# --------------------------------------------------------------------------- #

set -euo pipefail
cd "$(dirname "$0")"

# Make the package importable without requiring `pip install -e .`.
# An editable install also works and takes precedence.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------------------- #
# conda plumbing
# --------------------------------------------------------------------------- #
# A non-interactive shell never sources conda's shell hook, so `conda` is not on
# PATH even when it is installed. Find it rather than failing with a bare
# "conda: command not found".
if ! command -v conda >/dev/null 2>&1; then
  for d in "$HOME/miniconda3/bin" "$HOME/anaconda3/bin" "$HOME/miniforge3/bin" \
           /opt/conda/bin; do
    if [ -x "$d/conda" ]; then PATH="$d:$PATH"; break; fi
  done
fi
command -v conda >/dev/null 2>&1 || {
  echo "run.sh: conda not found. Install it or add it to PATH." >&2; exit 1; }

ENV_NAME="${ENV_NAME:-MLEP}"
conda env list | awk '$1 !~ /^#/ {print $1}' | grep -qx "$ENV_NAME" || {
  echo "run.sh: conda env '$ENV_NAME' not found. Create it with:" >&2
  echo "    conda env create -f environment.yaml" >&2
  exit 1
}

# --no-capture-output + python -u: `conda run` otherwise buffers the child's
# stdout until it exits, so a multi-hour sweep looks hung and prints everything
# at the end. The previous `| tail -40` then threw away every progress line on
# top of that; tee shows the run live AND keeps the whole log.
PY=(conda run --no-capture-output -n "$ENV_NAME" python -u)

DATAROOT="${DATAROOT:-datasets/TrainDatasets}"

# See the DISK I/O block above. STAGE=0 disables staging entirely.
STAGE="${STAGE:-1}"
STAGE_DIR="${STAGE_DIR:-/dev/shm/mlep_data}"
PER_DIR="${PER_DIR:-4000}"

mkdir -p results logs
STAMP="$(date +%Y%m%d_%H%M%S)"

# --------------------------------------------------------------------------- #
# Shared settings
# --------------------------------------------------------------------------- #
# batch_size is deliberately IDENTICAL for every config. experiment_windows.py's
# fairness argument ("every config trains for the same number of steps") only
# holds if a step means the same thing everywhere, so the batch is sized for the
# most expensive config rather than tuned per config. 16 leaves headroom on top
# of the 21.3 GB that multiwindow_multiscale needs at 32. Raise it only if you
# are comparing within the 2x2 family, where nothing goes near the limit.
BATCH="${BATCH:-16}"
CROP="${CROP:-224}"
WORKERS="${WORKERS:-8}"
LR="${LR:-2e-4}"
ARCH="${ARCH:-resnet18}"

# Extra classes are nearly free -- the step count is fixed, so more classes buys
# diversity at no training cost -- and they stop the result being a statement
# about cars specifically. Set CLASSES= (empty) to use every class folder.
CLASSES="${CLASSES:-car,cat,chair,horse,boat,person,dog,bird}"

# Train batches used to re-estimate BatchNorm running stats before evaluating.
# NOT optional in practice. Training leaves BN with a momentum-0.1 EMA over its
# last ~20 batches (~320 images at batch 16), which is off by enough that eval
# displaces every logit by 10-25 and every prediction collapses to one class --
# that is the acc=0.50-with-good-AP the earlier sweeps reported, and it moves AP
# too (baseline_2x2 clean: 0.5575 before, 0.9985 after). Set BN_RECAL=0 to
# reproduce the old broken behaviour for comparison.
BN_RECAL="${BN_RECAL:-50}"

# Several seeds is the single most important change over the old script. At 200
# steps the smoke test showed a 0.95 vs 0.73 clean-AP gap between configs that
# differ only in train-time augmentation -- almost certainly noise. Without a
# spread there is no way to tell a real ordering from a lucky init.
SEEDS="${SEEDS:-100 200 300}"

# Subset of configs to run; empty = all of them.
CONFIGS="${CONFIGS:-}"

stage_data () {   # stage_data <classes> -> echoes the dataroot to train from
  local classes="$1"
  if [ "$STAGE" != "1" ]; then
    echo "$DATAROOT"
    return
  fi
  # >&2 throughout: stdout is the return channel for the dataroot.
  echo "==> staging data into $STAGE_DIR (PER_DIR=$PER_DIR)" >&2
  "${PY[@]}" scripts/stage_dataset.py \
    --dataroot "$DATAROOT" --dest "$STAGE_DIR" \
    --classes "$classes" --per-dir "$PER_DIR" >&2
  echo "$STAGE_DIR"
}

launch () {   # launch <label> <classes> <steps> <val_batches> <seed> <configs> <out>
  local label="$1" classes="$2" steps="$3" vb="$4" seed="$5" configs="$6" out="$7"
  # Only the data-driven modes need this; 'list' and 'test' run without a dataset.
  [ -d "$DATAROOT/train" ] && [ -d "$DATAROOT/val" ] || {
    echo "run.sh: expected $DATAROOT/{train,val} to exist." >&2
    echo "    val/ is populated by datasets/TrainDatasets/val/progan_val.zip" >&2
    exit 1
  }
  # Idempotent, so calling it once per seed is fine: the first call copies, the
  # rest are no-ops that cost a directory listing. That also means a sweep
  # recovers by itself if /dev/shm is cleared out mid-run.
  local dataroot; dataroot="$(stage_data "$classes")"
  local log_freq=$(( steps / 10 )); [ "$log_freq" -lt 1 ] && log_freq=1
  local args=(
    --dataroot "$dataroot"
    --classes  "$classes"
    --arch     "$ARCH"
    --batch_size "$BATCH"
    --cropSize   "$CROP"
    --num_threads "$WORKERS"
    --lr "$LR"
    --max_train_steps "$steps"
    --max_val_batches "$vb"
    --log_freq "$log_freq"
    --bn_recal_batches "$BN_RECAL"
    --seed "$seed"
    --device cuda
    --out "$out"
  )
  [ -n "$configs" ] && args+=(--configs "$configs")
  # Extra flags for the two additional experiment groups (--experiment ...).
  # Empty for every existing mode, so their command lines are unchanged.
  args+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
  echo "==> $label  (seed=$seed batch=$BATCH steps=$steps classes=${classes:-ALL})"
  "${PY[@]}" -m mlep.experiments.windows "${args[@]}" 2>&1 | tee "logs/${label}.log"
}

MODE="${1:-smoke}"
case "$MODE" in

list)
  # ./run.sh list                 the window sweep's configs (unchanged)
  # ./run.sh list entropy         the entropy group's configs
  # ./run.sh list mlep_degradation
  "${PY[@]}" -m mlep.experiments.windows --list_configs ${2:+--experiment "$2"}
  ;;

# Model-level regression tests. Synthetic inputs only, so these run anywhere and
# do not touch the dataset -- worth running before starting a multi-hour sweep.
test)
  "${PY[@]}" tests/test_window_align.py
  "${PY[@]}" tests/test_bn_recalibration.py
  "${PY[@]}" tests/test_texture_split.py
  "${PY[@]}" tests/test_color_entropy.py
  "${PY[@]}" tests/test_entropy_modes.py
  "${PY[@]}" tests/test_degradation_dataset.py
  ;;

# Copy the subset into RAM up front. Optional -- smoke/sweep/confirm do it
# themselves -- but useful to run once by hand after a reboot, or to check how
# much space the subset actually takes before committing to a long run.
stage)
  stage_data "$CLASSES" >/dev/null
  echo "Staged. du -sh $STAGE_DIR:"
  du -sh "$STAGE_DIR"
  ;;

# --------------------------------------------------------------------------- #
# Smoke test -- plumbing only. One class, 200 steps, one seed. It proves the GPU
# path, the dataloader, AMP, BN recalibration and all five corruption scenarios
# work. Its numbers are NOT a result: 200 steps on one class is noise.
#
# If a config here reports acc ~0.50 alongside a high acc_best, BN recalibration
# is not doing its job -- that combination is the signature of the bug BN_RECAL
# exists to fix, not of a model that cannot classify.
# --------------------------------------------------------------------------- #
smoke)
  # Chosen to touch every distinct code path once: the fast hardcoded 2x2 route,
  # the general unfold+sort route with replicate padding (also the peak-memory
  # config, so if this fits the whole sweep fits), the rich/poor texture split in
  # both combine modes (18-channel concat and 9-channel diff), the train-time
  # augmentation plumbing, and the two COMPOSED together -- texture_split is a
  # build_model kwarg while train_aug is consumed by the harness, so nothing else
  # in the sweep exercises both at once. color_joint_2x2 covers the joint-colour
  # entropy path (the K x K same-colour matrix, which is the one new tensor big
  # enough to matter for memory).
  launch "smoke_${STAMP}" "car" 200 20 100 \
    "baseline_2x2,texsplit_2x2,texsplit_2x2_diff,multiwindow_multiscale_aligned,baseline_train_blur,baseline_train_jpeg,texsplit_2x2_train_jpeg,color_joint_2x2" \
    "results/smoke_${STAMP}.txt"
  ;;

# --------------------------------------------------------------------------- #
# The real comparison. Every config, several classes, several seeds.
# ~6 h for the default 13 configs x 3000 steps x 3 seeds at batch 16.
# Cut it down with CONFIGS=... or SEEDS=... or STEPS=... .
# --------------------------------------------------------------------------- #
sweep)
  STEPS="${STEPS:-3000}"
  for seed in $SEEDS; do
    launch "sweep_${STAMP}_seed${seed}" "$CLASSES" "$STEPS" 50 "$seed" \
      "$CONFIGS" "results/sweep_${STAMP}_seed${seed}.txt"
  done
  echo
  echo "Done. Aggregate the seeds with:"
  echo "  conda run -n $ENV_NAME python scripts/aggregate_results.py \\"
  echo "      results/sweep_${STAMP}_seed*.txt"
  ;;

# --------------------------------------------------------------------------- #
# Confirmation run: only the configs that survived the sweep, trained 4x longer
# and evaluated on 4x the validation data. Use this before believing a ranking.
# --------------------------------------------------------------------------- #
confirm)
  CONFIRM_CONFIGS="${2:-${CONFIGS:-}}"
  [ -n "$CONFIRM_CONFIGS" ] || {
    echo "usage: ./run.sh confirm <config1,config2,...>" >&2
    echo "       (run './run.sh list' to see the names)" >&2
    exit 1
  }
  STEPS="${STEPS:-12000}"
  for seed in $SEEDS; do
    launch "confirm_${STAMP}_seed${seed}" "$CLASSES" "$STEPS" 200 "$seed" \
      "$CONFIRM_CONFIGS" "results/confirm_${STAMP}_seed${seed}.txt"
  done
  echo
  echo "Done. Aggregate the seeds with:"
  echo "  conda run -n $ENV_NAME python scripts/aggregate_results.py \\"
  echo "      results/confirm_${STAMP}_seed*.txt"
  ;;

# --------------------------------------------------------------------------- #
# Experiment group 1: do other entropy definitions beat Shannon?
# Two stages (ENTROPY_STAGE=features|deep|both): cheap summary-statistic features
# + sklearn classifiers, then the MLEP network trained once per front-end.
# One seed by default -- 14 deep configs x 3 seeds is a sweep-sized run; set
# ENT_SEEDS="100 200 300" when the ordering matters.
# UNSEEN=<split> adds an unseen-generator evaluation on datasets/.../<split>.
# --------------------------------------------------------------------------- #
entropy)
  STEPS="${STEPS:-1000}"
  ENT_SEEDS="${ENT_SEEDS:-100}"
  # The feature stage is CPU-side and cheap; scale it with the deep stage rather
  # than leaving a 3000-step deep run to be compared against 60 feature batches.
  EXTRA_ARGS=(--experiment entropy --entropy_stage "${ENTROPY_STAGE:-both}"
              --entropy_feat_train_batches "${FEAT_TRAIN:-60}"
              --entropy_feat_val_batches "${FEAT_VAL:-30}")
  [ -n "${UNSEEN:-}" ] && EXTRA_ARGS+=(--unseen_split "$UNSEEN")
  for seed in $ENT_SEEDS; do
    launch "entropy_${STAMP}_seed${seed}" "$CLASSES" "$STEPS" 50 "$seed" \
      "$CONFIGS" "results/entropy_${STAMP}_seed${seed}.txt"
  done
  ;;

# --------------------------------------------------------------------------- #
# Experiment group 2: predict AI-vs-real, blurred-or-not and JPEG-or-not from
# MLEP's own 2x2 entropy front-end -- the stock model with 3 output logits
# instead of 1. VARIANTS=n degraded copies per source image; SEV_WEIGHT=w>0
# additionally turns on the optional quality/blur-level severity heads.
# --------------------------------------------------------------------------- #
degradation)
  STEPS="${STEPS:-2000}"
  DEG_SEEDS="${DEG_SEEDS:-100}"
  EXTRA_ARGS=(--experiment mlep_degradation
              --deg_variants "${VARIANTS:-4}"
              --deg_sev_weight "${SEV_WEIGHT:-0.0}")
  [ -n "${UNSEEN:-}" ] && EXTRA_ARGS+=(--unseen_split "$UNSEEN")
  for seed in $DEG_SEEDS; do
    launch "degradation_${STAMP}_seed${seed}" "$CLASSES" "$STEPS" 50 "$seed" \
      "$CONFIGS" "results/degradation_${STAMP}_seed${seed}.txt"
  done
  ;;

# --------------------------------------------------------------------------- #
# Full 4-output retrain: every source image contributes clean / blurred /
# JPEG-compressed / noisy variants, and the network gets four logits
# (ai, blur, jpeg, noise). Follows the released MLEP recipe -- Adam 1e-4,
# batch 64, lr*=0.9 every DELR epochs -- rather than the sweep's fixed-step one,
# so this is hours per epoch, not minutes. RESUME=<ckpt> START=<n> continues a
# run (see docs: optim_epoch_<n>.pth beside it makes the resume exact).
# --------------------------------------------------------------------------- #
pert4)
  args=(--dataroot "$DATAROOT" --name "${NAME:-mlep_pert4}"
        --batch_size "${BATCH:-64}" --lr "${LR4:-1e-4}"
        --niter "${EPOCHS:-50}" --delr_freq "${DELR:-20}"
        --num_threads "$WORKERS"
        --out "results/pert4_${STAMP}.txt")
  [ -n "${CLASSES4:-}" ] && args+=(--classes "$CLASSES4")
  [ -n "${RESUME:-}" ]   && args+=(--resume "$RESUME" --start_epoch "${START:-0}")
  [ -n "${BEST_AP:-}" ]  && args+=(--best_ap "$BEST_AP")
  mkdir -p logs
  "${PY[@]}" -m mlep.experiments.perturbation4 "${args[@]}" \
    2>&1 | tee "logs/pert4_${STAMP}.log"
  ;;

# --------------------------------------------------------------------------- #
# Score a 4-output checkpoint's blur / jpeg / noise heads on TestDatasets.
# Each sampled image is rendered at all ten conditions (clean + the nine
# training levels), so the comparison is paired, and the 26 generators are
# averaged unweighted. PER_LABEL=8 is the quick plumbing check.
# --------------------------------------------------------------------------- #
headtest)
  mkdir -p logs
  "${PY[@]}" -m mlep.evaluation.pert4_heads \
    --ckpt "${CKPT:-checkpoints/pert4_decay/model_epoch_best.pth}" \
    --per_label "${PER_LABEL:-500}" \
    --num_threads "$WORKERS" \
    --out "results/pert4_headtest_${STAMP}.txt" \
    2>&1 | tee "logs/pert4_headtest_${STAMP}.log"
  ;;

*)
  echo "usage: ./run.sh {list|test|stage|smoke|sweep|confirm <configs>|entropy|degradation|pert4|headtest}" >&2
  exit 1
  ;;
esac
