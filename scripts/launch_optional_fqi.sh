#!/usr/bin/env bash
# Resource-only launcher. The JSON recipe owns scientific budgets and identities.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${GPUS:?Set GPUS to the allocated physical GPU indices, for example 0,1}"
: "${RECIPE:?Set RECIPE to the generated campaign JSON}"
: "${RUN_DIR:?Set RUN_DIR to a new campaign directory, or the same directory to resume}"
SCENE_AUDIT="${SCENE_AUDIT:-configs/data/conveyor_optional/scene-audit.json}"
PYTHON="${PYTHON:-$PWD/.venv-policy-eval/bin/python}"
CONFIG="${CONFIG:-configs/experiments/conveyor_sort/meta_budget_optional_v2.yaml}"
export PYTHONPATH="$PWD/src:$PWD/third_party/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/outputs/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export META_ASSET_VERIFY_CACHE="$PWD/outputs/cache/asset-verification"
export TORCHINDUCTOR_CACHE_DIR="$PWD/outputs/cache/torchinductor"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="${LEARNER_CPU_THREADS:-8}" MKL_NUM_THREADS="${LEARNER_CPU_THREADS:-8}"
export OPENBLAS_NUM_THREADS=1
export CONVEYOR_STAGE_WANDB=0
mkdir -p "$RUN_DIR/operations" "$PWD/outputs/cache/fqi-temp"
export TMPDIR="$PWD/outputs/cache/fqi-temp"
ulimit -n "${NOFILE_LIMIT:-65536}"
exec "$PYTHON" -u scripts/run_fqi_experiment.py \
  --root "$RUN_DIR" --operations "$RUN_DIR/operations" --config "$CONFIG" \
  --recipe "$RECIPE" \
  --checkpoint-repo "${CHECKPOINT_REPO:-yypeng666/metamdp-conveyor-optional-budget-meta-v1}" \
  --project "${WANDB_PROJECT:-latency-meta-mdp-fqi-budget}" \
  --name "${RUN_NAME:-$(basename "$RUN_DIR")}" --poll-seconds 15 \
  --command "$PYTHON" -u scripts/run_conveyor_meta.py campaign \
  --config "$CONFIG" --scene-audit "$SCENE_AUDIT" --campaign-recipe "$RECIPE" \
  --output-dir "$RUN_DIR/campaign" --gpus "$GPUS" --train-gpus "${TRAIN_GPUS:-$GPUS}" \
  --validation-gpus "${VALIDATION_GPUS:-$GPUS}" \
  --rollout-workers-per-gpu "${ROLLOUT_WORKERS_PER_GPU:-1}" \
  --validation-workers-per-gpu "${VALIDATION_WORKERS_PER_GPU:-${ROLLOUT_WORKERS_PER_GPU:-1}}" \
  --cpu-threads "${CPU_THREADS:-4}" --validation-cpu-threads "${VALIDATION_CPU_THREADS:-${CPU_THREADS:-4}}" \
  --learner-cpu-threads "${LEARNER_CPU_THREADS:-8}" \
  --replay-prepare-workers "${REPLAY_PREPARE_WORKERS:-4}" \
  --episode-cache-size "${EPISODE_CACHE_SIZE:-256}" \
  --microbatch "${MICROBATCH:-full}" --bf16 --compile-learner --compact-replay \
  --checkpoint-every-updates 256 --min-free-gib 40 "$@"
