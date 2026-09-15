# Current training tasks

This file contains the active assignment only. Replace its contents for the next
round; keep experiment history and results in the external project record.

## Assignment

Run these tasks using the supplied configurations:

1. Train clean π0.5 from the public base checkpoint using [clean.yaml](../../configs/experiments/moving_ball/l2/clean.yaml).
2. Generate forecast inputs locally with the supplied frozen L2 Direct Belief and
   RGB decoder, then train the **current + forecast** policy from the final L2 clean checkpoint using
   [conditioned.yaml](../../configs/experiments/moving_ball/l2/conditioned.yaml).
   Inputs are current dual-camera RGB and current 16D proprio, plus forecast
   dual-camera RGB and forecast 16D proprio, query horizon and task prompt.
   Action supervision remains the observation-indexed H50 chunk.

3. Train L2 **forecast-only** using [l2/forecast_only.yaml](../../configs/experiments/moving_ball/l2/forecast_only.yaml),
   reusing the same clean checkpoint and generated cache, not the current + forecast checkpoint.
4. Train L3 **forecast-only** using [l3/forecast_only.yaml](../../configs/experiments/moving_ball/l3/forecast_only.yaml).
   It downloads the existing L3 clean6000 initializer; do not train another L3 clean policy.

Tasks 3–4 replace native RGB/proprio values with predictions and supervise actions
starting at the forecast endpoint. They retain the clean VLA input structure.

The combined launcher performs tasks 1–2, including local forecast generation. Use the configured 8 devices,
global batch 256 (32 per device). Launch one process; OpenPI handles data parallelism.
This assignment does not include Belief training, Meta training or evaluation.

## L2 clean and current + forecast

Follow [Docker setup](../../docker/README.md) in a persistent host terminal session
(for example, tmux). Export `HF_TOKEN` and `WANDB_API_KEY` on the host before starting
the container. The HF token needs official DINO weight access and permission to
upload to the model repositories specified in the configs.

Inside the container:

```bash
cd /workspace
export CONFIG=configs/experiments/moving_ball/l2/conditioned.yaml
export RUN_DIR=/data/l2-sft
mkdir -p "$RUN_DIR"

python scripts/run_policy_pipeline.py --config "$CONFIG" \
  --output-dir "$RUN_DIR" --check-access-only

python scripts/train_clean_policy.py \
  --config configs/experiments/moving_ball/l2/clean.yaml \
  --output-dir "$RUN_DIR/clean" --check-only

set -o pipefail
python -u scripts/run_policy_pipeline.py --config "$CONFIG" \
  --output-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/pipeline.log"
```

The access check downloads/caches frozen models. Clean preparation downloads the
public training bundle and checks the actual input batch and device topology.
The base model and tokenizer are downloaded by OpenPI as needed. After clean SFT,
the pipeline downloads source data, generates forecasts and starts conditioned SFT.
Rerun the final command with the same config and output directory after interruption;
the pipeline skips completed stages and selects the stage-specific recovery mode.

## L2 forecast-only

Reuse the L2 data and clean checkpoint from `/data/l2-sft`:

```bash
python -u scripts/train_conditioned_policy.py \
  --config configs/experiments/moving_ball/l2/forecast_only.yaml \
  --output-dir /data/l2-forecast-only \
  --clean-work-dir /data/l2-sft/clean \
  --forecast-dir /data/l2-sft/preparation/forecasts
```

## L3 forecast-only

Generate L3 forecasts locally, then train from the configured public clean6000:

```bash
/opt/belief/bin/python -u scripts/prepare_forecasts.py \
  --config configs/experiments/moving_ball/l3/forecast_only.yaml \
  --output-dir /data/l3-forecast-only/preparation

python -u scripts/train_conditioned_policy.py \
  --config configs/experiments/moving_ball/l3/forecast_only.yaml \
  --output-dir /data/l3-forecast-only
```

For either standalone SFT, append `--check-only` to check its batch before training,
`--resume` to resume interrupted training, or `--publish-only` to retry publication
after training completed. Keep all output directories distinct. Forecast preparation
resumes its existing cache when rerun with the same inputs.

## Expected delivery

- Clean milestones: steps **1000, 2000, 3000**.
- L2 current + forecast milestones: **3780, 7560**.
- L2 forecast-only milestones: **3505, 7010**.
- L3 forecast-only milestones: **3505, 7010**.
- Each conditioned run covers two balanced-query epochs; forecast-only excludes
  pairs without a real future action, so its step counts differ.
- Each stage publishes its milestones together after that stage finishes, to the
  HF model repo declared in its config. Uploads include inference parameters,
  normalization and required notices; the model README stays empty.
- Report the four HF repo links and published revisions, checkpoint steps, W&B run
  links and completion status to the operator. Keep optimizer states,
  logs and generated datasets on node storage until the operator arranges cleanup.

Completion requires `/data/l2-sft/pipeline.json` to show `phase: complete`, the two
forecast-only runs to have `conditioned/completion.json`, and all four milestone
publications to succeed. Generated forecast caches and optimizer states are not uploaded.
