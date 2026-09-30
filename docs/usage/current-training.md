# Training task list

Moving ball and conveyor are separate task families with their own data and model
checkpoints. The operator's startup request selects which jobs to run and their
priority. This list records configurations and dependencies, not an execution
queue or a history of runs.

Use [Docker setup](../../docker/README.md), then run the commands below from
`/workspace`. Shared training, resume and publication options are described in
[training commands](training.md).

## Conveyor

### Current + forecast conditioned SFT

Configuration: [conditioned_ddp.yaml](../../configs/experiments/conveyor_sort/conditioned_ddp.yaml).

- Initialize from the configured public conveyor clean checkpoint, step 5079.
- Reuse the conveyor clean training bundle and download the small terminal-observation
  supplement, conveyor Direct Belief, RGB decoder and DINO weights.
- Generate forecasts locally, then train with current and forecast dual-camera RGB,
  proprio, query horizon and the task prompt. Targets remain observation-indexed H50 actions.
- Train 15 additional source-equivalent epochs. At global batch 256, milestones
  are steps **7618, 15236, 22854** (approximately 5, 10 and 15 source epochs).
- Use one JAX process, eight devices, global batch 256 and replicated data
  parallelism (`fsdp_devices: 1`).
  The model uses the existing non-LoRA conditioned training scope.

```bash
export CONFIG=configs/experiments/conveyor_sort/conditioned_ddp.yaml
export RUN_DIR=/data/conveyor-conditioned
mkdir -p "$RUN_DIR"
python scripts/run_policy_pipeline.py --config "$CONFIG" \
  --output-dir "$RUN_DIR" --check-access-only

# After the preparation/training pilot described in training.md:
set -o pipefail
python -u scripts/run_policy_pipeline.py --config "$CONFIG" \
  --output-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/pipeline.log"
```

The pinned initializer makes this pipeline skip clean SFT. It prepares the full
forecast cache and then starts conditioned training automatically. Rerun with the
same configuration and directory to resume the incomplete stage. Use the same
`RUN_DIR/preparation` for a completed full preparation and the combined pipeline.
Keep a limited preparation pilot in a separate directory.

[conditioned.yaml](../../configs/experiments/conveyor_sort/conditioned.yaml) retains
the alternative global-batch-128 / FSDP-8 configuration. Select one configuration
per run. [clean_fsdp.yaml](../../configs/experiments/conveyor_sort/clean_fsdp.yaml)
is the separate clean-training entrypoint when clean retraining is assigned.

## Moving ball

### L2 clean and current + forecast

Configurations: [clean.yaml](../../configs/experiments/moving_ball/l2/clean.yaml)
and [conditioned.yaml](../../configs/experiments/moving_ball/l2/conditioned.yaml).

The combined pipeline trains clean from π0.5 base, prepares L2 forecasts and
trains current + forecast from the resulting final clean checkpoint. It uses
8 devices, global batch 256 and replicated model state (`fsdp_devices: 1`).

```bash
export CONFIG=configs/experiments/moving_ball/l2/conditioned.yaml
export RUN_DIR=/data/moving-ball-l2
mkdir -p "$RUN_DIR"
python scripts/run_policy_pipeline.py --config "$CONFIG" \
  --output-dir "$RUN_DIR" --check-access-only
set -o pipefail
python -u scripts/run_policy_pipeline.py --config "$CONFIG" \
  --output-dir "$RUN_DIR" 2>&1 | tee -a "$RUN_DIR/pipeline.log"
```

To run only clean SFT, use `scripts/train_clean_policy.py` with `clean.yaml` and
`--output-dir /data/moving-ball-l2/clean`. Base parameters and the tokenizer are
downloaded as needed. Forecast preparation also downloads the structured source
corpus and selects the configured L2 training split.

### L2 forecast-only

Configuration: [forecast_only.yaml](../../configs/experiments/moving_ball/l2/forecast_only.yaml).
Reuse the L2 clean checkpoint and forecast cache; this does not initialize from
the current + forecast checkpoint.

```bash
python -u scripts/train_conditioned_policy.py \
  --config configs/experiments/moving_ball/l2/forecast_only.yaml \
  --output-dir /data/moving-ball-l2-forecast-only \
  --clean-work-dir /data/moving-ball-l2/clean \
  --forecast-dir /data/moving-ball-l2/preparation/forecasts
```

### L3 forecast-only

Configuration: [forecast_only.yaml](../../configs/experiments/moving_ball/l3/forecast_only.yaml).
Download the configured clean6000 initializer, prepare L3 forecasts and train:

```bash
/opt/belief/bin/python -u scripts/prepare_forecasts.py \
  --config configs/experiments/moving_ball/l3/forecast_only.yaml \
  --output-dir /data/moving-ball-l3-forecast-only/preparation
python -u scripts/train_conditioned_policy.py \
  --config configs/experiments/moving_ball/l3/forecast_only.yaml \
  --output-dir /data/moving-ball-l3-forecast-only
```

Forecast-only replaces native RGB/proprio values with predictions and supervises
H50 actions starting at the forecast endpoint. Each moving-ball conditioned run
uses two balanced-query epochs. Configured milestones are L2 clean 1000/2000/3000,
L2 current + forecast 3780/7560, and L2/L3 forecast-only 3505/7010.

## Checkpoints and outputs

Each job declares its HF publication destination. Training keeps rolling recovery
checkpoints and permanent milestones; milestones are published together when the
stage finishes. A completed milestone can also be uploaded during training using
`scripts/publish_checkpoints.py` after its asynchronous save finishes.

The publisher includes inference parameters, normalization and required notices;
it leaves the model README empty. Generated data and optimizer state remain on
node storage. Return the configuration, checkpoint steps, W&B link, HF revision
and completion status to the operator. Keep run history outside this task list.
