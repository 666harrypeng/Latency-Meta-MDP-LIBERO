# Training commands

Run from the repository root with `PYTHONPATH=src`. Set `CONFIG` to the experiment
configuration and `RUN_DIR` to its output directory. Commands below use the Docker
environment paths; use the equivalent local Python environments outside Docker. Set `HF_TOKEN` and
`WANDB_API_KEY` in the shell when required.

## Clean policy

```bash
python scripts/train_clean_policy.py --config "$CONFIG" --output-dir "$RUN_DIR"
```

## JEPA belief

```bash
/opt/belief/bin/python scripts/check_belief.py --config "$CONFIG" --output-dir "$CHECK_DIR"
/opt/belief/bin/python scripts/train_belief.py --config "$CONFIG" \
  --preflight "$CHECK_DIR/preflight.json" --output-dir "$RUN_DIR"
```

## Belief-conditioned policy

```bash
/opt/belief/bin/python scripts/prepare_forecasts.py \
  --config "$CONFIG" --output-dir "$RUN_DIR/preparation"
python scripts/train_conditioned_policy.py --config "$CONFIG" --output-dir "$RUN_DIR"
```

Conditioned jobs default to `input_mode: current_and_forecast`. Set
`input_mode: forecast_only` to use native observation replacement with future-indexed
action targets. To reuse another run's clean checkpoint and cache, pass
`--clean-work-dir "$CLEAN_RUN_DIR" --forecast-dir "$FORECAST_DIR"`.
An `initialization` block in the job may instead pin a public clean checkpoint's
`repo_id`, `revision` and `step`. The standalone trainer then downloads that
checkpoint and the matching training bundle. Use a separate output directory per mode.

To run clean training, forecast preparation and conditioned training in sequence:

```bash
python scripts/run_policy_pipeline.py --config "$CONFIG" --output-dir "$RUN_DIR"
```

## Meta policy

```bash
/opt/belief/bin/python scripts/train_meta_policy.py --config "$CONFIG" \
  --replay-manifest "$REPLAY_MANIFEST" --output-dir "$RUN_DIR"
```

For the budgeted recipe, also pass `--cost-profile "$COST_PROFILE" --budget "$BUDGET"
--cost-multiplier "$MULTIPLIER"`. The budget is average weighted calls per episode;
the multiplier sets the training penalty.

To collect initial replay and run interaction/training rounds, copy and fill
[`meta_cycle.example.yaml`](../../configs/experiments/meta_cycle.example.yaml), then run:

```bash
/opt/belief/bin/python scripts/run_meta_pipeline.py --config "$CONFIG" \
  --output-dir "$RUN_DIR" --check-only
/opt/belief/bin/python scripts/run_meta_pipeline.py --config "$CONFIG" \
  --output-dir "$RUN_DIR"
```

Set `initial_replay` to reuse an existing replay snapshot, or leave it null to
collect one. Cohorts declare `train`/`validation` master groups; budget feedback
uses training masters only. Each phase fits Q, measures greedy feedback and
validation, then adds training interactions to the shared replay. `selected.json`
records the chosen checkpoint and whether any candidate met the measured budget.
Reportable evaluation is run separately. Use `--resume` after interruption.

Create the cost profile from isolated, single-worker clean-rule and
conditioned-Meta evaluation runs on the same reference device:

```bash
/opt/belief/bin/python scripts/calibrate_meta_cost.py \
  --process-results "$CLEAN_RESULTS" --process-results "$META_RESULTS" \
  --hardware "$DEVICE_LABEL" --output "$COST_PROFILE"
```

Each directory must contain results from one process, with at least three warm
measurements per component after its first call. The cycle uses the configured
evaluation environment (OpenPI, Torch and simulator); it does not install it.

## Publish policy checkpoints

```bash
python scripts/publish_checkpoints.py \
  --checkpoint-dir "$CHECKPOINT_DIR" --steps "$STEP" --repo "$HF_REPO"
```

Use `--resume` for policy/Belief training and `--resume-from` for Meta training.
The pipeline skips completed stages when rerun with the same configuration.
Use each script's `--help` for module-specific options.
