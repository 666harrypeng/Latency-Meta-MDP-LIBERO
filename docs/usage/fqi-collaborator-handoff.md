# Scratch budget-FQI training guide

## Start here

Clone this GitHub repository and pin the release commit supplied by the owner.
No HF source archive, overlay, ignored local output, or old Meta checkpoint is
needed. The runnable configuration is `configs/experiments/conveyor_sort/meta_budget_optional_v2.yaml`.
Use English for tracked first-party changes. The separate Isaac project is not
part of this training assignment.

Export `HF_TOKEN` and `WANDB_API_KEY` through the environment. HF credentials need
read access to the private starter and frozen assets (including gated DINO), and
write access to the designated Meta checkpoint repository. Never place credentials
in source, command arguments, configuration files, or published logs. The launcher
checks that both variables are set before creating a run. If credentials live in
`.bashrc`, use an interactive shell to load them. Export credentials in the
actual tmux pane that launches training: an existing tmux server need not inherit
new client variables, and non-interactive `bash -lc` may not load `.bashrc`.

## Installation and inputs

From the repository root, with Git and `uv` available:

```bash
bash scripts/setup_fqi_runtime.sh
export PYTHONPATH="$PWD/src"
export HF_HOME="$PWD/outputs/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
.venv-policy-eval/bin/hf download \
  yypeng666/metamdp-conveyor-optional-budget-starter-v1 --repo-type dataset \
  --revision b00053932f42e7c43f60a7af8415aad1efa1a1ea --include 'data/*' \
  --local-dir outputs/downloads/optional-starter
.venv-policy-eval/bin/python scripts/fqi_handoff.py restore-data \
  --bundle outputs/downloads/optional-starter/data \
  --output outputs/datasets/conveyor-optional-starter-v1
```

The pinned HF revision contains only data and its dataset card. The allowlist
downloads `data/*`; GitHub is the code source. The data identity is in `configs/data/conveyor_optional/hub.json`.
Restoration verifies archive checksums once and refuses to overwrite a cohort.
After successful restoration, the downloaded tar shards are disposable staging;
keep the restored dataset and small manifests, not both full copies.

The setup script creates `.venv-policy-eval` using Python 3.11.15 and
`requirements/fqi-runtime.txt`, restores pinned OpenPI/JEPA code, and downloads
frozen assets from `configs/data/conveyor_optional/asset-downloads.json`. It uses
project model caches and existing runtime patches. `uv` may reuse its user-level
Python/download cache; it does not install dependencies into system Python. Do not use the generic `uv sync`
environment for this job or install into system Python.

Frozen components: conditioned pi0.5 (step 15236), DINO, JEPA, and RGB decoder.
The Meta attention/memory network is initialized from scratch. Do not transfer
success-only weights, scale old Q values, or initialize formal training from an
engineering acceptance state.

## Data and reward

Starter: 16 training scenes × zero/nominal/family × four scheduler behaviors =
192 complete trajectories, 73,114 free decisions, approximately 24.93 GiB packed.
Behaviors are E25 current-only, E25 with forecast, E75 with forecast, and mixed
exploration. E25/E75 specify plan-age ticks (20 ms each), not epochs. The raw
records contain causal token observations, execution history, actions, operation
owners/timing, and parcel outcomes. They are Meta replay, not a VLA SFT dataset.

The source corpus was recorded under v1. V2 explicitly admits that unchanged
recording config and recomputes rewards from raw operation/outcome ledgers:

`reward = 10 * delivered - 5 * missed - 0.02 * compute_cost`

Relative prices are fixed in YAML: VLA 1, DINO 0.1603305, JEPA 0.1413566,
decoder 0.0214724, Meta 0.0849045. Zero logical latency still incurs compute cost.
Physical wall time and robot energy are not reward inputs. A miss is a parcel
crossing the configured miss-height plane; timeouts are recorded separately.
Rule-collected free decisions are priced as one deployment Meta decision;
`imputed_meta_calls` remains separate from the actual operation ledger.

V2 has an independent experiment/run identity. It does not modify original data
receipts, reuse old Bellman targets, or accept different action/scene protocols.
The starter delivered 83/1920 parcels (4.3%); it is scheduling coverage, not an
expert corpus. Engineering checks do not establish that this recipe will produce
a useful policy. Inspect early fixed validation, including hold/no-launch behavior.

## Scientific recipe

The unchanged temporal-memory architecture uses eight state-conditioned decision
query tokens, each 256-dimensional, with two shared-memory history reads and a
separate gated forecast read. The final readout concatenates queries and current
critical state. Observed history and predicted tokens remain distinct; missing
forecasts are masked. Protocol 3 changes scheduling permissions, not this memory
architecture. The schema-2 success-only documentation is historical background,
not the launch specification for this experiment.

| Item | Formal | Separate node qualification |
| --- | --- | --- |
| Meta initialization | Scratch, seed 0 | Independent scratch, seed 0 |
| Startup | 40 FQI passes × 1000 updates | 2 × 64 updates |
| Online | 100 cycles | 1 cycle |
| Each cycle | 24 new episodes, 2 × 1000 updates | 24 new episodes, 2 × 64 updates |
| Global batch | 256 | 256 |
| Validation | Initially, every 10 startup passes, each cycle end | Initially, startup end, cycle end |

Each validation has 9 greedy episodes: 3 fixed development scenes × 3 delays,
10 parcels each. These development scenes have historical evaluation exposure;
the independent test split remains unused. Temporal latency is not included here.
Collection rotates through eight train scenes per cycle with epsilon 0.2.
Replay online share rises smoothly from 10% to 50% across 100 cycles, with recent
preference, historical coverage, and oversampling caps. Target and effective
sampling probabilities, sample draws, and unique coverage are logged separately.

Protocol 3: G chooses Sleep/Observe; O chooses Sleep/direct Launch/Predict when
ready; P chooses Sleep/predicted Launch. Empty buffers hold with zero arm delta.
There is no starvation rescue, forced prediction, clean bootstrap, automatic
retry, or abort/replan. Absent forecast images are masked. Preserve the temporal
attention architecture and frozen model inputs.

Each pass freezes Bellman targets and performs the declared regression budget.
Adam persists. LR is 1e-4, gamma 1, gradient clip 0.5, CQL weight 0.01. Best is an
informational index based on all-regime mean budget return, with delivery/cost tie
breakers. All complete states are retained and published regardless of best.
Ordinary online continuation uses the preceding latest state.

## Hardware qualification and launch

Inspect allocated GPUs, available VRAM, CPU affinity AND cgroup quota, RAM, EGL,
and project disk/quota. The tested L40S node has two 46-GiB GPUs; this is evidence,
not a hardcoded requirement. Each rollout worker owns its own frozen model copy.
Measure completed trajectories/hour and peak VRAM before increasing workers.
More learner ranks need not be faster for this small network.

Generate a qualification recipe with `--acceptance` and use an independent
`RUN_DIR=outputs/qualification/optional-node-check`. It runs the bounded real
fit/collection/refit/validation path. Use a fresh formal run afterwards. Open a
persistent shell (for example `tmux new-session -s conveyor-fqi`), return to your
repository directory there, load your credentials, and run the following inside
that shell:

```bash
.venv-policy-eval/bin/python scripts/fqi_handoff.py recipe \
  --config configs/experiments/conveyor_sort/meta_budget_optional_v2.yaml \
  --starter outputs/datasets/conveyor-optional-starter-v1 \
  --output outputs/recipes/optional-formal-v2.json
export GPUS=0,1,2,3 TRAIN_GPUS=0,1
export ROLLOUT_WORKERS_PER_GPU=1 CPU_THREADS=4 LEARNER_CPU_THREADS=8
export EPISODE_CACHE_SIZE=256 MICROBATCH=full
export RECIPE=outputs/recipes/optional-formal-v2.json
export RUN_DIR=outputs/training/conveyor-budget-optional-scratch-v2
export WANDB_PROJECT=latency-meta-mdp-fqi-budget
export CHECKPOINT_REPO=yypeng666/metamdp-conveyor-optional-budget-meta-v1
bash scripts/launch_optional_fqi.sh
```

This allocation is an example, not an H100 optimum. Tune GPUs, `TRAIN_GPUS`,
`VALIDATION_GPUS`, workers/GPU, `VALIDATION_WORKERS_PER_GPU`, CPU thread settings,
`REPLAY_PREPARE_WORKERS`, and episode-context cache against completed throughput.
`MICROBATCH=full` processes each rank's full share of global batch256;
`auto` probes a fitting divisor without optimizer updates; integers request
accumulation. Resource tuning never silently changes global batch or data volume.
BF16 compilation is enabled; qualify CUDA/JAX/EGL on the actual node.

`EPISODE_CACHE_SIZE` bounds CPU contexts per learner rank. 256 avoids observed
64-entry cache thrashing on the 192-episode starter. Up to 768 episodes can be
active later; larger caches can help if aggregate rank RAM permits. There is no
promise of saturated GPU utilization or a universal optimal worker count.

## Monitoring, checkpoint return, and recovery

One central tmux console and W&B run cover startup, collection, target building,
updates, and validation. Training logs include loss, TD/CQL, Q diagnostics,
clipping, LR, sampling coverage, throughput, and hardware time series. Validation
uses cumulative actual updates on its x axis, includes per-delay and per-scene
outcomes/module calls/costs, and distinguishes collection from greedy evaluation.

Every atomically completed iteration state (including initialization and phase
starts) is asynchronously published to the public, manually gated model repo:

`runs/<wandb-run-id>/checkpoints/<campaign-relative-checkpoint>/`

The uploader requires an existing public repo with manual gating; it will not
silently create an unrestricted model repository. Access is provisioned by the
owner. Each checkpoint contains state, receipt, config, recipe and manifest.
`runs/<wandb-run-id>/index.json` maps checkpoint identity, actual total updates,
checkpoint lineage updates, validation results, Git commit and W&B URL. Missing
durable update events are marked with a null count, not inferred from planned work. The
W&B checkpoint table links directly to these files. Unvalidated states are marked
with null validation, never assigned another checkpoint's evaluation.

Publication is retryable and does not block GPU training. Pending counts/errors
remain visible. If training ends with unsent checkpoints/index, the supervisor
exits 75; the learned state is intact. Retry uploads without training:

```bash
.venv-policy-eval/bin/python scripts/fqi_handoff.py sync-checkpoints \
  --root "$RUN_DIR" --config configs/experiments/conveyor_sort/meta_budget_optional_v2.yaml \
  --recipe "$RECIPE" --repo "$CHECKPOINT_REPO" --run-id YOUR_WANDB_RUN_ID \
  --wandb-url YOUR_WANDB_RUN_URL
```

Resume training with the same launcher, commit, config, recipe, run directory and
W&B identity. Completed episodes/iterations are reused. A checkpoint download is
sufficient for model evaluation, but exact campaign continuation also requires the
journal and retained online replay. Do not claim HF weights alone restore replay.

## Storage lifecycle

- Immutable 192-episode starter and frozen models are reusable inputs.
- Keep the latest 20 online batches plus 96 historical episodes, subject to a
  96-GiB payload cap. Protect the two newest replay recovery snapshots.
- Rebuildable replay cache is capped at 192 GiB; campaign at 360 GiB; preserve
  at least 40 GiB free. Shared filesystem free space does not establish user quota.
- Retire unneeded online heavy payloads only after completed fit/validation and
  replay reference checks. Keep journal, inventory, compact metrics and tombstones.
- Validation saves compact audit/metrics/video rather than training feature copies.
- Keep all completed checkpoint states: approximately 52.5 MiB each, up to about
  17.5 GiB for 342 states including phase-start copies. Do not delete journal targets.

Allow at least 600 GiB project storage; 1 TiB is practical. Avoid duplicate model
caches, downloaded tar staging after restoration, and copied run trees.

## Collaborator completion checklist

Return the Git commit, config/recipe, W&B link, HF run index, final upload status,
and per-delay success/cost results. Review the first scheduled formal validation
before spending the full allocation. Do not change science silently, restore an
old Meta initializer, or stop solely for a single noisy validation fluctuation.
The engineering acceptance recipe is an infrastructure check, not an efficacy
experiment or the starting checkpoint for formal training.
