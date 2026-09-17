# Conveyor task and source recordings

Run from the repository root in the simulator/Belief environment described in
[environment setup](../../docker/README.md). It includes RoboSuite, MuJoCo and
PyArrow; video recording also uses FFmpeg. In Docker:

```bash
export CONVEYOR_PYTHON=/opt/belief/bin/python
export PYTHONPATH=src
export MUJOCO_GL=egl
```

For the existing local environment, set `CONVEYOR_PYTHON=.venv-jepa/bin/python`.
Keep longer batches in tmux. Each run needs a new output directory.

## Configure the scene

- [Task](../../configs/tasks/conveyor_sort/surface.yaml): supply duration, fixed
  parcel count, arrival groups, geometry, colors and selected camera.
- [Expert](../../configs/data/expert/conveyor_surface.yaml): interception and
  pick/transfer/release motion parameters.
- [Controller](../../configs/runtime/control/panda_osc_pose_delta_conveyor_v2.yaml):
  the shared action contract for this task's expert and learned policies.

`supply_ticks` uses 20 ms ticks. Set `spawn_count` for an exact count within that
window; set it to `null` for duration-only sampling. In fixed-count mode,
`moderate`/`sparse` intervals specify relative inter-group gap weights, while
`burst` intervals remain literal ticks. The actual schedule is saved in
`arrivals.json`. The final component of `goal_rgba` is opacity (0 to 1).

## Preview or execute the expert

```bash
"$CONVEYOR_PYTHON" scripts/preview_conveyor.py \
  --config configs/tasks/conveyor_sort/surface.yaml \
  --seed 4 --output-dir outputs/conveyor/preview-seed4

"$CONVEYOR_PYTHON" scripts/run_conveyor_expert.py \
  --config configs/tasks/conveyor_sort/surface.yaml \
  --expert-config configs/data/expert/conveyor_surface.yaml \
  --seed 4 --output-dir outputs/conveyor/expert-seed4
```

Videos are `preview.mp4` and `expert.mp4`; `status.json` reports outcomes and
resolved configuration. `trajectory.jsonl` and `arrivals.json` are diagnostics.
`--no-video` disables video output. One parcel's delivery does not reset the arm
or end the episode; supply stops first, then remaining parcels are processed up
to the configured drain limit.

## Record a source episode

```bash
"$CONVEYOR_PYTHON" scripts/run_conveyor_expert.py \
  --config configs/tasks/conveyor_sort/surface.yaml \
  --seed 4 --record-source --no-video \
  --output-dir outputs/conveyor/source-seed4
```

This still renders both cameras for the source. Only a fully successful expert
episode receives `source/manifest.json`; unsuccessful runs retain diagnostic
results and report `expert_admitted: false`. A completed simulator run alone is
not proof that its source was admitted.

The source contains:

- `frames.parquet`: N+1 observations, with lossless embedded PNG for both cameras
  and current 16D proprio at each 20 ms boundary.
- `transitions.npz`: N controller-native 7D actions, per-transition delivered
  parcel counts and a terminal flag only at the whole episode's end.
- `manifest.json`: episode/seed identity, control and scene configuration,
  outcome summary and completion metadata.

Standalone recordings are development sources. The corpus config below assigns
the source purpose for a collection. Existing moving-ball L2/L3 training jobs do
not accept conveyor data.

## Collect and prepare a development corpus

The smoke config declares eight training seeds and reserves validation/test
seeds without collecting them. Split membership is by complete episode seed.

```bash
"$CONVEYOR_PYTHON" scripts/collect_expert.py --task conveyor_sort \
  --config configs/data/conveyor_sort/source_smoke.yaml \
  --output-dir outputs/conveyor/corpus-smoke
```

Use `--resume` with the same configuration and source implementation to reuse
completed episodes. Failed episodes remain in the request inventory and prevent
the corpus from being marked ready. Incomplete or changed runs need inspection;
they are not silently overwritten or replaced with different seeds.

Policy preparation runs in the OpenPI environment (`python` in Docker or the
configured local `.venv-openpi/bin/python`). Run it as a fresh CLI process:

```bash
python scripts/prepare_policy.py --task conveyor_sort \
  --corpus-manifest outputs/conveyor/corpus-smoke/manifest.json \
  --output-dir outputs/conveyor/policy-smoke
```

This exports the train split to local LeRobot Parquet, fits train-only state and
action statistics, checks cross-delivery and episode-tail windows, and reads an
actual OpenPI batch. `--resume` reuses a matching completed export. The local
`--repo-id` names a dataset directory; these commands do not publish or download
a Hugging Face dataset. `preparation.json` records identities, counts and batch
checks. No model weights are trained or evaluated by this command.

For training sources, use
[`source.yaml`](../../configs/data/conveyor_sort/source.yaml) instead of the smoke
config and new corpus/preparation directories. It requests 200 train and 20
validation episodes, reserving 20 test seeds without collecting them. Use
`--repo-id metamdp/conveyor_sort_surface` when preparing that corpus. Only the
train split enters policy preparation; validation source episodes remain in the
corpus. Do not relabel smoke artifacts as training sources.

## Package and launch clean SFT

In the OpenPI environment, package a completed preparation and create a concrete
job bound to its manifest:

```bash
python scripts/package_policy.py \
  --preparation-dir "$PREPARATION_DIR" \
  --output-dir "$BUNDLE_DIR" --job-output "$JOB_FILE" \
  --run-name conveyor-clean

python scripts/train_clean_policy.py \
  --config "$JOB_FILE" --output-dir "$RUN_DIR" --check-data-only
```

The bundle contains the train dataset, normalization and preparation metadata.
On the same filesystem, Parquet payloads are hardlinked; otherwise they are
copied. Treat prepared and packaged payloads as immutable. Metadata and
normalization are copied independently. Verification hashes the small manifests
and normalization file and checks payload sizes; it does not hash the full corpus.

The generated job uses eight devices and global batch 256. The
[`conveyor_clean.yaml`](../../configs/training/policy/conveyor_clean.yaml) recipe
specifies three source epochs; the launcher derives the step count, warmup and
three equally spaced permanent milestones from the actual train frame count.
Other saves are rolling recovery checkpoints. `publish_repo: null` keeps the
checkpoints local. Setting a publication destination uploads the three milestones
after training completes, using the shared publisher.

`--check-data-only` reads a real OpenPI batch on the available device. It does not
load model weights, run an optimizer step or verify the requested GPU topology.
On the intended training node, check the topology and then launch:

```bash
python scripts/train_clean_policy.py \
  --config "$JOB_FILE" --output-dir "$RUN_DIR" --check-only
python scripts/train_clean_policy.py \
  --config "$JOB_FILE" --output-dir "$RUN_DIR"
```

Keep training in tmux and use `--resume` after interruption. Training requires a
`training_source` bundle; a `development_smoke` bundle permits checks only.
The local dataset path is relative to the generated job file. To distribute a
published bundle, replace its dataset source with `kind: huggingface`, `repo_id`
and a pinned 40-character `revision`, retaining `manifest_sha256`. Profile and
training recipe paths in these task jobs are repository-relative. Packaging does
not publish data or assign a collaborator training task.

## Data semantics

`ConveyorSource.policy_sample(h)` returns current RGB/proprio/prompt and the next
H50 actions. Only the true episode tail is padded and masked. A delivery in the
middle of the episode is not a boundary. `belief_window(h, q)` returns real
h−8/h−4/h history, executed controls, the masked q-step executable prefix and
separate future RGB/proprio supervision; encoding and normalization occur later.

Split by complete scene seed, never by frames, individual parcels or overlapping
windows. Keep current and forecast variants of the same episode in the same
split. Fit normalization on training data only. The full-success admission rule
applies to expert training sources; Meta RL replay must retain failed interactions.

## Prepare frozen vision features

In the Belief environment, encode completed train/validation sources with the
existing frozen DINO model:

```bash
"$CONVEYOR_PYTHON" scripts/prepare_vision.py --task conveyor_sort \
  --corpus-manifest "$CORPUS_DIR/manifest.json" \
  --output-dir "$FEATURE_DIR"
```

The encoder uses the pinned local model cache; add `--allow-download` if needed.
Each episode has one memory-mapped FP16 array with shape `[N+1, 2, 196, 384]`.
All observation boundaries, including the terminal observation and boundaries
around individual deliveries, are retained. `--boundary-batch-size` controls
encoding memory. `--resume` reuses completed episodes with matching source,
encoder and preparation identities. Test seeds are excluded.

This command prepares visual features only. Belief normalization is separate
from the clean-SFT statistics above.

## Prepare and check Direct Belief training

Set `CONFIG` to a task job such as
[`conveyor_sort/belief.yaml`](../../configs/experiments/conveyor_sort/belief.yaml).
It binds the corpus, frozen feature cache, controller and shared Direct recipe.

```bash
"$CONVEYOR_PYTHON" scripts/prepare_belief.py --config "$CONFIG"
"$CONVEYOR_PYTHON" scripts/check_belief.py --config "$CONFIG" \
  --output-dir "$CHECK_DIR" --optimizer-steps 20
"$CONVEYOR_PYTHON" scripts/train_belief.py --config "$CONFIG" \
  --preflight "$CHECK_DIR/preflight.json" --output-dir "$RUN_DIR"
```

Preparation fits proprio normalization on all train observation boundaries,
including the final recorded state. Validation uses the same train statistics.
The shared dataset samples q=1–20 from real endpoints, keeps h−8/h−4/h history
and masks controls after q. Individual deliveries never split a sequence.

Preflight runs real gradient updates and latency measurements without saving
weights. Formal training requires committed implementation code and a matching
preflight; run it in tmux. Task checkpoints carry the conveyor/controller
identity and cannot be substituted for moving-ball checkpoints. The existing
AR-comparison review command is for moving-ball; conveyor validation reporting
and decoded scene review require their task adapter before downstream admission.
