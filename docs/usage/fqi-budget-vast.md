# Budget FQI on Vast

Status as of 2026-10-25: the historical campaign stopped after complete C10 and
has been backed up. The user subsequently authorized abandoning the machine;
Vast instance `57559976` was destroyed and its absence verified on 2026-10-26 UTC.
See `outputs/operations/maintenance-backup-20261025/instance-release.json`.
It is not the starting checkpoint for the next experiment.
The user's corrected mainline is described in the
[independent scratch-budget design](../superpowers/specs/2026-10-25-budget-fqi-scratch-design.md)
and [implementation plan](../superpowers/plans/2026-10-25-budget-fqi-scratch.md).
That protocol is planned, not yet implemented or launched. Observe/Predict
binding is [under renewed design review](../superpowers/specs/2026-10-25-observe-predict-decision.md).

Historical continuation: [scale-v2 recipe and portable resources](fqi-budget-scale-v2.md).
The first cycle below is completed historical evidence. Scale v2 uses global
batch 256, microbatch 128 and 1,000 updates per FQI, with a per-cycle replay
curriculum. The operator cancelled the paired 50/50-versus-90/10 probe.

This is a new experiment on `vast-1024`, not a continuation of the old success-only
campaign. Ideas2 is a read-only migration source; do not launch new experiments
there. The Isaac project and ManiGuard are outside this task.

## Preserved contracts

The attention memory, observed/predicted information boundary, action vocabulary,
guarded starvation protocol, frozen models, task geometry and ten-object episodes
remain unchanged. The legacy campaign and its raw receipts remain unchanged.
The completed first cycle used global batch 64 independently of GPU count.

The source campaign completed 40 startup fits but began online collection from
startup best iteration 5. Distinguish total computation performed from checkpoint
lineage. Historical online cycle 1 raw payloads are no longer complete because of
retention. A paired diagnostic using the retained cycle 74 cohort cannot establish
what caused the original cycle 1 transition.

## New training semantics

- A derived, versioned reward view gives +10 per delivery and -5 per once-only
  `miss` event. Timeouts lose delivery credit but do not receive an additional
  drop penalty.
- Actual VLA, DINO, JEPA, decoder, Meta and bootstrap invocations use fixed prices.
  Physical wall time is a profiling diagnostic, not the reward. Initial bootstrap
  work contributes to episode cost but is not assigned to the first free decision.
- Forced work after a free decision folds into that preceding decision. Ledger
  boundary `t+1` belongs to the action executed at `t`.
- Initial sampling targets 90% startup and 10% online. Latest mass is capped at
  10%; each online group's per-decision probability is capped at four times the
  startup per-decision probability. Excess returns to startup. These are decision
  probabilities, not episode fractions.
- The S05 warm-start preserves representation weights and multiplies final Q
  head weights/biases by 100. Adam and sampler state reset. Huber beta and CQL
  temperature are both 100, with the temperature-prefactored log-sum-exp and the
  old conservative coefficient. This matches loss units, not optimizer dynamics.
- New configuration hashes and explicit hash-pinned old source configurations
  admit historical data only after immutable runtime contracts match. Never edit
  historical receipts to make them appear compatible.

## Acceptance gates

1. Verify CUDA, nonconstant MuJoCo EGL RGB, pinned DINO inference, credentials,
   frozen asset hashes and S05 checkpoint restore.
2. Migrate the complete 448 startup episodes, the retained 24-episode diagnostic
   cohort, frozen assets and representative checkpoints. Verify destination hashes
   against the migration manifest before deleting any source payload.
3. Check reward coverage on all admitted episodes. A truncated smoke may save
   diagnostic evidence but cannot claim complete budget accounting or enter replay.
4. Run a fresh S05 training-scene reference with real VLA inference. Fix module
   prices from that reference runtime and calibrate the initial coefficient to
   approximately two reward points per reference episode. Record the calibration
   data, warm-up exclusions, prices and identities before training.
5. Check replay disk requirements before building caches. Leave capacity for new
   collections, checkpoints and evaluation; do not assume the old Ideas2 cache
   fits in the Vast container.
6. Run the first budget training cycle. Require genuinely
   new episodes, finite optimizer updates, a hash-verified reloadable checkpoint,
   and fixed-protocol validation before extending the campaign.

TD and conservative losses, signed TD error, Q/target values, gradient clipping
and group/stage diagnostics are reported separately. Delivery ratio, drop count,
module invocations, fixed-price cost and scalar return must remain separate metrics.
No compute-saving or policy-improvement claim follows from a smaller fitting loss.

## Operational evidence

Local and remote qualification artifacts use
`outputs/operations/fqi-budget-vast-20261024/`. `progress.md` is the execution ledger.
The requested remote project location is `/root/meta_mdp_LIBERO`, with its
isolated `.venv-policy-eval`. During the initial transfer and collection it is a
symlink to the staged `/workspace/meta_mdp_LIBERO`; at the first safe boundary,
the finite continuation job performs a same-filesystem rename into home and
retains the old path only as a compatibility symlink for recorded launch paths.
Long jobs use named tmux sessions and retained console logs.

## One experiment console and W&B run

Use `conveyor-budget:main` on Vast as the experiment's continuous console:

```bash
ssh -t vast-1024 'tmux attach -t conveyor-budget'
```

`scripts/run_fqi_experiment.py` supervises the existing sequential controller.
The Vast launcher prints a timestamped summary every 10 seconds, including the current phase,
completed collection episodes, replay preparation, target construction, optimizer
updates, validation episodes, latest loss and elapsed time. Target construction
emits durable progress about every ten seconds; it does not count as an optimizer
update. Detailed child output remains in the operations directory's
`continuation-console.log`; the main console is appended to the run's `console.log`.

The supervisor owns one W&B identity at `telemetry-main/wandb-run.json` and one
writer lock. Relaunch with the same run directory and configuration to resume
that identity. It backfills durable collection and fit records, verifies the
already published source prefix, and continues cumulative optimizer/episode axes.
Collection curves follow manifest order so a worker's completion order or a
restart cannot reorder published points. Out-of-order completions appear in the
console count immediately; their curves wait for earlier manifest entries.

The child inherits `CONVEYOR_STAGE_WANDB=0`, which keeps local stage records but
disables per-collection and per-fit W&B runs. A collection already running before
deployment may finish its historical run normally; its episodes are also included
in the new main run. Future stages belong to the central run. Do not start a
second telemetry writer or controller for the same output directory.

The main run separates `train/*`, `fqi/*`, `collection/*` and `validation/*`.
Training uses cumulative actual optimizer work; FQI diagnostics use cumulative
iterations; fixed validation uses the online cycle. `delivery_ratio`,
`budget_return`, `compute_cost` and per-module `calls/*` remain distinct curves.
An interrupted and repeated update remains actual work, not checkpoint lineage.
Telemetry/network errors retain local evidence and do not stop training.

For a new experiment, launch the wrapper around that experiment's controller:

```bash
python -u scripts/run_fqi_experiment.py \
  --root "$RUN" --operations "$OPS" --config "$CONFIG" \
  --project "$WANDB_PROJECT" --name "$EXPERIMENT_NAME" \
  --poll-seconds 10 --command python -u "$CONTROLLER"
```

The current operations launcher can also bootstrap the supervisor from the
verified staged patch while the initial collection finishes. Source changes are
applied only at the idle stage boundary, before the new fit binds its source.
Restarting that launcher reuses verified migration and source-change receipts.

The instance's 500 GB workspace is container storage, not an attached persistent
volume. Keep irreplaceable checkpoints and receipts backed up before recycling or
destroying the instance. Transfer credentials are temporary, project-restricted,
and must be revoked after migration verification. User HF/W&B credentials must
never be copied into logs or tracked configuration.

The optional `--compact-replay` mode writes deduplicated raw features directly
to a verified mmap cache. FP16 is used only when every original FP32 value
round-trips bit-for-bit; other arrays remain FP32. This avoids a redundant large
FP32 cache during migration without weakening source or destination SHA checks.

The first continuation runs one new collection cycle with two 500-update fits
(1,000 optimizer updates total, not 500 FQI iterations). Its 24 newly collected
episodes and fitted checkpoints are reusable mainline experiment artifacts.
The paired sampler diagnostic was subsequently cancelled by the operator.
For this first cycle, the operator's speed priority skipped initial validation and performed the
nine fixed validation episodes once at the end, using four workers on the A100.
Use microbatch 64 with global batch 64, without changing the sampling recipe.
It waits for complete
migration and new collection, verifies all pinned destination files, and retains
a 64 GiB free-space gate before training. It does not extend the old campaign or
automatically schedule additional cycles. Apply the worker-count forwarding patch
only after the active collection and cache jobs finish; record both source trees
and preserve the existing collection receipts. A small real-context benchmark
guides microbatch selection but does not measure full-cohort wall time or efficacy.

## A100 stage resources

The instance has 80 GiB VRAM and a cgroup CPU quota of 23.04 cores. Current resident
rollout processes each use approximately 17.1 GiB; four replicas completed the
nine-episode validation in 21.6 minutes. Collection speedup remains to be measured. Five
replicas exceed VRAM at the observed footprint.

| Stage | Processes | CPU threads per process | Relevant options |
| --- | --- | --- | --- |
| New collection cohorts | 4 on GPU 0 | 5 | `--gpus 0 --rollout-workers-per-gpu 4 --cpu-threads 5` |
| Attention-fit validation | 4 on GPU 0 | 5 | `--rollout-workers-per-gpu 4 --validation-cpu-threads 5` |
| Attention learner | 1 on GPU 0 | 8 | `--cpu-threads 8 --global-batch 64 --microbatch 64 --bf16` |
| Replay preparation | 4 CPU workers | controlled separately | `--replay-prepare-workers 4` |

The already active first collection retains its two immutable shards. Do not
change its worker count mid-cohort. Subsequent cohorts use the four-worker
profile. These values are launch arguments, not scientific hyperparameters.
The bounded fit launcher records effective arguments in `fit-resource-config.json`.

The existing coordinator partitions each manifest exactly once, gives each worker
its own directory and log, and admits only verified complete attempts through a
single tracker. GPU/output locks prevent overlapping learner and rollout groups;
workers in the same rollout group share the GPU intentionally. On failure, the
coordinator terminates only its children. A cohort's shard count remains fixed on
resume. Four-worker tests cover a nine-episode uneven partition, verified merge,
failure cleanup and resume rejection when shard counts differ.

After transfer ends, destination verification runs concurrently with unfinished
collection/prepacking; training still waits for all three to complete. The active
source tree is unchanged until that boundary. Rollout startup peak memory and
episodes per minute still require observation during the actual four-worker stage.

## Replay preparation and cache reuse

The mmap cache contains immutable raw observations and execution histories.
Training admission checks only the summary, episode completion receipt and the
three consumed replay files; archived RGB frames and videos are not traversed.
Full artifact verification remains at publication and migration boundaries.
Admission and cache preparation share a persistent file-digest cache under
`replay/verification/`: the first encounter hashes a file, while subsequent
encounters check its device, inode, size, modification time and change time.
Unchanged source and packed arrays are not read again to recompute SHA. Changed
or replaced files are rehashed against their completion receipts. Receipt bindings,
training splits, episode outcomes and reward semantics are still checked.

Existing caches without verification metadata require one initial pass to populate
this metadata. Newly collected episodes require their initial conversion; earlier
converted episodes are reused. This optimization does not reuse learned features
or Bellman targets across FQI iterations. Those depend on the current model.

`fit/preparation-progress.json` records admission/preparation counts and elapsed
seconds for the current fit. The central console distinguishes these counts from
`historical_prepack`, which is the earlier migration preparation job. Operational
Python launchers must put all execution behind `if __name__ == "__main__"` so a
spawned learner imports the launcher without repeating replay preparation.

## Input throughput and standard console logs

The central run captures stdout (`console="wrap"`) and prints a status line every
10 seconds in the Vast launcher. During regression the line includes cycle,
FQI iteration, optimizer step/budget, cumulative updates, total loss, TD loss,
weighted conservative loss, gradient norm, learning rate, updates/second, input
wait time, and the remaining regression-time estimate. Target preparation and
validation remain separate named phases in the same window and run.

Attention FQI now transports lossless compact raw frames through a bounded CPU
producer. Shared frames are sent once per microbatch in their existing storage
precision, pinned in host memory, then restored to the exact dense FP32 public
inputs on the GPU. One future microbatch is prepared while the GPU processes the
current one. Learned visual pooling is still computed by the model; no learned
representations or target-network values are cached by this transport.

Admitted replay metadata has a disposable `metadata-v1` binary cache, keyed by
source path, selected fields, Python version, and file stat signature. Full packed
records are cached; reward reconstruction caches only operations and parcel
events from the larger source records. Canonical JSON remains authoritative.
Removing these derived caches causes reconstruction, not loss of source data.
The initial 472-episode cache was prepared with 12 CPU processes; ordinary cache
hits do not parse those source JSON files again.

A100 acceptance used actual replay, the unchanged batch of 64 and BF16. A warm
8-update comparison reduced target construction from 7.05 to 1.32 seconds and
regression from 6.12 to 1.25 seconds. Both deterministic runs produced the exact
same final model hash. These are controlled warm-input timings, not whole-cycle
speedups: cold episode reconstruction, collection, and validation are measured
separately. The first metadata build took 25.96 seconds and occupied 3.04 GiB.

Production iteration 2 then completed 500 new updates: target construction took
280.66 seconds, regression 58.21 seconds, and the complete fitter 381.19 seconds.
The preceding iteration took 971.89, 392.62, and 1405.24 seconds respectively.
These are before/after production observations with different target networks
and sample plans; the controlled comparison above establishes numerical equality.
Twenty two-second GPU samples during regression averaged 55.2% utilization
(45-61%). CPU context reconstruction remains a separate cold-path cost.

The repair preserved the complete iteration-1 checkpoint (500 updates), Adam,
and decision-sampler state. An explicit pinned runtime receipt records both the
original experiment identity and the full executed Python tree. Historical
checkpoint receipts and W&B scalar-history prefixes are unchanged. Evidence and
recovery launchers are under the local/remote
`outputs/operations/fqi-budget-vast-20261024/input-throughput` directory.

The metadata loader changes the converter's source file without changing any
conversion field other than that file's digest. The pinned compatibility receipt
therefore preserves the original converter protocol identity as well as the
experiment identity; the actual executed source digest remains separately
recorded. An exact comparison of all 472 episode conversion records established
this compatibility before resume. Do not remove binding checks or rewrite old
checkpoint receipts to accommodate a source change.
