# Conveyor information scheduler

`scripts/run_conveyor_meta.py` is the entrypoint for the conveyor information
scheduler. Its experiment configuration is
[`meta_information.yaml`](../../configs/experiments/conveyor_sort/meta_information.yaml).
This workflow trains the Meta scheduler; the conditioned VLA, visual encoder,
Belief predictor and RGB decoder remain frozen. It is separate from policy SFT
and from the older generic Meta pipeline.

## Execution and learning contract

The scheduler uses three stages: choose when to observe, whether to predict,
and whether to request a new VLA chunk. The current collection recipe uses
`bound` mode, which forces prediction after observation. The stage interface also
supports `split` mode; this does not mean the bound-mode dataset can be reused
unchanged for a split-mode experiment. Raw camera history continues to be
collected between scheduler decisions.

Actor inputs contain causal cached observations, history and execution state.
Simulator object state used for outcome auditing is kept outside actor inputs.
RTC remains responsible for buffer alignment and execution. Forced scheduling
steps are recorded and folded into free-decision transitions; they are not
treated as independent learned choices.

Each delivered ball earns 0.1 reward. The current configuration accepts only
undiscounted FQI with zero budget penalty. Misses and timeouts receive no added
penalty. Complete unsuccessful episodes remain in replay; interrupted or
otherwise invalid episodes are retained as artifacts but excluded from training.
Measured module costs and logical inference delays are separate quantities.

FQI recomputes recurrent history with a frozen target network for each fitted
iteration. Regression uses a Huber Bellman loss and the configured conservative
penalty over legal actions. Recurrent gradients are truncated in blocks;
optimizer updates occur at episode boundaries. Mixed replay weights the offline
and online sources equally by their mean transition loss.

## Explicit workflow stages

Run commands from the repository root, with `src` on `PYTHONPATH`. Use the pinned
environments described in [environment setup](../../docker/README.md). Inspect
GPU processes before launching model work and keep long commands in a persistent
terminal with a log. `--help` lists the entrypoint arguments.

| Command | Inputs and purpose |
| --- | --- |
| `preflight` | Verify configured assets, runtime dependencies and GPU availability. |
| `smoke`, `diagnose` | Bounded execution checks; not formal training data or evaluation. |
| `bootstrap` | Generate initial action caches for the starter cohort, using the separately loaded clean policy. Requires `--scene-audit`. |
| `collect` | Collect the starter cohort with the frozen conditioned services. Requires the same audit and bootstrap directory. |
| `train` | Fit the offline scheduler from `--cohort-dir`; `--stop-after-iterations` allows a bounded run. |
| `cycle-bootstrap` | Prepare bootstrap caches for an explicitly selected learned-policy cohort. Requires `--checkpoint`, `--phase`, and `--scene-audit`. |
| `cycle-collect` | Run that cohort with the fixed scheduler checkpoint and record real closed-loop outcomes. |
| `online-train` | Fit a new generation from `--offline-dir` and online `--cohort-dir`, initialized by `--checkpoint`. Requires `--compatibility` and `--scene-audit`. |
| `monitor-series` | Retrospectively evaluate saved iterations from a training directory supplied as `--checkpoint`. Requires `--bootstrap-dir` and `--scene-audit`. |
| `campaign` | Run a bounded, synchronous collection/refit/validation recipe. Requires `--campaign-recipe`, `--scene-audit`, `--gpus`, and `--train-gpus`. |

All commands require `--config` and `--output-dir`. `online-train` performs the
mixed-replay optimization stage; it does not collect online rollouts itself or
automatically repeat the collection/update cycle. Supply `--bootstrap-dir` to
enable simulator validation between its fitted iterations. Without that option,
it performs optimization without interleaved validation.

The starter recipe uses the first 16 configured training scenes, four scheduler
behaviors and two delay regimes. The online cohort uses the next 16 training
scenes: 24 exploratory learned-policy episodes and eight rule-policy episodes.
These are the default fixed cohorts. An explicit `--generation` JSON can select
additional audited training scenes for the cycle commands.

Scene-audit JSON and its referenced evidence, bootstrap caches, model assets and
replay archives are required run inputs. They are not installed by a Git checkout.
The collector verifies their identities. Compatibility records explicitly bind
the configuration and permitted collector code hashes; do not bypass identity
checks to combine datasets with different semantics.

## Validation, recovery and telemetry

The `monitor` phase evaluates the first three validation scenes in each of zero,
nominal and family delay, for nine paired episodes. It requires `--meta-only`.
The older `validation` phase uses four scenes and two delay regimes and can
include rule baselines; these are distinct protocols.

Interleaved validation runs at initialization and after every fitted iteration.
Report `validation/zero/reward`, `validation/nominal/reward` and
`validation/family/reward` separately. The mean of nominal and family reward
selects `best`; ties retain the incumbent. Zero-delay reward is a separate
diagnostic. These small development cohorts do not establish final test efficacy.

Training publishes `latest` after saving a complete iteration. Validated runs
retain `best` and `latest` weights, plus completion receipts for every iteration.
Retrospective monitoring retains intermediate weights until the entire series
has been evaluated. When evaluating an offline `train` checkpoint with
`monitor-series`, also pass `--offline-dir` pointing to its original source cohort;
the evaluator verifies and records that manifest hash. Keep receipts and retention metadata when moving a run.
Resume with the same inputs and directory: recovery verifies the saved binding,
restores learner RNG state, and finishes pending validation before the next update.

Local JSON records, execution logs and per-episode videos are the durable evidence.
W&B tracking reports optimizer loss, iteration progress and validation reward;
dataset reward statistics describe the collected behavior, not a newly evaluated
policy. Set `WANDB_ENTITY` to choose the destination account or team and optionally
`WANDB_PROJECT` to override the default project. With no entity override, the SDK
uses the authenticated account default. `scripts/track_meta_fqi.py` provides a separate log-tracking entrypoint for
offline training. External telemetry failures do not discard local results.

Validation binds the Python source hash. Do not edit the active checkout's Python
files during a validation series or change its recorded protocol to resume across
different implementations. Use a separate output directory for changed protocols.

## Serial and parallel execution

The default collector remains serial. `parallel-collect` partitions an unchanged
cycle manifest by episode index across one resident frozen-model worker per GPU.
For example, select physical devices with `--gpus 4,5,6,7`, and supply the same
`--checkpoint`, `--phase`, `--scene-audit`, `--config`, and `--meta-only` options
as `cycle-collect`, plus an existing `--bootstrap-dir`. Choose the combined
OpenPI/Belief runtime with `--worker-python /absolute/path/to/python`.
The same command accepts a single physical index for serial rollout execution.
Set `--cpu-threads` for each rollout/learner process (default four). Supply any
required compatibility overlays through the launch environment's `PYTHONPATH`;
the validator preserves it and appends this checkout's `src`. The interpreter
defaults to the calling Python executable. No host, device list or overlay path
is selected by the implementation.

Each worker writes its own directory. The coordinator verifies completion hashes,
publishes relative episode links, and reports per-episode results to W&B as they
finish. A failed worker stops its siblings; resuming the same command verifies and
reuses complete episodes. Preserve the entire `workers/` tree with its parent
output directory. Do not start a separate serial writer inside this directory.
The worker count, GPU assignment and runtime are fixed when resuming this output.

GPU admission uses `CUDA_VISIBLE_DEVICES`; without it, admission checks all GPUs.
Explicit `--allow-shared-pids` exceptions are available for operator-approved
contexts. They do not authorize terminating processes. Parallel coordinators also
hold per-device and output-directory locks. Other applications do not honor these
locks, so check device occupancy before launch. CUDA and EGL are mapped separately
so each worker renders and performs inference on its assigned physical device.

`online-train` and `monitor-series` accept `--gpus` and `--worker-python` for
parallel validation. Without `--train-gpus`, the existing episode-wise FQI learner
remains single-device. Use `--train-gpus 4,5,6,7 --gpus 4,5,6,7 --global-batch 4`
with `online-train` to parallelize both Bellman targets and regression. The CPU
coordinator publishes checkpoints and W&B events; learner ranks terminate before
all nine validation episodes run, releasing CUDA contexts and device locks.

The synchronous minibatch recipe is explicit: each global batch averages the
original source-weighted episode losses, with complete causal recurrent histories
and the existing truncated-gradient boundaries. Ranks SUM gradients, then clip
and update identically. A single device with `--train-gpus 0 --global-batch 4`
accumulates the same four episodes before each update. Global batch size is
independent of device count. With 160 episodes and two passes, batch four performs
80 updates; this differs from the original 320 episode-wise updates. Learning rate
and epoch count are not automatically scaled. Compare throughput at equal global
batch/data exposure and inspect closed-loop rewards when changing recipes.

`--replay-cache-gib` bounds the aggregate host episode cache across learner ranks
(default 128 GiB for the distributed path). A verified, lossless uncompressed
copy under `prepared-archive/` avoids repeated DEFLATE work across iterations;
source data and cohort bindings remain unchanged. Allow additional disk space for
these copies. Preparation checks both admitted source receipts and cached content.
Use `--replay-cache-dir` to place or reuse these copies on suitable local storage
and `--replay-prepare-workers` to bound concurrent preparation (default four).
Learner hardware, CPU/cache allocation and paths are recorded separately in
`execution-history.jsonl`; they are not part of the minibatch objective. A completed
iteration can resume with a different device count while keeping global batch,
replay, optimizer settings and source version fixed. Floating-point collective
reduction order can cause small numerical differences across device counts.

The synchronous learner also accepts `--replay-storage mmap`. Preparation validates
the original episode, folds forced stages using the canonical decoder, and stores
public packet metadata plus a deduplicated read-only NumPy visual array. Worker
processes attach these arrays without copying or decompressing repeated visual
inputs. Hashes are checked during admission; source and derived file identities
are checked on every access. Only raw inputs are cached: learned visual features,
recurrent histories and targets are recomputed. The mmap format has its own
versioned namespace within the selected cache directory. Mmap uses reclaimable
operating-system pages; the archive
LRU's `--replay-cache-gib` limit does not bound that page cache. Include one-time
conversion cost when reporting cold-start throughput.

`--warm-start-checkpoint` initializes a new fit from a later compatible scheduler
while `--checkpoint` continues to identify the policy that collected the online
data. Both identities are recorded separately. `--iterations` sets the new fit's
outer-iteration budget; `--stop-after-iterations` supports bounded verification.
Resume an unchanged recipe in the same output directory. Use a new output when
changing the batch recipe or source code; preserve previous receipts and evidence.

The selected runtime must contain both model
stacks, their fixed dependencies, cached vision weights and matching EGL libraries.

## Explicit collection and refit campaigns

A generation JSON contains `name`, `scenes`, `behaviors`, `regimes`, and `epsilon`.
`scenes` lists distinct audited training seeds; `behaviors` assigns one of `fqi`,
`periodic`, or `active` to each scene. Each scene is collected under every listed
regime (`zero`, `nominal`, or `family`). Epsilon applies only to FQI. The full
definition and behavior checkpoint identity enter the cohort binding. No machine
name, card count or device assignment belongs in this definition.

For an explicit cumulative refit, repeat `online-train --online-cohort PATH` for
each additional complete online cohort. Preserve `--cohort-dir` and `--checkpoint`
as the original online cohort and its collecting policy; pass the current policy
separately as `--warm-start-checkpoint`. Cohorts are admitted from this exact list,
never discovered by scanning output directories. Duplicate episode identities,
evaluation data, changed assets and unreviewed collector hashes are rejected.
The objective stays half offline and half cumulative online **transitions**;
it does not give half the samples to each source or add recency weighting.

`--optimizer-updates N` supplies an exact synchronous update budget. It shuffles
full replay passes and truncates the final pass when necessary; transition weights
still use the entire replay. Without this option each FQI iteration uses two full
passes. Receipts record episode visits, transition visits and actual updates.
`--skip-initial-validation` suppresses only the initial checkpoint's validation;
a saved fitted checkpoint with interrupted validation is validated on resume
before any further optimization.

The `campaign` recipe is a JSON object with these required fields:

| Field | Meaning |
|---|---|
| `initial_checkpoint` | Scheduler starting both the candidate lineage and incumbent selection. |
| `behavior_checkpoint` | Scheduler that collected the first existing online cohort. |
| `offline_dir`, `online_dirs` | One starter cohort and an explicit ordered list of existing online cohorts. |
| `compatibility` | Reviewed configuration/collector-code compatibility JSON for these existing cohorts. |
| `validation_bootstrap` | Existing bootstrap directory for the fixed validation scenes. |
| `initial_validation` | Complete `rollout-summary.json` for the initial checkpoint, with its sibling `cohort.json`. |
| `cycles` | Ordered list of `{ "generation": <definition or null>, "optimizer_updates": <positive integer or null> }`. |
| `global_batch`, `seed` | Scientific batch size and initial fit seed; cycle index increments the fit seed. |
| `drop_successes` | Delayed-mode ball-count regression triggering one full nine-episode repeat. |

Paths resolve against the repository root. A null generation makes a fixed-data
control and requires an explicit update budget. A non-null generation collects
before refitting; its null update budget means two full passes. All proposed
generations are checked for overlapping scenes and episode identities before collection.
Each cycle refreshes targets once, fits once, and evaluates the same nine greedy
episodes. It does not repeat initial validation or admit validation trajectories.
Every new collection uses the preceding candidate; independent `best` selection
uses mean nominal/family reward and retains the incumbent on ties. If both the
candidate and its repeat reach the configured loss relative to the initial
incumbent, subsequent cycles are not admitted. The threshold counts balls in the
fixed six delayed episodes; it is a development stopping rule, not a significance test.

Launch resources remain command arguments: `--gpus`, `--train-gpus`,
`--worker-python`, `--cpu-threads`, `--replay-cache-dir`, and
`--replay-prepare-workers`. One physical GPU is valid for each GPU list. Campaign
fits use mmap replay. The immutable recipe contains no server-specific scheduling.
Resume with the same recipe and output directory. `campaign-state.json` retains
phase attempts, wall times and artifact hashes; completed phases are reused only
after verification. Interrupted collection reuses completed episodes, and an
interrupted validation reuses its complete fit. The rollout topology must remain
unchanged when resuming a partial collection or validation phase. Bootstrap and
rollout cancellation terminate only children owned by this command.

`report.json` links the candidate lineage, separate three-delay rewards,
regression repeats and fit metrics. Each cycle also keeps its collection/validation
video index and per-episode summaries, module calls/costs, replay provenance and
training receipts. Report complete cycle wall time alongside model-compute time,
environment interactions and optimizer updates; host parallelism alone does not
establish better learning or lower per-rollout cost.

Changing machines does not make old runs portable by rewriting their receipts.
Model deployment verifies content hashes and accepts an explicitly supplied new
checkpoint location; create new collection outputs there. Existing training-resume
bindings can contain original source paths and must remain intact. Record hardware
and execution metadata alongside throughput measurements. Compare serial and
parallel throughput on the same hardware and cohort; keep their per-rollout model
compute times separate from simulator/video/storage wall time.
