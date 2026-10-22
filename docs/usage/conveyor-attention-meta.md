# Conveyor attention Meta

The v2 scheduler uses explicit causal history and temporal attention instead of
recurrent hidden state. Select
[`meta_attention_v2.yaml`](../../configs/experiments/conveyor_sort/meta_attention_v2.yaml).
The [v1 workflow](conveyor-meta.md) remains available for existing checkpoints.
These architectures have different checkpoint and action schemas.

This page describes the implemented LIBERO attention/FQI contract. For the
preserved Ideas2 execution version, results, and resume prerequisites, start with
the [legacy handoff](legacy-fqi-handoff.md). The newer local maintenance checkout
is not the source tree bound to that campaign. Do not copy it over the remote
run. The separate Isaac/PPO design has different contracts and is not implemented
by changing this campaign's configuration.

## Inputs and decisions

The model has eight 256D decision queries, four attention heads, two history
reading blocks and one gated forecast block. Both history blocks can read the
complete selected input bank. The readout concatenates all queries and a direct
projection of current critical state; it does not average the queries.

Public inputs include current proprioception, scheduler state, the known action
buffer, executed controls and resulting proprioception, paid visual observations,
and request/arrival/decision events. Selection retains up to 16 visual times,
64 executed controls and 32 scheduler events. The newest three visual frames
retain patch resolution; older frames use 16 learned spatial tokens per view.
Physical simulation time determines RoPE positions. Availability order separately
prevents future information from entering a decision. Simulator outcome metadata
is used only for reward and auditing.

JEPA forecasts remain a separate input bank, with target time and source metadata.
A forecast must be fresh for the current tick and buffer version. The history
reader conditions its gated forecast query. LayerNorm and trainable projections
are learned with the scheduler; the visual encoder, JEPA, decoder and VLA stay
frozen. Missing legacy fields are explicitly masked, never fabricated.

Observe is followed by forced Predict. The free choices are Sleep at
`[1, 2, 3, 4, 6, 8, 12, 16]` ticks, Observe, and Launch at the appropriate stage.
One tick is 20 ms. External masks enforce legality. `starvation_mode: guarded`
retains a 20-tick reserve. `allow_hold` is a distinct protocol: exhausted buffers
execute zero arm deltas and retain the gripper command. Its declared hold prefix
is shared by JEPA and RTC; arbitrary padding does not become valid. Qualify a
real hold/recovery rollout before using that mode in a research campaign. Legacy
guarded data is not automatically admitted into an allow-hold campaign.

## Learning and replay

Each delivered ball contributes 0.1 reward. Gamma is one and cost lambda is zero.
The conservative legal-action penalty is 0.01. Compute is measured, but the
objective does not reward compute savings.

Each FQI iteration clones a fixed target model, draws an explicit decision-context
sample stream, computes targets for its required next contexts, then fits with
persistent Adam. Forced stages are folded once into adjacent free decisions.
The target is refreshed between fitted iterations; it is not refreshed at every
optimizer step. Evaluation episodes never enter replay.

The sampler selects disjoint source groups with probability 0.50 startup,
0.25 older online and 0.25 latest online. When older online is absent its mass goes
to latest; startup alone receives all mass if no online data exists. Within each
group, valid decisions are sampled uniformly. Episode counts are not transition
weights. Replay grows only from explicitly named complete, hash-verified cohorts.

V1 sources preserve their original files and semantic action mapping. The v2
conversion records source receipts, raw-cache format, converter/selector hashes,
capabilities and decision counts. A legacy source missing execution history or a
three-frame bundle is usable only with those fields masked. Frozen features are
stored in deduplicated read-only mmap arrays; learned features and targets are
recomputed. Account separately for one-time conversion, host page cache and
per-worker Python context metadata.

## Bounded campaign recipe

`campaign --campaign-recipe recipe.json` accepts the following schema. Paths are
relative to the repository root. Obtain source hashes from existing verified
`cohort.json` files; the illustrative hash below must be replaced.

```json
{
  "schema_version": 2,
  "startup_sources": [
    {"path": "outputs/my-startup", "manifest_sha256": "<sha256>", "group": "startup"}
  ],
  "initial_checkpoint": null,
  "validation_bootstrap": "outputs/my-validation/bootstrap",
  "startup": {"fitted_iterations": 40, "optimizer_updates": 500, "validation_every": 5},
  "global_batch": 64,
  "seed": 7,
  "reward_regression_policy": "report_only",
  "retention": {
    "recent_batches": 20,
    "history_episodes": 96,
    "max_online_bytes": 128849018880,
    "max_cache_bytes": 240518168576,
    "max_run_bytes": 386547056640,
    "min_free_bytes": 34359738368,
    "episode_cache_size": 1024
  },
  "cycles": [
    {
      "generation": {
        "name": "online-g001", "trial": 0,
        "scenes": [10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007],
        "behaviors": ["fqi", "fqi", "fqi", "fqi", "fqi", "fqi", "periodic", "active"],
        "regimes": ["zero", "nominal", "family"], "epsilon": 0.2
      },
      "fitted_iterations": 2, "optimizer_updates": 1000, "validation_every": 2
    }
  ]
}
```

These numbers are configuration examples, not an automatic launch or a convergence
claim. `initial_checkpoint: null` initializes the new architecture; old GRU weights
are not transplanted. An external attention checkpoint must match the full model
and deployment binding. A zero startup iteration budget is useful for an
engineering pilot. Without `retention`, a null generation runs a fixed-data
continuation. A bounded replay recipe requires collection in every online cycle.

The startup incumbent is the best validated startup checkpoint. The first online
collection uses it; subsequent collections use the latest fully validated policy.
Each configured validation boundary has nine greedy episodes: three fixed seeds
under zero, nominal and family delay. `validation_every` defaults to one; the last
FQI iteration in each phase is always evaluated. The example evaluates startup
every five iterations, then once after both FQI iterations of each online cycle.
Initial startup validation adds nine more. Checkpoint saves remain independent
of validation cadence, and only evaluated checkpoints enter best-model selection.
Best maximizes the integer delivered-ball count over nominal and family, retaining
the incumbent on ties. Zero-delay results remain a separate curve. `report_only`
records reward regressions without introducing extra evaluation or changing the
budget; technical or integrity failures still stop advancement.

Repeat the cycle definition with distinct generation identities to specify the
desired number of cycles. With 100 cycles of 24 episodes, the example budgets
2,400 new online episodes, 220,000 optimizer updates including startup, and 981
validation episodes. Existing startup data and an optional separately collected
grid are reused for all startup iterations; 40 iterations do not create 40 datasets.

### Storage bounds

`retention` is optional. It never retires startup sources or pre-existing external
datasets. The active online replay contains the most recent 20 batches and a
disjoint pool of up to 96 older episodes. Seeded random priorities retain at most
32 episodes per delay regime, without selecting by success or length. A byte cap
may shorten the recent window or history pool. A latest batch too large to fit
causes a clear failure while preserving its files and the preceding checkpoint.

The example limits active online payloads to 120 GiB, derived replay cache to
224 GiB and the campaign directory to 360 GiB, with 32 GiB filesystem headroom.
Choose these independently for the actual storage allocation. The active byte
cap is not the instantaneous disk footprint: the current and previous cycle's
inputs remain protected, and collection temporarily writes a new batch. Run/free
space guards check phase boundaries; cache workers reserve bytes before writing
arrays. Startup assets outside the campaign directory still consume filesystem
quota. These checks do not reserve space against other users or external writers.

Every cycle publishes its selected episode IDs and source hashes under
`replay-state/`. Only after training and validation complete does cleanup retire
unprotected online feature arrays, source frame archives and training videos.
It preserves receipts, outcome summaries and lightweight provenance. A retirement
receipt distinguishes deliberate removal from missing or corrupted evidence and
permits the journal to resume past old phases. Old model checkpoints remain usable
for inference; exact retraining of retired datasets is no longer supported.

The shared mmap cache reuses identical packet/forecast features from observation
events without changing floating-point values. Each learner also has a bounded
LRU cache of reconstructed episode metadata (`episode_cache_size`); mapped feature
pages remain managed by the OS. Set that capacity with host RAM and learner count
in mind. Keeping the active episodes resident avoids repeated JSON/context work.
Evaluation saves videos, actions, scheduler/state/event traces, object results and
module timings, but omits unused training feature archives. Its trace format is
explicitly `evaluation_trace_v1` and cannot be admitted as training replay.

A generation can instead supply `patterns` shared across its listed scenes.
`grid_generation(name, scenes)` constructs five relative request-window positions
`rho=[0,.25,.5,.75,1]`, each with `on_request` or `every_4_ticks` refresh, plus E25
and legal random behaviors. With eight scenes and three delays this is 288
trajectories. The window begins when scheduling becomes possible after a request
and spans `max(remaining_buffer - reserve, 0)`; pending requests do not create
additional decision opportunities. Generation and trial identities permit revisits
to training scenes while preventing duplicate episodes. Keep failures as well as
successes, and retain behavior probabilities and checkpoint versions.

## Runtime and recovery

The same recipe runs with one or multiple GPUs. Select physical device IDs at
launch, without embedding hostnames or GPU lists in the experiment config:

```bash
PYTHONPATH=src python scripts/run_conveyor_meta.py campaign \
  --config configs/experiments/conveyor_sort/meta_attention_v2.yaml \
  --campaign-recipe recipe.json --scene-audit scene-audit.json \
  --output-dir outputs/my-attention-run \
  --gpus 0,1,2,3 --train-gpus 0,1,2,3 \
  --microbatch 2 --cpu-threads 4 --replay-prepare-workers 2
```

Use `--gpus 0 --train-gpus 0` for one device. Global batch comes from the recipe;
require `global_batch = learner_world_size × microbatch × accumulation` with
positive integer dimensions. Neither learning rate nor rollout count changes
with GPU count. The v2 batch unit is **decision contexts**, unlike v1's episode
loss batches. `--bf16` explicitly changes training arithmetic; FP32 is the default.
Use `--worker-python` when workers require a different pinned interpreter.

`--dry-run` produces planned episode identities and update/validation counts
without model processes. It validates recipe structure and scene membership;
full asset and source-content admission occurs at execution. Hash-pinned source
manifests and model assets must already exist for the actual run.

Run one controller in a persistent terminal. Collection workers keep frozen
services resident for their assigned episodes. The synchronous learner releases
its GPU processes before validation. On restart, retain the recipe, source
checkout and output path; the journal reuses complete verified phases, restores
optimizer/sampler state and finishes pending validation before further updates.
Learner device IDs and world size may change on restart while keeping global batch
compatible with the selected world size and microbatch; existing rollout shard
layouts remain fixed. Do not edit a running source snapshot.

For Slurm, keep the checkout, source manifests, caches and output directory on
persistent project storage, and use the same command and scientific recipe in
each allocation. Set `XDG_CACHE_HOME` explicitly; GPU locks use that cache and
include the compute hostname. A resumed rollout may use a different node, device
IDs or CPU allocation, while retaining its original worker count and episode
partition. Execution history records each allocation's host, devices and Slurm
job ID. Allocate only the devices assigned to the job.

Use `--checkpoint-every-updates 256` (the default) to periodically overwrite the
current iteration's `training-NNNN-latest.pt`. It saves model, Adam, fixed FQI target,
sampled decision order and current update position with an atomic replacement.
Restarting the same command loads this snapshot automatically; unsaved updates
may be repeated. The save interval is a runtime setting, independent of GPU count.
Completed iterations still publish `iteration-NNNN` and `latest`; recovery
snapshots do not enter model selection before the iteration and validation finish.

Complete episode receipts are reused; interrupted episodes restart. Pending
validation finishes before learning advances. After a job reaches its time limit,
submit the same command against the same persistent workspace and output directory.
There is no automatic job renewal or chaining. Count rolled-back optimization as
consumed work; resubmission does not increase the scientific update budget. A
process must release its writer lock before another controller resumes the run.

Per-episode artifacts include video, action/decision/event traces, object outcomes,
module invocation counts and total/mean/p50/p95 timings. `model_pipeline_seconds`
is the sum of the explicitly listed model components; simulator, video, archival,
process startup and conversion are separate overheads. Compare complete cycle
wall time at equal scientific budgets. A small batch can be slower on more GPUs.
An engineering pilot validates integrity and recovery, not learning efficacy or
statistical superiority on a three-scene development cohort.
