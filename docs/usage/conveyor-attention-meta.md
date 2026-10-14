# Conveyor attention Meta

The v2 scheduler uses explicit causal history and temporal attention instead of
recurrent hidden state. Select
[`meta_attention_v2.yaml`](../../configs/experiments/conveyor_sort/meta_attention_v2.yaml).
The [v1 workflow](conveyor-meta.md) remains available for existing checkpoints.
These architectures have different checkpoint and action schemas.

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
  "startup": {"fitted_iterations": 40, "optimizer_updates": 500},
  "global_batch": 64,
  "seed": 7,
  "reward_regression_policy": "report_only",
  "cycles": [
    {
      "generation": {
        "name": "online-g001", "trial": 0,
        "scenes": [10000, 10001, 10002, 10003, 10004, 10005, 10006, 10007],
        "behaviors": ["fqi", "fqi", "fqi", "fqi", "fqi", "fqi", "periodic", "active"],
        "regimes": ["zero", "nominal", "family"], "epsilon": 0.2
      },
      "fitted_iterations": 2, "optimizer_updates": 1000
    }
  ]
}
```

These numbers are configuration examples, not an automatic launch or a convergence
claim. `initial_checkpoint: null` initializes the new architecture; old GRU weights
are not transplanted. An external attention checkpoint must match the full model
and deployment binding. A zero startup iteration budget is useful for an
engineering pilot. A null generation runs a fixed-data continuation.

The startup incumbent is the best validated startup checkpoint. The first online
collection uses it; subsequent collections use the latest fully validated policy.
Every iteration has nine greedy validation episodes: three fixed validation seeds
under zero, nominal and family delay. Initial startup validation adds nine more.
Best maximizes the integer delivered-ball count over nominal and family, retaining
the incumbent on ties. Zero-delay results remain a separate curve. `report_only`
records reward regressions without introducing extra evaluation or changing the
budget; technical or integrity failures still stop advancement.

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
Runtime device changes are allowed at completed learner iteration boundaries;
existing rollout shard layouts remain fixed. Do not edit a running source snapshot.

Per-episode artifacts include video, action/decision/event traces, object outcomes,
module invocation counts and total/mean/p50/p95 timings. `model_pipeline_seconds`
is the sum of the explicitly listed model components; simulator, video, archival,
process startup and conversion are separate overheads. Compare complete cycle
wall time at equal scientific budgets. A small batch can be slower on more GPUs.
An engineering pilot validates integrity and recovery, not learning efficacy or
statistical superiority on a three-scene development cohort.
