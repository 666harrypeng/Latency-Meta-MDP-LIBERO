# Budget FQI: scale-v2 continuation

Historical execution record. The campaign stopped after completed C10 and its
selected artifacts passed maintenance-backup verification. Do not use C10 or
success-only S05 to initialize the next budget experiment. See the
[scratch-budget design](../superpowers/specs/2026-10-25-budget-fqi-scratch-design.md)
for the corrected initialization and scheduling protocol. The portable resource
and throughput work below remains reusable; the new protocol is not yet deployed.
The backed-up Vast instance has since been destroyed at the user's request.
Observe/Predict binding is [under renewed review](../superpowers/specs/2026-10-25-observe-predict-decision.md).

This is an explicit training-recipe revision after the completed Vast budget
cycle 1. It preserves the old results and continues from its iteration 2
checkpoint, including Adam and sampler RNG. It does not repeat startup or run
the cancelled 50/50-versus-90/10 probe.

## Scientific recipe

| Setting | Completed cycle 1 | Initial scale-v2 execution |
| --- | --- | --- |
| New trajectories per cycle | 24 | 24 |
| Global batch | 64 | 256 |
| Actual microbatch per GPU | 64 | 128 on this A100 |
| Accumulation, one GPU | 1 | 2 |
| Optimizer updates per frozen target | 500 | 1,000 |
| Draws per frozen target | 32,000 | 256,000 |
| Target refreshes per cycle | 2 | 2 |
| Draws / updates per cycle | 64,000 / 1,000 | 512,000 / 2,000 |
| Validation | 9 episodes at cycle end | unchanged |
| Learning rate / gradient clipping | 1e-4 / 0.5 | unchanged |
| Memory, reward, exploration, seeds | bound configuration | unchanged |

The microbatch rows above preserve the initial execution record. The later
compiled learner uses `--microbatch full`: 256 contexts and one backward pass
per optimizer update on the current single GPU. The scientific global batch
and update budget remain unchanged. See
[learner performance](fqi-learner-performance.md) for subsequent measurements
and the current execution controls.

Two fresh frozen-target regressions follow each completed collection. Increasing
regression work does not increase the number of Bellman refreshes. The base YAML
continues to bind deployment, reward and historical data compatibility; the
campaign JSON and checkpoint recipes explicitly bind the larger batch/update
budget and per-cycle sampling override.

The planned curriculum is

`online(c) = 0.10 + 0.40 * expm1(1.5 * (c-1)/99) / expm1(1.5)`.

The absolute cycle index runs from 1 to 100. Approximate online shares at cycles
1, 25, 50, 75 and 100 are 10%, 15%, 23%, 34% and 50%. A bounded invocation does
not compress the curriculum. The first execution window contains cycles 2-4;
the 100-cycle recipe is not an unattended run authorization.

Online sampling initially assigns half its probability to the latest cohort and
half to older online decisions. If only latest data exists, it receives the
online allocation. Latest mass cannot exceed 25% of all draws; each online group's
per-decision probability is capped at four times startup. Excess latest mass
goes to older online data when feasible. If the whole online pool is insufficient,
the effective online share falls below the target. Both shares, each group's
draws, distinct decisions, coverage and repeat rate are recorded under
`fqi/sampling/*`. Old online decisions are sampled uniformly within their group.

All 448 startup episodes remain. Existing bounded retention keeps 20 recent
online batches plus 96 historical episodes, subject to storage limits. On this
500 GB node, the continuation uses a 200 GiB replay-cache ceiling, 120 GiB online
payload ceiling, 160 GiB campaign ceiling and 64 GiB shared free-space reserve.
These are capacity guards, not claims that the node has a separate user quota.

## Measured A100 batch profile

The completed cycle 1 checkpoint and 472-episode replay were used for 12 updates
per setting. Reported update throughput excludes the first two updates; peak
memory is PyTorch allocated memory, not total device usage. Profile weights were
discarded and never entered collection. Target preparation is excluded from the
throughput below and reported separately in the raw JSON.

| Global / microbatch | Decisions/s | Peak allocated GiB |
| --- | ---: | ---: |
| 64 / 64 | 578 | 3.42 |
| 128 / 128 | 618 | 6.75 |
| 256 / 256 | 481 | 13.41 |
| 512 / 512 | 291 | 26.73 |
| **256 / 128** | **617** | **6.77** |
| 512 / 128 | 617 | 6.77 |

The selected 256/128 setting increases the training data budget eightfold per
target refresh and doubles actual per-forward batch size. It is not an eightfold
throughput gain. On this implementation and node, larger direct microbatches
were slower despite fitting in VRAM. Re-profile on another node; do not scale
learning rate automatically with batch or GPU count.

The measured update duration for 256/128 was 0.415 seconds: 1,000 updates imply
about 6.9 minutes of regression per FQI, before target construction and other
overhead. A full cycle also includes two target builds, 24 real-VLA rollouts and
nine validations. No end-to-end duration is claimed before observing it.

Raw evidence:
`outputs/operations/fqi-budget-vast-20261024/scale-v2/{batch-profile,accumulation-profile}.json`.
The batch sweep shared a progressively warmed replay; its target timings must
not be used as a controlled comparison of cold preparation costs.

## First completed collection comparison

| Cohort | Workers | Completed episodes | Rollout stage minutes | Episodes/hour |
| --- | ---: | ---: | ---: | ---: |
| Historical cycle 1 | 2 | 24 | 40.73 | 35.36 |
| Scale-v2 cycle 2 | 4 | 24 | 23.21 | 62.04 |

Cycle 2 completed with six episodes per worker and four zero exit codes. Merged
episode identities, scenes, regimes and random seeds matched all 24 manifest
entries exactly, with no duplicate or missing episode. Worker completion times
were 1,392.72, 1,342.84, 1,277.09 and 1,360.92 seconds. First-to-last completion
spread was 115.63 seconds (1.93 minutes); aggregate worker-slot idle time after
individual completion was 3.54% of the four-worker stage capacity. This does not
currently justify introducing a new dynamic dispatch mechanism.

These stage times include worker startup and exclude the preceding new-scene
bootstrap. The cohorts used different scenes and policies, and cycle 2 had the
initial test interference described below. Report the 1.75x observed throughput
ratio as an operational comparison, not an isolated causal scaling result.
Evidence: `scale-v2/acceptance.json`. Collection completion does not establish
the learning effect of the enlarged regression budget; new fits and validation
must be checked separately.

## First completed enlarged regression

Cycle 2, FQI iteration 1 finished 1,000 fresh optimizer updates with global batch
256. All 256,000 draws were processed, covering 184,737 of 371,641 available
decisions (49.71%). The target hash remained unchanged during regression.
Target construction took 348.37 seconds; regression took 418.39 seconds, or
611.86 decisions/s. The full FQI call took 810.94 seconds including overhead.

| Source | Available decisions | Actual draws | Distinct decisions | Coverage | Draws per distinct decision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Startup | 336,607 | 230,103 | 166,555 | 49.48% | 1.382 |
| Older online | 21,208 | 12,851 | 9,722 | 45.84% | 1.322 |
| Latest online | 13,826 | 13,046 | 8,460 | 61.19% | 1.542 |

Requested and effective online probability both equal 10.175%; the realized
draw fraction is 10.116%, as expected from random sampling rather than fixed
per-batch source counts. The completed checkpoint state SHA is
`c96bac4dddb0c9426ab520e378662c0c821a9d0e4ddaa1dac47fb02484481662`.
Evidence: `scale-v2/fit1-verified.json`. The initial continuation checkpoint also
matched the parent's model hash, Adam moments, sampler RNG and 64,000 prior
visits exactly; see `scale-v2/initial-resume-proof.json`.

Training-time mean TD losses for startup/older/latest G-stage samples were
0.288/6.956/4.355; mean signed TD errors were -0.288/-9.115/-5.598. These are
aggregate diagnostics over the changing model during regression, not a fresh
post-training validation or evidence of the original campaign's causal failure.
Gradient clipping applied on 96.1% of updates. Learning rate and clipping remain
unchanged in this bounded run; assess their adequacy with the pending fixed
validation rather than infer improvement from fitted loss.

Resource polling during this regression averaged 49.1% GPU utilization across
90 samples, with a 9.94 GiB peak device footprint. The implementation is not
claimed to saturate the learner or exhaust optimization opportunities. The
batch profiles support microbatch 128 among tested settings; the completed
larger update budget is a separate result from hardware saturation.

As of 2026-10-25 04:16 UTC, cycle 2 iteration 2 is building fresh targets. The
first fixed validation follows iteration 2; cycles 3-4 then continue under the
same controller. Their delivery and complete-module-cost results are pending.
A provisional whole-cycle budget is about 70-85 minutes, combining the measured
collection, preparation and FQI stages with the historical 21.6-minute
validation. This is an estimate, not a measured new full-cycle duration;
changed policy behavior and cache state can alter it.

## Portable resources and launch

Use the existing `campaign` command. Collection and validation share GPU IDs,
workers per GPU and CPU threads by default. Validation overrides are optional:
`--validation-gpus`, `--validation-workers-per-gpu`, `--validation-cpu-threads`.
The learner uses independent `--train-gpus`, `--learner-cpu-threads` and
`--microbatch`. Global batch and episode counts come from the scientific recipe.
Global batch must be divisible by learner GPU count times the resolved numeric
microbatch. `full` selects the whole per-rank batch; `auto` probes memory capacity.

On the current A100, the defaults are four rollout workers, five CPU threads per
worker, one learner with `MICROBATCH=full`, eight learner CPU threads and four
replay preparation workers. The cgroup quota is 23.04 CPU cores. Four complete
rollout replicas have already passed validation; resident memory was about
17.1 GiB each. Five such replicas would exceed this card's capacity even before
startup peaks. Other nodes must measure their own footprint and completed
trajectory throughput.

```bash
# Run from the project root; credentials must already be in the environment.
# Keep the resulting process in the existing conveyor-budget:main tmux window.
ROLLOUT_GPUS=0 TRAIN_GPUS=0 \
ROLLOUT_WORKERS_PER_GPU=4 ROLLOUT_CPU_THREADS=5 \
LEARNER_CPU_THREADS=8 MICROBATCH=full COMPILE_LEARNER=1 STOP_AFTER_CYCLES=3 \
bash outputs/operations/fqi-budget-vast-20261024/launch-scale-v2.sh
```

The launcher also accepts `VALIDATION_GPUS`, `VALIDATION_WORKERS_PER_GPU`,
`VALIDATION_CPU_THREADS`, `REPLAY_WORKERS` and `EPISODE_CACHE_SIZE`. Set the episode
cache large enough for the active replay when host RAM permits; a too-small LRU
can repeatedly reconstruct histories. This node uses capacity 1,024, allocating
entries only on demand. Do not alter shard count midway through an incomplete
collection; a subsequent complete cohort can use a different allocation.

The wrapper resumes W&B run `a04b647b` and its historical source-prefix digest.
This launcher is specific to the existing Vast campaign, including its paths,
environment, and audited source-change receipt. It is not a standalone new-node
installation or launch package. New campaigns use the tracked CLI with their
own configuration and output identity; do not copy this campaign's compatibility
receipt into an unrelated run.
New fits append metrics; existing records and identities are not rewritten.
`resource-config.json`, `resource-history.jsonl`, per-cycle recipes, checkpoint
receipts, and the scale-v2 source-change receipt record effective execution.
The central console prints worker count, completed episodes, elapsed stage time,
trajectory throughput, target progress and optimizer progress. `hardware.csv`
records GPU memory/utilization and container CPU consumption as diagnostics;
they do not enter reward.

## Acceptance and interpretation

- Confirm the 24 expected episode IDs appear exactly once in the completed
  merged collection, with fixed manifest identities/seeds and successful workers.
- Compare the first four-worker cohort against the historical two-worker
  collection's 40.7 minutes. Different checkpoints/scenes make this an operational
  comparison, not an isolated worker-count experiment.
- Use `worker_finished_seconds` to quantify tail waiting before changing static
  sharding. Shared VLA serving is deferred until measurements justify it.
- Inspect per-source actual coverage, TD/CQL diagnostics, target freezing,
  checkpoint continuity, and complete validation delivery and module costs.
- GPU saturation and fitted loss are not policy-quality evidence. No budget or
  success improvement is claimed from these throughput profiles.

## Deployment evidence

The targeted local suite passed 119 tests; the remote subset passed 26 tests
with bounded CPU threads. Ruff and whitespace checks passed before deployment.
Source replacement happened while the training controller was idle. The receipt
at `scale-v2/source-change.json` records the explicit scientific revision and the
unchanged starting checkpoint; this was not a compatibility-patch bypass.

The existing W&B history prefix (1,027 records) was verified before resuming.
The cloud API subsequently confirmed new cycle-2 records in the same running
run. Profiles used temporary model instances, and their weights were discarded.

The first four-worker collection overlapped a leftover test process until
2026-10-25 03:42:42 UTC. It had survived a requested termination, then exited
before the cleanup signal was sent. No rollout was restarted. Preserve the
full stage duration, mark this interference, and use subsequent cohorts for
uncontended whole-stage timing. The operations-only `scale-v2/acceptance.py`
summarizes manifests, completed checkpoints, validation and resource samples
without rereading large payloads. Its JSON distinguishes pending results from
completed identity checks and reports resource samples after the test exited.

### Subsequent transport and telemetry maintenance

See [learner performance](fqi-learner-performance.md) for the lossless input
transport repair and native W&B system monitoring. These are operational changes
to scale-v2; its reward, sampling curriculum, update budgets, microbatch, and
checkpoint lineage remain unchanged. Console worker counts now distinguish active
workers from configured slots.
