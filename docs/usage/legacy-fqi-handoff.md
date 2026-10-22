# Legacy LIBERO / FQI handoff

This repository preserves the LIBERO conveyor research and its original
experiments. The next Isaac Lab implementation is maintained in a separate
repository. Legacy cleanup A1-B4 was accepted on 2026-10-22; the dated
verification record below defines its evidence and limits. New runtime work
starts in the Isaac directory/session.
New design decisions do not retroactively change the old reward, observations,
action masks, latency conditions, or result interpretation.

## Read in this order

1. [Attention architecture and FQI protocol](conveyor-attention-meta.md).
2. [Conveyor task and source data](conveyor.md), then the
   [v1 scheduler](conveyor-meta.md) when working with older checkpoints.
3. The local, dated cleanup package:
   [`HANDOFF.md`](../../outputs/analysis/conveyor-sort/legacy-cleanup-20261022/HANDOFF.md),
   [`RUNBOOK.md`](../../outputs/analysis/conveyor-sort/legacy-cleanup-20261022/RUNBOOK.md),
   [`MANIFEST.json`](../../outputs/analysis/conveyor-sort/legacy-cleanup-20261022/MANIFEST.json),
   and [`verification.json`](../../outputs/analysis/conveyor-sort/legacy-cleanup-20261022/verification.json).
4. [Actor campaign documentation](conveyor-actor-training.md) only for that
   separate historical route; it is not the Ideas2 FQI launcher.

The linked `outputs/` files are deliberately not tracked. They exist in the
operator checkout and verified preservation copy, not in a fresh Git clone.
If absent, retrieve the preservation package before claiming recovery readiness.
Do not reconstruct provenance from a checkpoint filename or a W&B status alone.

## Code and execution identities

| Tree | Identity at the 2026-10-22 cleanup | Role |
| --- | --- | --- |
| Local maintenance checkout | `conveyor-sort`, `3ec126b326ad7f0f83e8cd57a1b3486220f37895`, with seven pre-existing repair files | Newer code; reviewed and tested independently |
| Ideas2 execution checkout | `4bc5401e1f3b460bd173a4d640d4f235e0b43ea2`, with six audited operational Python changes | Source tree bound to the preserved formal FQI campaign |
| New Isaac repository | `latency_meta_control_isaac` | Accepted design and staged implementation plan; no inherited campaign |

The seven local files are `attention_campaign.py`, `attention_training.py`,
`monitor.py`, `preflight.py`, `resources.py`, `runtime_identity.py`, and
`tests/test_information_runtime_repair.py`. The Python modules are under
`src/latency_meta_mdp/meta/information/`. They preserve an explicitly audited
source identity across resource-admission repairs; they do not permit arbitrary
scientific changes. The cleanup keeps their existing behavior after review. Git delivery records
these repairs and their regression tests in commit
`e0ad804fcfbbbc2cd8e6bf72526857a3aec5fff9`. Committing the local maintenance source does not
change the separately bound remote execution source.

Ideas2 retains compatibility identity
`da1406518cc164b5fc4c4eb14fdec8a51bb44889e32181460611b6d0741791f6`
and execution identity
`6d39bcb08aa29b15d475ac01cfc85dffbd38735c2401aa4280db7e9170ea6f21`.
The exact v3 receipt SHA is
`8fe5e0246058d53981ae46f222d1ca17a9d93d15cf6a16b64f30c0ad31c7f3ae`.
Its root and complete Python-file map are binding. Preserve the receipt; never
regenerate it to admit a refactor. The local archive includes the exact remote
execution Python tree as well as the newer local baseline and dirty bytes.

## What the existing experiment establishes

The scheduler has eight 256-dimensional queries, two history reads, and one
gated forecast read. It schedules a frozen visual encoder, JEPA, decoder, and
conditioned VLA. The preserved formal contract binds Observe to Predict and
uses guarded starvation semantics. It is success-only, off-policy FQI with
startup replay retained throughout online collection; it is not PPO or purely
online learning. Each delivered object receives 0.1 reward and cost lambda is 0.

The recipe uses 448 startup episodes (128 old offline, 32 old online, 288 grid),
40 startup iterations of 500 updates, then up to 100 online cycles with 24
episodes and two iterations of 1,000 updates each. Global batch is 64. Online
starts from startup best5, then continues from the previous cycle's latest.
Best selection is separate and uses nominal/family validation. Actual total
work and checkpoint ancestor updates must be reported separately.

The Oct21 evidence package records 74 complete online cycles and 168,000 total
updates, with C75 collection partially complete. The Oct22 read-only audit
confirmed latest C74 and best C03. A stale report saying `running`, or a live
telemetry sidecar, does not mean the training controller is running. Consult the
dated handoff for the latest observation and its timestamp.

The fixed development validation uses three scenes per delay and ten objects
per scene. Success means delivered objects / spawned objects. Selection across
many checkpoints on this small development set is not an independent test.
Historical E25 baselines were not rerun under the final runtime. Earlier gains
did not persist through training; do not splice C03's family score and C35's
zero score into a single policy. Module request counts and calibrated model
costs are descriptive diagnostics, not proof that lambda=0 learned a cost
objective or that physical energy was saved.

## Evidence and preservation

The local evidence index is
[`outputs/analysis/conveyor-sort/README.md`](../../outputs/analysis/conveyor-sort/README.md).
The Oct21 status, success comparison, and module-request packages contain source
snapshots, CSVs, builders, figures, and provenance. Rebuild in a mirrored copy:
the builders write beside their own scripts. Never overwrite the original
snapshot to refresh a figure. Historical evidence may retain its original
language; new tracked first-party text is English.

The cleanup manifest records a verified second copy outside the repository on
the operator host. This is same-host preservation, not an off-site backup;
machine-specific locations are retained in that package.
Large datasets and checkpoints remain at their bound locations with validated
references. They have not all been duplicated into the archive or Git.

## Recovery boundaries

- Keep the canonical remote root recorded in the pinned runtime receipt and
  formal output
  `outputs/training/conveyor-attention-bounded-100-ideas2` in place. Absolute
  checkpoint parents, data paths, receipts, and journal artifacts are binding.
- Check the whole execution tree, receipt, raw inputs, bootstrap, dependencies,
  full checkpoint, and storage. A filesystem `df` check is not a personal-quota
  check. Retain `harnesses/check_ideas2_storage.py` and its 32 GiB reserve.
- The host launcher is `launch_ideas2_formal.sh` under the acceptance package's
  `harnesses/`. It selects physical GPUs 4–7 and explicitly permits other-user
  occupancy while preserving writer locks. Never signal other users' processes.
- Reuse all completed startup, collection, fitting, and validation phases.
  Do not silently warm up again, replace old identities, or delete partial work.
- Keep main W&B run `cd0d98f6`; stage runs retain their own recovery identities.
  The sidecar is a separate writer and must not be duplicated.
- Full CPU checkpoint loading and exact synthetic continuation tests establish
  recovery preparation. They do not prove a new production GPU update or a
  complete simulator rollout. A formal restart needs its own authorization and
  observed new episodes/updates.
- Cleanup does not restart Ideas2 or Delta, renew monitoring, rent GPUs, or
  authorize deletion. The old repository remains available for future work.

## Dated closure acceptance

The read-only audit on **2026-10-22, 22:57-23:06 UTC** found no formal training
controller. The existing W&B summary sidecar remained. Personal quota was
568.0108/600 GiB, leaving 31.9892 GiB, below the original 32 GiB reserve.
At that point, 74 online cycles and 168,000 actual optimizer updates were
complete; latest C74 carried 150,500 ancestor updates. C75 collection had 22/24
complete episodes and no fitting.
That snapshot was not permission or sufficient storage to restart. Inspect the
live state again before any later operation; these numbers are historical.

| Required check | Result at closure |
| --- | --- |
| Local maintenance repairs | Seven files reviewed; no further correction needed; original content preserved |
| Actual execution source | All 489 Python files matched the existing audited source receipt |
| Actual latest/best checkpoints | C74 and C03 passed complete CPU load, strict model/Adam restoration, finite tensor, sampler/RNG and receipt/hash checks |
| Selected raw replay and frozen assets | 904 selected episodes plus 22 complete partial-C75 episodes; 1,018,370 files / 309,820,216,658 bytes read and hashed without errors |
| Mechanism verification | Main suite: 39 passed, 2 CUDA skips; independent CPU suite: 85 passed; suites overlap and are not additive; scoped Ruff passed |
| Historical results | Five PNGs reproduced byte- and pixel-identically; numerical tables matched; all 87 original evidence files unchanged |
| Independent handoff | Fresh-context review passed |

CUDA rollout equivalence, large-checkpoint production continuation, current cloud
W&B synchronization and full replay-cache hashing were not verified by closure.
The optional large-checkpoint continuation was explicitly not run. CPU loading
and fixture tests do not establish a successful production restart.

## Recovering a preserved campaign

A fresh clone contains the maintenance implementation and this guide. It does
not contain the old campaign, its private receipt/environment, machine-specific
launchers, weights, replay or credentials. Obtain the preserved campaign package
before trying to resume it. Do not regenerate those records from this guide or
substitute the newest Git checkout for the bound execution tree.

The following paths are relative to the original bound project root:

| Dependency | Preserved location |
| --- | --- |
| Formal output | `outputs/training/conveyor-attention-bounded-100-ideas2` |
| Runtime receipt | `outputs/operations/runtime-admission-repair-20261016/source-compatibility-v3.json` |
| Private runtime selector | Formal output's `runtime-patch.env` |
| Campaign launcher and quota guard | `outputs/analysis/conveyor-sort/meta-attention-acceptance-v2/harnesses/` |
| Worker interpreter | `.venv-policy-eval/bin/python` (audited Python 3.11.15, Torch 2.7.1+cu126) |

The receipt pins the original absolute root; the private runbook/manifest records
the cache location, account quota query and environment. Do not relocate them.
Load `CONVEYOR_RUNTIME_PATCH` and its pinned `CONVEYOR_RUNTIME_PATCH_SHA256`
through the preserved launcher/environment. Never replace the receipt to admit
new source code. `CONVEYOR_GPU_SHARING=allow` relaxes other-user GPU occupancy
admission only; writer locks, actual memory capacity and storage checks remain.

Before a separately authorized restart:

1. Confirm an idle authenticated shell and inspect existing controllers, tmux,
   sidecars and writer locks. Reuse the original run; do not start duplicates.
2. Verify the complete bound source tree, receipt/config/recipe SHA, bootstrap,
   frozen models, selected replay and retirement/partial-collection records.
   Check personal quota separately from filesystem capacity and actually read
   dependencies and checkpoint bytes. Keep the 32 GiB free reserve and nofile
   limit of at least 8192.
3. Validate the complete checkpoint against its receipt and load it on CPU.
   Preserve model, Adam, sampler/RNG, update position, fixed target and parent
   lineage. A periodic snapshot needs its in-progress target; a complete outer
   iteration uses its model for the next target. Do not refresh mid-iteration.
4. Set `BOUND_PROJECT_ROOT` to the original absolute root from the receipt. In a
   persistent, visible terminal, invoke the **preserved** launcher below only
   after preflight succeeds. This is a recovery entry, not a fresh-clone command.

   ```bash
   cd "$BOUND_PROJECT_ROOT"
   META_GPUS=4,5,6,7 bash outputs/analysis/conveyor-sort/meta-attention-acceptance-v2/harnesses/launch_ideas2_formal.sh
   ```

5. At the audited snapshot, reuse startup 40, cycles 1-74, latest C74/iteration 2,
   and the 22 valid C75 collection episodes. Continue the missing collection,
   then fitting; do not repeat warm-up or restart from selected best C03.
6. Require a genuinely new complete episode, new finite optimizer updates and
   a complete reloadable checkpoint. Check fixed targets, rank hashes, journal
   reuse and original W&B identities. Process liveness alone is insufficient.

For local maintenance verification, run from the repository root in the verified
legacy environment (Python 3.10.20, Torch 2.7.1+cu126 at closure):

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' .venv-belief/bin/python -m pytest -q \
  tests/test_information_runtime_repair.py \
  tests/test_information_preflight.py \
  tests/test_information_monitor_source.py \
  tests/test_information_online_resources.py \
  tests/test_information_campaign.py \
  tests/test_attention_campaign.py \
  tests/test_attention_training.py
```

The runtime identity module and its regression tests are maintained source, not
scratch files. Git records the local maintenance version; it does not replace
the separately preserved source identity of the original formal campaign.
