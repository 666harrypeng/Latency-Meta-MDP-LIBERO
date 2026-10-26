# FQI learner input transport and hardware telemetry

The FQI learner is one process on a single training GPU. Collection and validation
use separate VLA worker processes and finish before the learner starts. Increasing
rollout workers does not parallelize the learner. Global batch, actual microbatch,
and rollout workers remain separate controls.

The current attention scheduler contains 4,572,566 parameters. VLA inference is
frozen and belongs to rollout, not learner backpropagation. Its memory footprint
therefore should not be compared with VLA SFT. Device utilization measures GPU
activity over a sampling interval; it is not a measure of achieved FLOPS or a
requirement to occupy all VRAM.

## Diagnosis and scope

The A100 scale-v2 learner previously averaged about 49% device utilization. A
representative 128-decision input batch contains about 529 MiB of raw visual data.
CPU profiling found substantial array stacking, pinning, and repeated file-stat
work. Large free VRAM does not establish that the arithmetic kernels are the
bottleneck; compare completed decisions per second with the same global batch,
sampling, precision, and update count.

The transport revision preserves raw values, causal context selection, frozen
targets, sample order, gradient accumulation, optimizer state, and checkpoint
schema. It uses direct pinned-memory collation, two bounded preparation tasks,
and a CUDA copy stream. Episode-cache mutation remains serialized. Immutable
replay is admitted as before and checked at each FQI input pass; steady-state
minibatches do not repeat file-stat checks. Other replay consumers retain their
existing verification behavior.

The source repair receipt records the complete before/after trees and equivalence
evidence. Changes to the causal selector, architecture, reward, or optimizer are
not covered by this transport compatibility claim.

## Portability and optimization boundaries

The learner optimizations use PyTorch tensor operations, pinned host memory,
CUDA streams, and optional compilation. They do not select code by GPU model,
depend on a particular replay's sample count, or perform extra work to inflate
utilization. Scratch benchmark instrumentation under `outputs/operations/` is
not part of the training entrypoints.

| Control | Purpose | Scientific effect |
| --- | --- | --- |
| GPU lists, rollout workers, CPU threads, replay workers | Assign execution resources | Preserve episode identities, seeds and requested work |
| Microbatch `full`, `auto`, or a number | Fit each rank's batch into memory | Preserve global batch and optimizer update count |
| `--compile-learner` | Optional compiled CUDA execution | Preserve the model and objective; floating-point reductions can differ |
| Global batch, updates, collection count, target refreshes | Change training scale | Explicit recipe revision, not automatic resource tuning |

There is no automatic throughput tuner. `auto` selects memory capacity using
representative eager forward/backward probes before compilation; it does not
prove that every later input or compiled graph will fit, nor select the fastest
setting. Rollout worker count remains explicit. The two bounded input producers
are a shared default, not a GPU-specific branch; increasing them worsened the
measured A100 throughput.

A collaborator can reuse the application code and adjust these controls, but
must provide a compatible PyTorch/CUDA environment, assets, configuration and
output directory. A100 measurements do not validate B200/B300 execution or
justify copying an A100 virtual environment unchanged. Start a new node with a
short real-input forward/backward and checkpoint check, then compare completed
decisions and trajectories per second. Record compilation startup separately
from steady throughput. A higher utilization reading alone is not acceptance.

The dated Vast launcher is an existing-campaign resume command. Its W&B identity,
paths and source-compatibility receipt must not become library defaults or be
reused for an unrelated campaign. Further optimization should follow a measured
bottleneck and retain correctness checks; it must not change reward, history,
sampling or update budgets implicitly.

## Observability

The central campaign W&B writer previously disabled native system metrics.
Native monitoring is now enabled at a five-second interval. GPU utilization,
allocated-memory percentage, power and temperature, CPU, RAM, and disk appear in
W&B's **System** charts on the same campaign run. Native telemetry uses a separate
system stream and cannot advance or corrupt the training-event resume cursor.
Hardware history before this change is not fabricated; the earlier local
`hardware.csv` remains available.

Native CPU and host-memory metrics are not the container's allocation. Consult
`resource-config.json`, `resource-history.jsonl`, and the operation's cgroup CSV
for the actual container quota and usage. Console `input` measures exposed host
waiting/enqueue time, not all asynchronous copy time. Use profiler traces for
transfer/kernel overlap.

Measured A100 comparisons and deployment verification are recorded under
`outputs/operations/fqi-budget-vast-20261025/performance/`. Scratch profile weights
are discarded and never admitted to the campaign.

## A100 measurement, 2026-10-25

Both runs used the same 496-episode replay (371,641 decisions), cycle 2 iteration
2 checkpoint, sampling seed, BF16, global batch 256, microbatch 128, and eight
learner CPU threads. Each performed 32 scratch updates. The table uses updates
13-32, after the profiler's active window and initial warmup.

| Metric | Original input path | Pinned asynchronous prefetch |
| --- | ---: | ---: |
| Decisions per second | 593.6 | 848.8 |
| Mean update seconds | 0.4313 | 0.3016 |
| Exposed input wait per update | 134.9 ms | 4.8 ms |
| Sampled device utilization | 48.5% | 61.9% |
| Hardware samples in measurement window | 16 | 12 |
| Peak PyTorch allocated memory | 6.82 GiB | 9.99 GiB |

The observed regression throughput increase is 43.0%; update duration decreased
30.1%. The utilization estimates are short samples, not full-cycle averages.
The final update includes pipeline drain. Both scratch runs rebuilt cold
in-process replay contexts; do not extrapolate their target times to a complete
warm FQI target pass. A full formal cycle with the new transport is still pending.

The trace also shows CUDA copies, LayerNorm, small kernels, launches and CPU/GPU
synchronization. Removing exposed input wait does not eliminate these costs or
establish sustained device saturation. This revision does not claim that a larger
GPU alone, higher VRAM allocation, or more rollout processes would remove them.

The remote A100 passed 51 focused tests, including exact deterministic optimizer
and resume equivalence on CPU/CUDA with FP32/BF16. The broader local selection
passed 78 tests; relevant Ruff and whitespace checks passed. The source receipt
preserves the original selector and conversion identities, with an AST check
that `memory.py` changes only collation.

Formal execution resumed at cycle 3 from the complete cycle 2 checkpoint, with
3,000 prior budget-training updates and all nine cycle 2 validation episodes
retained. Its state SHA is
`eec6db09a37f11e151c5d566419b2e18ce1874706c8114935c89133e55fdccbb`.
The same W&B run `a04b647b` resumed; its cloud system stream contains new GPU
utilization, allocated-memory, power, temperature and clock samples. The console
prints active/configured rollout workers. No startup training was repeated.

## Compiled execution and microbatch selection

`run_conveyor_meta.py` accepts `--compile-learner` for CUDA FQI forward/backward
execution. The original model and Adam own checkpoint state; compilation does
not add `_orig_mod` keys or change the architecture, reward, sampling, update
budget, or target-refresh schedule. BF16 cast boundaries are retained where
supported, but fused reductions are not claimed to be bitwise equal to eager
execution. Tests check forward outputs, gradient direction, optimizer moments,
frozen targets, accumulation, short target batches, and resume continuity.

Microbatch is an execution choice:

| Option | Meaning |
| --- | --- |
| `--microbatch full` | One forward/backward for each rank's entire share of the global batch; no gradient accumulation. CLI default. |
| `--microbatch auto` | Probe real inputs and backward memory without optimizer steps; choose the largest divisor that fits with memory headroom. |
| `--microbatch 128` | Explicit 128-context microbatch, accumulating as needed. |

The central launcher exposes the same controls without a source edit:

```bash
# Large GPU: one full per-rank batch per optimizer update.
MICROBATCH=full COMPILE_LEARNER=1 bash outputs/operations/fqi-budget-vast-20261024/launch-scale-v2.sh

# Smaller GPU: probe available memory and retain the same global batch.
MICROBATCH=auto COMPILE_LEARNER=1 bash outputs/operations/fqi-budget-vast-20261024/launch-scale-v2.sh
```

Run only one controller for an output directory. These are alternative launch
commands, not concurrent jobs. `COMPILE_LEARNER=0` selects eager execution.

For a global batch of 256 on one GPU, `full` means 256 x 1; a 128 microbatch
means 128 x 2. On two GPUs, `full` means 128 x 1 per rank. Global batch and
optimizer update count never shrink to accommodate memory. Automatic probing
does not advance the sampler or RNG, and it does not update weights. It runs
before compilation, retains the real prefetch memory footprint, and leaves at
least 512 MiB or 5% of initially free VRAM as reserve. It chooses capacity, not
an empirically optimal throughput setting. Explicit `full` fails visibly if it
cannot fit, instead of silently changing the requested execution mode.

The effective batch, microbatch, accumulation count, and compiled-execution flag
appear in central training logs and W&B training metrics. Resource history keeps
the requested setting; FQI receipts keep both requested and effective values.
Compiler warmup is reported within phase duration and does not count as extra
optimizer updates. A first-call warmup is not repeated scientific startup training.

On the local RTX 4080, the full-shape fixture with global batch 256 and microbatch
128 improved from 277.7 ms to 166.6 ms per steady update. This is a fixture
measurement, not A100 real-replay throughput. A separate `auto` fixture run
encountered OOM at 256, selected 128, and completed 40 updates. A100 comparisons
and deployment status are recorded in the operation artifacts before claiming
formal training gains.

## A100 compiled execution comparison, 2026-10-25

Three disposable 64-update fits started from the same cycle 3 iteration 1
checkpoint and used the same 520-episode replay (390,251 decisions), sample seed,
global batch 256, BF16, and eight learner CPU threads. The table summarizes
updates 17-63, excluding compilation warmup and the final pipeline drain.

| Metric | Eager, 128 x 2 | Compiled, 128 x 2 | Compiled, 256 x 1 |
| --- | ---: | ---: | ---: |
| Median update time | 298.6 ms | 271.6 ms | 222.3 ms |
| Mean update time | 432.4 ms | 273.5 ms | 225.1 ms |
| Decisions/s from mean duration | 592.0 | 936.0 | 1137.1 |
| Exposed input wait per update | 5.3 ms | 55.6 ms | 56.8 ms |
| Sampled device utilization | 57.2% | 49.0% | 57.3% |
| Hardware samples | 50 | 46 | 37 |
| Peak PyTorch allocated memory | 10.03 GiB | 8.16 GiB | 16.17 GiB |

The eager window includes one 6.64-second update, so its mean substantially
overstates the typical step time. Median duration improves by 25.5% with compiled
full batches, equivalent to a 34.3% increase in typical update throughput. Full
batches also outperform compiled 128 x 2 by 18.2% in median duration. The default
for this A100 launcher is therefore `MICROBATCH=full`.

This is not sustained GPU saturation: input preparation remains exposed after
compute becomes faster. The compiler also reports that an indexed backward
operation cannot be CUDA-graph captured; enabling graph support does not imply
that the whole update is captured. No H100 or Blackwell speedup is established
by these A100 measurements. Do not compare target-phase times across these
sequential benchmarks: the first fit loads cold in-process replay contexts,
whereas later fits reuse them. All scratch fits retained frozen targets, and
their weights were discarded. Local RTX 4080 and remote A100 each passed the
58 focused execution, gradient, batch-selection, and resume checks.

Formal continuation was verified at cycle 3 iteration 2, attempt 2, resuming
iteration 1. The new partial checkpoint contains 200 additional optimizer updates
(4,200 actual budget-training updates in total), with an Adam step increment of
200 and frozen targets matching the resumed model. Its SHA256 is
`bd998c317256c6849efb03ff50a9baad0d6533fabbb11b52d83f46ef0cba6584`.
The new sampled plan is 1,000 x 256; sampler visits advance by 256,000 because
the complete iteration plan is drawn before fitting. This is not evidence that
all 1,000 updates have already completed.

The existing W&B run `a04b647b` is running and its cloud summary has new optimizer
updates beyond 4,000. Native system metrics continue on the same run. Collection
was retained at 24/24, with no repeated startup training. The exact source patch,
launcher, benchmark, checkpoint and resume evidence are in
`compiled-runtime-patch.json`, `compiled-resume-binding.json`,
`compiled-benchmark-summary.json`, `compiled-checkpoint-verified.json`, and
`compiled-resume-verified.json` under the operation's performance directory.

## Parallel visual copies, 2026-10-25

The next profile found that assembling a warm 256-decision CPU input took about
281 ms, including about 153 ms in NumPy stacking. Freezing Python GC did not
materially improve this profile. Increasing whole-batch preparation concurrency
was counterproductive: two, four and eight pending batches averaged 268, 378 and
641 ms per update, respectively. The campaign therefore retains two bounded
producers; more memory allocation is not treated as a performance gain.

An isolated exact-copy test of 1,083,801,600 raw bytes measured a warm median of
124.2 ms for NumPy stack, 54.1 ms for PyTorch stack, and about 100 ms for eight
separate NumPy copy tasks. Pinned collation now uses `torch.stack(..., out=pool)`
with the existing learner CPU thread budget. Input mmap views are read-only
operands; the output owns separate pinned storage. No trainable features are
cached. PyTorch may emit its once-per-process read-only NumPy view warning;
neither this operation nor any downstream consumer writes to those source views.

Two disposable real-replay fits compared the selected change with the original
copy path. Both used the same cycle 3 iteration 1 checkpoint, sampling seed,
520-episode replay, global/full batch 256, BF16, compiled learner, eight CPU
threads and 128 updates. Measurements cover updates 17-127, excluding initial
compilation and final input drain.

| Metric | NumPy pinned copy | Parallel PyTorch pinned copy |
| --- | ---: | ---: |
| Mean update duration | 234.8 ms | 203.3 ms |
| Median update duration | 220.9 ms | 204.9 ms |
| Decisions/s from mean duration | 1090.1 | 1259.0 |
| Exposed input wait per update | 68.6 ms | 33.4 ms |
| Sampled GPU utilization | 54.8% | 59.0% |
| Hardware samples | 94 | 80 |
| Peak PyTorch allocated memory | 17.41 GiB | 16.17 GiB |

Mean regression throughput increased 15.5%; median step duration decreased 7.2%.
These are short measurements with visible step-time variance, not full-campaign
averages. The first fit rebuilt cold replay contexts while the second reused
them, so their target-phase durations are not a controlled copy comparison.
Both plans visited 32,768 decisions with 31,472 distinct decisions, identical
source-group sampling and frozen target hashes. Compiled BF16 final weights were
not bitwise identical: mean losses were 1.77513 and 1.77561. Deterministic
CPU/CUDA tests separately verify exact inputs, Adam states and resume behavior;
this change does not claim bitwise equality of nondeterministic compiled runs.

The patch changes only raw pinned collation. An AST comparison verifies that the
history selector is unchanged. The related local and A100 test selections each
passed 33 tests. Results and the deployment receipt are retained under
`outputs/operations/fqi-input-throughput-vast-20261025/`.

The maintenance boundary preserves cycle 3's 5,000 actual optimizer updates and
all nine validation episodes. Its complete iteration 2 state SHA256 is
`768170150cb2a94cb80212246f6e14e032b435419748c37574f52d759cc2982d`.
The launcher keeps the same bounded cycle range, full batch 256, two input
producers, four rollout workers and W&B run `a04b647b`.

GPU saturation remains unproven. Residual input assembly and a compiled indexed
backward that cannot use CUDA Graph capture remain visible. Those are separate
optimization candidates; neither changes the acceptance criterion of higher
completed-update throughput under an unchanged scientific recipe.
