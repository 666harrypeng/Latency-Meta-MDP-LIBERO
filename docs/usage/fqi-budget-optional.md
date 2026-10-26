# Scratch budget FQI: optional prediction candidate

> Historical v1 implementation/qualification record. For the current 0.02-cost
> scratch delivery, use [the training guide](fqi-collaborator-handoff.md).

Implementation and bounded GPU qualification status: 2026-10-26. The protocol
has passed regression and four real-model evaluation episodes. It is **not** a
qualified training recipe or evidence of improved delivery/cost. The historical bound
success-only and Vast budget campaigns retain their original configurations.

## Execution contract

Use experiment schema 3, mode `optional_prediction`, a new nonempty
`experiment_id`, and action version 3 with `starvation_mode: allow_hold` and
`reserve: 0`. Existing schema 1/2 inputs remain supported on their own paths.
The memory/attention backbone and frozen conditioned VLA are retained. Three
learned value heads now represent G, O and P.

| Stage | Choices | Input authority |
| --- | --- | --- |
| G | Sleep, Observe | Proprioception, execution/request history and previously acquired memory |
| O | Sleep, direct Launch; Predict when ready | The observation explicitly acquired at this boundary |
| P | Sleep, predicted Launch | A forecast from this boundary and buffer version |

Predict readiness means its real history and executable-prefix inputs exist;
it is not privileged knowledge of prediction quality. Early Observe is valid
before the JEPA history is ready. Sleep advances physical time and returns to G.
A forecast prepared before Sleep cannot silently supply a later direct Launch.
There are no added event wakes, automatic replans, interrupts or forced requests.

The RTC client starts at tick zero with no VLA plan and buffer version zero.
Empty-buffer hold sends zero arm delta while retaining the last gripper command
(initially open). A known hold prefix can condition inference, but is not marked
as an unconsumed VLA chunk in the scheduler buffer. The first delayed action
chunk installs using its observation-time cursor. Collection and validation do
not prepare, load or call the clean-VLA bootstrap under schema 3.

A direct Launch passes an explicitly unavailable `DecodedForecast` to the same
conditioned VLA. Both forecast image masks are false; the current views and
proprioception remain present. A predicted Launch lazily decodes the prepared
forecast. Input support does not establish closed-loop skill on that branch.

Operator decision after the 2026-10-26 qualification: retain this mask-based
current-only path and the same frozen conditioned VLA. Additional current-only
capability ablations are deferred and are not a prerequisite for training.
Preserve the observed results as evidence; this decision does not establish
current-only competence or authorize restoring mandatory prediction.

## Data, reward and experiment identity

- Schema-3 raw and packed episodes preserve real, unforced O decisions and
  prediction readiness. Old bound replay cannot be relabeled as new-protocol
  data. Forced-stage records are rejected on the new path.
- Budget targets use derived `training_reward` from parcel events
  and calibrated module-call prices. The normalized historical `task_reward`
  remains an audit field. Protocol 3 prices every free decision as a deployment
  Meta decision, even when a rule collected it; actual calls and imputed service
  equivalents are reported separately. Same-tick costs and outcomes during hold
  retain their actual decision owners; no idle bonus is added.
- Fresh campaigns reject an external `initial_checkpoint`, nonzero cycle offset
  and imported online sources. Meta parameters start from the declared seed;
  no S05 representation, optimizer, sampler or Q-value scaling is imported.
- Checkpoints carry `initialization` with kind `scratch`, seed, initial model
  hash and `experiment_id`. Later fits preserve that identity and the existing
  optimizer/sampler recovery. Parent checkpoints must match the same config and
  experiment. Identities use experiment names rather than machine-local paths.
- To resume, use the original campaign journal/config and its retained states;
  do not create a new recipe pointing to an old task's checkpoint. The central
  W&B writer's existing output/config binding rejects reuse of an old run file.

Schema-3 reports add hold ticks, Observe/Predict/Launch counts, current-only and
predicted launches, and the fraction of launches using prediction. A zero
fraction with zero launches is reported as zero with the request counts beside
it, not as an efficiency claim. Validation retains separate delivery ratios,
budget returns and module costs. The new schema-3 candidate selects by mean
budget return across all three regimes, breaking ties by delivery and then cost.
Historical schema-1/2 selection remains nominal/family delivery. Inspect every
regime separately; see the [collaborator handoff](fqi-collaborator-handoff.md) for
the candidate recipe and current delivery acceptance boundary.

## Entry points and remaining qualification

Reuse `scripts/run_conveyor_meta.py` and the central
`scripts/run_fqi_experiment.py` supervisor. Resource controls remain the existing
GPU, worker, CPU-thread, full/microbatch and compiler options. Schema-3
collection/validation requires no bootstrap files; in a campaign recipe the
legacy `validation_bootstrap` field can be null. The legacy `diagnose` command
compares bound/split protocols and is explicitly unsupported for schema 3;
use the regular smoke/collection path for candidate qualification.

The new candidate is now specified in
`configs/experiments/conveyor_sort/meta_budget_optional_v1.yaml` and the recipe
builder. Follow the [collaborator handoff](fqi-collaborator-handoff.md) for the
192-episode starter, scratch update budget, resource controls, private HF transfer
and exact delivery acceptance receipt. Do not treat an in-progress delivery as
accepted until that receipt confirms collection, real fitting, learned collection,
validation, resume and publication. Formal weights must still start from scratch;
engineering acceptance states are not formal initializers.

## Local verification

The final related CPU regression passed 220 tests. Its two JAX-dependent
collection tests were rerun successfully in the existing OpenPI environment,
whose optional-protocol/forecast suite passed all 30 tests. These suites overlap;
their counts must not be summed as distinct tests. The later L40S run passed the
51-test GPU/optional suite, including CUDA/Inductor/pinned-memory checks and the
JAX-dependent cases; its separate preflight/binding suite passed 11 tests.
The compile test fixture now applies the production learner's deterministic,
TF32-disabled kernel policy; no numeric tolerance was relaxed. These regression
tests use lightweight fixtures. Actual deployed-model evaluation is below.

Independent review found and fixed an absent-bootstrap metadata access at
episode finalization and an unsupported legacy diagnostic entry path. Training
and evaluation entrypoint tests now exercise complete episode saving without
a bootstrap. Ruff and whitespace checks pass on the resulting change.

Local evidence and before-images are under
`outputs/operations/fqi-budget-scratch-20261026/`. Local planning notes also live
under `docs/superpowers/` (ignored by Git); this usage document is the portable
implementation summary and qualification checklist.

## Bounded real-VLA qualification

On two L40S GPUs, the same frozen conditioned VLA ran with a periodic E25 rule,
one worker per GPU, on historical scene 2000 (excluded from training):

| VLA input branch | Zero delay | Nominal delay |
| --- | ---: | ---: |
| Current-only | 0/10 delivered | 0/10 delivered |
| Current plus forecast | 3/10 delivered | 1/10 delivered |

Every episode completed, with 80 unique VLA requests and 80 matching arrivals.
Current-only made no JEPA/decoder calls; predicted episodes made 80 of each.
Both branches started with empty buffers, used zero-arm-delta hold, and retained
H50 action buffers. No clean bootstrap, old Meta weights, forced decisions or
training replay was used. The central W&B run `sskbjlnv` finished and its cloud
delivery/cost summaries match the local records. This is a rule-policy engineering
check on one scene, not a learned Meta result or a generalization benchmark.

The current-only traces contain actual grasps and losses, so absence of requests
does not explain the zero deliveries. The conditioned-SFT dataset exposes absent
forecasts at episode boundaries rather than throughout manipulation. Limited
training support for this branch is a concrete hypothesis, not an established
cause. The operator subsequently deferred additional ablations and chose to
proceed with masked current-only inputs. Record the limitation while preparing
training; do not silently replace the frozen VLA or restore a mandatory
Observe-to-Predict link.

Evidence, videos, deployed source manifests, environment pins and the post-run
report are under `outputs/operations/fqi-optional-vast-20261026/`. Existing module
prices were retained as diagnostic reference costs, not measured L40S energy or
a newly approved formal reward recipe. Source files were not modified during
the successful rollout; this documentation update is subsequent to that run.
