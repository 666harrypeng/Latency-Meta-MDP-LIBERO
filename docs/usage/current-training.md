# Current training assignment

The current RoboSuite/MuJoCo assignment is **scratch optional-prediction budget FQI**.

- Configuration: [meta_budget_optional_v2.yaml](../../configs/experiments/conveyor_sort/meta_budget_optional_v2.yaml)
- Setup, inputs, hardware qualification, launch, and checkpoint return:
  [FQI training guide](fqi-collaborator-handoff.md)
- Runtime installation: `bash scripts/setup_fqi_runtime.sh`
- Central launcher: `bash scripts/launch_optional_fqi.sh`

The conditioned VLA, DINO, JEPA, and decoder are frozen inputs. Train a new Meta
controller; do not initialize from historical success-only or acceptance weights.
Use reward `10 * delivered - 5 * missed - 0.02 * compute_cost`, optional Predict,
and empty-buffer hold. Resource parameters can be tuned on the assigned node.

Historical SFT commands remain in [training.md](training.md). Earlier Meta/FQI
results remain documented in [legacy-fqi-handoff.md](legacy-fqi-handoff.md).
These are not active training assignments. The separate Isaac project is outside
this repository's current job.
