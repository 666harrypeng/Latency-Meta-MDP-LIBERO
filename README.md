# Latency Meta-MDP

Learning when a robot should observe, predict, and request actions under inference
latency. This repository implements conveyor manipulation in RoboSuite/MuJoCo,
with a frozen vision-language-action policy and a trainable temporal-memory Meta
controller.

The Meta controller reads observed history and optional world-model forecasts.
It can wait with an empty action buffer, acquire a new observation, request a
prediction, or launch a conditioned VLA action chunk. The current FQI experiment
trains this scheduler from scratch for successful deliveries and lower inference
cost. The VLA, visual encoder, world model, and RGB decoder stay frozen.

## Quick start

Use an NVIDIA GPU node with CUDA/EGL support, Git, and `uv`. Dependencies use a
project virtual environment, with project-local model caches. Export `HF_TOKEN`
and `WANDB_API_KEY` in your shell;
HF access is required for the private dataset and frozen models.

```bash
git clone https://github.com/666harrypeng/Latency-Meta-MDP-LIBERO.git
cd Latency-Meta-MDP-LIBERO
bash scripts/setup_fqi_runtime.sh
export PYTHONPATH="$PWD/src"
.venv-policy-eval/bin/python scripts/fqi_handoff.py --help
```

Follow the [training guide](docs/usage/fqi-collaborator-handoff.md) to download the
192-episode starter, qualify your GPU node, and launch training. Hardware settings
are explicit launcher parameters; the scientific recipe does not depend on GPU type.
Code comes from GitHub; datasets and checkpoints come from Hugging Face.

## Training and evaluation

The current [experiment configuration](configs/experiments/conveyor_sort/meta_budget_optional_v2.yaml)
uses optional prediction and permits empty-buffer hold. Its reward is
`10 × delivered − 5 × missed − 0.02 × compute_cost`, with fixed relative module
prices. Offline FQI is followed by online collection and replay-based FQI with a
gradual data-mixture schedule. Evaluation uses fixed ten-parcel scenes under zero,
nominal, and family latency.

All completed Meta checkpoints are uploaded to the
[checkpoint repository](https://huggingface.co/yypeng666/metamdp-conveyor-optional-budget-meta-v1),
with manually approved download access and links to the corresponding W&B run.
The checkpoint index marks which states have validation results. Engineering
qualification is separate from evidence of improved task performance.

- [Training, evaluation, hardware, and storage guide](docs/usage/fqi-collaborator-handoff.md)
- [Conveyor task](docs/usage/conveyor.md) and [scheduler architecture](docs/usage/conveyor-attention-meta.md)
- [Policy SFT workflows](docs/usage/training.md)

Implementation: `src/latency_meta_mdp/`; commands: `scripts/`; experiments:
`configs/experiments/`. Earlier experiments remain available for reproduction.
