# Latency Meta-MDP

Research code for latency-aware robot policy learning and execution.

```bash
git submodule update --init third_party/openpi third_party/jepa-wms
uv sync --group dev
export PYTHONPATH=src
```

- [Environment setup](docker/README.md)
- [Training task list: moving ball and conveyor](docs/usage/current-training.md)
- [Training and publishing commands](docs/usage/training.md)
- [Experiment configurations](configs/experiments)
- [Conveyor task and source recordings](docs/usage/conveyor.md)
- [Conveyor information scheduler and FQI](docs/usage/conveyor-meta.md)
- [Conveyor attention scheduler and bounded FQI](docs/usage/conveyor-attention-meta.md)
- [Historical actor / PPO experiment](docs/usage/conveyor-actor-training.md)
- [Legacy FQI maintenance, evidence, and recovery handoff](docs/usage/legacy-fqi-handoff.md)

Implementations live in `src/latency_meta_mdp`; executable entrypoints are in
`scripts`. Historical implementations and configurations are isolated under
`legacy` and are not used by the current training entrypoints.

The LIBERO/FQI experiment is preserved here. The new Isaac Lab online-PPO design
lives in the separate private `latency_meta_control_isaac` repository. Its planned
action, latency, and reward contracts do not change this repository's experiments.
Use English for tracked first-party content; keep dated local evidence intact.
