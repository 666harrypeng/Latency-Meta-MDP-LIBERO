# Latency Meta-MDP

Research code for latency-aware robot policy learning and execution.

```bash
git submodule update --init third_party/openpi third_party/jepa-wms
uv sync --group dev
export PYTHONPATH=src
```

- [Environment setup](docker/README.md)
- [Training and publishing commands](docs/usage/training.md)
- [Experiment configurations](configs/experiments)
- [Conveyor task and source recordings](docs/usage/conveyor.md)

Implementations live in `src/latency_meta_mdp`; executable entrypoints are in
`scripts`. Historical implementations and configurations are isolated under
`legacy` and are not used by the current training entrypoints.
