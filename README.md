# Latency Meta-MDP

Research code for latency-aware asynchronous robot policy inference.

This repository is under active development.

## Installation

Python 3.10 and [uv](https://docs.astral.sh/uv/) are required.

```bash
git submodule update --init --recursive
uv sync --group dev
uv run python -m latency_meta_mdp.cli.check_runtime --help
```

## Clean policy SFT

Use the [Docker training instructions](docker/SFT.md) and the
[L2 job config](configs/training/pi05/l2_clean.yaml) for structured state16 data.
This training environment is separate from the simulation environment above.
