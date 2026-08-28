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
