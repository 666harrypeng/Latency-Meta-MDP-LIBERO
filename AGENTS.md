# Agent entrypoint

## Start here

- Read [README.md](README.md) for the repository layout.
- For an assigned training run, read the [current training tasks](docs/usage/current-training.md).
  That file specifies this round only; replace it for the next round rather than
  appending history. Long-term experiment tracking is maintained outside this repository.
- Use [Docker setup](docker/README.md) and [training commands](docs/usage/training.md)
  for environment and CLI instructions. Experiment YAML files are the source of
  truth for datasets, revisions, training settings and publication destinations.

## Code navigation

- `scripts/`: executable entrypoints; `configs/experiments/`: experiment inputs.
- `src/latency_meta_mdp/data/`: source data, policy exports and forecast preparation.
- `belief/`, `policy/`, `meta/`: predictor, VLA and scheduler implementations.
- `runtime/`, `envs/`, `io/`: execution, simulation and artifact handling.
- Historical implementations are under `src/latency_meta_mdp/legacy/` and
  `configs/legacy/`. Use the current scripts rather than old `outputs/launch/` files.

## Running assigned tasks

- Run commands from the repository root. Use the pinned environments and patches.
  In Docker, OpenPI uses `python`; Belief/forecast preparation uses
  `/opt/belief/bin/python`. The policy pipeline selects these environments itself.
- Keep long jobs in a persistent terminal session, such as host-side tmux, and
  retain logs and checkpoints on mounted storage. Resume using the documented CLI.
- Keep the assigned configuration unchanged unless the operator requests a change.
  Report startup failures rather than silently changing the training recipe.
- Supply credentials through environment variables; report only SET/UNSET.
  Use the provided checkpoint publisher rather than uploading whole run directories.
- Report completion, checkpoint steps, W&B links and published HF revisions to the
  operator. Do not append those results to the current task file or create a task archive.

## Development

Keep implementation in the corresponding package and entrypoint scripts thin.
Validate changes with relevant tests and Ruff. Commit or push only when requested;
running a training assignment does not require code edits or Git publication.
