# Agent entrypoint

## Start here

- Read [README.md](README.md) for the repository layout.
- For training, read the [training task list](docs/usage/current-training.md).
  It lists moving-ball and conveyor jobs separately. Run the task and configuration
  selected in the operator's startup request; list order does not set priority.
  Replace the list when assignments change rather than appending run history.
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
