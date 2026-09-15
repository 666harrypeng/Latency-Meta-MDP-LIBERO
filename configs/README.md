# Configuration layout

- `experiments/`: task, dataset, initialization, runtime overrides and output repositories.
- `training/`: reusable optimization recipes.
- `models/`: model architecture and feature-encoder definitions.
- `tasks/` and `runtime/`: environment and execution settings.
- `data/`: collection, conversion and split settings.
- `contracts/`: versioned snapshots required to read published datasets and artifacts.
- `legacy/`: configurations for historical implementations.

Change experiment and training settings without editing an existing data contract.
