# Training environment

Select the branch or revision containing the experiment configuration. See the
[training task list](../docs/usage/current-training.md) for task-specific entries.

For a new checkout, replace `<branch>` with the selected branch:

```bash
export BRANCH="<branch>"
git clone --branch "$BRANCH" https://github.com/666harrypeng/Latency-Meta-MDP-LIBERO.git
cd Latency-Meta-MDP-LIBERO
```

For an existing clean checkout, switch and update before building:

```bash
export BRANCH="<branch>"
git fetch origin
git switch "$BRANCH"
git merge --ff-only "origin/$BRANCH"
```

Use a separate checkout when the existing one has local changes or is in use.
For a pinned revision, check out that commit before initializing submodules and
building the image:

```bash
git submodule update --init third_party/openpi third_party/jepa-wms
docker build -f docker/sft.Dockerfile -t metamdp-sft .
```

Set `RUN_DIR` to an absolute host storage directory. Export `HF_TOKEN` and
`WANDB_API_KEY` in the host shell, then start the container in a persistent
terminal session such as tmux. The host needs a working NVIDIA Container Toolkit.

```bash
export RUN_DIR=/absolute/path/to/training-storage
mkdir -p "$RUN_DIR"/cache/tmp "$RUN_DIR"/wandb
docker run --rm -it --gpus all --ipc=host \
  -e HF_TOKEN -e WANDB_API_KEY \
  -e JAX_COMPILATION_CACHE_DIR=/data/cache/jax \
  -e TMPDIR=/data/cache/tmp \
  -v "$PWD:/workspace" -v "$RUN_DIR:/data" \
  metamdp-sft bash
```

Inside the container, run from `/workspace`. `python` uses the OpenPI environment;
`/opt/belief/bin/python` uses the separate Belief/forecast environment. Both are
included in the same image. Model caches, generated data, W&B files and outputs
remain under the mounted `/data` directory across container restarts.

Choose a job from the [training task list](../docs/usage/current-training.md).
Follow [training commands](../docs/usage/training.md) for pilots, resume and publishing.
The job configuration selects datasets, model revisions and training settings.
Downloads use `HF_TOKEN` when supplied; it needs access to the official DINO
weights and write access to the configured checkpoint repositories.
