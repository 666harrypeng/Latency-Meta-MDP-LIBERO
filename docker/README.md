# Training environment

Set `RUN_DIR` to the host directory used for outputs.

```bash
git submodule update --init third_party/openpi third_party/jepa-wms
docker build -f docker/sft.Dockerfile -t metamdp-sft .
mkdir -p "$RUN_DIR"
```

Pass credentials from the host environment and persist generated files outside
of the container:

```bash
docker run --rm -it --gpus all --ipc=host \
  -e HF_TOKEN -e WANDB_API_KEY \
  -v "$PWD:/workspace" -v "$RUN_DIR:/data" \
  metamdp-sft bash
```

Inside the container, `python` uses the OpenPI environment. Frozen forecast
preparation uses `/opt/belief/bin/python`; the policy pipeline selects it automatically.
Run [training commands](../docs/usage/training.md) from `/workspace`.
