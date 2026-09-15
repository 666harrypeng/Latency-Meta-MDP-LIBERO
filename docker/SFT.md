# Structured clean policy training

The job config selects the level, public dataset revision, batch and output repository.
The L2 job trains from the official π0.5 base with current 16D proprio, two current
camera images and the task prompt. Labels are clean H50 controller-native actions;
short episode tails retain their explicit loss mask. No future inputs or latency
shift are used in this stage.

## Build

```bash
git clone --branch rtc-forecast-policy https://github.com/666harrypeng/Latency-Meta-MDP-LIBERO.git
cd Latency-Meta-MDP-LIBERO
git submodule update --init third_party/openpi
docker build -f docker/sft.Dockerfile -t metamdp-sft .
mkdir -p "$PWD/sft-work/wandb"
```

Set `HF_TOKEN` (write access to the configured output repository) and
`WANDB_API_KEY` in the host shell before running. The dataset, base parameters and
tokenizer are public; missing assets download automatically and remain cached.
Tokens are passed at runtime, never embedded in the image.

## Run

Run this command inside a host tmux session:

```bash
docker run --name metamdp-l2-sft --gpus all --ipc=host \
  -e HF_TOKEN -e WANDB_API_KEY \
  -v "$PWD:/workspace" -v "$PWD/sft-work:/data" \
  metamdp-sft bash -o pipefail -c \
  'python -m latency_meta_mdp.cli.train_structured_pi05 \
    --config configs/training/pi05/l2_clean.yaml --work-dir /data/l2-clean \
    2>&1 | tee /data/l2-clean.log'
```

The mounted code checkout must include its Git metadata and initialized OpenPI
submodule. Training uses one JAX process with eight replicated devices, global
batch 256 (32/device), `fsdp_devices=1`. Do not wrap it in `torchrun`.

Before training, the command validates bundle file sizes, small metadata identities,
normalization, the actual loader batch and visible GPU count. Add `--check-only`
to run these checks without allocating the full model or starting optimization.
This is an input/topology check, not a full training benchmark.

The formal schedule is 3,000 optimizer steps / 768,000 examples, approximately
15.88 passes over the 48,373 L2 training sources. Milestones are 1,000, 2,000 and
3,000, with rolling resume checkpoints between them. W&B and stdout record progress.
Full training states remain under `/data/l2-clean/checkpoints` on the host mount.
At completion, the three inference milestones and normalization are uploaded to
the configured public model repository, with upstream notices and an empty README.
Optimizer state, logs and local run metadata are excluded from publication.

If interrupted, remove the stopped container (`docker rm metamdp-l2-sft`) and use
`--resume` with the same config and mounts. If training finished but upload failed,
use `--publish-only`; this retries publication without training again. Keep the
host work directory until the checkpoint upload has succeeded.

## Focused local checks

Inside the same container:

```bash
python -m pytest -q tests/test_sft_delivery.py tests/test_sft_launch.py
```

The full test suite remains available; it is not required to launch this job.
