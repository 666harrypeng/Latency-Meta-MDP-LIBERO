# Conveyor IQL initialization and on-policy PPO

This is the Delta parallel experiment. The existing FQI campaign remains a separate run. Both use the same frozen perception/VLA assets, public attention memory, action masks, and success-only environment reward.

`meta_attention_v2.yaml` pins the shared asset/input/action contract, including the original FQI source-data identity. The new algorithm and budgets are read from `attention_iql_ppo_v1.yaml`; its IQL/PPO runner does not execute the legacy `training` section. Do not edit the shared configuration to rename its old training algorithm: that would invalidate existing startup provenance.

The stage order is IQL value learning, weighted categorical actor extraction, fresh critic calibration, a two-cycle pilot, and 100 online PPO cycles. Online fitting uses only its current, completed stochastic rollout batch. Calibration and startup are never sampled by PPO. The actor and scalar critic have separate parameters and optimizers; Observe/Predict and starvation semantics stay unchanged.

## Inputs and runtime

Supply a JSON list of the three immutable startup sources, each with `path`, `manifest_sha256`, and `group: startup`. These must cover the approved448 episodes; the loader verifies actual completion receipts and hashes. Supply the existing scene audit and bootstrap directory separately. Training and validation seeds remain disjoint.

Run from the canonical repository checkout in the qualified environment, inside tmux or its Slurm allocation. Keep the output on persistent storage and the reconstructable cache on node scratch. GPU IDs, microbatch and CPU resources are runtime choices:

```bash
python -u scripts/run_conveyor_actor.py \
  --config configs/experiments/conveyor_sort/meta_attention_v2.yaml \
  --recipe configs/experiments/conveyor_sort/attention_iql_ppo_v1.yaml \
  --sources /absolute/path/startup-sources.json \
  --scene-audit /absolute/path/scene-inventory-audit.json \
  --bootstrap-dir /absolute/path/bootstrap \
  --output-dir /absolute/path/persistent/run \
  --cache-dir /absolute/path/node-scratch/replay-cache \
  --gpus 0,1 --microbatch 2 --cpu-threads 4 --stage pilot
```

`--stage offline` stops after actor selection; `pilot` stops after calibration and two24episode cycles. `formal` additionally checks measured storage requirements and proceeds with96episodes×100cycles. Pilot updates never initialize formal PPO. Use formal only after the single/multicard qualification and code review have passed. The script `verify_conveyor_actor.py` accepts a small, explicitly selected real-replay source manifest for bounded hardware qualification; it does not change formal budgets.

## Progress and resume

Use a distinct W&B project (default `latency-meta-mdp-robosuite-delta-ppo`) and the output directory's persistent run ID. The terminal prints `REPLAY`, `ACTOR_LEARN`, `ACTOR_SCORE`, `PARALLEL_PROGRESS`, `EPISODE_COMPLETE`, and `PPO_CAMPAIGN`. Collection/validation workers retain their logs and video indexes under their stage directories. Model computation and simulator/video overhead retain the existing per-episode component telemetry.

Rerun the same command/output directory after a time limit. The campaign checks recipe, source and data bindings; it restores complete checkpoints with Adam state. IQL/actor save every500updates; critic/PPO save at completed epoch boundaries. An interrupted epoch restarts from its prior completed boundary on the same frozen behavior-policy batch. Complete episodes are reused; interrupted episodes are excluded. The latest actor collects the next batch; best selection uses nominal/family validation, preserving the incumbent on ties.

Only a committed cycle with a verified successor checkpoint can retire its large collection payload. Lightweight summaries, receipts, validation videos, model and optimizer checkpoints remain. Startup archive is retained. The cache has one global allowance across stages. Storage admission fails clearly rather than deleting data needed for recovery. Separate runtime quotas and free space still need verification before launch.
