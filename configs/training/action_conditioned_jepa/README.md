# Direct-query predictor training

The current RTC forecast policy uses `DirectJepaPredictor`: one forward pass for
query ticks 1–20, with history at h−8, h−4 and h. `direct_query.yaml` is the shared
optimization recipe. A level job such as `direct_query_l2.yaml` selects the data,
normalization and execution device. Paths in the job resolve from the project root.

Use the JEPA environment with the pinned `third_party/jepa-wms` checkout initialized.
Run from the project root:

```bash
export PYTHONPATH=src
python -m latency_meta_mdp.cli.preflight_direct_jepa \
  --config configs/training/action_conditioned_jepa/direct_query_l2.yaml \
  --output-dir outputs/analysis/l2-direct/preflight --optimizer-steps 5
python -m latency_meta_mdp.cli.train_direct_jepa \
  --config configs/training/action_conditioned_jepa/direct_query_l2.yaml \
  --preflight outputs/analysis/l2-direct/preflight/preflight.json \
  --output-dir outputs/training/l2-direct
python -m latency_meta_mdp.cli.evaluate_direct_jepa \
  --config configs/training/action_conditioned_jepa/direct_query_l2.yaml \
  --run-dir outputs/training/l2-direct \
  --output-dir outputs/analysis/l2-direct/review
```

Set `WANDB_API_KEY` at runtime and keep training in tmux. Use a clean committed
checkout. The trainer starts from scratch and saves rolling resume state each
epoch plus immutable weights at epochs25/50/75. Add `--resume` to the training
command after an interruption; keep the same code, job and preflight identity.
Evaluation can include `--decoder-dir` to produce RGB review panels.
No command here publishes to Hugging Face.

An epoch is44,800 sampled real endpoint pairs, equally distributed across twenty
queries; it is not a complete pass through every possible pair. Real endpoint
requirements and the train/validation master split remain enforced. The current
runner checks existing immutable input metadata, file sizes and array contracts
without hashing large payloads again; the underlying loaders retain full-hash
verification by default for initial corpus certification.

`direct_query_l3_v1.yaml` and preflight without `--config` preserve the historical
L3 recipe/entry. `final_admission.yaml` is the separate autoregressive training
path; its weights are not interchangeable with Direct weights. Historical
experiment-only launchers under ignored outputs are not the release entry point.
