#!/usr/bin/env bash
# Install the pinned project-local FQI runtime and frozen model inputs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -f scripts/run_conveyor_meta.py
command -v uv >/dev/null
export HF_HOME="$PWD/outputs/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
for module in openpi jepa-wms; do
  if [ "$module" = openpi ]; then
    origin=https://github.com/Physical-Intelligence/openpi.git
    revision=15a9616a00943ada6c20a0f158e3adb39df2ccac
  else
    origin=https://github.com/facebookresearch/jepa-wms.git
    revision=13cf1d9c7e476f53c17714d2e0f1dc239a883ce0
  fi
  target="third_party/$module"
  if [ ! -e "$target/.git" ]; then
    git clone "$origin" "$target"
    git -C "$target" checkout "$revision"
  fi
  test "$(git -C "$target" rev-parse HEAD)" = "$revision"
  test "$(git -C "$target" remote get-url origin)" = "$origin"
done
uv venv --allow-existing --python 3.11.15 .venv-policy-eval
GIT_LFS_SKIP_SMUDGE=1 uv pip install --no-deps --python .venv-policy-eval/bin/python \
  -r requirements/fqi-runtime.txt
export PYTHONPATH="$PWD/src"
.venv-policy-eval/bin/python scripts/fqi_handoff.py restore-assets \
  --manifest configs/data/conveyor_optional/asset-downloads.json --root "$PWD"
echo 'Runtime and frozen assets restored; qualify CUDA/JAX/EGL before training.'
