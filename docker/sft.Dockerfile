# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs build-essential ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_INSTALL_DIR=/opt/python
COPY third_party/openpi /opt/openpi
WORKDIR /opt/openpi
RUN uv python install 3.11.15 \
    && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev --python 3.11.15 \
    && uv pip install --python /opt/venv/bin/python pytest==8.3.5
COPY docker/belief /opt/belief-env
RUN UV_PROJECT_ENVIRONMENT=/opt/belief uv sync --project /opt/belief-env --frozen --no-dev --python 3.10
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/workspace/src PYTHONUNBUFFERED=1 \
    HF_HOME=/data/cache/huggingface OPENPI_DATA_HOME=/data/cache/openpi \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 WANDB_DIR=/data/wandb
WORKDIR /workspace
CMD ["/bin/bash"]
