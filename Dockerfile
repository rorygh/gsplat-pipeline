# Lightweight CUDA devel base (not a full pytorch image) -- torch is pulled
# by uv from PyTorch's own CUDA-matched index below, so there's no bundled
# torch build to fight with or override.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# COLMAP via a standalone micromamba env: Ubuntu 22.04's apt colmap is an
# incompatible v3.x build. Use a CUDA-enabled build so SfM feature
# extraction + matching run on the GPU (the CPU builds' SIFT matcher is
# ~10+ min for 50 images). Pick a build whose CUDA floor is <= this image's
# CUDA (12.4): conda-forge's colmap=3.10=gpu* needs __cuda>=12.0; the 3.11+
# GPU builds need 12.6-12.9 and fail at runtime on a 12.4 driver.
# CONDA_OVERRIDE_CUDA lets the solver see a GPU at build time (there is none).
# QT_QPA_PLATFORM=offscreen: COLMAP builds a QApplication even for CLI
# subcommands and aborts on a headless host without it.
# Wrapped as a script rather than a global LD_LIBRARY_PATH, which can leak
# conda's OpenSSL into unrelated system tools (e.g. sshd).
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    CONDA_OVERRIDE_CUDA=12.4 micromamba create -y -p /opt/colmap-env -c conda-forge \
        "colmap=3.10=gpu*" "openimageio=3.1.*" && \
    printf '#!/bin/bash\nexec env LD_LIBRARY_PATH="/opt/colmap-env/lib:$LD_LIBRARY_PATH" QT_QPA_PLATFORM=offscreen /opt/colmap-env/bin/colmap "$@"\n' > /usr/local/bin/colmap && \
    chmod +x /usr/local/bin/colmap

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# `--frozen` installs exactly what's in uv.lock (no re-resolving at build
# time) into a venv at /app/.venv -- reproducible builds, and torch resolves
# to the CUDA 12.4 wheel per pyproject.toml's [tool.uv.sources] override.
RUN uv sync --frozen --no-dev

# gsplat JIT-compiles its CUDA kernels via torch's cpp_extension loader on
# first `rasterization()` call (i.e. the first `train` or `view`), not at
# image-build time -- expect a one-time multi-minute compile on first use
# inside a freshly built container.

# viser's websocket port, for the interactive viewer.
EXPOSE 7007

ENTRYPOINT ["uv", "run", "gsplat-pipeline"]
CMD ["--help"]
