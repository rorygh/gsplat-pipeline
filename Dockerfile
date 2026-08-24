# Lightweight CUDA devel base (not a full pytorch image) -- torch is pulled
# by uv from PyTorch's own CUDA-matched index below, so there's no bundled
# torch build to fight with or override.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates python3.11 python3.11-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# COLMAP via a standalone micromamba env: Ubuntu 22.04's apt colmap is an
# incompatible v3.x build, and libfaiss must be pinned to 1.10.0 (newer
# conda-forge builds are ABI-incompatible with this colmap build). Wrapped
# as a script rather than a global LD_LIBRARY_PATH, which can otherwise leak
# conda's OpenSSL into unrelated system tools (e.g. sshd).
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    micromamba create -y -p /opt/colmap-env -c conda-forge colmap=4.0.4 "libfaiss=1.10.0=cpu_openblas*" && \
    printf '#!/bin/bash\nexec env LD_LIBRARY_PATH="/opt/colmap-env/lib:$LD_LIBRARY_PATH" /opt/colmap-env/bin/colmap "$@"\n' > /usr/local/bin/colmap && \
    chmod +x /usr/local/bin/colmap

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src

# Install straight into the system interpreter -- a container is already an
# isolated environment, so a nested venv just adds an activation step for
# no benefit here.
RUN uv pip install --system -e .

# gsplat JIT-compiles its CUDA kernels via torch's cpp_extension loader on
# first `rasterization()` call (i.e. the first `train` or `view`), not at
# image-build time -- expect a one-time multi-minute compile on first use
# inside a freshly built container.

# viser's websocket port, for the interactive viewer.
EXPOSE 7007

ENTRYPOINT ["gsplat-pipeline"]
CMD ["--help"]
