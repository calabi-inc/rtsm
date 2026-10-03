# syntax=docker/dockerfile:1.7
# RTSM — the CUDA-ready image: Python 3.12, PyTorch cu128, the gpu / eval / mcp
# extras, no model weights. Published as ghcr.io/calabi-inc/rtsm:<version> by
# .github/workflows/release.yml; built without pushing on pull requests that
# touch it (.github/workflows/docker.yml).
#
#   docker build -f docker/Dockerfile -t rtsm .
#   docker run --rm --gpus all -v rtsm-models:/models -v "$PWD":/data rtsm eval my.bag --repeats 3
#   docker run --rm --gpus all -v rtsm-models:/models -p 8002:8002 -p 8765:8765 rtsm
#
# Needs an NVIDIA driver on the host and the NVIDIA Container Toolkit (--gpus).
# The PyTorch wheels carry their own CUDA libraries, so the base is plain
# Python, not a CUDA image. Weights download on first use into /models (mount
# a volume there or every container starts cold). The AGPL backends (dual,
# fastsam, yoloe) are not in the image: `pip install ultralytics` and mount
# their weights at /data/model_store if you opt in.

FROM python:3.12-slim-bookworm

ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    SAM2_BUILD_CUDA=0 \
    RTSM_CONTAINER=1

# cv2 needs libGL + glib; nothing else outside pip.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 rtsm \
    && mkdir -p /models/hf /models/torch /data \
    && chown -R rtsm:rtsm /models /data

# PyTorch first, from the CUDA index, in its own layer: it is the bulk of the
# image and does not change when rtsm does.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch==${TORCH_VERSION}" torchvision --index-url "${TORCH_INDEX}"

# rtsm from the build context (the tag being released, or the PR branch).
COPY --chown=rtsm:rtsm . /src
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "/src[gpu,eval,mcp]" \
    && rm -rf /src

USER rtsm
WORKDIR /data
EXPOSE 8002 8083 8765
ENTRYPOINT ["rtsm"]
CMD []
