# syntax=docker/dockerfile:1.7
# OCR (PP-OCRv6 + HunyuanOCR cascade) + hybrid retrieval + local LLM, GPU build.
# Needs on the host: NVIDIA driver >= 560 and the NVIDIA Container Toolkit (docker run --gpus all).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_XET=1 \
    TOKENIZERS_PARALLELISM=false \
    DOCSEARCH_WORK=/data/work \
    DOCSEARCH_INPUT=/data/in \
    HOME=/tmp \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    UV_HTTP_TIMEOUT=300 \
    UV_CONCURRENT_DOWNLOADS=4

# poppler-utils: pdfinfo / pdftoppm rasterise the scanned PDFs at 300 dpi
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv==0.13.7

# CUDA wheels bundle their own CUDA runtime, so no CUDA base image is needed.
# cu126 = widest driver support (>= 560). Newer driver and want a newer CUDA build? --build-arg TORCH_INDEX=.../cu130 (and TORCH_VERSION).
ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system torch==${TORCH_VERSION} --index-url ${TORCH_INDEX}

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r requirements.txt

COPY pyproject.toml ./
COPY docsearch ./docsearch
RUN uv pip install --system --no-deps .

# world-writable so the container can run as the host user (compose passes --user)
RUN mkdir -p /models /data/in /data/work && chmod 777 /models /data/work

# BAKE_MODELS=1 -> ~28 GB image (11.5 GB code+torch, ~16.6 GB weights) that works offline (build-time network needed).
# default 0 -> slim image; weights are downloaded once into the /models volume (`docsearch prefetch`).
ARG BAKE_MODELS=0
RUN if [ "$BAKE_MODELS" = "1" ]; then docsearch prefetch && chmod -R a+rwX /models; fi

ENTRYPOINT ["docsearch"]
CMD ["--help"]
