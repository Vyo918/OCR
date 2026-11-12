"""Paths and model ids. Everything is overridable through environment variables so the same code
runs on a laptop (./work) and inside the container (/data/work, models in /models)."""
from __future__ import annotations

import os
from pathlib import Path

WORK = Path(os.environ.get("DOCSEARCH_WORK", "work"))          # all generated artefacts
INPUT = Path(os.environ.get("DOCSEARCH_INPUT", "."))           # default folder scanned by `ingest`
CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))   # model weights (HF hub cache + rapidocr/)

OCR_ENGINE = os.environ.get("DOCSEARCH_OCR_ENGINE") or "hybrid"   # a key of engines.REGISTRY; "ppocrv6-m-u20" = PP-OCR only (6x faster, no pipe-label fix)
PADDLEVL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
HUNYUAN_ID = "tencent/HunyuanOCR"
EMBED_ID = os.environ.get("DOCSEARCH_EMBED_MODEL", "BAAI/bge-m3")
RERANK_ID = os.environ.get("DOCSEARCH_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_ID = os.environ.get("DOCSEARCH_LLM_MODEL", "Qwen/Qwen3-4B-Thinking-2507")

# Hugging Face repos that are always needed (repo id -> files never needed for inference, keeps the model volume small).
MODELS = {
    "embed": (EMBED_ID, ["*.onnx", "onnx/*", "*.msgpack", "*.h5", "imgs/*", "colbert_linear.pt", "sparse_linear.pt"]),
    "rerank": (RERANK_ID, ["*.onnx", "onnx/*"]),
    "llm": (LLM_ID, []),
}
# OCR engines that are VLMs live on the hub too; the PP-OCR engines download their ONNX files into CACHE/rapidocr on first use.
OCR_HF = {"paddlevl": [PADDLEVL_ID], "hunyuan": [HUNYUAN_ID], "hybrid": [HUNYUAN_ID]}   # engine -> HF repos it loads (PP-OCR ONNX files come from RapidOCR)

PAGES = WORK / "pages"
OCR_DIR = WORK / "ocr"
INDEX_DIR = WORK / "index"
BENCH_DIR = WORK / "bench"
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
