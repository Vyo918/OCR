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
LLM_BACKEND = os.environ.get("DOCSEARCH_LLM_BACKEND") or "transformers"   # transformers (in-process) | http (OpenAI-compatible server; openai is an alias)
if LLM_BACKEND == "openai":
    LLM_BACKEND = "http"
LLM_URL = (os.environ.get("DOCSEARCH_LLM_URL") or "http://127.0.0.1:8081/v1").rstrip("/")
LLM_API_KEY = os.environ.get("DOCSEARCH_LLM_API_KEY") or ""
CONTEXT_MODE = os.environ.get("DOCSEARCH_CONTEXT") or "chunks"   # chunks (top-k reranked) | sheet (every block/section chunk, same text for every question)
FACTS = (os.environ.get("DOCSEARCH_FACTS") or "1") != "0"   # structured fact rows in the prompt; on by default (paired fresh-question test: +4 correct of 39), DOCSEARCH_FACTS=0 turns them off
VERIFY = (os.environ.get("DOCSEARCH_VERIFY") or "0") != "0"   # answer check and at most one retry (docsearch/verify.py); off by default, DOCSEARCH_VERIFY=1 turns it on
TOP_K = int(os.environ.get("DOCSEARCH_TOP") or 12)
SEED = int(os.environ.get("DOCSEARCH_SEED") or 0)   # sampling seed of the answering LLM; vary it to measure run-to-run noise
REREAD = os.environ.get("DOCSEARCH_REREAD") or "0"   # crop re-read by a vision reader (docsearch/reread.py): 0 (off, default) | openai | hunyuan
if REREAD not in ("0", "openai", "hunyuan"):
    raise ValueError(f"DOCSEARCH_REREAD must be 0, openai or hunyuan, not {REREAD!r}")
REREAD_URL = (os.environ.get("DOCSEARCH_REREAD_URL") or "http://127.0.0.1:8082/v1").rstrip("/")   # 8081 is the LLM
REREAD_MODEL = os.environ.get("DOCSEARCH_REREAD_MODEL") or "mlx-community/Qwen3-VL-4B-Instruct-4bit"
REREAD_MAX = int(os.environ.get("DOCSEARCH_REREAD_MAX") or 12)   # crops re-read per question

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
