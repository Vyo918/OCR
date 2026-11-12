"""Download model weights into the model cache (HF_HOME; /models in the container)."""
from __future__ import annotations

import os

from . import config


def prefetch(only: list[str] | None = None) -> None:
    """Fetch the weights for the configured OCR engine plus embedder, reranker and LLM (or just the roles in `only`)."""
    # The xet transport stalls on many corporate / home links; plain HTTPS resumes reliably.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    roles = {"ocr": None, **config.MODELS}
    for role, spec in roles.items():
        if only and role not in only:
            continue
        if role == "ocr":
            for repo in config.OCR_HF.get(config.OCR_ENGINE, []):
                print(f"[prefetch] ocr: {repo}", flush=True)
                print(f"[prefetch] ocr ready at {snapshot_download(repo)}", flush=True)
            if config.OCR_ENGINE not in ("paddlevl", "hunyuan"):   # PP-OCR part: ONNX files from the RapidOCR release, fetched when the engine is built
                from . import engines
                print(f"[prefetch] ocr: PP-OCR ONNX -> {config.CACHE / 'rapidocr'}", flush=True)
                eng = engines.make(config.OCR_ENGINE)
                eng.warm()
                eng.close()
            continue
        repo, ignore = spec
        print(f"[prefetch] {role}: {repo}", flush=True)
        path = snapshot_download(repo, ignore_patterns=ignore or None)
        print(f"[prefetch] {role} ready at {path}", flush=True)
