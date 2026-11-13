"""Compute device selection: CUDA, Apple MPS or CPU, chosen from DOCSEARCH_DEVICE (auto by default).

torch is imported only by device() and empty_cache(), so the selection logic stays testable without it.
"""
from __future__ import annotations

import os

ACCEPTED = ("auto", "cuda", "mps", "cpu")
_cached: str | None = None


def requested_from_env(environ) -> str:
    return (environ.get("DOCSEARCH_DEVICE") or "").strip().lower() or "auto"


def select_device(requested: str, cuda_available: bool, mps_available: bool) -> str:
    r = (requested or "").strip().lower() or "auto"
    if r not in ACCEPTED:
        raise ValueError(f"unknown device {requested!r}; expected one of {', '.join(ACCEPTED)}")
    if r == "auto":
        return "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if r == "cuda" and not cuda_available:
        raise RuntimeError("DOCSEARCH_DEVICE=cuda but CUDA is not available")
    if r == "mps" and not mps_available:
        raise RuntimeError("DOCSEARCH_DEVICE=mps but Apple MPS is not available")
    return r


def device() -> str:
    global _cached
    if _cached is None:
        import torch
        _cached = select_device(requested_from_env(os.environ), torch.cuda.is_available(), torch.backends.mps.is_available())
    return _cached


def empty_cache() -> None:
    import torch
    d = device()
    if d == "cuda":
        torch.cuda.empty_cache()
    elif d == "mps":
        torch.mps.empty_cache()


def is_discrete() -> bool:
    """True when the GPU has its own memory, so models are moved on and off it around each use."""
    return device() == "cuda"
