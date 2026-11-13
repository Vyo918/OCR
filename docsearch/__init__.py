import os

# fewer out-of-memory failures on 12 GB cards: the 4B LLM plus a long context leaves little headroom for allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# ops unsupported on Apple MPS fall back to CPU instead of crashing; no effect on CUDA
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
