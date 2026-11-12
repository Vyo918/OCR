import os

# fewer out-of-memory failures on 12 GB cards: the 4B LLM plus a long context leaves little headroom for allocator fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
