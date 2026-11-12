# docsearch - ask questions about scanned drawings, locally

Give it a scanned PDF/image (drawings with Traditional Chinese + numbers, no text layer) and a question or a keyword.
It OCRs the sheet, indexes the text *with its position and neighbourhood*, answers with a local LLM that must quote the sheet, and
saves annotated crops showing **where on the sheet** the answer came from. Nothing leaves the machine. Shipped as one Docker image.

```
PDF / image ──► page image ──► OCR (PP-OCRv6 + HunyuanOCR) ──► layout-aware chunks ──► hybrid retrieval ──► Qwen3-4B-Thinking
 (scan, A3)     300 dpi          tiles + merge                  heading path,           BM25 + BGE-M3 +        grounded answer
                                                                neighbourhood           reranker               + boxed crops
```

Why this OCR engine: [MODEL_SELECTION.md](MODEL_SELECTION.md) (task analysis, 4 engine families + a cascade measured on hand-read ground truth).

## Run with Docker

Needs: Linux, NVIDIA GPU with >= 12 GB (developed on an RTX 3060), NVIDIA driver >= 560 + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Docker Compose v2, ~35 GB free disk (11.5 GB image + 16.6 GB weights + working files).

```bash
docker compose build                        # image: 11.5 GB on disk (4.0 GB as a gzipped `docker save`): python, torch+CUDA, poppler, the code; no model weights inside
./docsearch.sh prefetch                     # once: ~16.6 GB of weights (bge-m3, bge-reranker, Qwen3-4B-Thinking, HunyuanOCR, PP-OCR) into the models volume

./docsearch.sh ask -f ~/drawings/sheet.pdf "消防泵浦幾HP?" "初沉池幾槽?體積各多少?"    # OCR (cached) + answer + evidence crops
./docsearch.sh keyword -f ~/drawings/sheet.pdf 初沉池                              # keyword hits with sheet position + boxed crops
./docsearch.sh chat -f ~/drawings/sheet.pdf                                        # interactive, models stay loaded ("/kw 連通管" = keyword)
```

`-f` takes a PDF, an image, or a folder (repeatable). OCR results are cached per page and engine. In the bake-off, the sys sheet took 187 s and the raft sheet 858 s; these are indicative shared-resource measurements, not upper bounds.

The final Docker run took 671 s after reusing 835 s of cached raft OCR; it OCRed sys in 122 s and rebuilt a 662-chunk index. This is not a cold two-file time. See [BENCHMARK.md](BENCHMARK.md) for the run and qualifications.

For a two-source benchmark, repeat `-f` for each PDF:

```bash
./docsearch.sh bench -f ~/drawings/raft.pdf -f ~/drawings/sys.pdf
```
A second positional PDF is parsed as the question fixture, not another source.
Results land in `$WORK_DIR` (default `./work`): `ocr/` (lines + boxes as JSON), `index/`, `answers/<question>/` and `hits/<keyword>/` (overview + zoomed crops with red boxes), `bench/report.md`.

Useful environment variables for `docsearch.sh` / `docker compose`:

| Variable | Meaning | Default |
|---|---|---|
| `WORK_DIR` | generated files | `./work` (next to where you run the script) |
| `DOCS_DIR` | folder of PDFs for `build` | the current folder |
| `MODELS_DIR` | model weights; point at an existing `~/.cache/huggingface` to skip the download | named volume `models` |
| `HF_HUB_OFFLINE=1` | disable Hugging Face fetches (not a firewall; RapidOCR can still download missing ONNX weights) | `0` |
| `DOCSEARCH_OCR_ENGINE` | `hybrid` (default: PP-OCRv6, then HunyuanOCR only where PP-OCR is unsure), `ppocrv6-m-u20` (PP-OCR only: ~6x faster on the raft sheet, weaker on tiny pipe labels, needs 4.3 GB less weights), `ppocrv5`, `hunyuan`, `paddlevl` | see `docsearch/engines.py` |

### Shipping it to someone else

- **Image:** `docker save docsearch:latest | gzip > docsearch-image.tgz` -> on the other machine `docker load < docsearch-image.tgz`; copy `docsearch.sh` and `docker-compose.yml` (or `docker compose build` from this folder).
- **Weights, online:** `./docsearch.sh prefetch` on the other machine.
- **Weights, offline:** on a connected machine run `prefetch`, then `tar czf models.tgz -C <MODELS_DIR> .` (about 16.6 GB); on the target extract into a folder and run with `MODELS_DIR=<that folder> HF_HUB_OFFLINE=1`.

## Run without Docker

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126 && pip install -r requirements.txt && pip install --no-deps .
sudo apt install poppler-utils
docsearch prefetch
docsearch ask -f sheet.pdf "消防泵浦幾HP?"
```

## Commands

| Command | What it does |
|---|---|
| `docsearch ask [-f FILE] "question" ...` | OCR (cached) -> index -> retrieve -> answer; prints the answer, the search terms used, and writes boxed evidence crops |
| `docsearch keyword [-f FILE] TERM` | block-level keyword search (tolerates OCR noise, ignores Traditional/Simplified and case); prints sheet position and writes crops |
| `docsearch chat [-f FILE]` | interactive loop keeping the models loaded |
| `docsearch build [PDF\|DIR ...]` | OCR + index everything (no question); `ingest` = OCR only, `index` = index only |
| `docsearch bench [questions.json]` | run the reference questions, write `work/bench/report.md` (human answer next to system answer) |
| `docsearch prefetch [--only ocr embed rerank llm]` | download weights |

Offline unit tests (no GPU, no weights): `python -m unittest discover -s tests` - calculator and tank-size decoding.

## How it answers

1. **OCR** (`ocr_ingest.py`, `engines.py`): the embedded scan is reused as-is (else a 300 dpi render); the page is cut into overlapping 1280 px tiles, read by PP-OCRv6-medium (RapidOCR / ONNX), and duplicate lines from tile overlaps are merged (containment-aware). Tiles where at least two lines have a PP-OCR confidence below 0.75 (garbled tiny labels) are re-read by HunyuanOCR (`cascade.py`); its lines are added where PP-OCR found nothing or replace only the unsure PP-OCR lines. Every line keeps its box.
2. **Index** (`index.py`): lines are grouped into text blocks (size-capped union-find on box distance), headings give each block a section path, every block also gets a *neighbourhood chunk* (the block plus nearby blocks, in reading order) - a label on a plan only means something next to the tank it sits in - and every level-2 heading (`三、處理單元功能計算 > 1. 初沉池`) gets a *section chunk* holding everything under it, so one unit's design values are read together and apart from the next unit. Text is folded Traditional -> Simplified for matching (OpenCC + CAD variants such as 沈/沉, 汙/污), so `沈澱池` finds `沉淀池`.
3. **Retrieval** (`search.py`): the question is expanded with synonym groups from `glossary.json` (泵浦 / 水泵 / 幫浦; 連通管 / 聯通管 / 連接管; ...), then BM25 over character n-grams and BGE-M3 dense vectors are fused with RRF, re-ranked by BGE-reranker-v2-m3, and near-duplicate chunks are dropped so one region cannot fill the context.
4. **Answer:** Qwen3-4B-Thinking-2507 gets the top chunks and the sheet outline and must separate "stated" from "inferred". It never does arithmetic itself: it writes `{{7.9*3.525*3.2}}` and `calc.py` (a restricted evaluator, no `eval`) replaces it with `7.9*3.525*3.2 = 89.11`. If the information is not in the chunks it must say NOT FOUND instead of substituting another item (a question about a fire pump must not be answered with the sewage pump's rating).
5. **Show:** `answers/<question>/` holds an overview of each sheet with numbered red boxes round the cited lines, plus a zoomed crop per cited chunk.

## Limits (read before trusting an answer)

- OCR of the tiny coloured pipe labels on the raft sheet is the weak spot (MODEL_SELECTION.md sections 4.3-4.4): the cascade reads their sizes and counts but still garbles the words (`連通管` -> `通氧管`), and HunyuanOCR sometimes writes Simplified script. Always check the evidence crops for numbers that matter.
- Answers come from printed text only: the system does not count drawn symbols and does not read dimensions off the geometry.
- The LLM is sampled (fixed seed, temperature 0.6); wording can differ slightly between runs. A question takes minutes on an RTX 3060 (the Thinking model reasons before it answers).
- The benchmark is a small development demonstration, not statistical accuracy evidence. Human-reference wording means assistant-read visual references, not independent engineer validation; see [BENCHMARK.md](BENCHMARK.md).
- `HF_HUB_OFFLINE=1` disables Hugging Face fetches only; prefetch all weights. It does not block the network or RapidOCR missing-ONNX downloads. Use Docker network controls for actual isolation. The baked-weight image variant and cross-machine restore were not tested.
