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

### Local browser Q&A

The browser interface is same-origin and loopback-only; it uploads files to the
local service and does not send them to a hosted service. For Docker, start the
web service (the existing CLI service remains available separately):

```bash
docker compose up --build web
# Then open http://127.0.0.1:8000
```

The image does not contain model weights (it is `BAKE_MODELS=0` by default, to stay about 11.5 GB instead of about 28 GB). Weights download on first use into the `models` volume, about 16.6 GB in total: BGE-M3, the reranker and Qwen3-4B, plus HunyuanOCR the first time OCR runs. In testing, the first question took over 15 minutes for that reason, and the first real-drawing OCR waited on the HunyuanOCR download. Run `./docsearch.sh prefetch` once beforehand, or set `MODELS_DIR` to an existing populated Hugging Face cache, to avoid the wait.

Without Docker, install the dependencies and models as above, then run:

```bash
docsearch serve
# Then open http://127.0.0.1:8000
```

The browser accepts up to 10 PDF/image files per upload, 50 MiB per file and
100 MiB total. Supported extensions are PDF, PNG, JPG/JPEG, TIF/TIFF, BMP and
WebP. Extension checks are only preliminary; the service validates actual
files, and rejects PDFs over 20 pages and images over 100 megapixels. OCR and
indexing happen on the first upload, so processing can take minutes or longer
depending on page count, cache state and hardware. Subsequent questions are
independent, not a chat with earlier answers as context. Answers cite printed
text and may be wrong; inspect the displayed evidence crops, especially for
small labels and important numbers. The interface and its session are local;
uploaded material and generated results are retained in the configured work
area on disk, not automatically erased when the browser closes. The page history
and collection ownership exist only in the current browser session; restarting
the service loses that history and session access, but does not delete retained
files. Remove data from that work area when it is no longer needed.

### Reaching it from other devices (opt-in)

The default is loopback-only: the service answers only on `127.0.0.1` and accepts only `localhost`, `127.0.0.1` and `::1` as Host. To serve a Tailscale address, bind to that address and allow its name explicitly:

```bash
DOCSEARCH_BIND=100.104.213.53 DOCSEARCH_ALLOWED_HOSTS=100.104.213.53 docker compose up -d web
# Then open http://100.104.213.53:8000 from a tailnet device
```

Without Docker: `DOCSEARCH_ALLOWED_HOSTS=100.104.213.53 docsearch serve --host 100.104.213.53`.

There is no login. Every device that can reach that address can upload files and use the GPU. The Host and Origin checks defend against browser-based attacks (DNS rebinding, cross-site requests); they do not defend against an untrusted network peer. Bind to the tailnet address only, never `0.0.0.0`. Traffic is plain HTTP; Tailscale (WireGuard) encrypts it in transit.



## Run without Docker

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126 && pip install -r requirements.txt && pip install --no-deps .
sudo apt install poppler-utils
docsearch prefetch
docsearch ask -f sheet.pdf "消防泵浦幾HP?"
```

## Run on a Mac (Apple Silicon)

The Mac runs docsearch natively, not in Docker: Docker Desktop on macOS cannot pass the GPU to containers, and the Docker image is NVIDIA-only (CUDA). Natively, PyTorch uses Apple's MPS backend.

Requirements: Apple Silicon, macOS 15, [Homebrew](https://brew.sh), [uv](https://docs.astral.sh/uv/), about 35 GB free disk (16.6 GB weights plus the environment), 24 GB of memory recommended.

```bash
deploy/macos/install.sh                                                  # once: .venv, torch (version from the Dockerfile), poppler, weights
DOCSEARCH_BIND=100.76.130.63 DOCSEARCH_PORT=8010 deploy/macos/run.sh     # foreground server
deploy/macos/launchagent.sh install --bind 100.76.130.63 --port 8010     # start at login, restart on crash
deploy/macos/launchagent.sh status                                       # also: restart, uninstall
```

`install.sh` is idempotent and takes `--no-weights` (skip the download) and `--allow-cpu` (continue when MPS is unavailable). It never uses sudo. The LaunchAgent logs to `~/Library/Logs/docsearch/`, and its plist is `~/Library/LaunchAgents/com.docsearch.web.plist`. It runs only while you are logged in.

Device selection is automatic (CUDA, then MPS, then CPU). Set `DOCSEARCH_DEVICE=cpu` or `DOCSEARCH_DEVICE=mps` to override it. `run.sh` takes the work folder from `DOCSEARCH_WORK` (default `~/ocr-docsearch/work`).

Limits:

- There is no login. Bind only to the tailnet address (`run.sh` and `launchagent.sh` refuse `0.0.0.0` and `::`). The default allowed Host is the bind address; traffic is plain HTTP inside Tailscale's encrypted tunnel.
- `install.sh` runs `brew install poppler` only when `pdftoppm` is missing, and brew may upgrade poppler's dependency formulae at that point.
- The first OCR on MPS is expected to be slower than on the NVIDIA GPU. The speed on this hardware is not measured yet.

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
