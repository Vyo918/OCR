# OCR model selection

**Question:** which locally installable OCR model is best *for this task* - not which one tops a leaderboard.
**Short answer:** PP-OCRv6-medium (RapidOCR/ONNX, detector box padding `unclip_ratio = 2.0`) reads the dense text and the numbers best and is ~10x faster than the VLMs;
HunyuanOCR reads the tiny coloured pipe labels better but is too slow and too weak on dense tables to be the primary engine. Details, numbers and caveats below.

## 1. The task, measured on the two input drawings

| Fact (measured) | `系統平面圖、剖面圖` ("sys") | `筏基層聯通管平面圖` ("raft") |
|---|---|---|
| File | 1 page, A3, one embedded 4964x3510 px JPEG (300 dpi), no fonts / no text layer (`pdffonts`, `pdfimages`) | same |
| OCR lines (PP-OCRv6-m-u20) / characters | 391 / 5 440 | 774 / 3 953 |
| Line height (OCR box), 10th pct / median / 90th pct | 33 / 42 / 53 px | 20 / 28 / 41 px (1.7 - 3.5 mm on paper for the 10th - 90th pct) |
| Lines containing Chinese / digits | 74 % / 66 % | 38 % / 67 % |
| Content | calculation sheet (thin CAD font, formulas), equipment legend table, plan labels (green/red/blue over hatching and grid) | pump data tables, coloured pipe labels on a grid (`4"PVC連通管x1支(壁上方)`, `B.O.P+320`) |

What this asks of the OCR:

1. **Digit fidelity.** One wrong digit flips an answer (`2HP` vs `5HP`, `D320` vs `D305`, `4"` vs `6"`). Scored as *digit%* on the strings the three benchmark questions depend on ("CRITICAL").
2. **Line boxes.** Retrieval builds neighbourhood context (a label means something only next to its tank / table) and the answer is shown *on the sheet*; a plain text dump is not enough.
3. **No invented text.** Question 1 has no answer in the drawings (there is no fire pump). A model that completes a spec table from language priors manufactures one. Scored as *phantom%* (boxes on blank paper) and by reading the misses.
4. **Traditional Chinese with CAD variants** (污/汙, 沈/沉, 聯/連), mixed with ASCII units (`m3/min`, `φ50mm`, `×`, `≧`).
5. **Tiny, thin, coloured strokes** on top of line work.
6. **Runs locally**: 12 GB GPU (RTX 3060) / few CPU cores, shippable in a container, usable per file (minutes, not hours).

Research context (used to frame the task, not to rank models):
- AECV-Bench (arXiv [2601.04819](https://arxiv.org/abs/2601.04819)): on architecture/engineering drawings, OCR and text-centric QA are the strongest capability (up to 0.95 accuracy); symbol counting and spatial reasoning are weak (counting doors/windows 0.40-0.55). Hence this system answers from *printed labels* (`x2支`), never from counting drawn symbols.
- Bridge-drawing OCR study ("Optimizing 2D bridge engineering drawing digitization: a comparative study of text recognition tools...", Results in Engineering, 2026, [S2590123026012211](https://www.sciencedirect.com/science/article/pii/S2590123026012211)): compares conventional OCR and vision-language tools on engineering drawings, i.e. the same family question as here. Publisher page is behind a bot wall; I could confirm the title and scope but not re-read its numbers, so no figure from it is used below.
- OmniDocBench (arXiv [2412.07626](https://arxiv.org/abs/2412.07626)) ranks *document parsing* over nine document sources (academic papers, textbooks, handwritten notes, dense newspapers, ...). Its abstract lists no engineering-drawing category, so its ranking is a hypothesis for this task, not evidence.

## 2. Candidates and what each was built for

| Candidate | Built for | Why it is / is not a fit here | Source |
|---|---|---|---|
| **PP-OCRv5 server** (det + rec, ONNX) | Text detection + line recognition; one 5M-class pipeline claimed to rival billion-parameter VLMs on OCR | Line boxes by construction, no autoregressive hallucination, CPU/GPU, Traditional Chinese in the unified recogniser | arXiv [2603.24373](https://arxiv.org/abs/2603.24373) |
| **PP-OCRv6 medium** (det + rec, ONNX) | Same family, newer backbone (PPLCNetV4), 50 languages, a Traditional-Chinese recognition setting (`LangRec.CHINESE_CHT` in RapidOCR 3.10); loss and neck redesigned for small / dense text | Best match on paper: line-level, small dense text, faithful reproduction; vendor claims it beats VLMs on hallucination ("VLMs introduce hallucinated corrections based on linguistic priors") | [PaddleOCR PP-OCRv6 docs](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-OCRv6/PP-OCRv6.html) |
| **PaddleOCR-VL-1.6** (0.9B VLM) | Whole-page *document parsing* (layout, tables, formulas, reading order); #1 on OmniDocBench-type boards. Used here in its text-spotting prompt, 640 px tiles, 2x zoom | Designed for pages of running text, not 4000-px CAD plots with hundreds of 20-px labels; slow (autoregressive) | arXiv [2510.14528](https://arxiv.org/abs/2510.14528) |
| **HunyuanOCR** (1B VLM) | End-to-end OCR VLM that explicitly includes text *spotting* (text + boxes), parsing, IE | Closest VLM to this task (spotting), but autoregressive: slow, can drop / repeat text at tile edges | arXiv [2511.19575](https://arxiv.org/abs/2511.19575) |
| not run: general VLMs (Qwen-VL class), GOT-OCR, olmOCR, dots.ocr | page-to-markdown / chat | same family as PaddleOCR-VL (generative, page-level), larger or slower, no boxes | - |
| not run: Tesseract / EasyOCR | classic / older engines | excluded to keep the bake-off to current models; not measured here | - |

## 3. Protocol

`python -m docsearch.modelsel run <candidate>` / `score` / `zone` (code in `docsearch/modelsel/`).

- **Ground truth:** `docsearch/modelsel/gt.py` - 115 text lines (81 sys, 34 raft) that I read by eye from 3-4x zoomed crops of the scans. 40 are CRITICAL (the digits decide one of the three questions). Single annotator, no second pass: a few GT strings may themselves contain a misreading.
- **Matching:** each GT string is searched in the engine's OCR text of the same sheet (single lines plus concatenations of up to 4 neighbouring lines, after width/case/quote normalisation) with a fuzzy best-alignment.
  *hit* = at least 85 % of GT characters recovered; *char%* = mean character accuracy of the best alignment; *digit%* = share of the numbers in the GT string found intact in that alignment; *exact* = verbatim.
  Matching is sheet-wide, not position-checked, so a label that is repeated on the sheet (`B.O.P+320`) is credited from any occurrence. This flatters every engine equally; the **zone** test below is position-restricted.
- **Same page-handling for every engine** (`ocr_ingest.read_page`): overlapping tiles, containment-aware merge of duplicates. PP-OCR: 1280 px tiles / 160 px overlap; VLMs: 640 px tiles / 120 px overlap / 2x zoom.
- **Variants:** `unclip` = detector box padding (`Det.unclip_ratio`, default 1.6 -> 2.0 / 2.5, because the default clipped the first characters of some thin-font lines); `-x2` = 2x upscaled tiles.
- **Timings are upper bounds:** several runs shared one RTX 3060 and a few CPU cores at the same time; VLMs ran through `transformers.generate` (no vLLM).

## 4. Results

### 4.1 Both sheets, PP-OCR family (115 GT strings, 40 CRITICAL)

| Candidate | exact % | hit % | char % | digit % | CRITICAL hit % | CRITICAL digit % | lines | phantom % | seconds (2 sheets) |
|---|---|---|---|---|---|---|---|---|---|
| ppocrv5 (server) | 38.3 | 73.9 | 89.0 | 89.5 | 77.5 | 91.4 | 1 177 | 0.1 | 318 |
| ppocrv5-x2 (2x zoom) | 24.3 | 49.6 | 83.1 | 84.9 | 50.0 | 83.0 | 1 396 | 0.3 | 1 071 |
| ppocrv5-u20 | 38.3 | 70.4 | 88.5 | 87.7 | 75.0 | 89.3 | 1 089 | 0.2 | 359 |
| ppocrv6-m | 45.2 | 74.8 | 90.5 | 90.8 | 72.5 | 92.5 | 1 173 | 0.1 | 433 |
| **ppocrv6-m-u20** | 43.5 | 75.7 | 90.4 | **94.0** | 72.5 | **96.7** | 1 165 | 0.1 | **261** |
| ppocrv6-m-u25 | 48.7 | 76.5 | 90.9 | 92.5 | 77.5 | 93.9 | 1 119 | 0.3 | 402 |

Oracle union of the six PP-OCR variants: hit 80.0 %, char 93.0 % - most remaining misses are *systematic* (tiny coloured labels), not random.

### 4.2 The sys sheet, all four families (81 GT strings, 25 CRITICAL)

| Candidate | exact % | hit % | char % | digit % | CRITICAL hit % | CRITICAL digit % | lines | phantom % | seconds |
|---|---|---|---|---|---|---|---|---|---|
| ppocrv5 | 40.7 | 82.7 | 91.4 | 93.6 | 96.0 | 93.6 | 428 | 0.0 | 139 |
| ppocrv6-m | 48.1 | 84.0 | 93.3 | 93.3 | 92.0 | 95.2 | 393 | 0.0 | 190 |
| **ppocrv6-m-u20** | 45.7 | 84.0 | 93.1 | 94.8 | 92.0 | **99.2** | 391 | 0.0 | 107 |
| ppocrv6-m-u25 | 54.3 | 86.4 | 93.7 | 92.8 | 96.0 | 95.2 | 385 | 0.5 | 118 |
| hunyuan | 42.0 | 67.9 | 89.4 | 90.4 | 68.0 | 91.7 | 580 | 0.0 | 1 536 |
| paddlevl | 33.3 | 55.6 | 85.8 | 86.6 | 60.0 | 91.5 | 602 | 0.2 | 2 209 |

Per category (hit % | digit %), sys sheet:

| Candidate | legend table | calc sheet | plan labels |
|---|---|---|---|
| ppocrv5 | 100.0 \| 92.9 | 88.2 \| 93.8 | 66.7 \| 93.8 |
| ppocrv6-m-u20 | 94.1 \| 100.0 | 85.3 \| 94.9 | 76.7 \| 92.0 |
| ppocrv6-m-u25 | 100.0 \| 100.0 | 85.3 \| 93.5 | 80.0 \| 88.3 |
| hunyuan | 47.1 \| 85.7 | 55.9 \| 89.0 | **93.3 \| 94.7** |
| paddlevl | 41.2 \| 71.4 | 41.2 \| 86.4 | 80.0 \| 94.7 |

Oracle union of all six on the sys sheet: hit 93.8 %, char 97.5 % - the VLMs are right where PP-OCR is wrong (plan labels) and wrong where PP-OCR is right (tables, formulas).

### 4.3 Zone test: the pipe-label band that decides question 3 (raft sheet, x 0.28-0.52, y 0.60-0.78; 7 CRITICAL strings, position-restricted)

| Candidate | lines | hit | char % | digit % | seconds |
|---|---|---|---|---|---|
| ppocrv5 | 107 | 2/7 | 73.0 | 61.5 | 45 |
| ppocrv5-u20 | 106 | 2/7 | 73.1 | 76.9 | 22 |
| ppocrv6-m | 116 | 1/7 | 64.4 | 84.6 | 16 |
| ppocrv6-m-u20 | 112 | 4/7 | 81.4 | 76.9 | 16 |
| ppocrv6-m-u25 | 103 | 1/7 | 66.9 | 69.2 | 15 |
| paddlevl | 78 | 0/7 | 24.5 | 53.8 | 401 |
| **hunyuan** | 118 | **5/7** | **87.1** | **92.3** | 111 |

Typical PP-OCRv6 failure here: `4"PVC連通管x1支(壁上方)` comes out as `4PvC延载警x1文(信上方)` - digits and unit survive, the Chinese does not.

## 5. Decision

**Default engine: PP-OCRv6-medium with `unclip_ratio = 2.0` (`ppocrv6-m-u20`)**, because for this task:

- It has the best digit accuracy on the CRITICAL strings (96.7 % both sheets, 99.2 % on the sys sheet) - the property that decides the answers - with 0.1 % phantom boxes (no invented text).
- It is the fastest: the whole sys sheet in ~2 minutes on a few CPU cores, versus 26 min (HunyuanOCR) and 37 min (PaddleOCR-VL) on the GPU.
- It dominates on the dense legend and calculation text (the evidence for questions 1 and 2), where the VLMs lose 30-50 points of hit rate.
- The leaderboard leader (PaddleOCR-VL) is the *worst* of the four on these drawings (55.6 % hit vs 84.0 %), which is why the choice was measured instead of inherited from OmniDocBench.

**Known weakness, and what is done about it.** All PP-OCR variants read the tiny coloured pipe labels poorly (4/7 at best in the zone test; the Chinese part of `連通管` is the usual casualty). HunyuanOCR reads that band best (5/7, 92.3 % digits) and the plan labels best (93.3 %). It is available as an alternative engine (`DOCSEARCH_OCR_ENGINE=hunyuan`) but is not the default: 25+ minutes per sheet, and 47-56 % hit on tables/formulas.

## 6. Limits of this evidence

- 115 GT lines from one annotator (me), two sheets from one project. Differences of a few points between PP-OCR variants (e.g. u20 vs u25) are within noise; the family-level gaps (PP-OCR vs VLM on tables, VLM vs PP-OCR on labels) are large and consistent.
- The GT match is sheet-wide; only the zone test is position-restricted.
- VLM numbers are for one prompt and one tiling per model (640 px / 2x zoom); a different prompt or tile size could move them. PaddleOCR-VL was run in its spotting mode, not its page-parsing mode, because only spotting gives line boxes.
- Timings were measured with concurrent jobs on one GPU; relative order is reliable, absolute values are upper bounds.
- The full-page VLM runs on the raft sheet were stopped unfinished; VLM evidence for the raft sheet is the zone test only.
