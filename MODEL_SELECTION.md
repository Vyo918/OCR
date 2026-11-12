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
| **PaddleOCR-VL-1.6** (0.9B VLM) | Whole-page *document parsing* (layout, tables, formulas, reading order); its paper reports state of the art on page-level parsing benchmarks. Used here in its text-spotting prompt, 640 px tiles, 2x zoom | Designed for pages of running text, not 4000-px CAD plots with hundreds of 20-px labels; slow (autoregressive) | arXiv [2510.14528](https://arxiv.org/abs/2510.14528) |
| **HunyuanOCR** (1B VLM) | End-to-end OCR VLM that explicitly includes text *spotting* (text + boxes), parsing, IE | Closest VLM to this task (spotting), but autoregressive: slow, can drop / repeat text at tile edges | arXiv [2511.19575](https://arxiv.org/abs/2511.19575) |
| **Cascade `hybrid`** (this project: PP-OCRv6-m-u20, then HunyuanOCR where PP-OCR is unsure) | Built from the measured failure profiles in section 4: PP-OCR is right on dense text and digits, wrong on tiny coloured labels; the VLM is the reverse. PP-OCR's own line confidence separates the two cases (see `docsearch/cascade.py`) | Keeps PP-OCR output wherever it is confident (no VLM invention on tables / formulas), pays the VLM's ~15 s per 640 px tile only on tiles holding at least two lines scored below 0.75 | - |
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

### 4.1 Both sheets, PP-OCR family and the cascade (115 GT strings, 40 CRITICAL)

| Candidate | exact % | hit % | char % | digit % | CRITICAL hit % | CRITICAL digit % | lines | phantom % | seconds (2 sheets) |
|---|---|---|---|---|---|---|---|---|---|
| ppocrv5 (server) | 38.3 | 73.9 | 89.0 | 89.5 | 77.5 | 91.4 | 1 177 | 0.1 | 318 |
| ppocrv5-x2 (2x zoom) | 24.3 | 49.6 | 83.1 | 84.9 | 50.0 | 83.0 | 1 396 | 0.3 | 1 071 |
| ppocrv5-u20 | 38.3 | 70.4 | 88.5 | 87.7 | 75.0 | 89.3 | 1 089 | 0.2 | 359 |
| ppocrv6-m | 45.2 | 74.8 | 90.5 | 90.8 | 72.5 | 92.5 | 1 173 | 0.1 | 433 |
| **ppocrv6-m-u20** | 43.5 | 75.7 | 90.4 | **94.0** | 72.5 | **96.7** | 1 165 | 0.1 | **261** |
| ppocrv6-m-u25 | 48.7 | 76.5 | 90.9 | 92.5 | 77.5 | 93.9 | 1 119 | 0.3 | 402 |
| hybrid (PP-OCRv6-m-u20 + HunyuanOCR cascade) | 43.5 | 79.1 | 91.6 | 94.5 | 82.5 | 95.3 | 1 524 | 0.1 | 1 045 (raft 858, sys 187; shared GPU / CPU) |

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

Two ways of reading the band, because a full-page VLM run was not affordable for every candidate:

**(a) Band re-read** - the candidate reads only the cropped band (same tiling / merge code as ingestion):

| Candidate | lines | hit | char % | digit % | seconds |
|---|---|---|---|---|---|
| ppocrv5 | 107 | 2/7 | 73.0 | 61.5 | 45 |
| ppocrv5-u20 | 106 | 2/7 | 73.1 | 76.9 | 22 |
| ppocrv6-m | 116 | 1/7 | 64.4 | 84.6 | 16 |
| ppocrv6-m-u20 | 112 | 4/7 | 81.4 | 76.9 | 16 |
| ppocrv6-m-u25 | 103 | 1/7 | 66.9 | 69.2 | 15 |
| paddlevl | 78 | 0/7 | 24.5 | 53.8 | 401 |
| hunyuan | 118 | 5/7 | 87.1 | 92.3 | 111 |

**(b) In situ** - the lines of the *stored full-page run* that lie inside the band (`zone --from-run`), i.e. what ingestion of the whole sheet really delivers there:

| Candidate | lines | hit | char % | digit % |
|---|---|---|---|---|
| ppocrv5 | 106 | 2/7 | 76.6 | 69.2 |
| ppocrv6-m | 115 | 1/7 | 69.8 | 76.9 |
| ppocrv6-m-u20 | 114 | 1/7 | 67.8 | 84.6 |
| ppocrv6-m-u25 | 107 | 2/7 | 76.1 | 84.6 |
| **hybrid (cascade)** | 154 | **4/7** | **82.6** | **92.3** |

Reading the two tables together:

- The same engine (`ppocrv6-m-u20`) scores 4/7 as a band re-read and 1/7 in situ: only the tile alignment changed. Differences between PP-OCR variants in this band are therefore **tile-alignment noise**; what is consistent is that every PP-OCR variant is weak here (1-2/7 in situ, 61-85 % digits) and that the VLM (re-read) and the cascade (in situ) are clearly ahead.
- Typical PP-OCRv6 failure: `4"PVC連通管x1支(壁上方)` comes out as `4PvC延载警x1文(信上方)` - digits and unit survive, the Chinese does not, and the line confidence is low (0.58-0.73; correct labels score 0.83-0.97). That is the signal the cascade uses.
- HunyuanOCR gets the sizes and counts, but not the words: it reads the same label as `4°PVC,通氧管×1支(图上方)` (`連通管` -> `通氧管`, `壁` -> `图`, inch mark -> degree sign, Traditional -> Simplified). Counts it read in the band: `4"x1` six times and `4"x3` once; `6"x2` seven times and `6"x1` three times - 4 of 17 lines deviate. PP-OCRv6 parsed a `6"` + `x2` label zero times and `4"x1` three times. An answer must therefore be a *vote over the repeated labels*, which is how the retrieval chunks (one per wall) are used.

### 4.4 The cascade (`hybrid`) on whole sheets

`docsearch/cascade.py`: PP-OCRv6-m-u20 reads the sheet; HunyuanOCR (640 px tiles, 120 px overlap) re-reads only the tiles that hold at least 2 lines with PP-OCR confidence below 0.75; a VLM line is added where PP-OCR has nothing under it, replaces PP-OCR lines that are all unsure, and is dropped where a confident PP-OCR line already covers it.

| Sheet | PP-OCR lines | below 0.75 | VLM tiles | VLM lines added / replacing / dropped | seconds (PP-OCR part) |
|---|---|---|---|---|---|
| raft | 774 | 226 | 27 of 70 | 347 / 147 / 244 | 858 (89) |
| sys | 391 | 16 | 4 of 70 | 17 / 0 / 40 | 187 (138; CPU shared with another job) |

Scores against the same 115-string GT (full table in 4.1):

| Sheet | Candidate | hit % | char % | digit % | CRITICAL hit % | CRITICAL digit % | pipe-label hit % \| digit % |
|---|---|---|---|---|---|---|---|
| raft (34 GT, 15 CRITICAL) | ppocrv5 | 52.9 | 83.4 | 76.1 | 46.7 | 86.4 | 45.5 \| 71.1 |
| | ppocrv6-m-u20 | 55.9 | 84.0 | 91.3 | 40.0 | 90.9 | 45.5 \| 89.5 |
| | ppocrv6-m-u25 | 52.9 | 84.0 | 91.3 | 46.7 | 90.9 | 45.5 \| 89.5 |
| | **hybrid** | **67.6** | **87.8** | **93.5** | **66.7** | 86.4 | **63.6 \| 92.1** |
| sys (81 GT, 25 CRITICAL) | ppocrv6-m-u20 | 84.0 | 93.1 | 94.8 | 92.0 | 99.2 | - |
| | hybrid | 84.0 | 93.1 | 94.8 | 92.0 | 99.2 | - |

- On the sys sheet the cascade changes no GT-scored string (the VLM was consulted on 4 tiles and replaced nothing): **no regression on dense text**, which is what the confidence gate is for.
- On the raft sheet it recovers 12 points of hit rate and the pipe labels; the one number that goes *down* is CRITICAL digit % (90.9 -> 86.4). That is a scoring artefact, not a misread digit: the VLM read `调整池(2-1)` correctly but in Simplified script, so the fuzzy sheet-wide match preferred the Traditional `調整池(2-2)` entry. The index folds Traditional / Simplified for matching, so retrieval is not affected, but the scorer is script-sensitive and under-credits VLM output.
- Cost: 858 s for the raft sheet instead of 154 s (PP-OCR alone), all of it VLM time on 27 tiles, plus a 4.3 GB model on the GPU. Pages PP-OCR reads confidently (sys: 4 tiles, +49 s) barely pay.

## 5. Decision

**Default engine: the cascade `hybrid` = PP-OCRv6-medium (`unclip_ratio = 2.0`) on the whole page, HunyuanOCR only on the tiles where PP-OCR is unsure.** Single-engine fallback: `ppocrv6-m-u20` (`DOCSEARCH_OCR_ENGINE=ppocrv6-m-u20`).

Why this and not one model:

- **No single model wins both halves of the task.** PP-OCR is best on dense, digit-heavy text (the legend and calculation sheet behind questions 1 and 2): 84.0 % hit and 99.2 % CRITICAL digits on the sys sheet, against 67.9 % (HunyuanOCR) and 55.6 % (PaddleOCR-VL), with 0.1 % phantom boxes. HunyuanOCR is best on the tiny coloured pipe labels behind question 3 (zone re-read: 5/7 and 92.3 % digits against 4/7 and 76.9 % for the best PP-OCR variant). The cascade takes each model where it is strong, with PP-OCR's own confidence as the switch.
- **Measured gain where it matters, no loss elsewhere.** Raft sheet: hit 55.9 -> 67.6 %, pipe-label band in situ 1/7 -> 4/7 and digits 84.6 -> 92.3 %. Sys sheet: every GT-scored string identical (the VLM was consulted on 4 tiles and replaced nothing), because lines PP-OCR is confident about are never overwritten.
- **It beats the leaderboard pick on this task.** PaddleOCR-VL leads OmniDocBench but was the worst engine here on the sys sheet (55.6 % hit) and failed the pipe-label band (0/7); that is why the choice was measured instead of inherited.

What it costs, stated plainly:

- **Time and weights.** Raft sheet 858 s against 154 s for PP-OCR alone (27 of 70 tiles go through the VLM; measured with the GPU shared with other jobs, so an upper bound); +4.3 GB of weights and the GPU. OCR is cached per page and engine, so this is paid once per sheet. A user who does not need pipe labels can take the single-engine fallback.
- **Residual errors.** The cascade reads sizes and counts on the pipe labels but not the words (`連通管` -> `通氧管`), and sometimes writes Simplified script. HunyuanOCR on its own misses a third of the dense text (67.9 % hit on the sys sheet) and can loop on a tile; the arbitration dropped 244 of its lines on the raft sheet (already covered by a confident PP-OCR line, or looped). Retrieval is script-folded and the answer is a vote over the repeated wall labels, so these do not stop question 3, but a single label should always be checked against its evidence crop.
- **The scorer under-credits it.** CRITICAL digit % on both sheets is 95.3 for the cascade against 96.7 for PP-OCR alone, because the GT match is script-sensitive (4.4); I did not adjust the scorer after seeing this.

## 6. Limits of this evidence

- 115 GT lines from one annotator (me), two sheets from one project. Differences of a few points between PP-OCR variants (e.g. u20 vs u25) are within noise; the family-level gaps (PP-OCR vs VLM on tables, VLM vs PP-OCR on labels) are large and consistent.
- The GT match is sheet-wide; only the zone test is position-restricted.
- VLM numbers are for one prompt and one tiling per model (640 px / 2x zoom); a different prompt or tile size could move them. PaddleOCR-VL was run in its spotting mode, not its page-parsing mode, because only spotting gives line boxes.
- Timings were measured with concurrent jobs on one GPU; relative order is reliable, absolute values are upper bounds.
- The full-page VLM runs on the raft sheet were stopped unfinished; VLM evidence for the raft sheet is the zone test only.
