# Models: tested, judged, and planned

One place for every model this project has run or is considering, with the result and the judgement. Details of the OCR bake-off live in [MODEL_SELECTION.md](MODEL_SELECTION.md); the three-question development run is in [BENCHMARK.md](BENCHMARK.md). This file adds the answer-quality work and the list of models still to test.

**Status legend:** TESTED = run here, numbers below are mine. NOT TESTED = candidate only; every claim about it is from a vendor card or third-party report and is unverified by this project.

## 0. How to read the evidence

- Two scanned drawings (one A3 page each), one reviewer, no independent validation.
- OCR scores use a 115-string ground truth read by eye from zoomed crops (40 marked CRITICAL).
- Answer quality uses a private 155-question Chinese benchmark file (not committed). Experiments use a stratified 40-question subset (`--subset 40 --seed 1`). **I graded these 40 answers by hand**; the automatic grader in `docsearch/benchjson.py` is a conservative guard (it never passed a wrong answer, but it rejected many correct ones), so hand grading is the number that counts.
- 40 questions is small. A repeat of the v2 run with the same index and the same sampling seed returned byte-identical answers for all 40 questions (the pipeline is deterministic for a fixed seed), so it measured nothing about noise. Between the original and the re-ordered index (different chunks, so different prompts) 4 questions flipped to right and 3 to wrong on the 38 development questions (hand labels); that is the only noise-like evidence so far. Run-to-run sampling noise needs runs with different seeds (`DOCSEARCH_SEED`). Until measured, treat a difference below about 4 questions net as noise.
- Several benchmark questions were seen during development (prompts, chunking, calculator were tuned). This is not held-out accuracy.
- Timings were taken on a shared RTX 3060 12 GB and on an Apple M4 Mac mini 24 GB shared with other services. They are indicative.

## 1. OCR engines (TESTED)

Scores: both sheets, 115 GT strings, 40 CRITICAL; "digit%" = numbers in the GT string found intact. Full tables, protocol and caveats in MODEL_SELECTION.md.

| Model | hit % | CRITICAL digit % | seconds (2 sheets) | Judgement |
|---|---|---|---|---|
| PP-OCRv5 server | 73.9 | 91.4 | 318 | Solid baseline, weaker than v6 on digits. |
| PP-OCRv5 2x zoom | 49.6 | 83.0 | 1 071 | **Rejected.** Upscaling hurt hits and was 3x slower. |
| PP-OCRv5 unclip 2.0 | 70.4 | 89.3 | 359 | Rejected, worse than plain v5. |
| PP-OCRv6 medium | 74.8 | 92.5 | 433 | Good. |
| **PP-OCRv6 medium, unclip 2.0** | 75.7 | **96.7** | **261** | Best single engine: best digits, fastest. Fallback engine. |
| PP-OCRv6 medium, unclip 2.5 | 76.5 | 93.9 | 402 | Slightly more hits, fewer intact digits; not chosen. |
| PaddleOCR-VL-1.6 (0.9B VLM) | 55.6 (sys sheet) | 91.5 (sys) | 2 209 (sys sheet only) | **Rejected.** Built for page parsing; on dense CAD sheets it missed text, 0/7 on the pipe-label band, very slow. |
| HunyuanOCR (1B VLM) | 67.9 (sys) | 91.7 (sys) | 1 536 (sys sheet only) | Poor on tables and formulas, **best on tiny coloured pipe labels** (5/7 and 92.3 % digits on the band). Slow, can loop, writes Simplified script, changes words (連通管 to 通氧管). |
| **Hybrid: PP-OCRv6-m-u20 + HunyuanOCR on unsure tiles** | 79.1 | 95.3 | 1 045 | **Current default.** Raft hit 55.9 to 67.6 %, pipe-label band 1/7 to 4/7. Adverse: CRITICAL digits fell 96.7 to 95.3 overall (86.4 on raft). Costs ~858 s on the raft sheet and +4.3 GB weights. Defensible, not proven: the digit regression is unexplained. |

Judgement on OCR as an answer bottleneck: **not the main one.** In my check of 16 wrong answers (hand-graded run), the correct value was already present in the OCR text for 11 of them. The rest are computed values or counts that need arithmetic. Better OCR will not by itself raise the answer rate; retrieval, reading and arithmetic will.

## 2. Answering LLM (TESTED, one model)

| Model | Role | Result | Judgement |
|---|---|---|---|
| **Qwen3-4B-Thinking-2507**, bf16, `transformers` | Reads top reranked chunks as text; answers in Traditional Chinese; a calculator step handles dimension arithmetic | Hand-graded on the 40-question subset: **23 correct, 1 partial, 16 wrong (58 %)**, in both the original index and the re-ordered index ("v2"). Latency: mean 209 s/question (original) and 187 s (v2) on the RTX 3060 with cached OCR; 462 s through the CLI on the Mac mini (15.2 GB peak); 10 to 15 min through the web service under memory pressure. Thinking length: 934 tokens on a "not found" question, 4 659 and 5 567 on the calculation questions. | Works for lookups, correctly refuses when the drawing lacks the item, adequate on unit arithmetic. Fails on: (a) picking the wrong one of near-identical tanks or rows, (b) counting, (c) saying NOT FOUND when the value is in the text, (d) reading a wrong number, (e) sums across both sheets. Too slow for interactive use. **Accuracy is the limit I am working on, so a stronger reader is the first thing to test.** |

The automatic grader scored the same two runs at 16 and 13 passes; it wrongly failed 8 and 11 correct answers. This is why hand grading is the reference.

On the three original development questions the final run (hybrid OCR, same LLM) satisfied all coarse fact checks (1/1, 3/3, 2/2; no forbidden hits). Earlier configurations did worse on the same checks: PP-OCR-only index 1/1, 0/4, 0/3; hybrid before prompt/chunk tuning 1/1, 0/4, 2/3. The final checks were reached **after tuning on these very questions**, so they show the system can be made to pass them, not that it generalises.

Two answering LLMs have been run: Qwen3-4B (the fixed reader for pipeline tests) and Qwen3.5-9B 4-bit (section 5).

Reader decision: with fact rows the 9B was no better than the 4B and took about 1.2 times as long per question (270 against 225 s on different machines; 343 against 185 s without fact rows, section 5), so Qwen3-4B stays the fixed reader for pipeline tests. The remaining failures are retrieval and data, not reader size.

## 3. Retrieval stack (TESTED as a fixed stack, not compared)

| Model | Role | Evidence | Judgement |
|---|---|---|---|
| `BAAI/bge-m3` (dense) + BM25 | candidate retrieval over layout chunks | In use in every run. Not A/B-tested against another embedder. | Keep. Swap test listed in section 4. |
| `BAAI/bge-reranker-v2-m3` | rerank candidates | In use in every run. Not A/B-tested. | Keep. A third-party report (simplified-Chinese corpora, GPU, not drawings) favours it over Qwen3-Reranker-0.6B. |

Chunking experiment (same 40 questions, same LLM):

| Variant | Chunks | Hand-graded | Judgement |
|---|---|---|---|
| v1 chunking | 662 (joint index) | 22 correct | Had a real bug: block grouping linked a line to the one two lines away across an overlapping neighbour, so a value could land under the wrong heading (effluent BOD limit under the influent heading). |
| v2 reading-order fix | 1 246 | 23 correct | Fixes the bug (the affected question, Q020, flipped to correct) but the total is nearly unchanged: 4 questions flipped to right (including Q020) and 3 flipped to wrong on the 38 development questions. **Kept** because the bug was real; net effect on accuracy is +1, within noise. |

Optional answer check (off by default, `DOCSEARCH_VERIFY=1`): the same 40 v1.4 questions scored 31 / 0 / 9 with one retry versus 33 / 1 / 6 without it. It flagged and retried 30 answers, corrected no wrong answer, and caused three regressions; mean time rose from 195 to 346 s. The net -2 is within the roughly two-question seed noise, but the adoption gate was not met, so keep it off.

## 4. Models under consideration (NOT TESTED)

Sources and sizes are from a research pass over model cards, Hugging Face file trees and community benchmarks; none of it was run here. Mac budget: about 12 to 14 GB usable (24 GB machine shared with other services). 3060 budget: 12 GB VRAM.

### 4.1 Answering / reading LLMs

| # | Model (runtime) | Size | Why consider | Fit | Priority |
|---|---|---|---|---|---|
| 1 | `mlx-community/Qwen3.5-9B-4bit` (MLX); same weights as GGUF `unsloth/Qwen3.5-9B-GGUF` Q4_K_M + mmproj on llama.cpp for the RTX 3060 | 5.95 GB (GGUF 5.68 + 0.92) | Newer family; text and image in one model; vendor card lists OCRBench 89.2, CC-OCR 79.3, MMLongBench-Doc 57.7, CharXiv-RQ 73.0 (above Qwen3-VL-30B-A3B on the same card); hybrid attention keeps the long-context KV cache small, so feeding the whole sheet is feasible | Third-party: ~21 tok/s at 1k and ~16 tok/s at 16k on an M4 16 GB, peak 9.1 GB at 16k; 57 tok/s and 5.5 GB on a 3060 | **First** |
| 2 | `mlx-community/Qwen3.5-9B-8bit` | 10.43 GB | Quantisation check | Borderline, context up to 16k only | After #1 |
| 3 | `mlx-community/Qwen3.5-4B-4bit` | 3.03 GB | Cheap upgrade over the current 4B; ~38 tok/s on M4 24 GB | Fits | After #1 |
| 4 | `mlx-community/Qwen3-VL-8B-Instruct-4bit` (or `-Thinking-4bit`) | 5.76 GB | Vision baseline from the previous family | Fits | Control |
| 5 | `mlx-community/Qwen3-14B-4bit` | 8.31 GB | Larger text-only control; ~10 tok/s | Fits only up to ~16k context | Control |
| 6 | `mlx-community/gemma-4-12B-it-4bit` | 6.74 GB | Different-family control; conflicting Traditional-Chinese reports, so do not exclude on them | Fits; ~14 tok/s at 1k, thermal pressure seen | Control |

**Ruled out for memory** (do not fit the shared Mac, and not the 3060 either): Qwen3-30B-A3B (17.2 GB), Qwen3-VL-30B-A3B (18.3 GB), Qwen3.5-35B-A3B (20.4 GB). Qwen3.6 exists only as 27B and 35B-A3B.
**Rejected on evidence** for numeric work: Breeze2-8B (older, no numeric-reasoning evidence), Gemma-3-TAIDE-12b-Chat-2602 (a third-party test scored 42 % on calculation, below base Gemma 3 12B at 50 %).
**Researched and not recommended:** Phi-4-mini, SmolLM3, Llama 3.2 (English-centred; SmolLM3 GPQA-Diamond 41.7 and Phi-4-mini-flash 45.1 against 76.2 for Qwen3.5-4B, vendor figures); Granite 4.2 8B and Ministral 3 8B (about 5.2 to 5.4 GB, Apache-2.0, no Chinese listed: control only, low priority). Not researched in depth: GLM-4, DeepSeek-R1 distills, InternVL3.

Small-model reading at long context degrades sharply: on NoLiMa (Adobe, ICML 2025), Gemma 3 12B scores 74.7 / 27.4 / 7.3 at 1K / 8K / 32K tokens and Llama 3.1 8B 65.7 / 31.9 / 14.2. No independent result exists for Qwen3.5. This is why the whole-sheet arm (about 33k tokens) is demoted to an escalation path and a generous top-k (20 to 30 chunks, 6 to 10k tokens) is tested instead.

No Taiwan-specific benchmark number exists for Qwen3.5-9B; it has to be measured on our set.

### 4.2 Retrieval

| Model | Why | Priority |
|---|---|---|
| `mlx-community/Qwen3-Embedding-0.6B-8bit` (0.63 GB) replacing bge-m3, keeping the reranker | A third-party test on simplified Chinese retrieval favours it by 3.9 nDCG on T2Retrieval and ties on DuRetrieval; one-hour A/B | Low |
| Qwen3-Embedding-4B | Same test: +3.8 and +5.7 over the 0.6B on the two sets; about 2.5 GB at 4-bit [INFERENCE] | Low, after the 0.6B |
| Qwen3-Reranker-4B (0.6B scored below bge-reranker-v2-m3 on Chinese in the same test; 4B has no Chinese test found) | Candidate to replace bge-reranker-v2-m3. Pitfall: clip the document before applying the template, or nDCG falls (84.3 to 68.3 reported) | Low |
| `jina-reranker-m0`, `jina-reranker-v3`, `jina-embeddings-v5` | CC-BY-NC licence | Not recommended |
| ColQwen2.5, ColPali, VisRAG, M3DocRAG, ViDoRAG, VDocRAG (page-level visual retrieval) | They choose which page to read; we have two pages. Our failures are chunk choice, counting and digit reading | Skipped on purpose |

### 4.3 OCR / vision readers

| Model | Why | Priority |
|---|---|---|
| Qwen3.5-9B (native vision) or Qwen3-VL-4B / MiniCPM-V 4.5 as a **crop re-reader**: crop the OCR line boxes of the retrieved chunk at native resolution and let the VLM read them | A preprint on large-format drawings (LED-Bench, arXiv 2610.19571) reports crops lifting accuracy 36 to 49 %; on MechVQA (arXiv 2605.30794) MiniCPM-V 4.5 scored 60.5, Qwen3-VL-4B 60.2 and InternVL3.5-8B 49.8. Different benchmarks, preprints; treat as a hypothesis. Only helps if the fact is not already in the OCR text, which for most of our wrong answers it is | IMPLEMENTED, NOT EVALUATED (`DOCSEARCH_REREAD`, default off) |
| dots.mocr (3B, MIT), olmOCR-2-7B (English-focused) | Other OCR specialists; no drawing evidence; MLX support unverified | Low |
| PaddleOCR-VL-1.6 in place of HunyuanOCR on unsure tiles | Already measured worse in spotting mode; only worth a retry in a different mode | Very low |
| Agentic zoom models (DeepEyes, ZoomEye) | Trained on natural images | Skipped on purpose |

### 4.4 Structured extraction (lookup, count, sum by code)

| Model | Why | Priority |
|---|---|---|
| Qwen3.5-9B / 4B | Native tool calling, BFCL-V4 66.1 and 50.3 (vendor); no independent Chinese function-calling evidence | With the extraction step |
| MiniCPM5-2B (1.56 GB, Apache-2.0, English and Chinese, BFCL v4 66.6 self-reported) | Small candidate extractor | Low |

The extraction step is mostly code (regex over the calculation tables and labels, with the LLM only as fallback), so model choice matters less than the schema.

### 4.5 Test protocol for the candidates

1. Freeze the top-k chunks of the 40 questions once and reuse them for every model. Same prompt, same calculator, same thinking cap. Only the model changes.
2. Arms: **A** frozen top-12 chunks as text; **B** generous top-k (30 chunks, `DOCSEARCH_TOP=30`), later grouped by sheet and section; **B2** whole-sheet text only as an escalation when the answer is NOT FOUND; **C** chunks plus image crops (vision models only); **D** crops only.
3. Hand-grade every answer. Record failure class (a)-(e), false refusals, thinking tokens, wall time, peak memory.
4. Count a model as better only on a net gain of 4 or more questions on the same 40 (or a paired test on the 13 persistent failures). Repeat the winner on the other machine to check that the two runtimes and quantisations agree.
5. Run the broad sweep on the RTX 3060 (about 2 000 tok/s prompt processing against 83 to 160 on the M4), confirm the winner on the Mac.

## 5. Open items

Results on the retained questions, hand-graded by one reviewer. Benchmark v1.1 gave 38 retained questions; v1.2 removed Q143 (37 left) and added Q157, Q158 and Q160; v1.3 replaced Q160 with Q162 after Q160 proved ambiguous (the legend says 76 diffusers, the calculation sheet says 76 including 4 in the disinfection tank). The retained set is now **40 questions**: the 37 plus Q157, Q158, Q162. The three added questions were run afterwards on every configuration with the same settings: Q158 and Q162 are answered correctly by all seven configurations, Q157 (pump A bore) is wrong in all seven because OCR reads the 3-inch bore as 5.

Per-question hand labels were reconstructed cell by cell on 2026-11-17 and checked against the reviewer's sheets for Q040 to Q045, Q050, Q101, Q113, Q115, Q125, Q136, Q154, Q155 and Q157. The recount differs from the earlier totals by up to 2 correct per run, because partial credit is now applied consistently: a partial is an answer where the right value appears only among several unselected alternatives, or the verdict is right but supporting values are wrong, or the answer is incomplete.

The 37 older questions are the 40 retained minus Q157, Q158 and Q162. The 38-question development set used in section 3 and for the fact-row gains below adds Q143.

Benchmark v1.4 (2026-11-17) corrected references for Q036 and Q044 and reworded Q043 and Q045. Earlier answers to the reworded questions are excluded; all other rows are counted on the v1.4 basis. Across all runs, 71 of 150 questions have never been tested.

| Configuration | v1.4 tested | Correct / partial / wrong |
|---|---:|---|
| Qwen3-4B top-12, seed 0 (v2 order) | 38 | 25 / 1 / 12 |
| Qwen3-4B top-12, seed 1 | 38 | 26 / 1 / 11 |
| Qwen3-4B top-30, seed 0 | 38 | 23 / 2 / 13 |
| Qwen3.5-9B 4-bit top-12 (Mac mini) | 38 | 25 / 5 / 8 |
| Qwen3.5-9B 4-bit top-30 (Mac mini) | 38 | 22 / 5 / 11 |
| Qwen3-4B top-12 + fact rows (earlier code) | 38 | 30 / 1 / 7 |
| Qwen3.5-9B 4-bit top-12 + fact rows | 38 | 29 / 3 / 6 |
| **Current code, fact rows on, v1.4** | **40** | **33 / 1 / 6** |

Fresh-question pair (40 questions never used in development, 4B top-12, seed 0, RTX 3060; 39 compared because Q141 was removed from the benchmark between the two arms):

| Arm | Correct / partial / wrong | Seconds per question |
|---|---|---|
| Fact rows off | 30 / 1 / 8 | 133 |
| **Fact rows on** | **34 / 0 / 5** | 150 |

Fact rows won on Q052, Q057, Q084, Q123 (wrong to correct) and Q028 (partial to correct) and lost Q082 (the rows led it to 0.21 m3/min at 4.2 mH). Net +4 correct, the gate set beforehand (net +4 and no refusal question turning wrong) is met, so **`DOCSEARCH_FACTS` now defaults to on** (`DOCSEARCH_FACTS=0` turns it off).

- Sampling noise is about 1 to 2 questions; more chunks (top-30) and the larger reader (9B) did not give a clear gain. On 38 comparable v1.4 questions, the earlier 4B fact-row run scored 30 / 1 / 7 and the 9B scored 29 / 3 / 6.
- The development-set gain and the fresh-pair gain point the same way; each is a single reviewer on a small set, so treat the size of the gain as approximate.
- The row-selection fix gained Q113 and Q115. Current code remains wrong on Q037, Q040, Q041, Q043, Q136 and Q157, with Q101 partial. On the fresh set: Q036, Q051, Q082, Q083, Q109.
- Nothing else in section 4 has been run.

## 6. Current plan and status

Machines: the Mac mini (Apple M4, 24 GB) is the default deployment machine; the RTX 3060 is the benchmark workstation for quality tests. Timings on both are indicative (section 0).

Running:

- Timed build on the Mac mini, with the Q058 trace: upload-to-ready and question-to-answer times.

Planned:

- Answer check with one retry: measured on the 40 retained questions; the adoption gate was not met, so it remains off (section 3).
- Crop re-read with a vision model (section 4.3): design done. Digit misreads explain only the two pump-bore questions (Q136, Q157), so the realistic ceiling is +2, exactly the gate. Build, then a model-free precheck, then decide on the 40-question run. Reader: Qwen3-VL-4B-Instruct, control HunyuanOCR.
- Qwen3.5-9B 4-bit: one final check of the finished pipeline. Qwen3.5-9B 8-bit: not planned. Qwen3.5-4B 4-bit (MLX): default machine speed check. Embedding and reranker swaps: low priority.

The pipeline reader stays Qwen3-4B-Thinking-2507 (section 2).
