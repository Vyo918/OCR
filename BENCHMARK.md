# OCR/QA benchmark

## Run

One completed local Docker run answered the three development questions below. The command was:

```sh
./docsearch.sh bench -f '<raft.pdf>' -f '<sys.pdf>'
```

Repeat `-f` for each source: a second positional path is parsed as the question fixture, not another PDF. The final run used `WORK_DIR=work/final`, `HF_HUB_OFFLINE=1`, and the hybrid OCR engine. It reused raft OCR from an earlier 835-second invocation, OCRed the system drawing in 122 seconds, rebuilt a joint 662-chunk index, and answered the questions. Elapsed time was 671 seconds; this is **not** a cold two-sheet ingestion time. The failed earlier invocation is not an accuracy result. No new network weights were needed.

| Question | Assistant-read visual reference | Exact system answer | Comparison; answer time |
|---|---|---|---|
| 消防泵浦幾HP? | Neither drawing specifies a fire-pump rating. Other pump ratings are not substitutes. | `消防泵浦幾HP? NOT FOUND` | Matches the drawing-based reference; no rating invented. **63 s** |
| 初沉池幾槽?體積各多少? | The calculation sheet's “三、處理單元功能計算 > 1. 初沉池” set has three tanks: dimensions in cm are L790×W352.5×H350/D320 for tanks 1–2, and L467.5×W300×H350/D320 for tank 3. Using effective depth D gives approximately 89.11, 89.11, and 44.88 m³. | `初沉池有3槽，體積分別為89.11 m³、89.11 m³、44.88 m³。` | Matches the calculation-sheet set and per-tank volumes. The sheet prints total 223.12 m³, while recomputing its dimensions yields 223.104 m³ (about 223.10); total was not asked. **212 s** |
| 初沉至調整池連通管數量?及尺寸? | For the identified initial-sedimentation-to-adjustment-pool connection/wall set: one 4-inch PVC pipe and two 6-inch PVC pipes (three total). This is not a count of all pipes in the building. | `初沉至調整池連通管數量: 1支 4\" PVC管 和 2支 6\" PVC管` | Matches the identified connection and set, not a building-wide total. Drawing elevations are supplementary, not required. **261 s** |

## Qualifications

The Q2 count is specifically the three-tank set in the calculation sheet, not an unqualified count of every similarly named plan compartment: the plan also labels 初沉池(1-4) through (1-6) in 槽(1), at H=1.85 m, without individual dimensions in that calculation set. The stated volumes use D, the effective water depth, rather than full structural H. A separate whole-plan tank total is 239.5 m³. The drawings also disagree on flow (410 vs 425 CMD) and pump B rating (2 HP in the table vs 1 HP on the plan). These ambiguities and discrepancies are not reconciled by the answers.

This was a self-run development exercise on the same two pages and three repeatedly used questions—not held-out testing, general accuracy measurement, or independent human-engineer validation. Visual references were read by the assistant from enlarged scans. After earlier failures, prompts, section chunking, tank-dimension decoding, and calculator behavior were tuned; Q2 total volume and Q3 elevation were removed from required facts after earlier runs because they were not asked. Historical failures remain separately in the raw results. The final 1+3+2 fact-presence checks and zero forbidden hits are only a coarse screen, not 100% accuracy. No independent review, preregistration, confidence intervals, aggregate accuracy, or comparable old/new rubric scores are claimed.

OCR remains imperfect: some text garbles 連通管 as 通氧管 and inch marks as degree symbols. Retrieval and OCR can miss or misread drawing content; absence claims and plan interpretation have these limits.

## Reproducibility artifacts

Raw local, ignored outputs (not committed): [final report](work/final/bench/report.md), [final results](work/final/bench/results.json), [OCR/index run log](work/final/benchmark.log), and [manifest](work/final/manifest.json). Historical QA results are retained separately under `work/bakeoff/qa_*.results.json`.

Run identity: image `sha256:1da16377d049a76eacd77aaf30c59d45c69bc3ef23cefb36937783048e603a19`; Python 3.12.16, torch 2.14.1+cu126, transformers 5.22.2, RapidOCR 3.10.0, ONNX Runtime 1.31.1, sentence-transformers 6.3.0. Source PDF SHA-256: raft `67de06a455422b65a7ff34e963224fff9e1774abb0f2a826b72538fa7c8f4674`; system `e4102346cc90a5b70708a8cdf2c17693ca93334b0d11924dd9c43d704dfe49b1`.
