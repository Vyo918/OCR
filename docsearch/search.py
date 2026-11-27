"""Stage 3 - retrieval + grounded answer.

  question --> drawing-vocabulary expansion
           --> BM25 (char n-grams) + BGE-M3 dense --RRF--> candidates
           --> bge-reranker-v2-m3 cross-encoder     --> top-k evidence chunks
           --> local LLM (Qwen3-4B-Thinking) answers *only* from the evidence, citing chunk ids.

`keyword()` is the plain keyword lookup (exact + OCR-tolerant) with sheet position and crops.

GPU budget (12 GB card): the LLM (8 GB) stays resident; embedder and re-ranker (1.1 GB each) are
parked on the CPU and moved to the GPU only while retrieval runs.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from . import answerfmt, calc, config, facts, factquery, verify
from .device import device, empty_cache, is_discrete, retrieval_placement
from .index import fold, norm, tokens


# Synonym groups of drawing vocabulary (docsearch/glossary.json, plain list of lists - extend it for your own trade).
GLOSSARY: list[list[str]] = json.loads((Path(__file__).with_name("glossary.json")).read_text())
CLOSING_THOUGHT = "\n\nI have to stop thinking now and write the final answer from what I have found so far.\n</think>\n\n"


def _stream(messages: list[dict], extra: dict) -> tuple[str, str, str]:
    """One streamed /chat/completions call -> (reasoning, content, finish_reason). Reasoning arrives as `reasoning_content`
    (llama.cpp) or `reasoning` (mlx_lm); a server that inlines it leaves both empty and the caller splits content."""
    from .progress import report
    body = {"model": config.LLM_ID, "messages": messages, "stream": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, **extra}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    url = f"{config.LLM_URL}/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"LLM server {url} returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
    except OSError as exc:
        raise RuntimeError(f"cannot reach LLM server at {url}: {exc}") from exc
    reasoning, content, finish, events, done = [], [], None, 0, False
    with resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            choices = json.loads(payload).get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            content.append(delta.get("content") or "")
            finish = choices[0].get("finish_reason") or finish
            events += 1
            if events % 16 == 0:
                report('reason', events, None, 'tokens')
    if not finish and not done:
        raise RuntimeError(f"LLM server {url} stream ended without finish_reason or [DONE]")
    return "".join(reasoning), "".join(content), finish

def _seed(seed: int) -> dict:
    """Seed field of an HTTP request. mlx_lm.server batches concurrent requests only when no seed is sent, and serves seeded ones one at a time, so the seed goes out only when DOCSEARCH_SEED pins it."""
    return {"seed": seed} if config.SEED_SET else {}

def _complete(prompt: str, seed: int, max_tokens: int) -> str:
    body = {"model": config.LLM_ID, "prompt": prompt, "max_tokens": max_tokens, **_seed(seed),
            "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if config.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {config.LLM_API_KEY}"
    req = urllib.request.Request(f"{config.LLM_URL}/completions", data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            return (json.load(resp)["choices"][0]["text"] or "").strip()
    except urllib.error.HTTPError as exc:
        exc.close()
        return ""
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return ""


class Engine:
    def __init__(self, index_dir: Path | None = None, only: set[str] | None = None):
        """`only`: restrict retrieval to these document names (the files the user pointed at)."""
        from rank_bm25 import BM25Okapi

        index_dir = index_dir or config.INDEX_DIR
        data = json.loads((index_dir / "chunks.json").read_text())
        keep = [i for i, c in enumerate(data["chunks"]) if only is None or c["doc"] in only]
        if not keep:
            raise SystemExit(f"none of {sorted(only or [])} is in the index ({', '.join(data['docs'])})")
        self.chunks: list[dict] = [data["chunks"][i] for i in keep]
        self.docs: dict = {d: v for d, v in data["docs"].items() if only is None or d in only}
        self.facts = facts.extract(self.chunks, self.docs)
        self.outlines: dict = {d: v for d, v in data["outlines"].items() if d in self.docs}
        self.emb = np.load(index_dir / "emb.npy").astype(np.float32)[keep]
        self.bm25 = BM25Okapi([tokens(c["text"]) for c in self.chunks])
        self._embedder = self._reranker = self._llm = None
        self.seed = config.SEED

    # ---------------------------------------------------------------- models
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            build, half = retrieval_placement(device())
            self._embedder = SentenceTransformer(config.EMBED_ID, device=build)
            if half:
                self._embedder.half()
        return self._embedder

    def reranker(self):
        if self._reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            build, half = retrieval_placement(device())
            self._reranker = CrossEncoder(config.RERANK_ID, device=build, model_kwargs={"dtype": torch.float16 if half else torch.float32})
        return self._reranker

    def llm(self):
        if self._llm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(config.LLM_ID)
            model = AutoModelForCausalLM.from_pretrained(config.LLM_ID, dtype=torch.bfloat16).to(device()).eval()
            self._llm = (tok, model)
        return self._llm

    @contextlib.contextmanager
    def on_gpu(self, model):
        """Move the LLM onto the GPU for one use, then back to host memory (CUDA only; MPS and CPU keep it in place)."""
        if not is_discrete():
            yield model
            return
        model.to(device())
        try:
            yield model
        finally:
            model.to("cpu")
            empty_cache()

    def chat(self, system: str, user: str, max_new_tokens: int | None = None, answer_tokens: int = 1500) -> tuple[str, str]:
        if max_new_tokens is None:
            max_new_tokens = config.THINK_TOKENS
        if config.LLM_BACKEND == "http":
            return self._chat_http(system, user, max_new_tokens, answer_tokens)
        if config.LLM_BACKEND != "transformers":
            raise ValueError(f"DOCSEARCH_LLM_BACKEND must be 'transformers' or 'http' ('openai' is an alias), not {config.LLM_BACKEND!r}")
        return self._chat_transformers(system, user, max_new_tokens, answer_tokens)

    def _chat_http(self, system: str, user: str, max_new_tokens: int, answer_tokens: int) -> tuple[str, str]:
        """Same (answer, reasoning) contract as the transformers path, served by an OpenAI-compatible server."""
        from .progress import report
        report('reason', 0, None, 'tokens')
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        extra = {**_seed(self.seed), "max_tokens": max_new_tokens + answer_tokens}
        if config.LLM_THINK != "default":
            extra["chat_template_kwargs"] = {"enable_thinking": config.LLM_THINK == "on"}
        reasoning, content, finish = _stream(msgs, extra)
        if not reasoning and "</think>" in content:   # server does not split reasoning out: split as the transformers path does
            head, _, content = content.partition("</think>")
            reasoning = head
        elif not reasoning and finish == "length" and config.LLM_THINK != "off":
            reasoning, content = content, ""
        reasoning, content = reasoning.replace("<think>", "").strip(), content.strip()
        if finish == "length" and not content:
            prompt = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                      f"<|im_start|>assistant\n<think>\n{reasoning}" + CLOSING_THOUGHT)
            content = _complete(prompt, self.seed, answer_tokens)
            if not content:
                follow = user + "\n\nYour earlier reasoning ran out of budget; its last part follows. Write the final answer only.\n" + reasoning[-4000:]
                _, content, _ = _stream([{"role": "system", "content": system}, {"role": "user", "content": follow}],
                                        {"chat_template_kwargs": {"enable_thinking": False}, "max_tokens": answer_tokens})
                content = content.strip()
        if not content:
            raise RuntimeError(f"LLM server {config.LLM_URL}/chat/completions returned no answer text")
        return content, reasoning

    def _chat_transformers(self, system: str, user: str, max_new_tokens: int, answer_tokens: int) -> tuple[str, str]:
        """-> (answer, reasoning). Qwen3-4B-Thinking-2507 always reasons first, then answers after </think>.
        If the thinking budget runs out, the thought is closed for it ("budget forcing") and it writes the answer from what it has."""
        import torch
        tok, model = self.llm()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        def generate(text: str, budget: int) -> str:
            from transformers import DynamicCache, StoppingCriteria, StoppingCriteriaList
            ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device())
            n = ids.shape[1]
            cache = DynamicCache(config=model.config)   # hybrid models (Qwen3.5 linear attention) need per-layer state slots from the config
            class ReportProgress(StoppingCriteria):
                def __init__(self): self.count = 0
                def __call__(self, input_ids, scores, **kwargs):
                    self.count += 1
                    if self.count % 16 == 0:
                        from .progress import report
                        report('reason', self.count, None, 'tokens')
                    return False
            reporter = ReportProgress()
            with torch.inference_mode():
                # prefill in 2k-token slices: one 13k-token forward pass (prompt + 7k tokens of thought) needs >1 GB of MLP
                # activations on top of the 8 GB of weights and runs a 12 GB card out of memory
                for i in range(0, n - 1, 2048):
                    model(input_ids=ids[:, i:min(i + 2048, n - 1)], past_key_values=cache, use_cache=True, logits_to_keep=1)
                torch.manual_seed(self.seed)      # sampling (recommended for the Thinking model) but repeatable run to run
                out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), past_key_values=cache, max_new_tokens=budget,
                                     do_sample=True, temperature=0.6, top_p=0.95, top_k=20, stopping_criteria=StoppingCriteriaList([reporter]))
            return tok.decode(out[0][n:], skip_special_tokens=True)

        from .progress import report
        report('reason', 0, None, 'tokens')
        text = generate(prompt, max_new_tokens)
        if "</think>" in text:
            reasoning, _, answer = text.partition("</think>")
            return answer.strip(), reasoning.replace("<think>", "").strip()
        return generate(prompt + text + CLOSING_THOUGHT, answer_tokens).strip(), text.strip()

    # ------------------------------------------------------------- retrieval
    def lexical(self, q: str, k: int) -> list[int]:
        s = self.bm25.get_scores(tokens(q))
        return [int(i) for i in np.argsort(-s)[:k] if s[i] > 0]

    def dense(self, queries: list[str], k: int) -> list[list[int]]:
        with self.on_gpu(self.embedder()) as m:
            v = m.encode(queries, normalize_embeddings=True)
        return [[int(i) for i in np.argsort(-(self.emb @ q))[:k]] for q in v]

    def hybrid(self, queries: list[str], k: int = 60, rrf_k: int = 60) -> list[int]:
        score: dict[int, float] = {}
        rankings = [self.lexical(q, k) for q in queries] + self.dense(queries, k)
        for ranking in rankings:
            for r, i in enumerate(ranking):
                score[i] = score.get(i, 0.0) + 1.0 / (rrf_k + r + 1)
        return [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1])][:k]

    def rerank(self, question: str, cand: list[int], top: int) -> list[int]:
        with self.on_gpu(self.reranker().model) as _:
            s = self.reranker().predict([(question, self.chunks[i]["text"]) for i in cand], batch_size=8)
        return self.novel([cand[j] for j in np.argsort(-np.asarray(s))], top)

    def novel(self, ranked: list[int], top: int, min_new: float = 0.25) -> list[int]:
        """Walk the ranking and keep a chunk only if >= 25 % of its character bigrams are new. Neighbourhood chunks overlap
        heavily (one per label cluster), and without this a dozen near-copies of the same plan labels push the calculation
        sheet out of the top-k."""
        def grams(t: str) -> set[str]:
            s = fold(norm(t)).replace(" ", "").replace("\n", "")
            return {s[k:k + 2] for k in range(len(s) - 1)}

        kept, seen = [], set()
        for i in ranked:
            g = grams(self.chunks[i]["body"])
            if g and (not kept or len(g - seen) / len(g) >= min_new):
                kept.append(i)
                seen |= g
            if len(kept) == top:
                break
        return kept

    def evidence_text(self, ev: list[dict], mode: str) -> str:
        """The evidence the LLM reads. `chunks`: the reranked top-k `ev`. `sheet`: every block and section chunk of every document
        in index (reading) order, the same text for every question so a server-side prompt cache can reuse it. Neighbourhood
        chunks are left out: they repeat the same labels."""
        if mode == "sheet":
            ev = [c for d in self.docs for c in self.chunks if c["doc"] == d and c["kind"] in ("block", "section")]
        elif mode != "chunks":
            raise ValueError(f"DOCSEARCH_CONTEXT must be 'chunks' or 'sheet', not {mode!r}")
        return "\n\n".join(f"<chunk id=\"{c['id']}\" kind=\"{c['kind']}\">\n{c['text']}\n</chunk>" for c in ev)

    # ---------------------------------------------------------------- answer
    def expand(self, question: str) -> list[str]:
        """Engineers write 泵浦 / 連通管 / 初沉池; users say 水泵 / 連接管 / 沉澱池. Every glossary group the question touches
        adds one extra lexical query with the whole group; the dense model covers the rest."""
        q = fold(norm(question)).lower()
        extra = [" ".join(g) for g in GLOSSARY if any(fold(t).lower() in q for t in g)]
        return [question] + extra

    def build_prompt(self, question: str, ev: list[dict], fx: dict | None, rr: str | None = None) -> tuple[str, str]:
        """(system, user) messages the LLM gets for `question` over the evidence chunks `ev`; `fx` the fact rows or None; `rr` the crop re-read block (Rereader.run()["text"]) or None."""
        outline = "\n".join(f"## {d}\n" + "\n".join(o) for d, o in self.outlines.items())
        evidence = self.evidence_text(ev, config.CONTEXT_MODE)
        system = ("You answer questions about OCR'd construction drawings (Taiwanese MEP / wastewater) using ONLY the chunks provided.\n"
                  "Reading conventions:\n"
                  "- OCR is noisy (look-alike characters, 1/l, 0/O, dropped glyphs). Chunks overlap and repeat the same labels: count distinct "
                  "items, not mentions. Reconcile garbled text with neighbouring chunks and say when you did.\n"
                  "- Calculation sheets (計算書) and equipment tables carry design values; plan labels show where things are. Prefer the former "
                  "for counts, sizes and ratings. When two sources disagree or cover different items, report each one with its source.\n"
                  "- Drawing dimensions are cm unless written with m. `L790×W352.5×H350/D320` = length x width x total height / effective "
                  "depth, so volume = L x W x D; lines `[tank size read: ...]` are that decoding done by the system, with the volume already computed - "
                  "use them. `4\"PVC連通管x2支` = two 4-inch PVC connecting pipes; `B.O.P+320` = pipe bottom level.\n"
                  "- A plan title with a volume under it (`<plan name>平面圖` followed by a sum of L x W x H terms `= ... m3`) gives the total of EVERYTHING "
                  "drawn on that plan level, not of one named unit: never use it as the size of one named unit. A unit's own size "
                  "comes from its calculation sheet or its own label. A `(1)(2)(3)` list on the calculation sheet is the list of that unit's tanks.\n"
                  "- Never do arithmetic in your head and never write a computed result yourself. Write only the expression, as {{7.9*3.525*3.2}}; "
                  "the system replaces it by `expression = exact value`.\n"
                  "- Separate what the drawing states from what you infer. If the information is not in the chunks, say it was NOT FOUND - "
                  "never guess, never substitute a different item (e.g. do not answer a fire-pump question with another pump's rating).\n"
                  "- When a fact row or chunk gives a value, quote it with its unit; do not re-derive it. If the drawing prints both a required value (需要/所需/以上/以下) and a selected value (採用/設計/選用), answer with the selected one and mention the required one only as context. For multi-item questions ('各', '分別', a count plus values), give the value for EVERY item found, from both the calculation sheet and the plan, each with its source.\n"
                  "Answer in the question's language, concisely, then list the chunk ids used.")
        if fx:
            system = system.replace("Answer in the question's language", '- "Fact rows" were parsed by code from the same OCR text, not read by a person: each row is one printed value with its chunk id, and "Computed" lines are exact counts, sums, comparisons and conflicts over those rows. When a row\'s entity and attribute match the question, answer from that row and quote its value and unit with its row id; a matching row is authoritative over your own reading of the chunks. Use a row only if its item is the one asked about (same name and number); a Computed `name match:` line ties the question\'s name to a shorter drawing name that OCR clipped, so rows of that drawing name ARE the asked item. When rows disagree, report each value with its source. If neither a row nor a chunk supports an answer, say NOT FOUND. A Computed `sized tanks: A by calc sheet + B by plan measurement = N` line means the calculation sheet and the plan size DIFFERENT tanks: the count is N (not A), and the answer lists every volume on the `per-tank volumes` line with its source.\nAnswer in the question\'s language')
        if config.LLM_THINK == "off":
            # Without a reasoning pass small models fill gaps with a nearby item's value; make them show the line first.
            system = system.replace("Answer in the question's language", "- Start the answer with `依據:` and copy the exact chunk line or fact row that names the asked item itself together with the value. If no line names the asked item, the answer is NOT FOUND: a value printed for another pump, tank or table is never the answer, and a question asking whether the drawings list something is answered 'no' when no line names it.\nAnswer in the question's language")
        if rr:
            system = system.replace("Answer in the question's language", "- `Re-read of drawing crops` lines: a vision model looked at the named OCR line on the drawing image again. Where its digits differ from the OCR text or from a fact row parsed from that line, prefer the re-read digits, quote them with the [Rn] id, and say the OCR read them differently.\nAnswer in the question's language")
        block = f"\n\n{rr}" if rr else ""
        user = f"Document outline:\n{outline}\n\nEvidence chunks:\n{evidence}{block}\n\nQuestion: {question}"
        if fx:
            user = f"Document outline:\n{outline}\n\nEvidence chunks:\n{evidence}\n\n{fx['text']}{block}\n\nQuestion: {question}"
        return system, user

    def prepare(self, question: str, top: int = config.TOP_K) -> dict:
        """The retrieval half of answer(): queries, candidate ids, the reranked top-k ids `best`, their chunks `ev` and the fact rows `fx`."""
        from .progress import report
        report('retrieve')
        queries = self.expand(question)
        cand = self.hybrid(queries)
        from .entities import anchor, question_entities
        if any(ident is not None for _, ident in question_entities(question)):
            with self.on_gpu(self.reranker().model):
                scores = self.reranker().predict([(question, self.chunks[i]["text"]) for i in cand], batch_size=8)
            pool = [cand[j] for j in np.argsort(-np.asarray(scores))]
            best = anchor(question, self.novel(pool, top), pool, self.chunks, top)
        else:
            best = self.rerank(question, cand, top)   # expansions only widened the pool
        ev = [self.chunks[i] for i in best]
        fx = factquery.context(question, self.facts) if config.FACTS else None
        return {"queries": queries, "cand": cand, "best": best, "ev": ev, "fx": fx}

    def rereader(self):
        """Crop re-read (docsearch/reread.py), built on first use and kept for the life of the Engine."""
        if getattr(self, "_rereader", None) is None:
            from . import reread
            self._rereader = reread.Rereader(self.docs, self.chunks)
        return self._rereader

    def answer(self, question: str, top: int = config.TOP_K, lock=None) -> dict:
        with lock if lock is not None else contextlib.nullcontext():
            p = self.prepare(question, top)
            queries, cand, best, ev, fx = p["queries"], p["cand"], p["best"], p["ev"], p["fx"]
            rr = self.rereader().run(ev, fx) if config.REREAD != "0" else None
            system, user = self.build_prompt(question, ev, fx, rr["text"] if rr else None)
        ans, reasoning = self.chat(system, user)
        verdict = None
        if config.VERIFY:
            if lock is None:
                verdict, ans, reasoning, ev = self._verify(question, cand, best, ev, fx, system, user, ans, reasoning, rr)
            else:
                verdict, ans, reasoning, ev = self._verify(question, cand, best, ev, fx, system, user, ans, reasoning, rr, lock)
        answer_raw = ans
        ans = calc.fill(ans)
        if os.environ.get("DOCSEARCH_ANSWER_FORMAT", "1") == "1":
            if config.CONTEXT_MODE == "sheet":
                contexts = [c["text"] for c in self.chunks if c["doc"] in self.docs and c["kind"] in ("block", "section")]
            else:
                contexts = [c["text"] for c in ev]
            ans = answerfmt.normalize(ans, question, contexts + ([fx["text"]] if fx else []))
        if fx:
            by_id = {c["id"]: c for c in self.chunks}
            seen = {c["id"] for c in ev}
            for row in fx["rows"]:
                for src in row["src"]:
                    ident = src["chunk"]
                    if ident not in seen and ident in by_id and len(seen) - len(best) < 8:
                        ev.append(by_id[ident])
                        seen.add(ident)
        result = {"question": question, "queries": queries, "answer": ans, "reasoning": reasoning,
                  "evidence": [{"id": c["id"], "doc": c["doc"], "bbox": c["bbox"], "text": c["body"]} for c in ev], "context": config.CONTEXT_MODE, "llm": config.LLM_ID}
        if os.environ.get("DOCSEARCH_ANSWER_FORMAT", "1") == "1":
            result["answer_raw"] = answer_raw
        if fx:
            result.update(facts=fx["rows"], computed=fx["computed"])
        if rr is not None:
            result["reread"] = rr["records"]
        if verdict:
            result["verify"] = verdict
        return result

    def _verify(self, question: str, cand: list[int], best: list[int], ev: list[dict], fx: dict | None,
                system: str, user: str, ans: str, reasoning: str, rr: dict | None, lock=None) -> tuple[dict, str, str, list[dict]]:
        """Check the raw answer (verify.check), then retry at most once. Returns (record, answer, reasoning, evidence): the record
        is {verdict, retried, first_answer, unsupported}; the answer and evidence are the retry's when it ran."""
        rows, computed = (fx["rows"], fx["computed"]) if fx else ((), ())
        texts = [c["text"] for c in ev] + (rr["evidence"] if rr else [])
        first = verify.check(ans, texts, rows, computed)
        rec = {"verdict": first["verdict"], "retried": False, "first_answer": ans, "unsupported": first["unsupported"]}
        if first["verdict"] == "unsupported_number":
            ans, reasoning = self.chat(system, user + verify.note(first["unsupported"]))
            rec["retried"] = True
        elif first["verdict"] == "refusal" and best and verify.missing(question, [c["text"] for c in self.chunks], texts):
            with lock if lock is not None else contextlib.nullcontext():
                top = self.chunks[best[0]]
                section = [j for j, c in enumerate(self.chunks) if c["kind"] == "section" and c["doc"] == top["doc"]
                           and c["section"] == top["section"]][:1]
                more = [i for i in self.rerank(question, cand, len(best) + 12) if i not in best][:12]
                ev = [self.chunks[i] for i in dict.fromkeys(section + list(best) + more)]
                system, user = self.build_prompt(question, ev, fx, rr["text"] if rr else None)
            ans, reasoning = self.chat(system, user)
            rec["retried"] = True
        return rec, ans, reasoning, ev

    # --------------------------------------------------------------- keyword
    def keyword(self, kw: str, min_score: float = 0.75) -> list[dict]:
        """Block chunks containing `kw` exactly (score 1) or with OCR noise (share of the keyword's bigrams present).
        Matching ignores spaces, case and Traditional/Simplified script. `hit["matches"]` are the individual OCR lines
        that carry the keyword (what gets boxed); a keyword spanning two lines falls back to the whole block."""
        key = lambda s: fold(norm(s)).lower().replace(" ", "").replace("\n", "")
        k = key(kw)
        grams = {k[i:i + 2] for i in range(len(k) - 1)} or {k}

        def score(text: str) -> float:
            if k in text:
                return 1.0
            s = sum(g in text for g in grams) / len(grams)
            return s if len(k) >= 3 and s >= min_score else 0.0

        hits = []
        for c in self.chunks:
            if c["kind"] != "block":
                continue
            s = score(key(c["body"]))
            if not s:
                continue
            per_line = [(score(key(ln["text"])), ln) for ln in c["lines"]]
            matches = [ln for sc, ln in per_line if sc and sc >= s]
            hits.append({**c, "score": round(s, 2), "matches": matches or [{"text": c["body"], "bbox": c["bbox"]}]})
        return sorted(hits, key=lambda h: (-h["score"], h["doc"], h["bbox"][1]))

    def annotate(self, hits: list[dict], out: Path) -> list[Path]:
        """Overview image per sheet with numbered red boxes round the matching lines + one zoomed crop per hit."""
        from PIL import Image, ImageDraw

        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for doc in sorted({h["doc"] for h in hits}):
            src = Image.open(self.docs[doc]["image"]).convert("RGB")
            im = src.copy()
            dr = ImageDraw.Draw(im)
            for n, h in enumerate([h for h in hits if h["doc"] == doc], 1):
                for m in h["matches"]:
                    dr.rectangle(m["bbox"], outline=(255, 0, 0), width=6)
                b = h["bbox"]
                dr.text((b[0], max(b[1] - 30, 0)), str(n), fill=(255, 0, 0))
                pad = 140
                crop = src.crop((max(b[0] - pad, 0), max(b[1] - pad, 0), b[2] + pad, b[3] + pad))
                d2 = ImageDraw.Draw(crop)
                for m in h["matches"]:
                    mb = m["bbox"]
                    d2.rectangle((mb[0] - max(b[0] - pad, 0), mb[1] - max(b[1] - pad, 0), mb[2] - max(b[0] - pad, 0), mb[3] - max(b[1] - pad, 0)), outline=(255, 0, 0), width=3)
                p = out / f"{doc[:30]}.hit{n}.png"
                crop.save(p)
                paths.append(p)
            ov = out / f"{doc[:30]}.overview.jpg"
            im.resize((im.width // 2, im.height // 2)).save(ov, quality=85)
            paths.append(ov)
        return paths

    def locate(self, result: dict, out: Path, limit: int = 4) -> list[Path]:
        """Boxed crops of the evidence an answer cites (the top evidence when it cites none): the 'show me where' output."""
        byid = {c["id"]: c for c in self.chunks}
        ev = [byid[e["id"]] for e in result["evidence"]]
        num = lambda c: c["id"].rsplit("#", 1)[-1]   # the model cites chunks as '#220', not by the full 'sheet#220' id
        cited = [c for c in ev if re.search(rf"#{re.escape(num(c))}(?!\d)", result["answer"])] or ev
        hits = [{**c, "matches": c["lines"] or [{"text": c["body"], "bbox": c["bbox"]}]} for c in cited[:limit]]
        return self.annotate(hits, out)
