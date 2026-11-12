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
import re
from pathlib import Path

import numpy as np

from . import calc, config
from .index import fold, norm, tokens


# Synonym groups of drawing vocabulary (docsearch/glossary.json, plain list of lists - extend it for your own trade).
GLOSSARY: list[list[str]] = json.loads((Path(__file__).with_name("glossary.json")).read_text())


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
        self.outlines: dict = {d: v for d, v in data["outlines"].items() if d in self.docs}
        self.emb = np.load(index_dir / "emb.npy").astype(np.float32)[keep]
        self.bm25 = BM25Okapi([tokens(c["text"]) for c in self.chunks])
        self._embedder = self._reranker = self._llm = None
        self.seed = 0

    # ---------------------------------------------------------------- models
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(config.EMBED_ID, device="cpu").half()
        return self._embedder

    def reranker(self):
        if self._reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.RERANK_ID, device="cpu", model_kwargs={"dtype": torch.float16})
        return self._reranker

    def llm(self):
        if self._llm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(config.LLM_ID)
            model = AutoModelForCausalLM.from_pretrained(config.LLM_ID, dtype=torch.bfloat16).to("cuda").eval()
            self._llm = (tok, model)
        return self._llm

    @contextlib.contextmanager
    def on_gpu(self, model):
        import torch
        model.to("cuda")
        try:
            yield model
        finally:
            model.to("cpu")
            torch.cuda.empty_cache()

    def chat(self, system: str, user: str, max_new_tokens: int = 7000, answer_tokens: int = 1500) -> tuple[str, str]:
        """-> (answer, reasoning). Qwen3-4B-Thinking-2507 always reasons first, then answers after </think>.
        If the thinking budget runs out, the thought is closed for it ("budget forcing") and it writes the answer from what it has."""
        import torch
        tok, model = self.llm()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        def generate(text: str, budget: int) -> str:
            enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.inference_mode():
                torch.manual_seed(self.seed)      # sampling (recommended for the Thinking model) but repeatable run to run
                out = model.generate(**enc, max_new_tokens=budget, do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
            return tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)

        text = generate(prompt, max_new_tokens)
        if "</think>" in text:
            reasoning, _, answer = text.partition("</think>")
            return answer.strip(), reasoning.replace("<think>", "").strip()
        closing = "\n\nI have to stop thinking now and write the final answer from what I have found so far.\n</think>\n\n"
        return generate(prompt + text + closing, answer_tokens).strip(), text.strip()

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

    # ---------------------------------------------------------------- answer
    def expand(self, question: str) -> list[str]:
        """Engineers write 泵浦 / 連通管 / 初沉池; users say 水泵 / 連接管 / 沉澱池. Every glossary group the question touches
        adds one extra lexical query with the whole group; the dense model covers the rest."""
        q = fold(norm(question)).lower()
        extra = [" ".join(g) for g in GLOSSARY if any(fold(t).lower() in q for t in g)]
        return [question] + extra

    def answer(self, question: str, top: int = 12) -> dict:
        queries = self.expand(question)
        cand = self.hybrid(queries)
        best = self.rerank(question, cand, top)   # the cross-encoder judges the *question*; expansions only widened the pool
        ev = [self.chunks[i] for i in best]
        outline = "\n".join(f"## {d}\n" + "\n".join(o) for d, o in self.outlines.items())
        evidence = "\n\n".join(f"<chunk id=\"{c['id']}\" kind=\"{c['kind']}\">\n{c['text']}\n</chunk>" for c in ev)
        system = ("You answer questions about OCR'd construction drawings (Taiwanese MEP / wastewater) using ONLY the chunks provided.\n"
                  "Reading conventions:\n"
                  "- OCR is noisy (look-alike characters, 1/l, 0/O, dropped glyphs). Chunks overlap and repeat the same labels: count distinct "
                  "items, not mentions. Reconcile garbled text with neighbouring chunks and say when you did.\n"
                  "- Calculation sheets (計算書) and equipment tables carry design values; plan labels show where things are. Prefer the former "
                  "for counts, sizes and ratings. When two sources disagree or cover different items, report each one with its source.\n"
                  "- Drawing dimensions are cm unless written with m. `L790×W352.5×H350/D320` = length x width x total height / effective "
                  "depth, so volume = L x W x D. `4\"PVC連通管x2支` = two 4-inch PVC connecting pipes; `B.O.P+320` = pipe bottom level.\n"
                  "- A plan title with a volume under it (`<plan name>平面圖` followed by a sum of L x W x H terms `= ... m3`) gives the total of EVERYTHING "
                  "drawn on that plan level, not of one named unit: never use it as the size of one named unit. A unit's own size "
                  "comes from its calculation sheet or its own label. A `(1)(2)(3)` list on the calculation sheet is the list of that unit's tanks.\n"
                  "- Never do arithmetic in your head and never write a computed result yourself. Write only the expression, as {{7.9*3.525*3.2}}; "
                  "the system replaces it by `expression = exact value`.\n"
                  "- Separate what the drawing states from what you infer. If the information is not in the chunks, say it was NOT FOUND - "
                  "never guess, never substitute a different item (e.g. do not answer a fire-pump question with another pump's rating).\n"
                  "Answer in the question's language, concisely, then list the chunk ids used.")
        user = f"Document outline:\n{outline}\n\nEvidence chunks:\n{evidence}\n\nQuestion: {question}"
        ans, reasoning = self.chat(system, user)
        ans = calc.fill(ans)
        return {"question": question, "queries": queries, "answer": ans, "reasoning": reasoning,
                "evidence": [{"id": c["id"], "doc": c["doc"], "bbox": c["bbox"], "text": c["body"]} for c in ev]}

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
