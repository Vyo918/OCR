"""Build the testing and benchmarking history page (one offline HTML file).

    python -m docsearch.history --curated docs/history/history.json \
        [--bench work/bench-json] [--benchmark PRIVATE.json] --out work/history.html

The curated JSON is the source of truth and carries the last measured
per-question rows, so the page builds without --bench. --bench re-reads the
results.jsonl files written by the QA runner and replaces the per-question rows
of runs with the same name. Only id, category, difficulty, grade and seconds are
read from result rows, and never question, reference or answer text.

--benchmark embeds private question, reference and engine answer text in the question grid's
detail views. That output contains benchmark text: never commit or publish it. The generator
refuses an --out path under docs/.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 1
TEMPLATE = Path(__file__).with_name("history_template.html")
PLACEHOLDER = "__HISTORY_BODY__"
GRADES = ("pass", "fail", "review", "error")
GRADE_LETTER = {"pass": "P", "fail": "F", "review": "R", "error": "E"}
RUN_STATUSES = ("done", "running", "queued")
FORBIDDEN_KEYS = {"question", "reference", "answer", "model_answer", "evidence_ids", "computed"}
ROW_FIELDS = ("id", "category", "difficulty", "grade", "seconds")
ANSWER_CAP = 3000
DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"
CITATION_LINE = re.compile(r"^.+?#\d+(?:\s*,\s*.+?#\d+)*$")
LIST_SECTIONS = {
    "experiments": ("id", "title", "date", "status", "purpose", "method", "dataset", "models",
                    "configuration", "environment", "duration", "results", "judgement", "caveats", "artifacts"),
    "runs": ("run", "label", "experiment", "model", "machine", "dataset", "status", "config", "hand",
             "notes", "grid", "per_question"),
    "models": ("id", "name", "role", "size", "where", "status", "verdict"),
    "machines": ("id", "name", "hardware", "notes"),
    "datasets": ("id", "name", "description"),
    "milestones": ("date", "commit", "title"),
    "not_tested": ("name", "role", "protocol", "priority"),
    "discarded_questions": ("id", "reason"),
}
TEXT_LISTS = ("findings", "caveats", "open_items")
PIPELINE_FIELDS = ("stage", "component", "version", "params", "role")
PIPELINE_STAGES = 18
COMPARE_HEADERS = {"config": "Configuration", "reader": "Reader", "runtime": "Runtime",
                   "facts": "Fact rows", "hand": "Hand grade"}
TOP_FIELDS = ("as_of", "project", "default", "best", "retained40")
NAV = (
    ("overview", "Overview"), ("timeline", "Timeline"), ("leaderboard", "Leaderboard"),
    ("experiments", "Experiments"), ("heatgrid", "Question grid"), ("models", "Models"),
    ("findings", "Findings"), ("not-tested", "Not tested"), ("caveats", "Caveats"),
    ("add-record", "Add a record"),
)


def esc(value: object) -> str:
    return html.escape(str(value))


def validate(data: dict) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    errors += [f"missing top-level field {k}" for k in TOP_FIELDS if k not in data]
    for section, fields in LIST_SECTIONS.items():
        items = data.get(section)
        if not isinstance(items, list) or not items:
            errors.append(f"{section}: must be a non-empty list")
            continue
        for i, item in enumerate(items):
            missing = [f for f in fields if not isinstance(item, dict) or f not in item]
            if missing:
                errors.append(f"{section}[{i}]: missing {', '.join(missing)}")
    errors += [f"{k}: must be a list of strings" for k in TEXT_LISTS if not isinstance(data.get(k), list)]
    if errors:
        return errors

    for key in ("default", "best"):
        stages = data[key].get("pipeline")
        if not isinstance(stages, list) or len(stages) != PIPELINE_STAGES:
            errors.append(f"{key}.pipeline: needs {PIPELINE_STAGES} stages")
            continue
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict) or set(PIPELINE_FIELDS) - set(stage):
                errors.append(f"{key}.pipeline[{i}]: needs {', '.join(PIPELINE_FIELDS)}")
    comparison = data["best"].get("comparison")
    if not isinstance(comparison, list) or not comparison:
        errors.append("best.comparison: must be a non-empty list")
    else:
        for i, row in enumerate(comparison):
            if not isinstance(row, dict) or set(COMPARE_HEADERS) - set(row):
                errors.append(f"best.comparison[{i}]: needs {', '.join(COMPARE_HEADERS)}")
    def dup_check(section: str, values: list) -> None:
        dup = sorted(k for k, c in Counter(values).items() if c > 1)
        if dup:
            errors.append(f"{section}: duplicate ids {dup}")

    exp_ids = [e["id"] for e in data["experiments"]]
    model_ids = [m["id"] for m in data["models"]]
    machine_ids = [m["id"] for m in data["machines"]]
    dataset_ids = [d["id"] for d in data["datasets"]]
    run_names = [r["run"] for r in data["runs"]]
    for section, values in (("experiments", exp_ids), ("models", model_ids), ("machines", machine_ids),
                            ("datasets", dataset_ids), ("runs", run_names)):
        dup_check(section, values)

    for e in data["experiments"]:
        for mid in e["models"]:
            if mid not in model_ids:
                errors.append(f"experiment {e['id']}: unknown model {mid}")
        columns = e["results"].get("columns", [])
        if any(len(row) != len(columns) for row in e["results"].get("rows", [])):
            errors.append(f"experiment {e['id']}: result rows do not match columns")

    for r in data["runs"]:
        where = f"run {r['run']}"
        if r["experiment"] not in exp_ids:
            errors.append(f"{where}: unknown experiment {r['experiment']}")
        if r["model"] not in model_ids:
            errors.append(f"{where}: unknown model {r['model']}")
        if r["machine"] not in machine_ids:
            errors.append(f"{where}: unknown machine {r['machine']}")
        if r["dataset"] not in dataset_ids:
            errors.append(f"{where}: unknown dataset {r['dataset']}")
        if r["status"] not in RUN_STATUSES:
            errors.append(f"{where}: status must be one of {RUN_STATUSES}")
        for h in r["hand"]:
            if not isinstance(h, dict) or not {"scope", "correct", "partial", "wrong"} <= set(h):
                errors.append(f"{where}: hand entries need scope, correct, partial, wrong")
        rows = r["per_question"]
        for row in rows:
            if any(f not in row for f in ROW_FIELDS):
                errors.append(f"{where}: per_question row missing fields")
            elif row["grade"] not in GRADES or not isinstance(row["seconds"], (int, float)):
                errors.append(f"{where}: bad grade or seconds in row {row.get('id')}")
        appended = r.get("appended_questions", [])
        ids = {row.get("id") for row in rows}
        if not set(appended) <= ids:
            errors.append(f"{where}: appended_questions not in per_question")

    retained = data["retained40"]
    if not isinstance(retained, dict) or not isinstance(retained.get("ids"), list) or "note" not in retained:
        errors.append("retained40: needs ids and note")
    else:
        dup_check("retained40", retained["ids"])

    discarded = {d["id"] for d in data["discarded_questions"]}
    for r in data["runs"]:
        leaked = sorted(discarded & {row.get("id") for row in r["per_question"]})
        if leaked:
            errors.append(f"run {r['run']}: discarded questions present {leaked}")

    bad = sorted({k for k in _walk_keys(data) if k in FORBIDDEN_KEYS})
    if bad:
        errors.append(f"forbidden keys present (benchmark text must stay out): {bad}")
    return errors


def _walk_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_keys(value)


def read_bench(bench: Path) -> dict[str, dict]:
    """Map run name (folder path under bench) to whitelisted rows and Config."""
    found: dict[str, dict] = {}
    for path in _result_files(bench):
        rows: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rows[rec["id"]] = {
                "id": rec["id"],
                "category": rec.get("類別", ""),
                "difficulty": rec.get("難度", ""),
                "grade": rec["grade"],
                "seconds": round(float(rec["seconds"]), 1),
            }
        config = ""
        summary = path.with_name("summary.md")
        if summary.exists():
            match = re.search(r"^Config: (.+)$", summary.read_text(encoding="utf-8"), re.M)
            config = match.group(1).strip() if match else ""
        name = path.parent.relative_to(bench).as_posix()
        found[name] = {"per_question": list(rows.values()), "config": config}
    return found


def _result_files(bench: Path) -> list[Path]:
    return sorted([*bench.glob("*/results.jsonl"), *bench.glob("facts-ab/*/results.jsonl")])


def clean_answer(text: str | None) -> str:
    """Drop trailing chunk-id citation lines (e.g. 'drawing#153, drawing#391') and cap the length."""
    lines = (text or "").strip().splitlines()
    while lines and CITATION_LINE.match(lines[-1].strip()):
        lines.pop()
    body = "\n".join(lines).strip()
    return body if len(body) <= ANSWER_CAP else body[:ANSWER_CAP] + " [cut]"


def read_benchmark(path: Path) -> dict[str, dict]:
    """Map question id to question and reference text from the private benchmark file."""
    items = json.loads(path.read_text(encoding="utf-8"))["題目"]
    return {q["編號"]: {"question": q["問題"], "reference": q["答案"]} for q in items}


def read_answers(bench: Path) -> dict[str, dict[str, str]]:
    """Map run name to {question id: cleaned engine answer}. Used only with --benchmark."""
    found: dict[str, dict[str, str]] = {}
    for path in _result_files(bench):
        rows: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                rows[rec["id"]] = clean_answer(rec.get("answer"))
        found[path.parent.relative_to(bench).as_posix()] = rows
    return found


def merge_bench(data: dict, bench: Path) -> list[str]:
    measured = read_bench(bench)
    runs = {r["run"]: r for r in data["runs"]}
    discarded = {d["id"] for d in data["discarded_questions"]}
    unmatched = []
    for name, found in measured.items():
        run = runs.get(name)
        if run is None:
            unmatched.append(name)
            continue
        run["per_question"] = [row for row in found["per_question"] if row["id"] not in discarded]
        if not run["config"] and found["config"]:
            run["config"] = found["config"]
    return unmatched


def metrics(run: dict) -> dict:
    appended = set(run.get("appended_questions", []))
    rows = [r for r in run["per_question"] if r["id"] not in appended]
    counts = Counter(r["grade"] for r in rows)
    secs = [r["seconds"] for r in rows]
    n = len(rows)
    return {
        "n": n,
        "expected": run.get("expected", n),
        "pass": counts["pass"],
        "fail": counts["fail"],
        "review": counts["review"],
        "error": counts["error"],
        "pct": 100 * counts["pass"] / n if n else None,
        "mean": statistics.mean(secs) if secs else None,
        "median": statistics.median(secs) if secs else None,
        "total": sum(secs),
        "appended": len(run["per_question"]) - n,
    }


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")


def _fmt(value: float | None, digits: int = 1, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def _or_dash(value: object) -> str:
    return "-" if value is None else str(value)


def _sort_key(date: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", date)
    return match.group(0) if match else "9999-99-99"


def _details(cls: str, anchor: str | None, summary: str, body: str, open_: bool = False) -> str:
    anchor_attr = f' id="{esc(anchor)}"' if anchor else ""
    open_attr = " open" if open_ else ""
    return f'<details class="{cls}"{anchor_attr}{open_attr}><summary>{summary}</summary>{body}</details>'


def _section(anchor: str, title: str, note: str, body: str) -> str:
    summary = f'<span class="title">{esc(title)}</span>'
    if note:
        summary += f'<span class="note">{esc(note)}</span>'
    return _details("section", anchor, summary, body)


def _table(columns: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="wrap"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _ul(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul>"


def _p(text: str) -> str:
    return f"<p>{esc(text)}</p>"


def _hand_text(run: dict) -> str:
    if not run["hand"]:
        return "not hand-graded"
    return "; ".join(
        f"{_or_dash(h['correct'])} / {_or_dash(h['partial'])} / {_or_dash(h['wrong'])} ({h['scope']})"
        for h in run["hand"]
    )


def _select(name: str, values: list[str]) -> str:
    options = '<option value="">all</option>' + "".join(
        f'<option value="{esc(v)}">{esc(v)}</option>' for v in values
    )
    return f'<label>{esc(name)} <select data-filter="{esc(name)}">{options}</select></label>'


def _header(data: dict) -> str:
    best = data["best"]
    links = "".join(f'<a href="#{anchor}">{esc(label)}</a>' for anchor, label in NAV)
    return (
        '<header class="top">'
        '<div class="bar"><span class="name">Testing and benchmarking history</span>'
        f'<span class="best">Best measured: {esc(best["label"])}, hand {esc(best["hand"])}</span></div>'
        f'<nav>{links}</nav></header>'
    )


def _kv(rows: list[tuple[str, str]]) -> str:
    return '<dl class="kv">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in rows) + "</dl>"


def _pipeline_details(stages: list[dict]) -> str:
    rows = [[esc(s[field]) for field in PIPELINE_FIELDS] for s in stages]
    table = _table(["Stage", "Component / model", "Version / precision", "Parameters", "Role"], rows, "pipeline")
    return _details("sub", None, f"Full pipeline ({len(stages)} stages)", table)


def _comparison(rows: list[dict]) -> str:
    first = rows[0]
    body = [
        [esc(row[key]) if row[key] == first[key] else f"<mark>{esc(row[key])}</mark>" for key in COMPARE_HEADERS]
        for row in rows
    ]
    return _table(list(COMPARE_HEADERS.values()), body, "compare")


def _measured(rows: list[dict]) -> str:
    body = [[esc(r["scope"]), esc(r["hand"]), esc(r["seconds"])] for r in rows]
    return _table(["Measured", "Hand grade", "s per question"], body, "measured")


def _overview(data: dict) -> str:
    d, b = data["default"], data["best"]
    counts = [
        ("experiments", len(data["experiments"])), ("QA runs", len(data["runs"])),
        ("models and engines", len(data["models"])), ("datasets", len(data["datasets"])),
        ("machines", len(data["machines"])),
    ]
    chips = "".join(f"<li><b>{n}</b> {esc(label)}</li>" for label, n in counts)
    facts = "on" if d["facts"] else "off"
    default_card = (
        '<div class="card"><h3>Current default</h3>'
        f'<p class="headline">{esc(d["headline"])}</p>'
        + _kv([
            ("Reader", d["llm"]),
            ("OCR", d["ocr"]),
            ("Retrieval", f'top-{d["top_k"]} chunks, seed {d["seed"]}, fact rows {facts}'),
            ("Runtime", d["runtime"]),
            ("Hand score", d["hand_score"]),
        ])
        + _measured(d["measured"])
        + f'<p class="caveat">{esc(d["caveat"])}</p>'
        + _pipeline_details(d["pipeline"]) + "</div>"
    )
    best_card = (
        '<div class="card"><h3>Best measured</h3>'
        f'<p class="headline">{esc(b["headline"])}</p>'
        + _comparison(b["comparison"])
        + f'<p class="caveat">{esc(b["caveat"])}</p>'
        + _pipeline_details(b["pipeline"]) + "</div>"
    )
    return (
        '<section id="overview" class="overview"><h2>Overview</h2><div class="cards">'
        + default_card + best_card
        + f'</div><ul class="counts">{chips}</ul>'
        '<p class="note">Two grades are reported and never merged. The automatic grader is conservative: '
        'it never passes a wrong answer but misses some correct ones. Hand grading by one reviewer is the '
        'reference. Per-question hand labels are not in the repository, so the question grid shows grader '
        'grades.</p></section>'
    )


def _timeline(data: dict) -> str:
    items: list[tuple[str, str]] = []
    for e in data["experiments"]:
        items.append((e["date"], f'<li><span class="date">{esc(e["date"])}</span> '
                                 f'<a href="#exp-{esc(e["id"])}">{esc(e["title"])}</a> '
                                 f'<span class="badge s-{esc(e["status"])}">{esc(e["status"])}</span></li>'))
    for m in data["milestones"]:
        items.append((m["date"], f'<li><span class="date">{esc(m["date"])}</span> {esc(m["title"])} '
                                 f'<code>{esc(m["commit"])}</code></li>'))
    items.sort(key=lambda item: _sort_key(item[0]))
    return '<ol class="timeline">' + "".join(html_item for _, html_item in items) + "</ol>"


def _leaderboard(data: dict) -> str:
    runs = data["runs"]
    filters = "".join(
        _select(name, sorted({r[key] for r in runs}))
        for name, key in (("experiment", "experiment"), ("model", "model"), ("machine", "machine"))
    )
    head = ["Run", "Experiment", "Model", "Machine", "Dataset", "Rows", "Grader pass", "Pass %",
            "Hand correct / partial / wrong", "s/question", "Status", "Config"]
    rows = []
    for run in runs:
        m = metrics(run)
        count = f"{m['n']} of {m['expected']}" if m["expected"] != m["n"] else str(m["n"])
        pct = "" if m["pct"] is None else f"{m['pct']:.4f}"
        mean = "" if m["mean"] is None else f"{m['mean']:.2f}"
        rows.append(
            f'<tr data-row data-experiment="{esc(run["experiment"])}" data-model="{esc(run["model"])}" '
            f'data-machine="{esc(run["machine"])}">'
            f'<td><a href="#run-{_slug(run["run"])}">{esc(run["label"])}</a><br><code>{esc(run["run"])}</code></td>'
            f'<td><a href="#exp-{esc(run["experiment"])}">{esc(run["experiment"])}</a></td>'
            f'<td>{esc(run["model"])}</td><td>{esc(run["machine"])}</td><td>{esc(run["dataset"])}</td>'
            f'<td data-v="{m["n"]}">{count}</td>'
            f'<td data-v="{m["pass"]}">{m["pass"]} / {m["n"]}</td>'
            f'<td data-v="{pct}">{_fmt(m["pct"], 1, "%")}</td>'
            f'<td>{esc(_hand_text(run))}</td>'
            f'<td data-v="{mean}">{_fmt(m["mean"], 1)}</td>'
            f'<td><span class="badge s-{esc(run["status"])}">{esc(run["status"])}</span></td>'
            f'<td class="cfg">{esc(run["config"] or "not recorded")}</td></tr>'
        )
    heads = "".join(f'<th class="sort">{esc(h)}</th>' for h in head)
    return (
        f'<div class="filters">{filters}</div>'
        '<p class="note">Click a column header to sort. Grader pass counts exclude rows appended after a run '
        'started. Hand grades are shown where recorded for that run and scope.</p>'
        f'<div class="wrap"><table class="sortable"><thead><tr>{heads}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _run(run: dict) -> str:
    m = metrics(run)
    count = f"{m['n']} of {m['expected']}" if m["expected"] != m["n"] else str(m["n"])
    head = (f'<span class="title">{esc(run["label"])}</span>'
            f'<span class="meta"><code>{esc(run["run"])}</code> · '
            f'<span class="badge s-{esc(run["status"])}">{esc(run["status"])}</span> · '
            f'grader {m["pass"]} / {m["n"]}</span>')
    stats = _table(
        ["Rows", "Grader pass", "Grader fail", "Grader review", "Pass %", "Mean s", "Median s", "Total s",
         "Rows appended after start"],
        [[esc(count), str(m["pass"]), str(m["fail"]), str(m["review"]),
          esc(_fmt(m["pct"], 1, "%")), esc(_fmt(m["mean"])), esc(_fmt(m["median"])),
          esc(_fmt(m["total"], 0)), esc(", ".join(run.get("appended_questions", [])) or "none")]],
    )
    hand = _table(
        ["Scope", "Correct", "Partial", "Wrong", "Source"],
        [[esc(h["scope"]), esc(_or_dash(h["correct"])), esc(_or_dash(h["partial"])),
          esc(_or_dash(h["wrong"])), esc(h.get("source", ""))] for h in run["hand"]],
    ) if run["hand"] else _p("No hand grades recorded for this run.")
    per_question = _table(
        ["ID", "Category", "Difficulty", "Grade", "Seconds"],
        [[esc(r["id"]), esc(r["category"]), esc(r["difficulty"]),
          f'<span class="grade g-{esc(r["grade"])}">{esc(r["grade"])}</span>', esc(f'{r["seconds"]:.1f}')]
         for r in run["per_question"]],
        cls="sortable",
    )
    body = (
        _p(run["notes"])
        + _ul([f"Config: {run['config'] or 'not recorded'}", f"Machine: {run['machine']}",
               f"Dataset: {run['dataset']}"])
        + "<h4>Grader</h4>" + stats
        + "<h4>Hand grades</h4>" + hand
        + "<h4>Per question (grader grades)</h4>" + per_question
    )
    return _details("run", f"run-{_slug(run['run'])}", head, body)


def _experiment(e: dict, data: dict, models: dict) -> str:
    runs = [r for r in data["runs"] if r["experiment"] == e["id"]]
    model_names = [models[mid]["name"] for mid in e["models"]]
    subs = [
        ("Purpose", _p(e["purpose"])),
        ("Method", _p(e["method"])),
        ("Dataset", _p(e["dataset"])),
        ("Models and versions", _ul(model_names)),
        ("Configuration", _ul(e["configuration"])),
        ("Environment and hardware", _p(e["environment"])),
        ("Duration", _p(e["duration"])),
        ("Results", _table(e["results"]["columns"],
                           [[esc(c) for c in row] for row in e["results"]["rows"]])),
        ("Judgement", _p(e["judgement"])),
        ("Caveats", _ul(e["caveats"])),
        ("Artifacts", _ul(e["artifacts"])),
        (f"Runs ({len(runs)})", "".join(_run(r) for r in runs) or _p("No measured runs recorded.")),
    ]
    head = (f'<span class="title">{esc(e["title"])}</span>'
            f'<span class="meta">{esc(e["date"])} · '
            f'<span class="badge s-{esc(e["status"])}">{esc(e["status"])}</span></span>')
    body = "".join(_details("sub", None, esc(title), part) for title, part in subs)
    return _details("exp", f"exp-{e['id']}", head, body)


def grid_runs(data: dict) -> list[dict]:
    return [r for r in data["runs"] if r["grid"]]


def _qdetail(data: dict, texts: dict | None, answers: dict | None) -> dict:
    """Detail for the question grid; results align with grid_runs(). Text only with --benchmark."""
    models = {m["id"]: m["name"] for m in data["models"]}
    runs = grid_runs(data)
    by_run = {r["run"]: {row["id"]: row for row in r["per_question"]} for r in runs}
    questions = {}
    for qid in data["retained40"]["ids"]:
        rows = [by_run[r["run"]].get(qid) for r in runs]
        first = next((row for row in rows if row), {"category": "", "difficulty": ""})
        entry = {"category": first["category"], "difficulty": first["difficulty"], "results": []}
        for r, row in zip(runs, rows):
            if row is None:
                entry["results"].append(None)
                continue
            result = {"grade": row["grade"], "seconds": row["seconds"]}
            if answers is not None and qid in answers.get(r["run"], {}):
                result["answer"] = answers[r["run"]][qid]
            entry["results"].append(result)
        if texts is not None:
            entry.update(texts.get(qid, {}))
        questions[qid] = entry
    return {
        "text": texts is not None,
        "runs": [{"run": r["run"], "label": r["label"], "model": models[r["model"]], "config": r["config"]}
                 for r in runs],
        "questions": questions,
    }


def _qdetail_block(detail: dict) -> str:
    """One JSON island. '</' and '<!--' are escaped so no text can close the script element."""
    text = json.dumps(detail, ensure_ascii=False, separators=(",", ":"))
    text = text.replace("</", "<\\/").replace("<!--", "\\u003c!--")
    return f'<script type="application/json" id="qdetail">{text}</script>'


def _heatgrid(data: dict, detail: dict) -> str:
    runs = grid_runs(data)
    grades = {r["run"]: {row["id"]: row["grade"] for row in r["per_question"]} for r in runs}
    retained = data["retained40"]
    head = "".join(f'<th scope="col"><a href="#run-{_slug(r["run"])}">{esc(r["run"])}</a></th>' for r in runs)
    rows = []
    for qid in retained["ids"]:
        cells = []
        for i, r in enumerate(runs):
            grade = grades[r["run"]].get(qid)
            if grade is None:
                cells.append('<td class="g g-none"></td>')
            else:
                cells.append(f'<td class="g g-{grade}" tabindex="0" data-q="{esc(qid)}" data-r="{i}" '
                             f'aria-label="{esc(qid)} {esc(r["run"])}: {grade}">{GRADE_LETTER[grade]}</td>')
        rows.append(f'<tr><th scope="row" class="qid" tabindex="0" data-q="{esc(qid)}">{esc(qid)}</th>'
                    f'{"".join(cells)}</tr>')
    legend = (
        '<p class="legend"><span class="g g-pass">P</span> pass <span class="g g-fail">F</span> fail '
        '<span class="g g-review">R</span> review <span class="g g-none"></span> not run in that run. '
        'Grader grades only; hand grades are not recorded per question. Hover a cell for a summary; '
        'click a cell or a question id for details.</p>'
    )
    return (
        f'<p class="note">{esc(retained["note"])}</p>' + legend
        + f'<div class="wrap"><table class="grid"><thead><tr><th scope="col">Question</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        '<div id="qpanel" class="qpanel" hidden></div>'
        '<div id="qtip" class="qtip" role="tooltip" hidden></div>'
        + _qdetail_block(detail)
    )


def _models(data: dict, exps: list[dict]) -> str:
    rows = []
    for m in data["models"]:
        used = " ".join(f'<a href="#exp-{esc(e["id"])}">{esc(e["id"])}</a>'
                        for e in exps if m["id"] in e["models"]) or "-"
        rows.append([f'{esc(m["name"])}<br><code>{esc(m["id"])}</code>', esc(m["role"]), esc(m["size"]),
                     esc(m["where"]), esc(m["status"]), esc(m["verdict"]), used])
    return _table(["Model or engine", "Role", "Size and quantisation", "Where it ran", "Status", "Verdict",
                   "Experiments"], rows)


def _not_tested(data: dict) -> str:
    rows = [[esc(t["name"]), esc(t["role"]), esc(t["protocol"]), esc(t["priority"])] for t in data["not_tested"]]
    return _table(["Candidate", "Role", "Test protocol", "Priority"], rows)


def _add_record() -> str:
    steps = [
        "Run the QA benchmark so that work/bench-json/&lt;run&gt;/ holds results.jsonl and summary.md.",
        "Add one object to runs in docs/history/history.json: the same run name, its experiment, model, "
        "machine, dataset, status, config, hand grades in hand, and appended_questions for any id added after "
        "the run started. Set status to running while the run is in progress.",
        "Regenerate: python -m docsearch.history --curated docs/history/history.json "
        "--bench work/bench-json --out work/history.html",
        "Run the tests: python -m unittest discover -s tests",
    ]
    items = "".join(f"<li>{s}</li>" for s in steps)
    return (
        f"<ol>{items}</ol>"
        "<p>Never paste benchmark question text, reference answers or model answers into the repository. "
        "The generator reads only id, category, difficulty, grade and seconds from result rows, and validation "
        "rejects the forbidden keys listed in history.py.</p>"
    )


def render(data: dict, texts: dict | None = None, answers: dict | None = None) -> str:
    models = {m["id"]: m for m in data["models"]}
    detail = _qdetail(data, texts, answers)
    body = "".join([
        _header(data),
        "<main>",
        _overview(data),
        _section("timeline", "Timeline", f"{len(data['experiments'])} experiments, "
                 f"{len(data['milestones'])} milestones", _timeline(data)),
        _section("leaderboard", "Leaderboard", f"{len(data['runs'])} QA runs", _leaderboard(data)),
        _section("experiments", "Experiments", f"{len(data['experiments'])} cards",
                 "".join(_experiment(e, data, models) for e in data["experiments"])),
        _section("heatgrid", "Question grid", f"{len(data['retained40']['ids'])} questions", _heatgrid(data, detail)),
        _section("models", "Models and engines", f"{len(data['models'])} entries", _models(data, data["experiments"])),
        _section("findings", "Findings so far", "", _ul(data["findings"])),
        _section("not-tested", "Not yet tested", "", _not_tested(data)),
        _section("caveats", "Caveats", "", _ul(data["caveats"])
                 + _details("sub", None, "Open items", _ul(data["open_items"]))),
        _section("add-record", "How to add a record", "", _add_record()),
        "</main>",
    ])
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count(PLACEHOLDER) != 1:
        raise ValueError(f"{TEMPLATE} must contain {PLACEHOLDER} exactly once")
    return template.replace(PLACEHOLDER, body)


def build(curated: Path, bench: Path | None, out: Path, benchmark: Path | None = None) -> int:
    data = json.loads(curated.read_text(encoding="utf-8"))
    errors = validate(data)
    if errors:
        raise SystemExit("invalid curated history:\n  " + "\n  ".join(errors))
    if bench is not None and bench.is_dir():
        for name in merge_bench(data, bench):
            print(f"history: no curated run for {name}; skipped", file=sys.stderr)
        errors = validate(data)
        if errors:
            raise SystemExit("invalid history after merging results:\n  " + "\n  ".join(errors))
    elif bench is not None:
        print(f"history: {bench} not found; building from curated data only", file=sys.stderr)
    out.parent.mkdir(parents=True, exist_ok=True)
    texts = answers = None
    if benchmark is not None:
        texts = read_benchmark(benchmark)
        if bench is not None and bench.is_dir():
            answers = read_answers(bench)
    page = render(data, texts, answers)
    out.write_text(page, encoding="utf-8")
    return len(page.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--curated", type=Path, default=Path("docs/history/history.json"))
    parser.add_argument("--bench", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("work/history.html"))
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="private benchmark JSON: also embeds question, reference and engine answer text")
    args = parser.parse_args(argv)
    if args.benchmark is not None:
        if args.out.resolve().is_relative_to(DOCS_DIR):
            print(f"history: refusing to write private benchmark text under {DOCS_DIR}; "
                  "choose an --out outside docs/", file=sys.stderr)
            return 2
        print("history: output contains private benchmark text; do not commit or publish", file=sys.stderr)
    size = build(args.curated, args.bench, args.out, args.benchmark)
    print(f"wrote {args.out} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
