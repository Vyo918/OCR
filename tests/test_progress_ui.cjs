const assert = require("node:assert/strict");
const progress = require("../docsearch/static/progress.js");

assert.equal(progress.formatDuration(-2), "0:00");
assert.equal(progress.formatDuration(65.9), "1:05");
assert.equal(progress.formatDuration(3600), "1:00:00");
assert.equal(progress.stageLabel("collection", "ocr"), "Reading text (OCR)");
assert.equal(progress.stageLabel("question", "reason"), "Reasoning");
assert.equal(progress.stageLabel("question", "future"), "Processing");
assert.equal(progress.estimateRemaining(9, 30), null);
assert.equal(progress.estimateRemaining(10, 4), null);
assert.equal(progress.estimateRemaining(10, 5), 190);
assert.equal(progress.estimateRemaining(10, 99), 10 / 99);
assert.equal(progress.estimateRemaining(10, 100), null);
assert.equal(progress.smoothEstimate(100, 200), 125);
assert.equal(progress.smoothEstimate(100, 20), 75);
assert.equal(progress.smoothEstimate(null, 20), 20);
assert.equal(progress.smoothEstimate(20, null), null);
assert.equal(progress.describeEstimate(30), "Estimate: under a minute left");
assert.equal(progress.describeEstimate(120), "Estimate: about 2 min left");
assert.equal(progress.describeEstimate(5400), "Estimate: about 2 h left");
assert.match(progress.describeEstimate(30, { reasoning: true }), /reasoning length varies/);

class Node {
  constructor() { this.attributes = {}; this.style = { setProperty() {} }; this.classList = { add() {}, remove() {} }; }
  setAttribute(key, value) { this.attributes[key] = value; }
}
function meter(kind) {
  let time = 0;
  let timer = null;
  const statusNode = new Node(), barNode = new Node(), noteNode = new Node(), clockNode = new Node();
  const result = progress.createMeter({ kind, statusNode, barNode, noteNode, clockNode,
    now: () => time, setTimer: (fn) => { timer = fn; return 1; }, clearTimer: () => { timer = null; } });
  return { ...result, statusNode, barNode, noteNode, clockNode,
    tick(seconds = 1) { time += seconds * 1000; if (timer) timer(); }, pending: () => timer !== null };
}
for (const [kind, finished] of [["collection", "Processed in 1:35"], ["question", "Answered in 1:35"]]) {
  const m = meter(kind);
  m.start();
  assert.equal(m.statusNode.textContent, kind === "collection" ? "Uploading" : "Submitting");
  m.sync({ status: "queued", queued_seconds: 0, progress: { percent: null } });
  m.sync({ status: "running", stage: kind === "collection" ? "ocr" : "reason", run_seconds: 20,
    progress: { percent: 30 } });
  assert.equal(m.statusNode.textContent, kind === "collection" ? "Reading text (OCR)" : "Reasoning");
  assert.equal(String(m.barNode.attributes["aria-valuenow"]), "30");
  assert.match(m.noteNode.textContent, /Estimate/);
  m.tick();
  assert.equal(m.clockNode.textContent, "0:21");
  m.sync({ status: "running", stage: kind === "collection" ? "ocr" : "reason", run_seconds: 20,
    progress: { percent: 30 } });
  m.tick(61);
  assert.equal(m.noteNode.textContent, "Still working");
  m.finish({ status: "ready", run_seconds: 95, progress: { percent: 100 } });
  assert.equal(m.statusNode.textContent, finished);
  assert.equal(m.barNode.attributes["aria-valuenow"], "100");
  assert.equal(m.pending(), false);
  m.stop(); m.stop();
}
const failed = meter("collection");
failed.start(); failed.finish({ status: "failed", run_seconds: 12 });
assert.equal(failed.statusNode.textContent, "Stopped after 0:12");
assert.equal(failed.pending(), false);
console.log("progress UI unit tests passed");
