"use strict";

function formatDuration(seconds) {
  const value = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const remainder = value % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`
    : `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function stageLabel(kind, stage) {
  const labels = kind === "collection"
    ? { queued: "Waiting in the queue", prepare: "Preparing", ocr: "Reading text (OCR)",
      index: "Building the search index" }
    : { queued: "Waiting in the queue", retrieve: "Searching the drawings", reason: "Reasoning",
      answer: "Writing the answer" };
  return labels[stage] || "Processing";
}

function estimateRemaining(runSeconds, percent) {
  if (!Number.isFinite(percent) || percent < 15 || percent > 99 || runSeconds < 20) return null;
  return runSeconds * (100 - percent) / percent;
}

function smoothEstimate(previous, next) {
  if (next === null) return null;
  if (previous === null) return next;
  return Math.max(previous * 0.75, Math.min(previous * 1.25, next));
}

function describeEstimate(seconds, options = {}) {
  let text;
  if (seconds < 45) text = "Estimate: under a minute left (rough)";
  else if (seconds < 5400) text = `Estimate: about ${Math.round(seconds / 60)} min left (rough)`;
  else text = `Estimate: about ${Math.round(seconds / 3600)} h left (rough)`;
  return options.reasoning ? `${text.replace(" (rough)", "")} (estimated, reasoning length varies)` : text;
}

function createMeter({ kind, statusNode, barNode, noteNode, clockNode, now = () => performance.now(),
  setTimer = (fn, ms) => setInterval(fn, ms), clearTimer = (id) => clearInterval(id) }) {
  let timer = null, base = 0, anchor = 0, elapsed = 0, lastPercent = null, progressAt = 0;
  let estimate = null, stopped = false, label = "";
  const clock = () => { if (clockNode) clockNode.textContent = formatDuration(elapsed + Math.max(0, now() - anchor) / 1000); };
  const refresh = () => {
    clock();
    if (noteNode && now() - progressAt >= 60000) noteNode.textContent = "Still working";
  };
  const start = () => {
    stopped = false; anchor = now(); base = 0; elapsed = 0;
    statusNode.textContent = kind === "collection" ? "Uploading" : "Submitting";
    clock();
    if (timer === null) timer = setTimer(refresh, 1000);
  };
  const sync = (job) => {
    const progress = job.progress || {};
    const seconds = job.status === "queued" ? job.queued_seconds : job.run_seconds;
    elapsed = Number.isFinite(seconds) ? Math.max(0, seconds) : elapsed;
    anchor = now();
    label = stageLabel(kind, job.stage || (job.status === "queued" ? "queued" : ""));
    statusNode.textContent = label;
    clock();
    const percent = Number.isFinite(progress.percent) ? progress.percent : null;
    barNode.setAttribute("role", "progressbar");
    barNode.setAttribute("aria-valuemin", "0"); barNode.setAttribute("aria-valuemax", "100");
    if (percent === null) {
      barNode.removeAttribute?.("aria-valuenow"); barNode.setAttribute("aria-valuetext", label);
      barNode.classList.add("indeterminate"); barNode.style.setProperty("--progress", "0%");
    } else {
      barNode.setAttribute("aria-valuenow", percent); barNode.setAttribute("aria-valuetext", `${percent}%`);
      barNode.classList.remove("indeterminate"); barNode.style.setProperty("--progress", `${percent}%`);
    }
    if (percent !== lastPercent) { lastPercent = percent; progressAt = now(); }
    const nextEstimate = estimateRemaining(elapsed, percent);
    estimate = nextEstimate === null ? null : smoothEstimate(estimate, nextEstimate);
    if (noteNode) noteNode.textContent = now() - progressAt >= 60000 ? "Still working"
      : estimate !== null ? describeEstimate(estimate, { reasoning: kind === "question" && job.stage === "reason" })
        : percent !== null && percent < 15 && job.status === "running" && elapsed >= 10 ? "Estimating..." : "";
  };
  const finish = (job) => {
    stop();
    const seconds = Number.isFinite(job.run_seconds) ? job.run_seconds : elapsed + Math.max(0, now() - anchor) / 1000;
    statusNode.textContent = job.status === "failed" ? `Stopped after ${formatDuration(seconds)}`
      : `${kind === "collection" ? "Processed in" : "Answered in"} ${formatDuration(seconds)}`;
    if (job.status === "ready") {
      barNode.classList.remove("indeterminate"); barNode.setAttribute("aria-valuenow", "100");
      barNode.setAttribute("aria-valuetext", "100%"); barNode.style.setProperty("--progress", "100%");
    }
    if (clockNode) clockNode.textContent = "";
  };
  const stop = () => { if (timer !== null) clearTimer(timer); timer = null; stopped = true; };
  return { start, sync, finish, stop };
}

const DocsearchProgress = { formatDuration, stageLabel, estimateRemaining, smoothEstimate, describeEstimate, createMeter };
if (typeof module !== "undefined" && module.exports) module.exports = DocsearchProgress;
if (typeof window !== "undefined") window.DocsearchProgress = DocsearchProgress;
