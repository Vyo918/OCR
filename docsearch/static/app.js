"use strict";

const $ = (selector) => document.querySelector(selector);
const picker = $("#file-picker");
const dropZone = $("#drop-zone");
const fileList = $("#file-list");
const uploadButton = $("#upload-button");
const uploadStatus = $("#upload-status");
const uploadError = $("#upload-error");
const question = $("#question");
const askButton = $("#ask-button");
const questionError = $("#question-error");
const history = $("#history");
const supported = new Set([".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]);
let limits = { max_files: 10, max_file_bytes: 52428800, max_batch_bytes: 104857600 };
let csrfToken = "";
let selected = [];
let collection = null;
let uploadBusy = false;
let activeQuestions = 0;
let generation = 0;
let collectionPoll = null;
const jobPolls = new Set();

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let size = bytes / 1024, unit = 0;
  while (size >= 1024 && unit < units.length - 1) { size /= 1024; unit++; }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${units[unit]}`;
}
function showError(node, message) { node.textContent = message; node.hidden = !message; }
function detailMessage(data, fallback) {
  const detail = data && data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => typeof item === "string" ? item : item.msg || "Invalid request").join("; ");
  return fallback;
}
async function request(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.method && options.method !== "GET") headers.set("X-CSRF-Token", csrfToken);
  const response = await fetch(url, { ...options, headers, credentials: "same-origin" });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const error = new Error(detailMessage(data, `Request failed (${response.status})`));
    error.status = response.status;
    throw error;
  }
  return data;
}
function validateSelection() {
  const bytes = selected.reduce((total, file) => total + file.size, 0);
  const invalid = selected.find((file) => file.size <= 0 || !supported.has(file.name.slice(file.name.lastIndexOf(".")).toLowerCase()));
  const tooLarge = selected.find((file) => file.size > limits.max_file_bytes);
  const message = invalid ? (invalid.size <= 0 ? `${invalid.name} is empty.` : `Unsupported file type: ${invalid.name}`)
    : tooLarge ? `${tooLarge.name} exceeds the ${formatSize(limits.max_file_bytes)} per-file limit.`
    : selected.length > limits.max_files ? `Choose no more than ${limits.max_files} files.`
    : bytes > limits.max_batch_bytes ? `The selected files exceed the ${formatSize(limits.max_batch_bytes)} batch limit.` : "";
  showError(uploadError, message);
  uploadStatus.textContent = selected.length && !message
    ? `${selected.length} file${selected.length === 1 ? "" : "s"} selected · ${formatSize(bytes)} total`
    : collection ? uploadStatus.textContent : "Choose drawings to start a new batch.";
  return !message;
}
function renderFiles() {
  fileList.replaceChildren();
  selected.forEach((file, index) => {
    const item = document.createElement("li"); item.className = "file-item";
    const info = document.createElement("div"); info.className = "file-info";
    const name = document.createElement("strong"); name.textContent = file.name;
    const size = document.createElement("span"); size.textContent = formatSize(file.size);
    info.append(name, size);
    const remove = document.createElement("button"); remove.type = "button"; remove.className = "remove-button"; remove.textContent = "Remove"; remove.setAttribute("aria-label", `Remove ${file.name}`);
    remove.disabled = uploadBusy;
    remove.addEventListener("click", () => { selected.splice(index, 1); renderFiles(); });
    item.append(info, remove); fileList.append(item);
  });
  const valid = validateSelection();
  uploadButton.disabled = !csrfToken || !selected.length || !valid || uploadBusy || collection !== null;
  dropZone.setAttribute("aria-disabled", String(uploadBusy || collection !== null));
}
function addFiles(files) {
  if (uploadBusy || collection) return;
  const additions = Array.from(files);
  if (!additions.length) return;
  const seen = new Set(selected.map((file) => `${file.name}\0${file.size}\0${file.lastModified}`));
  const unique = additions.filter((file) => {
    const key = `${file.name}\0${file.size}\0${file.lastModified}`;
    if (seen.has(key)) return false;
    seen.add(key); return true;
  });
  selected.push(...unique);
  renderFiles();
}
picker.addEventListener("change", () => { addFiles(picker.files); picker.value = ""; });
dropZone.addEventListener("click", (event) => { if (event.target !== picker && !uploadBusy && !collection) picker.click(); });
dropZone.addEventListener("keydown", (event) => { if ((event.key === "Enter" || event.key === " ") && !uploadBusy && !collection) { event.preventDefault(); picker.click(); } });
for (const eventName of ["dragenter", "dragover"]) dropZone.addEventListener(eventName, (event) => { event.preventDefault(); if (!uploadBusy && !collection) dropZone.classList.add("drag-over"); });
for (const eventName of ["dragleave", "drop"]) dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove("drag-over"); });
dropZone.addEventListener("drop", (event) => addFiles(event.dataTransfer.files));

async function pollCollection(id, run, failures = 0) {
  clearTimeout(collectionPoll);
  if (run !== generation || !collection) return;
  try {
    const current = await request(`/api/collections/${encodeURIComponent(id)}`);
    if (run !== generation) return;
    showError(uploadError, "");
    collection = current;
    if (current.status === "ready") {
      uploadStatus.textContent = "Drawings are ready. You can ask a question.";
      $("#collection-state").textContent = "Ready to ask";
      $("#qa-panel").setAttribute("aria-disabled", "false");
      question.disabled = false; askButton.disabled = !question.value.trim();
      $("#new-upload").hidden = false;
      renderFiles(); return;
    }
    if (current.status === "failed") {
      question.disabled = true; askButton.disabled = true; $("#qa-panel").setAttribute("aria-disabled", "true");
      uploadStatus.textContent = "Drawing processing failed.";
      showError(uploadError, current.error || "The drawings could not be processed.");
      $("#collection-state").textContent = "Processing failed"; $("#new-upload").hidden = false; renderFiles(); return;
    }
    uploadStatus.textContent = current.status === "queued" ? "Waiting for the local processing queue…" : `Processing drawings · ${current.stage || "working"}`;
    collectionPoll = setTimeout(() => pollCollection(id, run), 1500);
  } catch (error) {
    if (run !== generation) return;
    failures++;
    uploadStatus.textContent = "Could not check processing status.";
    showError(uploadError, error.message);
    if ((error.status >= 400 && error.status < 500) || failures >= 5) {
      $("#collection-state").textContent = "Status unavailable";
      $("#new-upload").hidden = false;
      return;
    }
    collectionPoll = setTimeout(() => pollCollection(id, run, failures), 3000);
  }
}
uploadButton.addEventListener("click", async () => {
  if (uploadBusy || collection || !selected.length || !csrfToken || !validateSelection()) return;
  uploadBusy = true; uploadButton.textContent = "Uploading…"; showError(uploadError, ""); renderFiles();
  const body = new FormData(); selected.forEach((file) => body.append("files", file, file.name));
  try {
    const result = await request("/api/collections", { method: "POST", body });
    collection = result;
    $("#new-upload").hidden = false;
    uploadStatus.textContent = result.status === "ready" ? "Drawings are ready. You can ask a question." : result.status === "failed" ? "Drawing processing failed." : "Drawings uploaded; processing started…";
    $("#collection-state").textContent = result.status === "ready" ? "Ready to ask" : result.status === "failed" ? "Processing failed" : "Processing";
    if (result.status === "ready") { $("#qa-panel").setAttribute("aria-disabled", "false"); question.disabled = false; }
    else if (result.status === "failed") { question.disabled = true; askButton.disabled = true; $("#qa-panel").setAttribute("aria-disabled", "true"); showError(uploadError, result.error || "The drawings could not be processed."); }
    else pollCollection(result.id, generation);
  } catch (error) { showError(uploadError, error.message); uploadStatus.textContent = "Upload failed. Check the files and try again."; }
  finally { uploadBusy = false; uploadButton.textContent = "Upload drawings"; renderFiles(); }
});

function appendText(parent, tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; node.textContent = text; parent.append(node); return node; }
function makeEvidenceImage(image, figure, label) {
  if (!image || typeof image.url !== "string" || !image.url.startsWith("/api/")) return;
  const url = new URL(image.url, window.location.origin);
  if (url.origin !== window.location.origin) return;
  const img = document.createElement("img"); img.src = url.href; img.alt = label; img.loading = "lazy";
  const button = document.createElement("button"); button.type = "button"; button.className = "evidence-expand"; button.setAttribute("aria-label", `Enlarge ${label}`);
  button.append(img);
  button.addEventListener("click", () => { $("#dialog-image").src = url.href; $("#image-dialog").showModal(); });
  figure.append(button);
}
function renderResult(card, result) {
  appendText(card, "h4", "answer-label", "Answer");
  appendText(card, "p", "answer-text", result.answer || "No answer was returned.");
  if (Array.isArray(result.evidence) && result.evidence.length) {
    appendText(card, "h4", "evidence-label", "Evidence");
    const list = document.createElement("div"); list.className = "evidence-list";
    result.evidence.forEach((item) => {
      const block = document.createElement("article"); block.className = "evidence-item";
      appendText(block, "p", "evidence-text", item.text || "");
      appendText(block, "span", "evidence-source", `Page / document: ${item.doc || "unknown"} · ${item.id || ""}`);
      list.append(block);
    }); card.append(list);
  }
  if (Array.isArray(result.images) && result.images.length) {
    const images = document.createElement("div"); images.className = "evidence-images";
    result.images.forEach((image) => { const figure = document.createElement("figure"); makeEvidenceImage(image, figure, `Evidence crop ${image.name || ""}`); if (figure.childElementCount) { appendText(figure, "figcaption", "", image.name || "View evidence crop"); images.append(figure); } });
    if (images.childElementCount) card.append(images);
  }
}
function pollQuestion(id, jobId, card, run) {
  let failures = 0;
  const tick = async () => {
    if (run !== generation) return;
    try {
      const job = await request(`/api/collections/${encodeURIComponent(id)}/questions/${encodeURIComponent(jobId)}`);
      failures = 0;
      if (run !== generation) return;
      if (job.status === "ready") { card.replaceChildren(); appendText(card, "h3", "asked-question", job.question || "Question"); renderResult(card, job.result || {}); return; }
      if (job.status === "failed") { card.replaceChildren(); appendText(card, "h3", "asked-question", job.question || "Question"); appendText(card, "p", "error", job.error || "The question could not be answered."); return; }
      const status = card.querySelector(".job-status"); status.textContent = job.status === "queued" ? "Waiting in the local queue…" : `Finding an answer · ${job.stage || "working"}`;
    } catch (error) {
      if (run !== generation) return;
      failures++;
      if (failures >= 5) { card.replaceChildren(); appendText(card, "p", "error", `Could not retrieve this answer: ${error.message}`); return; }
      card.querySelector(".job-status").textContent = "Temporary status issue · retrying…";
    }
    const handle = setTimeout(tick, failures ? 2500 : 1500); jobPolls.add(handle);
  };
  tick();
}
question.addEventListener("input", () => { $("#question-count").textContent = `${question.value.length} / 2000`; askButton.disabled = !collection || collection.status !== "ready" || !question.value.trim(); });
$("#question-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text || !collection || collection.status !== "ready" || activeQuestions) { if (!text) question.focus(); return; }
  activeQuestions++;
  const id = collection.id, run = generation;
  askButton.disabled = true; askButton.textContent = "Submitting…"; showError(questionError, "");
  const card = document.createElement("article"); card.className = "qa-card";
  appendText(card, "h3", "asked-question", text); appendText(card, "p", "job-status", "Submitting question…"); history.prepend(card);
  question.value = ""; question.dispatchEvent(new Event("input"));
  try {
    const job = await request(`/api/collections/${encodeURIComponent(id)}/questions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text }) });
    if (run === generation) pollQuestion(id, job.id, card, run);
  } catch (error) { if (run === generation) { card.replaceChildren(); appendText(card, "h3", "asked-question", text); appendText(card, "p", "error", error.message); showError(questionError, "Question was not submitted."); } }
  finally { if (run === generation) activeQuestions--; if (run === generation) { askButton.textContent = "Ask question →"; askButton.disabled = !collection || collection.status !== "ready" || !question.value.trim(); } }
});
$("#new-upload").addEventListener("click", () => {
  $("#qa-panel").setAttribute("aria-disabled", "true");
  generation++; clearTimeout(collectionPoll); jobPolls.forEach(clearTimeout); jobPolls.clear();
  activeQuestions = 0; collection = null; selected = []; history.replaceChildren(); question.value = ""; question.disabled = true; askButton.disabled = true;
  $("#question-count").textContent = "0 / 2000"; $("#collection-state").textContent = "Upload to begin";
  $("#new-upload").hidden = true; uploadStatus.textContent = "Choose drawings to start a new batch.";
  showError(uploadError, ""); showError(questionError, ""); renderFiles();
});

$("#close-image").addEventListener("click", () => $("#image-dialog").close());
$("#image-dialog").addEventListener("click", (event) => { if (event.target === event.currentTarget) event.currentTarget.close(); });

async function bootstrap() {
  try {
    const session = await request("/api/session");
    csrfToken = session.csrf_token; limits = { ...limits, ...session.limits };
    const extensions = Array.isArray(session.supported_extensions) ? session.supported_extensions.join(", ") : "supported files";
    $("#file-guidance").textContent = `Up to ${limits.max_files} files · ${formatSize(limits.max_file_bytes)} each · ${formatSize(limits.max_batch_bytes)} total · ${extensions}`;
    uploadStatus.textContent = "Ready when you are. Files are processed on this machine."; renderFiles();
  } catch (error) {
    uploadStatus.textContent = "The local service is unavailable.";
    showError(uploadError, `Could not initialize a secure session: ${error.message}`);
  }
}
bootstrap();
