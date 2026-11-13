const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor(tag = "") { this.tagName = tag; this.listeners = {}; this.children = []; this.attributes = {}; this.classList = { add() {}, remove() {} }; this.style = { setProperty() {} }; this.hidden = false; this.disabled = false; this.value = ""; this.textContent = ""; this.files = []; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  setAttribute(name, value) { this.attributes[name] = value; }
  append(...nodes) { this.children.push(...nodes); }
  prepend(...nodes) { this.children.unshift(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  querySelector() { return new Element(); }
  click() { for (const callback of this.listeners.click || []) callback({ target: this }); }
  focus() {}
  dispatchEvent(event) { for (const callback of this.listeners[event.type] || []) callback({ target: this }); }
  get childElementCount() { return this.children.length; }
}
const ids = ["file-picker", "drop-zone", "file-list", "upload-button", "upload-status", "upload-error", "upload-progress", "upload-clock", "upload-note", "question", "ask-button", "question-error", "history", "new-upload", "collection-state", "question-count", "question-form", "image-dialog", "dialog-image", "close-image", "file-guidance", "qa-panel"];
const elements = new Map(ids.map((id) => [id, new Element()]));
const document = {
  querySelector(selector) { return elements.get(selector.slice(1)) || new Element(); },
  createElement() { return new Element(); },
};
let responses = [];
const context = { document, window: { location: { origin: "http://localhost" } }, fetch: async (url) => {
  const response = responses.length ? responses.shift() : { ok: true, json: async () => ({ csrf_token: "token", limits: { max_files: 10, max_file_bytes: 10, max_batch_bytes: 20 } }) };
  return typeof response === "function" ? response(url) : response;
}, Headers, FormData: class { append() {} }, URL, Set, Map, Array, String, Number, Event, setTimeout, clearTimeout, console };
vm.runInNewContext(fs.readFileSync("docsearch/static/progress.js", "utf8"), { window: context.window, module: { exports: {} }, performance, setInterval, clearInterval });
context.window.DocsearchProgress = require("../docsearch/static/progress.js");
vm.runInNewContext(fs.readFileSync("docsearch/static/app.js", "utf8"), context);
const fire = (id, event, value = {}) => elements.get(id).listeners[event].forEach((callback) => callback({ target: elements.get(id), preventDefault() {}, ...value }));
const response = (body, ok = true, status = 200) => ({ ok, status, json: async () => body });
const html = fs.readFileSync("docsearch/static/index.html", "utf8");
assert.ok(html.indexOf("/static/style.css") < html.indexOf("/static/progress.js"));
assert.ok(html.indexOf("/static/progress.js") < html.indexOf("/static/app.js"));
let scheduled;
context.setTimeout = (callback) => { scheduled = callback; return 1; };
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise((resolve) => setImmediate(resolve)); };

(async () => {
  await flush();
  const picker = elements.get("file-picker");
  picker.files = [{ name: "bad.exe", size: 2, lastModified: 1 }]; fire("file-picker", "change");
  assert.equal(elements.get("upload-button").disabled, true);
  assert.match(elements.get("upload-error").textContent, /Unsupported/);
  elements.get("file-list").children[0].children[1].click();
  assert.equal(elements.get("upload-button").disabled, true);
  assert.equal(elements.get("upload-error").hidden, true);
  assert.match(elements.get("upload-status").textContent, /Choose drawings/);

  picker.files = [{ name: "valid.pdf", size: 2, lastModified: 2 }]; fire("file-picker", "change");
  responses.push(response({ detail: "upload rejected" }, false, 413));
  const upload = elements.get("upload-button").listeners.click[0]({ target: elements.get("upload-button") });
  await upload;
  assert.equal(elements.get("upload-error").textContent, "upload rejected");
  assert.equal(elements.get("upload-status").textContent, "Upload failed. Check the files and try again.");

  responses.push(response({ id: "c1", status: "failed", error: "backend diagnostic", stage: "failed" }));
  const accepted = elements.get("upload-button").listeners.click[0]({ target: elements.get("upload-button") });
  await accepted;
  assert.equal(elements.get("file-list").children[0].children[1].disabled, true);
  await flush();
  assert.equal(elements.get("upload-error").textContent, "backend diagnostic");
  assert.equal(elements.get("upload-status").textContent, "Drawing processing failed.");
  assert.equal(elements.get("new-upload").hidden, false);
  elements.get("new-upload").click();
  picker.files = [{ name: "next.pdf", size: 3, lastModified: 3 }]; fire("file-picker", "change");
  assert.equal(elements.get("file-list").children.length, 1);
  assert.match(elements.get("upload-status").textContent, /1 file/);

  responses.push(response({ id: "c2", status: "ready" }));
  await elements.get("upload-button").listeners.click[0]({ target: elements.get("upload-button") });
  assert.equal(elements.get("upload-status").textContent, "Drawings are ready. You can ask a question.");
  assert.equal(elements.get("collection-state").textContent, "Ready to ask");
  assert.equal(elements.get("qa-panel").attributes["aria-disabled"], "false");
  assert.equal(elements.get("question").disabled, false);
  elements.get("new-upload").click();
  picker.files = [{ name: "flow.pdf", size: 2, lastModified: 4 }]; fire("file-picker", "change");
  responses.push(response({ id: "c3", status: "queued", queued_seconds: 0, progress: { percent: null } }));
  responses.push(response({ status: "running", stage: "ocr", run_seconds: 20, progress: { stage: "ocr", percent: 30 } }));
  responses.push(response({ status: "ready", stage: "ready", run_seconds: 95, progress: { percent: 100 } }));
  const uploadFlow = elements.get("upload-button").listeners.click[0]({ target: elements.get("upload-button") });
  await uploadFlow;
  await flush();
  while (scheduled) { const callback = scheduled; scheduled = null; callback(); await flush(); }
  assert.match(elements.get("upload-status").textContent, /Processed in 1:35/);
  assert.equal(elements.get("upload-progress").attributes["aria-valuenow"], "100");
  responses.push(response({ id: "q1", status: "queued", question: "What is shown?", queued_seconds: 0, progress: { percent: null } }));
  responses.push(response({ id: "q1", status: "ready", question: "What is shown?", run_seconds: 95, result: {}, progress: { percent: 100 } }));
  const questionFlow = elements.get("question-form").listeners.submit[0]({ preventDefault() {} });
  await questionFlow;
  await flush();
  assert.match(elements.get("history").children[0].children[3].textContent, /Answered in 1:35/);
  console.log("browser UI state regression passed");
})().catch((error) => { console.error(error); process.exitCode = 1; });
