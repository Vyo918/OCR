const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

class Element {
  constructor() { this.listeners = {}; this.children = []; this.attributes = {}; this.classList = { add() {}, remove() {} }; this.hidden = false; this.disabled = false; this.value = ""; this.textContent = ""; }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  setAttribute(name, value) { this.attributes[name] = value; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  querySelector() { return new Element(); }
  click() { for (const callback of this.listeners.click || []) callback({ target: this }); }
  focus() {}
}
const ids = ["file-picker", "drop-zone", "file-list", "upload-button", "upload-status", "upload-error", "question", "ask-button", "question-error", "history", "new-upload", "collection-state", "question-count", "question-form", "image-dialog", "dialog-image", "close-image", "file-guidance"];
const elements = new Map(ids.map((id) => [id, new Element()]));
const document = {
  querySelector(selector) { return elements.get(selector.slice(1)) || new Element(); },
  createElement() { return new Element(); },
};
const context = { document, window: { location: { origin: "http://localhost" } }, fetch: async () => ({ ok: true, json: async () => ({ csrf_token: "token", limits: { max_files: 10, max_file_bytes: 10, max_batch_bytes: 20 } }) }), Headers, FormData, URL, Set, Map, Array, String, Number, Event, setTimeout, clearTimeout, console };
vm.runInNewContext(fs.readFileSync("docsearch/static/app.js", "utf8"), context);
const fire = (id, event, value = {}) => elements.get(id).listeners[event].forEach((callback) => callback({ target: elements.get(id), preventDefault() {}, ...value }));
(async () => {
  await new Promise((resolve) => setImmediate(resolve));
  const picker = elements.get("file-picker");
  picker.files = [{ name: "bad.exe", size: 2, lastModified: 1 }]; fire("file-picker", "change");
  assert.equal(elements.get("upload-button").disabled, true);
  assert.match(elements.get("upload-error").textContent, /Unsupported/);
  elements.get("file-list").children[0].children[1].click();
  assert.equal(elements.get("upload-button").disabled, true); // no selection
  assert.equal(elements.get("upload-error").hidden, true);
  assert.match(elements.get("upload-status").textContent, /Choose drawings/);
  console.log("browser UI selection validation regression passed");
})().catch((error) => { console.error(error); process.exitCode = 1; });
