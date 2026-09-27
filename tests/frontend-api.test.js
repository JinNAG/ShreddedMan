const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const vm = require("node:vm");
const { once } = require("node:events");
const { before, beforeEach, after, test } = require("node:test");
const express = require("express");
const multer = require("multer");

const root = path.resolve(__dirname, "..");
const html = fs.readFileSync(path.join(root, "public/index.html"), "utf8");
const browserScript = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const imageBytes = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=", "base64");
const id = "0123456789abcdef0123456789abcdef";
const statusURL = `/api/submissions/${id}`;
const documentURL = `${statusURL}/document`;
const reportURL = `${statusURL}/join-report`;
const quietConsole = { log() {}, error() {} };
let backendServer, frontendServer, temporary, origin;
let mode, uploads, polls, documentRequests;

function job(status) {
  return {
    submission_id: id, status, status_url: statusURL,
    document_url: status === "complete" ? documentURL : null,
    join_report_url: status === "complete" ? reportURL : null,
    review_required: status === "complete" && mode === "review",
    error: status === "failed" ? "Document reconstruction failed. Check the photos and try again." : null
  };
}

before(async () => {
  const backend = express();
  backend.post("/api/submissions", multer().any(), (req, res) => {
    uploads.push({ files: req.files, rotation: req.body.rotation });
    if (mode === "offline") return req.socket.destroy();
    if (mode === "busy") return res.status(503).set("Retry-After", "5").json({ detail: "The processing queue is full." });
    if (mode === "oldServer") return res.type("text").send("Upload successful");
    res.status(202).set("Location", statusURL).json(job("queued"));
  });
  backend.get(statusURL, (req, res) => {
    polls += 1;
    res.json(job(mode === "failed" ? "failed" : polls === 1 ? "normalized" : "complete"));
  });
  backend.get(documentURL, (req, res) => {
    documentRequests += 1;
    if (mode === "missingImage") return res.status(404).json({ detail: "Document not found." });
    if (polls < 2) return res.status(409).json({ detail: "Document not available yet." });
    res.type("png").set("Content-Disposition", 'inline; filename="document.png"').send(imageBytes);
  });
  backend.get(reportURL, (req, res) => res.type("html").send("<h1>Join report</h1>"));
  backend.get(`${statusURL}/document.png`, (req, res) => res.type("png").send(imageBytes));
  backend.get(`${statusURL}/join_report.json`, (req, res) => res.json({ joins: [] }));
  backendServer = backend.listen(0, "127.0.0.1");
  await once(backendServer, "listening");

  temporary = fs.mkdtempSync(path.join(os.tmpdir(), "shreddedman-web-test-"));
  const runServer = vm.compileFunction(fs.readFileSync(path.join(root, "server.js"), "utf8"),
    ["__dirname", "require", "process", "console"]);
  runServer(temporary, name => {
    if (name !== "express") return require(name);
    return Object.assign(() => {
      const app = express();
      const listen = app.listen.bind(app);
      app.listen = () => (frontendServer = listen(0, "127.0.0.1"));
      return app;
    }, express);
  }, { env: { BACKEND_URL: `http://127.0.0.1:${backendServer.address().port}` } }, quietConsole);
  await once(frontendServer, "listening");
  origin = `http://127.0.0.1:${frontendServer.address().port}`;
});

after(async () => {
  for (const server of [frontendServer, backendServer]) {
    if (server) await new Promise(resolve => server.close(resolve));
  }
  if (temporary) {
    assert.equal(path.dirname(path.resolve(temporary)), path.resolve(os.tmpdir()));
    assert.ok(path.basename(temporary).startsWith("shreddedman-web-test-"));
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});

beforeEach(() => { mode = "success"; uploads = []; polls = documentRequests = 0; });

function browser() {
  let submit;
  const button = { disabled: false };
  const messages = [];
  const links = [];
  const nodes = {
    uploadForm: {
      addEventListener(name, listener) { if (name === "submit") submit = listener; },
      querySelector() { return button; }
    },
    Upload: { files: [
      new File([imageBytes], "photo.png", { type: "image/png" }),
      new File(["second photo"], "photo.jpeg", { type: "image/jpeg" })
    ] },
    originalImages: {
      style: {}, children: [],
      replaceChildren(...images) { this.children = images; }
    },
    originalMessage: { style: {} },
    editedMessage: {
      style: {},
      set textContent(value) { messages.push(value); this.value = value; },
      get textContent() { return this.value; }
    },
    editedImage: {
      style: { display: "none" },
      removeAttribute(name) { delete this[name]; },
      async decode() {
        assert.equal(this.style.display, "none", "Keep incomplete results hidden");
        const response = await fetch(origin + this.src);
        assert.equal(response.status, 200);
        assert.equal(response.headers.get("content-type"), "image/png");
        assert.deepEqual(Buffer.from(await response.arrayBuffer()), imageBytes);
      }
    },
    downloadImage: {
      hidden: true,
      removeAttribute(name) { delete this[name]; }
    }
  };
  vm.runInNewContext(browserScript, {
    document: {
      getElementById: id => nodes[id],
      createElement: tag => {
        assert.equal(tag, "img");
        return {};
      }
    },
    FormData, AbortSignal, console: quietConsole,
    URL: { createObjectURL: () => "blob:original-image", revokeObjectURL() {} },
    fetch: (url, options) => {
      assert.ok(url.startsWith("/"), "Use same-origin requests through Node");
      return fetch(origin + url, options);
    },
    setTimeout: callback => setTimeout(callback, 1)
  });
  return { nodes, button, messages, links, submit: () => submit({ preventDefault() {} }) };
}

test("all photos reach one submission and the finished PNG appears in Edited Image", async () => {
  const page = browser();
  const pending = page.submit();
  assert.equal(page.button.disabled, true);
  await page.submit(); // A double click must not enqueue another reconstruction.
  await pending;
  assert.equal(uploads.length, 1);
  assert.equal(uploads[0].rotation, "auto");
  assert.deepEqual(uploads[0].files.map(file => [file.fieldname, file.originalname, file.mimetype]),
    [["files", "photo.png", "image/png"], ["files", "photo.jpeg", "image/jpeg"]]);
  assert.deepEqual(uploads[0].files[0].buffer, imageBytes);
  assert.equal(polls, 2);
  assert.equal(documentRequests, 1);
  assert.equal(page.nodes.editedImage.src, documentURL);
  assert.equal(page.nodes.editedImage.style.display, "block");
  assert.equal(page.nodes.editedMessage.style.display, "none");
  assert.equal(page.nodes.originalImages.style.display, "grid");
  assert.deepEqual(page.nodes.originalImages.children.map(image => image.alt),
    ["Uploaded photo 1", "Uploaded photo 2"]);
  assert.equal(page.nodes.downloadImage.href, documentURL);
  assert.equal(page.nodes.downloadImage.hidden, false);
  assert.ok(page.messages.includes("Sorting strips and reconstructing your document..."));
  assert.equal(page.button.disabled, false);
});

test("a review warning links to the report and its page assets", async () => {
  mode = "review";
  const page = browser();
  await page.submit();
  assert.match(page.nodes.uploadStatus.textContent, /need review/);
  assert.equal(page.links.length, 1);
  assert.equal(page.links[0].href, reportURL);
  assert.equal(page.links[0].textContent, "View join report");
  assert.equal((await fetch(origin + reportURL)).status, 200);
  assert.equal((await fetch(origin + `${statusURL}/document.png`)).status, 200);
  assert.equal((await fetch(origin + `${statusURL}/join_report.json`)).status, 200);
});

test("a later failed job hides the old image and displays the backend error", async () => {
  const page = browser();
  await page.submit();
  mode = "failed";
  await page.submit();
  assert.equal(uploads.length, 2);
  assert.equal(documentRequests, 1);
  assert.equal(page.nodes.editedImage.style.display, "none");
  assert.equal(page.nodes.editedMessage.style.display, "block");
  assert.equal(page.nodes.downloadImage.hidden, true);
  assert.match(page.nodes.editedMessage.textContent, /Check the photos/);
  assert.equal(page.button.disabled, false);
});

test("an unavailable PNG does not report completion or leave an empty result box", async () => {
  mode = "missingImage";
  const page = browser();
  await page.submit();
  assert.match(page.nodes.editedMessage.textContent, /document could not be loaded/);
  assert.equal(page.nodes.editedMessage.style.display, "block");
  assert.equal(page.nodes.editedImage.style.display, "none");
  assert.equal(page.button.disabled, false);
});

test("queue, connection and old-server errors are shown instead of Ready for processing", async () => {
  for (const [scenario, message] of [["busy", /queue is full/], ["offline", /service is unavailable/], ["oldServer", /unexpected response/]]) {
    mode = scenario;
    const page = browser();
    await page.submit();
    assert.match(page.nodes.editedMessage.textContent, message);
    assert.equal(page.button.disabled, false);
    assert.equal(page.nodes.editedImage.style.display, "none");
  }
});

test("proxy preserves HTTP errors, no-cache headers and PNG disposition", async () => {
  assert.equal((await fetch(origin + documentURL)).status, 409);
  assert.equal((await fetch(origin + "/api/submissions/not-an-id")).status, 404);
  await fetch(origin + statusURL);
  const status = await fetch(origin + statusURL);
  assert.equal(status.headers.get("cache-control"), "no-store");
  const image = await fetch(origin + documentURL);
  assert.equal(image.headers.get("content-disposition"), 'inline; filename="document.png"');
  assert.deepEqual(Buffer.from(await image.arrayBuffer()), imageBytes);
  const empty = await fetch(origin + "/upload", { method: "POST", body: new FormData() });
  assert.equal(empty.status, 400);
  assert.equal(typeof (await empty.json()).detail, "string");
});
