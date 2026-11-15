const $ = (id) => document.getElementById(id);
const drop = $("drop"), input = $("file"), statusEl = $("status");
let previewUrl = null;

fetch("/api/warm", { method: "POST" }).catch(() => {});

function setStatus(text, err = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("err", err);
}

function fmtPct(p) {
  if (p >= 0.999) return ">99.9%";
  if (p <= 0.001) return "<0.1%";
  return (p * 100).toFixed(p > 0.99 || p < 0.01 ? 1 : 0) + "%";
}

// ---------- nav + popups ----------
function setActive(name) {
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.go === name));
}
document.querySelectorAll(".nav-btn").forEach((b) => b.addEventListener("click", () => {
  const go = b.dataset.go;
  if (go === "detect") return showUpload();
  setActive(go);
  $(go).showModal();
}));
document.querySelectorAll(".pop").forEach((d) => {
  d.addEventListener("close", () => setActive("detect"));
  d.addEventListener("click", (e) => { if (e.target === d) d.close(); });  // click backdrop
  d.querySelector(".x").addEventListener("click", () => d.close());
});

// ---------- upload ----------
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault(); drop.classList.remove("over");
  if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]);
});
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
input.addEventListener("change", () => { if (input.files[0]) analyze(input.files[0]); input.value = ""; });
document.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) analyze(item.getAsFile());
});
$("again").addEventListener("click", () => { showUpload(); input.click(); });

function showUpload() {
  setActive("detect");
  $("result").hidden = true; $("upload").hidden = false;
  document.body.className = "landing";
}

async function analyze(file) {
  if (file.size > 15 * 1024 * 1024) return setStatus("That file is over 15 MB.", true);
  const fd = new FormData();
  fd.append("file", file, file.name || "pasted.png");
  drop.classList.add("busy"); $("btnText").textContent = "Detecting…";
  setStatus("Analyzing with both models…");
  const slow = setTimeout(() => setStatus("Starting the GPU. The first check after a quiet period can take a minute…"), 8000);
  let res, body;
  try {
    res = await fetch("/api/analyze", { method: "POST", body: fd });
    body = await res.json().catch(() => ({}));
  } catch {
    body = null;
  } finally {
    clearTimeout(slow);
    drop.classList.remove("busy"); $("btnText").textContent = "Upload & Detect";
  }
  if (!body) return setStatus("Network error. Please try again.", true);
  if (!res.ok) return setStatus(body.detail || `Error ${res.status}`, true);
  setStatus("");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  render(body, previewUrl);
}

// ---------- result ----------
function fill(prefix, cardId, p, verdict) {
  $(cardId).className = "score-card " + verdict.key + (cardId === "cardSwin" ? " alt" : "");
  $(prefix + "Pct").textContent = fmtPct(p);
  $(prefix + "Verdict").textContent = verdict.label;
  const bar = $(prefix + "Bar");
  bar.style.width = "0";
  requestAnimationFrame(() => requestAnimationFrame(() => { bar.style.width = Math.max(p * 100, 1.5) + "%"; }));
}

function render(r, src) {
  $("preview").src = src;
  $("frameBg").style.backgroundImage = `url("${src}")`;
  fill("swin", "cardSwin", r.likelihood, r.verdict);
  if (r.resnet) fill("ours", "cardOurs", r.resnet.likelihood, r.resnet.verdict);
  $("cardOurs").hidden = !r.resnet;
  $("upload").hidden = true; $("result").hidden = false;
  document.body.className = "result";
  setActive("detect");
  window.scrollTo({ top: 0, behavior: "smooth" });
}
