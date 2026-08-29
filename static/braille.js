const statusEl = document.getElementById("status");
const portSelect = document.getElementById("portSelect");
const connectStatus = document.getElementById("connectStatus");
const uploadStatus = document.getElementById("uploadStatus");
const textInput = document.getElementById("textInput");
const textPreview = document.getElementById("textPreview");
const cellGlyph = document.getElementById("cellGlyph");
const cellLetter = document.getElementById("cellLetter");
const cellCaption = document.getElementById("cellCaption");
const btnPlay = document.getElementById("btnPlay");
const speedSlider = document.getElementById("speedSlider");
const speedLabel = document.getElementById("speedLabel");
const pdfInput = document.getElementById("pdfInput");
const fileDrop = document.getElementById("fileDrop");

const POLL_MS = 300;

function setTopStatus(connected) {
  statusEl.textContent = connected ? "connected" : "disconnected";
  statusEl.className = "status " + (connected ? "status-live" : "status-connecting");
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  return res.json();
}

async function refreshPorts() {
  const data = await api("/braille/ports");
  const prev = portSelect.value;
  portSelect.innerHTML = "";
  data.ports.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = p;
    portSelect.appendChild(opt);
  });
  if (data.ports.includes(prev)) portSelect.value = prev;
  if (!data.ports.length) {
    const opt = document.createElement("option");
    opt.textContent = "No serial ports found";
    opt.disabled = true;
    portSelect.appendChild(opt);
  }
}

async function connect() {
  const port = portSelect.value;
  if (!port) return;
  connectStatus.textContent = "Connecting…";
  connectStatus.className = "connect-status";
  const data = await api("/braille/connect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ port }),
  });
  if (data.connected) {
    connectStatus.textContent = `Connected to ${data.port}`;
    connectStatus.className = "connect-status ok";
  } else {
    connectStatus.textContent = "Failed: " + (data.error || "unknown error");
    connectStatus.className = "connect-status bad";
  }
  applyStatus(data);
}

async function disconnect() {
  const data = await api("/braille/disconnect", { method: "POST" });
  connectStatus.textContent = "Disconnected";
  connectStatus.className = "connect-status";
  applyStatus(data);
}

async function loadText() {
  const data = await api("/braille/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: textInput.value }),
  });
  applyStatus(data);
}

async function uploadPdf(file) {
  uploadStatus.textContent = "Uploading…";
  uploadStatus.className = "connect-status";
  const form = new FormData();
  form.append("pdf", file);
  const res = await fetch("/braille/upload", { method: "POST", body: form });
  const data = await res.json();
  if (data.error) {
    uploadStatus.textContent = data.error;
    uploadStatus.className = "connect-status bad";
    return;
  }
  uploadStatus.textContent = `Loaded ${data.length} characters from PDF`;
  uploadStatus.className = "connect-status ok";
  applyStatus(data);
}

async function control(action, extra) {
  const data = await api("/braille/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, ...extra }),
  });
  applyStatus(data);
}

function applyStatus(data) {
  setTopStatus(data.connected);
  cellGlyph.textContent = data.current_unicode || "⠀";
  cellLetter.textContent = data.current_char || "–";
  cellCaption.textContent = `${data.index} / ${data.length}`;
  btnPlay.textContent = data.playing ? "⏸ Pause" : "▶ Play";
  textPreview.innerHTML = renderPreview(data.text, data.index);
}

function renderPreview(text, index) {
  if (!text) return "(nothing loaded yet)";
  const before = escapeHtml(text.slice(0, index));
  const at = escapeHtml(text[index] || "");
  const after = escapeHtml(text.slice(index + 1));
  return `${before}<mark>${at}</mark>${after}`;
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

async function poll() {
  try {
    const data = await api("/braille/status");
    applyStatus(data);
  } catch (err) {
    // server not reachable this tick; try again next interval
  }
  setTimeout(poll, POLL_MS);
}

document.getElementById("btnRefreshPorts").addEventListener("click", refreshPorts);
document.getElementById("btnConnect").addEventListener("click", connect);
document.getElementById("btnDisconnect").addEventListener("click", disconnect);
document.getElementById("btnSendText").addEventListener("click", loadText);
document.getElementById("btnPlay").addEventListener("click", () => {
  control(btnPlay.textContent.includes("Play") ? "play" : "pause");
});
document.getElementById("btnNext").addEventListener("click", () => control("next"));
document.getElementById("btnPrev").addEventListener("click", () => control("prev"));
document.getElementById("btnReset").addEventListener("click", () => control("reset"));

speedSlider.addEventListener("input", () => {
  speedLabel.textContent = `${speedSlider.value}ms / letter`;
});
speedSlider.addEventListener("change", () => {
  control("speed", { delay_ms: Number(speedSlider.value) });
});

fileDrop.addEventListener("click", () => pdfInput.click());
pdfInput.addEventListener("change", () => {
  if (pdfInput.files[0]) uploadPdf(pdfInput.files[0]);
});
fileDrop.addEventListener("dragover", (e) => e.preventDefault());
fileDrop.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (file) uploadPdf(file);
});

refreshPorts();
poll();
