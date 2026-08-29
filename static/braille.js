const statusEl = document.getElementById("status");
const portSelect = document.getElementById("portSelect");
const connectStatus = document.getElementById("connectStatus");
const uploadStatus = document.getElementById("uploadStatus");
const textInput = document.getElementById("textInput");
const textPreview = document.getElementById("textPreview");
const dotGrid = document.getElementById("dotGrid");
const cellHero = document.getElementById("cellHero");
const cellLetter = document.getElementById("cellLetter");
const cellCaption = document.getElementById("cellCaption");
const btnPlay = document.getElementById("btnPlay");
const speedSlider = document.getElementById("speedSlider");
const speedLabel = document.getElementById("speedLabel");
const pdfInput = document.getElementById("pdfInput");
const fileDrop = document.getElementById("fileDrop");

const POLL_MS = 300;

// Standard Grade-1 English braille dot numbering:
//   1 4
//   2 5
//   3 6
const BRAILLE_DOTS = {
  A: [1], B: [1, 2], C: [1, 4], D: [1, 4, 5], E: [1, 5],
  F: [1, 2, 4], G: [1, 2, 4, 5], H: [1, 2, 5], I: [2, 4], J: [2, 4, 5],
  K: [1, 3], L: [1, 2, 3], M: [1, 3, 4], N: [1, 3, 4, 5], O: [1, 3, 5],
  P: [1, 2, 3, 4], Q: [1, 2, 3, 4, 5], R: [1, 2, 3, 5], S: [2, 3, 4], T: [2, 3, 4, 5],
  U: [1, 3, 6], V: [1, 2, 3, 6], W: [2, 4, 5, 6], X: [1, 3, 4, 6], Y: [1, 3, 4, 5, 6],
  Z: [1, 3, 5, 6],
};

function renderDots(letter) {
  const active = new Set(BRAILLE_DOTS[letter] || []);
  dotGrid.querySelectorAll("circle").forEach((circle) => {
    const dotNum = Number(circle.dataset.dot);
    circle.classList.toggle("dot-on", active.has(dotNum));
    circle.classList.toggle("dot-off", !active.has(dotNum));
  });
}

function setTopStatus(connected) {
  statusEl.querySelector(".label").textContent = connected ? "connected" : "disconnected";
  statusEl.className = "status-badge state-" + (connected ? "live" : "idle");
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
  renderDots(data.current_char);
  cellLetter.textContent = data.current_char || "–";
  cellCaption.textContent = `${data.index} / ${data.length}`;
  btnPlay.textContent = data.playing ? "⏸ Pause" : "▶ Play";
  textPreview.innerHTML = renderPreview(data.text, data.index);

  const signal = Boolean(data.current_char);
  cellHero.classList.toggle("has-signal", signal);
  cellHero.classList.toggle("hero-pulse", signal && data.playing);
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
