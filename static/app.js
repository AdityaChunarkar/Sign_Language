const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const statusEl = document.getElementById("status");
const letterBadge = document.getElementById("letterBadge");
const progressFill = document.getElementById("progressFill");
const sentenceEl = document.getElementById("sentence");

const cameraSelect = document.getElementById("cameraSelect");

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

const PREDICT_INTERVAL_MS = 200;
let busy = false;
let currentStream = null;

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "status " + cls;
}

async function listCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((d) => d.kind === "videoinput");
  cameraSelect.innerHTML = "";
  cams.forEach((cam, i) => {
    const opt = document.createElement("option");
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Camera ${i + 1}`;
    cameraSelect.appendChild(opt);
  });
  return cams;
}

async function startCamera(deviceId) {
  try {
    if (currentStream) {
      currentStream.getTracks().forEach((t) => t.stop());
    }
    const constraints = {
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: 640, height: 480 }
        : { width: 640, height: 480 },
      audio: false,
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    currentStream = stream;
    video.srcObject = stream;
    await video.play();
    captureCanvas.width = video.videoWidth || 640;
    captureCanvas.height = video.videoHeight || 480;
    resizeOverlay();
    setStatus("live", "status-live");

    // Labels are only populated after permission is granted, so refresh
    // the dropdown once we actually have a stream.
    await listCameras();
    const activeTrack = stream.getVideoTracks()[0];
    const activeId = activeTrack && activeTrack.getSettings().deviceId;
    if (activeId) cameraSelect.value = activeId;
  } catch (err) {
    setStatus("camera error: " + err.message, "status-error");
    await listCameras();
  }
}

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
}
window.addEventListener("resize", resizeOverlay);

function drawBBox(bbox) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  if (!bbox) return;
  const x1 = bbox.x1 * overlay.width;
  const y1 = bbox.y1 * overlay.height;
  const x2 = bbox.x2 * overlay.width;
  const y2 = bbox.y2 * overlay.height;
  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
}

async function tick() {
  if (!busy) {
    busy = true;
    try {
      captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
      const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7);
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
      });
      const data = await res.json();
      applyPrediction(data);
    } catch (err) {
      setStatus("connection error", "status-error");
    }
    busy = false;
  }
  setTimeout(tick, PREDICT_INTERVAL_MS);
}

function applyPrediction(data) {
  drawBBox(data.bbox);

  if (!data.hand_detected) {
    letterBadge.textContent = "–";
    letterBadge.style.color = "#8a93a3";
  } else if (data.pred_conf < data.conf_gate) {
    letterBadge.textContent = `${data.pred_letter} ${(data.pred_conf * 100).toFixed(0)}%`;
    letterBadge.style.color = "#f59e0b";
  } else {
    letterBadge.textContent = `${data.pred_letter} ${(data.pred_conf * 100).toFixed(0)}%`;
    letterBadge.style.color = "#34d399";
  }

  progressFill.style.width = `${Math.round((data.progress || 0) * 100)}%`;
  sentenceEl.textContent = data.sentence || "";
}

async function sendControl(action) {
  const res = await fetch("/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  const data = await res.json();
  sentenceEl.textContent = data.sentence || "";
}

document.getElementById("btnSpace").addEventListener("click", () => sendControl("space"));
document.getElementById("btnBackspace").addEventListener("click", () => sendControl("backspace"));
document.getElementById("btnClear").addEventListener("click", () => sendControl("clear"));

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.code === "Space") { e.preventDefault(); sendControl("space"); }
  else if (e.code === "Backspace") { e.preventDefault(); sendControl("backspace"); }
});

cameraSelect.addEventListener("change", () => startCamera(cameraSelect.value));

if (navigator.mediaDevices.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", listCameras);
}

listCameras().then(() => startCamera());
