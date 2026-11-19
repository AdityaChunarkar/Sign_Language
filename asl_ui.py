#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASL Studio - a browser interface for asl_local.py
=================================================

Run it from the same folder as asl_local.py:

    python asl_ui.py

It opens http://127.0.0.1:8765 in your browser. Four pages:

    Live     camera + typing sentences by signing (A lock | SPACE | D delete | C clear)
    Record   guided recording of your own hand (the same recorder as --collect-personal)
    Train    fine-tune, final model, charts, export - with the live log
    Data     what you have recorded, sample pictures, saved charts

Nothing new to install. Everything is saved in the same asl_project/ folder as before.
Press Ctrl+C in the terminal to stop the app (this also turns the camera off).

Options:
    --port 8765          change the port
    --camera 0           webcam index (try 1 if 0 is not your camera)
    --project-dir DIR    use another asl_project folder
    --no-browser         do not open the browser automatically
"""

import argparse
import codecs
import hmac
import json
import os
import queue
import re
import secrets
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from collections import Counter, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
if not (HERE / "asl_local.py").exists():
    sys.exit("asl_ui.py must be in the SAME folder as asl_local.py. Move it there and run it again.")
sys.path.insert(0, str(HERE))

print("Loading (TensorFlow takes a few seconds) ...", flush=True)
import asl_local as asl  # noqa: E402

np, cv2 = asl.np, asl.cv2
TOKEN = secrets.token_urlsafe(18)

# What the Train page can run. Fixed command lines only - nothing from the browser is passed to a shell.
JOBS = {
    "finetune": ("Fine-tune on my hand", ["--finetune"]),
    "finetune_final": ("Make the final model", ["--finetune", "--final"]),
    "report": ("Make charts of my data", ["--personal-report"]),
    "export": ("Export the models", ["--export"]),
    "download": ("Download the Kaggle pictures", ["--download"]),
    "preprocess": ("Prepare the Kaggle pictures", ["--preprocess"]),
    "train": ("Train a new base model", ["--train"]),
    "evaluate": ("Check the base model", ["--evaluate"]),
}

TONES = {(0, 255, 0): "good", (60, 60, 255): "bad", (0, 165, 255): "warn", (0, 200, 255): "info",
         (0, 255, 255): "info"}


class Busy(Exception):
    pass


class BadRequest(Exception):
    pass


def build_cmd(args):
    return [sys.executable, "-u", str(Path(asl.__file__).resolve()), *args, "--project-dir", str(asl.P.base)]


# ======================================================================
#  Log buffer for jobs (understands \r progress bars)
# ======================================================================
class LogBuffer:
    def __init__(self, maxlines=1500):
        self.lines, self.cur, self.cr, self.version, self.maxlines = [], "", False, 0, maxlines
        self.lock = threading.Lock()

    def feed(self, text):
        with self.lock:
            for ch in text:
                if ch == "\n":
                    self.lines.append(self.cur)
                    self.cur, self.cr = "", False
                    if len(self.lines) > self.maxlines:
                        del self.lines[:200]
                elif ch == "\r":
                    self.cr = True
                elif ch == "\x08":
                    continue
                else:
                    if self.cr:
                        self.cur, self.cr = "", False
                    self.cur += ch
            self.version += 1

    def tail(self, n=400):
        with self.lock:
            return (self.lines + ([self.cur] if self.cur else []))[-n:], self.version


# ======================================================================
#  Hub: the one shared state object (one activity at a time)
# ======================================================================
class Hub:
    def __init__(self):
        self.lock = threading.RLock()
        self.mode = "idle"                       # idle | live | record | job
        self.stop_evt = threading.Event()
        self.thread = None
        self.error = ""
        self.frame, self.frame_id = None, 0
        self.frame_cv = threading.Condition()
        self.cmds = queue.Queue()                # live commands: lock / space / delete / clear
        self.keys = queue.Queue()                # recording keys: s / p / n / q
        self.models, self.models_sig = None, None
        self.model_info = {}
        self.settings = {"alpha": 0.6, "conf": 0.95, "hold": 1.0}
        self.camera = 0
        self.sentence, self.saved_sentence = "", ""
        self.flash_n = 0
        self.live = self.blank_live()
        self.record = {"phase": "idle"}
        self.job = None

    # ---------- state ----------
    def blank_live(self):
        return {"phase": "idle", "hand": False, "letter": "", "conf": 0.0, "top": [], "stable": "",
                "stable_conf": 0.0, "progress": 0.0, "fps": 0.0, "flash": "", "flash_id": self.flash_n}

    def set_live(self, **kw):
        with self.lock:
            self.live.update(kw)

    def snapshot(self):
        with self.lock:
            job = None
            if self.job:
                j = self.job
                job = {k: j[k] for k in ("key", "title", "running", "returncode", "started", "ended")}
                job["log_version"] = j["log"].version
            return {"mode": self.mode, "error": self.error, "camera": self.camera,
                    "live": dict(self.live, sentence=self.sentence), "settings": dict(self.settings),
                    "record": dict(self.record), "model": dict(self.model_info), "job": job}

    # ---------- frames ----------
    def publish_frame(self, jpg):
        with self.frame_cv:
            self.frame, self.frame_id = jpg, self.frame_id + 1
            self.frame_cv.notify_all()

    def wait_frame(self, last_id, timeout):
        with self.frame_cv:
            self.frame_cv.wait_for(lambda: self.frame_id != last_id, timeout)
            return self.frame, self.frame_id

    # ---------- running things ----------
    def launch(self, kind, target):
        with self.lock:
            if self.mode != "idle":
                what = {"live": "Live recognition", "record": "Recording", "job": "A training job"}[self.mode]
                raise Busy(f"{what} is running. Stop it first.")
            self.mode, self.error = kind, ""
            self.stop_evt.clear()
            with self.frame_cv:
                self.frame = None
            self.thread = threading.Thread(target=self._run, args=(kind, target), daemon=True, name=f"asl-{kind}")
            self.thread.start()

    def _run(self, kind, target):
        try:
            target()
        except SystemExit as e:                  # asl_local reports user errors with sys.exit("message")
            if isinstance(e.code, str) and e.code.strip():
                self.error = e.code.strip()
        except RuntimeError as e:                # asl_local raises these with a plain-language message (camera etc.)
            self.error = str(e)
        except Exception as e:                   # noqa: BLE001 - unexpected: shown in the UI and printed here
            traceback.print_exc()
            self.error = f"{type(e).__name__}: {e}"
        finally:
            with self.lock:
                if kind == "live":
                    self.live.update(phase="idle", hand=False, letter="", top=[], stable="", progress=0.0, fps=0.0)
                    self.save_sentence()
                elif kind == "record" and self.record.get("phase") not in ("done", "stopped"):
                    self.record["phase"] = "error" if self.error else "stopped"
                self.mode = "idle"
            self.publish_frame(None)

    def stop(self):
        with self.lock:
            self.stop_evt.set()
            if self.mode == "record":
                self.keys.put("q")
            if self.mode == "job" and self.job and self.job.get("proc") and self.job["running"]:
                try:
                    self.job["proc"].terminate()
                except OSError:
                    pass

    def shutdown(self):
        self.stop()
        t = self.thread
        if t and t.is_alive():
            t.join(8)
        self.save_sentence()

    # ---------- sentence history ----------
    def save_sentence(self):
        s = self.sentence.strip()
        if not s or s == self.saved_sentence:
            return
        path = asl.P.transcripts
        stamp = datetime.now().isoformat(timespec="seconds")
        lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        if self.saved_sentence and lines and lines[-1].endswith("\t" + self.saved_sentence) \
                and s.startswith(self.saved_sentence):
            lines[-1] = f"{stamp}\t{s}"          # same sentence carried on -> update the line, do not repeat it
        else:
            lines.append(f"{stamp}\t{s}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.saved_sentence = s

    # ---------- models ----------
    def get_models(self):
        pick = lambda pers, base: pers if asl.ckpt(pers).exists() else base          # noqa: E731
        paths = (asl.ckpt(pick(asl.IMG_PERS, asl.IMG_BEST)), asl.ckpt(pick(asl.LM_PERS, asl.LM_BEST)))
        sig = tuple(file_sig(p) for p in paths) + (str(asl.P.base),)
        if self.models is None or sig != self.models_sig:
            img, lm = asl.load_models_for_inference()          # also loads config.json into CFG
            with self.lock:
                self.models, self.models_sig = (img, lm), sig
                self.settings.update(alpha=float(asl.CFG["FUSION_ALPHA"]), conf=float(asl.CFG["CONF_THRESHOLD"]),
                                     hold=float(asl.CFG["STABLE_SECONDS"]))
                self.model_info = {"kind": model_kind(), "image": paths[0].name, "landmark": paths[1].name}
        return self.models


hub = Hub()


def file_sig(p):
    try:
        st = p.stat()
        return str(p), st.st_mtime_ns, st.st_size
    except OSError:
        return str(p), 0, 0


def model_kind():
    if asl.ckpt(asl.IMG_PERS).exists() and asl.ckpt(asl.LM_PERS).exists():
        return "personal"
    if asl.ckpt(asl.IMG_BEST).exists() and asl.ckpt(asl.LM_BEST).exists():
        return "base"
    return "none"


def encode_jpeg(img, quality=78):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


# ======================================================================
#  Live recognition engine (same acceptance rules as asl_local --realtime)
# ======================================================================
def run_live():
    hub.set_live(phase="loading")
    image_model, landmark_model = hub.get_models()
    hub.set_live(phase="camera")
    cam = asl.LocalWebcam(hub.camera)
    try:
        landmarker = asl.make_landmarker(video=True)
    except BaseException:
        cam.release()
        raise
    clock = asl.VideoClock()
    while not hub.cmds.empty():                              # forget keys pressed while it was off
        hub.cmds.get_nowait()

    history = deque()
    last_accepted, ready_for_repeat, release_start = None, True, None
    flash_text, flash_until = "", 0.0
    RELEASE_SECS = 0.4
    fps_ema, prev_t, misses = 0.0, time.time(), 0
    n_cls = asl.NUM_CLASSES

    def flash(text):
        nonlocal flash_text, flash_until
        hub.flash_n += 1
        flash_text, flash_until = text, time.time() + 1.0
        hub.set_live(flash_id=hub.flash_n)

    hub.set_live(phase="running")
    try:
        while not hub.stop_evt.is_set():
            frame = cam.read()
            if frame is None:
                misses += 1
                if misses > 60:
                    hub.error = "The camera stopped sending pictures. Check that no other app is using it."
                    break
                time.sleep(0.01)
                continue
            misses = 0
            now = time.time()
            alpha, conf_gate, stable_secs = (float(hub.settings[k]) for k in ("alpha", "conf", "hold"))
            crop, vec, bbox, hand_lms = asl.extract_from_image(frame, landmarker, clock())

            pred_letter, pred_conf, top = "", 0.0, []
            if crop is not None:
                p_img = image_model(crop[None].astype(np.float32), training=False).numpy()[0] if alpha > 0 else np.zeros(n_cls)
                p_lm = landmark_model(vec[None], training=False).numpy()[0] if alpha < 1 else np.zeros(n_cls)
                p = alpha * p_img + (1 - alpha) * p_lm
                pred_idx = int(np.argmax(p))
                pred_letter, pred_conf = asl.LETTERS[pred_idx], float(p[pred_idx])
                top = [[asl.LETTERS[i], round(float(p[i]), 3)] for i in np.argsort(p)[::-1][:5]]
                history.append((now, pred_idx, pred_conf))
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                asl.draw_hand_landmarks(frame, hand_lms)

            # ---- commands from the browser ----
            manual_lock = False
            while True:
                try:
                    cmd = hub.cmds.get_nowait()
                except queue.Empty:
                    break
                s = hub.sentence
                if cmd == "space":
                    if s and not s.endswith(" "):
                        hub.sentence = s + " "
                        flash("Space added")
                    else:
                        flash("Type a letter first")
                    history.clear()
                elif cmd == "delete":
                    if s:
                        flash("Deleted space" if s[-1] == " " else f"Deleted {s[-1]}")
                        hub.sentence = s[:-1]
                    else:
                        flash("Nothing to delete")
                    history.clear()
                elif cmd == "lock":
                    manual_lock = True
                elif cmd == "clear":
                    hub.save_sentence()
                    hub.sentence, hub.saved_sentence = "", ""
                    last_accepted, ready_for_repeat = None, True
                    flash("Cleared")
                    history.clear()

            if manual_lock:
                if pred_letter:
                    hub.sentence += pred_letter
                    last_accepted, ready_for_repeat, release_start = pred_letter, False, None
                    history.clear()
                    flash(f"Locked {pred_letter}")
                else:
                    flash("No hand to lock")

            while history and now - history[0][0] > stable_secs * 1.2:
                history.popleft()

            if last_accepted is not None and not ready_for_repeat:
                releasing = (crop is None) or (pred_letter != last_accepted)
                if releasing:
                    release_start = release_start or now
                    if now - release_start >= RELEASE_SECS:
                        ready_for_repeat = True
                else:
                    release_start = None

            stable_letter, stable_conf, progress = "", 0.0, 0.0
            if len(history) >= 3:
                span = history[-1][0] - history[0][0]
                top_idx, top_n = Counter(h[1] for h in history).most_common(1)[0]
                stable_frac = top_n / len(history)
                stable_conf = float(np.mean([h[2] for h in history if h[1] == top_idx]))
                stable_letter = asl.LETTERS[top_idx]
                if stable_conf >= conf_gate:
                    progress = min(1.0, span / stable_secs) * stable_frac
                if (span >= stable_secs and stable_frac >= 0.75 and stable_conf >= conf_gate
                        and (stable_letter != last_accepted or ready_for_repeat)):
                    hub.sentence += stable_letter
                    last_accepted, ready_for_repeat, release_start = stable_letter, False, None
                    history.clear()
                    progress = 0.0

            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt

            hub.set_live(hand=crop is not None, letter=pred_letter, conf=round(pred_conf, 4), top=top,
                         stable=stable_letter, stable_conf=round(stable_conf, 4), progress=round(progress, 3),
                         fps=round(fps_ema, 1), flash=flash_text if now < flash_until else "")
            jpg = encode_jpeg(frame)
            if jpg:
                hub.publish_frame(jpg)
    finally:
        landmarker.close()
        cam.release()


# ======================================================================
#  Recording: the existing collect_personal_dataset(), wired to the browser
# ======================================================================
def run_record(letters, per_letter, session):
    before = len(asl.load_meta())
    while not hub.keys.empty():
        hub.keys.get_nowait()
    with hub.lock:
        hub.record = {"phase": "starting", "letters": letters, "session": session, "per_letter": per_letter,
                      "letter": "", "have": 0, "target": per_letter, "capturing": False, "msg": "Opening the camera",
                      "tone": "info", "angle": 0, "angle_name": "", "angle_text": "", "angle_have": 0, "quota": 0,
                      "angles": len(asl.ANGLE_PHASES), "saved": 0}
    asl.collect_personal_dataset(letters, per_letter, session, hub.camera)
    with hub.lock:
        hub.record["phase"] = "stopped" if hub.stop_evt.is_set() else "done"
        hub.record["saved"] = max(0, len(asl.load_meta()) - before)


def install_bridge():
    """asl_local's recorder draws to an OpenCV window and reads OpenCV keys. These four small functions send the
    pictures to the browser instead, take keys from the browser, and report progress."""
    def ui_show(window, img):
        if hub.mode == "record":
            jpg = encode_jpeg(img)
            if jpg:
                hub.publish_frame(jpg)

    def read_keys(delay=1):
        time.sleep(max(delay, 1) / 1000.0)
        out = []
        while True:
            try:
                out.append(hub.keys.get_nowait())
            except queue.Empty:
                return out

    def ui_closed(window):
        return hub.stop_evt.is_set()

    def draw_hud(overlay, letter, have, target, phase, phase_have, quota, capturing, msg, msg_color):
        name, text = asl.ANGLE_PHASES[phase]
        with hub.lock:
            hub.record.update(phase="recording", letter=letter, have=int(have), target=int(target), angle=int(phase),
                              angle_name=name, angle_text=text, angle_have=int(phase_have), quota=int(quota),
                              capturing=bool(capturing), msg=msg, tone=TONES.get(tuple(msg_color), "info"))

    asl.ui_show, asl.read_keys, asl.ui_closed, asl.draw_hud = ui_show, read_keys, ui_closed, draw_hud
    try:                                           # the recorder calls this when it ends; there is no window to close
        cv2.destroyAllWindows = lambda: None
    except Exception:                              # noqa: BLE001
        pass


# ======================================================================
#  Jobs: --finetune, --export, ... as sub-processes with a live log
# ======================================================================
def run_job(key):
    title, args = JOBS[key]
    job = {"key": key, "title": title, "running": True, "returncode": None, "started": time.time(), "ended": None,
           "log": LogBuffer(), "proc": None}
    with hub.lock:
        hub.job = job
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TF_CPP_MIN_LOG_LEVEL="2")
    try:
        proc = subprocess.Popen(build_cmd(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(HERE),
                                env=env)
    except OSError as e:
        job["running"], job["returncode"], job["ended"] = False, -1, time.time()
        raise RuntimeError(f"Could not start the job: {e}")
    job["proc"] = proc
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while True:
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            job["log"].feed(dec.decode(chunk))
    finally:
        rc = proc.wait()
        with hub.lock:
            job["running"], job["returncode"], job["ended"] = False, rc, time.time()
        job["log"].feed(f"\n[finished with exit code {rc}]\n" if rc else "\n[finished]\n")


# ======================================================================
#  Read-only data for the pages
# ======================================================================
def next_session_name(sessions):
    i = 1
    while f"s{i:02d}" in sessions:
        i += 1
    return f"s{i:02d}"


def build_status():
    df = asl.load_meta()
    sessions = list(dict.fromkeys(df["session"])) if len(df) else []
    per = {l: {} for l in asl.LETTERS}
    if len(df):
        for (l, s), n in Counter(zip(df["letter"], df["session"])).items():
            if l in per:
                per[l][s] = int(n)
    cached = [l for l in asl.LETTERS if (asl.P.kaggle_cache / f"{l}.npz").exists()]
    have_kag = asl.P.dataset_root.is_dir() or asl.P.dataset_zip.exists()
    kind = model_kind()
    pers_ok = kind == "personal"
    strength = asl.split_personal(df)[1] if len(df) else None
    if kind == "none":
        nxt = {"page": "train", "text": "No models yet. Import the Colab models with --import-models, or train a base model on the Train page."}
    elif not len(df):
        nxt = {"page": "record", "text": "Record your own hand. 25 pictures per letter is a good start."}
    elif not pers_ok:
        nxt = {"page": "train", "text": "Fine-tune the model on your hand."}
    else:
        nxt = {"page": "live", "text": "Everything is ready. Try the Live page, or record another session in new light for a fairer test."}
    return {
        "project_dir": str(asl.P.base), "sessions": sessions, "next_session": next_session_name(sessions),
        "per_letter": per, "total": int(len(df)), "letters": asl.LETTERS, "split": strength,
        "pipeline": {"kaggle_downloaded": bool(have_kag), "kaggle_cached": len(cached), "model_kind": kind,
                     "personal_ok": bool(pers_ok), "next": nxt},
        "angles": [{"name": n, "text": t} for n, t in asl.ANGLE_PHASES],
    }


def list_reports():
    out = []
    if asl.P.reports.is_dir():
        for f in sorted(asl.P.reports.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.suffix.lower() in (".png", ".txt") and re.fullmatch(r"[A-Za-z0-9_.\-]+", f.name):
                out.append({"name": f.name, "kind": "image" if f.suffix.lower() == ".png" else "text",
                            "mtime": int(f.stat().st_mtime), "size": f.stat().st_size})
    return out


def read_transcripts(limit=8):
    p = asl.P.transcripts
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]:
        stamp, _, text = line.partition("\t")
        rows.append({"time": stamp, "text": text})
    return rows[::-1]


def sample_path(letter):
    d = asl.P.crops / letter
    if letter not in asl.LETTERS or not d.is_dir():
        return None
    files = sorted(f for f in d.iterdir() if f.suffix.lower() == ".jpg")
    return files[len(files) // 2] if files else None


def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


# ======================================================================
#  HTTP
# ======================================================================
INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="asl-token" content="@@TOKEN@@">
<title>ASL Studio</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect x='3' y='2' width='26' height='28' rx='7' fill='%230B7F63'/%3E%3Crect x='3' y='24' width='26' height='6' rx='3' fill='%23085F4A'/%3E%3Ctext x='16' y='21' font-family='Arial' font-weight='700' font-size='17' text-anchor='middle' fill='white'%3EA%3C/text%3E%3C/svg%3E">
<script>
(function () {
  var t = null;
  try { t = localStorage.getItem('asl-theme'); } catch (e) {}
  if (!t) t = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.dataset.theme = t;
})();
</script>
<style>
:root {
  color-scheme: light;
  --bg: #EBEEF3; --surface: #FFFFFF; --sunk: #DFE4EC; --ink: #1A2030; --ink-2: #556077;
  --line: #D0D7E2; --line-2: #B3BCCD;
  --accent: #0B7F63; --on-accent: #FFFFFF; --fill: #7ADFC0; --accent-soft: #CFEFE4;
  --warn: #9A5B00; --warn-soft: #FBE9CC; --bad: #B3392C; --bad-soft: #F9DDD8;
  --stage: #101520; --stage-ink: #F2F5FA;
  --key-top: #FFFFFF; --key-edge: #B3BCCD;
  --display: "Bahnschrift", "DIN Alternate", "Avenir Next Condensed", "Segoe UI Semibold", "Helvetica Neue", Arial, sans-serif;
  --body: "Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif;
  --r-stage: 22px; --r-tray: 18px; --r-ctl: 11px;
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0E121A; --surface: #171D29; --sunk: #0A0D13; --ink: #E8ECF4; --ink-2: #9AA6BC;
  --line: #283043; --line-2: #3A455C;
  --accent: #3FD7AE; --on-accent: #05261C; --fill: #1E8F72; --accent-soft: #12372D;
  --warn: #F0B35A; --warn-soft: #3A2B10; --bad: #FF8E80; --bad-soft: #3F1B17;
  --stage: #06080C; --key-top: #222B3B; --key-edge: #080A0F;
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 var(--body); }
h1, h2, p, ul, ol, figure { margin: 0; }
button, input, select { font: inherit; color: inherit; }
[hidden] { display: none !important; }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
.muted { color: var(--ink-2); }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }

/* ---------- header ---------- */
.bar { display: flex; align-items: center; gap: 28px; padding: 12px 28px; background: var(--surface); border-bottom: 1px solid var(--line); }
.brand { display: flex; align-items: center; gap: 12px; color: inherit; text-decoration: none; }
.brand-name { font: 700 21px/1 var(--display); letter-spacing: .01em; }
.tabs { display: flex; gap: 4px; }
.tabs a { padding: 8px 16px; border-radius: 999px; color: var(--ink-2); text-decoration: none; font-weight: 600; }
.tabs a:hover { background: var(--sunk); color: var(--ink); }
.tabs a[aria-current="page"] { background: var(--ink); color: var(--bg); }
.bar-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
.pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 13px; border-radius: 999px; background: var(--sunk); font-size: 13px; font-weight: 600; white-space: nowrap; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: var(--line-2); flex: none; }
.pill[data-state="live"] .dot, .pill[data-state="job"] .dot { background: var(--accent); box-shadow: 0 0 0 4px var(--accent-soft); }
.pill[data-state="record"] .dot { background: var(--bad); box-shadow: 0 0 0 4px var(--bad-soft); }
.icon-btn { width: 38px; height: 38px; display: grid; place-items: center; border-radius: 50%; border: 1px solid var(--line); background: transparent; cursor: pointer; }
.icon-btn:hover { background: var(--sunk); }

.banner { display: flex; align-items: center; gap: 16px; padding: 11px 28px; background: var(--bad-soft); color: var(--bad); border-bottom: 1px solid var(--line); font-weight: 600; }
.banner button { margin-left: auto; background: none; border: 0; color: inherit; text-decoration: underline; cursor: pointer; font-weight: 600; }

main { max-width: 1280px; margin: 0 auto; padding: 28px; }
.page h1 { font: 700 30px/1.1 var(--display); margin-bottom: 8px; }
.page h2 { font: 600 14px/1.3 var(--body); color: var(--ink-2); margin-bottom: 12px; }
.lead { color: var(--ink-2); max-width: 62ch; }

/* ---------- keycaps ---------- */
.key {
  display: inline-grid; place-items: center; width: var(--k, 46px); height: calc(var(--k, 46px) * 1.16);
  border-radius: 10px; background: var(--key-top); color: var(--ink);
  border: 1px solid var(--line); border-bottom: 4px solid var(--key-edge);
  font-family: var(--display); font-weight: 700; font-size: calc(var(--k, 46px) * .62); line-height: 1; user-select: none;
}
.key.new { animation: drop .3s cubic-bezier(.2, .9, .3, 1.25); }
@keyframes drop { from { transform: translateY(-16px) scale(1.1); opacity: .15; } to { transform: none; opacity: 1; } }
.brand .key { --k: 30px; background: var(--accent); color: var(--on-accent); border-color: var(--accent); border-bottom-color: color-mix(in srgb, var(--accent) 70%, #000); border-radius: 8px; }
kbd { display: inline-grid; place-items: center; min-width: 22px; height: 22px; padding: 0 6px; border-radius: 6px; background: var(--key-top); color: var(--ink); border: 1px solid var(--line-2); border-bottom-width: 2px; font: 600 12px/1 var(--body); }
.btn.dark kbd { background: transparent; color: var(--bg); border-color: rgba(255, 255, 255, .45); }

/* ---------- buttons and fields ---------- */
.btn { white-space: nowrap; display: inline-flex; align-items: center; justify-content: center; gap: 10px; height: 44px; padding: 0 18px; border-radius: var(--r-ctl); border: 1px solid var(--line-2); background: var(--surface); color: var(--ink); font-weight: 600; cursor: pointer; transition: background .12s, transform .06s; }
.btn:hover:not(:disabled) { background: var(--sunk); }
.btn:active:not(:disabled) { transform: translateY(1px); }
.btn:disabled { opacity: .45; cursor: not-allowed; }
.btn.primary { background: var(--accent); border-color: var(--accent); color: var(--on-accent); }
.btn.primary:hover:not(:disabled) { background: var(--accent); filter: brightness(1.08); }
.btn.dark { background: var(--ink); border-color: var(--ink); color: var(--bg); }
.btn.dark:hover:not(:disabled) { background: var(--ink); filter: brightness(1.2); }
.btn.big { height: 50px; padding: 0 26px; font-size: 16px; }
.btn.small { height: 34px; padding: 0 12px; font-size: 13px; }
.field { display: grid; gap: 6px; align-content: start; }
.field > span { display: flex; justify-content: space-between; gap: 12px; font-weight: 600; font-size: 14px; }
.field output { color: var(--ink-2); font-weight: 500; font-variant-numeric: tabular-nums; }
input[type="text"], input[type="number"] { height: 40px; padding: 0 12px; border-radius: 10px; border: 1px solid var(--line-2); background: var(--surface); width: 100%; }
input[type="range"] { width: 100%; accent-color: var(--accent); }
.scale { display: flex; justify-content: space-between; color: var(--ink-2); font-size: 12px; }
small { font-size: 13px; color: var(--ink-2); }

/* ---------- stage (video) ---------- */
.stage-wrap { display: grid; gap: 12px; }
.stage { position: relative; aspect-ratio: 4 / 3; width: min(100%, calc((100vh - 190px) * 4 / 3)); min-width: 320px; border-radius: var(--r-stage); background: var(--stage); color: var(--stage-ink); overflow: hidden; }
.video { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
.stage-ph, .stage-loading { position: absolute; inset: 0; display: grid; place-content: center; justify-items: center; gap: 8px; text-align: center; padding: 24px; }
.stage-ph svg { width: 160px; height: 160px; margin-bottom: 4px; }
.ph-title { font: 700 28px/1.1 var(--display); }
.ph-sub { color: #AEB8CC; max-width: 36ch; margin-bottom: 12px; }
.stage-loading { background: rgba(6, 8, 12, .6); grid-auto-flow: column; place-content: center; gap: 14px; font-weight: 600; font-size: 17px; }
.spin { width: 18px; height: 18px; border-radius: 50%; border: 3px solid rgba(255, 255, 255, .25); border-top-color: #fff; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.chip { position: absolute; display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px; background: rgba(8, 10, 15, .68); color: #fff; font-size: 13px; font-weight: 600; backdrop-filter: blur(6px); }
.chip.tl { left: 14px; top: 14px; } .chip.tr { right: 14px; top: 14px; }
.chip[data-tone="good"] .dot, .chip[data-on="true"] .dot { background: #46E0B4; }
.chip[data-tone="bad"] .dot, .chip[data-on="false"] .dot { background: #FF7A6B; }
.chip[data-tone="warn"] .dot { background: #FFB84D; }
.chip[data-tone="info"] .dot { background: #9FB0CC; }
.toast { position: absolute; left: 50%; bottom: 18px; padding: 9px 18px; border-radius: 12px; background: #F2F5FA; color: #101520; font-weight: 700; opacity: 0; pointer-events: none; transform: translate(-50%, 8px); }
.toast.show { animation: toast 1.4s ease both; }
@keyframes toast { 0% { opacity: 0; transform: translate(-50%, 8px); } 12% { opacity: 1; transform: translate(-50%, 0); } 80% { opacity: 1; } 100% { opacity: 0; } }
.caption { position: absolute; left: 0; right: 0; bottom: 0; padding: 44px 24px 20px; background: linear-gradient(to top, rgba(6, 8, 12, .86), rgba(6, 8, 12, 0)); font: 700 30px/1.15 var(--display); }
.stage-foot small { flex: 1; }
.stage-foot .btn { flex: none; }
.stage-foot { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 40px; width: min(100%, calc((100vh - 190px) * 4 / 3)); min-width: 320px; }

/* ---------- live page ---------- */
.live, .rec { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 36px; align-items: start; }
.side { display: grid; gap: 26px; }
.tray { background: var(--sunk); border-radius: var(--r-tray); padding: 20px 20px 16px; min-height: 200px; display: flex; flex-direction: column; gap: 14px; }
.tiles { --k: 46px; display: flex; flex-wrap: wrap; align-content: flex-start; gap: 12px 20px; flex: 1; max-height: 250px; overflow: auto; padding: 2px 2px 4px; }
.tiles.scrolled { -webkit-mask-image: linear-gradient(to bottom, transparent 0, #000 26px); mask-image: linear-gradient(to bottom, transparent 0, #000 26px); }
.word { display: inline-flex; gap: 6px; }
.slot { --p: 0; order: 1; width: 56px; height: 66px; border-radius: 12px; border: 2px dashed var(--line-2); display: grid; place-items: center; font: 700 36px/1 var(--display); color: var(--ink-2); background: linear-gradient(to top, var(--fill) calc(var(--p) * 100%), transparent calc(var(--p) * 100%)); transition: border-color .15s; }
.slot[data-state="ready"] { border-color: var(--accent); color: var(--ink); }
.slot[data-state="unsure"] { border-color: var(--warn); color: var(--warn); border-style: solid; }
.slot[data-state="off"] { opacity: .5; }
.tiles-empty { order: 2; align-self: center; color: var(--ink-2); max-width: 24ch; }
.hint { min-height: 1.5em; color: var(--ink-2); }
.hint[data-tone="warn"] { color: var(--warn); font-weight: 600; }
.actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.actions .btn span { flex: 1; text-align: left; }
.actions .btn kbd { flex: none; }
.actions .btn.plain span { text-align: center; }
.guess { list-style: none; padding: 0; display: grid; gap: 8px; }
.guess li { display: grid; grid-template-columns: 30px 1fr 44px; align-items: center; gap: 12px; }
.g-letter { font: 700 19px/1 var(--display); text-align: center; }
.g-bar { height: 8px; border-radius: 99px; background: var(--sunk); overflow: hidden; }
.g-bar i { display: block; width: 0; height: 100%; background: var(--line-2); transition: width .12s linear; }
.guess li[data-top="true"] .g-bar i { background: var(--accent); }
.g-pct { text-align: right; color: var(--ink-2); font-variant-numeric: tabular-nums; font-size: 13px; }
details.tuning { border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 12px 0; }
details.tuning summary { cursor: pointer; font-weight: 600; }
details.tuning .fields { display: grid; gap: 16px; padding-top: 16px; }
.recent { list-style: none; padding: 0; }
.recent li { display: flex; align-items: center; gap: 12px; padding: 10px 0; border-top: 1px solid var(--line); }
.recent li:first-child { border-top: 0; }
.recent .r-text { font: 600 18px/1.25 var(--display); flex: 1; word-break: break-word; }
.recent time { color: var(--ink-2); font-size: 13px; font-variant-numeric: tabular-nums; }

/* ---------- record page ---------- */
.panel { display: grid; gap: 22px; }
.letters { display: grid; grid-template-columns: repeat(auto-fill, minmax(54px, 1fr)); gap: 10px; }
.pick { display: grid; justify-items: center; gap: 2px; padding: 0; border: 0; background: none; cursor: pointer; color: inherit; }
.pick .key { --k: 40px; opacity: .45; transition: opacity .1s, transform .1s; }
.pick[aria-pressed="true"] .key { opacity: 1; border-bottom-color: var(--accent); }
.pick small { font-variant-numeric: tabular-nums; font-size: 12px; }
.pick[data-short="true"] small { color: var(--warn); font-weight: 600; }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.two { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.run-top { display: flex; align-items: center; gap: 18px; }
.run-top .key { --k: 64px; }
.bar-track { height: 10px; border-radius: 99px; background: var(--sunk); overflow: hidden; }
.bar-track i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .2s; }
.angles { display: grid; grid-template-columns: repeat(10, 1fr); gap: 5px; }
.angles i { height: 8px; border-radius: 99px; background: var(--sunk); }
.angles i.done { background: var(--accent); }
.angles i.now { background: var(--ink); }
.coach-msg { font-weight: 600; }
.coach-msg[data-tone="good"] { color: var(--accent); }
.coach-msg[data-tone="bad"] { color: var(--bad); }
.coach-msg[data-tone="warn"] { color: var(--warn); }
.strip { display: flex; flex-wrap: wrap; gap: 5px; }
.strip .key { --k: 24px; border-bottom-width: 3px; border-radius: 6px; opacity: .4; }
.strip .key.done { opacity: 1; border-bottom-color: var(--accent); }
.strip .key.now { opacity: 1; background: var(--ink); color: var(--bg); border-color: var(--ink); }

/* ---------- train page ---------- */
.train { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.1fr); gap: 36px; align-items: start; }
.joblist { list-style: none; padding: 0; margin-top: 18px; }
.joblist li { display: flex; gap: 18px; align-items: center; padding: 16px 0; border-top: 1px solid var(--line); }
.joblist .j-text { flex: 1; }
.joblist strong { display: block; font: 700 18px/1.2 var(--display); margin-bottom: 3px; }
.joblist p { color: var(--ink-2); }
details.adv { margin-top: 16px; }
details.adv summary { cursor: pointer; font-weight: 600; padding: 8px 0; }
.console { position: sticky; top: 20px; display: grid; gap: 14px; }
.console-head { display: flex; align-items: center; gap: 12px; }
.console-head strong { font: 700 18px/1 var(--display); flex: 1; }
.log { margin: 0; height: 340px; overflow: auto; padding: 16px; border-radius: 14px; background: var(--stage); color: #D5DCEA; font: 12.5px/1.55 ui-monospace, "Cascadia Mono", Consolas, Menlo, monospace; white-space: pre-wrap; word-break: break-word; }
.result { display: grid; gap: 12px; }
.result pre { margin: 0; padding: 16px; border-radius: 14px; background: var(--sunk); white-space: pre-wrap; font: 13px/1.5 ui-monospace, Consolas, Menlo, monospace; }
.result img { width: 100%; border-radius: 12px; border: 1px solid var(--line); background: #fff; }

/* ---------- data page ---------- */
.data-top { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr); gap: 40px; align-items: start; margin-bottom: 40px; }
.steps { list-style: none; padding: 0; margin-top: 20px; }
.steps li { display: flex; gap: 14px; padding: 13px 0; border-top: 1px solid var(--line); }
.steps .tick { flex: none; width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; background: var(--sunk); color: var(--ink-2); font-weight: 700; font-size: 13px; }
.steps li[data-ok="true"] .tick { background: var(--accent); color: var(--on-accent); }
.steps b { display: block; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; font-size: 13px; color: var(--ink-2); }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; }
#chart svg { width: 100%; height: auto; display: block; }
.samples { display: grid; grid-template-columns: repeat(auto-fill, minmax(96px, 1fr)); gap: 14px; }
.samples figure { display: grid; gap: 6px; }
.samples .ph { aspect-ratio: 1; border-radius: 14px; overflow: hidden; background: var(--sunk); display: grid; place-items: center; font: 700 30px var(--display); color: var(--line-2); }
.samples img { width: 100%; height: 100%; object-fit: cover; display: block; }
.samples figcaption { display: flex; justify-content: space-between; font-size: 13px; color: var(--ink-2); }
.samples figcaption b { font: 700 15px var(--display); color: var(--ink); }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
.gallery a { display: grid; gap: 6px; color: inherit; text-decoration: none; }
.gallery img { width: 100%; border-radius: 12px; border: 1px solid var(--line); background: #fff; }
section.block-gap { margin-top: 44px; }

@media (max-width: 1000px) {
  .live, .rec, .train, .data-top { grid-template-columns: 1fr; }
  .bar { flex-wrap: wrap; gap: 12px 20px; padding: 12px 16px; }
  .tabs { order: 3; width: 100%; overflow-x: auto; }
  main { padding: 18px 16px; }
  .stage, .stage-foot { width: 100%; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
}
</style>
</head>
<body>
<noscript>ASL Studio needs JavaScript. Turn it on in your browser and reload.</noscript>

<header class="bar">
  <a class="brand" href="#live" aria-label="ASL Studio, Live page"><span class="key" aria-hidden="true">A</span><span class="brand-name">ASL Studio</span></a>
  <nav class="tabs" aria-label="Pages">
    <a href="#live" data-page="live">Live</a>
    <a href="#record" data-page="record">Record</a>
    <a href="#train" data-page="train">Train</a>
    <a href="#data" data-page="data">Data</a>
  </nav>
  <div class="bar-right">
    <span class="pill" id="pill-mode" data-state="idle"><i class="dot"></i><span>Idle</span></span>
    <span class="pill" id="pill-model">Checking models</span>
    <button class="icon-btn" id="theme" type="button" aria-label="Switch between light and dark">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8.5"/><path d="M12 3.5v17a8.5 8.5 0 0 0 0-17z" fill="currentColor"/></svg>
    </button>
  </div>
</header>

<div class="banner" id="banner" role="alert" hidden><span id="banner-text"></span><button type="button" id="banner-close">Dismiss</button></div>

<main>
<!-- ======================= LIVE ======================= -->
<section class="page live" id="page-live">
  <h1 class="sr">Live recognition</h1>
  <div class="stage-wrap">
    <div class="stage" id="stage-live">
      <img class="video" id="video-live" alt="Live camera view with the detected hand" hidden>
      <div class="stage-ph" id="ph-live">
        <svg class="hand-art" viewBox="0 0 200 200" aria-hidden="true"></svg>
        <p class="ph-title">Camera is off</p>
        <p class="ph-sub" id="ph-sub-live">Sign a letter and hold it steady. It is typed for you.</p>
        <button class="btn primary big" id="start-live" type="button">Start camera</button>
      </div>
      <div class="stage-loading" id="loading-live" hidden><span class="spin"></span><span id="loading-text">Loading</span></div>
      <span class="chip tl" id="hand-chip" hidden><i class="dot"></i><span>No hand in view</span></span>
      <span class="chip tr" id="fps-chip" hidden></span>
      <div class="toast" id="toast-live"></div>
    </div>
    <div class="stage-foot">
      <small id="model-line"></small>
      <button class="btn small" id="stop-live" type="button" hidden>Stop camera</button>
    </div>
  </div>

  <div class="side">
    <div>
      <div class="tray">
        <div class="tiles" id="tiles">
          <div class="slot" id="slot" data-state="off" aria-hidden="true"></div>
          <span class="tiles-empty" id="tiles-empty">Your sentence builds here, one letter at a time.</span>
        </div>
        <p class="hint" id="hint">Start the camera to begin.</p>
      </div>
    </div>

    <div class="actions">
      <button class="btn dark" data-act="lock" type="button"><span>Lock letter</span><kbd>A</kbd></button>
      <button class="btn" data-act="space" type="button"><span>Space</span><kbd>Space</kbd></button>
      <button class="btn" data-act="delete" type="button"><span>Delete</span><kbd>D</kbd></button>
      <button class="btn" data-act="clear" type="button"><span>Clear</span><kbd>C</kbd></button>
      <button class="btn plain" id="speak" type="button"><span>Read aloud</span></button>
      <button class="btn plain" id="copy" type="button"><span>Copy</span></button>
    </div>

    <section>
      <h2>What the model sees</h2>
      <p class="muted" id="guess-empty">The five most likely letters show here while the camera is on.</p>
      <ol class="guess" id="guess" hidden>
        <li><span class="g-letter"></span><span class="g-bar"><i></i></span><span class="g-pct"></span></li>
        <li><span class="g-letter"></span><span class="g-bar"><i></i></span><span class="g-pct"></span></li>
        <li><span class="g-letter"></span><span class="g-bar"><i></i></span><span class="g-pct"></span></li>
        <li><span class="g-letter"></span><span class="g-bar"><i></i></span><span class="g-pct"></span></li>
        <li><span class="g-letter"></span><span class="g-bar"><i></i></span><span class="g-pct"></span></li>
      </ol>
    </section>

    <details class="tuning">
      <summary>Tuning</summary>
      <div class="fields">
        <label class="field"><span>Confidence needed <output id="o-conf"></output></span><input type="range" id="r-conf" min="0.5" max="0.99" step="0.01"></label>
        <label class="field"><span>Hold time <output id="o-hold"></output></span><input type="range" id="r-hold" min="0.3" max="2.5" step="0.1"></label>
        <label class="field"><span>Model mix <output id="o-alpha"></output></span><input type="range" id="r-alpha" min="0" max="1" step="0.05"><span class="scale"><span>Hand skeleton</span><span>Photo</span></span></label>
        <label class="field"><span>Camera number</span><input type="number" class="cam-input" min="0" max="9" value="0"></label>
      </div>
    </details>

    <section>
      <h2>Recent sentences</h2>
      <ul class="recent" id="recent"></ul>
      <p class="muted" id="recent-empty">Sentences you type are saved here.</p>
    </section>
  </div>
</section>

<!-- ======================= RECORD ======================= -->
<section class="page rec" id="page-record" hidden>
  <div class="stage-wrap">
    <div class="stage" id="stage-rec">
      <img class="video" id="video-record" alt="Camera view while recording" hidden>
      <div class="stage-ph" id="ph-rec">
        <svg class="hand-art" viewBox="0 0 200 200" aria-hidden="true"></svg>
        <p class="ph-title">Record your own hand</p>
        <p class="ph-sub">Pick the letters on the right and press Start recording. The camera guides you through 10 angles per letter.</p>
      </div>
      <div class="stage-loading" id="loading-rec" hidden><span class="spin"></span><span>Opening the camera</span></div>
      <span class="chip tl" id="rec-chip" hidden data-tone="info"><i class="dot"></i><span></span></span>
      <div class="caption" id="rec-caption" hidden></div>
    </div>
  </div>

  <div class="side">
    <!-- setup -->
    <div class="panel" id="rec-setup">
      <div>
        <h1>Record</h1>
        <p class="lead">Pictures of your own hand make the model accurate for you. Use a new session name whenever the light or background changes. The next fine-tune then tests on that session, so the accuracy it reports is fair.</p>
      </div>
      <div class="two">
        <label class="field"><span>Session name</span><input type="text" id="in-session" maxlength="24" autocomplete="off" spellcheck="false"><small id="session-note"></small></label>
        <label class="field"><span>Pictures per letter</span><input type="number" id="in-per" min="5" max="500" value="25"><small>25 is a good start. 50 or more helps the weak letters.</small></label>
      </div>
      <div>
        <h2>Letters to record</h2>
        <div class="letters" id="letters"></div>
        <div class="row" style="margin-top:12px">
          <button class="btn small" id="sel-all" type="button">All letters</button>
          <button class="btn small" id="sel-need" type="button">Only letters that need more</button>
          <button class="btn small" id="sel-none" type="button">None</button>
        </div>
      </div>
      <div class="row" style="align-items:flex-end;gap:16px"><button class="btn primary big" id="rec-start" type="button">Start recording</button><label class="field" style="width:130px"><span>Camera number</span><input type="number" class="cam-input" min="0" max="9" value="0"></label></div>
    </div>

    <!-- running -->
    <div class="panel" id="rec-run" hidden>
      <div class="run-top">
        <span class="key" id="run-key">A</span>
        <div style="flex:1">
          <p id="run-count" style="font:700 22px/1.2 var(--display)"></p>
          <div class="bar-track" style="margin-top:8px"><i id="run-bar"></i></div>
        </div>
      </div>
      <div>
        <div class="angles" id="run-angles"></div>
        <p class="muted" id="run-angle" style="margin-top:8px"></p>
      </div>
      <p class="coach-msg" id="run-msg"></p>
      <div class="actions">
        <button class="btn primary" id="rec-toggle" type="button"><span>Start capturing</span><kbd>S</kbd></button>
        <button class="btn" id="rec-next" type="button"><span>Next angle</span><kbd>P</kbd></button>
        <button class="btn" id="rec-skip" type="button"><span>Skip letter</span><kbd>N</kbd></button>
        <button class="btn" id="rec-stop" type="button"><span>Stop</span><kbd>Q</kbd></button>
      </div>
      <div><h2>Letters in this run</h2><div class="strip" id="run-strip"></div></div>
    </div>

    <!-- finished -->
    <div class="panel" id="rec-done" hidden>
      <div><h1 id="done-title">Recording finished</h1><p class="lead" id="done-text"></p></div>
      <div class="row"><button class="btn primary" id="rec-train" type="button">Go to Train</button><button class="btn" id="rec-more" type="button">Record more</button></div>
    </div>
  </div>
</section>

<!-- ======================= TRAIN ======================= -->
<section class="page train" id="page-train" hidden>
  <div>
    <h1>Train</h1>
    <p class="lead">Run these in order after you record. Each one runs on this computer, and its progress shows on the right. Close nothing while one is running.</p>
    <ul class="joblist" id="joblist"></ul>
    <details class="adv"><summary>Advanced</summary><ul class="joblist" id="joblist-adv"></ul></details>
  </div>
  <div class="console">
    <div class="console-head"><strong id="job-title">Nothing has run yet</strong><span class="pill" id="job-pill" hidden></span><button class="btn small" id="job-stop" type="button" hidden>Stop</button></div>
    <pre class="log" id="log" tabindex="0">The log of the job you run appears here.</pre>
    <div class="result" id="result" hidden><h2>Result</h2><pre id="result-text"></pre><img id="result-img" alt="Chart saved by the job" hidden></div>
  </div>
</section>

<!-- ======================= DATA ======================= -->
<section class="page data" id="page-data" hidden>
  <div class="data-top">
    <div>
      <h1>Data</h1>
      <p class="lead" id="next-text"></p>
      <div class="row" style="margin-top:14px"><button class="btn primary" id="next-go" type="button" hidden></button></div>
      <ul class="steps" id="pipeline"></ul>
    </div>
    <div>
      <h2>Pictures per letter</h2>
      <div id="chart"></div>
      <div class="legend" id="legend"></div>
    </div>
  </div>
  <section>
    <h2>One of your pictures for each letter</h2>
    <div class="samples" id="samples"></div>
  </section>
  <section class="block-gap">
    <h2>Saved charts and reports</h2>
    <div class="gallery" id="gallery"></div>
    <p class="muted" id="gallery-empty">Charts appear here after you run &ldquo;Make charts of my data&rdquo; or a fine-tune.</p>
  </section>
</section>
</main>

<script>
(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const token = $('meta[name="asl-token"]').content;
const PAGES = ['live', 'record', 'train', 'data'];
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

let S = null;          // newest snapshot from the server
let status = null;     // /api/status
let page = 'live';
let lastError = '';

/* ---------------- helpers ---------------- */
async function api(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-ASL-Token': token }, body: JSON.stringify(body || {}) });
  let j = {};
  try { j = await r.json(); } catch (e) { /* no body */ }
  if (!r.ok) throw new Error(j.error || ('Request failed (' + r.status + ')'));
  return j;
}
async function getJSON(path) {
  const r = await fetch(path, { cache: 'no-store' });
  if (!r.ok) throw new Error('Request failed (' + r.status + ')');
  return r.json();
}
const act = (path, body) => api(path, body).catch(e => showBanner(e.message));
function showBanner(msg) { $('#banner-text').textContent = msg; $('#banner').hidden = false; }
$('#banner-close').onclick = () => { $('#banner').hidden = true; };
function toast(sel, text) {
  const el = $(sel); el.textContent = text;
  el.classList.remove('show'); void el.offsetWidth; el.classList.add('show');
}
function blurAfter(fn) { return e => { fn(e); if (e.currentTarget && e.currentTarget.blur) e.currentTarget.blur(); }; }

/* ---------------- theme ---------------- */
$('#theme').onclick = () => {
  const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('asl-theme', t); } catch (e) { /* private mode */ }
};

/* ---------------- hand drawing for the empty stage ---------------- */
(function drawHand() {
  const pts = [[100,180],[78,165],[62,145],[52,125],[44,108],[80,110],[76,82],[74,62],[73,44],[100,104],[100,72],[100,50],[100,30],[120,108],[124,80],[126,60],[127,43],[138,120],[146,98],[150,82],[153,68]];
  const links = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];
  const lines = links.map(([a, b]) => `<line x1="${pts[a][0]}" y1="${pts[a][1]}" x2="${pts[b][0]}" y2="${pts[b][1]}" stroke="#F2F5FA" stroke-opacity=".55" stroke-width="2" stroke-linecap="round"/>`).join('');
  const dots = pts.map((p, i) => `<circle cx="${p[0]}" cy="${p[1]}" r="${[4,8,12,16,20].includes(i) ? 5 : 3.5}" fill="${[4,8,12,16,20].includes(i) ? '#46E0B4' : '#F2F5FA'}"/>`).join('');
  $$('.hand-art').forEach(s => { s.innerHTML = lines + dots; });
})();

/* ---------------- router ---------------- */
function route() {
  const h = (location.hash || '#live').slice(1);
  page = PAGES.includes(h) ? h : 'live';
  PAGES.forEach(p => { $('#page-' + p).hidden = p !== page; });
  $$('.tabs a').forEach(a => a.setAttribute('aria-current', a.dataset.page === page ? 'page' : 'false'));
  if (page === 'record' || page === 'data') loadStatus();
  if (page === 'data' || page === 'train') loadReports();
  if (S) { syncVideo(); renderAll(); }
}
addEventListener('hashchange', route);

/* ---------------- video ---------------- */
let videoFor = null;
function syncVideo() {
  const want = S && ((page === 'live' && S.mode === 'live') ? 'live' : (page === 'record' && S.mode === 'record') ? 'record' : null);
  if (want === videoFor) return;
  if (videoFor) { const old = $('#video-' + videoFor); old.removeAttribute('src'); old.hidden = true; }
  videoFor = want;
  if (want) { const v = $('#video-' + want); v.src = '/stream.mjpg?t=' + Date.now(); v.hidden = false; }
}

/* ---------------- server events ---------------- */
function connect() {
  const es = new EventSource('/events');
  es.onopen = () => { if ($('#banner-text').dataset.lost) { $('#banner').hidden = true; delete $('#banner-text').dataset.lost; } };
  es.onmessage = ev => {
    const prev = S;
    S = JSON.parse(ev.data);
    if (S.error && S.error !== lastError) showBanner(S.error);
    lastError = S.error;
    if (prev && prev.mode !== 'idle' && S.mode === 'idle') { loadStatus(); loadReports(); if (prev.mode === 'job') finishJob(); }
    if (!prev || prev.mode !== S.mode) syncVideo();
    renderAll();
  };
  es.onerror = () => {
    showBanner('Lost contact with ASL Studio. Check that the terminal window is still running, then reload this page.');
    $('#banner-text').dataset.lost = '1';
  };
}

async function loadStatus() { try { status = await getJSON('/api/status'); renderHeader(); if (page === 'record') drawLetterGrid(); if (page === 'data') renderData(); if (page === 'live') renderModelLine(); } catch (e) { /* retried on next visit */ } }

/* ---------------- header ---------------- */
function renderHeader() {
  if (!S) return;
  const m = S.mode, p = $('#pill-mode');
  const label = m === 'live' ? (S.live.phase === 'running' ? 'Live' : 'Starting') : m === 'record' ? 'Recording' : m === 'job' ? 'Training' : 'Idle';
  p.dataset.state = m; p.lastElementChild.textContent = label;
  const kind = (S.model && S.model.kind) || (status && status.pipeline.model_kind) || '';
  $('#pill-model').textContent = kind === 'personal' ? 'Your fine-tuned model' : kind === 'base' ? 'Base model' : kind === 'none' ? 'No model yet' : 'Checking models';
}
function renderModelLine() {
  const k = status && status.pipeline.model_kind;
  $('#model-line').textContent = k === 'personal' ? 'Using the model fine-tuned on your hand.' : k === 'base' ? 'Using the base model. Record and fine-tune for better accuracy on your hand.' : k === 'none' ? 'No model found. See the Data page.' : '';
}

/* ---------------- LIVE ---------------- */
const tiles = $('#tiles'), slot = $('#slot'), tilesEmpty = $('#tiles-empty');
let shown = null, lastFlash = null;
tiles.addEventListener('scroll', () => tiles.classList.toggle('scrolled', tiles.scrollTop > 2));

function renderTiles(s) {
  if (s === shown) return;
  const grew = shown !== null && s.length === shown.length + 1 && s.startsWith(shown) && s.slice(-1) !== ' ';
  shown = s;
  $$('.word', tiles).forEach(w => w.remove());
  if (s.length) {
    s.split(' ').forEach(word => {
      const el = document.createElement('span'); el.className = 'word';
      for (const ch of word) { const k = document.createElement('span'); k.className = 'key'; k.textContent = ch; el.append(k); }
      tiles.insertBefore(el, slot);
    });
    if (grew) { const ks = $$('.key', tiles); ks[ks.length - 1].classList.add('new'); }
  }
  tilesEmpty.hidden = s.length > 0;
  tiles.scrollTop = tiles.scrollHeight;
  tiles.classList.toggle('scrolled', tiles.scrollTop > 2);
}

function renderLive() {
  const L = S.live, on = S.mode === 'live', run = on && L.phase === 'running', other = S.mode !== 'idle' && !on;
  $('#ph-live').hidden = on;
  $('#ph-sub-live').textContent = other ? (S.mode === 'record' ? 'Recording is using the camera. Stop it first.' : 'A training job is running. Wait for it to finish, or stop it.') : 'Sign a letter and hold it steady. It is typed for you.';
  $('#start-live').disabled = S.mode !== 'idle';
  $('#loading-live').hidden = !(on && !run);
  if (on && !run) $('#loading-text').textContent = L.phase === 'loading' ? 'Loading the models' : 'Opening the camera';
  $('#stop-live').hidden = !on;
  const chip = $('#hand-chip'); chip.hidden = !run; chip.dataset.on = String(L.hand); chip.lastElementChild.textContent = L.hand ? 'Hand found' : 'No hand in view';
  $('#fps-chip').hidden = !run; $('#fps-chip').textContent = Math.round(L.fps) + ' fps';

  renderTiles(L.sentence);

  const gate = S.settings.conf;
  let state = 'off', letter = '', p = 0, hint = 'Start the camera to begin.', tone = '';
  if (on && !run) { hint = 'Getting ready.'; }
  else if (run && !L.hand) { state = 'off'; hint = 'Show your hand to the camera.'; }
  else if (run && L.conf >= gate) { state = 'ready'; letter = L.letter; p = L.progress; hint = p > 0.05 ? 'Keep holding.' : 'Hold the sign steady, or press A to lock it now.'; }
  else if (run) { state = 'unsure'; letter = L.letter; hint = `Not sure. Best guess is ${L.letter} at ${Math.round(L.conf * 100)}%. Press A to lock it in.`; tone = 'warn'; }
  slot.dataset.state = state; slot.textContent = letter; slot.style.setProperty('--p', p);
  const h = $('#hint'); h.textContent = hint; h.dataset.tone = tone;

  $('#guess').hidden = !run; $('#guess-empty').hidden = run;
  const top = L.top || [];
  $$('#guess li').forEach((li, i) => {
    const t = run ? top[i] : null;
    $('.g-letter', li).textContent = t ? t[0] : '';
    $('.g-bar i', li).style.width = t ? Math.round(t[1] * 100) + '%' : '0%';
    $('.g-pct', li).textContent = t ? Math.round(t[1] * 100) + '%' : '';
    li.dataset.top = String(i === 0);
  });

  if (L.flash_id !== lastFlash) { if (lastFlash !== null && L.flash) toast('#toast-live', L.flash); lastFlash = L.flash_id; }
  $$('[data-act]').forEach(b => { b.disabled = !run; });
  $('#speak').disabled = !L.sentence.trim();
  $('#copy').disabled = !L.sentence.trim();
  syncSliders();
}

$$('[data-act]').forEach(b => b.addEventListener('click', blurAfter(() => act('/api/live/key', { key: b.dataset.act }))));
$('#start-live').onclick = () => act('/api/live/start', { camera: camValue() });
$('#stop-live').onclick = () => act('/api/stop');
$('#speak').onclick = blurAfter(() => { if (S && S.live.sentence.trim() && 'speechSynthesis' in window) { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(S.live.sentence.trim().toLowerCase())); } });
$('#copy').onclick = blurAfter(async () => {
  if (!S) return;
  try { await navigator.clipboard.writeText(S.live.sentence.trim()); toast('#toast-live', 'Copied'); } catch (e) { showBanner('Could not copy. Select the text and copy it by hand.'); }
});

/* tuning sliders */
const SL = { conf: $('#r-conf'), hold: $('#r-hold'), alpha: $('#r-alpha') };
const OUT = { conf: $('#o-conf'), hold: $('#o-hold'), alpha: $('#o-alpha') };
const fmt = {
  conf: v => Math.round(v * 100) + '%',
  hold: v => (+v).toFixed(1) + ' s',
  alpha: v => v <= 0 ? 'Hand skeleton only' : v >= 1 ? 'Photo only' : Math.round((1 - v) * 100) + '% skeleton, ' + Math.round(v * 100) + '% photo',
};
let touched = 0, sendTimer = null;
Object.keys(SL).forEach(k => SL[k].addEventListener('input', () => {
  touched = Date.now(); OUT[k].textContent = fmt[k](SL[k].value);
  clearTimeout(sendTimer);
  sendTimer = setTimeout(() => act('/api/live/settings', { [k]: +SL[k].value }), 120);
}));
function syncSliders() {
  if (Date.now() - touched < 900) return;
  Object.keys(SL).forEach(k => { SL[k].value = S.settings[k]; OUT[k].textContent = fmt[k](S.settings[k]); });
}

/* camera number (the Live and Record pages share it) */
const camInputs = $$('.cam-input');
function camValue() { return +camInputs[0].value || 0; }
camInputs.forEach(i => i.addEventListener('input', () => camInputs.forEach(o => { if (o !== i) o.value = i.value; })));

/* recent sentences */
async function loadRecent() {
  try {
    const rows = await getJSON('/api/transcripts');
    $('#recent-empty').hidden = rows.length > 0;
    $('#recent').innerHTML = rows.map(r => `<li><span class="r-text">${esc(r.text)}</span><time>${esc(r.time.slice(11, 16))}</time></li>`).join('');
  } catch (e) { /* ignore */ }
}

/* ---------------- RECORD ---------------- */
let selected = new Set(LETTERS), recDismissed = false, sessionTouched = false;

function sessionName() { return $('#in-session').value.trim(); }
function perLetter() { return Math.max(1, +$('#in-per').value || 25); }
function countFor(l) { return (status && status.per_letter[l] && status.per_letter[l][sessionName()]) || 0; }
function drawLetterGrid() {
  if (!status) return;
  if (!sessionTouched) $('#in-session').value = status.next_session;
  const sn = sessionName(), t = perLetter();
  $('#session-note').textContent = status.sessions.includes(sn) ? 'This session exists. Recording tops up each letter.' : 'A new session.';
  $('#letters').innerHTML = LETTERS.map(l => {
    const c = countFor(l);
    return `<button type="button" class="pick" data-l="${l}" aria-pressed="${selected.has(l)}" data-short="${c < t}" aria-label="${l}, ${c} pictures"><span class="key">${l}</span><small>${c}</small></button>`;
  }).join('');
}
$('#letters').addEventListener('click', e => {
  const b = e.target.closest('.pick'); if (!b) return;
  const l = b.dataset.l; selected.has(l) ? selected.delete(l) : selected.add(l);
  b.setAttribute('aria-pressed', String(selected.has(l)));
});
$('#sel-all').onclick = () => { selected = new Set(LETTERS); drawLetterGrid(); };
$('#sel-none').onclick = () => { selected = new Set(); drawLetterGrid(); };
$('#sel-need').onclick = () => { selected = new Set(LETTERS.filter(l => countFor(l) < perLetter())); drawLetterGrid(); };
$('#in-session').addEventListener('input', () => { sessionTouched = true; drawLetterGrid(); });
$('#in-per').addEventListener('input', drawLetterGrid);
$('#rec-start').onclick = async () => {
  try {
    await api('/api/record/start', { letters: [...selected].sort().join(''), session: sessionName(), per_letter: perLetter(), camera: camValue() });
    recDismissed = false; sessionTouched = true;
  } catch (e) { showBanner(e.message); }
};
const recKey = k => () => act('/api/record/key', { key: k });
$('#rec-toggle').onclick = blurAfter(recKey('s'));
$('#rec-next').onclick = blurAfter(recKey('p'));
$('#rec-skip').onclick = blurAfter(recKey('n'));
$('#rec-stop').onclick = () => act('/api/stop');
$('#rec-more').onclick = () => { recDismissed = true; renderRecord(); };
$('#rec-train').onclick = () => { location.hash = '#train'; };

function renderRecord() {
  const R = S.record, on = S.mode === 'record';
  const finished = !on && (R.phase === 'done' || R.phase === 'stopped') && !recDismissed;
  $('#rec-setup').hidden = on || finished;
  $('#rec-run').hidden = !on;
  $('#rec-done').hidden = !finished;
  $('#ph-rec').hidden = on;
  $('#loading-rec').hidden = !(on && R.phase === 'starting');
  $('#rec-start').disabled = S.mode !== 'idle';
  const chip = $('#rec-chip'), cap = $('#rec-caption');
  chip.hidden = !(on && R.phase === 'recording'); cap.hidden = chip.hidden;
  if (!chip.hidden) { chip.dataset.tone = R.tone; chip.lastElementChild.textContent = R.msg; cap.textContent = R.angle_text; }
  if (on && R.phase === 'recording') {
    $('#run-key').textContent = R.letter;
    $('#run-count').textContent = `${R.have} of ${R.target} pictures`;
    $('#run-bar').style.width = Math.min(100, 100 * R.have / Math.max(1, R.target)) + '%';
    const ang = $('#run-angles');
    if (ang.children.length !== R.angles) ang.innerHTML = '<i></i>'.repeat(R.angles);
    [...ang.children].forEach((el, i) => { el.className = i < R.angle ? 'done' : i === R.angle ? 'now' : ''; });
    $('#run-angle').textContent = `Angle ${R.angle + 1} of ${R.angles}: ${R.angle_text}. ${R.angle_have} of ${R.quota} here.`;
    const m = $('#run-msg'); m.textContent = R.msg; m.dataset.tone = R.tone;
    $('#rec-toggle').firstElementChild.textContent = R.capturing ? 'Pause' : 'Start capturing';
    const idx = R.letters.indexOf(R.letter);
    $('#run-strip').innerHTML = R.letters.map((l, i) => `<span class="key ${i < idx ? 'done' : i === idx ? 'now' : ''}">${l}</span>`).join('');
  }
  if (finished) {
    $('#done-title').textContent = R.phase === 'done' ? 'Recording finished' : 'Recording stopped';
    $('#done-text').textContent = `Saved ${R.saved} new picture${R.saved === 1 ? '' : 's'} in session ${R.session}. Next, fine-tune the model on the Train page.`;
  }
}

/* ---------------- TRAIN ---------------- */
const JOB_UI = {
  finetune: ['Fine-tune on my hand', 'Teaches the model your hand from the pictures you recorded, then tests it on pictures it has not seen.', 'About 5 to 20 minutes'],
  finetune_final: ['Make the final model', 'Trains again on every picture you recorded. Do this once the test result looks good.', 'About 5 to 20 minutes'],
  report: ['Make charts of my data', 'Saves pictures of how much data you have for each letter and angle. Good for slides.', 'A few seconds'],
  export: ['Export the models', 'Packs the models into asl_model_export.zip so you can copy them to another computer.', 'A few seconds'],
  download: ['Download the Kaggle pictures', 'About 1 GB. Needs a Kaggle account token on this computer.', '10 to 30 minutes'],
  preprocess: ['Prepare the Kaggle pictures', 'Finds the hand in every Kaggle picture. Needed to train a new base model, and it makes fine-tuning more reliable.', '20 to 60 minutes'],
  train: ['Train a new base model', 'Trains from scratch on the Kaggle pictures. Slow on a computer without a graphics card.', 'Hours'],
  evaluate: ['Check the base model', 'Scores the base model on Kaggle pictures and shows which letters get mixed up.', 'A few minutes'],
};
const RESULT_FILES = { finetune: ['evaluation_personal.txt', 'confusion_personal.png'], finetune_final: ['evaluation_personal_final.txt', 'confusion_personal_final.png'], evaluate: ['evaluation_kaggle.txt', 'confusion_kaggle.png'], report: [null, 'personal_images_per_letter.png'] };
function jobRows(keys) {
  return keys.map(k => `<li><div class="j-text"><strong>${esc(JOB_UI[k][0])}</strong><p>${esc(JOB_UI[k][1])}</p><small>${esc(JOB_UI[k][2])}</small></div><button type="button" class="btn" data-job="${k}">Run</button></li>`).join('');
}
$('#joblist').innerHTML = jobRows(['finetune', 'finetune_final', 'report', 'export']);
$('#joblist-adv').innerHTML = jobRows(['download', 'preprocess', 'train', 'evaluate']);
$$('[data-job]').forEach(b => b.addEventListener('click', () => { act('/api/job/run', { job: b.dataset.job }); $('#result').hidden = true; }));
$('#job-stop').onclick = () => act('/api/stop');

let logVersion = -1, logBusy = false, logFor = null;
async function pullLog() {
  if (logBusy) return; logBusy = true;
  try {
    const j = await getJSON('/api/job/log'), el = $('#log');
    const near = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    el.textContent = j.lines.join('\n') || 'Starting.'; logVersion = j.version;
    if (near) el.scrollTop = el.scrollHeight;
  } catch (e) { /* next event retries */ } finally { logBusy = false; }
}
function renderTrain() {
  const J = S.job, busy = S.mode !== 'idle';
  $$('[data-job]').forEach(b => { b.disabled = busy; });
  const pill = $('#job-pill');
  if (!J) { $('#job-title').textContent = 'Nothing has run yet'; pill.hidden = true; $('#job-stop').hidden = true; return; }
  $('#job-title').textContent = J.title;
  pill.hidden = false;
  pill.dataset.state = J.running ? 'job' : 'idle';
  pill.innerHTML = `<i class="dot"></i><span>${J.running ? 'Running' : J.returncode === 0 ? 'Finished' : 'Stopped or failed'}</span>`;
  $('#job-stop').hidden = !(J.running);
  if (J.log_version !== logVersion && (page === 'train')) pullLog();
  if (logFor !== J.started) { logFor = J.started; logVersion = -1; }
}
async function finishJob() {
  await pullLog();
  const J = S && S.job; if (!J || J.returncode !== 0) return;
  const files = RESULT_FILES[J.key]; if (!files) return;
  const box = $('#result'), txt = $('#result-text'), img = $('#result-img');
  box.hidden = false; txt.hidden = true; img.hidden = true;
  const stamp = Date.now();
  if (files[0]) { try { const r = await fetch('/reports/' + files[0] + '?t=' + stamp); if (r.ok) { txt.textContent = await r.text(); txt.hidden = false; } } catch (e) { /* none */ } }
  if (files[1]) { img.onload = () => { img.hidden = false; }; img.onerror = () => { img.hidden = true; }; img.src = '/reports/' + files[1] + '?t=' + stamp; }
  if (txt.hidden && !files[1]) box.hidden = true;
}

/* ---------------- DATA ---------------- */
const PALETTE = ['#0B7F63', '#3B6FD8', '#C77D1A', '#8A4FBF', '#C0392B', '#2F8FA8'];
function renderData() {
  if (!status) return;
  const p = status.pipeline, n = status.pipeline.next;
  $('#next-text').textContent = n.text;
  const go = $('#next-go'); go.hidden = false; go.textContent = { live: 'Open Live', record: 'Open Record', train: 'Open Train' }[n.page] || 'Open'; go.onclick = () => { location.hash = '#' + n.page; };
  const splitText = { STRICT: 'The test uses a whole separate session, so the accuracy it reports is fair.', MEDIUM: 'The test uses angles it has not seen from one session. Record a second session in new light for a fairer test.', WEAK: 'The test is optimistic. Record a fresh session to get a fair number.' }[status.split] || '';
  const steps = [
    [p.model_kind !== 'none', 'Models', p.model_kind === 'personal' ? 'Base model and your fine-tuned model are ready.' : p.model_kind === 'base' ? 'Base model is ready. Fine-tune it on your hand.' : 'No model yet. Import the Colab models with --import-models, or train one on the Train page.'],
    [status.total > 0, 'Your pictures', status.total ? `${status.total} pictures in ${status.sessions.length} session${status.sessions.length === 1 ? '' : 's'}: ${status.sessions.join(', ')}. ${splitText}` : 'None yet. Record some on the Record page.'],
    [p.personal_ok, 'Fine-tuned on your hand', p.personal_ok ? 'Done. Live recognition uses it.' : 'Not yet.'],
    [p.kaggle_cached === 26, 'Kaggle pictures (optional)', p.kaggle_cached === 26 ? 'Prepared. Fine-tuning mixes some in so the model does not forget.' : `${p.kaggle_cached} of 26 letters prepared. Needed only to train a new base model, and they make fine-tuning more reliable.`],
  ];
  $('#pipeline').innerHTML = steps.map(s => `<li data-ok="${s[0]}"><span class="tick">${s[0] ? '&#10003;' : ''}</span><div><b>${esc(s[1])}</b><span class="muted">${esc(s[2])}</span></div></li>`).join('');
  drawChart();
  $('#samples').innerHTML = LETTERS.map(l => {
    const c = Object.values(status.per_letter[l] || {}).reduce((a, b) => a + b, 0);
    return `<figure><div class="ph">${c ? `<img loading="lazy" alt="Your sign for ${l}" src="/api/sample/${l}?v=${status.total}" onerror="this.remove()">` : l}</div><figcaption><b>${l}</b><span>${c}</span></figcaption></figure>`;
  }).join('');
}
function drawChart() {
  const per = status.per_letter, sess = status.sessions;
  const W = 760, H = 250, L = 34, R = 8, T = 10, B = 28, cw = (W - L - R) / 26;
  const totals = LETTERS.map(l => Object.values(per[l] || {}).reduce((a, b) => a + b, 0));
  const max = Math.max(10, ...totals), top = Math.ceil(max / 10) * 10;
  const y = v => T + (H - T - B) * (1 - v / top);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Pictures per letter">`;
  for (let i = 0; i <= 4; i++) { const v = top * i / 4; svg += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" stroke="var(--line)"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--ink-2)">${Math.round(v)}</text>`; }
  LETTERS.forEach((l, i) => {
    let acc = 0;
    sess.forEach((s, j) => { const v = (per[l] || {})[s] || 0; if (!v) return; svg += `<rect x="${L + i * cw + cw * .18}" width="${cw * .64}" y="${y(acc + v)}" height="${y(acc) - y(acc + v)}" rx="2" fill="${PALETTE[j % PALETTE.length]}"/>`; acc += v; });
    svg += `<text x="${L + i * cw + cw / 2}" y="${H - 8}" text-anchor="middle" font-size="12" font-weight="700" fill="var(--ink)">${l}</text>`;
  });
  $('#chart').innerHTML = svg + '</svg>';
  $('#legend').innerHTML = sess.map((s, j) => `<span><i style="background:${PALETTE[j % PALETTE.length]}"></i>${esc(s)}</span>`).join('') || '<span>No pictures yet.</span>';
}
async function loadReports() {
  try {
    const rows = await getJSON('/api/reports');
    const imgs = rows.filter(r => r.kind === 'image');
    $('#gallery').innerHTML = imgs.map(r => `<a href="/reports/${esc(r.name)}" target="_blank" rel="noopener"><img loading="lazy" alt="${esc(r.name)}" src="/reports/${esc(r.name)}?v=${r.mtime}"><small>${esc(r.name)}</small></a>`).join('');
    $('#gallery-empty').hidden = imgs.length > 0;
  } catch (e) { /* ignore */ }
}

/* ---------------- keyboard ---------------- */
function typingInField(t) { return !!t && (t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || (t.tagName === 'INPUT' && !['range', 'checkbox', 'radio', 'button'].includes(t.type))); }
addEventListener('keydown', e => {
  if (e.ctrlKey || e.metaKey || e.altKey || !S || typingInField(e.target)) return;
  const k = e.key.toLowerCase();
  if (page === 'live' && S.mode === 'live' && S.live.phase === 'running') {
    const map = { a: 'lock', s: 'lock', ' ': 'space', d: 'delete', backspace: 'delete', c: 'clear' };
    if (map[k]) { e.preventDefault(); if (!e.repeat) act('/api/live/key', { key: map[k] }); }
  } else if (page === 'record' && S.mode === 'record') {
    if ('spnq'.includes(k) && k.length === 1) { e.preventDefault(); if (!e.repeat) act(k === 'q' ? '/api/stop' : '/api/record/key', k === 'q' ? {} : { key: k }); }
  }
});
addEventListener('keyup', e => { if (e.key === ' ' && page === 'live' && S && S.mode === 'live' && !typingInField(e.target)) e.preventDefault(); });

/* ---------------- render dispatch ---------------- */
let lastRecentFetch = 0;
function renderAll() {
  if (!S) return;
  renderHeader();
  if (page === 'live') { renderLive(); if (Date.now() - lastRecentFetch > 4000) { lastRecentFetch = Date.now(); loadRecent(); } }
  else if (page === 'record') renderRecord();
  else if (page === 'train') renderTrain();
}

route();
loadStatus();
connect();
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "ASLStudio"
    sys_version = ""

    def log_message(self, *args):
        pass

    # ---------- helpers ----------
    def host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1")

    def send_bytes(self, code, body, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, obj, code=200):
        self.send_bytes(code, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 100_000:
            raise BadRequest("Request too large.")
        if not n:
            return {}
        try:
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except ValueError:
            raise BadRequest("Could not read the request.")
        return data if isinstance(data, dict) else {}

    # ---------- GET ----------
    def do_GET(self):
        if not self.host_ok():
            return self.send_json({"error": "Not allowed."}, 403)
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/":
                return self.send_bytes(200, INDEX_HTML.replace("@@TOKEN@@", TOKEN).encode("utf-8"),
                                       "text/html; charset=utf-8")
            if path == "/stream.mjpg":
                return self.stream()
            if path == "/events":
                return self.events()
            if path == "/api/status":
                return self.send_json(build_status())
            if path == "/api/transcripts":
                return self.send_json(read_transcripts())
            if path == "/api/reports":
                return self.send_json(list_reports())
            if path == "/api/job/log":
                job = hub.job
                lines, version = job["log"].tail() if job else ([], 0)
                return self.send_json({"lines": lines, "version": version})
            m = re.fullmatch(r"/api/sample/([A-Z])", path)
            if m:
                f = sample_path(m.group(1))
                if not f:
                    return self.send_json({"error": "No picture yet."}, 404)
                return self.send_bytes(200, f.read_bytes(), "image/jpeg")
            m = re.fullmatch(r"/reports/([A-Za-z0-9_.\-]+)", path)
            if m:
                f = asl.P.reports / m.group(1)
                if f.is_file() and f.suffix.lower() in (".png", ".txt"):
                    ctype = "image/png" if f.suffix.lower() == ".png" else "text/plain; charset=utf-8"
                    return self.send_bytes(200, f.read_bytes(), ctype)
            self.send_json({"error": "Not found."}, 404)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:                    # noqa: BLE001
            traceback.print_exc()
            try:
                self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
            except OSError:
                pass

    # ---------- streams ----------
    def stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last = -1
        try:
            while True:
                jpg, fid = hub.wait_frame(last, 1.0)
                if fid == last:
                    if hub.mode not in ("live", "record"):
                        return
                    continue
                last = fid
                if jpg is None:
                    if hub.mode not in ("live", "record"):
                        return
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last, beat = "", time.time()
        try:
            while True:
                payload = json.dumps(hub.snapshot(), separators=(",", ":"))
                if payload != last:
                    self.wfile.write(b"data: " + payload.encode("utf-8") + b"\n\n")
                    self.wfile.flush()
                    last, beat = payload, time.time()
                elif time.time() - beat > 10:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    beat = time.time()
                time.sleep(0.07)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    # ---------- POST ----------
    def do_POST(self):
        if not self.host_ok():
            return self.send_json({"error": "Not allowed."}, 403)
        origin = self.headers.get("Origin")
        if origin and origin.split("://", 1)[-1] != self.headers.get("Host"):
            return self.send_json({"error": "Not allowed."}, 403)
        if not hmac.compare_digest(self.headers.get("X-ASL-Token", ""), TOKEN):
            return self.send_json({"error": "Reload the page and try again."}, 403)
        path = unquote(urlparse(self.path).path)
        try:
            body = self.body_json()
            self.send_json(self.route_post(path, body))
        except Busy as e:
            self.send_json({"error": str(e)}, 409)
        except BadRequest as e:
            self.send_json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:                    # noqa: BLE001
            traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def route_post(self, path, body):
        if path == "/api/live/start":
            hub.camera = int(clamp(body.get("camera", hub.camera), 0, 9))
            hub.launch("live", run_live)
            return {"ok": True}
        if path == "/api/live/key":
            cmd = body.get("key")
            if cmd not in ("lock", "space", "delete", "clear"):
                raise BadRequest("Unknown key.")
            if hub.mode != "live":
                raise BadRequest("Start the camera first.")
            hub.cmds.put(cmd)
            return {"ok": True}
        if path == "/api/live/settings":
            with hub.lock:
                if "alpha" in body:
                    hub.settings["alpha"] = clamp(body["alpha"], 0.0, 1.0)
                if "conf" in body:
                    hub.settings["conf"] = clamp(body["conf"], 0.30, 0.999)
                if "hold" in body:
                    hub.settings["hold"] = clamp(body["hold"], 0.2, 3.0)
            return {"ok": True}
        if path == "/api/record/start":
            letters = "".join(dict.fromkeys(c for c in str(body.get("letters", "")).upper() if c in asl.LETTERS))
            session = str(body.get("session", "")).strip()
            if not letters:
                raise BadRequest("Pick at least one letter.")
            if not re.fullmatch(r"[A-Za-z0-9-]{1,24}", session) or session.lower() == "legacy":
                raise BadRequest("The session name can use letters, digits and - only (and not 'legacy').")
            per_letter = int(clamp(body.get("per_letter", 25), 1, 500))
            hub.camera = int(clamp(body.get("camera", hub.camera), 0, 9))
            hub.launch("record", lambda: run_record(list(letters), per_letter, session))
            return {"ok": True}
        if path == "/api/record/key":
            k = body.get("key")
            if k not in ("s", "p", "n", "q"):
                raise BadRequest("Unknown key.")
            if hub.mode != "record":
                raise BadRequest("Recording is not running.")
            hub.keys.put(k)
            return {"ok": True}
        if path == "/api/job/run":
            key = body.get("job")
            if key not in JOBS:
                raise BadRequest("Unknown job.")
            hub.launch("job", lambda: run_job(key))
            return {"ok": True}
        if path == "/api/stop":
            hub.stop()
            return {"ok": True}
        raise BadRequest("Not found.")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"          # on Windows this would let two copies share one port


def make_server(port):
    last = None
    for p in range(port, port + 10):
        try:
            return Server(("127.0.0.1", p), Handler)
        except OSError as e:
            last = e
    sys.exit(f"Could not open a port near {port}: {last}")


def main():
    ap = argparse.ArgumentParser(description="ASL Studio - browser interface for asl_local.py")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is not your camera)")
    ap.add_argument("--project-dir", default=None, help="use another asl_project folder")
    ap.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = ap.parse_args()

    if args.project_dir:
        asl.P = asl.make_paths(args.project_dir)
    asl.load_saved_config()
    hub.camera = args.camera
    hub.settings.update(alpha=float(asl.CFG["FUSION_ALPHA"]), conf=float(asl.CFG["CONF_THRESHOLD"]),
                        hold=float(asl.CFG["STABLE_SECONDS"]))
    install_bridge()

    httpd = make_server(args.port)
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    print(f"\nASL Studio is running at  {url}")
    print(f"Project folder: {asl.P.base}")
    print("Leave this window open while you use it. Press Ctrl+C here to stop.\n")
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        httpd.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        hub.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()