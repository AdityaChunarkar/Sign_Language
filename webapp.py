#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Browser front end for the exported ASL model.

Reuses model loading / hand extraction / fusion logic straight from
asl_model.py (no duplicated inference code). The browser captures webcam
frames (works with a real webcam or a phone-as-webcam app like DroidCam,
since both show up as a normal camera device to the browser) and posts
them to /predict; this server runs MediaPipe + the two Keras models and
returns the fused prediction, mirroring the stabilization logic from
run_realtime_recognition() in asl_model.py.

Run with:
    python webapp.py
then open http://localhost:5000
"""

import base64
import io
import time
from collections import Counter, deque

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from pypdf import PdfReader

from asl_model import (
    CFG,
    LETTERS,
    extract_from_image,
    load_models_for_inference,
    mp_hands,
)
from braille_controller import BrailleController

app = Flask(__name__)
braille_ctl = BrailleController()

print("Loading models...")
IMAGE_MODEL, LANDMARK_MODEL = load_models_for_inference()
HANDS = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1,
    min_detection_confidence=0.6, min_tracking_confidence=0.6,
    model_complexity=1,
)

RELEASE_SECS = 0.4

STATE = {
    "history": deque(),
    "sentence": "",
    "last_accepted": None,
    "ready_for_repeat": True,
    "release_start": None,
}


def _decode_frame(data_url):
    header, encoded = data_url.split(",", 1)
    raw = base64.b64decode(encoded)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.flip(frame, 1)  # mirror, matches LocalWebcam.read()


@app.route("/")
def index():
    return render_template(
        "index.html",
        conf_threshold=CFG["CONF_THRESHOLD"],
        stable_seconds=CFG["STABLE_SECONDS"],
    )


@app.route("/braille")
def braille_page():
    return render_template("braille.html")


@app.route("/braille/ports")
def braille_ports():
    return jsonify({"ports": braille_ctl.list_ports()})


@app.route("/braille/connect", methods=["POST"])
def braille_connect():
    port = request.get_json(force=True).get("port")
    try:
        braille_ctl.connect(port)
        return jsonify(braille_ctl.status())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/braille/disconnect", methods=["POST"])
def braille_disconnect():
    braille_ctl.disconnect()
    return jsonify(braille_ctl.status())


@app.route("/braille/text", methods=["POST"])
def braille_text():
    text = request.get_json(force=True).get("text", "")
    braille_ctl.set_text(text)
    return jsonify(braille_ctl.status())


@app.route("/braille/upload", methods=["POST"])
def braille_upload():
    f = request.files.get("pdf")
    if f is None:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400
    try:
        reader = PdfReader(io.BytesIO(f.read()))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return jsonify({"ok": False, "error": f"Could not read PDF: {e}"}), 400
    braille_ctl.set_text(text)
    return jsonify(braille_ctl.status())


@app.route("/braille/control", methods=["POST"])
def braille_control():
    body = request.get_json(force=True)
    action = body.get("action")
    if action == "play":
        braille_ctl.play()
    elif action == "pause":
        braille_ctl.pause()
    elif action == "next":
        braille_ctl.next_char()
    elif action == "prev":
        braille_ctl.prev_char()
    elif action == "reset":
        braille_ctl.reset()
    elif action == "speed":
        braille_ctl.set_speed(body.get("delay_ms", 800))
    return jsonify(braille_ctl.status())


@app.route("/braille/status")
def braille_status():
    return jsonify(braille_ctl.status())


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    frame = _decode_frame(payload["image"])
    now = time.time()
    alpha = float(CFG["FUSION_ALPHA"])
    conf_gate = float(CFG["CONF_THRESHOLD"])
    stable_secs = float(CFG["STABLE_SECONDS"])

    crop, vec, bbox, _hand_lms = extract_from_image(frame, HANDS)

    pred_letter, pred_conf = None, 0.0
    if crop is not None:
        p_img = IMAGE_MODEL(crop[None].astype(np.float32), training=False).numpy()[0]
        p_lm = LANDMARK_MODEL(vec[None], training=False).numpy()[0]
        p = alpha * p_img + (1 - alpha) * p_lm
        pred_idx = int(np.argmax(p))
        pred_letter, pred_conf = LETTERS[pred_idx], float(p[pred_idx])
        STATE["history"].append((now, pred_idx, pred_conf))

    history = STATE["history"]
    while history and now - history[0][0] > stable_secs * 1.2:
        history.popleft()

    if STATE["last_accepted"] is not None and not STATE["ready_for_repeat"]:
        releasing = (crop is None) or (pred_letter != STATE["last_accepted"])
        if releasing:
            STATE["release_start"] = STATE["release_start"] or now
            if now - STATE["release_start"] >= RELEASE_SECS:
                STATE["ready_for_repeat"] = True
        else:
            STATE["release_start"] = None

    stable_letter, stable_frac, stable_conf, progress = None, 0.0, 0.0, 0.0
    if len(history) >= 3:
        span = history[-1][0] - history[0][0]
        counts = Counter(h[1] for h in history)
        top_idx, top_n = counts.most_common(1)[0]
        stable_frac = top_n / len(history)
        confs = [h[2] for h in history if h[1] == top_idx]
        stable_conf = float(np.mean(confs))
        stable_letter = LETTERS[top_idx]
        progress = min(1.0, span / stable_secs) * stable_frac

        if (span >= stable_secs and stable_frac >= 0.75 and stable_conf >= conf_gate
                and (stable_letter != STATE["last_accepted"] or STATE["ready_for_repeat"])):
            STATE["sentence"] += stable_letter
            STATE["last_accepted"] = stable_letter
            STATE["ready_for_repeat"] = False
            STATE["release_start"] = None
            history.clear()

    bbox_out = None
    if bbox is not None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bbox_out = {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h}

    return jsonify({
        "hand_detected": crop is not None,
        "pred_letter": pred_letter,
        "pred_conf": pred_conf,
        "conf_gate": conf_gate,
        "stable_letter": stable_letter,
        "progress": progress,
        "sentence": STATE["sentence"],
        "bbox": bbox_out,
    })


@app.route("/control", methods=["POST"])
def control():
    action = request.get_json(force=True).get("action")
    if action == "space":
        STATE["sentence"] += " "
        STATE["last_accepted"] = None
        STATE["ready_for_repeat"] = True
    elif action == "backspace":
        STATE["sentence"] = STATE["sentence"][:-1]
    elif action == "clear":
        STATE["sentence"] = ""
        STATE["last_accepted"] = None
        STATE["ready_for_repeat"] = True
        STATE["history"].clear()
    return jsonify({"sentence": STATE["sentence"]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
