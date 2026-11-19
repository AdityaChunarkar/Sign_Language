#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time ASL Alphabet Recognition - Hybrid Image + Landmark Model
VS Code / local version  (everything is saved to disk, every stage can resume)
==============================================================================

Same model as the Colab notebook (EfficientNetB0 image branch + MediaPipe
landmark branch, fused), but built to live on your own machine:

    * Uses the NEW MediaPipe Tasks API (mp.tasks.vision.HandLandmarker), like the
      latest Colab notebook. The old mp.solutions API is not used anywhere.
    * No Colab code: the browser<->Python webcam bridge is replaced by a normal
      OpenCV window, /content paths by one project folder, !pip by requirements.txt.
    * Everything is written into  asl_project/  next to this script - nothing
      disappears when you close VS Code.

SETUP (once)
------------
    python -m venv .venv
    .venv\\Scripts\\activate            (Windows)      source .venv/bin/activate   (Mac/Linux)
    pip install -r requirements.txt

COMMANDS  (run them one after another, or combine several in one line)
---------------------------------------------------------------------
    python asl_local.py --status                  what exists + what to run next
    python asl_local.py --download                Kaggle ASL alphabet (or put the zip in asl_project/)
    python asl_local.py --preprocess              MediaPipe crops + landmarks (resumes if interrupted)
    python asl_local.py --train                   train image + landmark models
    python asl_local.py --evaluate                honest metrics, confusion matrix, confusion-driven retrain

    python asl_local.py --import-models  asl_model_export.zip     reuse models trained in Colab
    python asl_local.py --import-npz     personal_asl.npz         reuse your OLD Colab personal data

    python asl_local.py --collect-personal --session s01          record YOUR hand (guided angles)
    python asl_local.py --collect-personal --session s02          record again in other light/background
    python asl_local.py --personal-report                         counts, angle coverage, poster (PNG)
    python asl_local.py --finetune                                fine-tune on your hand, HONEST accuracy
    python asl_local.py --finetune --final                        then retrain on ALL your images for real use
    python asl_local.py --export                                  portable zip of the final models
    python asl_local.py --realtime                                live webcam recognition

PROJECT FOLDER
--------------
    asl_project/
      models/            *.keras checkpoints, config.json (fusion alpha, thresholds), label_encoder.json
      cache/kaggle/      preprocessed Kaggle crops + landmarks (one file per letter)
      personal/          YOUR dataset:  crops/<L>/*.jpg  frames/<L>/*.jpg  metadata.csv
      reports/           PNG charts + text reports
      asl_model_export/  + asl_model_export.zip   (from --export)
      transcripts.txt    every sentence you typed in --realtime

KEYS
----
    recording : s start/pause | p next angle | n skip letter | q quit      (click the video window first)
    realtime  : A lock the letter now | SPACE new word | D delete last | C clear | Q quit
"""

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

# ----------------------------------------------------------------------
# Dependency check - fail fast with a clear message instead of a traceback
# ----------------------------------------------------------------------
try:
    import numpy as np
    import pandas as pd
    import cv2
    import matplotlib
    matplotlib.use("Agg")                      # always save PNGs; never block on a plot window
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm.auto import tqdm

    import tensorflow as tf
    import keras
    from tensorflow.keras import layers, models, callbacks, optimizers

    import mediapipe as mp
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 precision_recall_fscore_support, accuracy_score)
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n\nInstall everything first with:\n"
             "    pip install -r requirements.txt\n")

if not hasattr(mp, "tasks") or not hasattr(mp.tasks.vision, "HandLandmarker"):
    sys.exit("Your mediapipe is too old for the Tasks API. Run:  pip install -U mediapipe")

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------------------------
# Global configuration (every important knob lives here)
# ------------------------------------------------------------------
CFG = {
    "IMG_SIZE": 224,                 # input size for EfficientNetB0
    "MAX_PER_CLASS": 400,            # Kaggle images kept per letter (raise with --max-per-class)
    "BATCH_SIZE": 32,
    "VAL_SPLIT": 0.15,
    "CONF_THRESHOLD": 0.95,          # real-time acceptance confidence
    "STABLE_SECONDS": 1.0,           # letter must stay stable this long
    "HAND_MARGIN": 0.30,             # crop margin around the detected hand
    "HARD_LETTERS": list("ASTEMNRUVWPKGHXY"),
    "HARD_BOOST": 1.5,
    "HARD_OVERSAMPLE": 0.5,
    "FUSION_ALPHA": 0.6,             # image-model weight in late fusion (auto-tuned)
    "FOCAL_GAMMA": 2.0,
    # ---- personal recording ----
    "PERSONAL_PER_LETTER": 200,      # images per letter per session
    "MIN_POSE_CHANGE": 0.012,        # a frame is saved only if the hand moved this much (fraction of frame)
    "RECENT_K": 8,                   # ...compared with each of the last K saved frames
    "MIN_SAVE_GAP_S": 0.10,
    "BLUR_MIN": 25.0,                # skip blurry crops (0 = off)
    "MIN_HAND_SCORE": 0.70,
    "GRACE_SECONDS": 2.0,            # pause after each angle change
    "SAVE_FULL_FRAME": True,
}
LETTERS = [chr(ord("A") + i) for i in range(26)]
NUM_CLASSES = len(LETTERS)
LETTER_TO_IDX = {l: i for i, l in enumerate(LETTERS)}

# 10 angles you are guided through for every letter (the video is mirrored like a mirror).
ANGLE_PHASES = [
    ("front",      "Straight at the camera"),
    ("tilt_left",  "Tilt your hand LEFT"),
    ("tilt_right", "Tilt your hand RIGHT"),
    ("turn_left",  "Turn wrist: palm faces LEFT"),
    ("turn_right", "Turn wrist: palm faces RIGHT"),
    ("pitch",      "Tip fingers UP, then DOWN"),
    ("close",      "Move hand CLOSER to camera"),
    ("far",        "Move hand FARTHER away"),
    ("high_low",   "Hand HIGH, then LOW in frame"),
    ("left_right", "Hand at LEFT / RIGHT edge"),
]
HOLDOUT_PHASE_IDS = [4, 9]           # angles held out when only ONE recording session exists

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")

# Standard 21-point MediaPipe hand skeleton (hard-coded so it never depends on a mediapipe version)
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                    (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
                    (17, 18), (18, 19), (19, 20)]

LM_COLS = [f"lm{i}_{a}" for i in range(21) for a in "xyz"]
META_COLS = (["file", "frame_file", "letter", "session", "phase_id", "phase", "handedness", "hand_score",
              "x1", "y1", "x2", "y2", "frame_w", "frame_h", "blur", "timestamp"] + LM_COLS)


# ------------------------------------------------------------------
# Project paths (replaces Colab's /content/...)
# ------------------------------------------------------------------
def make_paths(base):
    base = Path(base).expanduser().resolve()
    return SimpleNamespace(
        base=base, models=base / "models", kaggle_cache=base / "cache" / "kaggle",
        personal=base / "personal", crops=base / "personal" / "crops", frames=base / "personal" / "frames",
        meta=base / "personal" / "metadata.csv", reports=base / "reports",
        hand_model=base / "hand_landmarker.task", dataset_zip=base / "asl-alphabet.zip",
        dataset_root=base / "asl_alphabet_train" / "asl_alphabet_train",
        export=base / "asl_model_export", transcripts=base / "transcripts.txt")


P = make_paths(os.environ.get("ASL_PROJECT_DIR", Path(__file__).resolve().parent / "asl_project"))

IMG_BEST, LM_BEST = "image_model_best", "landmark_model_best"
IMG_PERS, LM_PERS = "image_model_personal", "landmark_model_personal"


def ckpt(name):
    return P.models / f"{name}.keras"


def ensure_dirs():
    for d in (P.base, P.models, P.kaggle_cache, P.personal, P.reports):
        d.mkdir(parents=True, exist_ok=True)


def save_config():
    P.models.mkdir(parents=True, exist_ok=True)
    (P.models / "config.json").write_text(json.dumps(CFG, indent=2))
    (P.models / "label_encoder.json").write_text(json.dumps({"classes": LETTERS}, indent=2))


def load_saved_config():
    p = P.models / "config.json"
    if p.exists():
        CFG.update(json.loads(p.read_text()))


# ------------------------------------------------------------------
# GPU + mixed precision (only when a GPU exists)
# ------------------------------------------------------------------
def setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs available:", gpus)
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            print("Mixed precision ENABLED (mixed_float16).")
        except Exception as e:
            print("Mixed precision not enabled:", e)
    else:
        print("No GPU detected - training runs on CPU (slower). Fine-tuning and live recognition are OK on CPU.")


# ==================================================================
#  MediaPipe (Tasks API) helpers
# ==================================================================
def ensure_hand_model():
    p = P.hand_model
    if p.exists() and p.stat().st_size > 1_000_000:
        return str(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the MediaPipe hand model (~8 MB) ...")
    tmp = p.with_suffix(".part")
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, tmp)
        tmp.replace(p)
    except Exception as e:
        sys.exit(f"Could not download the hand model ({e}).\nDownload it in your browser from\n  {HAND_MODEL_URL}\n"
                 f"and save it as:\n  {p}")
    return str(p)


def make_landmarker(video=False):
    """IMAGE mode for still pictures (Kaggle), VIDEO mode for the webcam."""
    vis = mp.tasks.vision
    kw = dict(base_options=mp.tasks.BaseOptions(model_asset_path=ensure_hand_model()), num_hands=1)
    if video:
        kw.update(running_mode=vis.RunningMode.VIDEO, min_hand_detection_confidence=0.6,
                  min_tracking_confidence=0.6)
    else:
        kw.update(running_mode=vis.RunningMode.IMAGE, min_hand_detection_confidence=0.5)
    return vis.HandLandmarker.create_from_options(vis.HandLandmarkerOptions(**kw))


class VideoClock:
    """Strictly increasing millisecond timestamps (required by MediaPipe VIDEO mode)."""
    def __init__(self):
        self.t0, self.last = time.monotonic(), -1

    def __call__(self):
        ts = max(int((time.monotonic() - self.t0) * 1000), self.last + 1)
        self.last = ts
        return ts


def normalize_landmarks(landmarks_21x3, handedness_label):
    """Wrist-centred, scale-normalised, mirror-corrected 63-dim landmark vector."""
    lm = np.asarray(landmarks_21x3, dtype=np.float32).copy()
    lm -= lm[0]
    lm /= (np.max(np.linalg.norm(lm, axis=1)) + 1e-6)
    if handedness_label == "Left":
        lm[:, 0] *= -1.0
    return lm.flatten()


def crop_hand_region(image_bgr, hand_landmarks, margin=None):
    """Crop around the hand with a safety margin -> (224x224 BGR crop, bbox) or (None, None)."""
    margin = CFG["HAND_MARGIN"] if margin is None else margin
    h, w = image_bgr.shape[:2]
    xs = [p.x for p in hand_landmarks]
    ys = [p.y for p in hand_landmarks]
    x1, x2 = min(xs) * w, max(xs) * w
    y1, y2 = min(ys) * h, max(ys) * h
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1 = int(max(0, x1 - mx)); x2 = int(min(w, x2 + mx))
    y1 = int(max(0, y1 - my)); y2 = int(min(h, y2 + my))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, None
    crop = cv2.resize(image_bgr[y1:y2, x1:x2], (CFG["IMG_SIZE"], CFG["IMG_SIZE"]), interpolation=cv2.INTER_AREA)
    return crop, (x1, y1, x2, y2)


def detect_hand(image_bgr, landmarker, ts_ms=None):
    """Run MediaPipe on one frame. ts_ms=None -> IMAGE mode, else VIDEO mode (increasing ms)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(mp_image) if ts_ms is None else landmarker.detect_for_video(mp_image, ts_ms)
    if not res.hand_landmarks:
        return None
    hand = res.hand_landmarks[0]
    cat = res.handedness[0][0] if res.handedness else None
    crop, bbox = crop_hand_region(image_bgr, hand)
    return {"crop_bgr": crop, "bbox": bbox, "hand_lms": hand,
            "raw": np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32),
            "handed": cat.category_name if cat else "Right",
            "score": float(cat.score) if cat else 0.0}


def extract_from_image(image_bgr, landmarker, ts_ms=None):
    """-> (crop_rgb_uint8, landmark_vec_63, bbox, hand_landmarks) or four Nones."""
    s = detect_hand(image_bgr, landmarker, ts_ms)
    if s is None or s["crop_bgr"] is None:
        return None, None, None, None
    return (cv2.cvtColor(s["crop_bgr"], cv2.COLOR_BGR2RGB),
            normalize_landmarks(s["raw"], s["handed"]), s["bbox"], s["hand_lms"])


def draw_hand_landmarks(image_bgr, hand_landmarks):
    h, w = image_bgr.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image_bgr, pts[a], pts[b], (255, 255, 255), 2)
    for x, y in pts:
        cv2.circle(image_bgr, (x, y), 4, (0, 140, 255), -1)


# ==================================================================
#  Local webcam + keyboard (replaces the Colab JavaScript bridge)
# ==================================================================
class LocalWebcam:
    def __init__(self, cam_index=0, width=640, height=480):
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        self.cap = None
        for b in backends:
            cap = cv2.VideoCapture(cam_index, b)
            if cap.isOpened():
                self.cap = cap
                break
            cap.release()
        if self.cap is None:
            raise RuntimeError(
                f"Could not open webcam (index {cam_index}). Check that a camera is connected and not used by "
                "another app. Try --camera 1. On macOS allow camera access for your terminal / VS Code "
                "(System Settings > Privacy & Security > Camera).")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return cv2.flip(frame, 1)       # mirrored like a mirror - same as the Colab notebook

    def release(self):
        self.cap.release()


def ui_show(window, img):
    cv2.imshow(window, img)


def ui_closed(window):
    try:
        return cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def read_keys(delay=1):
    """Poll the keyboard; returns a list like ['s'], [' '], ['backspace']."""
    k = cv2.waitKey(delay)
    if k == -1:
        return []
    k &= 0xFF
    if k in (8, 127):
        return ["backspace"]
    if k == 27:
        return ["q"]
    return [chr(k).lower()] if 32 <= k < 127 else []


# ==================================================================
#  Kaggle dataset: download -> preprocess (resumable) -> load
# ==================================================================
def download_dataset():
    if P.dataset_root.is_dir():
        print("Dataset already present at", P.dataset_root)
        return
    ensure_dirs()
    if not P.dataset_zip.exists():
        print("Downloading the Kaggle ASL alphabet dataset (~1 GB) ...")
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files("grassknoted/asl-alphabet", path=str(P.base), unzip=False, quiet=False)
        except Exception as e:
            sys.exit(
                f"Automatic download failed ({type(e).__name__}: {e}).\n\n"
                "Option A: put your Kaggle API token in ~/.kaggle/kaggle.json "
                "(kaggle.com > Settings > API > Create New Token) and run --download again.\n"
                "Option B: download the zip yourself from\n"
                "  https://www.kaggle.com/datasets/grassknoted/asl-alphabet\n"
                f"and save it as:\n  {P.dataset_zip}\nthen run --download again.")
    print("Unzipping (A-Z training folders only; a couple of minutes) ...")
    with zipfile.ZipFile(P.dataset_zip) as zf:
        wanted = []
        for m in zf.namelist():
            parts = m.split("/")
            if len(parts) >= 4 and parts[0] == "asl_alphabet_train" and parts[2] in LETTERS and parts[-1]:
                wanted.append(m)
        for m in tqdm(wanted or zf.namelist(), desc="Extracting"):
            zf.extract(m, P.base)
    print("Dataset ready:", P.dataset_root.is_dir())
    print("Next: python asl_local.py --preprocess")


def _process_letter(letter, landmarker, max_per_class):
    files = sorted(glob.glob(str(P.dataset_root / letter / "*.jpg")))
    random.Random(SEED).shuffle(files)
    imgs, lms = [], []
    for fp in files:
        if len(imgs) >= max_per_class:
            break
        img = cv2.imread(fp)
        if img is None:
            continue
        crop, vec, _, _ = extract_from_image(img, landmarker)
        if crop is None:
            continue                                   # no hand found -> skip (keeps image/landmark data paired)
        imgs.append(crop.astype(np.uint8))
        lms.append(vec)
    return imgs, lms, len(imgs) < max_per_class


def preprocess_dataset(max_per_class=None, force=False):
    """MediaPipe crop + landmarks for every Kaggle image. One cache file per letter -> safe to interrupt."""
    max_per_class = int(max_per_class or CFG["MAX_PER_CLASS"])
    if not P.dataset_root.is_dir():
        sys.exit("Kaggle dataset not found. Run:  python asl_local.py --download")
    ensure_dirs()
    S = CFG["IMG_SIZE"]
    landmarker = make_landmarker(video=False)
    try:
        for letter in tqdm(LETTERS, desc="Letters"):
            path = P.kaggle_cache / f"{letter}.npz"
            if path.exists() and not force:
                with np.load(path) as z:
                    if int(z["max_per_class"]) >= max_per_class or bool(z["exhausted"]):
                        continue                       # already have enough for this letter
            imgs, lms, exhausted = _process_letter(letter, landmarker, max_per_class)
            if not imgs:
                print(f"WARNING: no usable hands found for letter {letter}")
            tmp = P.kaggle_cache / f"{letter}.tmp.npz"
            np.savez(tmp,
                     X_img=np.stack(imgs) if imgs else np.zeros((0, S, S, 3), np.uint8),
                     X_lm=np.stack(lms).astype(np.float32) if lms else np.zeros((0, 63), np.float32),
                     n=np.int64(len(imgs)), max_per_class=np.int64(max_per_class), exhausted=np.bool_(exhausted))
            tmp.replace(path)
    finally:
        landmarker.close()
    X_img, X_lm, y = load_kaggle_cache(max_per_class)
    print("Images:", X_img.shape, "| Landmarks:", X_lm.shape, "| Labels:", y.shape)
    print("Per-class counts:", dict(sorted(Counter(LETTERS[i] for i in y).items())))
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    for ax, letter in zip(axes.flat, CFG["HARD_LETTERS"]):
        idxs = np.where(y == LETTER_TO_IDX[letter])[0]
        if len(idxs):
            ax.imshow(X_img[idxs[0]]); ax.set_title(letter)
        ax.axis("off")
    plt.suptitle("Cropped hand images - the 16 'hard' letters")
    out = P.reports / "kaggle_sanity_check.png"
    fig.savefig(out, dpi=120, bbox_inches="tight"); plt.close(fig)
    print("Saved", out)
    print("Next: python asl_local.py --train")


def load_kaggle_cache(max_per_class=None, required=True):
    max_per_class = int(max_per_class or CFG["MAX_PER_CLASS"])
    S = CFG["IMG_SIZE"]
    files = [(l, P.kaggle_cache / f"{l}.npz") for l in LETTERS]
    missing = [l for l, f in files if not f.exists()]
    if missing:
        if required:
            sys.exit(f"Kaggle cache incomplete (missing: {' '.join(missing)}). Run:  python asl_local.py --preprocess")
        return None
    counts = []
    for _, f in files:
        with np.load(f) as z:
            counts.append(min(int(z["n"]), max_per_class))
    N = sum(counts)
    X_img = np.empty((N, S, S, 3), np.uint8)
    X_lm = np.empty((N, 63), np.float32)
    y = np.empty(N, np.int64)
    pos = 0
    for (letter, f), c in zip(files, counts):
        with np.load(f) as z:
            X_img[pos:pos + c] = z["X_img"][:c]
            X_lm[pos:pos + c] = z["X_lm"][:c]
        y[pos:pos + c] = LETTER_TO_IDX[letter]
        pos += c
    return X_img, X_lm, y


def kaggle_split(y_all):
    return train_test_split(np.arange(len(y_all)), test_size=CFG["VAL_SPLIT"], stratify=y_all, random_state=SEED)


# ==================================================================
#  Augmentation, loss, architectures
# ==================================================================
@keras.saving.register_keras_serializable(package="asl_hybrid")
class PixelScaleGaussianNoise(layers.Layer):
    """Additive Gaussian noise for inputs on the raw 0-255 pixel scale."""
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.stddev, dtype=inputs.dtype)
            return inputs + noise
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config


def build_image_augmentation():
    """Only active while training. No horizontal flips: mirroring changes the meaning of many ASL signs."""
    return tf.keras.Sequential([
        layers.RandomRotation(0.06, fill_mode="nearest"),
        layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
        layers.RandomZoom(0.12, fill_mode="nearest"),
        layers.RandomBrightness(0.20, value_range=(0, 255)),
        layers.RandomContrast(0.20),
        PixelScaleGaussianNoise(4.0),
    ], name="augmentation")


def augment_landmark_vector(vec63, rng):
    """Meaning-preserving landmark augmentation: rotate in-plane, rescale, jitter."""
    lm = vec63.reshape(21, 3).copy()
    theta = rng.uniform(-0.20, 0.20)
    c, s = math.cos(theta), math.sin(theta)
    x, yc = lm[:, 0].copy(), lm[:, 1].copy()
    lm[:, 0] = c * x - s * yc
    lm[:, 1] = s * x + c * yc
    lm *= rng.uniform(0.92, 1.08)
    lm += rng.normal(0.0, 0.012, lm.shape)
    return lm.flatten().astype(np.float32)


def expand_landmark_dataset(X, y, w, copies=3, seed=SEED):
    rng = np.random.RandomState(seed)
    Xs, ys, ws = [X], [y], [w]
    for _ in range(copies):
        Xs.append(np.stack([augment_landmark_vector(v, rng) for v in X]))
        ys.append(y); ws.append(w)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


def categorical_focal_loss(gamma=None):
    gamma = CFG["FOCAL_GAMMA"] if gamma is None else gamma

    def loss_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(tf.pow(1.0 - y_pred, gamma) * ce, axis=-1)
    return loss_fn


def custom_objects():
    return {"loss_fn": categorical_focal_loss(), "PixelScaleGaussianNoise": PixelScaleGaussianNoise}


def build_image_model(num_classes=NUM_CLASSES):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(CFG["IMG_SIZE"], CFG["IMG_SIZE"], 3))
    base.trainable = False
    inp = layers.Input((CFG["IMG_SIZE"], CFG["IMG_SIZE"], 3), name="hand_image")
    x = build_image_augmentation()(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="probs")(x)
    return models.Model(inp, out, name="image_branch"), base


def build_landmark_model(num_classes=NUM_CLASSES):
    inp = layers.Input((63,), name="landmarks")
    x = layers.Dense(256)(inp); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(256)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128)(x); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="probs")(x)
    return models.Model(inp, out, name="landmark_branch")


# ==================================================================
#  Class weights, oversampling, tf.data, callbacks, prediction helpers
# ==================================================================
def compute_sample_weights(y_indices, extra_boost=None):
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_indices)))
    for l in CFG["HARD_LETTERS"]:
        cw[LETTER_TO_IDX[l]] *= CFG["HARD_BOOST"]
    for cls_idx, mult in (extra_boost or {}).items():
        cw[cls_idx] *= mult
    return np.array([cw[c] for c in y_indices], dtype=np.float32), cw


def oversample_indices(idx_array, y_indices, frac=None, seed=SEED):
    frac = CFG["HARD_OVERSAMPLE"] if frac is None else frac
    rng = np.random.RandomState(seed)
    extra = []
    for cls in {LETTER_TO_IDX[l] for l in CFG["HARD_LETTERS"]}:
        pos = idx_array[y_indices == cls]
        n_extra = int(len(pos) * frac)
        if n_extra > 0:
            extra.append(rng.choice(pos, n_extra, replace=True))
    out = np.concatenate([idx_array] + extra) if extra else idx_array.copy()
    rng.shuffle(out)
    return out


def make_image_ds(X_img, y_all, sample_w, indices, training):
    """tf.data over a big uint8 array. tf.numpy_function avoids embedding a multi-GB constant in the graph."""
    S = CFG["IMG_SIZE"]

    def _fetch(i):
        i = int(i)
        return (X_img[i].astype(np.float32), np.eye(NUM_CLASSES, dtype=np.float32)[y_all[i]],
                np.float32(sample_w[i]))

    ds = tf.data.Dataset.from_tensor_slices(indices.astype(np.int64))
    if training:
        ds = ds.shuffle(len(indices), seed=SEED, reshuffle_each_iteration=True)

    def _map(i):
        img, lab, w = tf.numpy_function(_fetch, [i], (tf.float32, tf.float32, tf.float32))
        img.set_shape((S, S, 3)); lab.set_shape((NUM_CLASSES,)); w.set_shape(())
        return img, lab, w

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(CFG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)


def make_mixed_ds(personal_img, kaggle_img, source, index, y, w, training):
    """Fine-tuning pipeline mixing two arrays: source 0 = personal, 1 = Kaggle replay."""
    S = CFG["IMG_SIZE"]

    def _fetch(src, idx):
        src, idx = int(src), int(idx)
        return (personal_img[idx] if src == 0 else kaggle_img[idx]).astype(np.float32)

    y_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y]
    ds = tf.data.Dataset.from_tensor_slices((source.astype(np.int64), index.astype(np.int64), y_oh,
                                             w.astype(np.float32)))
    if training:
        ds = ds.shuffle(len(index), seed=SEED, reshuffle_each_iteration=True)

    def _map(s, i, yy, ww):
        img = tf.numpy_function(_fetch, [s, i], tf.float32)
        img.set_shape((S, S, 3)); yy.set_shape((NUM_CLASSES,)); ww.set_shape(())
        return img, yy, ww

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(CFG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)


def make_callbacks(ckpt_path, patience=6, keep_if_better_than=None):
    """keep_if_better_than: a fine-tune phase may only overwrite the checkpoint if it beats this val_accuracy."""
    kw = {"initial_value_threshold": float(keep_if_better_than)} if keep_if_better_than is not None else {}
    return [
        callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True, verbose=1, **kw),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=patience, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]


def predict_chunked(model, X_uint8, chunk=512, batch_size=64):
    """Predict on a big uint8 array without ever holding it all as float32."""
    out = []
    for i in range(0, len(X_uint8), chunk):
        out.append(model.predict(X_uint8[i:i + chunk].astype(np.float32), batch_size=batch_size, verbose=0))
    return np.concatenate(out) if out else np.zeros((0, NUM_CLASSES), np.float32)


def tune_alpha(p_img, p_lm, y_true, tol=0.002):
    """Pick the fusion weight alpha. Among alphas whose accuracy is within `tol` of the best, take the one with
    the lowest log-loss. (The old loop kept the FIRST alpha that reached the top score, so a saturated 100%
    validation set always produced alpha=0.00 = image branch silently switched off.)"""
    rows = []
    idx = np.arange(len(y_true))
    for a in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        p = a * p_img + (1 - a) * p_lm
        acc = float(accuracy_score(y_true, p.argmax(1)))
        nll = float(-np.mean(np.log(np.clip(p[idx, y_true], 1e-7, 1.0))))
        rows.append((float(a), acc, nll))
    best_acc = max(r[1] for r in rows)
    cands = [r for r in rows if r[1] >= best_acc - tol]
    # round the log-loss so float noise / negligible differences count as ties -> then prefer the middle (0.5)
    return min(cands, key=lambda r: (round(r[2], 3), abs(r[0] - 0.5)))[:2]


def merge_hist(*hs):
    out = {}
    for h in hs:
        for k, v in h.history.items():
            out.setdefault(k, []).extend(float(x) for x in v)
    return out


def plot_curves(hist_img, hist_lm, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for key, style in [("accuracy", "-"), ("val_accuracy", "--")]:
        axes[0].plot(hist_img[key], style, label="img " + key)
        axes[0].plot(hist_lm[key], style, label="lm " + key)
    for key, style in [("loss", "-"), ("val_loss", "--")]:
        axes[1].plot(hist_img[key], style, label="img " + key)
        axes[1].plot(hist_lm[key], style, label="lm " + key)
    axes[0].set_title("Accuracy"); axes[1].set_title("Loss (focal)")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("epoch")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_confusion(cm, title, path):
    fig = plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LETTERS, yticklabels=LETTERS, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def plot_per_class(cm, title, path):
    acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    fig = plt.figure(figsize=(13, 3))
    plt.bar(LETTERS, acc, color=["#d62728" if l in CFG["HARD_LETTERS"] else "#1f77b4" for l in LETTERS])
    plt.ylim(min(0.8, float(acc.min()) - 0.02), 1.0)
    plt.title(title + "  (red = designated hard letters)"); plt.grid(alpha=0.3, axis="y")
    fig.savefig(path, dpi=120, bbox_inches="tight"); plt.close(fig)


def confused_pairs(cm):
    pairs = [(int(cm[i, j]), LETTERS[i], LETTERS[j])
             for i in range(NUM_CLASSES) for j in range(NUM_CLASSES) if i != j and cm[i, j] > 0]
    return sorted(pairs, reverse=True)


# ==================================================================
#  Stage: train
# ==================================================================
def train_models():
    X_img, X_lm, y_all = load_kaggle_cache()
    train_idx, val_idx = kaggle_split(y_all)
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    print(f"Train: {len(train_idx)}  |  Val: {len(val_idx)}")
    sample_w, _ = compute_sample_weights(y_all)
    train_idx_os = oversample_indices(train_idx, y_train)
    print(f"Training samples after hard-letter oversampling: {len(train_idx_os)} (was {len(train_idx)})")
    train_ds = make_image_ds(X_img, y_all, sample_w, train_idx_os, training=True)
    val_ds = make_image_ds(X_img, y_all, sample_w, val_idx, training=False)

    image_model, effnet_base = build_image_model()
    landmark_model = build_landmark_model()
    ensure_dirs()

    # Phase 1: train the head (backbone frozen)
    image_model.compile(optimizer=optimizers.Adam(1e-3), loss=categorical_focal_loss(),
                        metrics=["accuracy"], weighted_metrics=[])
    h1 = image_model.fit(train_ds, validation_data=val_ds, epochs=12, callbacks=make_callbacks(ckpt(IMG_BEST)))

    # Phase 2: unfreeze the top of EfficientNet, very low LR (may only overwrite the checkpoint if it improves)
    effnet_base.trainable = True
    for layer in effnet_base.layers[:-60]:
        layer.trainable = False
    image_model.compile(optimizer=optimizers.Adam(1e-5), loss=categorical_focal_loss(),
                        metrics=["accuracy"], weighted_metrics=[])
    h2 = image_model.fit(train_ds, validation_data=val_ds, epochs=15,
                         callbacks=make_callbacks(ckpt(IMG_BEST), patience=5,
                                                  keep_if_better_than=max(h1.history["val_accuracy"])))

    # Landmark model on augmented + oversampled landmark data
    wlm = sample_w[train_idx_os]
    Xlm, ylm, wlm = expand_landmark_dataset(X_lm[train_idx_os], y_all[train_idx_os], wlm, copies=3)
    landmark_model.compile(optimizer=optimizers.Adam(1e-3), loss=categorical_focal_loss(),
                           metrics=["accuracy"], weighted_metrics=[])
    hl = landmark_model.fit(Xlm, np.eye(NUM_CLASSES, dtype=np.float32)[ylm], sample_weight=wlm,
                            validation_data=(X_lm[val_idx], np.eye(NUM_CLASSES, dtype=np.float32)[y_val]),
                            epochs=80, batch_size=128, callbacks=make_callbacks(ckpt(LM_BEST), patience=10),
                            verbose=2)

    hist_img = merge_hist(h1, h2)
    (P.models / "training_history.json").write_text(json.dumps({"image_model": hist_img,
                                                                  "landmark_model": merge_hist(hl)}))
    plot_curves(hist_img, merge_hist(hl), P.reports / "training_curves.png")
    save_config()
    print("Training complete. Checkpoints saved to", P.models)
    print("Next: python asl_local.py --evaluate")


# ==================================================================
#  Stage: evaluate (+ confusion-driven retraining)
# ==================================================================
def _load_pair(img_name, lm_name):
    for n in (img_name, lm_name):
        if not ckpt(n).exists():
            sys.exit(f"Missing {ckpt(n)}.  Train first (--train) or reuse Colab models (--import-models).")
    return (tf.keras.models.load_model(str(ckpt(img_name)), compile=False, custom_objects=custom_objects()),
            tf.keras.models.load_model(str(ckpt(lm_name)), compile=False, custom_objects=custom_objects()))


def evaluate_models():
    X_img, X_lm, y_all = load_kaggle_cache()
    train_idx, val_idx = kaggle_split(y_all)
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    image_model, landmark_model = _load_pair(IMG_BEST, LM_BEST)

    def probs(idx):
        return predict_chunked(image_model, X_img[idx]), landmark_model.predict(X_lm[idx], batch_size=256, verbose=0)

    p_img, p_lm = probs(val_idx)
    alpha, acc = tune_alpha(p_img, p_lm, y_val)
    CFG["FUSION_ALPHA"] = alpha
    y_pred = np.argmax(alpha * p_img + (1 - alpha) * p_lm, axis=1)
    lines = [f"Image-only val accuracy   : {accuracy_score(y_val, p_img.argmax(1)):.4f}",
             f"Landmark-only val accuracy: {accuracy_score(y_val, p_lm.argmax(1)):.4f}",
             f"FUSED val accuracy (alpha={alpha:.2f}): {acc:.4f}"]
    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="macro", zero_division=0)
    lines.append(f"Macro Precision: {prec:.4f} | Macro Recall: {rec:.4f} | Macro F1: {f1:.4f}\n")
    lines.append(classification_report(y_val, y_pred, target_names=LETTERS, zero_division=0))
    cm = confusion_matrix(y_val, y_pred, labels=np.arange(NUM_CLASSES))
    pairs = confused_pairs(cm)
    lines.append("Most confused pairs (true -> predicted):")
    lines += [f"  {a} -> {b}: {n} errors" for n, a, b in pairs[:12]]
    plot_confusion(cm, "Fused-model confusion matrix (Kaggle validation)", P.reports / "confusion_kaggle.png")
    plot_per_class(cm, "Per-class accuracy (Kaggle validation)", P.reports / "per_class_kaggle.png")

    # Confusion-driven retraining: boost the classes in the worst confusions, oversample, fine-tune again
    extra_boost = {}
    for n, a, b in pairs:
        if n > 2:
            for l in (a, b):
                extra_boost[LETTER_TO_IDX[l]] = max(extra_boost.get(LETTER_TO_IDX[l], 1.0), 1.5)
    if extra_boost:
        lines.append("Boosting classes: " + " ".join(sorted(LETTERS[c] for c in extra_boost)))
        print("\n".join(lines))
        sample_w, _ = compute_sample_weights(y_all, extra_boost=extra_boost)
        train_idx_os = oversample_indices(train_idx, y_train, frac=CFG["HARD_OVERSAMPLE"] + 0.25)
        image_model.compile(optimizer=optimizers.Adam(5e-6), loss=categorical_focal_loss(),
                            metrics=["accuracy"], weighted_metrics=[])
        image_model.fit(make_image_ds(X_img, y_all, sample_w, train_idx_os, True),
                        validation_data=make_image_ds(X_img, y_all, sample_w, val_idx, False), epochs=6,
                        callbacks=make_callbacks(ckpt(IMG_BEST), patience=3,
                                                 keep_if_better_than=accuracy_score(y_val, p_img.argmax(1))))
        Xb, yb, wb = expand_landmark_dataset(X_lm[train_idx_os], y_all[train_idx_os], sample_w[train_idx_os], copies=2)
        landmark_model.compile(optimizer=optimizers.Adam(2e-4), loss=categorical_focal_loss(),
                               metrics=["accuracy"], weighted_metrics=[])
        landmark_model.fit(Xb, np.eye(NUM_CLASSES, dtype=np.float32)[yb], sample_weight=wb,
                           validation_data=(X_lm[val_idx], np.eye(NUM_CLASSES, dtype=np.float32)[y_val]),
                           epochs=20, batch_size=128,
                           callbacks=make_callbacks(ckpt(LM_BEST), patience=6,
                                                    keep_if_better_than=accuracy_score(y_val, p_lm.argmax(1))),
                           verbose=2)
        image_model, landmark_model = _load_pair(IMG_BEST, LM_BEST)
        p_img, p_lm = probs(val_idx)
        alpha, acc = tune_alpha(p_img, p_lm, y_val)
        CFG["FUSION_ALPHA"] = alpha
        final = f"\nPost-retrain FUSED val accuracy (alpha={alpha:.2f}): {acc:.4f}"
        print(final)
        lines.append(final)
    else:
        print("\n".join(lines))
    (P.reports / "evaluation_kaggle.txt").write_text("\n".join(lines))
    save_config()
    print("\nSaved reports to", P.reports)
    print("NOTE: this is accuracy on the Kaggle validation split. Real-world accuracy on YOUR hand is lower and "
          "is measured by --finetune on a held-out recording.")
    print("Next: python asl_local.py --collect-personal --session s01")


# ==================================================================
#  Personal dataset (same folder format as the Dataset Builder notebook)
# ==================================================================
def load_meta(check_files=True):
    if not P.meta.exists():
        return pd.DataFrame(columns=META_COLS)
    df = pd.read_csv(P.meta, keep_default_na=False)
    if check_files and len(df):                                # images you deleted by hand are ignored
        present = {l: set(os.listdir(P.crops / l)) if (P.crops / l).is_dir() else set() for l in df["letter"].unique()}
        keep = [os.path.basename(f) in present[l] for l, f in zip(df["letter"], df["file"])]
        df = df[keep].reset_index(drop=True)
    return df


def prune_metadata():
    df = load_meta()
    df.to_csv(P.meta, index=False)
    print(f"metadata.csv now has {len(df):,} rows.")


def _append_meta(rows):
    if rows:
        pd.DataFrame(rows, columns=META_COLS).to_csv(P.meta, mode="a", header=not P.meta.exists(), index=False)


def next_index(letter, session):
    prefix, best = f"{letter}_{session}_", 0
    d = P.crops / letter
    for fn in (os.listdir(d) if d.is_dir() else []):
        if fn.startswith(prefix):
            try:
                best = max(best, int(os.path.splitext(fn)[0].split("_")[-1]))
            except ValueError:
                pass
    return best + 1


def check_quality(s):
    """-> (ok, message, blur)"""
    if s["crop_bgr"] is None:
        return False, "Hand too small - move closer", 0.0
    raw = s["raw"]
    if raw[:, 0].min() < 0 or raw[:, 0].max() > 1 or raw[:, 1].min() < 0 or raw[:, 1].max() > 1:
        return False, "Keep your WHOLE hand inside the frame", 0.0
    if s["score"] < CFG["MIN_HAND_SCORE"]:
        return False, "Hand not clear - show it better", 0.0
    blur = float(cv2.Laplacian(cv2.cvtColor(s["crop_bgr"], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if blur < CFG["BLUR_MIN"]:
        return False, "Blurry - move a bit slower", blur
    return True, "", blur


def is_new_pose(raw_xy, recent):
    return all(np.abs(raw_xy - r).mean() >= CFG["MIN_POSE_CHANGE"] for r in recent)


def save_sample(s, frame_bgr, letter, session, phase_id, blur, idx):
    name = f"{letter}_{session}_{idx:04d}.jpg"
    (P.crops / letter).mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(P.crops / letter / name), s["crop_bgr"], [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Could not write {P.crops / letter / name} - is the disk full?")
    if CFG["SAVE_FULL_FRAME"]:
        (P.frames / letter).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(P.frames / letter / name), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    x1, y1, x2, y2 = [int(v) for v in s["bbox"]]
    row = {"file": f"{letter}/{name}", "frame_file": f"{letter}/{name}" if CFG["SAVE_FULL_FRAME"] else "",
           "letter": letter, "session": session, "phase_id": int(phase_id), "phase": ANGLE_PHASES[phase_id][0],
           "handedness": s["handed"], "hand_score": round(s["score"], 4), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
           "frame_w": int(frame_bgr.shape[1]), "frame_h": int(frame_bgr.shape[0]), "blur": round(blur, 1),
           "timestamp": datetime.now().isoformat(timespec="seconds")}
    row.update(dict(zip(LM_COLS, [float(v) for v in s["raw"].reshape(-1)])))
    return row


def draw_hud(overlay, letter, have, target, phase, phase_have, quota, capturing, msg, msg_color):
    h, w = overlay.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(overlay, f"Letter {letter}  [{have}/{target}]  " + ("CAPTURING" if capturing else "PAUSED"),
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0) if capturing else (0, 200, 255), 2)
    cv2.rectangle(overlay, (0, 34), (w, 68), (40, 40, 40), -1)
    cv2.putText(overlay, f"Angle {phase + 1}/{len(ANGLE_PHASES)}: {ANGLE_PHASES[phase][1]}  [{phase_have}/{quota}]",
                (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(overlay, (0, 68), (int(w * min(1.0, have / max(1, target))), 74), (0, 200, 0), -1)
    cv2.rectangle(overlay, (0, h - 34), (w, h), (0, 0, 0), -1)
    cv2.putText(overlay, msg, (8, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, msg_color, 2)
    cv2.putText(overlay, letter, (w - 90, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5)


def collect_personal_dataset(letters=None, per_letter=None, session="s01", cam_index=0):
    """Record YOUR hand, guided through 10 angles per letter. Resumable: run again with the same --session."""
    letters = letters or LETTERS
    per_letter = int(per_letter or CFG["PERSONAL_PER_LETTER"])
    if not re.fullmatch(r"[A-Za-z0-9-]+", session) or session.lower() == "legacy":
        sys.exit("--session may contain only letters, digits and '-' (and 'legacy' is reserved for imported data).")
    ensure_dirs()
    n_phases = len(ANGLE_PHASES)
    quota = math.ceil(per_letter / n_phases)
    df0 = load_meta()
    existing = list(dict.fromkeys(df0["session"])) if len(df0) else []
    print(f"Existing sessions: {existing or 'none'}  |  recording into session '{session}'")
    counts = Counter(zip(df0["session"], df0["letter"], df0["phase_id"].astype(int))) if len(df0) else Counter()

    cam = LocalWebcam(cam_index)
    landmarker = make_landmarker(video=True)
    clock = VideoClock()
    window = "ASL - record your hand   (s start/pause | p next angle | n skip letter | q quit)"
    buffer, saved_run, quit_all, misses = [], 0, False, 0
    try:
        for letter in letters:
            if quit_all:
                break
            pc = [counts.get((session, letter, p), 0) for p in range(n_phases)]
            have = sum(pc)
            if have >= per_letter:
                print(f"{letter}: already has {have} images in session {session}, skipping.")
                continue
            phase = next((p for p in range(n_phases) if pc[p] < quota), n_phases - 1)
            idx = next_index(letter, session)
            capturing, skip_letter = False, False
            recent = deque(maxlen=CFG["RECENT_K"])
            last_save, grace_until = 0.0, 0.0
            msg, msg_color = "Press s to start capturing", (0, 200, 255)
            while have < per_letter:
                frame = cam.read()
                if frame is None:
                    misses += 1
                    if misses > 60:
                        print("Camera is not delivering frames - stopping.")
                        quit_all = True
                        break
                    read_keys(10)
                    continue
                misses = 0
                s = detect_hand(frame, landmarker, clock())
                overlay = frame.copy()
                if s is not None:
                    if s["bbox"] is not None:
                        x1, y1, x2, y2 = s["bbox"]
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    draw_hand_landmarks(overlay, s["hand_lms"])
                now = time.time()
                if not capturing:
                    msg, msg_color = "PAUSED - press s to start", (0, 200, 255)
                elif now < grace_until:
                    msg, msg_color = "NEXT: " + ANGLE_PHASES[phase][1], (0, 255, 255)
                elif s is None:
                    msg, msg_color = "No hand detected", (60, 60, 255)
                else:
                    ok, why, blur = check_quality(s)
                    if not ok:
                        msg, msg_color = why, (60, 60, 255)
                    elif now - last_save < CFG["MIN_SAVE_GAP_S"]:
                        pass
                    elif not is_new_pose(s["raw"][:, :2], recent):
                        msg, msg_color = "Move your hand a little - need a NEW angle", (0, 165, 255)
                    else:
                        buffer.append(save_sample(s, frame, letter, session, phase, blur, idx))
                        counts[(session, letter, phase)] += 1
                        pc[phase] += 1
                        have += 1; idx += 1; saved_run += 1
                        last_save = now
                        recent.append(s["raw"][:, :2].copy())
                        msg, msg_color = f"Saved {letter} #{have}", (0, 255, 0)
                        if len(buffer) >= 25:
                            _append_meta(buffer); buffer = []
                        if pc[phase] >= quota and phase < n_phases - 1:
                            phase += 1
                            recent.clear()
                            grace_until = now + CFG["GRACE_SECONDS"]
                draw_hud(overlay, letter, have, per_letter, phase, pc[phase], quota, capturing, msg, msg_color)
                ui_show(window, overlay)
                for k in read_keys():
                    if k == "s":
                        capturing = not capturing
                        recent.clear()
                    elif k == "p" and phase < n_phases - 1:
                        phase += 1
                        recent.clear()
                        grace_until = time.time() + CFG["GRACE_SECONDS"]
                    elif k == "n":
                        skip_letter = True
                    elif k == "q":
                        quit_all = True
                if ui_closed(window):
                    quit_all = True
                if quit_all or skip_letter:
                    break
            _append_meta(buffer); buffer = []
            tag = "skipped" if skip_letter else ("stopped" if quit_all else "done")
            print(f"{letter}: {tag} - {have}/{per_letter} images in session {session}")
    finally:
        _append_meta(buffer)
        landmarker.close()
        cam.release()
        cv2.destroyAllWindows()
    print(f"\nSaved {saved_run} new images this run. Personal dataset total: {len(load_meta()):,} images "
          f"(folder: {P.personal})")
    print("Next: python asl_local.py --personal-report   then   python asl_local.py --finetune")


def import_legacy_npz(path):
    """Convert an OLD Colab personal_asl.npz (X_img, X_lm, y) into the folder format, session 'legacy'."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"File not found: {path}")
    ensure_dirs()
    z = np.load(path)
    for k in ("X_img", "X_lm", "y"):
        if k not in z.files:
            sys.exit(f"{path.name} has no '{k}' array - is it really a personal_asl.npz from the notebook?")
    X_img, X_lm, y = z["X_img"], z["X_lm"], z["y"]
    S = CFG["IMG_SIZE"]
    if X_img.ndim != 4 or X_img.shape[3] != 3 or X_lm.shape != (len(y), 63) or len(X_img) != len(y):
        sys.exit(f"Unexpected shapes: X_img {X_img.shape}, X_lm {X_lm.shape}, y {y.shape}")
    df0 = load_meta()
    if len(df0) and (df0["session"] == "legacy").any():
        sys.exit("A 'legacy' session was already imported. To re-import, delete those rows/files first.")
    counters, rows = Counter(), []
    for i in tqdm(range(len(y)), desc="Importing"):
        letter = LETTERS[int(y[i])]
        counters[letter] += 1
        name = f"{letter}_legacy_{counters[letter]:04d}.jpg"
        (P.crops / letter).mkdir(parents=True, exist_ok=True)
        img = X_img[i]
        if img.shape[:2] != (S, S):
            img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(P.crops / letter / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        row = {"file": f"{letter}/{name}", "frame_file": "", "letter": letter, "session": "legacy",
               "phase_id": -1, "phase": "legacy", "handedness": "Right", "hand_score": 1.0,
               "x1": 0, "y1": 0, "x2": 0, "y2": 0, "frame_w": 0, "frame_h": 0,
               "blur": round(float(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()), 1),
               "timestamp": "legacy-import"}
        # the old npz already stores NORMALISED landmarks (wrist-centred, left hands mirrored) -> keep as is
        row.update(dict(zip(LM_COLS, [float(v) for v in X_lm[i]])))
        rows.append(row)
    _append_meta(rows)
    print(f"Imported {len(rows):,} images as session 'legacy' ({len(counters)} letters).")
    print("Heads-up: legacy frames are near-duplicates from ONE recording, so they help TRAINING but cannot give "
          "an honest accuracy number. Record a fresh session to measure real accuracy:")
    print("  python asl_local.py --collect-personal --session s01")


def import_models(path):
    """Copy models trained in Colab (zip or folder) into asl_project/models/."""
    path = Path(path)
    if not path.exists():
        sys.exit(f"Not found: {path}")
    ensure_dirs()
    tmp = None
    src = path
    if path.is_file():
        if path.suffix.lower() != ".zip":
            sys.exit("Give me the asl_model_export.zip (or the unzipped folder).")
        tmp = tempfile.mkdtemp(prefix="asl_models_")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        src = Path(tmp)
    copied = []
    try:
        for pattern in ("*.keras", "config.json", "label_encoder.json", "training_history.*"):
            for f in src.rglob(pattern):
                shutil.copy(f, P.models / f.name)
                copied.append(f.name)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    if not copied:
        sys.exit("No .keras models found there.")
    print("Imported:", ", ".join(sorted(set(copied))))
    load_saved_config()
    if CFG["FUSION_ALPHA"] in (0.0, 1.0):
        print(f"\nNOTE: the imported config has FUSION_ALPHA={CFG['FUSION_ALPHA']:.2f}, i.e. ONE branch is switched "
              "off (a tie at 100% validation accuracy picks alpha=0). Re-tune it with --finetune, or test "
              "with  --realtime --alpha 0.5")
    print("Next: python asl_local.py --realtime     (or --finetune after recording your hand)")


def split_personal(df):
    """Honest train/validation split of the personal data -> (is_val, strength, description).

    STRICT : the newest recorded session is held out entirely (different light/background/day)
    MEDIUM : only one session -> whole ANGLES are held out
    WEAK   : only legacy frames (no angle info) -> last 20% of each letter's frames (same session; optimistic)
    """
    sessions = list(dict.fromkeys(df["session"]))
    new_sessions = [s for s in sessions if s != "legacy"]
    if len(sessions) >= 2 and new_sessions:
        val_s = new_sessions[-1]
        others = [s for s in sessions if s != val_s]
        return ((df["session"] == val_s).to_numpy(), "STRICT",
                f"validation = whole session '{val_s}' (never trained on); training = {', '.join(others)}")
    if len(new_sessions) == 1:
        is_val = df["phase_id"].isin(HOLDOUT_PHASE_IDS).to_numpy()
        if is_val.any():
            names = [ANGLE_PHASES[p][0] for p in HOLDOUT_PHASE_IDS]
            return (is_val, "MEDIUM", f"one session only -> validation = unseen angles {names}")
    is_val = np.zeros(len(df), bool)
    letters = df["letter"].to_numpy()
    for letter in np.unique(letters):
        pos = np.where(letters == letter)[0]
        k = max(1, int(round(len(pos) * 0.2)))
        is_val[pos[-k:]] = True
    return (is_val, "WEAK", "legacy frames only -> validation = last 20% of each letter's frames (same recording; "
                            "accuracy will be OPTIMISTIC)")


def load_personal_arrays(df):
    S = CFG["IMG_SIZE"]
    n = len(df)
    X_img = np.empty((n, S, S, 3), np.uint8)
    X_lm = np.empty((n, 63), np.float32)
    raw_all = df[LM_COLS].to_numpy(np.float32).reshape(n, 21, 3)
    for i, (f, hand) in enumerate(tqdm(list(zip(df["file"], df["handedness"])), desc="Loading personal images")):
        img = cv2.imread(str(P.crops / f))
        if img is None:
            raise RuntimeError(f"Cannot read {P.crops / f}")
        if img.shape[:2] != (S, S):
            img = cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)
        X_img[i] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_lm[i] = normalize_landmarks(raw_all[i], hand)
    y = df["letter"].map(LETTER_TO_IDX).to_numpy(np.int64)
    return X_img, X_lm, y


# ==================================================================
#  Personal data report (charts for slides)
# ==================================================================
def personal_report():
    df = load_meta()
    if len(df) == 0:
        sys.exit("No personal images yet. Record some with --collect-personal or import with --import-npz.")
    ensure_dirs()
    per = df["letter"].value_counts().reindex(LETTERS, fill_value=0)
    print(f"TOTAL: {len(df):,} images | {int((per > 0).sum())}/26 letters | sessions: "
          f"{list(dict.fromkeys(df['session']))}")
    tab = pd.crosstab(df["letter"], df["session"]).reindex(LETTERS, fill_value=0)
    print(tab.to_string())
    few = [l for l in LETTERS if per[l] < 50]
    if few:
        print("Letters with fewer than 50 images:", " ".join(few))
    is_val, strength, how = split_personal(df)
    print(f"\nAccuracy honesty if you fine-tune now: {strength} - {how}")

    fig, ax = plt.subplots(figsize=(12, 4.2))
    bottom = np.zeros(len(LETTERS))
    for sess in tab.columns:
        ax.bar(LETTERS, tab[sess].values, bottom=bottom, label=str(sess), width=0.7)
        bottom += tab[sess].values
    ax.set_title(f"{len(df):,} images of one signer's hand"); ax.set_ylabel("images per letter")
    ax.legend(title="session"); ax.grid(axis="y", alpha=0.3)
    fig.savefig(P.reports / "personal_images_per_letter.png", dpi=150, bbox_inches="tight"); plt.close(fig)

    new = df[df["phase_id"].astype(int) >= 0]
    if len(new):
        grid = pd.crosstab(new["letter"], new["phase_id"].astype(int)).reindex(
            index=LETTERS, columns=range(len(ANGLE_PHASES)), fill_value=0)
        fig, ax = plt.subplots(figsize=(10.5, 8))
        im = ax.imshow(grid.values, cmap="Blues", aspect="auto", vmin=0)
        ax.set_xticks(range(len(ANGLE_PHASES)))
        ax.set_xticklabels([p[0].replace("_", " ") for p in ANGLE_PHASES], rotation=35, ha="right")
        ax.set_yticks(range(len(LETTERS))); ax.set_yticklabels(LETTERS)
        ax.set_title("Angle coverage: images per letter and angle (recorded sessions)")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("images")
        fig.savefig(P.reports / "personal_angle_coverage.png", dpi=150, bbox_inches="tight"); plt.close(fig)

        tile, label_w, n_ph = 96, 56, len(ANGLE_PHASES)
        rng = np.random.RandomState(1)
        canvas = np.full((len(LETTERS) * tile, label_w + n_ph * tile, 3), 252, np.uint8)
        for r, letter in enumerate(LETTERS):
            cv2.putText(canvas, letter, (12, r * tile + tile // 2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (11, 11, 11), 2)
            for c in range(n_ph):
                sub = new[(new["letter"] == letter) & (new["phase_id"].astype(int) == c)]
                if len(sub) == 0:
                    continue
                img = cv2.imread(str(P.crops / sub.iloc[rng.randint(len(sub))]["file"]))
                if img is not None:
                    x0 = label_w + c * tile
                    canvas[r * tile:(r + 1) * tile, x0:x0 + tile] = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(P.reports / "personal_poster.png"), canvas)
    print("Saved charts to", P.reports)


# ==================================================================
#  Stage: finetune on personal data (honest validation)
# ==================================================================
def finetune_personal(final=False):
    """final=False: honest run (newest session / angles held out -> a number you can quote).
    final=True : retrain on EVERY recorded image for the model you actually use; keeps the fusion alpha from your
                 last honest run; the accuracy it prints is no longer honest."""
    image_model, landmark_model = _load_pair(IMG_BEST, LM_BEST)
    df = load_meta()
    if len(df) == 0:
        sys.exit("No personal dataset. Run --collect-personal (or --import-npz) first.")
    ensure_dirs()
    P_img, P_lm, P_y = load_personal_arrays(df)
    is_val, strength, how = split_personal(df)
    p_tr, p_va = np.where(~is_val)[0], np.where(is_val)[0]
    if len(p_va) == 0 or len(p_tr) == 0:
        sys.exit("Not enough personal data to split into train/validation.")
    if final:
        p_tr = np.arange(len(df))
        print("FINAL MODE: training on ALL recorded images. The held-out images are now part of the training data, "
              "so the accuracy printed at the end is NOT honest - quote the number from your previous run.")
    print(f"Personal samples: {len(P_y):,} | train {len(p_tr):,} | validation {len(p_va):,}")
    print(f"Split [{strength}]: {how}")
    no_val = [LETTERS[c] for c in np.unique(P_y) if not (P_y[p_va] == c).any()]
    if no_val:
        print("WARNING - no validation images for letters:", " ".join(no_val))

    kag = load_kaggle_cache(required=False)
    rng = np.random.RandomState(SEED)
    if kag is not None:
        X_img, X_lm, y_all = kag
        k_train, k_val = kaggle_split(y_all)
        replay = np.concatenate([rng.choice(k_train[y_all[k_train] == c], size=min(80, int(np.sum(y_all[k_train] == c))),
                                            replace=False) for c in range(NUM_CLASSES)])
    else:
        X_img = X_lm = y_all = k_val = None
        replay = np.array([], np.int64)
        print("WARNING - Kaggle cache not found: fine-tuning on personal data only (risk of forgetting other "
              "conditions). Run --download and --preprocess for the best result.")

    # personal x3 (weight 2) + Kaggle replay slice (weight 1); hard letters boosted
    source = np.concatenate([np.zeros(len(p_tr) * 3, np.int64), np.ones(len(replay), np.int64)])
    index = np.concatenate([np.repeat(p_tr, 3), replay]).astype(np.int64)
    ft_y = np.concatenate([np.repeat(P_y[p_tr], 3), y_all[replay] if kag is not None else np.array([], np.int64)])
    ft_w = np.concatenate([np.full(len(p_tr) * 3, 2.0, np.float32), np.full(len(replay), 1.0, np.float32)])
    for l in CFG["HARD_LETTERS"]:
        ft_w[ft_y == LETTER_TO_IDX[l]] *= CFG["HARD_BOOST"]
    perm = rng.permutation(len(ft_y))
    source, index, ft_y, ft_w = source[perm], index[perm], ft_y[perm], ft_w[perm]

    train_ds = make_mixed_ds(P_img, X_img, source, index, ft_y, ft_w, training=True)
    val_ds = make_mixed_ds(P_img, X_img, np.zeros(len(p_va), np.int64), p_va, P_y[p_va],
                           np.ones(len(p_va), np.float32), training=False)

    pers_img_ckpt, pers_lm_ckpt = ckpt(IMG_PERS), ckpt(LM_PERS)
    image_model.compile(optimizer=optimizers.Adam(1e-5), loss=categorical_focal_loss(),
                        metrics=["accuracy"], weighted_metrics=[])
    image_model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=make_callbacks(pers_img_ckpt, patience=4))

    if kag is not None:
        ft_lm = np.where((source == 0)[:, None], P_lm[np.where(source == 0, index, 0)], X_lm[np.where(source == 1, index, 0)])
    else:
        ft_lm = P_lm[index]
    Xl, yl, wl = expand_landmark_dataset(ft_lm.astype(np.float32), ft_y, ft_w, copies=2)
    landmark_model.compile(optimizer=optimizers.Adam(3e-4), loss=categorical_focal_loss(),
                           metrics=["accuracy"], weighted_metrics=[])
    landmark_model.fit(Xl, np.eye(NUM_CLASSES, dtype=np.float32)[yl], sample_weight=wl,
                       validation_data=(P_lm[p_va], np.eye(NUM_CLASSES, dtype=np.float32)[P_y[p_va]]),
                       epochs=30, batch_size=128, callbacks=make_callbacks(pers_lm_ckpt, patience=8), verbose=2)

    image_model, landmark_model = _load_pair(IMG_PERS, LM_PERS)
    pv_img = predict_chunked(image_model, P_img[p_va])
    pv_lm = landmark_model.predict(P_lm[p_va], batch_size=256, verbose=0)
    if final:
        alpha = float(CFG["FUSION_ALPHA"])           # keep the alpha tuned on honest data - do not re-tune on seen data
        acc = float(accuracy_score(P_y[p_va], np.argmax(alpha * pv_img + (1 - alpha) * pv_lm, 1)))
    else:
        alpha, acc = tune_alpha(pv_img, pv_lm, P_y[p_va])
        CFG["FUSION_ALPHA"] = alpha
    pred = np.argmax(alpha * pv_img + (1 - alpha) * pv_lm, axis=1)

    cm = confusion_matrix(P_y[p_va], pred, labels=np.arange(NUM_CLASSES))
    per_cls = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    present = [c for c in range(NUM_CLASSES) if cm[c].sum() > 0]
    worst = sorted(present, key=lambda c: per_cls[c])[:5]
    if final:
        lines = ["FINAL MODEL - trained on ALL recorded images.",
                 "The accuracy below is on images the model has now SEEN: do NOT quote it.",
                 f"Fusion alpha kept from your last honest run: {alpha:.2f}",
                 f"(seen-data) fused accuracy: {acc:.4f}"]
    else:
        lines = [f"Split strength: {strength}", how,
                 f"Image-only accuracy on held-out personal data   : {accuracy_score(P_y[p_va], pv_img.argmax(1)):.4f}",
                 f"Landmark-only accuracy on held-out personal data: {accuracy_score(P_y[p_va], pv_lm.argmax(1)):.4f}",
                 f"FUSED accuracy on held-out personal data (alpha={alpha:.2f}): {acc:.4f}",
                 "Weakest letters: " + ", ".join(f"{LETTERS[c]} {per_cls[c]:.0%}" for c in worst)]
    if kag is not None:                                      # forgetting check on the original Kaggle validation set
        kp_img = predict_chunked(image_model, X_img[k_val])
        kp_lm = landmark_model.predict(X_lm[k_val], batch_size=256, verbose=0)
        lines.append(f"Kaggle validation accuracy after fine-tune (forgetting check): "
                     f"{accuracy_score(y_all[k_val], np.argmax(alpha * kp_img + (1 - alpha) * kp_lm, 1)):.4f}")
    if strength != "STRICT" and not final:
        lines.append("TIP: this number is optimistic. Record a fresh session in different light/background "
                     "(--collect-personal --session s02) and run --finetune again for a number you can defend.")
    print("\n" + "\n".join(lines))
    suffix = "_final" if final else ""
    plot_confusion(cm, f"Personal data ({'FINAL, seen data' if final else 'held-out, ' + strength})",
                   P.reports / f"confusion_personal{suffix}.png")
    (P.reports / f"evaluation_personal{suffix}.txt").write_text("\n".join(lines))
    save_config()
    print("\nSaved reports to", P.reports)
    if not final:
        print("Happy with this honest number? Then train the model you will actually USE on every image:\n"
              "  python asl_local.py --finetune --final")
    print("Next: python asl_local.py --realtime")


# ==================================================================
#  Stage: export
# ==================================================================
def export_models():
    load_saved_config()
    P.export.mkdir(parents=True, exist_ok=True)
    for name in (IMG_BEST, LM_BEST, IMG_PERS, LM_PERS):
        if ckpt(name).exists():
            shutil.copy(ckpt(name), P.export / ckpt(name).name)
    (P.export / "label_encoder.json").write_text(json.dumps({"classes": LETTERS}, indent=2))
    (P.export / "config.json").write_text(json.dumps(CFG, indent=2))
    hist = P.models / "training_history.json"
    if hist.exists():
        shutil.copy(hist, P.export / hist.name)
    print("Exported files:")
    for f in sorted(P.export.iterdir()):
        print("  ", f.name, f"({f.stat().st_size / 1e6:.1f} MB)")
    shutil.make_archive(str(P.base / "asl_model_export"), "zip", P.export)
    print(f"\nZip ready at {P.base / 'asl_model_export.zip'}")


# ==================================================================
#  Stage: realtime recognition
# ==================================================================
def load_models_for_inference():
    load_saved_config()
    pick = lambda pers, base: pers if ckpt(pers).exists() else base
    img_name, lm_name = pick(IMG_PERS, IMG_BEST), pick(LM_PERS, LM_BEST)
    image_model, landmark_model = _load_pair(img_name, lm_name)
    print("Loaded image model   :", ckpt(img_name))
    print("Loaded landmark model:", ckpt(lm_name))
    return image_model, landmark_model


def wrap_sentence(text, width=34):
    """Split text into lines of at most `width` characters, breaking at spaces (a longer single word is cut)."""
    lines, cur = [], ""
    for word in re.split(r"( )", text):                   # keep the spaces as separate tokens
        if len(cur) + len(word) <= width:
            cur += word
            continue
        if word == " ":                                   # a space that would overflow ends the line
            lines.append(cur)
            cur = ""
            continue
        if cur.strip():
            lines.append(cur)
            cur = ""
        while len(word) > width:                          # a single word longer than a whole line
            lines.append(word[:width])
            word = word[width:]
        cur = word
    lines.append(cur)
    return lines


def run_realtime_recognition(image_model, landmark_model, max_minutes=10, cam_index=0):
    conf_gate = float(CFG["CONF_THRESHOLD"])
    stable_secs = float(CFG["STABLE_SECONDS"])
    alpha = float(CFG["FUSION_ALPHA"])
    print(f"Fusion alpha = {alpha:.2f} | confidence gate = {conf_gate:.2f}")
    print("Keys (click the video window first):  A = lock the letter now | SPACE = new word | "
          "D = delete last | C = clear all | Q = quit")

    history = deque()
    sentence, last_accepted, ready_for_repeat, release_start = "", None, True, None
    flash_text, flash_until = "", 0.0                      # short on-screen confirmation of a key press
    RELEASE_SECS = 0.4
    fps_ema, prev_t, t_start, misses = 0.0, time.time(), time.time(), 0
    window = "ASL Real-Time Recognition"
    cam = LocalWebcam(cam_index)
    landmarker = make_landmarker(video=True)
    clock = VideoClock()
    running = True
    try:
        while running and (time.time() - t_start) < max_minutes * 60:
            frame = cam.read()
            if frame is None:
                misses += 1
                if misses > 60:
                    print("Camera is not delivering frames - stopping.")
                    break
                read_keys(10)
                continue
            misses = 0
            now = time.time()
            crop, vec, bbox, hand_lms = extract_from_image(frame, landmarker, clock())
            overlay = frame.copy()

            pred_letter, pred_conf = None, 0.0
            if crop is not None:
                n = NUM_CLASSES
                p_img = image_model(crop[None].astype(np.float32), training=False).numpy()[0] if alpha > 0 else np.zeros(n)
                p_lm = landmark_model(vec[None], training=False).numpy()[0] if alpha < 1 else np.zeros(n)
                p = alpha * p_img + (1 - alpha) * p_lm
                pred_idx = int(np.argmax(p))
                pred_letter, pred_conf = LETTERS[pred_idx], float(p[pred_idx])
                history.append((now, pred_idx, pred_conf))
                x1, y1, x2, y2 = bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_hand_landmarks(overlay, hand_lms)

            manual_lock = False
            for k in read_keys():
                if k == " ":                                    # SPACE: end the word (no leading / double spaces)
                    if sentence and not sentence.endswith(" "):
                        sentence += " "
                        flash_text, flash_until = "Space added", now + 1.0
                    else:
                        flash_text, flash_until = "Type a letter before adding a space", now + 1.0
                    history.clear()                             # a half-built hold must not add a letter right after
                elif k in ("d", "backspace"):                   # D (or BACKSPACE): delete the last character
                    if sentence:
                        flash_text = "Deleted space" if sentence[-1] == " " else f"Deleted {sentence[-1]}"
                        flash_until = now + 1.0
                        sentence = sentence[:-1]
                    else:
                        flash_text, flash_until = "Nothing to delete", now + 1.0
                    history.clear()
                elif k in ("a", "s"):                           # A (or S): lock in the current guess right now
                    manual_lock = True
                elif k == "c":
                    sentence = ""; last_accepted = None; ready_for_repeat = True
                    flash_text, flash_until = "Cleared", now + 1.0
                    history.clear()
                elif k == "q":
                    running = False
            if ui_closed(window) and fps_ema > 0:
                running = False

            if manual_lock:                                  # accepts the current guess regardless of confidence
                if pred_letter is not None:
                    sentence += pred_letter
                    last_accepted, ready_for_repeat, release_start = pred_letter, False, None
                    history.clear()
                    flash_text, flash_until = f"Locked {pred_letter}", now + 1.0
                else:
                    flash_text, flash_until = "No hand to lock - show your sign", now + 1.0

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

            stable_letter, stable_conf, progress = None, 0.0, 0.0
            if len(history) >= 3:
                span = history[-1][0] - history[0][0]
                top_idx, top_n = Counter(h[1] for h in history).most_common(1)[0]
                stable_frac = top_n / len(history)
                stable_conf = float(np.mean([h[2] for h in history if h[1] == top_idx]))
                stable_letter = LETTERS[top_idx]
                progress = min(1.0, span / stable_secs) * stable_frac
                if (span >= stable_secs and stable_frac >= 0.75 and stable_conf >= conf_gate
                        and (stable_letter != last_accepted or ready_for_repeat)):
                    sentence += stable_letter
                    last_accepted, ready_for_repeat, release_start = stable_letter, False, None
                    history.clear()

            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt

            cv2.rectangle(overlay, (0, 0), (640, 96), (0, 0, 0), -1)
            if crop is None:
                head, color = "No hand detected", (0, 0, 255)
            elif pred_conf < conf_gate:
                head, color = f"Uncertain ({pred_letter} {pred_conf * 100:.0f}%) - press A to lock it in", (0, 165, 255)
            else:
                head, color = f"{pred_letter}  {pred_conf * 100:.1f}%", (0, 255, 0)
            cv2.putText(overlay, head, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(overlay, f"FPS: {fps_ema:.1f}", (545, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if stable_letter and stable_conf >= conf_gate:
                cv2.rectangle(overlay, (10, 44), (10 + int(300 * progress), 56), (0, 255, 0), -1)
                cv2.rectangle(overlay, (10, 44), (310, 56), (255, 255, 255), 1)
                cv2.putText(overlay, f"hold {stable_letter}", (320, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            cv2.putText(overlay, f"Word: {sentence.split(' ')[-1]}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if flash_text and now < flash_until:                 # confirmation of the last key press
                cv2.putText(overlay, flash_text, (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            h, w = overlay.shape[:2]
            cursor = "_" if int(now * 2) % 2 == 0 else " "     # blinking cursor makes a trailing space visible
            text = sentence + cursor
            lines = wrap_sentence(text)[-2:]                   # newest two lines, broken at spaces
            cv2.rectangle(overlay, (0, h - 92), (w, h), (0, 0, 0), -1)
            cv2.putText(overlay, "A=lock letter   SPACE=space   D=delete   C=clear   Q=quit", (10, h - 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            for j, line in enumerate(reversed(lines)):         # newest line at the bottom
                cv2.putText(overlay, line, (10, h - 12 - 30 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            ui_show(window, overlay)
    finally:
        landmarker.close()
        cam.release()
        cv2.destroyAllWindows()

    print("\nSession ended.")
    print("Final text:", repr(sentence))
    if sentence.strip():
        with open(P.transcripts, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{sentence}\n")
        print("Saved to", P.transcripts)
    return sentence


# ==================================================================
#  Status + CLI
# ==================================================================
def print_status():
    print(f"Project folder: {P.base}\n")
    have_kag = P.dataset_root.is_dir() or P.dataset_zip.exists()
    cached = [l for l in LETTERS if (P.kaggle_cache / f"{l}.npz").exists()]
    base_ok = ckpt(IMG_BEST).exists() and ckpt(LM_BEST).exists()
    pers_ok = ckpt(IMG_PERS).exists() and ckpt(LM_PERS).exists()
    df = load_meta()
    mark = lambda ok: "[x]" if ok else "[ ]"
    print(f"{mark(have_kag)} Kaggle dataset downloaded")
    print(f"{mark(len(cached) == 26)} Kaggle preprocessed  ({len(cached)}/26 letters cached)")
    print(f"{mark(base_ok)} Base models (image + landmark)")
    if len(df):
        sess = list(dict.fromkeys(df["session"]))
        _, strength, _ = split_personal(df)
        print(f"[x] Personal dataset: {len(df):,} images, sessions {sess}  (accuracy honesty: {strength})")
    else:
        print("[ ] Personal dataset (none yet)")
    print(f"{mark(pers_ok)} Personalised models")
    if not base_ok and len(cached) < 26 and not have_kag:
        nxt = "--download   (or reuse Colab models:  --import-models asl_model_export.zip)"
    elif not base_ok and len(cached) < 26:
        nxt = "--preprocess"
    elif not base_ok:
        nxt = "--train   (or reuse Colab models:  --import-models asl_model_export.zip)"
    elif not len(df):
        nxt = "--collect-personal --session s01"
    elif not pers_ok:
        nxt = "--finetune"
    else:
        nxt = "--realtime      (or record another session for a stricter, higher-quality model)"
    print(f"\nSuggested next step:  python asl_local.py {nxt}")


def main():
    ap = argparse.ArgumentParser(description="ASL alphabet recognition - hybrid image + landmark model (local).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="COMMANDS" + __doc__.split("COMMANDS", 1)[1] if "COMMANDS" in (__doc__ or "") else None)
    a = ap.add_argument
    a("--status", action="store_true", help="show what exists and what to run next")
    a("--download", action="store_true", help="download + unzip the Kaggle ASL alphabet dataset")
    a("--preprocess", action="store_true", help="MediaPipe crops + landmarks (resumable)")
    a("--train", action="store_true", help="train the image + landmark models")
    a("--evaluate", action="store_true", help="metrics, confusion matrix, confusion-driven retrain")
    a("--import-models", metavar="ZIP_OR_FOLDER", help="reuse models trained in Colab (asl_model_export.zip)")
    a("--import-npz", metavar="FILE", help="reuse your OLD Colab personal_asl.npz")
    a("--collect-personal", action="store_true", help="record your hand with the webcam (guided angles)")
    a("--personal-report", action="store_true", help="counts, angle coverage and poster of your personal data")
    a("--finetune", action="store_true", help="fine-tune on your personal data with an honest validation split")
    a("--final", action="store_true", help="with --finetune: train on ALL recorded images (do this after you have "
                                          "checked the honest number)")
    a("--export", action="store_true", help="copy models + config into a portable zip")
    a("--realtime", action="store_true", help="live webcam recognition")
    a("--project-dir", default=None, help="where to keep everything (default: ./asl_project next to this script)")
    a("--session", default="s01", help="label for a recording run, e.g. s01, s02-night (new light/background = new session)")
    a("--letters", default=None, help="restrict --collect-personal to these letters, e.g. MNST")
    a("--per-letter", type=int, default=None, help="images per letter per session (default 200)")
    a("--max-per-class", type=int, default=None, help="Kaggle images kept per letter (default 400)")
    a("--camera", type=int, default=0, help="webcam index (try 1 if 0 is not your camera)")
    a("--no-frames", action="store_true", help="do not save full camera frames while recording")
    a("--max-minutes", type=float, default=10, help="max length of a --realtime session")
    a("--conf", type=float, default=None, help="override the confidence gate for --realtime (default 0.95)")
    a("--alpha", type=float, default=None, help="override the fusion weight for --realtime (0=landmarks only, 1=image only)")
    args = ap.parse_args()

    global P
    if args.project_dir:
        P = make_paths(args.project_dir)
    if args.no_frames:
        CFG["SAVE_FULL_FRAME"] = False
    if args.max_per_class:
        CFG["MAX_PER_CLASS"] = args.max_per_class
    load_saved_config()
    if args.max_per_class:                    # CLI wins over a saved config
        CFG["MAX_PER_CLASS"] = args.max_per_class

    todo = [args.status, args.download, args.preprocess, args.train, args.evaluate, args.import_models,
            args.import_npz, args.collect_personal, args.personal_report, args.finetune, args.export, args.realtime]
    if not any(todo):
        ap.print_help()
        print()
        print_status()
        return

    if args.status:
        print_status()
    if args.download:
        download_dataset()
    if args.preprocess:
        preprocess_dataset(args.max_per_class)
    if args.train or args.evaluate or args.finetune:
        setup_gpu()
    if args.train:
        train_models()
    if args.evaluate:
        evaluate_models()
    if args.import_models:
        import_models(args.import_models)
    if args.import_npz:
        import_legacy_npz(args.import_npz)
    if args.collect_personal:
        letters = [c for c in args.letters.upper() if c in LETTERS] if args.letters else LETTERS
        collect_personal_dataset(letters, args.per_letter, args.session, args.camera)
    if args.personal_report:
        personal_report()
    if args.finetune:
        finetune_personal(final=args.final)
    if args.export:
        export_models()
    if args.realtime:
        image_model, landmark_model = load_models_for_inference()
        if args.conf is not None:
            CFG["CONF_THRESHOLD"] = args.conf
        if args.alpha is not None:
            CFG["FUSION_ALPHA"] = args.alpha
        run_realtime_recognition(image_model, landmark_model, args.max_minutes, args.camera)


if __name__ == "__main__":
    main()