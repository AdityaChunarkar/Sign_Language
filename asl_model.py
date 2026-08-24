#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-Time ASL Alphabet Recognition — Hybrid Image + Landmark Model
====================================================================
Local / VS Code version, converted from a Google Colab notebook export.

WHAT CHANGED FROM THE COLAB VERSION
------------------------------------
* `!pip install ...` shell-magic lines (invalid outside a notebook cell) were
  removed. Install dependencies yourself first:
      pip install -r requirements.txt
* `google.colab.files.upload()` / `google.colab.output.eval_js()` and the
  browser<->Python JS webcam bridge (Colab-only, does not exist locally) were
  replaced with plain OpenCV `cv2.VideoCapture` + `cv2.imshow`.
* Hardcoded `/content/...` paths (Colab's ephemeral VM) were replaced with a
  local `asl_project/` folder next to this script.
* The single giant "run every cell top to bottom" notebook was split into
  independent stages you select on the command line, so you don't have to
  redownload/retrain everything every time you run the script.
* Fixed a stray `n#` syntax bug left over from the notebook export.

USAGE
-----
    pip install -r requirements.txt

    python asl_model.py --download            # fetch the Kaggle ASL alphabet dataset
    python asl_model.py --preprocess          # MediaPipe crop + landmark extraction (cached)
    python asl_model.py --train                # train image + landmark models
    python asl_model.py --evaluate             # metrics, confusion matrix, confusion-driven retrain
    python asl_model.py --collect-personal      # collect your own hand via webcam
    python asl_model.py --finetune              # fine-tune on your personal data
    python asl_model.py --export                # export models + config to a portable folder
    python asl_model.py --realtime              # live webcam recognition + word building
    python asl_model.py --all                   # run every stage above, in order

Stages can be combined, e.g.: python asl_model.py --train --evaluate

NOTES
-----
* GPU is used automatically if TensorFlow detects one; otherwise it falls
  back to CPU (training will just be slower).
* `--realtime` and `--collect-personal` open a real OS window (cv2.imshow) and
  need a display + a working webcam — they won't work on a headless server.
* Kaggle dataset download needs `~/.kaggle/kaggle.json` (Kaggle -> Account ->
  Create New API Token), OR you can manually download the zip yourself (see
  the message `--download` prints).
"""

import argparse
import json
import math
import os
import pickle
import random
import shutil
import sys
import time
import zipfile
from collections import Counter, deque

# ----------------------------------------------------------------------
# Dependency check — fail fast with a clear message instead of a traceback
# ----------------------------------------------------------------------
try:
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm.auto import tqdm

    import tensorflow as tf
    import keras
    from tensorflow.keras import layers, models, callbacks, optimizers

    import mediapipe as mp
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        accuracy_score,
    )
except ImportError as e:
    sys.exit(
        f"Missing dependency: {e}\n\n"
        "Install everything first with:\n"
        "    pip install -r requirements.txt\n"
    )

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
    "MAX_PER_CLASS": 400,            # images sampled per letter from the Kaggle set (raise for more accuracy)
    "BATCH_SIZE": 32,
    "VAL_SPLIT": 0.15,
    "CONF_THRESHOLD": 0.95,          # real-time acceptance confidence
    "STABLE_SECONDS": 1.0,           # letter must stay stable this long
    "HAND_MARGIN": 0.30,             # crop margin around the detected hand
    "HARD_LETTERS": list("ASTEMNRUVWPKGHXY"),  # visually similar letters
    "HARD_BOOST": 1.5,               # class-weight multiplier for hard letters
    "HARD_OVERSAMPLE": 0.5,          # +50% extra training samples for hard letters
    "FUSION_ALPHA": 0.6,             # image-model weight in late fusion (auto-tuned later)
    "FOCAL_GAMMA": 2.0,              # focal loss focusing parameter
    "PERSONAL_PER_LETTER": 150,      # webcam images to collect per letter for fine-tuning
}
LETTERS = [chr(ord('A') + i) for i in range(26)]
NUM_CLASSES = len(LETTERS)
LETTER_TO_IDX = {l: i for i, l in enumerate(LETTERS)}

# ------------------------------------------------------------------
# Local paths (replaces Colab's /content/...)
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "asl_project")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_CACHE = os.path.join(BASE_DIR, "processed_asl.npz")
PERSONAL_CACHE = os.path.join(BASE_DIR, "personal_asl.npz")
TUNED_CFG_PATH = os.path.join(MODELS_DIR, "tuned_config.json")
EXPORT_DIR = os.path.join(BASE_DIR, "asl_model_export")
DATASET_ZIP = os.path.join(BASE_DIR, "asl-alphabet.zip")
DATASET_ROOT = os.path.join(BASE_DIR, "asl_alphabet_train", "asl_alphabet_train")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

IMG_CKPT = os.path.join(MODELS_DIR, "image_model_best.keras")
LM_CKPT = os.path.join(MODELS_DIR, "landmark_model_best.keras")
IMG_CKPT_PERSONAL = os.path.join(MODELS_DIR, "image_model_personal.keras")
LM_CKPT_PERSONAL = os.path.join(MODELS_DIR, "landmark_model_personal.keras")


# ------------------------------------------------------------------
# GPU + Mixed Precision (speeds up training with no accuracy loss)
# ------------------------------------------------------------------
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    print("GPUs available:", gpus)
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("Mixed precision ENABLED (mixed_float16).")
        except Exception as e:
            print("Mixed precision not enabled:", e)
    else:
        print("No GPU detected — training will run on CPU and be slower.")


# ------------------------------------------------------------------
# Dataset download (replaces the Colab kaggle.json upload + !kaggle + !unzip)
# ------------------------------------------------------------------
def download_dataset():
    if os.path.isdir(DATASET_ROOT):
        print("Dataset already present at", DATASET_ROOT)
        return
    if not os.path.exists(DATASET_ZIP):
        kaggle_cfg = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
        if not os.path.exists(kaggle_cfg):
            sys.exit(
                "ASL alphabet dataset not found.\n\n"
                "Option A (automatic): put your Kaggle API token at\n"
                f"  {kaggle_cfg}\n"
                "(Kaggle.com -> Account -> API -> Create New API Token), then re-run:\n"
                "  python asl_model.py --download\n\n"
                "Option B (manual): download the zip yourself from\n"
                "  https://www.kaggle.com/datasets/grassknoted/asl-alphabet\n"
                f"and place it at:\n  {DATASET_ZIP}\n"
                "then re-run: python asl_model.py --download"
            )
        print("Downloading dataset via the Kaggle CLI...")
        import subprocess
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "grassknoted/asl-alphabet", "-p", BASE_DIR],
            check=True,
        )
    print("Unzipping dataset (this can take a couple of minutes)...")
    with zipfile.ZipFile(DATASET_ZIP) as zf:
        zf.extractall(BASE_DIR)
    print("Dataset ready:", os.path.isdir(DATASET_ROOT))
    if os.path.isdir(DATASET_ROOT):
        print("Classes found:", sorted(os.listdir(DATASET_ROOT))[:10], "...")


# ------------------------------------------------------------------
# MediaPipe hand detection, cropping & landmark extraction
# ------------------------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def normalize_landmarks(landmarks_21x3, handedness_label):
    """Wrist-centered, scale-normalized, mirror-corrected 63-dim landmark vector."""
    lm = landmarks_21x3.copy().astype(np.float32)
    lm -= lm[0]                                       # wrist to origin
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-6
    lm /= scale
    if handedness_label == "Left":
        lm[:, 0] *= -1.0
    return lm.flatten()                               # shape (63,)


def crop_hand_region(image_bgr, hand_landmarks, margin=None):
    """Crop a square-ish region around the detected hand with a safety margin."""
    if margin is None:
        margin = CFG["HAND_MARGIN"]
    h, w = image_bgr.shape[:2]
    xs = [p.x for p in hand_landmarks.landmark]
    ys = [p.y for p in hand_landmarks.landmark]
    x1, x2 = min(xs) * w, max(xs) * w
    y1, y2 = min(ys) * h, max(ys) * h
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1 = int(max(0, x1 - mx)); x2 = int(min(w, x2 + mx))
    y1 = int(max(0, y1 - my)); y2 = int(min(h, y2 + my))
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, None
    crop = cv2.resize(image_bgr[y1:y2, x1:x2], (CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
                       interpolation=cv2.INTER_AREA)
    return crop, (x1, y1, x2, y2)


def extract_from_image(image_bgr, hands_detector):
    """Run MediaPipe on one BGR image -> (crop_rgb_uint8, landmark_vec_63, bbox, hand) or Nones."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = hands_detector.process(rgb)
    if not res.multi_hand_landmarks:
        return None, None, None, None
    hand = res.multi_hand_landmarks[0]
    handed = res.multi_handedness[0].classification[0].label if res.multi_handedness else "Right"
    crop, bbox = crop_hand_region(image_bgr, hand)
    if crop is None:
        return None, None, None, None
    lm = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
    vec = normalize_landmarks(lm, handed)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), vec, bbox, hand


# ------------------------------------------------------------------
# Preprocessing stage — builds DATA_CACHE from the raw Kaggle images
# ------------------------------------------------------------------
def preprocess_dataset(show_sanity_plot=True):
    if not os.path.isdir(DATASET_ROOT):
        sys.exit("Dataset not found. Run: python asl_model.py --download")

    if os.path.exists(DATA_CACHE):
        print("Cache already exists at", DATA_CACHE, "- delete it to reprocess.")
        return

    imgs, lms, labels = [], [], []
    import glob
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.5, model_complexity=1) as hands:
        for letter in tqdm(LETTERS, desc="Letters"):
            folder = os.path.join(DATASET_ROOT, letter)
            files_list = sorted(glob.glob(os.path.join(folder, "*.jpg")))
            random.Random(SEED).shuffle(files_list)
            kept = 0
            for fp in files_list:
                if kept >= CFG["MAX_PER_CLASS"]:
                    break
                img = cv2.imread(fp)
                if img is None:
                    continue
                crop, vec, _, _ = extract_from_image(img, hands)
                if crop is None:
                    continue  # no hand detected -> skip (keeps image/landmark data paired)
                imgs.append(crop.astype(np.uint8))
                lms.append(vec)
                labels.append(LETTER_TO_IDX[letter])
                kept += 1

    X_img = np.stack(imgs)                    # (N, 224, 224, 3) uint8 RGB
    X_lm = np.stack(lms).astype(np.float32)   # (N, 63)
    y_all = np.array(labels, dtype=np.int64)  # (N,)
    np.savez_compressed(DATA_CACHE, X_img=X_img, X_lm=X_lm, y=y_all)
    print("Cached to", DATA_CACHE)
    print("Images:", X_img.shape, "| Landmarks:", X_lm.shape, "| Labels:", y_all.shape)
    print("Per-class counts:", dict(sorted(Counter([LETTERS[i] for i in y_all]).items())))

    if show_sanity_plot:
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for ax, letter in zip(axes.flat, CFG["HARD_LETTERS"]):
            idxs = np.where(y_all == LETTER_TO_IDX[letter])[0]
            if len(idxs):
                ax.imshow(X_img[idxs[0]]); ax.set_title(letter)
            ax.axis("off")
        plt.suptitle("Cropped hand images — the 16 'hard' letters")
        plt.show()


# ------------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="asl_hybrid")
class PixelScaleGaussianNoise(layers.Layer):
    """Additive Gaussian noise for inputs on the raw 0-255 pixel scale."""
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0,
                                      stddev=self.stddev, dtype=inputs.dtype)
            return inputs + noise
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config


def build_image_augmentation():
    """Keras preprocessing pipeline — only runs when training=True.
    No horizontal flips: mirroring changes the meaning of many ASL signs."""
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
    x, ycoord = lm[:, 0].copy(), lm[:, 1].copy()
    lm[:, 0] = c * x - s * ycoord
    lm[:, 1] = s * x + c * ycoord
    lm *= rng.uniform(0.92, 1.08)
    lm += rng.normal(0.0, 0.012, lm.shape)
    return lm.flatten().astype(np.float32)


def expand_landmark_dataset(X, y, w, copies=3, seed=SEED):
    """Offline oversampling of landmark data with augmentation."""
    rng = np.random.RandomState(seed)
    Xs, ys, ws = [X], [y], [w]
    for _ in range(copies):
        Xs.append(np.stack([augment_landmark_vector(v, rng) for v in X]))
        ys.append(y); ws.append(w)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)


# ------------------------------------------------------------------
# Loss
# ------------------------------------------------------------------
def categorical_focal_loss(gamma=None):
    if gamma is None:
        gamma = CFG["FOCAL_GAMMA"]

    def loss_fn(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        focal = tf.pow(1.0 - y_pred, gamma) * ce
        return tf.reduce_sum(focal, axis=-1)
    return loss_fn


CUSTOM_OBJECTS = {
    "loss_fn": categorical_focal_loss(),
    "PixelScaleGaussianNoise": PixelScaleGaussianNoise,
}


# ------------------------------------------------------------------
# Model architectures
# ------------------------------------------------------------------
def build_image_model(num_classes=NUM_CLASSES):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(CFG["IMG_SIZE"], CFG["IMG_SIZE"], 3))
    base.trainable = False

    inp = layers.Input((CFG["IMG_SIZE"], CFG["IMG_SIZE"], 3), name="hand_image")
    x = build_image_augmentation()(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="probs")(x)
    model = models.Model(inp, out, name="image_branch")
    return model, base


def build_landmark_model(num_classes=NUM_CLASSES):
    inp = layers.Input((63,), name="landmarks")
    x = layers.Dense(256)(inp); x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(256)(x);  x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128)(x);  x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(num_classes, activation="softmax", dtype="float32", name="probs")(x)
    return models.Model(inp, out, name="landmark_branch")


# ------------------------------------------------------------------
# Class weights / oversampling / tf.data pipelines
# ------------------------------------------------------------------
def compute_sample_weights(y_indices, extra_boost=None):
    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_indices)
    cw = dict(enumerate(cw))
    for l in CFG["HARD_LETTERS"]:
        cw[LETTER_TO_IDX[l]] *= CFG["HARD_BOOST"]
    if extra_boost:
        for cls_idx, mult in extra_boost.items():
            cw[cls_idx] *= mult
    return np.array([cw[c] for c in y_indices], dtype=np.float32), cw


def oversample_indices(idx_array, y_indices, frac=None, seed=SEED):
    """Duplicate a fraction of hard-letter samples in the training index list."""
    if frac is None:
        frac = CFG["HARD_OVERSAMPLE"]
    rng = np.random.RandomState(seed)
    hard_cls = {LETTER_TO_IDX[l] for l in CFG["HARD_LETTERS"]}
    extra = []
    for cls in hard_cls:
        cls_pos = idx_array[y_indices == cls]
        n_extra = int(len(cls_pos) * frac)
        if n_extra > 0:
            extra.append(rng.choice(cls_pos, n_extra, replace=True))
    out = np.concatenate([idx_array] + extra) if extra else idx_array.copy()
    rng.shuffle(out)
    return out


def make_image_ds(X_img, y_all, sample_w_lookup, indices, training, batch_size=None):
    """tf.data pipeline. Indexes into big numpy arrays via tf.numpy_function
    to avoid embedding a multi-GB constant in the graph."""
    if batch_size is None:
        batch_size = CFG["BATCH_SIZE"]

    def _fetch(i):
        i = int(i)
        return (X_img[i].astype(np.float32),
                np.eye(NUM_CLASSES, dtype=np.float32)[y_all[i]],
                np.float32(sample_w_lookup[i]))

    ds = tf.data.Dataset.from_tensor_slices(indices.astype(np.int64))
    if training:
        ds = ds.shuffle(len(indices), seed=SEED, reshuffle_each_iteration=True)

    def _map(i):
        img, lab, w = tf.numpy_function(_fetch, [i], (tf.float32, tf.float32, tf.float32))
        img.set_shape((CFG["IMG_SIZE"], CFG["IMG_SIZE"], 3))
        lab.set_shape((NUM_CLASSES,)); w.set_shape(())
        return img, lab, w

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_callbacks(ckpt_path, patience=6):
    return [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                   save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                 restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                     patience=3, min_lr=1e-7, verbose=1),
    ]


def _load_data_cache():
    if not os.path.exists(DATA_CACHE):
        sys.exit("processed_asl.npz not found. Run: python asl_model.py --preprocess")
    c = np.load(DATA_CACHE)
    return c["X_img"], c["X_lm"], c["y"]


def _train_val_split(y_all):
    train_idx, val_idx = train_test_split(
        np.arange(len(y_all)), test_size=CFG["VAL_SPLIT"],
        stratify=y_all, random_state=SEED)
    return train_idx, val_idx


# ------------------------------------------------------------------
# Stage: train
# ------------------------------------------------------------------
def train_models():
    X_img, X_lm, y_all = _load_data_cache()
    train_idx, val_idx = _train_val_split(y_all)
    y_train, y_val = y_all[train_idx], y_all[val_idx]
    print(f"Train: {len(train_idx)}  |  Val: {len(val_idx)}")

    sample_w_lookup, _ = compute_sample_weights(y_all)
    train_idx_os = oversample_indices(train_idx, y_train)
    print(f"Training samples after hard-letter oversampling: {len(train_idx_os)} (was {len(train_idx)})")

    train_ds = make_image_ds(X_img, y_all, sample_w_lookup, train_idx_os, training=True)
    val_ds = make_image_ds(X_img, y_all, sample_w_lookup, val_idx, training=False)

    image_model, effnet_base = build_image_model()
    landmark_model = build_landmark_model()
    image_model.summary(line_length=100)
    landmark_model.summary(line_length=100)

    # Phase 1: train the head (backbone frozen)
    image_model.compile(optimizer=optimizers.Adam(1e-3),
                         loss=categorical_focal_loss(), metrics=["accuracy"],
                         weighted_metrics=[])
    hist_img_1 = image_model.fit(train_ds, validation_data=val_ds,
                                  epochs=12, callbacks=make_callbacks(IMG_CKPT))

    # Phase 2: unfreeze the top of EfficientNet, low LR
    effnet_base.trainable = True
    for layer in effnet_base.layers[:-60]:
        layer.trainable = False
    image_model.compile(optimizer=optimizers.Adam(1e-5),
                         loss=categorical_focal_loss(), metrics=["accuracy"],
                         weighted_metrics=[])
    hist_img_2 = image_model.fit(train_ds, validation_data=val_ds,
                                  epochs=15, callbacks=make_callbacks(IMG_CKPT, patience=5))

    # Landmark model
    Xlm_tr = X_lm[train_idx_os]
    ylm_tr = y_all[train_idx_os]
    wlm_tr = sample_w_lookup[train_idx_os]
    Xlm_tr_aug, ylm_tr_aug, wlm_tr_aug = expand_landmark_dataset(Xlm_tr, ylm_tr, wlm_tr, copies=3)
    ylm_tr_oh = np.eye(NUM_CLASSES, dtype=np.float32)[ylm_tr_aug]
    ylm_val_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_val]

    landmark_model.compile(optimizer=optimizers.Adam(1e-3),
                            loss=categorical_focal_loss(), metrics=["accuracy"],
                            weighted_metrics=[])
    hist_lm = landmark_model.fit(
        Xlm_tr_aug, ylm_tr_oh, sample_weight=wlm_tr_aug,
        validation_data=(X_lm[val_idx], ylm_val_oh),
        epochs=80, batch_size=128,
        callbacks=make_callbacks(LM_CKPT, patience=10), verbose=2)

    # Training curves
    def merge_hist(*hs):
        out = {}
        for h in hs:
            for k, v in h.history.items():
                out.setdefault(k, []).extend(v)
        return out

    hist_img_all = merge_hist(hist_img_1, hist_img_2)
    with open(os.path.join(MODELS_DIR, "training_history.pkl"), "wb") as f:
        pickle.dump({"image_model": hist_img_all, "landmark_model": hist_lm.history}, f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for key, style in [("accuracy", "-"), ("val_accuracy", "--")]:
        axes[0].plot(hist_img_all[key], style, label="img " + key)
        axes[0].plot(hist_lm.history[key], style, label="lm " + key)
    for key, style in [("loss", "-"), ("val_loss", "--")]:
        axes[1].plot(hist_img_all[key], style, label="img " + key)
        axes[1].plot(hist_lm.history[key], style, label="lm " + key)
    axes[0].set_title("Accuracy"); axes[1].set_title("Loss (focal)")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel("epoch")
    plt.show()

    print("Training complete. Checkpoints saved to", MODELS_DIR)
    print("Next: python asl_model.py --evaluate")


# ------------------------------------------------------------------
# Stage: evaluate (+ confusion-driven retraining)
# ------------------------------------------------------------------
def evaluate_models():
    if not (os.path.exists(IMG_CKPT) and os.path.exists(LM_CKPT)):
        sys.exit("Checkpoints not found. Run: python asl_model.py --train")

    X_img, X_lm, y_all = _load_data_cache()
    train_idx, val_idx = _train_val_split(y_all)
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    image_model = tf.keras.models.load_model(IMG_CKPT, compile=False, custom_objects=CUSTOM_OBJECTS)
    landmark_model = tf.keras.models.load_model(LM_CKPT, compile=False, custom_objects=CUSTOM_OBJECTS)

    def predict_probs(idx):
        p_img = image_model.predict(X_img[idx].astype(np.float32),
                                     batch_size=CFG["BATCH_SIZE"], verbose=0)
        p_lm = landmark_model.predict(X_lm[idx], batch_size=256, verbose=0)
        return p_img, p_lm

    def tune_alpha(p_img_val, p_lm_val, y_true):
        best_alpha, best_acc = 0.5, 0.0
        for alpha in np.arange(0.0, 1.01, 0.05):
            acc = accuracy_score(y_true, np.argmax(alpha * p_img_val + (1 - alpha) * p_lm_val, axis=1))
            if acc > best_acc:
                best_alpha, best_acc = float(alpha), acc
        return best_alpha, best_acc

    p_img_val, p_lm_val = predict_probs(val_idx)
    best_alpha, best_acc = tune_alpha(p_img_val, p_lm_val, y_val)
    CFG["FUSION_ALPHA"] = best_alpha
    p_fused = best_alpha * p_img_val + (1 - best_alpha) * p_lm_val
    y_pred = np.argmax(p_fused, axis=1)

    print(f"Image-only val accuracy   : {accuracy_score(y_val, np.argmax(p_img_val, 1)):.4f}")
    print(f"Landmark-only val accuracy: {accuracy_score(y_val, np.argmax(p_lm_val, 1)):.4f}")
    print(f"FUSED val accuracy (alpha={best_alpha:.2f}): {best_acc:.4f}")

    prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="macro", zero_division=0)
    print(f"\nMacro Precision: {prec:.4f} | Macro Recall: {rec:.4f} | Macro F1: {f1:.4f}\n")
    print(classification_report(y_val, y_pred, target_names=LETTERS, zero_division=0))

    cm = confusion_matrix(y_val, y_pred, labels=np.arange(NUM_CLASSES))
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LETTERS, yticklabels=LETTERS, cbar=False)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Fused-model confusion matrix (validation)")
    plt.show()

    per_class_acc = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    plt.figure(figsize=(13, 3))
    colors = ["#d62728" if l in CFG["HARD_LETTERS"] else "#1f77b4" for l in LETTERS]
    plt.bar(LETTERS, per_class_acc, color=colors)
    plt.ylim(min(0.8, per_class_acc.min() - 0.02), 1.0)
    plt.title("Per-class accuracy (red = designated hard letters)"); plt.grid(alpha=0.3, axis="y")
    plt.show()

    pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], LETTERS[i], LETTERS[j]))
    pairs.sort(reverse=True)
    print("Most confused pairs (true -> predicted):")
    for n, a, b in pairs[:12]:
        print(f"  {a} -> {b}: {n} errors")

    # Confusion-driven retraining
    CONFUSION_ERROR_THRESHOLD = 2
    extra_boost = {}
    for n, a, b in pairs:
        if n > CONFUSION_ERROR_THRESHOLD:
            for l in (a, b):
                extra_boost[LETTER_TO_IDX[l]] = max(extra_boost.get(LETTER_TO_IDX[l], 1.0), 1.5)

    if extra_boost:
        boosted_letters = sorted(LETTERS[c] for c in extra_boost)
        print("Boosting classes:", boosted_letters)

        sample_w_lookup, _ = compute_sample_weights(y_all, extra_boost=extra_boost)
        train_idx_os = oversample_indices(train_idx, y_train, frac=CFG["HARD_OVERSAMPLE"] + 0.25)
        train_ds = make_image_ds(X_img, y_all, sample_w_lookup, train_idx_os, training=True)
        val_ds = make_image_ds(X_img, y_all, sample_w_lookup, val_idx, training=False)

        image_model.compile(optimizer=optimizers.Adam(5e-6),
                             loss=categorical_focal_loss(), metrics=["accuracy"],
                             weighted_metrics=[])
        image_model.fit(train_ds, validation_data=val_ds, epochs=6,
                         callbacks=make_callbacks(IMG_CKPT, patience=3))

        Xb, yb, wb = expand_landmark_dataset(
            X_lm[train_idx_os], y_all[train_idx_os], sample_w_lookup[train_idx_os], copies=2)
        ylm_val_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_val]
        landmark_model.compile(optimizer=optimizers.Adam(2e-4),
                                loss=categorical_focal_loss(), metrics=["accuracy"],
                                weighted_metrics=[])
        landmark_model.fit(Xb, np.eye(NUM_CLASSES, dtype=np.float32)[yb], sample_weight=wb,
                            validation_data=(X_lm[val_idx], ylm_val_oh),
                            epochs=20, batch_size=128,
                            callbacks=make_callbacks(LM_CKPT, patience=6), verbose=2)

        image_model = tf.keras.models.load_model(IMG_CKPT, compile=False, custom_objects=CUSTOM_OBJECTS)
        landmark_model = tf.keras.models.load_model(LM_CKPT, compile=False, custom_objects=CUSTOM_OBJECTS)
        p_img_val, p_lm_val = predict_probs(val_idx)
        best_alpha, best_acc = tune_alpha(p_img_val, p_lm_val, y_val)
        CFG["FUSION_ALPHA"] = best_alpha
        print(f"\nPost-retrain FUSED val accuracy (alpha={best_alpha:.2f}): {best_acc:.4f}")

    with open(TUNED_CFG_PATH, "w") as f:
        json.dump(CFG, f, indent=2)
    print("\nSaved tuned config to", TUNED_CFG_PATH)
    print("Next: python asl_model.py --collect-personal   (optional, for best real-world accuracy)")
    print("  or: python asl_model.py --export")


# ------------------------------------------------------------------
# Local webcam helper — replaces the Colab browser<->Python JS bridge
# ------------------------------------------------------------------
class LocalWebcam:
    def __init__(self, cam_index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam (index {cam_index}). "
                "Check that a camera is connected and not already in use."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.flip(frame, 1)   # mirror, matches original behavior

    def release(self):
        self.cap.release()


# ------------------------------------------------------------------
# Stage: collect-personal
# ------------------------------------------------------------------
def collect_personal_dataset(letters=None, per_letter=None):
    letters = letters or LETTERS
    per_letter = per_letter or CFG["PERSONAL_PER_LETTER"]

    if os.path.exists(PERSONAL_CACHE):
        _p = np.load(PERSONAL_CACHE)
        P_img, P_lm, P_y = list(_p["X_img"]), list(_p["X_lm"]), list(_p["y"])
    else:
        P_img, P_lm, P_y = [], [], []
    done_counts = Counter(P_y)

    cam = LocalWebcam()
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                            min_detection_confidence=0.6,
                            min_tracking_confidence=0.6, model_complexity=1)
    window = "ASL - Personal Data Collection  (s: capture  n: skip letter  q: quit)"
    quit_all = False
    try:
        for letter in letters:
            cls = LETTER_TO_IDX[letter]
            have = done_counts.get(cls, 0)
            if have >= per_letter:
                print(f"{letter}: already have {have} samples, skipping.")
                continue
            capturing = False
            while have < per_letter:
                frame = cam.read()
                if frame is None:
                    continue
                crop, vec, bbox, hand_lms = extract_from_image(frame, hands)
                overlay = frame.copy()
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(overlay, hand_lms, mp_hands.HAND_CONNECTIONS)
                msg = f"SIGN '{letter}'  [{have}/{per_letter}]  " + \
                      ("CAPTURING - vary angle/distance" if capturing else "press 's' to capture")
                cv2.rectangle(overlay, (0, 0), (640, 34), (0, 0, 0), -1)
                cv2.putText(overlay, msg, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            (0, 255, 0) if capturing else (0, 200, 255), 2)
                cv2.imshow(window, overlay)

                if capturing and crop is not None:
                    P_img.append(crop.astype(np.uint8))
                    P_lm.append(vec); P_y.append(cls)
                    have += 1

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('s'), ord('S')):
                    capturing = True
                elif key in (ord('n'), ord('N')):
                    have = per_letter
                elif key in (ord('q'), ord('Q')):
                    quit_all = True
                    break

            np.savez_compressed(
                PERSONAL_CACHE,
                X_img=np.stack(P_img) if P_img else np.zeros((0, CFG["IMG_SIZE"], CFG["IMG_SIZE"], 3), np.uint8),
                X_lm=np.stack(P_lm) if P_lm else np.zeros((0, 63), np.float32),
                y=np.array(P_y, dtype=np.int64),
            )
            print(f"{letter}: saved (total personal samples: {len(P_y)})")
            if quit_all:
                break
    finally:
        hands.close()
        cam.release()
        cv2.destroyAllWindows()
    print("Personal dataset size:", len(P_y))
    print("Next: python asl_model.py --finetune")


# ------------------------------------------------------------------
# Stage: finetune (on personal data)
# ------------------------------------------------------------------
def finetune_personal():
    if not (os.path.exists(IMG_CKPT) and os.path.exists(LM_CKPT)):
        sys.exit("Base checkpoints not found. Run: python asl_model.py --train")
    if not os.path.exists(PERSONAL_CACHE):
        sys.exit("No personal dataset found. Run: python asl_model.py --collect-personal")

    X_img, X_lm, y_all = _load_data_cache()
    train_idx, _ = _train_val_split(y_all)
    y_train = y_all[train_idx]

    image_model = tf.keras.models.load_model(IMG_CKPT, compile=False, custom_objects=CUSTOM_OBJECTS)
    landmark_model = tf.keras.models.load_model(LM_CKPT, compile=False, custom_objects=CUSTOM_OBJECTS)

    _p = np.load(PERSONAL_CACHE)
    P_img, P_lm, P_y = _p["X_img"], _p["X_lm"], _p["y"]
    print("Personal samples:", P_img.shape[0], "| per-class:",
          dict(sorted(Counter([LETTERS[i] for i in P_y]).items())))

    p_tr, p_va = train_test_split(np.arange(len(P_y)), test_size=0.15,
                                   stratify=P_y, random_state=SEED)

    rng = np.random.RandomState(SEED)
    replay = np.concatenate([
        rng.choice(train_idx[y_train == c], size=min(80, np.sum(y_train == c)), replace=False)
        for c in range(NUM_CLASSES)
    ])

    FT_source = np.concatenate([np.zeros(len(p_tr) * 3, dtype=np.int64),
                                 np.ones(len(replay), dtype=np.int64)])
    FT_index = np.concatenate([np.repeat(p_tr, 3), replay])
    FT_y = np.concatenate([np.repeat(P_y[p_tr], 3), y_all[replay]])
    FT_w = np.concatenate([np.full(len(p_tr) * 3, 2.0, np.float32),
                            np.full(len(replay), 1.0, np.float32)])
    for l in CFG["HARD_LETTERS"]:
        FT_w[FT_y == LETTER_TO_IDX[l]] *= CFG["HARD_BOOST"]
    perm = rng.permutation(len(FT_y))
    FT_source, FT_index, FT_y, FT_w = FT_source[perm], FT_index[perm], FT_y[perm], FT_w[perm]

    Pva_y_oh = np.eye(NUM_CLASSES, dtype=np.float32)[P_y[p_va]]

    def _fetch_ft_image(source, idx):
        source, idx = int(source), int(idx)
        img = P_img[idx] if source == 0 else X_img[idx]
        return img.astype(np.float32)

    def make_ft_image_ds(source_arr, index_arr, y_arr, w_arr, batch_size, training):
        y_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_arr]
        ds = tf.data.Dataset.from_tensor_slices(
            (source_arr, index_arr, y_oh.astype(np.float32), w_arr.astype(np.float32)))
        if training:
            ds = ds.shuffle(len(index_arr), seed=SEED, reshuffle_each_iteration=True)

        def _map(s, i, y, w):
            img = tf.numpy_function(_fetch_ft_image, [s, i], tf.float32)
            img.set_shape((CFG["IMG_SIZE"], CFG["IMG_SIZE"], 3))
            y.set_shape((NUM_CLASSES,)); w.set_shape(())
            return img, y, w

        ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ft_train_ds = make_ft_image_ds(FT_source, FT_index, FT_y, FT_w, CFG["BATCH_SIZE"], training=True)
    ft_val_imgs = P_img[p_va].astype(np.float32)

    image_model.compile(optimizer=optimizers.Adam(1e-5),
                         loss=categorical_focal_loss(), metrics=["accuracy"],
                         weighted_metrics=[])
    image_model.fit(ft_train_ds, validation_data=(ft_val_imgs, Pva_y_oh), epochs=10,
                     callbacks=make_callbacks(IMG_CKPT_PERSONAL, patience=4))
    del ft_val_imgs

    FT_X_lm = np.empty((len(FT_index), 63), dtype=np.float32)
    is_personal = FT_source == 0
    FT_X_lm[is_personal] = P_lm[FT_index[is_personal]]
    FT_X_lm[~is_personal] = X_lm[FT_index[~is_personal]]
    FTlm_X, FTlm_y, FTlm_w = expand_landmark_dataset(FT_X_lm, FT_y, FT_w, copies=2)
    landmark_model.compile(optimizer=optimizers.Adam(3e-4),
                            loss=categorical_focal_loss(), metrics=["accuracy"],
                            weighted_metrics=[])
    landmark_model.fit(FTlm_X, np.eye(NUM_CLASSES, dtype=np.float32)[FTlm_y], sample_weight=FTlm_w,
                        validation_data=(P_lm[p_va], Pva_y_oh),
                        epochs=30, batch_size=128,
                        callbacks=make_callbacks(LM_CKPT_PERSONAL, patience=8), verbose=2)

    image_model = tf.keras.models.load_model(IMG_CKPT_PERSONAL, compile=False, custom_objects=CUSTOM_OBJECTS)
    landmark_model = tf.keras.models.load_model(LM_CKPT_PERSONAL, compile=False, custom_objects=CUSTOM_OBJECTS)
    pv_img = image_model.predict(P_img[p_va].astype(np.float32), verbose=0)
    pv_lm = landmark_model.predict(P_lm[p_va], verbose=0)
    best_alpha, best_acc = 0.5, 0.0
    for alpha in np.arange(0.0, 1.01, 0.05):
        acc = accuracy_score(P_y[p_va], np.argmax(alpha * pv_img + (1 - alpha) * pv_lm, 1))
        if acc > best_acc:
            best_alpha, best_acc = float(alpha), acc
    CFG["FUSION_ALPHA"] = best_alpha
    print(f"\nPersonalized fused accuracy on YOUR hand: {best_acc:.4f} (alpha={best_alpha:.2f})")

    with open(TUNED_CFG_PATH, "w") as f:
        json.dump(CFG, f, indent=2)
    print("Next: python asl_model.py --export")


# ------------------------------------------------------------------
# Stage: export
# ------------------------------------------------------------------
def export_models():
    if os.path.exists(TUNED_CFG_PATH):
        with open(TUNED_CFG_PATH) as f:
            CFG.update(json.load(f))
    else:
        print("No tuned_config.json found yet (run --evaluate first) — exporting defaults.")

    os.makedirs(EXPORT_DIR, exist_ok=True)
    for src in [IMG_CKPT, LM_CKPT, IMG_CKPT_PERSONAL, LM_CKPT_PERSONAL]:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(EXPORT_DIR, os.path.basename(src)))

    with open(os.path.join(EXPORT_DIR, "label_encoder.json"), "w") as f:
        json.dump({"classes": LETTERS}, f, indent=2)
    with open(os.path.join(EXPORT_DIR, "config.json"), "w") as f:
        json.dump(CFG, f, indent=2)

    hist_path = os.path.join(MODELS_DIR, "training_history.pkl")
    if os.path.exists(hist_path):
        shutil.copy(hist_path, os.path.join(EXPORT_DIR, "training_history.pkl"))

    print("Exported files:")
    for p in sorted(os.listdir(EXPORT_DIR)):
        print("  ", p, f"({os.path.getsize(os.path.join(EXPORT_DIR, p)) / 1e6:.1f} MB)")

    zip_base = os.path.join(BASE_DIR, "asl_model_export")
    shutil.make_archive(zip_base, "zip", EXPORT_DIR)
    print(f"\nZip ready at {zip_base}.zip")
    print("Next: python asl_model.py --realtime")


# ------------------------------------------------------------------
# Load models for inference (personalized if available, else base)
# ------------------------------------------------------------------
def load_models_for_inference():
    global LETTERS, LETTER_TO_IDX

    label_path = os.path.join(EXPORT_DIR, "label_encoder.json")
    cfg_path = os.path.join(EXPORT_DIR, "config.json")
    if not (os.path.exists(label_path) and os.path.exists(cfg_path)):
        sys.exit("No exported model found. Run: python asl_model.py --export")

    with open(label_path) as f:
        LETTERS = json.load(f)["classes"]
    LETTER_TO_IDX = {l: i for i, l in enumerate(LETTERS)}
    with open(cfg_path) as f:
        CFG.update(json.load(f))

    def _pick(personal, base):
        p = os.path.join(EXPORT_DIR, personal)
        return p if os.path.exists(p) else os.path.join(EXPORT_DIR, base)

    img_path = _pick("image_model_personal.keras", "image_model_best.keras")
    lm_path = _pick("landmark_model_personal.keras", "landmark_model_best.keras")
    image_model = tf.keras.models.load_model(img_path, compile=False, custom_objects=CUSTOM_OBJECTS)
    landmark_model = tf.keras.models.load_model(lm_path, compile=False, custom_objects=CUSTOM_OBJECTS)
    print("Loaded image model   :", img_path)
    print("Loaded landmark model:", lm_path)
    print(f"Fusion alpha = {CFG['FUSION_ALPHA']:.2f} | confidence gate = {CFG['CONF_THRESHOLD']:.2f}")
    return image_model, landmark_model


# ------------------------------------------------------------------
# Stage: realtime — live webcam inference, stabilization, word building
# ------------------------------------------------------------------
def run_realtime_recognition(image_model, landmark_model, max_minutes=10):
    conf_gate = float(CFG["CONF_THRESHOLD"])
    stable_secs = float(CFG["STABLE_SECONDS"])
    alpha = float(CFG["FUSION_ALPHA"])

    history = deque()
    sentence = ""
    last_accepted = None
    ready_for_repeat = True
    release_start = None
    RELEASE_SECS = 0.4

    fps_ema, prev_t = 0.0, time.time()
    t_start = time.time()

    cam = LocalWebcam()
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                            min_detection_confidence=0.6,
                            min_tracking_confidence=0.6, model_complexity=1)
    window = "ASL Real-Time Recognition"
    try:
        while (time.time() - t_start) < max_minutes * 60:
            frame = cam.read()
            if frame is None:
                continue
            now = time.time()
            crop, vec, bbox, hand_lms = extract_from_image(frame, hands)
            overlay = frame.copy()

            pred_letter, pred_conf = None, 0.0
            if crop is not None:
                p_img = image_model(crop[None].astype(np.float32), training=False).numpy()[0]
                p_lm = landmark_model(vec[None], training=False).numpy()[0]
                p = alpha * p_img + (1 - alpha) * p_lm
                pred_idx = int(np.argmax(p))
                pred_letter, pred_conf = LETTERS[pred_idx], float(p[pred_idx])
                history.append((now, pred_idx, pred_conf))

                x1, y1, x2, y2 = bbox
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                mp_drawing.draw_landmarks(overlay, hand_lms, mp_hands.HAND_CONNECTIONS)

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
                        and (stable_letter != last_accepted or ready_for_repeat)):
                    sentence += stable_letter
                    last_accepted = stable_letter
                    ready_for_repeat = False
                    release_start = None
                    history.clear()

            dt = now - prev_t; prev_t = now
            if dt > 0:
                fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if fps_ema else 1.0 / dt

            cv2.rectangle(overlay, (0, 0), (640, 96), (0, 0, 0), -1)
            if crop is None:
                head, color = "No hand detected", (0, 0, 255)
            elif pred_conf < conf_gate:
                head, color = f"Uncertain ({pred_letter} {pred_conf*100:.0f}%)", (0, 165, 255)
            else:
                head, color = f"{pred_letter}  {pred_conf*100:.1f}%", (0, 255, 0)
            cv2.putText(overlay, head, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(overlay, f"FPS: {fps_ema:.1f}", (545, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if stable_letter and stable_conf >= conf_gate:
                cv2.rectangle(overlay, (10, 44), (10 + int(300 * progress), 56), (0, 255, 0), -1)
                cv2.rectangle(overlay, (10, 44), (310, 56), (255, 255, 255), 1)
                cv2.putText(overlay, f"hold {stable_letter}", (320, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            word = sentence.split(" ")[-1]
            cv2.putText(overlay, f"Word: {word}", (10, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(overlay, (0, 448), (640, 480), (0, 0, 0), -1)
            cv2.putText(overlay, f"Sentence: {sentence[-46:]}", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(overlay, "SPACE:space  BACKSPACE:delete  C:clear  Q:quit", (150, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            cv2.imshow(window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                sentence += " "; last_accepted = None; ready_for_repeat = True
            elif key == 8:  # Backspace
                sentence = sentence[:-1]
            elif key in (ord('c'), ord('C')):
                sentence = ""; last_accepted = None; ready_for_repeat = True
            elif key in (ord('q'), ord('Q')):
                break
    finally:
        hands.close()
        cam.release()
        cv2.destroyAllWindows()

    print("\nSession ended.")
    print("Final text:", repr(sentence))
    return sentence


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ASL alphabet recognition — hybrid image + landmark model (local version).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--download", action="store_true", help="Download & unzip the Kaggle ASL alphabet dataset")
    parser.add_argument("--preprocess", action="store_true", help="MediaPipe crop + landmark extraction (cached)")
    parser.add_argument("--train", action="store_true", help="Train the image + landmark models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate + confusion-driven retraining")
    parser.add_argument("--collect-personal", action="store_true", help="Collect your own hand dataset via webcam")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune models on your personal dataset")
    parser.add_argument("--export", action="store_true", help="Export models + config to a portable folder")
    parser.add_argument("--realtime", action="store_true", help="Run real-time webcam recognition")
    parser.add_argument("--all", action="store_true", help="Run every stage above, in order")
    parser.add_argument("--letters", type=str, default=None,
                         help="Restrict --collect-personal to these letters, e.g. MNST")
    parser.add_argument("--per-letter", type=int, default=None,
                         help="Personal images to collect per letter (default from CFG)")
    parser.add_argument("--max-minutes", type=float, default=10, help="Max minutes for --realtime session")
    args = parser.parse_args()

    ran_anything = any([
        args.download, args.preprocess, args.train, args.evaluate,
        args.collect_personal, args.finetune, args.export, args.realtime, args.all,
    ])
    if not ran_anything:
        parser.print_help()
        return

    setup_gpu()

    if args.all:
        download_dataset()
        preprocess_dataset()
        train_models()
        evaluate_models()
        collect_personal_dataset()
        finetune_personal()
        export_models()
        image_model, landmark_model = load_models_for_inference()
        run_realtime_recognition(image_model, landmark_model, max_minutes=args.max_minutes)
        return

    if args.download:
        download_dataset()
    if args.preprocess:
        preprocess_dataset()
    if args.train:
        train_models()
    if args.evaluate:
        evaluate_models()
    if args.collect_personal:
        letters = list(args.letters.upper()) if args.letters else LETTERS
        per_letter = args.per_letter or CFG["PERSONAL_PER_LETTER"]
        collect_personal_dataset(letters=letters, per_letter=per_letter)
    if args.finetune:
        finetune_personal()
    if args.export:
        export_models()
    if args.realtime:
        image_model, landmark_model = load_models_for_inference()
        run_realtime_recognition(image_model, landmark_model, max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()
