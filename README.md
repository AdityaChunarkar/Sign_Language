# ASL Recognition + Refreshable Braille Display

Two related accessibility tools sharing one Flask web app:

1. **Sign → Text** — real-time American Sign Language alphabet recognition
   from a webcam, using a hybrid image + hand-landmark model.
2. **Text → Braille** — type text or upload a PDF and have it streamed,
   one letter at a time, to a DIY single-cell refreshable braille display
   built from an Arduino Uno and six solenoids.

Both are served from the same local Flask process at `http://localhost:5000`.

---

## Architecture overview

```
                         ┌─────────────────────────────┐
                         │           Browser            │
                         │                               │
                         │  index.html + app.js          │  templates/braille.html
                         │  (webcam capture, canvas       │  + braille.js
                         │   overlay, sentence builder)   │  (text/PDF UI, playback
                         │                               │   controls, port picker)
                         └───────────────┬───────────────┘
                                         │ HTTP (JSON / multipart)
                                         ▼
                         ┌─────────────────────────────┐
                         │         webapp.py             │
                         │        (Flask app)            │
                         │                               │
                         │  /predict   /control           │  /braille/*
                         │  (ASL inference loop)          │  (text queue, PDF
                         │                               │   extraction, serial
                         │                               │   control)
                         └───────┬───────────────┬───────┘
                                 │               │
                    imports      │               │  imports
                                 ▼               ▼
                    ┌─────────────────┐   ┌───────────────────────┐
                    │  asl_model.py    │   │ braille_controller.py  │
                    │                  │   │                        │
                    │ MediaPipe hand   │   │ pyserial bridge +      │
                    │ extraction,      │   │ background playback    │
                    │ EfficientNetB0   │   │ thread (play/pause/    │
                    │ + landmark MLP,  │   │ next/prev)             │
                    │ late fusion      │   │                        │
                    └────────┬─────────┘   └───────────┬────────────┘
                             │                          │
                             ▼                          ▼
                  asl_project/asl_model_export/   USB serial (9600 baud)
                  (exported .keras models,               │
                   label encoder, config)                ▼
                                                 ┌────────────────────┐
                                                 │   Arduino Uno       │
                                                 │  + ULN2803A driver  │
                                                 │  + 6 solenoids      │
                                                 │  (one braille cell) │
                                                 └────────────────────┘
```

---

## 1. Sign → Text (ASL recognition)

### Model pipeline (`asl_model.py`)

A single CLI script drives the whole offline pipeline, run in stages so you
never have to redo expensive steps:

| Stage | Flag | What it does |
|---|---|---|
| Download | `--download` | Fetches the Kaggle ASL alphabet dataset (or use a manually placed zip) |
| Preprocess | `--preprocess` | Runs MediaPipe Hands over every raw image, crops to the hand region, extracts a 63-dim landmark vector, caches both to `processed_asl.npz` |
| Train | `--train` | Trains two models in parallel: an EfficientNetB0-based image classifier and a small MLP on hand landmarks |
| Evaluate | `--evaluate` | Tunes the late-fusion weight between the two models, prints a confusion matrix, and does one round of confusion-driven retraining on the most-confused letter pairs |
| Collect personal | `--collect-personal` | Optional: capture your own hand via webcam to personalize the model |
| Finetune | `--finetune` | Optional: fine-tunes both models on your personal samples (with replay of the original data to avoid forgetting) |
| Export | `--export` | Copies the best checkpoints + label encoder + tuned config into a portable `asl_model_export/` folder (this is what the web app loads) |
| Realtime | `--realtime` | Standalone `cv2.imshow`-based live demo (the original, non-web version) |

**Why two models fused together, not one:** the image branch (EfficientNetB0 on
the cropped hand) captures overall hand shape and texture; the landmark
branch (MediaPipe's 21 hand keypoints, wrist-centered and scale-normalized)
captures precise finger geometry and is far more robust to lighting/background.
`--evaluate` searches for the blend weight (`FUSION_ALPHA`) that maximizes
validation accuracy, since the two branches make different mistakes.

**Stability logic (used identically in `--realtime` and `webapp.py`):** a
single frame's prediction is noisy, so a short rolling history of recent
predictions is kept. A letter is only "accepted" into the output once it's
been the dominant prediction for a configurable window (`STABLE_SECONDS`)
at high confidence (`CONF_THRESHOLD`), which prevents a hand transitioning
between signs from spamming random letters.

### Web version (`webapp.py` + `/predict`)

The browser captures a webcam frame every 200ms (`static/app.js`), sends it
as a base64 JPEG to `POST /predict`, and the server:

1. Runs MediaPipe hand detection + cropping (`extract_from_image`, imported
   directly from `asl_model.py` — no duplicated logic).
2. Runs both Keras models on the crop/landmarks and fuses their output.
3. Applies the same temporal-stability algorithm as `--realtime`, using a
   server-side global `STATE` dict (single-user local app, so no per-session
   handling was needed).
4. Returns the predicted letter, confidence, stabilization progress, hand
   bounding box (for the on-canvas overlay), and the built-up sentence.

`POST /control` handles Space / Backspace / Clear on the sentence.

---

## 2. Text → Braille

### Hardware protocol

The display is a **single braille cell** (2×3 = 6 dots) driven by 6
solenoids through a ULN2803A darlington array, controlled by an Arduino Uno.
The Arduino sketch is intentionally simple: it blocks on
`Serial.readStringUntil('\n')`, looks the received character up in a
26-letter dot-pattern table, and fires the corresponding solenoids. So the
entire protocol from the PC side is: **write `"<LETTER>\n"` at 9600 baud.**

### Software side (`braille_controller.py`)

`BrailleController` is a small thread-safe class that owns:
- The `pyserial` connection (open/close, with the ~2s delay needed because
  opening the port reboots most Arduino Unos).
- A text queue (typed text or PDF-extracted text, uppercased, filtered to
  only the letters the Arduino's table supports, whitespace collapsed).
- A background daemon thread for playback: on `play()`, it walks the queue
  one character at a time, writing each to serial and sleeping for the
  configured delay, until paused or the text ends.
- `next_char()` / `prev_char()` / `reset()` for manual stepping.
- `status()` — a snapshot (connected?, current index/char, a Unicode
  braille glyph for browser preview, playing state) polled by the frontend
  every 300ms.

### Web routes (`webapp.py`)

| Route | Purpose |
|---|---|
| `GET /braille` | Serves the page |
| `GET /braille/ports` | Lists available serial ports (`serial.tools.list_ports`) |
| `POST /braille/connect` / `/braille/disconnect` | Opens/closes the Arduino connection |
| `POST /braille/text` | Loads typed text into the queue |
| `POST /braille/upload` | Extracts text from an uploaded PDF (`pypdf`) and loads it |
| `POST /braille/control` | `play` / `pause` / `next` / `prev` / `reset` / `speed` |
| `GET /braille/status` | Polled by the frontend to keep the UI in sync |

The frontend (`templates/braille.html` + `static/braille.js`) never talks to
the serial port directly — everything goes through these routes, since only
the Python process (not the browser) can access the OS's COM ports.

---

## Project layout

```
asl_model.py            Offline pipeline: data prep, training, evaluation, export
braille_controller.py   Serial bridge + playback thread for the braille cell
webapp.py               Flask app: routes for both features, model loading
templates/
  index.html            Sign → Text page
  braille.html           Text → Braille page
static/
  app.js / style.css     Shared styling + ASL page logic
  braille.js             Braille page logic
requirements.txt        Python dependencies
asl_project/            Generated at runtime — dataset cache, model
                        checkpoints, exported model (gitignored, regenerate
                        via the asl_model.py stages above)
```

---

## Running it locally

```powershell
py -3.11 -m venv .venv311
& ".venv311\Scripts\python.exe" -m pip install -r requirements.txt

# one-time: train + export a model (see asl_model.py stages above),
# or copy in an already-exported asl_project/asl_model_export/ folder

& ".venv311\Scripts\python.exe" webapp.py
```

Then open `http://localhost:5000` (Sign → Text) or `http://localhost:5000/braille`
(Text → Braille). The server binds to `0.0.0.0`, so it's also reachable from
other devices on the same network at `http://<your-LAN-IP>:5000`.

**Note:** Python 3.11 or 3.12 is required — TensorFlow and MediaPipe don't
yet publish wheels for newer Python releases.

## Why the braille page can't be hosted publicly

The Arduino is connected over USB serial directly to whichever machine runs
`webapp.py`. A cloud host has no physical path to that port, so `/braille`
only makes sense run locally (or on your LAN) — hosting it publicly could
only ever demo the text/PDF/playback UI with no hardware attached, not
actually drive the display.
