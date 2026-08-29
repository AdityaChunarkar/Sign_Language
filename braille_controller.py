#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serial bridge to the DIY single-cell refreshable braille display (Arduino
Uno + ULN2803A + 6 solenoids). The Arduino reads one line of text at a
time over USB serial at 9600 baud and fires the dot pattern for that
character, so the whole protocol is: write "<CHAR>\n" and it displays.

This module owns the serial connection and a background "player" thread
that walks through a piece of text (typed, or extracted from a PDF) one
character at a time on a timer, so the web UI just needs to set text and
call play/pause/next/prev.
"""

import threading
import time

import serial
import serial.tools.list_ports

SUPPORTED_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Standard Grade-1 English Braille Unicode code points, purely for showing
# a visual preview of the current cell in the browser (not sent to the
# Arduino, which does its own dot-pattern lookup from the plain letter).
LETTER_TO_UNICODE_BRAILLE = {
    "A": "⠁", "B": "⠃", "C": "⠉", "D": "⠙", "E": "⠑",
    "F": "⠋", "G": "⠛", "H": "⠓", "I": "⠊", "J": "⠚",
    "K": "⠅", "L": "⠇", "M": "⠍", "N": "⠝", "O": "⠕",
    "P": "⠏", "Q": "⠟", "R": "⠗", "S": "⠎", "T": "⠞",
    "U": "⠥", "V": "⠧", "W": "⠺", "X": "⠭", "Y": "⠽",
    "Z": "⠵",
}

DEFAULT_BAUD = 9600
DEFAULT_DELAY_MS = 800
MIN_DELAY_MS = 150
ARDUINO_RESET_SECS = 2.0  # opening the port reboots most Arduino Unos


class BrailleController:
    """Thread-safe wrapper around the serial link + a play/pause text feed."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ser = None
        self._port = None
        self._text = ""
        self._index = 0
        self._playing = False
        self._delay_ms = DEFAULT_DELAY_MS
        self._stop_event = threading.Event()
        self._thread = None
        self._last_error = None

    # ---- connection -----------------------------------------------------
    def list_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, port, baud=DEFAULT_BAUD):
        with self._lock:
            self._disconnect_locked()
            self._ser = serial.Serial(port, baud, timeout=1)
            self._port = port
            self._last_error = None
        time.sleep(ARDUINO_RESET_SECS)  # let the board finish its auto-reset

    def disconnect(self):
        with self._lock:
            self._disconnect_locked()

    def _disconnect_locked(self):
        self._playing = False
        self._stop_event.set()
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self._port = None

    # ---- text queue -------------------------------------------------------
    def set_text(self, text):
        with self._lock:
            cleaned = "".join(
                c for c in text.upper() if c in SUPPORTED_LETTERS or c.isspace()
            )
            cleaned = " ".join(cleaned.split())  # collapse whitespace runs
            self._text = cleaned
            self._index = 0

    def set_speed(self, delay_ms):
        with self._lock:
            self._delay_ms = max(MIN_DELAY_MS, int(delay_ms))

    # ---- playback ---------------------------------------------------------
    def play(self):
        with self._lock:
            if self._playing or not self._text or self._ser is None:
                return
            self._playing = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def pause(self):
        with self._lock:
            self._playing = False
            self._stop_event.set()

    def next_char(self):
        with self._lock:
            if self._index < len(self._text):
                self._send_char_locked(self._text[self._index])
                self._index += 1

    def prev_char(self):
        with self._lock:
            if self._index > 0:
                self._index -= 1

    def reset(self):
        with self._lock:
            self._index = 0

    def _run(self):
        while not self._stop_event.is_set():
            with self._lock:
                if self._index >= len(self._text):
                    self._playing = False
                    break
                ch = self._text[self._index]
                self._send_char_locked(ch)
                self._index += 1
                delay = self._delay_ms / 1000.0
            self._stop_event.wait(delay)

    def _send_char_locked(self, ch):
        if ch.isspace():
            return  # natural pause between words; nothing to send
        if self._ser is None:
            return
        try:
            self._ser.write((ch + "\n").encode("ascii"))
        except Exception as e:
            self._last_error = str(e)
            self._playing = False
            self._stop_event.set()

    # ---- status -------------------------------------------------------
    def status(self):
        with self._lock:
            current = self._text[self._index] if self._index < len(self._text) else None
            return {
                "connected": self._ser is not None,
                "port": self._port,
                "text": self._text,
                "index": self._index,
                "length": len(self._text),
                "current_char": current,
                "current_unicode": LETTER_TO_UNICODE_BRAILLE.get(current),
                "playing": self._playing,
                "delay_ms": self._delay_ms,
                "error": self._last_error,
            }
