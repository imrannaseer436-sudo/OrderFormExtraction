"""
Automated deskew + contrast normalization for images sent to a VLM,
built 2026-08-17 specifically for extract_ollama_cloud.py's mistral call
path -- NOT wired into grid.py or ocr_cell_read.py, and not imported by
either. That CV+OCR path is already validated working (confirmed exact
match on sample 5.jpeg) and untuned new preprocessing here could change
its behavior in ways that haven't been checked; keeping this file
completely separate means it can only ever affect the mistral code path
that explicitly imports it.

Motivation: on sample 8.jpeg, one row's values bled into the row below it
when read by mistral-large-3:675b. A prompt-only fix made this WORSE, not
better (see CLAUDE.md's "Row-crops confirmed to fix the row-bleed bug"
section) -- the bleed only actually stopped when the user supplied a
manually deskewed, tightly-cropped version of the same photo (see
CLAUDE.md's "Row-bleed solved -- but by a clean input image" section).
This module is an attempt to automate that specific manual step (deskew;
contrast normalization is a reasonable companion for real, unevenly-lit
phone photos) so it becomes a pipeline step instead of a one-off manual
workaround -- NOT yet confirmed to reproduce the same fix; see CLAUDE.md's
dated section on this for the actual validated result.

Deliberately does not attempt to replicate cropping or perspective
correction -- the manual fix that worked on sample 8.jpeg may have
involved more than just deskew+contrast (a better framing, a flatter
angle), and this module only automates the two specific, well-defined
transforms named in the task this was built from.
"""

from __future__ import annotations

import cv2
import numpy as np

from grid import _estimate_skew_deg


def preprocess_for_vlm(image_bytes: bytes) -> bytes:
    """Deskew (small-angle camera-tilt correction, reusing grid.py's own
    Hough-transform-based estimator) + CLAHE contrast normalization
    (applied on the L channel in LAB space, so color balance is preserved
    while local contrast is boosted -- helps an unevenly-lit real photo
    without the color-shifting a plain global histogram equalization on
    RGB would cause). Returns re-encoded JPEG bytes."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    skew = _estimate_skew_deg(binary)
    if abs(skew) > 0.05:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else image_bytes
