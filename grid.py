"""
Deterministic row-boundary detection via classical CV, replacing the two
earlier attempts at having the VLM *guess* row positions (both failed —
see CLAUDE.md). The ruled table lines on these forms are real, physical
lines; a photographed (not flat-scanned) page shows them with a few
degrees of rotation plus mild perspective warp, but they're still
detectable precisely once you look for relative density peaks instead of
requiring one pixel-perfect straight line.

Deliberately row-only, not full grid: column (vertical line) detection
was tried and stayed sparse/unreliable on the test image, but Stage B
already gets the column headers as text (from Stage A) plus a header
image crop, so column labeling doesn't depend on this module.

**Multiple candidate strategies (2026-08-0X)**: the original approach
(total dark-pixel count per row) works well on some forms but was
confirmed to fail on a second, lower-resolution/denser form -- it found a
"clean" uniform run of lines in the letterhead/party-info-box area
instead of the real table, because generic text-line density there looked
similar enough by total pixel count. A second profile (longest
contiguous dark run per row, after tolerantly closing small gaps) can
tell a real ruled line apart from text in that case, since a physical
line has one long unbroken stretch while handwriting is many short
strokes -- but tuning showed no single (profile, threshold) combination
works for both forms; the settings that succeed on one fail outright on
the other. Rather than chase a universal threshold, `detect_row_boundaries`
now tries several strategies and lets the caller pick the best candidate
using its own independent validation (extract_ollama.py checks whether
the resulting header crop actually contains the real header text) --
see `iter_row_boundary_candidates`.
"""

from __future__ import annotations

import cv2
import numpy as np


def _estimate_skew_deg(binary: np.ndarray) -> float:
    """Median angle of near-horizontal line segments via probabilistic
    Hough transform. Small (<15 deg) rotations only -- this corrects
    camera tilt, not the residual per-line warp handled by peak detection
    below."""
    lines = cv2.HoughLinesP(
        binary, 1, np.pi / 360, threshold=150,
        minLineLength=binary.shape[1] // 6, maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in lines.reshape(-1, 4)
    ]
    angles = [a for a in angles if abs(a) < 15]
    return float(np.median(angles)) if angles else 0.0


def _find_peaks(profile: np.ndarray, min_gap: int, min_prominence: float) -> list[int]:
    """Local maxima at least `min_gap' apart, each required to stand out
    from its flanking local minima by `min_prominence' -- tolerant of a
    noisy baseline (handwriting/text also contributes dark pixels)."""
    peaks = []
    n = len(profile)
    for i in range(min_gap, n - min_gap):
        window = profile[i - min_gap:i + min_gap + 1]
        if profile[i] != window.max():
            continue
        left_min = profile[max(0, i - min_gap):i].min() if i > 0 else profile[i]
        right_min = profile[i:min(n, i + min_gap)].min()
        if profile[i] - max(left_min, right_min) >= min_prominence:
            peaks.append(i)
    peaks.sort(key=lambda i: -profile[i])
    kept = []
    for p in peaks:
        if all(abs(p - k) >= min_gap for k in kept):
            kept.append(p)
    return sorted(kept)


def _find_uniform_run(points: list[int], run_len: int) -> list[int] | None:
    """First run of `run_len' consecutive points whose gaps stay in a
    plausible per-row-height range and vary smoothly row-to-row (no
    sudden jumps) -- this is what distinguishes real, evenly-ruled table
    rows from sparser/irregular lines elsewhere on the page (letterhead
    rules, box borders, notes-area rules)."""
    if len(points) < run_len:
        return None
    total_span = points[-1] - points[0]
    approx_row_h = total_span / max(len(points) - 1, 1)
    lo, hi = approx_row_h * 0.3, approx_row_h * 3.0

    for start in range(len(points) - run_len + 1):
        window = points[start:start + run_len]
        gaps = [b - a for a, b in zip(window, window[1:])]
        if not all(lo <= g <= hi for g in gaps):
            continue
        if all(0.6 <= g2 / g1 <= 1.7 for g1, g2 in zip(gaps, gaps[1:])):
            return window
    return None


def detect_column_boundaries(
    image_bytes: bytes, skew_deg: float, y_top_frac: float, y_bottom_frac: float, n_columns: int
) -> list[float] | None:
    """Returns n_columns+1 x-fractions (0.0-1.0 of image width, in the
    DESKEWED frame -- caller must rotate by skew_deg first, same
    convention as detect_row_boundaries) bounding n_columns evenly-spaced
    columns, found the same way as row boundaries: a uniform run among all
    detected vertical ruled-line candidates.

    Restricted to [y_top_frac, y_bottom_frac] (the whole table band,
    header through last row) rather than the full page -- a full-page
    attempt at this was tried previously and was too sparse/unreliable
    (letterhead, party-name box, notes text all add noise); scoping to
    just the ruled table, and aggregating the line signal across every
    row's height (not just one row or just the header), gives each true
    column line many rows worth of pixel evidence to be detected from,
    which also naturally averages out the mild perspective shear that
    makes a single row's columns drift a few pixels from where the header
    row's columns sit."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), skew_deg, 1.0)
    gray_ds = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    binary_ds = cv2.adaptiveThreshold(gray_ds, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)

    y_top = max(0, int(y_top_frac * h))
    y_bottom = min(h, int(y_bottom_frac * h))
    band = binary_ds[y_top:y_bottom, :]

    col_profile = (band > 0).sum(axis=0)
    candidates = _find_peaks(col_profile, min_gap=15, min_prominence=(y_bottom - y_top) * 0.15)

    run = _find_uniform_run(candidates, n_columns + 1)
    if run is None:
        return None
    return [x / w for x in run]


def _row_profile_sum(binary_ds: np.ndarray) -> np.ndarray:
    """Total dark-pixel count per row -- the original profile. Works well
    when the real ruled lines are clearly the darkest/densest rows on the
    page, but confirmed to fail when other page regions (letterhead,
    info boxes) have text-line density in a similar range."""
    return (binary_ds > 0).sum(axis=1)


def _row_profile_longest_run(binary_ds: np.ndarray, close_width: int) -> np.ndarray:
    """Longest contiguous dark run per row, after horizontally closing
    gaps up to close_width pixels -- tolerates a warp-fragmented ruled
    line (a few small gaps) while still distinguishing it from generic
    handwriting/text density, which is many short strokes with bigger,
    more irregular gaps even after the same closing."""
    kernel = np.ones((1, close_width), np.uint8)
    closed = cv2.morphologyEx(binary_ds, cv2.MORPH_CLOSE, kernel) > 0
    h, _ = closed.shape
    result = np.zeros(h, dtype=np.int32)
    for y in range(h):
        row = closed[y]
        if not row.any():
            continue
        diffs = np.diff(row.astype(np.int8))
        starts = list(np.where(diffs == 1)[0] + 1)
        ends = list(np.where(diffs == -1)[0] + 1)
        if row[0]:
            starts = [0] + starts
        if row[-1]:
            ends = ends + [len(row)]
        if starts:
            result[y] = max(e - s for s, e in zip(starts, ends))
    return result


# Tried in order; the caller (extract_ollama.py) validates each
# resulting candidate independently (OCR on the header crop) and keeps
# the best-scoring one, since no single one of these has been found to
# work across differently-scaled/lit photos -- confirmed the settings
# that work on one real form fail outright on another, and vice versa.
_CANDIDATE_CONFIGS: list[tuple[str, dict]] = [
    ("sum", dict(min_gap=25, min_prominence=150)),
    ("longest_run", dict(close_width=20, min_prominence_frac=0.15, min_gap=20)),
    ("longest_run", dict(close_width=10, min_prominence_frac=0.15, min_gap=20)),
    ("longest_run", dict(close_width=10, min_prominence_frac=0.25, min_gap=20)),
    ("longest_run", dict(close_width=20, min_prominence_frac=0.25, min_gap=20)),
    ("longest_run", dict(close_width=30, min_prominence_frac=0.35, min_gap=20)),
]


def iter_row_boundary_candidates(image_bytes: bytes, n_items: int):
    """Yields (y_fractions, skew_deg) for every candidate strategy in
    _CANDIDATE_CONFIGS that finds a valid uniform run of n_items+1
    boundaries. y_fractions are 0.0-1.0 of image height in the DESKEWED
    frame (caller must rotate by skew_deg, same convention as PIL's
    Image.rotate, before cropping). Callers should independently validate
    each candidate (e.g. does the resulting header crop contain the real
    header text?) and pick the best -- see this module's docstring for why
    there's no single reliable choice to make internally."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)

    skew = _estimate_skew_deg(binary)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    gray_ds = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    binary_ds = cv2.adaptiveThreshold(gray_ds, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)

    for kind, params in _CANDIDATE_CONFIGS:
        if kind == "sum":
            profile = _row_profile_sum(binary_ds)
            min_prominence = params["min_prominence"]
        else:
            profile = _row_profile_longest_run(binary_ds, params["close_width"])
            min_prominence = w * params["min_prominence_frac"]
        candidates = _find_peaks(profile, min_gap=params["min_gap"], min_prominence=min_prominence)
        run = _find_uniform_run(candidates, n_items + 1)
        if run is not None:
            yield [y / h for y in run], skew


def _upscale_image_bytes(image_bytes: bytes, scale: float) -> bytes:
    """Upscale raw image bytes by `scale` via cv2, re-encoded as PNG.
    Confirmed necessary 2026-08-17: every _CANDIDATE_CONFIGS strategy
    requires a ~20-25px minimum gap (`min_gap`) between candidate row
    lines -- a low-resolution or tightly-cropped photo can have real ruled
    lines closer together than that in absolute pixels (confirmed on
    `sample 10 crop.jpg`, ~24px/row: zero candidates at native resolution,
    despite the lines being perfectly visible to a human) and
    iter_row_boundary_candidates then returns nothing at all, not just an
    imprecise result. 4x upscaling (~96px/row on that same image) fixed it
    outright -- see CLAUDE.md's dated sections on sample 10."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".png", resized)
    return buf.tobytes()


def iter_row_boundary_candidates_auto(image_bytes: bytes, n_items: int, upscale_factors=(1, 2, 4)):
    """Wraps iter_row_boundary_candidates with an automatic upscale retry:
    tries native resolution, then each further factor in upscale_factors,
    in order -- see _upscale_image_bytes for why a low-resolution image can
    otherwise return zero candidates outright rather than just an
    imprecise one.

    Deliberately does NOT stop at the first scale that yields any
    candidate -- confirmed necessary 2026-08-17: on a real form
    (sample 10.jpeg) 2x upscaling found a geometrically-uniform run of
    lines that was actually wrong (it locked onto the boundary above an
    extra letterhead-style row instead of the real column-header row,
    confirmed by inspecting the resulting header crop directly), and every
    _CANDIDATE_CONFIGS strategy at that scale converged on the same wrong
    answer. A caller validates each candidate independently (does the
    header crop contain real header text? does each row's own content
    match what's expected there?) and rejects a bad one -- stopping the
    upscale retry just because *something* was geometrically found, before
    that validation has a chance to run, would silently give up on a scale
    that might have found the right answer instead. Relies on the caller's
    own early-exit (stop as soon as a fully-validated candidate is found,
    e.g. extract_ollama.py's `if good_count == n_items: break`) to avoid
    wastefully validating every remaining scale once a good one is found --
    this generator is lazy, so an unrequested scale is never even upscaled.

    Yields (y_fractions, skew_deg, working_image_bytes, scale).
    y_fractions/skew_deg are scale-independent (0.0-1.0 fractions /
    degrees, same convention as iter_row_boundary_candidates), but callers
    MUST build every downstream crop from working_image_bytes, not the
    original image_bytes passed in here -- the pixel data itself differs
    whenever scale != 1, and using the wrong one silently reintroduces the
    exact under-resolution problem this function exists to route around."""
    for scale in upscale_factors:
        working_bytes = image_bytes if scale == 1 else _upscale_image_bytes(image_bytes, scale)
        for y_fractions, skew in iter_row_boundary_candidates(working_bytes, n_items):
            yield y_fractions, skew, working_bytes, scale
