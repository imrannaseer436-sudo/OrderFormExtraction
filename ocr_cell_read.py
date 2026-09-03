"""
v6: Stage B via OCR instead of a VLM call, for images where grid.py's row
detection succeeded (a header+row crop is available).

Why: even given a clean, correctly-isolated single-row crop, the local 7B
VLM still misread cells -- confidently hallucinating the form's second
header row as data in blank columns, and shifting real values into the
wrong column (see CLAUDE.md's v5 fresh-run results, 2026-08-03). Both
failure modes are inherent to asking a generative model to LOCATE and READ
every cell in one pass. PaddleOCR instead only reports text where ink
actually exists (no hallucination possible for a blank cell) and each
detected digit carries its own pixel position, which is matched to the
nearest header column -- turning "read 13 columns while holding your
position in your head" into "recognize this isolated text box," a much
easier task for either a VLM or an OCR engine, but especially reliable for
a non-generative recognizer.

Column assignment is order-preserving (a digit's left-to-right RANK is
reliable even though the photo's mild perspective warp shifts absolute
pixel positions row to row), solved via a small DP (monotonic assignment,
not independent nearest-neighbor per digit) so one ambiguous digit can't
get tie-broken onto the wrong header independent of its neighbors.

v6.1 (2026-08-03) adds two things, both requiring every row of an image to
be processed together rather than independently (see `read_rows`):

- **Drift self-calibration**: the photographed page's mild perspective
  means a column's true pixel position drifts further from the (fixed)
  header crop's position the deeper a row sits in the table -- confirmed
  empirically on a real form (~0px drift near the header, ~50-58px by the
  table's last row). Two independent attempts to re-derive this
  geometrically (fitting boundary lines via Hough; a Canny+contour
  document-scanner corner detection) were tried and both failed on a real
  photo -- see CLAUDE.md. Instead of a fresh geometric measurement, this
  self-calibrates FROM the same image's own confidently-correct rows: rows
  where the initial (uncorrected) DP assignment is already a tight fit
  give a direct, trustworthy measurement of the residual (actual digit
  position minus assigned header position) at that row's depth. A linear
  fit across those residuals predicts the correction for deeper, less
  confident rows, then assignment is redone with corrected header
  positions. No new CV detection, reuses data already computed.
- **Item-name cross-check**: alongside digits, the leftmost sufficiently-
  long non-digit text token in each row's data band is also returned
  (`item_name_ocr`). The caller (extract_ollama.py) uses this to catch
  cases where Stage A's VLM call misreads a short alphanumeric item code
  (confirmed: "F.G-3005"/"F.G-3025" misread as "F.G1-3005"/"F.G1-302S") --
  OCR reads these correctly since it's a direct pixel-level recognition,
  not a holistic VLM read.

Falls back to the VLM per-row call (extract_ollama.stage_b_row) whenever
grid detection didn't find a clean set of row boundaries for an image, or
an individual row's OCR can't even read its own crop's header labels.

v6.2 (2026-08-03) adds, in `ocr_row`: a digit-lookalike-letter recovery
(`_try_digit_correct`) for handwritten digits OCR misread as a similar
letter (confirmed: "S"->5, "LO"->10, "so"->50, previously silently
dropped), and a left-side `x_floor` mirroring the existing right-side
`x_cutoff` (confirmed: a misread ditto-mark produced a spurious digit in
the item-name region on a real form; the left cutoff excludes that
region from digit consideration entirely).

2026-09-01: ported `_split_merged_qty_token` from extract_ollama_cloud.py
(built there 2026-08-19) -- PaddleOCR occasionally fuses two adjacent
handwritten numbers into one detection (e.g. two "30"s read as one
"3030" token), which `ocr_row` previously accepted unguarded via a bare
`int(text)` with no upper bound. Now re-segments a token this large back
into its two real values when there's exactly one valid, unambiguous
split; otherwise drops it, matching this module's existing "leave it out
rather than invent a value" principle. `extract_ollama_cloud.py` now
imports this shared definition instead of keeping its own duplicate.
"""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from schema import LETTER_SIZE_CODES

# A row's initial (uncorrected) DP fit below this average per-digit
# distance (px) is trusted as a genuine, undrifted measurement point for
# calibration -- well under half a column's spacing (~62px on the form
# this was tuned against), so a row this tight almost certainly landed on
# the right header, not just the closest available one.
_CONFIDENT_AVG_COST_PX = 12.0
# Need calibration points spanning at least this fraction of the table's
# height before trusting an extrapolated slope -- a tiny y-range invites
# a wild extrapolation to the far (uncalibrated) rows. Confirmed against a
# real form (2026-08-03): only the first ~4 rows are typically tight
# enough to qualify as confident, spanning ~0.10 of the table -- a first
# attempt at 0.15 was too strict and silently disabled calibration
# entirely; 0.08 includes that real, useful signal while still requiring
# more than 2 near-adjacent rows.
_MIN_CALIBRATION_Y_SPREAD = 0.08

# Confirmed on real handwriting (2026-08-03): OCR occasionally reads a
# handwritten digit as a similar-looking letter instead -- "S"->5, "LO"->10,
# "so"->50 were all observed directly in rows.txt, previously silently
# dropped since they failed the pure-isdigit() check. Deliberately a small,
# conservative table (not every letter that resembles a digit) since this
# recovers values that were otherwise being dropped, not overriding an
# existing read -- a wrong substitution is worse than a dropped cell, so
# only near-unambiguous look-alikes are included.
_DIGIT_LOOKALIKES = {"O": "0", "o": "0", "S": "5", "s": "5", "L": "1", "l": "1", "I": "1"}

# How close (px) a later row's unclaimed overflow digit must sit to an
# earlier row's discovered overflow-label x to be treated as belonging to
# that same label, when propagating a label written once across several
# rows (see read_rows()). Matches the scale of other cross-row x-position
# tolerances already used in this project's OCR work (e.g. the cloud
# pipeline's DUP_LABEL_TOLERANCE_PX = 15.0) rather than a per-row spacing
# fraction, since both x-positions being compared here already come from
# the same shared, unscaled header-crop coordinate frame.
_OVERFLOW_PROPAGATION_TOLERANCE_PX = 15.0

_ocr_singleton = None


def _try_digit_correct(text: str) -> int | None:
    """Recover a quantity from a token OCR misread as digit-lookalike
    letters. Scoped to short tokens (real quantities on this form are 1-3
    digits) and only applied to candidates already inside the plausible
    size-column x-range (see ocr_row) -- not attempted on item-name-region
    text, which lives well to the left of that range."""
    if not (1 <= len(text) <= 3):
        return None
    corrected = "".join(_DIGIT_LOOKALIKES.get(c, c) for c in text)
    if not corrected.isdigit():
        return None
    qty = int(corrected)
    return qty if qty > 0 else None


def _split_merged_qty_token(text: str, box: list[float]) -> list[tuple[int, float]] | None:
    """Re-segments a single OCR token that's almost certainly two adjacent
    quantity numbers PaddleOCR fused into one detection (e.g. two
    handwritten "30"s next to each other read as one token "3030") into two
    separate (qty, x_center) candidates, instead of a bare `int(text)`
    silently accepting the fused value outright (ported from
    extract_ollama_cloud.py, built there 2026-08-19).

    Tries every internal split point and keeps it ONLY if EXACTLY ONE split
    point yields two valid (1-500, no spurious leading zero) integers --
    checked at every position rather than just the middle, since a merge of
    two differently-sized numbers (e.g. "5"+"30" -> "530") won't split
    evenly. Requiring uniqueness matters because a token like "1010" splits
    validly at two different points (10|10 and 101|0 -- the latter rejected
    by the leading-zero-on-neither-side check, but "255" would split at both
    2|55 and 25|5 with nothing to prefer one over the other) -- an ambiguous
    split is worse than no value at all, matching this module's existing
    "leave it out rather than invent a value" principle. On a unique split,
    the box's pixel width is divided proportionally by character position to
    estimate each half's own x-center, so downstream digit-to-header
    assignment (which keys entirely off x-position) still works on the
    recovered values."""
    if not text.isdigit() or len(text) < 2:
        return None
    x0, x1 = box[0], box[2]
    width = x1 - x0
    valid_splits = []
    for i in range(1, len(text)):
        left, right = text[:i], text[i:]
        if (len(left) > 1 and left.startswith("0")) or (len(right) > 1 and right.startswith("0")):
            continue
        lv, rv = int(left), int(right)
        if 1 <= lv <= 500 and 1 <= rv <= 500:
            valid_splits.append((i, lv, rv))
    if len(valid_splits) != 1:
        return None
    i, lv, rv = valid_splits[0]
    split_x = x0 + width * (i / len(text))
    return [(lv, (x0 + split_x) / 2), (rv, (split_x + x1) / 2)]


def _add_windows_gpu_dll_dirs() -> None:
    """Make GPU-inference DLLs (cuDNN, CUDA runtime) discoverable even when
    this process's own PATH is stale.

    Confirmed necessary on this machine (2026-09-01): installing the CUDA
    Toolkit updates the system PATH, but a process tree that was already
    running when the installer ran doesn't see the update. Confirmed by
    direct testing that `os.add_dll_directory` (Win32 AddDllDirectory)
    does NOT fix this for Paddle specifically -- ctypes.WinDLL picked up
    an AddDllDirectory-registered path fine, but Paddle's own native
    dynamic loader still failed to find cublasLt64_12.dll even with the
    same directory registered, meaning its loader calls plain
    LoadLibrary (classic search order: app dir, System32, PATH -- NOT the
    AddDllDirectory-extended list, which only LoadLibraryEx with the
    LOAD_LIBRARY_SEARCH_* flags honors). Prepending to the process's own
    `os.environ["PATH"]` instead is what actually worked, confirmed via a
    real predict() call. Called every time get_ocr() builds the pipeline
    rather than relied on once at import time; cheap and idempotent. No-op
    on any non-Windows platform or if a directory isn't found, since GPU
    is opportunistic here -- CPU remains the fallback either way (see
    _paddle_device)."""
    if sys.platform != "win32":
        return

    candidates: list[Path] = []
    try:
        import nvidia.cudnn
        cudnn_bin = Path(nvidia.cudnn.__path__[0]) / "bin"
        if cudnn_bin.is_dir():
            candidates.append(cudnn_bin)
    except Exception:
        pass

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        candidates.append(Path(cuda_path) / "bin")
    else:
        # CUDA_PATH itself may be unset in this stale-env process even
        # though the installer set it system-wide -- fall back to the
        # standard install location, newest version first.
        toolkit_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        for version_dir in sorted(toolkit_root.glob("v*"), reverse=True):
            candidates.append(version_dir / "bin")
            break

    existing_path = os.environ.get("PATH", "")
    for d in candidates:
        if not d.is_dir():
            continue
        try:
            os.add_dll_directory(str(d))  # cheap, harmless belt-and-suspenders
        except (FileNotFoundError, OSError):
            pass
        d_str = str(d)
        if d_str not in existing_path:
            existing_path = d_str + os.pathsep + existing_path
    os.environ["PATH"] = existing_path


def _paddle_device() -> str:
    """Select CUDA/GPU when available and fall back to CPU otherwise.

    This keeps the existing CPU-safe behavior for environments without a
    compatible NVIDIA build while automatically taking advantage of a
    recently-added GPU on machines where Paddle has CUDA support.
    """
    try:
        import paddle
    except Exception:
        return "cpu"

    try:
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"


def get_ocr():
    """Lazily construct and cache the PaddleOCR pipeline.

    Prefer GPU when the local Paddle build supports CUDA; otherwise keep the
    known-safe CPU fallback. enable_mkldnn=False remains on both paths to
    avoid the oneDNN NotImplementedError seen on some builds here.
    """
    global _ocr_singleton
    if _ocr_singleton is None:
        _add_windows_gpu_dll_dirs()
        from paddleocr import PaddleOCR

        device = _paddle_device()
        _ocr_singleton = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
            enable_mkldnn=False,
        )
    return _ocr_singleton


def _monotonic_assign(digit_xs: list[float], header_xs: list[float]) -> tuple[dict[int, int], float]:
    """Best order-preserving assignment of digits (sorted by x) to a
    subset of headers (sorted by x), minimizing total |distance|. Plain
    distance-minimizing DP, no contiguity bias -- an interior-gap penalty
    (to prefer this form's typically-contiguous data block) was tried and
    tuned against a real scanned form: it fixed a couple of rows but
    pulled an already-correct anchor away from its true header on others,
    trying to preserve contiguity elsewhere. The plain version is the
    more predictable, safer default.

    Returns (assignment, avg_cost) -- avg_cost (total distance / digit
    count) is used by read_rows() as a per-row confidence signal for drift
    calibration: a tight fit is trustworthy evidence of the real drift at
    that row's depth, a loose one probably already picked the wrong header
    for at least one digit.

    dp[i][j] = min cost to assign digits 0..i where digit i lands on
    header j exactly (need the exact header, not just "some header <= j",
    to know the gap to the next digit -- unused here since there's no gap
    penalty, but kept in this shape in case that's revisited)."""
    m, n = len(digit_xs), len(header_xs)
    INF = float("inf")
    dp = [[INF] * n for _ in range(m)]
    prev_j: list[list[int | None]] = [[None] * n for _ in range(m)]

    for j in range(n):
        dp[0][j] = abs(digit_xs[0] - header_xs[j])

    for i in range(1, m):
        for j in range(i, n):
            for jp in range(i - 1, j):
                cost = dp[i - 1][jp] + abs(digit_xs[i] - header_xs[j])
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    prev_j[i][j] = jp

    j_best = min(range(n), key=lambda j: dp[m - 1][j])
    assignment: dict[int, int] = {}
    i, j = m - 1, j_best
    while i >= 0:
        assignment[i] = j
        j = prev_j[i][j] if i > 0 else None
        i -= 1
    return assignment, dp[m - 1][j_best] / m


def _split_by_largest_gap(ys: list[float]) -> tuple[list[int], list[int]]:
    """Split indices into a low-y and high-y group at the largest gap in
    sorted y-order -- used to separate the header band's two stacked text
    lines (primary size number on top, secondary style-code number
    directly below it) without a fixed pixel threshold, since crops vary
    in scale/DPI across photos. Returns (low_group, high_group) as index
    lists into the original (unsorted) input."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    if len(order) < 2:
        return order, []
    gap_i = max(range(len(order) - 1), key=lambda i: ys[order[i + 1]] - ys[order[i]])
    return order[:gap_i + 1], order[gap_i + 1:]


def _detect_stacked_labels(
    texts: list[str], boxes: list[list[float]], scores: list[float], header_h_px: int,
    score_threshold: float, x_floor: float, x_cutoff: float, spacing: float,
) -> tuple[list[tuple[str, float, int]], set[int], list[tuple[float, int]]]:
    """Finds "label stacked directly above its own quantity" pairs in a
    row's own data band -- the real-world convention this form's writers
    use both for an unprinted overflow size column past the last printed
    header (e.g. a handwritten "110" past this form's last real header,
    because there's no room to add a real column) and for letter-coded
    clothing sizes sitting INSIDE the normal printed columns (e.g.
    S/M/L/XL/XXL when a product's real catalog sizing has no relation to
    the form's generic numeric grid) -- both are the same physical shape,
    just at different x-positions, per CLAUDE.md's 2026-08-14 observation
    that a genuine overflow column and letter-coded sizes are one real-
    world convention, not two.

    A candidate label is only accepted with a genuine companion quantity
    below it -- a bare stray digit or letter near the margin isn't enough
    evidence on its own, matching this module's existing "leave it out
    rather than invent a value" principle (already used for ambiguous
    digit-lookalikes).

    Returns (overflow_items, consumed_indices, unclaimed_overflow_digits).
    overflow_items is (label_text, label_x_center, qty) -- pairs found
    entirely within THIS row. consumed_indices lets the caller skip these
    token indices in the normal digit-parsing loop -- required, not just
    tidy: a letter-size candidate's companion digit sits INSIDE the normal
    header x-range and would otherwise ALSO be read as a normal quantity
    there (double-counted against the wrong real header), and a letter
    label itself (e.g. "S") could otherwise be misread as a digit via
    _try_digit_correct's lookalike table (S->5, L->1) -- this detection
    must run, and its results be respected, before that fallback gets a
    chance to fire on the same token. unclaimed_overflow_digits is every
    digit found PAST the last real header with no label directly above it
    in THIS row -- e.g. a row relying on an overflow label written only
    once, above an earlier row (confirmed real: sample 5.jpeg's
    `Fairlady Plain` has its own "110" quantity but no "110" label of its
    own, relying on the label above `Fairlady Print`). read_rows() matches
    these against a label position already discovered on an earlier row of
    the same image -- deliberately only done for the numeric-overflow
    region (past x_cutoff), not the letter-size region (which overlaps the
    normal header x-range and would risk the same double-counting problem
    described above if propagated the same way; see read_rows())."""
    overflow_margin = spacing  # one extra column-width past the last real header
    candidates: list[tuple[int, str, list[float]]] = []
    for i, (text, box, score) in enumerate(zip(texts, boxes, scores)):
        if box[1] < header_h_px or score < score_threshold:
            continue
        cx = (box[0] + box[2]) / 2
        upper = text.upper()
        is_numeric_overflow = text.isdigit() and 1 <= len(text) <= 3 and x_cutoff < cx <= x_cutoff + overflow_margin
        is_letter_size = upper in LETTER_SIZE_CODES and x_floor <= cx <= x_cutoff + overflow_margin
        if is_numeric_overflow or is_letter_size:
            candidates.append((i, text if is_numeric_overflow else upper, box))

    overflow_items: list[tuple[str, float, int]] = []
    consumed: set[int] = set()
    for label_idx, label_text, label_box in candidates:
        label_cx = (label_box[0] + label_box[2]) / 2
        best: tuple[int, list[float], int] | None = None
        for j, (text, box, score) in enumerate(zip(texts, boxes, scores)):
            if j == label_idx or j in consumed or box[1] < header_h_px or score < score_threshold:
                continue
            if not text.isdigit():
                continue
            cx = (box[0] + box[2]) / 2
            if abs(cx - label_cx) > spacing * 0.25 or box[1] <= label_box[1]:
                continue  # not below the label, or too far off in x -- not a companion
            qty = int(text)
            if not (1 <= qty <= 500):
                continue
            if best is None or box[1] < best[1][1]:  # closest-below wins
                best = (j, box, qty)
        if best is not None:
            j, _, qty = best
            overflow_items.append((label_text, label_cx, qty))
            consumed.add(label_idx)
            consumed.add(j)

    # Deliberately does not exclude a token just because it was a
    # candidate label above -- a numeric token past x_cutoff that never
    # found its own companion is functionally just a bare digit (e.g.
    # sample 5.jpeg's `Fairlady Plain` quantity looks exactly like a
    # label candidate in isolation, with nothing else in its own row's
    # crop) -- eligible to be matched against a label discovered on an
    # earlier row instead. Only tokens actually CONSUMED (a genuine local
    # pairing) are excluded.
    unclaimed_overflow_digits: list[tuple[float, int]] = []
    for i, (text, box, score) in enumerate(zip(texts, boxes, scores)):
        if i in consumed:
            continue
        if box[1] < header_h_px or score < score_threshold or not text.isdigit():
            continue
        cx = (box[0] + box[2]) / 2
        if not (x_cutoff < cx <= x_cutoff + overflow_margin):
            continue
        qty = int(text)
        if 1 <= qty <= 500:
            unclaimed_overflow_digits.append((cx, qty))

    return overflow_items, consumed, unclaimed_overflow_digits


_NO_STACKED_LABELS = {"overflow_items": [], "unclaimed_overflow_digits": []}


def ocr_row(
    stitched_png_bytes: bytes, header_h_px: int, headers: list[str], score_threshold: float
) -> tuple[dict[str, float] | None, list[tuple[float, int]], str | None, str, dict]:
    """OCR one stitched header+row crop. Returns (header_x, digit_items,
    item_name_ocr, debug, stacked_labels) where header_x is None if not
    every header label was found (caller should fall back to the VLM for
    this row), digit_items is a sorted list of (x_center, qty) already
    filtered by score/cutoff -- NOT yet assigned to headers, since that
    needs the (possibly drift-corrected) header positions from
    read_rows() -- and stacked_labels is this row's own detected stacked
    label/quantity pairs plus any unclaimed overflow digits (see
    _detect_stacked_labels for both), not yet merged with a label
    discovered on an earlier row of the same image (that cross-row
    propagation happens in read_rows(), which sees every row together)."""
    if not headers:
        # No shared header row to align against at all (e.g. a free-form
        # page -- see extract_ollama.py's own guard against calling this
        # in that case). Defense in depth: the DP/spacing math below
        # divides by len(headers)-1 and indexes into an empty list if
        # headers is empty, so this must be checked before any of that
        # runs, not just relied on at the call site.
        return None, [], None, "(no headers to align against)", dict(_NO_STACKED_LABELS)

    image = Image.open(io.BytesIO(stitched_png_bytes)).convert("RGB")
    arr = np.array(image)[:, :, ::-1]  # RGB -> BGR, what cv2/Paddle expect

    ocr = get_ocr()
    result = list(ocr.predict(arr))
    if not result:
        return None, [], None, "(no OCR result)", dict(_NO_STACKED_LABELS)
    res = result[0]
    texts = res["rec_texts"]
    boxes = res["rec_boxes"].tolist()
    scores = res["rec_scores"]

    debug = ", ".join(
        f"{t}@{(b[0] + b[2]) / 2:.0f}(y{b[1]}-{b[3]},s{s:.2f})" for t, b, s in zip(texts, boxes, scores)
    )

    header_x: dict[str, float] = {}
    for text, box in zip(texts, boxes):
        if text in headers and box[3] <= header_h_px and text not in header_x:
            header_x[text] = (box[0] + box[2]) / 2

    if len(header_x) < len(headers):
        # Exact-text matching didn't find every header -- confirmed on a
        # lower-resolution real photo (2026-08-04, 998x1138 vs the
        # original test form's 1448x1490) this can happen even though the
        # header row itself is fully legible: PaddleOCR misread a single
        # digit ("60" -> "80"), which collided with the real "80" token in
        # this dict (first occurrence wins, by x-order "60"'s position),
        # silently discarding BOTH -- the real "80" is skipped as
        # "already found" under the wrong x, and "60" never appears at
        # all. Recover positionally instead of by value: `headers` is
        # known, ordered, and evenly spaced (from Stage A's size_headers),
        # so if the header band's TOP line (the primary size-number row,
        # not the secondary style-code line stacked directly below it on
        # this form) has exactly the right COUNT of digit-like tokens,
        # their left-to-right order alone is enough to assign them
        # correctly even when individual digits were misread -- the same
        # "order over exact value" principle already used for row-boundary
        # and digit-to-header assignment elsewhere in this module.
        # header_h_px (from build_header_crop's avg_row_h_f*1.3) isn't
        # guaranteed to be tall enough to include BOTH the primary
        # size-number line and the secondary style-code line stacked below
        # it -- confirmed on a real crop (2026-08-04) it was tall enough
        # for only the primary line (header_h_px=46px cut the secondary
        # line's boxes, which sit at y~29-64, entirely out). So `numeric`
        # here may hold just one line's worth of tokens, not two --
        # splitting it into a "top" and "bottom" group unconditionally
        # would carve that single line in half at whatever its largest
        # internal y-jitter gap happens to be, never matching len(headers).
        # Only split when there's evidence of a second line (roughly 2x
        # the expected count); if the count already matches exactly, it's
        # already just the one line we want.
        numeric = [
            (box, (box[0] + box[2]) / 2, box[1])
            for text, box in zip(texts, boxes)
            if box[3] <= header_h_px and text.isdigit() and 1 <= len(text) <= 3
        ]
        if len(numeric) == len(headers):
            top_line = sorted(numeric, key=lambda n: n[1])
        elif len(numeric) > len(headers):
            top_idx, _ = _split_by_largest_gap([n[2] for n in numeric])
            top_line = sorted((numeric[i] for i in top_idx), key=lambda n: n[1])
        else:
            top_line = []
        if len(top_line) == len(headers):
            header_x = {h: n[1] for h, n in zip(headers, top_line)}

    if len(header_x) < len(headers):
        return None, [], None, f"(only found {len(header_x)}/{len(headers)} headers) " + debug, dict(_NO_STACKED_LABELS)

    header_xs_sorted = [header_x[h] for h in headers]  # `headers` is already left-to-right
    spacing = (header_xs_sorted[-1] - header_xs_sorted[0]) / (len(header_xs_sorted) - 1)
    x_cutoff = header_xs_sorted[-1] + spacing * 0.5  # excludes the trailing "Total Dozen" column
    # Mirrors x_cutoff on the left -- excludes stray marks in the
    # item-name/Style/ditto-mark region from ever being mistaken for a
    # size value (confirmed: a misread ditto-mark produced a spurious
    # digit here on a real form).
    x_floor = header_xs_sorted[0] - spacing * 0.5

    overflow_items, consumed_indices, unclaimed_overflow_digits = _detect_stacked_labels(
        texts, boxes, scores, header_h_px, score_threshold, x_floor, x_cutoff, spacing
    )

    digit_items = []
    name_candidates: list[tuple[float, str]] = []
    ratio_marker_notes: list[str] = []
    for idx, (text, box, score) in enumerate(zip(texts, boxes, scores)):
        if idx in consumed_indices:
            continue
        if box[1] < header_h_px or score < score_threshold:
            continue
        cx = (box[0] + box[2]) / 2

        qty = int(text) if text.lstrip("-").isdigit() else _try_digit_correct(text)
        if qty is not None and qty > 500 and text.isdigit():
            # Almost certainly two adjacent quantities PaddleOCR merged
            # into one detection (see _split_merged_qty_token) rather than
            # a real quantity -- this form's real values never reach 500.
            # Recover both real values when there's exactly one safe split;
            # otherwise drop the token entirely rather than keep an
            # implausible value or guess which half is real.
            for split_qty, split_cx in (_split_merged_qty_token(text, box) or []):
                if x_floor <= split_cx <= x_cutoff:
                    digit_items.append((split_cx, split_qty))
        elif qty is not None and qty > 0 and x_floor <= cx <= x_cutoff:
            digit_items.append((cx, qty))
        elif qty is None and len(text) >= 2:
            # candidate for the item-name cross-check -- leftmost
            # sufficiently-long non-digit token in the data row. Item name
            # is always the leftmost column on this form, so this doesn't
            # need to explicitly exclude the Style/type-code token.
            name_candidates.append((box[0], text))
        elif qty is None and len(text) <= 2 and x_floor <= cx <= x_cutoff:
            # Some forms (e.g. a typed grid using "*" for "per standard
            # ratio" instead of a digit) mark a cell with a short non-
            # digit symbol rather than a real quantity. PaddleOCR often
            # can't read the glyph itself reliably (seen misread as ".",
            # a stray punctuation mark, or dropped at low confidence), so
            # this doesn't try to identify WHICH symbol it was -- it just
            # surfaces that SOMETHING non-numeric sits in a real quantity
            # cell, so a human reviewer sees it in rows.txt instead of an
            # unexplained gap, rather than silently dropping it (already
            # safe -- never invents a fake number -- but previously
            # invisible).
            nearest_header = min(range(len(header_xs_sorted)), key=lambda k: abs(header_xs_sorted[k] - cx))
            ratio_marker_notes.append(f"{headers[nearest_header]}:marker({text!r})")
    digit_items.sort()
    item_name_ocr = min(name_candidates)[1] if name_candidates else None
    if ratio_marker_notes:
        debug = debug + " | ratio markers: " + ", ".join(ratio_marker_notes)

    stacked_labels = {"overflow_items": overflow_items, "unclaimed_overflow_digits": unclaimed_overflow_digits}
    return header_x, digit_items, item_name_ocr, debug, stacked_labels


def count_headers_found(header_png_bytes: bytes, headers: list[str], score_threshold: float = 0.5) -> int:
    """How many of `headers` are recognized as text anywhere in this crop.
    Used by extract_ollama.py as a sanity check BEFORE committing to
    grid-based row cropping for a whole image: confirmed on a real form
    (2026-08-0X, denser/differently-laid-out than the original test form)
    that grid.py's row-boundary detection can find a "clean" run of
    evenly-spaced lines in entirely the wrong part of the page (there: the
    letterhead/party-info-box area, not the actual item table) while still
    returning successfully. Every row's header crop then shows the wrong
    region, so every row's OCR fails identically -- silently building 18
    per-row crops around a wrong position and only discovering the problem
    one row at a time is both slow and, worse, the VLM fallback used to
    reuse that same wrong crop instead of the real image. Checking the
    header position ONCE, before building any row crops, catches this
    up front so the whole image can fall back to full-image reading
    instead."""
    image = Image.open(io.BytesIO(header_png_bytes)).convert("RGB")
    arr = np.array(image)[:, :, ::-1]
    ocr = get_ocr()
    result = list(ocr.predict(arr))
    if not result:
        return 0
    res = result[0]
    found = {t for t, s in zip(res["rec_texts"], res["rec_scores"]) if s >= score_threshold}
    return sum(1 for h in headers if h in found)


def item_alignment_ok(item_name_ocr: str | None, expected_item: str) -> bool:
    """Does an already-OCR'd item-name reading plausibly match the item
    Stage A expects at this row position? Split out from row_alignment_ok
    (pure comparison, no OCR call) so a caller that's already run
    ocr_row() for this crop -- e.g. extract_ollama.py's candidate-scoring
    loop -- doesn't have to pay for a second, redundant OCR call (~14s on
    this machine's CPU-only PaddleOCR build, confirmed 2026-08-04) just to
    validate a result it already has.

    Deliberately a forgiving substring/word-overlap match, not exact
    equality -- Stage A (VLM) and OCR frequently disagree on exact
    spelling/wording for the same real item (confirmed repeatedly this
    project), so requiring an exact match would reject correctly-aligned
    rows just as often as it catches misaligned ones. Returns True if OCR
    found no text at all (a separate, unrelated problem this check isn't
    trying to catch) or if the expected name has no long alpha word to
    check against (e.g. a bare code like "F.G-3005") -- but NOT merely
    because the OCR'd text lacks one: a first version did that and missed
    a real misalignment, because the WRONG row's item happened to be a
    short code ("MM K4532") with no 3+-letter word either, so the "can't
    check, allow it" fallback let a genuinely wrong row through."""
    if not item_name_ocr:
        return True
    expected_words = [w.upper() for w in re.findall(r"[A-Za-z]{3,}", expected_item)]
    if not expected_words:
        return True
    return any(w in item_name_ocr.upper() for w in expected_words)


def row_alignment_ok(row_png_bytes: bytes, header_h_px: int, headers: list[str], expected_item: str) -> bool:
    """Does this row crop's own OCR'd item-name text plausibly match the
    item Stage A expects at this position? Used alongside
    count_headers_found to validate a candidate grid: confirmed on a real
    form that a header crop can be correctly positioned while individual
    ROW boundaries still drift (missing/extra candidate lines mid-table
    shift every row after them a position or two) -- e.g. row 17's crop
    showed row 14's content. The header check alone can't catch that,
    since it only looks at the one fixed header position, never at
    whether a specific row index landed on the right row.

    Convenience wrapper around item_alignment_ok() for callers that don't
    already have an ocr_row() result to reuse -- see that function's
    docstring for the matching logic itself."""
    _, _, item_name_ocr, _, _ = ocr_row(row_png_bytes, header_h_px, headers, score_threshold=0.5)
    return item_alignment_ok(item_name_ocr, expected_item)


def read_rows(
    rows: list[tuple[bytes, int, float]], headers: list[str], score_threshold: float = 0.5,
    precomputed: dict[int, tuple] | None = None,
) -> list[tuple[dict[str, int], str, str | None]]:
    """Process every row-crop of ONE image together -- required (not just
    convenient) for drift self-calibration, which needs to see every row's
    initial fit before correcting any of them.

    `rows` is (stitched_png_bytes, header_h_px, row_mid_y_fraction) per
    row, in row order; row_mid_y_fraction is the row's vertical position
    as a 0-1 fraction of the table (e.g. from grid.py's boundaries), used
    only as the calibration's x-axis (relative depth), not as a physical
    unit.

    `precomputed`, if given, maps row index -> an ocr_row() result already
    computed for that exact crop (e.g. by extract_ollama.py's candidate-
    scoring pass, which OCRs every row of a candidate to check alignment
    before it's known to be the winner). Without this, the winning
    candidate's rows get OCR'd twice -- once to validate the candidate,
    again here to actually read it -- for no benefit, since the crop
    pixels are identical both times. Confirmed costly, not just
    theoretically wasteful: ~14s per OCR call on this machine's CPU-only
    PaddleOCR build (2026-08-04), so this roughly halves the OCR cost of
    the winning candidate's rows.

    Returns, per row, (quantities, debug_string, item_name_ocr).
    quantities is {} with a debug string starting "(only found" if OCR
    couldn't even read this row's own header labels -- the caller should
    fall back to the VLM for that specific row."""
    precomputed = precomputed or {}
    parsed = [
        precomputed[i] if i in precomputed else ocr_row(png, h, headers, score_threshold)
        for i, (png, h, _) in enumerate(rows)
    ]
    row_ys = [y for _, _, y in rows]

    calib_points: list[tuple[float, float]] = []
    for (header_x, digit_items, _, _, _), row_y in zip(parsed, row_ys):
        if header_x is None or not digit_items:
            continue
        header_xs_sorted = [header_x[h] for h in headers]
        digit_xs = [d[0] for d in digit_items]
        assignment, avg_cost = _monotonic_assign(digit_xs, header_xs_sorted)
        if avg_cost < _CONFIDENT_AVG_COST_PX:
            residuals = [digit_xs[k] - header_xs_sorted[assignment[k]] for k in assignment]
            calib_points.append((row_y, sum(residuals) / len(residuals)))

    correction = None
    if len(calib_points) >= 3:
        ys = [p[0] for p in calib_points]
        if max(ys) - min(ys) >= _MIN_CALIBRATION_Y_SPREAD:
            a, b = np.polyfit(ys, [p[1] for p in calib_points], 1)
            correction = (float(a), float(b))

    # Numeric-overflow labels (e.g. a handwritten "110" past the last
    # printed header) discovered on any row so far, in row order --
    # carried forward so a LATER row with no label of its own can still
    # pick up its own value at that x (see _detect_stacked_labels'
    # docstring: sample 5.jpeg's `Fairlady Plain` relies on the label
    # written once, above `Fairlady Print`). Deliberately raw-pixel
    # matching, independent of the drift-calibration `correction` above --
    # both the label's discovered x and a later row's unclaimed digit x
    # come from the same shared, unscaled header-crop coordinate frame, so
    # no correction is needed here even though one is applied to the real
    # numeric headers below. Letter-size labels are NOT tracked here --
    # see _detect_stacked_labels for why that propagation isn't safe yet.
    discovered_numeric_labels: dict[str, float] = {}

    results: list[tuple[dict[str, int], str, str | None]] = []
    for (header_x, digit_items, item_name_ocr, debug, stacked_labels), row_y in zip(parsed, row_ys):
        extra_quantities: dict[str, int] = {}
        for label, label_x, qty in stacked_labels["overflow_items"]:
            extra_quantities[label] = qty
            if label.isdigit():
                discovered_numeric_labels.setdefault(label, label_x)

        unclaimed = list(stacked_labels["unclaimed_overflow_digits"])
        if unclaimed and discovered_numeric_labels:
            for label, label_x in discovered_numeric_labels.items():
                if label in extra_quantities:
                    continue
                best_k = min(
                    (k for k, (ux, _) in enumerate(unclaimed) if abs(ux - label_x) <= _OVERFLOW_PROPAGATION_TOLERANCE_PX),
                    key=lambda k: abs(unclaimed[k][0] - label_x),
                    default=None,
                )
                if best_k is not None:
                    extra_quantities[label] = unclaimed[best_k][1]
                    del unclaimed[best_k]

        if header_x is None:
            results.append(({}, debug, item_name_ocr))
            continue
        header_xs_sorted = [header_x[h] for h in headers]
        if correction is not None:
            a, b = correction
            shift = a * row_y + b
            header_xs_sorted = [x + shift for x in header_xs_sorted]
        if not digit_items:
            results.append((dict(extra_quantities), debug, item_name_ocr))
            continue
        digit_xs = [d[0] for d in digit_items]
        assignment, _ = _monotonic_assign(digit_xs, header_xs_sorted)
        quantities = {headers[assignment[k]]: digit_items[k][1] for k in assignment}
        quantities.update(extra_quantities)
        results.append((quantities, debug, item_name_ocr))
    return results
