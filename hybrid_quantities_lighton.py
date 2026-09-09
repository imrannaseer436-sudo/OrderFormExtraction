"""
Experimental alternative to extract_ollama_cloud.py's PaddleOCR-based
`_hybrid_ocr_quantities` -- same job (an independent, second reading of
each row's quantities to reconcile against mistral's own main-call
reading), different mechanism: instead of a classical detect-then-
recognize OCR pass feeding a pixel-position column-assignment algorithm,
this sends the whole page (or a table-only crop) to a second VLM
(`maternion/LightOnOCR-2:1b`, local Ollama) and parses its own assembled
table back out.

NOT WIRED INTO extract_ollama_cloud.py OR THE REVIEW APP. This is a
standalone module for direct comparison against the production
PaddleOCR path (see compare_hybrid_backends.py) -- CLAUDE.md's
documented production behavior is unchanged until this earns that
switch on real evidence, not just this module existing.

Do we still need mistral? Yes. This module takes the exact same
`extracted: ExtractedForm` input as `_hybrid_ocr_quantities` and reuses
it the same way: the item skeleton (names, order, count), the printed
size_headers list, and each row's own printed row_total (used for
reconciliation, not re-derived from LightOnOCR's own reading -- see
`_reconcile_hybrid_with_vlm`) all still come from mistral's main call.
LightOnOCR-2 only replaces PaddleOCR's specific role: an independent
re-read of the quantity CELLS. It is not a schema-structuring model --
party_name/order_no/order_date, style/type codes, and the mistral-only
struck_out/date_present/letter_sizes flags all still need mistral (or
an equivalent VLM) regardless of which quantity backend wins.

Session findings this module encodes (see HISTORY.md for the full
narrative once logged):
- The real fix for LightOnOCR-2's mid-table truncation was Ollama's
  `num_ctx` (raised to 16384), not `num_predict` -- a prior evaluation
  attributed the cutoff to "the model's fixed context budget" without
  ever trying num_ctx.
- A single crop (or even the full page) in ONE call read every test
  form's quantities as well as or better than either glm-ocr's
  single-crop treatment or LightOnOCR-2's own earlier per-row grid.py
  crop approach, on 3 of 4 test forms -- `sample 4-scanned` (mostly
  single tally marks) remains the hard case.
- Feeding the FULL page (letterhead included) needed no more context
  than the table-only crop and produced byte-identical table content on
  the two forms tested -- confirmed the letterhead-exclusion step that
  mattered for glm-ocr is NOT load-bearing for LightOnOCR-2. This module
  defaults to the full page for that reason: one fewer moving part
  (no table-region cropping, no letterhead-margin tuning) than a
  crop-based approach would need.

Quirks this module was written to handle, ported from patterns
extract_claude.py's own prompt (SYSTEM_PROMPT_TEMPLATE) already teaches
mistral, or from extract_ollama_cloud.py's PaddleOCR path already
defending against -- see _parse_cell and the reconciliation loop below
for where each one lives:
- SIZE LABEL OVERRIDE / LETTER SIZES (the "size override" / "available
  size" concept): a cell sometimes holds a tiny handwritten size label
  (a number past the printed grid, or a clothing letter like M/XL)
  stacked over its own quantity, because the row's real product doesn't
  match this column's printed header at all -- e.g. sample 5-scanned's
  "Fairlady Print" row, where the printed grid stops at 105 but the
  product needed sizes 105 and 110, so the writer squeezed
  "105 (small) / 10 (quantity)" into one cell. LightOnOCR-2's own <br>
  normalization surfaces this as a two-token cell ("105 10"); _parse_cell
  splits it and reports the size ACTUALLY WRITTEN (not the printed
  column position) as the key -- exactly extract_claude.py's SIZE LABEL
  OVERRIDE rule, applied to LightOnOCR-2's transcription instead of
  asking a second model to reason about it. _reconcile_hybrid_with_vlm
  already knows what to do with a hybrid_map key outside size_headers
  (trust it wholesale, see that function's own docstring) -- no change
  needed there. The tiny label is often written only ONCE per page, on
  the first row that needs the repurposed column -- a later row reusing
  the SAME physical column (a fixed ruled line) often just writes the
  bare quantity, the same "write it once, ditto after" convention these
  forms already use for item names (confirmed on sample 5-scanned: the
  row directly below "Fairlady Print" shares its overflow column with a
  bare, unlabeled quantity). resolved_override_col makes one pass over
  every aligned row's unmapped columns first, so a label seen on any row
  resolves that column for every row on the page.
- Two-row (two-printed-line) products: an item whose name/description
  wraps onto a second printed line (e.g. sample 4-scanned's "1. B.3825"
  / "SHORT SET RNBS") is read by LightOnOCR-2 as ONE table row with a
  <br> inside the cell, not two separate rows -- confirmed on a real
  response, no special handling needed here beyond the <br>-to-space
  normalization _cell_text already does for every cell.
- Struck-out rows: mistral's own direct "this row is crossed out"
  judgment (MistralExtractedItem.struck_out) is threaded through from
  extract_one() as struck_out_hints, exactly the way letter_size_hints
  already is -- a row flagged there has its LightOnOCR-2 reading forced
  to empty IF LightOnOCR-2 also found nothing, which is the ordinary
  case for a real cancelled row. Deliberately NOT the blank-quantities/
  blank-total proxy extract_ollama_cloud.py's PaddleOCR path falls back
  to for model-agnostic struck-out detection (see
  _realign_row_clusters_by_total's docstring) -- that proxy was tried
  here first and produced a real false positive during testing.
  struck_out ITSELF then also turned out unreliable on a real row
  (sample 13-scanned's "B 4457 COLLAR": mistral reported struck_out=True
  on a row confirmed, by zooming into the photo, to have no strike-
  through at all -- a normal row with a clean printed total matching its
  own quantities). Since both of struck_out's failure directions are
  real (silently keeping a genuinely cancelled row ships an unwanted
  order; silently zeroing a real one ships an order short), a row
  flagged struck_out where LightOnOCR-2 still found a substantial
  reading is treated as a genuine conflict and FLAGGED for a human
  instead of silently decided either way -- same "flag, don't guess"
  principle _flag_hybrid_total_mismatch already applies to a checksum
  disagreement. struck_out_hints is empty for a non-mistral model or
  when this function is called without it (e.g.
  compare_hybrid_backends.py) -- no struck-out handling happens in that
  case, same as before this fix existed.
- The "dividend/divisor" two-line header (a size number stacked over an
  unrelated alt/dozen-equivalent number, e.g. sample 12-scanned's
  "35<br>14") -- _header_cols_in_block matches on the first line only,
  so the alt number is never mistaken for a size header or read as a
  quantity.

Known gaps still open (expect to find more -- iterate here, not in the
production file):
- Row alignment falls back to fuzzy text matching only when row counts
  disagree -- a row that's present but whose leading text came back
  empty could still misalign. Not yet given the same scrutiny as
  grid.py's item_alignment_ok.
- No brandlist-driven letter-size-to-number resolution is attempted
  here -- a LETTER SIZES row's cells come back keyed by the letter
  itself (e.g. "M"), same as mistral's own raw reading; resolving that
  to a real catalog number is brandlist_match.py's job downstream,
  unchanged by which hybrid backend ran.
"""
from __future__ import annotations

import difflib
import io
import json
import re
from pathlib import Path

import ollama
from PIL import Image

from extract_claude import _parse_int_or_none
from extract_ollama_cloud import (
    OLLAMA_REQUEST_TIMEOUT,
    ExtractedForm,
    _exif_corrected_bytes,
    _flag_hybrid_total_mismatch,
    _hybrid_ocr_quantities,
    _reconcile_hybrid_with_vlm,
)

# A plain `from ollama import chat` (the bare module-level convenience
# function) used to be used here -- confirmed 2026-09-08 that its own
# underlying default client has NO request timeout at all
# (`chat.__self__._client.timeout` -> `Timeout(timeout=None)`), same gap
# OLLAMA_REQUEST_TIMEOUT's own comment in extract_ollama_cloud.py
# documents for `ollama.Client()`. This module's own call runs on
# EVERY extraction by default (this is the production hybrid-quantities
# backend), so a genuinely stuck local call here -- not just in the main
# call -- would hang the review app's single-worker executor forever
# with no exception ever raised for extract_one()'s own try/except
# around this call to catch.
_CLIENT = ollama.Client(timeout=OLLAMA_REQUEST_TIMEOUT)

LIGHTON_MODEL = "maternion/LightOnOCR-2:1b"
NUM_CTX = 16384

PROMPT = (
    "Transcribe this order form table exactly as written. Output a markdown "
    "table: first column the item/description, next columns the printed "
    "size headers, then quantities. Leave a cell blank if nothing is "
    "written there -- do not guess or fill in a value. Preserve every row. "
    "If a cell has a small handwritten size label -- a number past the "
    "printed grid, or a clothing letter size like S, M, L, XL, XXL -- "
    "written above or beside its quantity, overriding the printed column "
    "header for that cell, write BOTH the label and the quantity in that "
    "cell separated by a space (e.g. '105 10', 'M 6'), not just the "
    "quantity alone under the printed header's column."
)

CIRCLED = {chr(0x2460 + i): i + 1 for i in range(9)}  # ①-⑨ -> 1-9

TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.S)
TBODY_RE = re.compile(r"<tbody[^>]*>(.*?)</tbody>", re.S)
THEAD_RE = re.compile(r"<thead[^>]*>(.*?)</thead>", re.S)
TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
TD_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S)


def _cell_text(raw: str) -> str:
    raw = re.sub(r"<br\s*/?>", " ", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    return raw.strip()


def _parse_tr_block(block: str) -> list[list[str]]:
    return [[_cell_text(td) for td in TD_RE.findall(tr)] for tr in TR_RE.findall(block) if TD_RE.findall(tr)]


LETTER_SIZES = {"0", "XS", "S", "M", "L", "XL", "XXL", "2XL", "3XL", "4XL"}


def _as_plain_int(token: str) -> int | None:
    if token in CIRCLED:
        return CIRCLED[token]
    if re.fullmatch(r"\d+", token):
        return int(token)
    return None


def _parse_cell(raw: str) -> tuple[str | None, int | None]:
    """Returns (override_size, quantity). override_size is None for an
    ordinary cell -- the printed column header it sits under applies as-is.
    Non-None means this cell is a SIZE LABEL OVERRIDE or LETTER SIZES pair
    (see module docstring): a tiny handwritten size label -- a number past
    the printed grid, or a clothing letter -- stacked over its own
    quantity, surfaced by LightOnOCR-2 as a two-token cell once <br>
    becomes a space (e.g. "105 10", "M 5"). The label is reported exactly
    as written (never converted, per extract_claude.py's own LETTER SIZES
    rule) and used as the dict KEY instead of the printed header."""
    raw = raw.strip()
    qty = _as_plain_int(raw)
    if qty is not None:
        return None, qty
    tokens = raw.split()
    if len(tokens) == 2:
        label, qty_token = tokens
        qty = _as_plain_int(qty_token)
        if qty is not None and (label.isdigit() or label.upper() in LETTER_SIZES):
            return (label if label.isdigit() else label.upper()), qty
    return None, None  # blank, dash, "x", or anything else unparsed -- no quantity, not a guess


def _call_lighton(image_bytes: bytes) -> tuple[str, str | None]:
    response = _CLIENT.chat(
        model=LIGHTON_MODEL,
        messages=[{"role": "user", "content": PROMPT, "images": [image_bytes]}],
        options={"temperature": 0, "num_ctx": NUM_CTX, "num_predict": -1},
    )
    return response.message.content, getattr(response, "done_reason", None)


def _header_cols_in_block(block: str, size_headers: list[str]) -> dict[str, int]:
    """Maps each of mistral's known size_headers to a column index, by
    exact string match against every <thead> row IN THIS ONE TABLE BLOCK
    (a two-row header, main label + alternate numbering, is common --
    e.g. sample 12-scanned's "35/14" pairs -- so every thead row in the
    block is searched, not just the first)."""
    header_rows = []
    for thead_body in THEAD_RE.findall(block):
        header_rows.extend(_parse_tr_block(thead_body))
    if not header_rows:
        header_rows = _parse_tr_block(block)[:1]  # malformed block with no <thead> at all: fall back to its own row 0
    col_for_header: dict[str, int] = {}
    for header_row in header_rows:
        for idx, cell in enumerate(header_row):
            # A header cell is sometimes two stacked lines in ONE <th>
            # (the size number over its alt/dozen-equivalent numbering,
            # e.g. "45<br>18") rather than two separate header rows --
            # confirmed on a real response where "45<br>18" survives
            # _cell_text's <br>-to-space normalization as "45 18", which
            # never equals size_headers' plain "45" on a whole-string
            # match. The first whitespace-separated token is checked too
            # for exactly this shape.
            tokens = cell.split()
            candidates = [cell] if not tokens else [cell, tokens[0]]
            for candidate in candidates:
                if candidate in size_headers and candidate not in col_for_header:
                    col_for_header[candidate] = idx
                    break
    return col_for_header


def _find_item_table(text: str, size_headers: list[str]) -> tuple[dict[str, int], list[list[str]]]:
    """LightOnOCR-2 transcribes the WHOLE page as separate <table> blocks
    when no crop excludes the letterhead -- confirmed on a real response
    (sample 4-scanned's full-page read produced 3 separate <table>
    elements: a "Party Name" mini-table, a "LORRY/Booking Station"
    mini-table, and only THEN the real size/quantity table). Blindly
    taking the first <thead>/<tbody> in the whole document (this
    function's first version) grabbed the Party Name mini-table's own
    header/body instead -- found zero real size headers and skipped the
    row every time. Scores every <table> block by how many known
    size_headers its own header row(s) contain and uses the best-scoring
    one for both header columns AND its own tbody rows, so a header match
    and its data always come from the SAME table block."""
    table_blocks = TABLE_RE.findall(text) or [text]  # malformed output with no <table> tag at all: treat it as one block
    best_cols: dict[str, int] = {}
    best_block = table_blocks[0]
    for block in table_blocks:
        cols = _header_cols_in_block(block, size_headers)
        if len(cols) > len(best_cols):
            best_cols, best_block = cols, block

    tbody_bodies = TBODY_RE.findall(best_block)
    if tbody_bodies:
        rows = [r for body in tbody_bodies for r in _parse_tr_block(body)]
    else:
        rows = _parse_tr_block(best_block)[1:]  # no <tbody> tag: skip row 0 (the header row itself)

    # Drop rows with no real data -- but ONLY checking the item-label cell
    # and the matched size-header columns, not every cell in the row.
    # Confirmed a real bug checking every cell (this function's first
    # version): a page's own trailing "Grand Total" row (blank item name,
    # blank size cells, but a real sum in the trailing Total column) then
    # survives as a fake extra row, making len(data_rows) one MORE than
    # len(item_names) even when every real row parsed cleanly -- which
    # forces _align_rows_to_items's fuzzy-matching fallback instead of a
    # clean 1:1 positional match, and that fallback then misaligned two
    # unrelated rows on a real 20-item form (confirmed: the model's own
    # raw response had item 0 exactly right, but this bug's downstream
    # fuzzy match handed item 0 a completely different row's data instead).
    size_cols = set(best_cols.values())

    def _is_summary_row(r: list[str]) -> bool:
        # A page-footer "Grand Total" row has no real per-item data in any
        # size column, whichever shape it takes: a blank item-label cell
        # (sample 3-scanned), or the label ITSELF holding the summary text
        # via a wide colspan (sample 13-scanned's `colspan="16">GRAND
        # TOTAL`, which TD_RE still captures as ONE cell -- colspan isn't
        # parsed, just whatever text sits in that one <td>). Checking the
        # label text too (not just blankness) matters here specifically:
        # a blank-label version of this same row already proved capable of
        # forcing the fragile fuzzy-match fallback and misaligning two
        # unrelated real rows on sample 3-scanned -- this is the same
        # latent risk on a form where it hasn't (yet) caused a visible
        # symptom, not a difference worth trusting to luck twice.
        if any(c < len(r) and r[c] for c in size_cols):
            return False
        label = (r[0] if r else "").strip().upper()
        return label == "" or "GRAND TOTAL" in label or label in ("TOTAL", "TOTALS")

    rows = [r for r in rows if not _is_summary_row(r)]
    return best_cols, rows


def _row_label(row: list[str], size_cols: set[int]) -> str:
    """Every leading cell before the first matched size column, joined --
    works whether the form has 1 leading column (item) or 3 (serial no,
    article name, style), without needing to know which up front."""
    first_size_col = min(size_cols) if size_cols else len(row)
    return " ".join(c for c in row[:first_size_col] if c).strip()


def _partial_ratio(a: str, b: str) -> float:
    """fuzzywuzzy-style partial_ratio via stdlib difflib: best-matching
    contiguous window of the longer string against the shorter one.
    Confirmed necessary on sample 4-scanned: mistral named its own last
    two rows just "PANT" (down from a fuller "CREED B4756 1/4\" PANT" on
    an earlier run of the same image) while LightOnOCR-2 still read the
    full "10. CREED B4756 1/4\" PANT" label -- plain SequenceMatcher.ratio()
    scores that pair ~0.27 (penalized by the ~20-character length gap
    between a 4-char query and a 26-char label) even though "PANT" is an
    exact substring, so both rows silently fell back to mistral's own
    unverified reading instead of being hybrid-corrected."""
    if len(a) > len(b):
        a, b = b, a
    if not a:
        return 0.0
    best = 0.0
    for block in difflib.SequenceMatcher(None, a, b).get_matching_blocks():
        if block.size == 0:
            continue
        start = max(0, min(block.b - block.a, len(b) - len(a)))
        window = b[start:start + len(a)]
        best = max(best, difflib.SequenceMatcher(None, a, window).ratio())
    return best


def _align_rows_to_items(data_rows: list[list[str]], item_names: list[str], size_cols: set[int]) -> dict[int, list[str]]:
    """Row i of data_rows -> item index. Positional when counts already
    match (the common case); otherwise a greedy order-preserving fuzzy
    match on each row's leading text, so a dropped/extra row (confirmed
    to happen -- e.g. sample 4-scanned's row 12, which LightOnOCR-2
    drops from the table entirely) doesn't silently misalign every row
    after it."""
    if len(data_rows) == len(item_names):
        return {i: data_rows[i] for i in range(len(item_names))}

    labels = [_row_label(r, size_cols) for r in data_rows]
    used: set[int] = set()
    aligned: dict[int, list[str]] = {}
    for i, item_name in enumerate(item_names):
        if not item_name:
            continue
        best_j, best_score = None, 0.0
        for j, label in enumerate(labels):
            if j in used or not label:
                continue
            score = max(
                difflib.SequenceMatcher(None, item_name.lower(), label.lower()).ratio(),
                _partial_ratio(item_name.lower(), label.lower()),
            )
            if score > best_score:
                best_j, best_score = j, score
        if best_j is not None and best_score >= 0.35:
            aligned[i] = data_rows[best_j]
            used.add(best_j)
    return aligned


def lighton_hybrid_quantities(
    image_path: Path,
    extracted: ExtractedForm,
    outdir: Path | None = None,
    struck_out_hints: set[int] = frozenset(),
    letter_size_hints: set[int] = frozenset(),
) -> tuple[dict[int, dict[str, int]], dict[int, dict]]:
    """Drop-in-shaped alternative to extract_ollama_cloud._hybrid_ocr_quantities
    -- same (image_path, extracted, outdir) -> (quantities_by_index, flags)
    contract, so a future integration can select between the two behind a
    flag without changing extract_one()'s merge loop. See module docstring
    for what this does and does not handle yet."""
    size_headers = extracted.size_headers
    n_items = len(extracted.items)
    if n_items == 0 or not size_headers:
        return {}, {}

    image_bytes = _exif_corrected_bytes(image_path)
    text, done_reason = _call_lighton(image_bytes)

    if outdir is not None:
        debug = {"done_reason": done_reason, "raw_response_chars": len(text), "raw_response": text}

    col_for_header, data_rows = _find_item_table(text, size_headers)
    if not col_for_header:
        print("  LightOnOCR-2 hybrid: found none of this form's known size headers in the response -- skipping.")
        if outdir is not None:
            debug["skipped"] = "no size headers matched"
            (outdir / f"{image_path.stem}.lighton_debug.json").write_text(json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8")
        return {}, {}

    size_cols = set(col_for_header.values())
    item_names = [it.item for it in extracted.items]
    aligned = _align_rows_to_items(data_rows, item_names, size_cols)

    # Narrow item-NAME auto-fill, not just quantities. Confirmed real on
    # sample 4-scanned's last rows: mistral's own main call collapsed
    # "CREED B4756 1/4\" PANT" / "LOOPER ZB 3965 1/4\" PANT" down to a bare
    # "PANT" and even duplicated it into `type` ("PANT"/"PANT"), while
    # LightOnOCR-2's own aligned row for the same index still has the full
    # text. Deliberately NOT a general "prefer the longer name" rule --
    # LightOnOCR-2's raw table reads are noisier than mistral's on other
    # forms (e.g. sample 12-scanned's "ESSD Premium Rm" / "EscoD Mimi
    # drumk o/c" against mistral's own cleaner "ESSA Premium" / "Goodloom
    # Mini Adult"), so overwriting a merely-short-but-correct name would
    # trade a real problem for an invisible one. Only fires when mistral's
    # own name is a clear break -- empty, or identical to its own `type`
    # field (both are signals the main call collapsed the row rather than
    # actually reading it), which the row's own row_total/quantities can't
    # catch since those can still be right even when the name isn't.
    name_corrections: dict[int, str] = {}
    for idx, row in aligned.items():
        item = extracted.items[idx]
        mistral_name = item.item.strip()
        mistral_type = item.type.strip()
        if mistral_name and mistral_name.lower() != mistral_type.lower():
            continue
        label = _row_label(row, size_cols)
        candidate = re.sub(r"^\d+\.\s*", "", label).strip()
        if candidate and candidate.lower() != mistral_name.lower():
            item.item = candidate
            name_corrections[idx] = candidate

    header_for_col = {col: header for header, col in col_for_header.items()}
    first_col = min(size_cols) if size_cols else 0

    # Page-level override-column resolution: a SIZE LABEL OVERRIDE cell's
    # tiny handwritten label is often written only ONCE per page, on the
    # first row that needs that repurposed column -- every later row
    # reusing the same physical column (a fixed ruled line, same position
    # top to bottom) just writes the bare quantity, the same "write it
    # once, ditto after" convention these forms already use for item
    # names. Confirmed real: sample 5-scanned's "Fairlady Print" row
    # labels its overflow column "110"; the very next row ("Fairlady
    # Plain"), sharing that same column, has a bare "10" there with no
    # label at all -- a per-row-only override lookup misses it entirely.
    # One pass over every aligned row's UNMAPPED columns first, so a label
    # seen on any row resolves that column for every row.
    resolved_override_col: dict[int, str] = {}
    for row in aligned.values():
        for col in range(first_col, len(row)):
            if col in header_for_col:
                continue
            override_size, qty = _parse_cell(row[col])
            if override_size is not None:
                resolved_override_col.setdefault(col, override_size)

    quantities_by_index: dict[int, dict[str, int]] = {}
    for idx, row in aligned.items():
        cells: dict[str, int] = {}
        # Every column from the first matched size header onward, not just
        # the matched header columns themselves -- a SIZE LABEL OVERRIDE
        # cell is self-labeled and can sit ANYWHERE, including a column
        # this form doesn't even print a size header for at all (confirmed
        # on sample 5-scanned's "Fairlady Print" row: its second override,
        # size 110, occupies the TRAILING "Total Dozen" column's position,
        # which is deliberately never in col_for_header -- see SIZE
        # HEADERS' own prompt rule excluding running-total columns).
        for col in range(first_col, len(row)):
            override_size, qty = _parse_cell(row[col])
            if qty is None:
                continue
            header = header_for_col.get(col)
            if override_size is not None:
                cells[override_size] = qty
            elif header is not None:
                cells[header] = qty
            elif col in resolved_override_col:
                # This row's own cell has no label, but another row already
                # resolved what this column means -- see the pass above.
                cells[resolved_override_col[col]] = qty
        quantities_by_index[idx] = cells  # explicit empty dict when a matched row had no readable cells

    if outdir is not None:
        debug["headers_matched"] = col_for_header
        debug["rows_found"] = len(data_rows)
        debug["rows_aligned"] = len(aligned)
        debug["unaligned_items"] = [item_names[i] for i in range(n_items) if i not in aligned]
        debug["name_corrections"] = {str(i): {"from": item_names[i], "to": name} for i, name in name_corrections.items()}
        (outdir / f"{image_path.stem}.lighton_debug.json").write_text(json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8")

    if name_corrections:
        print(f"  LightOnOCR-2 hybrid: filled in {len(name_corrections)} row name(s) mistral left empty/duplicated from its own type field.")

    print(f"  LightOnOCR-2 hybrid quantity read: aligned {len(aligned)}/{n_items} rows "
          f"({len(data_rows)} rows found in the transcription, {len(col_for_header)}/{len(size_headers)} size headers matched).")

    # Letter-size rows (MistralExtractedItem.letter_sizes, threaded through
    # as letter_size_hints): confirmed LightOnOCR-2 can't be prompted into
    # preserving a cramped stacked S/M/L/XL/XXL label instead of guessing a
    # plain digit under the nearest printed numeric column -- tried an
    # explicit PROMPT instruction for exactly this and it made no
    # difference on a real re-run of sample 3-scanned's MM K4532 row. This
    # is a real gap the SIZE LABEL OVERRIDE mechanism above can't close on
    # its own: it can only use a label LightOnOCR-2 actually reported, and
    # LightOnOCR-2 never reports this one.
    #
    # Rather than build a second, from-scratch pixel-OCR recovery pass,
    # this reuses extract_ollama_cloud._hybrid_ocr_quantities's own real
    # one wholesale -- _recover_letter_size_row_digits was already written
    # and tuned specifically against this exact row (see its own
    # docstring: two prior crop designs tried and failed before landing on
    # a per-column, y-offset-swept re-OCR), deeply entangled with
    # PaddleOCR's own whole-page header/row-position state. Reimplementing
    # that here would be re-deriving already-hard-won tuning, not writing
    # something new.
    #
    # Gated on letter_size_hints ALONE (mistral's own flag) -- NOT unioned
    # with a code-derived "this row's total doesn't add up" signal, which
    # was tried and reverted the same day: it does correctly widen the net
    # to catch rows like MM K4532 that mistral never flags, but a large sum
    # mismatch turns out to correlate with ordinary column-shift misreads
    # just as much as with a real letter-size row -- confirmed on a live
    # run where it fired on 4 unrelated rows of sample 5-scanned (a form
    # with no letter-size row at all), each paying the full ~20-40s
    # PaddleOCR whole-page pass (see this project's own usage log) for no
    # benefit. That defeats the reason this backend swap happened in the
    # first place. Net effect of reverting: MM K4532-shaped rows stay
    # unresolved (mistral practically never sets the flag for this exact
    # cramped letter-over-digit shape -- confirmed via 5 fresh live calls,
    # all letter_sizes=False) rather than fixed, in exchange for keeping
    # LightOnOCR-2's latency advantage intact on every other form. A
    # cheaper, genuinely targeted pre-check (e.g. a lightweight letter-
    # detection-only scan at just a suspect row's position, before ever
    # committing to the full expensive path) would be the way to recover
    # this specific row without the cost -- not yet built.
    letter_resolved: dict[int, dict] = {}  # idx -> flag (or None), for rows PaddleOCR's recovery already settled
    letter_or_mismatch_hints = letter_size_hints
    if letter_or_mismatch_hints:
        paddle_qty, paddle_flags = _hybrid_ocr_quantities(image_path, extracted, outdir, letter_or_mismatch_hints)
        for idx in letter_or_mismatch_hints:
            recovered = paddle_qty.get(idx)
            if recovered:
                quantities_by_index[idx] = recovered
                letter_resolved[idx] = paddle_flags.get(idx)

    # Same reconciliation the production PaddleOCR path uses -- keeps this
    # module's disagreement-handling identical and comparable, rather than
    # inventing a second policy to also evaluate.
    size_headers_set = set(size_headers)
    flags: dict[int, dict] = {}
    for idx, qty in quantities_by_index.items():
        if idx in letter_resolved:
            # Already resolved via PaddleOCR's own letter-size recovery --
            # that path already reconciled against vlm internally
            # (_hybrid_ocr_quantities calls the same
            # _reconcile_hybrid_with_vlm this loop uses), so running it
            # again here would just be re-deciding an already-settled row.
            if letter_resolved[idx] is not None:
                flags[idx] = letter_resolved[idx]
            continue
        printed_total = _parse_int_or_none(extracted.items[idx].row_total)
        vlm_map = {qp.size: qp.quantity for qp in extracted.items[idx].quantities}
        if idx in struck_out_hints:
            # Mistral's own direct "this row is crossed out" judgment
            # (MistralExtractedItem.struck_out, threaded through from
            # extract_one() -- empty by default, e.g. when called from
            # compare_hybrid_backends.py or a non-mistral model). The
            # ordinary case: LightOnOCR-2 also found nothing here, which is
            # exactly what a real cancelled row should look like -- silently
            # keep it empty, nothing to review.
            #
            # But struck_out itself isn't perfectly reliable either --
            # confirmed on a real run of sample 13-scanned.jpg's "B 4457
            # COLLAR" row: mistral reported struck_out=True (and, before
            # that, the empty-quantities/blank-total proxy this replaced
            # made the same call) on a row later confirmed, by zooming into
            # the photo, to have NO strike-through at all -- a normal row
            # with a clean printed total (51) matching its own quantities
            # exactly. Both of struck_out's own failure directions are
            # real: silently keeping a genuinely cancelled row ships an
            # order that was never wanted; silently zeroing a real one
            # ships an order short. When LightOnOCR-2 still found a
            # substantial reading despite the flag, that's a genuine
            # conflict between two real signals, not a clear case either
            # way -- flag it for a human instead of silently picking a
            # side (same "flag, don't guess" principle
            # _flag_hybrid_total_mismatch already applies to a checksum
            # disagreement). The flagged reading is LightOnOCR-2's own,
            # not zeroed -- an extra row a reviewer deletes in one click is
            # cheaper to catch than a missing one is to notice was ever
            # dropped.
            if qty:
                flags[idx] = {
                    "status": "unresolved",
                    "sizes": sorted(qty.keys(), key=lambda s: int(s) if s.isdigit() else 0),
                    "note": ("LightOnOCR-2 hybrid: mistral flagged this row as struck-through/cancelled, but "
                             "an independent OCR pass still found real quantities here -- compare against the "
                             "photo before trusting either reading."),
                }
                quantities_by_index[idx] = qty
            else:
                quantities_by_index[idx] = {}
            continue
        merged, conflict_flag = _reconcile_hybrid_with_vlm(vlm_map, qty, printed_total, size_headers_set)
        quantities_by_index[idx] = merged
        if conflict_flag is not None:
            flags[idx] = conflict_flag
            continue
        flag_text = _flag_hybrid_total_mismatch(merged, printed_total)
        if flag_text is not None:
            diff = sum(merged.values()) - printed_total
            status = "unresolved" if abs(diff) > 2 else "unverified"
            flags[idx] = {
                "status": status,
                "sizes": sorted(merged.keys(), key=lambda s: int(s) if s.isdigit() else 0),
                "note": f"LightOnOCR-2 hybrid: {flag_text}.",
            }
    return quantities_by_index, flags
