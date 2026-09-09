"""
chandra_parser.py -- turns datalab-to/chandra-ocr-2's own native output
(bbox-annotated HTML blocks, an unconstrained "extract this order form"
call, NOT a schema-constrained one) into this project's ExtractedForm
shape, so extract_ollama_cloud.extract_one() can run Chandra through
exactly the same downstream pipeline (hybrid quantities, brandlist
cross-check, review app) every other model already goes through.

Why native format, not schema-constrained: confirmed by direct testing
(see HISTORY.md's 2026-09-07 section) that forcing Chandra through this
project's Pydantic JSON schema (the same format= mechanism mistral uses)
badly degrades its output -- it was trained to emit its own HTML/bbox
format, not an arbitrary externally-imposed JSON shape. This module is
the real integration that session's evaluation left unbuilt: every rule
below is grounded in Chandra's ACTUAL output on this project's own
regression forms (sample 5/3/4/13-scanned, sample 2.jpeg free-form,
sample 12-scanned's To/From template), captured and inspected directly
before writing the corresponding parsing rule -- not guessed at.

Chandra's own output shape, confirmed across all 6 forms above:
- A flat sequence of top-level `<div data-bbox="x1 y1 x2 y2"
  data-label="LABEL">...</div>` blocks in reading order. Coordinates are
  normalized to a fixed 0-1000 scale (Qwen-VL's own convention, confirmed
  by an outer table bbox of "0 322 998 998" on a real response) --
  independent of the image's real pixel dimensions or aspect ratio, so
  x/y fracs are both just value/1000.0.
- data-label is NOT a reliable field-role signal on its own: the SAME
  visual element (this business's own letterhead company name, the
  Party Name box) comes back labeled "Section-Header" on one form and
  plain "Text" or "Form" on another, confirmed on 3 different real ESSA
  forms. Every header-field extractor below matches on the block's own
  TEXT CONTENT (a label word like "Party Name" or "From", a date-shaped
  substring, a company-name-suffix word), never on data-label alone.
- The item/quantity table lives inside one or more "Table"-labeled
  blocks containing real `<table>` HTML. Confirmed (sample 3-scanned):
  Chandra sometimes splits ONE visual table into a header-only `<table>`
  (its own `<div data-label="Table">`, a `<thead>` with no `<tbody>`) and
  a separate body-only `<table>` (a second `<div data-label="Table">`, a
  `<tbody>` with no `<thead>`) -- every "Table" block found is
  concatenated, not just the first, and table_top_frac/table_bottom_frac
  span every such block's own bbox, not just one.
- A second header row beneath the real size headers (an alternate
  dozen/box-count numbering, sometimes with a clothing-letter-size label
  stacked into a cell via `<br/>`, e.g. sample 12-scanned's "30<br/>XS")
  is a confirmed decoy on every form tested -- this parser never reads
  ANY header row but the first one under `<thead>` for size_headers, per
  the same DITTO MARKS/SIZE HEADERS rule extract_claude.py's own prompt
  already teaches mistral ("never put it in size_headers").
- A free-form page (no printed grid, e.g. sample 2.jpeg) has NO `<table>`
  at all -- every item name and its size:quantity pairs are separate
  plain-text blocks, quantities rendered as `<math>\\frac{A}{B}</math>`
  (or occasionally plain `<math>A/B</math>`) with A=size, B=quantity.
  A wrapped continuation line (more pairs for the item above, no new
  name) is its own subsequent block with no text but the pairs -- see
  _parse_freeform_blocks.
- Chandra sometimes describes a non-text visual element with bracketed
  placeholder prose instead of leaving the block empty (confirmed real:
  "[Empty box for GST]", "[Signature]" in a row's own total cell on
  sample 4-scanned). _is_placeholder() filters these out everywhere a
  field value is extracted, so a placeholder description never becomes a
  fabricated order_no/party_name/row_total.

NOT wired for: MistralExtractedItem's struck_out/date_present/letter_sizes
signals -- Chandra gets no schema and no addendum prompt at all (see
CHANDRA_PROMPT below), so extract_one() leaves letter_size_hints/
struck_out_hints empty for this model, same as it already does for any
non-mistral model. Confirmed acceptable by direct testing (HISTORY.md):
Chandra's own reading already comes back with empty quantities for a
genuine struck-out row with no special mechanism needed, and letter-size
rows remain a shared, still-open gap across every model in this project,
not something this integration regresses.
"""
from __future__ import annotations

import re
import time

import ollama

from extract_claude import ExtractedForm, ExtractedItem, KNOWN_STYLE_CODES, QuantityPair

CHANDRA_MODEL = "hf.co/mradermacher/chandra-ocr-2-GGUF:Q5_K_M"

# Confirmed necessary (HISTORY.md, 2026-09-07): local Ollama's default
# context window truncates this model's real per-form output; num_ctx=16384
# fixed it with no accuracy cost across every form tested. num_predict=-1
# (uncapped, bounded only by num_ctx) mirrors hybrid_quantities_lighton.py's
# own already-working _call_lighton pattern.
NUM_CTX = 16384

# Plain, unconstrained prompt -- NOT this project's schema-constrained
# SYSTEM_PROMPT_TEMPLATE/USER_PROMPT (see module docstring: forcing the
# schema onto Chandra made its output categorically worse, not better).
CHANDRA_PROMPT = "Extract this order form as markdown, preserving the table structure exactly"


# --------------------------------------------------------------- calling

def call_chandra(client: ollama.Client, model: str, image_bytes: bytes) -> tuple[str | None, str | None, dict]:
    """One Chandra call. Returns (raw_text, error, usage) -- same shape
    convention as extract_ollama_cloud._call_schema, so extract_one()'s
    existing usage-logging call sites work unchanged regardless of which
    model path produced the call.

    think=False: confirmed by direct testing (HISTORY.md) this model's
    "thinking" field routing is a qwen3.5 chat-template quirk, not a real
    hidden reasoning phase -- output lands in .thinking instead of
    .content when think isn't explicitly passed, with zero difference in
    token count or generation time either way. Explicitly False here so
    .message.content is always where the real output is."""
    t0 = time.perf_counter()
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": CHANDRA_PROMPT, "images": [image_bytes]}],
            options={"temperature": 0, "num_ctx": NUM_CTX, "num_predict": -1},
            think=False,
        )
    except Exception as exc:
        return None, str(exc), {"prompt_eval_count": None, "eval_count": None, "duration_seconds": time.perf_counter() - t0}

    usage = {
        "prompt_eval_count": resp.prompt_eval_count,
        "eval_count": resp.eval_count,
        "duration_seconds": time.perf_counter() - t0,
    }
    if resp.done_reason not in ("stop", None):
        return None, f"generation stopped early (done_reason={resp.done_reason})", usage
    text = resp.message.content or ""
    if not text.strip():
        return None, "empty response", usage
    return text, None, usage


# ------------------------------------------------------------ block split

_BLOCK_START_RE = re.compile(r'<div\s+data-bbox="(?P<bbox>[\d\s]+)"\s+data-label="(?P<label>[^"]*)"\s*>')


def _iter_blocks(raw_text: str) -> list[tuple[list[int], str, str]]:
    """Splits Chandra's flat sequence of top-level bbox divs into
    (bbox, label, inner_html) tuples, in document order. Chunk boundaries
    are found by locating every block-start tag first, then slicing
    between consecutive starts (rather than a naive non-greedy
    `(.*?)</div>` per block), so a block's own inner content never needs
    to be assumed div-free -- confirmed unnecessary in practice (no block
    seen across 6 real forms nests another data-bbox div), but this is
    the version that wouldn't silently truncate if one did."""
    starts = list(_BLOCK_START_RE.finditer(raw_text))
    blocks = []
    for i, m in enumerate(starts):
        start = m.end()
        end = starts[i + 1].start() if i + 1 < len(starts) else len(raw_text)
        inner = raw_text[start:end]
        inner = re.sub(r"</div>\s*$", "", inner.rstrip())
        bbox = [int(x) for x in m.group("bbox").split()]
        blocks.append((bbox, m.group("label"), inner))
    return blocks


def _frac(v: int) -> float:
    return v / 1000.0


# ------------------------------------------------------------- text utils

_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_WS_RE = re.compile(r"\s+")
_PLACEHOLDER_RE = re.compile(r"^\[.*\]$")


def _strip_tags(html: str) -> str:
    text = _BR_RE.sub(" ", html)
    text = _TAG_RE.sub("", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
    return _WS_RE.sub(" ", text).strip()


def _cell_text(html: str) -> str:
    return _strip_tags(html)


def _is_placeholder(text: str) -> bool:
    """True for Chandra's own bracketed description of a non-text visual
    element ("[Empty box for GST]", "[Signature]") -- confirmed real on
    sample 4-scanned, both in a header field's value position and in a
    row's own total cell. Never real form content."""
    return bool(_PLACEHOLDER_RE.match(text.strip()))


def _clean(text: str) -> str:
    text = text.strip()
    return "" if _is_placeholder(text) else text


# ------------------------------------------------------ header-field text

_DATE_SEARCH_RE = re.compile(r"date\W{0,10}?(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})", re.I)
_ORDER_NO_RE = re.compile(r"order\s*(?:form)?\s*no\.?\s*[:\-]?\s*(?P<val>\S+)", re.I)
_ORDER_NO_RE2 = re.compile(r"^no\.?\s*[:\-]?\s*(?P<val>\S+)", re.I)
_PARTY_NAME_RE = re.compile(r"party\s*name\s*[:\-]?\s*(.+)", re.I)
_GENERIC_TITLE_RE = re.compile(
    r"^(order\s*form(\s*no\.?)?|invoice|proforma\s*invoice|bill|challan|delivery\s*challan|memo)$", re.I
)
_HEADING_TAG_RE = re.compile(r"<h[12][ >]|<b[ >]|<strong[ >]", re.I)
_EXCLUDE_PREFIX_RE = re.compile(r"^(gstin|cin|ph\.?|tel|contact|date|party\s*name|gst\b)", re.I)
_LABEL_FRAGMENT_RE = re.compile(r"^(m/s\.?|mr\.?|mrs\.?|ms\.?|to,?|from\s*:?)$", re.I)


def _normalize_date(d: str, mo: str, y: str) -> str:
    if len(y) == 2:
        y = "20" + y
    return f"{int(d):02d}/{int(mo):02d}/{y}"


def _extract_order_date(header_blocks: list[tuple[list[int], str, str]]) -> str:
    for _bbox, _label, inner in header_blocks:
        text = _cell_text(inner)
        m = _DATE_SEARCH_RE.search(text)
        if m:
            return _normalize_date(*m.groups())
    return ""


def _looks_like_order_no(val: str) -> bool:
    if _is_placeholder(val) or len(val) < 2:
        return False
    return bool(re.match(r"^[A-Za-z0-9][A-Za-z0-9\-/]*$", val))


def _extract_order_no(header_blocks: list[tuple[list[int], str, str]]) -> str:
    texts = [_cell_text(inner) for _bbox, _label, inner in header_blocks]
    for pattern in (_ORDER_NO_RE, _ORDER_NO_RE2):
        for text in texts:
            m = pattern.search(text)
            if m and _looks_like_order_no(m.group("val").strip()):
                return m.group("val").strip()
    return ""


def _extract_party_name(header_blocks: list[tuple[list[int], str, str]]) -> str:
    # Priority 1: an explicit "Party Name" field -- this business's own
    # ESSA-style printed grid forms (sample 5/3/13-scanned all confirmed).
    # Only the first line of the block counts (a second <br/> line is
    # typically the buyer's city, not part of the name -- confirmed on
    # sample 3-scanned's "Party Name A.T. Dhirendrababu<br/>HYDERABAD").
    for _bbox, _label, inner in header_blocks:
        first_line = re.split(r"<br\s*/?>", inner, maxsplit=1, flags=re.I)[0]
        text = _cell_text(first_line)
        m = _PARTY_NAME_RE.search(text)
        if m:
            val = _clean(m.group(1))
            if val:
                return val

    # Priority 2: a To/From booking-agent template with no "Party Name"
    # field at all (confirmed real, sample 12-scanned -- see CLAUDE.md's
    # documented reversed-role convention: "From" names the real buyer).
    value = _value_after_label(header_blocks, r"^from\s*:?$")
    return re.sub(r"^m/s\.?\s*", "", value, flags=re.I).strip()


def _extract_seller_name(header_blocks: list[tuple[list[int], str, str]]) -> str:
    candidates = []
    for _bbox, _label, inner in header_blocks:
        text = _cell_text(inner)
        if not text or _is_placeholder(text) or _EXCLUDE_PREFIX_RE.match(text):
            continue
        norm = re.sub(r"[^A-Za-z ]", "", text).strip()
        if not norm or _GENERIC_TITLE_RE.match(norm):
            continue
        # Priority 1: data-label alone (Section-Header) -- matches when
        # Chandra used it (sample 5-scanned first probe, sample 2.jpeg).
        if _label == "Section-Header":
            candidates.append(("A", text))
            continue
        # Priority 2: emphasis/heading markup even under a plain "Text"
        # label -- confirmed necessary, sample 3-scanned's/4-scanned's own
        # letterhead company name came back as data-label="Text" with an
        # <h1>/<h2>/<b> tag rather than "Section-Header". An ordinary
        # address/contact line under the same letterhead never carries
        # this markup on any form tested -- that's what tells them apart,
        # not a company-name keyword list (which wouldn't generalize past
        # this business's own "GARMENTS"/"LIMITED" wording).
        if _HEADING_TAG_RE.search(inner) and len(norm) >= 6:
            candidates.append(("B", text))
    if candidates:
        # Prefer Section-Header hits; within a tier, the longest text
        # (confirmed necessary: sample 3-scanned has both "<h1>ESSA</h1>"
        # and the fuller "<h2>ESSA GARMENTS PRIVATE LIMITED</h2>").
        candidates.sort(key=lambda c: (c[0], -len(c[1])))
        return candidates[0][1]

    # Fallback: a To/From template's own "To" (recipient/seller) field --
    # confirmed real, sample 12-scanned, where "ORDER FORM" itself is
    # mislabeled Section-Header (filtered above by _GENERIC_TITLE_RE) and
    # the actual seller name has no heading markup at all.
    return _value_after_label(header_blocks, r"^to,?$")


def _bbox_overlaps(b1: list[int], b2: list[int]) -> bool:
    x_overlap = b1[0] < b2[2] and b2[0] < b1[2]
    y_overlap = b1[1] < b2[3] and b2[1] < b1[3]
    return x_overlap or y_overlap


def _value_after_label(blocks: list[tuple[list[int], str, str]], label_pattern: str, window: int = 6) -> str:
    """Finds the value block belonging to a short standalone label block
    (e.g. "To," / "From :") -- used only for templates with no combined
    "Label <u>value</u>" block to regex directly. Candidates are any
    nearby block (within `window` blocks, following document order) whose
    bbox overlaps the label's own bbox on EITHER axis: confirmed both
    layouts appear on the same real form (sample 12-scanned) -- "To,"'s
    value sits in the same row, offset well to the right ("From :"'s
    value sits in the same column, stacked below with a small gap).
    Ties/near-misses are broken by picking the LONGEST surviving text,
    which reliably prefers a real name over a short label fragment
    ("M/s.") that also happens to sit nearby -- confirmed necessary on
    that exact form."""
    label_re = re.compile(label_pattern, re.I)
    for i, (bbox, label, inner) in enumerate(blocks):
        if label == "Table":
            break
        text = _cell_text(inner).strip()
        if not label_re.match(text):
            continue
        candidates = []
        for bbox2, label2, inner2 in blocks[i + 1 : i + 1 + window]:
            if label2 == "Table":
                break
            if not _bbox_overlaps(bbox, bbox2):
                continue
            value = _cell_text(inner2).strip()
            if not value or _is_placeholder(value) or label_re.match(value) or _LABEL_FRAGMENT_RE.match(value):
                continue
            candidates.append(value)
        if candidates:
            return max(candidates, key=len)
    return ""


# ------------------------------------------------------------- ditto marks

# Confirmed real (sample 5-scanned): "Fairleady Print" then next row
# "— " — Plain" -> "Fairleady Plain" (replace the PREVIOUS row's own
# trailing modifier word with this row's new one, not append to it -- see
# extract_claude.py's own DITTO MARKS rule/example, which this mirrors).
# A bare ditto with no modifier ("— " —") inherits the previous name
# unchanged (confirmed real: "MYNA" -> "— " —" stays "MYNA", only the
# Style column changes on that row). Deliberately its own implementation,
# not a call into ExtractedForm's own _forward_fill_ditto_item_names --
# that validator only fills an EMPTY item string; Chandra transcribes the
# ditto mark itself as real (non-empty) text, so that validator alone
# would never fire here.
_DITTO_MARKER_RE = re.compile(r'^([\s\-–—"\'`]{2,})(.*)$')


def _expand_ditto(item_text: str, prev_item: str) -> str:
    text = item_text.strip()
    m = _DITTO_MARKER_RE.match(text)
    if not m or not prev_item:
        return item_text
    modifier = m.group(2).strip()
    if not modifier:
        return prev_item
    base = prev_item.rsplit(" ", 1)[0] if " " in prev_item else prev_item
    return f"{base} {modifier}"


# ------------------------------------------------------ bullet-mark strip

# Free-form pages number/bullet each item by hand (e.g. "12 *) BABYCARE
# JETTY IE", "*) COOLD COLOUR RN") -- confirmed real, sample 2.jpeg, every
# item block. Not part of the item name.
_BULLET_RE = re.compile(r"^\s*\d*\s*\*\)\s*")


# ----------------------------------------------------------- table parsing

_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.S)
_THEAD_RE = re.compile(r"<thead[^>]*>(.*?)</thead>", re.S)
_TBODY_RE = re.compile(r"<tbody[^>]*>(.*?)</tbody>", re.S)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_TH_RE = re.compile(r'<th((?:\s+[\w-]+="[^"]*")*)\s*>(.*?)</th>', re.S)
_TD_RE = re.compile(r'<td((?:\s+[\w-]+="[^"]*")*)\s*>(.*?)</td>', re.S)
_COLSPAN_RE = re.compile(r'colspan="(\d+)"')

_STYLE_HEADER_RE = re.compile(r"\bstyle\b", re.I)
_SERIAL_HEADER_RE = re.compile(r"^(sr\.?|s\.?|sl\.?)\s*no\.?$", re.I)
_TOTAL_HEADER_RE = re.compile(r"\btotal\b", re.I)

# A page's own signature/sign-off band ("Stock Entry By:", "Checked By:",
# etc., below the real item grid) that Chandra sometimes hallucinates as an
# ordinary body <tr> -- see _parse_grid_tables's own comment on the
# sample 4-scanned case this was confirmed against.
_FOOTER_LABEL_RE = re.compile(r"^(stock\s*entry|checked|received|approved|prepared|verified|packed|dispatched)\s*by\s*:?\s*$", re.I)


def _get_colspan(attrs: str) -> int:
    m = _COLSPAN_RE.search(attrs)
    return int(m.group(1)) if m else 1


def _split_table(table_inner: str) -> tuple[str, str]:
    """(thead_html, tbody_html) for one <table>'s inner HTML -- tolerates
    either wrapper being entirely absent, since Chandra's own multi-table
    split (see module docstring) produces a header-only <table> (a
    <thead> with no <tbody>) and a body-only <table> (a <tbody> with no
    <thead>) as two SEPARATE blocks on some forms."""
    thead_m = _THEAD_RE.search(table_inner)
    if thead_m:
        thead_html = thead_m.group(1)
        remainder = table_inner[: thead_m.start()] + table_inner[thead_m.end() :]
    else:
        thead_html = ""
        remainder = table_inner
    tbody_m = _TBODY_RE.search(remainder)
    tbody_html = tbody_m.group(1) if tbody_m else remainder
    return thead_html, tbody_html


def _expand_header_row(tr_html: str) -> list[str]:
    """Flat list of header cell texts, a colspan="N" cell repeated N times
    so its position lines up with the N real body columns it visually
    spans (confirmed necessary: sample 12-scanned's `<th colspan="2">
    Rate</th>` sits above two real body <td> cells, "Per"/"Box"). rowspan
    is a non-issue here -- see _parse_grid_tables's own comment on why
    only the FIRST <thead> row is ever read at all."""
    cells = []
    for attrs, inner in _TH_RE.findall(tr_html):
        text = _cell_text(inner)
        cells.extend([text] * max(_get_colspan(attrs), 1))
    return cells


def _split_header_row(flat: list[str]) -> tuple[list[str], list[str], list[str]]:
    n = len(flat)
    i = 0
    while i < n and not flat[i].strip().isdigit():
        i += 1
    j = i
    while j < n and flat[j].strip().isdigit():
        j += 1
    return flat[:i], [flat[k].strip() for k in range(i, j)], flat[j:]


def _classify_leading_roles(leading: list[str]) -> tuple[int | None, int | None]:
    """(item_idx, type_idx) among the leading non-size header columns --
    generalized from 3 real, structurally different forms: 2 columns with
    an explicit Style column (sample 5-scanned: Particulars/Style), 3
    columns with a serial number (sample 13-scanned: Sr. No/Article
    Name/Style), and 2 columns with NEITHER a Style nor a serial column
    (sample 12-scanned: QUALITY/SHADE SHAPE -- confirmed by its own real
    body data that SHADE SHAPE, the more varied/descriptive column, is
    the real item name; QUALITY, a short fabric-grade code repeated
    across many rows, plays the type/style role instead)."""
    style_idx = next((i for i, h in enumerate(leading) if _STYLE_HEADER_RE.search(h)), None)
    remaining = [i for i, h in enumerate(leading) if i != style_idx and not _SERIAL_HEADER_RE.match(h.strip())]
    if not remaining:
        return None, style_idx
    item_idx = remaining[-1]
    type_idx = style_idx if style_idx is not None else (remaining[0] if len(remaining) > 1 else None)
    return item_idx, type_idx


def _parse_quantity_cell(text: str, header: str) -> tuple[str, int] | None:
    """(size_key, qty) for an ordinary cell (size_key == header) or a SIZE
    LABEL OVERRIDE cell (size_key is whatever label was actually written,
    per extract_claude.py's own rule of the same name) -- surfaced by
    Chandra as a two-token cell once whitespace/<br> collapses to a
    space, mirroring hybrid_quantities_lighton.py's own _parse_cell for
    the same real-world quirk on a different model's output. Anything
    else ("x", "-", a bare letter size with no adjoining quantity --
    still-open gap, see module docstring) returns None: no quantity, not
    a guess."""
    text = text.strip()
    if re.fullmatch(r"\d+", text):
        return header, int(text)
    tokens = text.split()
    if len(tokens) == 2 and tokens[0].isdigit() and tokens[1].isdigit():
        return tokens[0], int(tokens[1])
    return None


def _interpolate_row_fracs(table_top: float, table_bottom: float, n_header_rows: int, n_body_rows: int) -> list[tuple[float, float]]:
    """Per-row row_top_frac/row_bottom_frac when Chandra's own output
    carries no per-row bbox at all (confirmed: only the outer <table>
    block gets a bbox: header rows and body rows share it with no further
    breakdown). An even split, reserving `n_header_rows` units of space
    for the header band first, is a coarse approximation -- but
    row_top_frac/row_bottom_frac are documented (extract_ollama_cloud.py)
    as only ever needed for a coarse top-to-bottom ORDERING by the
    production hybrid-quantities backend (LightOnOCR-2), not pixel-exact
    crops; the one thing that matters is a strictly-increasing, well-
    formed partition, which this guarantees by construction."""
    total_units = max(n_header_rows, 0) + max(n_body_rows, 1)
    span = table_bottom - table_top
    unit = span / total_units if total_units else 0.0
    return [(table_top + unit * (n_header_rows + i), table_top + unit * (n_header_rows + i + 1)) for i in range(n_body_rows)]


def _parse_grid_tables(table_blocks: list[tuple[list[int], str, str]]) -> tuple[list[str], list[dict], list[str], float, float]:
    thead_chunks: list[str] = []
    tbody_chunks: list[str] = []
    ys = [y for bbox, _l, _i in table_blocks for y in (bbox[1], bbox[3])]
    table_top, table_bottom = _frac(min(ys)), _frac(max(ys))

    for _bbox, _label, inner in table_blocks:
        for table_inner in _TABLE_RE.findall(inner):
            thead_html, tbody_html = _split_table(table_inner)
            if thead_html.strip():
                thead_chunks.append(thead_html)
            if tbody_html.strip():
                tbody_chunks.append(tbody_html)

    leading_headers: list[str] = []
    size_headers: list[str] = []
    trailing_headers: list[str] = []
    n_header_rows = 0
    for thead_html in thead_chunks:
        rows = _TR_RE.findall(thead_html)
        n_header_rows = max(n_header_rows, len(rows))
        if rows and not size_headers:
            flat = _expand_header_row(rows[0])
            if flat:
                leading_headers, size_headers, trailing_headers = _split_header_row(flat)

    item_idx, type_idx = _classify_leading_roles(leading_headers)
    total_idx = next((i for i, h in enumerate(trailing_headers) if _TOTAL_HEADER_RE.search(h)), None)
    n_leading = len(leading_headers)
    n_size = len(size_headers)
    n_trailing = len(trailing_headers)
    # A merged label cell spanning roughly half the table or more is a
    # footer/summary row ("Old Rate Supply only", "GRAND TOTAL"), not a
    # real item -- confirmed real on 2 different forms with 2 different
    # shapes: a single <td colspan="15"> (sample 5-scanned) and a 3-cell
    # row where only the FIRST cell is merged, colspan="23", with the
    # actual grand-total number and unit as two ordinary trailing cells
    # after it (sample 13-scanned) -- the earlier "exactly one cell"
    # check caught only the first shape and let the second one through as
    # a fake item (item="512", type="BOXES").
    _footer_colspan_floor = max((n_leading + n_size) // 2, 3)

    body_trs: list[str] = []
    for tbody_html in tbody_chunks:
        body_trs.extend(_TR_RE.findall(tbody_html))
    row_fracs = _interpolate_row_fracs(table_top, table_bottom, n_header_rows, len(body_trs))

    items: list[dict] = []
    notes: list[str] = []
    for row_idx, tr_html in enumerate(body_trs):
        cells = _TD_RE.findall(tr_html)
        if not cells:
            continue
        texts = [_cell_text(inner) for _attrs, inner in cells]
        max_colspan = max(_get_colspan(attrs) for attrs, _inner in cells)
        # A second footer shape, distinct from the merged-colspan one above:
        # confirmed real on sample 4-scanned, where Chandra hallucinates the
        # page's own "Stock Entry By:"/"Checked By:" signature band (below
        # the real grid, with its own stray tally marks and a circled total)
        # as an ordinary, un-merged <tr> -- normal individual <td> cells, so
        # the colspan check above never fires. It landed as a fake item
        # ("Stock Entry By :") with 10 fabricated quantities of "1" pulled
        # from marks that belong to that band, not to any item -- and pushed
        # the REAL next row (the last item on the page) down one slot with
        # its own quantities now misattributed to the phantom row above it.
        # Matched on the item-column TEXT alone, not position, since this
        # phantom row can appear anywhere a signature block sits relative to
        # the real last row.
        item_probe = texts[item_idx] if item_idx is not None and item_idx < len(texts) else ""
        is_footer_row = max_colspan >= _footer_colspan_floor or _FOOTER_LABEL_RE.match(item_probe.strip())
        if is_footer_row:
            if max_colspan >= _footer_colspan_floor:
                label_cell = next((c for c in cells if _get_colspan(c[0]) == max_colspan), cells[0])
                text = _clean(_cell_text(label_cell[1]))
            else:
                text = _clean(item_probe)
            if text:
                notes.append(text)
            continue

        item_text = item_probe
        type_text = texts[type_idx] if type_idx is not None and type_idx < len(texts) else ""
        # Leading and trailing cells are trusted by position from the
        # START and END of the row respectively; whatever falls between
        # them is zipped against size_headers left-aligned. Deliberately
        # NOT a fixed texts[n_leading:n_leading+n_size] slice -- confirmed
        # real on sample 13-scanned: Chandra's own body rows there
        # consistently carry ONE MORE cell in the size-column region than
        # its own header row lists (a genuine Chandra quirk, not a
        # counting bug here), which silently pushed row_total one column
        # left on every single row under the old fixed-offset slice. A
        # short zip just drops the one extra (empty, in every row
        # checked) middle cell instead of corrupting every column's
        # alignment.
        if n_trailing and len(texts) >= n_leading + n_trailing:
            size_cells = texts[n_leading : len(texts) - n_trailing]
            trailing_cells = texts[len(texts) - n_trailing :]
        else:
            size_cells = texts[n_leading:]
            trailing_cells = []
        row_total = trailing_cells[total_idx] if total_idx is not None and total_idx < len(trailing_cells) else ""

        quantities: dict[str, int] = {}
        for header, cell_text in zip(size_headers, size_cells):
            parsed = _parse_quantity_cell(cell_text, header)
            if parsed is not None:
                sk, qty = parsed
                quantities[sk] = qty

        item_name = _expand_ditto(item_text.strip(), items[-1]["item"] if items else "")
        top, bottom = row_fracs[row_idx]
        items.append({
            "item": item_name,
            "type": _clean(type_text),
            "quantities": quantities,
            "row_total": _clean(row_total),
            "row_top_frac": top,
            "row_bottom_frac": bottom,
        })

    return size_headers, items, notes, table_top, table_bottom


# --------------------------------------------------------- free-form path

_FRAC_TAG_RE = re.compile(r"<math>\s*\\frac\{(\d+)\}\{(\d+)\}\s*</math>")
_PLAIN_MATH_RE = re.compile(r"<math>\s*(\d+)\s*/\s*(\d+)\s*</math>")
_PAIR_RE = re.compile(r"(?<!\d)(\d{1,3})\s*/\s*(\d{1,3})(?!\d)")
_ITEM_SUFFIX_RE = re.compile(r"\b(" + "|".join(re.escape(c) for c in KNOWN_STYLE_CODES) + r")\s*$", re.I)


def _normalize_math(inner_html: str) -> str:
    text = _FRAC_TAG_RE.sub(lambda m: f"{m.group(1)}/{m.group(2)}", inner_html)
    text = _PLAIN_MATH_RE.sub(lambda m: f"{m.group(1)}/{m.group(2)}", text)
    return text


def _extract_pairs(text: str) -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in _PAIR_RE.finditer(text)]


def _clean_item_name(text: str) -> str:
    return _BULLET_RE.sub("", text).strip()


def _looks_like_item_name(text: str) -> bool:
    return bool(_ITEM_SUFFIX_RE.search(text))


def _parse_freeform_blocks(blocks: list[tuple[list[int], str, str]]) -> tuple[list[dict], float, float]:
    """FREE-FORM LAYOUT (no printed grid, e.g. sample 2.jpeg): every item
    is its own text block, immediately followed by one or more pure-
    quantity blocks (a wrapped continuation line is its own block with no
    new item name -- confirmed real, sample 2.jpeg's "BABYCARE DRAWER"
    row). Blocks are consumed strictly in document order; a block with
    fraction-shaped pairs and no other letters is treated as a
    continuation of whatever item is currently open, never a new one.

    A name-only block that never accumulates any quantity pairs and
    doesn't end in a known style code is dropped rather than kept as an
    empty item -- confirmed necessary: this business's own letterhead
    ("MURUGAN TIE CHENNAI") and a pre-printed diary label ("MONDAY", the
    documented date-fabrication test case on this exact form) both come
    through as ordinary multi-word text blocks with no fraction pairs,
    and would otherwise show up as fake empty product rows. The real
    tradeoff: a genuinely blank-quantity item with no style-code suffix
    is dropped too rather than kept -- accepted here since every item
    name on every free-form form tested so far ends in one."""
    parsed = []
    for bbox, _label, inner in blocks:
        norm = _normalize_math(inner)
        text = _cell_text(norm)
        if not text or _is_placeholder(text):
            continue
        pairs = _extract_pairs(text)
        residual = _PAIR_RE.sub(" ", text)
        has_letters = bool(re.sub(r"[^A-Za-z]", "", residual))
        parsed.append({"bbox": bbox, "text": text, "pairs": pairs, "has_letters": has_letters})

    items: list[dict] = []
    current: dict | None = None
    for blk in parsed:
        if blk["pairs"] and not blk["has_letters"] and current is not None:
            current["quantities"].extend(blk["pairs"])
            current["bottom"] = max(current["bottom"], blk["bbox"][3])
            continue

        name_text = _clean_item_name(blk["text"])
        if not name_text:
            continue
        current = {
            "item": name_text,
            "quantities": list(blk["pairs"]),
            "top": blk["bbox"][1],
            "bottom": blk["bbox"][3],
        }
        items.append(current)

    items = [it for it in items if it["quantities"] or _looks_like_item_name(it["item"])]

    if not items:
        return [], 0.0, 0.0
    table_top = _frac(min(it["top"] for it in items))
    table_bottom = _frac(max(it["bottom"] for it in items))
    out = []
    for it in items:
        quantities: dict[str, int] = {}
        for size, qty in it["quantities"]:
            quantities[size] = qty
        out.append({
            "item": it["item"],
            "type": "",
            "quantities": quantities,
            "row_total": "",
            "row_top_frac": _frac(it["top"]),
            "row_bottom_frac": _frac(it["bottom"]),
        })
    return out, table_top, table_bottom


# ----------------------------------------------------------------- public

def parse_chandra_output(raw_text: str) -> ExtractedForm:
    blocks = _iter_blocks(raw_text)
    table_blocks = [b for b in blocks if b[1] == "Table"]
    header_blocks = blocks if not table_blocks else [b for b in blocks if b[1] != "Table"]

    seller_name = _extract_seller_name(header_blocks)
    party_name = _extract_party_name(header_blocks)
    order_no = _extract_order_no(header_blocks)
    order_date = _extract_order_date(header_blocks)

    if table_blocks:
        size_headers, items_raw, notes, table_top, table_bottom = _parse_grid_tables(table_blocks)
    else:
        size_headers = []
        items_raw, table_top, table_bottom = _parse_freeform_blocks(blocks)
        notes = []

    items = [
        ExtractedItem(
            item=it["item"],
            type=it["type"],
            quantities=[QuantityPair(size=size, quantity=qty) for size, qty in it["quantities"].items()],
            row_total=it["row_total"],
            row_top_frac=it["row_top_frac"],
            row_bottom_frac=it["row_bottom_frac"],
        )
        for it in items_raw
    ]

    return ExtractedForm(
        seller_name=seller_name,
        party_name=party_name,
        order_no=order_no,
        order_date=order_date,
        size_headers=size_headers,
        items=items,
        notes=notes,
        table_top_frac=table_top,
        table_bottom_frac=max(table_bottom, table_top + 0.01),
    )
