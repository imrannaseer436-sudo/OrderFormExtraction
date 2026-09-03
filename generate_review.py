#!/usr/bin/env python3
"""
Generates a self-contained HTML review page from one image's extraction
output -- the "Review UI" noted as not-yet-built in CLAUDE.md. Everything
(source photo, extracted data) is embedded inline so the file works by
double-clicking it, no server or network needed, matching how
`review.csv` already works today.

Renders the item x size grid in the same shape as the physical form
(Particulars/Style down the left, size columns across the top) since
that's the fastest way for a human to visually cross-check output against
the photo it came from -- side by side, not as a flat list. Quantity and
item-name/style cells are editable inputs; an "Export corrected JSON"
button reconstructs the OrderForm JSON from whatever is currently in the
page (including edits) and downloads it, so a reviewer can fix a handful
of cells here rather than hand-editing the raw JSON file.

Usage:
    python3 generate_review.py "sample 5"
    python3 generate_review.py "sample 5" --outdir extracted --images Images
"""

import argparse
import base64
import html
import json
from pathlib import Path

try:
    from brandlist_match import SUGGEST_SCORE_THRESHOLD
except Exception:
    SUGGEST_SCORE_THRESHOLD = 70.0  # brandlist_match not importable (e.g. pyodbc/rapidfuzz not
                                     # installed here) -- fine, this is only used to filter which
                                     # already-computed notes are worth showing, no DB call needed

IMAGE_EXTENSIONS = {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png", ".webp": "webp", ".gif": "gif"}


def find_source_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_size_headers(order_form: dict, stage_a: dict | None) -> list[str]:
    headers = list(stage_a["size_headers"]) if stage_a else []
    seen = set(headers)
    extra = set()
    for item in order_form["items"]:
        extra.update(item.get("quantities", {}).keys())
    # keep any size the model actually used but the header list somehow
    # missed, appended in numeric order after the real headers
    for size in sorted(extra - seen, key=lambda s: (len(s), s)):
        headers.append(size)
    return headers


def _brandlist_note_html(note: dict | None) -> str:
    """One compact line summarizing brandlist_match.py's suggestion for this
    row, if there is one worth showing -- never auto-applied to item.item,
    so this is the human's chance to see and act on it. Empty string if
    there's nothing (no match found, or match too weak to be useful).

    2026-08-08 fix: a plain score check here was hiding real matches --
    confirmed on sample 4.jpeg's 'CRERO B4756 H/PANT' (score 51.6) and
    'LOOPER ZB3965 H/PANT' (score 69.4), both genuine unique-code matches
    (a digit code that narrows the catalog to exactly ONE product -- see
    brandlist_match.py's AUTO_APPLY_UNIQUE_CODE_SCORE comment for why a low
    text score there means the surrounding words were misread, not that the
    product identity is in doubt). annotate_and_resolve() already documents
    this exact bypass ("not just the auto-apply one") but this display gate
    never implemented it -- it only checked the raw score. Now mirrors the
    same _has_unique_code_match rule the backend uses."""
    if not note or not note.get("match"):
        return ""
    match = note["match"]
    is_unique_code_match = match.get("code_narrowed") and match.get("narrowed_pool_size") == 1
    if match["score"] < SUGGEST_SCORE_THRESHOLD and not is_unique_code_match:
        return ""

    parts = [f'DB match: {html.escape(match["bname"])} (score {match["score"]})']
    if is_unique_code_match and match["score"] < SUGGEST_SCORE_THRESHOLD:
        parts.append("⚠ low text score, but a product code uniquely narrows it to this one item")
    if match.get("styles"):
        parts.append(f'styles: {", ".join(html.escape(s) for s in match["styles"])}')
    if note.get("code_not_in_catalog"):
        parts.append(f'⚠ code "{html.escape(note["code_not_in_catalog"])}" not in catalog')
    if note.get("style_mismatch"):
        parts.append(f'⚠ style mismatch, catalog says: {", ".join(html.escape(s) for s in note["style_mismatch"])}')
    if note.get("type_filled_from_catalog"):
        parts.append(f'✓ type auto-filled: {html.escape(note["type_filled_from_catalog"])}')
    if note.get("resolved_sizes"):
        resolved = ", ".join(f"{html.escape(k)}={v}" for k, v in note["resolved_sizes"].items())
        parts.append(f'✓ letter sizes resolved: {resolved}')
    if note.get("unresolved_sizes"):
        parts.append(f'⚠ unresolved size letters: {", ".join(html.escape(s) for s in note["unresolved_sizes"])}')
    if note.get("size_conflicts"):
        conflicts = ", ".join(f"{html.escape(k)}→{v}" for k, v in note["size_conflicts"].items())
        parts.append(f'⚠ size conflict, needs manual check: {conflicts}')
    if note.get("likely_column_shift"):
        shift = note["likely_column_shift"]
        correction = ", ".join(f"{html.escape(k)}→{v}" for k, v in shift["suggested_correction"].items())
        parts.append(f'⚠⚠ likely column shift (offset {shift["offset"]:+d}), check against photo: {correction}')
    elif note.get("sizes_outside_catalog_range"):
        sizes = ", ".join(str(s) for s in note["sizes_outside_catalog_range"])
        parts.append(f'⚠ size(s) not in this product\'s catalog range: {sizes}')
    return f'<div class="db-note">{" &middot; ".join(parts)}</div>'


def _sorted_sizes(sizes: list[str]) -> list[str]:
    return sorted(sizes, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))


def _friendly_recount_summary(flag: dict) -> str:
    """A short, plain-language sentence for someone checking this form
    against the photo -- not an engineer. The full technical note (which
    exact numbers each of the two independent reads produced) is still
    available as a hover tooltip on both this line and the flagged cells
    themselves; this line is just the "what should I do" summary a general
    reviewer actually needs at a glance."""
    sizes = ", ".join(_sorted_sizes(flag.get("sizes", [])))
    status = flag.get("status")
    note = flag.get("note", "")
    # Flags from extract_ollama_cloud.py's hybrid-OCR total-mismatch and
    # catalog checks (2026-08-22) already write a specific, plain-language
    # reason into their own note text -- showing it directly is more
    # accurate than the recount-pass phrasing below, which describes a
    # DIFFERENT mechanism (two independent VLM reads disagreeing) these
    # flags never go through.
    if note.startswith("Hybrid OCR+VLM:") or note.startswith("Catalog check:"):
        return f"⚠️ {html.escape(note)}"
    if status == "unresolved":
        return f"⚠️ Please check sizes {sizes} against the photo — two automatic checks disagreed and we couldn't tell which is right."
    if status == "resolved":
        return f"⚠️ Worth a quick check on sizes {sizes} — two automatic checks disagreed, so we picked the one that matched the order total or product catalog."
    if status == "auto_corrected":
        return f"⚠️ Please verify sizes {sizes} — both checks agreed, but the sizes looked shifted, so we corrected them to match the product catalog."
    if status == "unverified":
        return "This row's quantities were not double-checked — please verify against the photo."
    return html.escape(flag.get("note", ""))


def _recount_note_html(flag: dict | None) -> str:
    """Row-level detail line for extract_claude.py's recount pass (2026-08-08)
    -- separate from _brandlist_note_html's DB cross-check line, since these
    come from a different signal (main vs. row-crop recount agreement,
    catalog-grounded drift correction) and can both be present on one row.
    Shows a plain-language summary (_friendly_recount_summary) rather than
    the raw technical note -- the raw note (main=X vs recount=Y for every
    differing size) is still reachable via this line's own tooltip and via
    hovering the flagged cells directly, for anyone who wants the detail."""
    if not flag or flag.get("status") in (None, "ok", "no_recount") or not flag.get("note"):
        return ""
    summary = _friendly_recount_summary(flag)
    title_attr = f' title="{html.escape(flag["note"])}"' if flag.get("note") else ""
    return f'<div class="recount-note recount-{html.escape(flag["status"])}"{title_attr}>{summary}</div>'


def _party_check_html(party_check: dict | None) -> str:
    """Banner for brandlist_match.resolve_party_name()'s output -- flags
    when a form's letterhead isn't Essa and the handwritten Party Name
    doesn't look like it matches Essa's own buyer records either (or
    matches a DIFFERENT one than the letterhead does). Whole-form-level,
    not per-row, so it renders once near the Party Name field rather than
    per item. Empty string when there's nothing to flag (resolve_party_name
    already returns None for the common, unproblematic cases)."""
    if not party_check or not party_check.get("note"):
        return ""
    parts = [html.escape(party_check["note"])]
    if party_check.get("suggested_party_name"):
        parts.append(f'Suggested party: <strong>{html.escape(party_check["suggested_party_name"])}</strong>')
    return f'<div class="party-check-banner">⚠️ {" ".join(parts)}</div>'


def render_html(order_form: dict, stage_a: dict | None, headers: list[str], image_path: Path | None, source_name: str, brandlist: list[dict] | None = None, recount_flags: list[dict] | None = None, party_check: dict | None = None) -> str:
    if image_path is not None:
        ext = IMAGE_EXTENSIONS[image_path.suffix.lower()]
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        image_data_uri = f"data:image/{ext};base64,{b64}"
    else:
        image_data_uri = ""

    seller = html.escape(stage_a.get("seller_name", "")) if stage_a else ""
    party = html.escape(order_form.get("party_name", ""))
    order_no = html.escape(order_form.get("order_no", ""))
    order_date = html.escape(order_form.get("order_date", ""))
    # The recount pass's own technical notes (e.g. "NEEDS REVIEW (UNRESOLVED)
    # -- 'X': main reading and row-crop recount disagree (50: main=1 vs
    # recount=-; ...)") are already shown per-row, in plain language, right
    # next to the flagged cells they describe (_recount_note_html) -- much
    # more useful there than as a wall of technical text at the top of the
    # page. Excluded here by exact text match against recount_flags so the
    # top Notes field is left for genuine page-level notes only (e.g.
    # handwritten remarks, GST marks) -- what a general reviewer actually
    # wants from a page-level "Notes" field.
    recount_note_texts = {f["note"] for f in (recount_flags or []) if f.get("note")}
    page_notes = [n for n in order_form.get("notes", []) if n not in recount_note_texts]
    notes = "; ".join(page_notes)

    header_cells = "".join(f'<th class="size-col">{html.escape(h)}</th>' for h in headers)

    flagged_cell_count = 0
    rows_html = []
    for ri, item in enumerate(order_form["items"]):
        qty = item.get("quantities", {})
        is_empty_row = len(qty) == 0
        flag = recount_flags[ri] if recount_flags and ri < len(recount_flags) else None
        flagged_sizes = set(flag["sizes"]) if flag and flag.get("sizes") else set()
        cell_status = flag["status"] if flag and flag.get("sizes") else None

        cells = []
        for h in headers:
            val = qty.get(h, "")
            is_flagged = h in flagged_sizes
            cell_class = f"qty-cell flag-{cell_status}" if is_flagged else "qty-cell"
            title_attr = f' title="{html.escape(flag["note"])}"' if is_flagged and flag.get("note") else ""
            cells.append(
                f'<td class="{cell_class}"{title_attr}><input type="number" min="0" inputmode="numeric" '
                f'data-row="{ri}" data-size="{html.escape(h)}" value="{val}" '
                f'placeholder="–" /></td>'
            )
            if is_flagged:
                flagged_cell_count += 1
        row_class = "row-empty" if is_empty_row else ""
        brandlist_note = brandlist[ri] if brandlist and ri < len(brandlist) else None
        note_html = _brandlist_note_html(brandlist_note)
        note_html += _recount_note_html(flag)

        # Shift controls: ◀/▶ always available (nudge every currently-filled
        # cell in this row one header-column left/right, operating on
        # whatever's live in the DOM right now -- including prior manual
        # edits), plus a one-click "Fix" button pre-loaded with the exact
        # offset when brandlist_match.detect_column_shift already suggested
        # one for this row (see likely_column_shift note above) -- so the
        # human doesn't have to re-derive a number this page already knows.
        suggested_shift = brandlist_note.get("likely_column_shift") if brandlist_note else None
        suggested_offset = suggested_shift["offset"] if suggested_shift else None
        fix_btn = (
            f'<button type="button" class="shift-btn shift-suggested" data-row="{ri}" data-offset="{suggested_offset}" '
            f'title="Apply the suggested {suggested_offset:+d}-column shift">Fix {suggested_offset:+d}</button>'
            if suggested_offset else ""
        )
        shift_controls = (
            '<div class="shift-controls">'
            f'<button type="button" class="shift-btn" data-row="{ri}" data-offset="-1" title="Shift this row\'s quantities one column left">◀</button>'
            f'<button type="button" class="shift-btn" data-row="{ri}" data-offset="1" title="Shift this row\'s quantities one column right">▶</button>'
            f'{fix_btn}'
            '</div>'
        )

        rows_html.append(
            f'<tr class="{row_class}" data-row="{ri}">'
            f'<td class="item-col"><input type="text" class="item-name" data-row="{ri}" '
            f'value="{html.escape(item["item"])}" />{shift_controls}{note_html}</td>'
            f'<td class="style-col"><input type="text" class="item-style" data-row="{ri}" '
            f'value="{html.escape(item.get("type", ""))}" /></td>'
            f'{"".join(cells)}'
            "</tr>"
        )

    order_form_json = json.dumps(order_form, ensure_ascii=False)
    headers_json = json.dumps(headers, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Review — {html.escape(source_name)}</title>
<style>
  :root {{
    --paper: #f4f5f6;
    --panel: #ffffff;
    --ink: #1c2024;
    --ink-soft: #5b6470;
    --rule: #d7dbe0;
    --accent: #2f5f8a;
    --accent-soft: #e7eef4;
    --warn: #a15c00;
    --warn-soft: #fbeedd;
    --focus: #2f5f8a;
    --flag-unresolved: #b3261e;
    --flag-unresolved-soft: #fbe4e1;
    --flag-resolved: #a15c00;
    --flag-resolved-soft: #fbeedd;
    --flag-corrected: #6a4c93;
    --flag-corrected-soft: #efe7f5;
    --flag-unverified: #5b6470;
    --flag-unverified-soft: #e7e9eb;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --paper: #1a1d21;
      --panel: #23272c;
      --ink: #e8eaed;
      --ink-soft: #9aa3ad;
      --rule: #383e45;
      --accent: #7fb3da;
      --accent-soft: #253544;
      --warn: #e0a446;
      --warn-soft: #3a2e1a;
      --focus: #7fb3da;
      --flag-unresolved: #e2665c;
      --flag-unresolved-soft: #402221;
      --flag-resolved: #e0a446;
      --flag-resolved-soft: #3a2e1a;
      --flag-corrected: #b79ce0;
      --flag-corrected-soft: #332a42;
      --flag-unverified: #9aa3ad;
      --flag-unverified-soft: #2c3036;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
  }}
  header.topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.75rem 1.25rem;
    background: var(--panel);
    border-bottom: 1px solid var(--rule);
    position: sticky;
    top: 0;
    z-index: 20;
  }}
  header.topbar h1 {{
    font-size: 1rem;
    margin: 0;
    font-weight: 600;
  }}
  header.topbar .meta {{
    color: var(--ink-soft);
    font-size: 0.82rem;
  }}
  button.export {{
    font: inherit;
    font-weight: 600;
    background: var(--accent);
    color: white;
    border: none;
    padding: 0.55rem 1.1rem;
    border-radius: 5px;
    cursor: pointer;
  }}
  button.export:hover {{ filter: brightness(1.08); }}
  button.export:active {{ filter: brightness(0.95); }}

  .layout {{
    display: grid;
    grid-template-columns: minmax(260px, 38%) 1fr;
    gap: 1px;
    background: var(--rule);
    min-height: calc(100vh - 54px);
  }}
  @media (max-width: 900px) {{
    .layout {{ grid-template-columns: 1fr; }}
  }}

  .image-pane {{
    background: var(--panel);
    padding: 0.75rem;
    overflow: auto;
    max-height: calc(100vh - 54px);
    position: sticky;
    top: 54px;
    align-self: start;
  }}
  .image-pane img {{
    width: 100%;
    height: auto;
    display: block;
    border: 1px solid var(--rule);
    border-radius: 4px;
  }}
  .image-pane .no-image {{
    color: var(--ink-soft);
    font-style: italic;
    padding: 2rem;
    text-align: center;
  }}

  .data-pane {{
    background: var(--panel);
    padding: 1rem 1.25rem 2rem;
    overflow: auto;
    max-height: calc(100vh - 54px); /* bounds this pane so it scrolls
      WITHIN itself (like .image-pane already does) instead of growing
      past the viewport and letting the whole PAGE scroll -- without this,
      table.grid's sticky thead has no real scrolling ancestor to stick
      within and the effect is easy to miss. */
  }}

  .order-meta {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.6rem 1.5rem;
    margin-bottom: 1rem;
    padding-bottom: 0.9rem;
    border-bottom: 1px solid var(--rule);
  }}
  .party-check-banner {{
    margin: -0.4rem 0 1rem;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    line-height: 1.45;
    background: var(--flag-resolved-soft);
    color: var(--ink);
    border: 1px solid var(--flag-resolved);
    border-radius: 5px;
  }}
  .order-meta .field label {{
    display: block;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--ink-soft);
    margin-bottom: 0.2rem;
  }}
  .order-meta .field .value {{
    font-size: 0.95rem;
    font-weight: 500;
  }}
  .order-meta .field.notes .value {{ font-weight: 400; color: var(--ink-soft); }}

  table.grid {{
    border-collapse: collapse;
    width: 100%;
    font-variant-numeric: tabular-nums;
  }}
  table.grid th, table.grid td {{
    border: 1px solid var(--rule);
    padding: 0;
  }}
  table.grid thead th {{
    background: var(--accent-soft);
    color: var(--ink);
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.5rem 0.4rem;
    position: sticky;
    top: 0; /* sticks to the top of .data-pane's OWN scroll box (now that
               .data-pane has a bounded max-height + overflow:auto above),
               not the page -- .data-pane's own top edge already sits
               below header.topbar, so no extra offset is needed here. */
    z-index: 10;
  }}
  table.grid thead th.item-col, table.grid thead th.style-col {{ text-align: left; }}
  table.grid thead th.size-col {{ text-align: center; min-width: 3.4rem; }}

  td.item-col {{ min-width: 180px; position: sticky; left: 0; background: var(--panel); z-index: 5; }}
  td.style-col {{ min-width: 60px; position: sticky; left: 180px; background: var(--panel); z-index: 5; }}

  .db-note {{
    font-size: 0.7rem;
    line-height: 1.35;
    color: var(--ink-soft);
    padding: 0 0.5rem 0.45rem;
    word-break: break-word;
  }}
  .recount-note {{
    font-size: 0.7rem;
    line-height: 1.35;
    padding: 0 0.5rem 0.45rem;
    word-break: break-word;
    font-weight: 600;
  }}
  .recount-unresolved {{ color: var(--flag-unresolved); }}
  .recount-resolved {{ color: var(--flag-resolved); }}
  .recount-auto_corrected {{ color: var(--flag-corrected); }}
  .recount-unverified {{ color: var(--flag-unverified); font-weight: 400; }}

  /* Quantity cells the recount pass thinks are worth a second look -- see
     _apply_recount()'s status meanings in extract_claude.py. Border, not
     just background, so it stays visible whether the cell is empty or
     filled, and cursor:help + title attr surface the specific reason on
     hover without needing to scroll to the row's note text below it. */
  td.qty-cell[class*="flag-"] {{ cursor: help; }}
  td.flag-unresolved {{ background: var(--flag-unresolved-soft); box-shadow: inset 0 0 0 2px var(--flag-unresolved); }}
  td.flag-resolved {{ background: var(--flag-resolved-soft); box-shadow: inset 0 0 0 2px var(--flag-resolved); }}
  td.flag-auto_corrected {{ background: var(--flag-corrected-soft); box-shadow: inset 0 0 0 2px var(--flag-corrected); }}
  td.flag-unverified {{ background: var(--flag-unverified-soft); box-shadow: inset 0 0 0 1px var(--flag-unverified); }}

  .shift-controls {{
    display: flex;
    gap: 0.3rem;
    padding: 0 0.5rem 0.4rem;
  }}
  button.shift-btn {{
    font: inherit;
    font-size: 0.72rem;
    line-height: 1;
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid var(--rule);
    border-radius: 4px;
    padding: 0.25rem 0.5rem;
    cursor: pointer;
  }}
  button.shift-btn:hover {{ filter: brightness(1.08); }}
  button.shift-btn.shift-suggested {{
    background: var(--flag-corrected-soft);
    color: var(--flag-corrected);
    border-color: var(--flag-corrected);
    font-weight: 600;
  }}

  table.grid input {{
    font: inherit;
    font-variant-numeric: tabular-nums;
    width: 100%;
    border: none;
    background: transparent;
    color: var(--ink);
    padding: 0.5rem 0.5rem;
  }}
  table.grid input:focus {{
    outline: 2px solid var(--focus);
    outline-offset: -2px;
    background: var(--accent-soft);
  }}
  td.qty-cell input {{ text-align: right; }}
  td.qty-cell input::placeholder {{ color: var(--ink-soft); opacity: 0.5; text-align: center; }}

  tr.row-empty td.item-col, tr.row-empty td.style-col {{ background: var(--warn-soft); }}
  tr.row-empty .item-name {{ color: var(--warn); }}

  input.edited {{ background: var(--accent-soft); }}

  footer.note {{
    margin-top: 1rem;
    font-size: 0.78rem;
    color: var(--ink-soft);
  }}
</style>
</head>
<body>
<header class="topbar">
  <div>
    <h1>{html.escape(source_name)}</h1>
    <div class="meta">{seller and f"Seller: {seller} &nbsp;·&nbsp; " or ""}Party: {party or "—"}{f' &nbsp;·&nbsp; <span style="color:var(--flag-unresolved);font-weight:600;">{flagged_cell_count} quantity cell(s) flagged for review</span>' if flagged_cell_count else ''}</div>
  </div>
  <button class="export" id="exportBtn">Export corrected JSON</button>
</header>

<div class="layout">
  <div class="image-pane">
    {f'<img src="{image_data_uri}" alt="Source order form" />' if image_data_uri else '<div class="no-image">Source image not found</div>'}
  </div>

  <div class="data-pane">
    <div class="order-meta">
      <div class="field"><label>Party name</label><div class="value" id="partyName" contenteditable="true">{party}</div></div>
      <div class="field"><label>Order no.</label><div class="value" id="orderNo" contenteditable="true">{order_no}</div></div>
      <div class="field"><label>Order date</label><div class="value" id="orderDate" contenteditable="true">{order_date}</div></div>
      <div class="field notes"><label>Notes</label><div class="value" id="notes" contenteditable="true">{html.escape(notes)}</div></div>
    </div>
    {_party_check_html(party_check)}

    <div style="overflow-x:auto">
      <table class="grid">
        <thead>
          <tr>
            <th class="item-col">Particulars</th>
            <th class="style-col">Style</th>
            {header_cells}
          </tr>
        </thead>
        <tbody id="gridBody">
          {"".join(rows_html)}
        </tbody>
      </table>
    </div>
    <footer class="note">
      Rows highlighted in amber have no quantities at all — check these first.
      Individual cells with a colored border are flagged as doubtful — hover a cell for the specific reason:
      <span style="color:var(--flag-unresolved);font-weight:600;">red</span> = a strong, specific problem (row doesn't sum to its own printed total, or two independent reads disagreed with no automatic resolution) — most important to check,
      <span style="color:var(--flag-resolved);font-weight:600;">amber</span> = a likely fix was found automatically (a probable column shift, or a disagreement resolved via Total Dozen/catalog) — worth a glance,
      <span style="color:var(--flag-corrected);font-weight:600;">purple</span> = both reads agreed but were auto-corrected against the product catalog (shared-bias check),
      <span style="color:var(--flag-unverified);font-weight:600;">gray</span> = a weaker signal (sizes outside the matched product's catalog range, or a row recount couldn't independently verify) — worth a glance.
      Each row also has ◀/▶ buttons to shift every filled cell in that row one size column left/right (for a column-drift misread), and a "Fix" button when the catalog check already found a specific likely offset.
      Edit any cell directly, then use "Export corrected JSON" to download the corrected file.
    </footer>
  </div>
</div>

<script>
const ORIGINAL_ORDER = {order_form_json};
const HEADERS = {headers_json};

document.querySelectorAll('table.grid input, .order-meta [contenteditable]').forEach(el => {{
  const evt = el.tagName === 'INPUT' ? 'input' : 'input';
  el.addEventListener(evt, () => el.classList.add('edited'));
}});

// Shifts every currently-filled quantity in row `rowIdx` by `offset` header
// columns, operating on whatever's live in the DOM right now (including
// prior manual edits) -- not the original extraction -- since that's what
// a reviewer expects "shift this row" to act on. A cell shifted past the
// first/last header would lose its value entirely, so that's confirmed
// before proceeding rather than silently dropped.
function shiftRow(rowIdx, offset) {{
  const tr = document.querySelector(`tr[data-row="${{rowIdx}}"]`);
  if (!tr) return;
  const inputsBySize = {{}};
  tr.querySelectorAll('.qty-cell input').forEach(inp => {{ inputsBySize[inp.dataset.size] = inp; }});
  const oldValues = HEADERS.map(h => (inputsBySize[h] ? inputsBySize[h].value.trim() : ''));
  if (!oldValues.some(v => v !== '')) return;

  const n = HEADERS.length;
  let wouldLoseData = false;
  for (let i = 0; i < n; i++) {{
    const j = i + offset;
    if (oldValues[i] !== '' && (j < 0 || j >= n)) wouldLoseData = true;
  }}
  if (wouldLoseData && !confirm(
    `Shifting by ${{offset > 0 ? '+' : ''}}${{offset}} would push at least one filled cell past the ` +
    `${{offset > 0 ? 'last' : 'first'}} size column and lose its value. Continue anyway?`
  )) return;

  const newValues = new Array(n).fill('');
  for (let i = 0; i < n; i++) {{
    const j = i + offset;
    if (j >= 0 && j < n) newValues[j] = oldValues[i];
  }}
  HEADERS.forEach((h, i) => {{
    const inp = inputsBySize[h];
    if (!inp) return;
    inp.value = newValues[i];
    inp.classList.add('edited');
  }});
}}

document.getElementById('gridBody').addEventListener('click', (e) => {{
  const btn = e.target.closest('.shift-btn');
  if (!btn) return;
  shiftRow(Number(btn.dataset.row), Number(btn.dataset.offset));
}});

document.getElementById('exportBtn').addEventListener('click', () => {{
  const items = [];
  document.querySelectorAll('#gridBody tr').forEach(tr => {{
    const row = tr.dataset.row;
    const name = tr.querySelector('.item-name').value.trim();
    const type = tr.querySelector('.item-style').value.trim();
    const quantities = {{}};
    tr.querySelectorAll('.qty-cell input').forEach(inp => {{
      const v = inp.value.trim();
      if (v !== '' && Number(v) > 0) quantities[inp.dataset.size] = Number(v);
    }});
    items.push({{ item: name, type: type, quantities: quantities }});
  }});

  const result = {{
    party_name: document.getElementById('partyName').textContent.trim(),
    order_no: document.getElementById('orderNo').textContent.trim(),
    order_date: document.getElementById('orderDate').textContent.trim(),
    items: items,
    notes: document.getElementById('notes').textContent.split(';').map(s => s.trim()).filter(Boolean),
  }};

  const blob = new Blob([JSON.stringify(result, null, 2)], {{ type: 'application/json' }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = {json.dumps(source_name)} + '.corrected.json';
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}});
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate a self-contained HTML review page for one extracted image.")
    parser.add_argument("name", help="Image stem, e.g. 'sample 5' (matches extracted/<name>.json)")
    parser.add_argument("--outdir", default="extracted", help="Directory holding <name>.json / <name>.stageA.json (default: extracted)")
    parser.add_argument("--images", default="Images", help="Directory holding the source image (default: Images)")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    order_form = json.loads((outdir / f"{args.name}.json").read_text(encoding="utf-8"))
    stage_a_path = outdir / f"{args.name}.stageA.json"
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8")) if stage_a_path.exists() else None
    brandlist_path = outdir / f"{args.name}.brandlist.json"
    brandlist = json.loads(brandlist_path.read_text(encoding="utf-8")) if brandlist_path.exists() else None
    recount_flags_path = outdir / f"{args.name}.recount_flags.json"
    recount_flags = json.loads(recount_flags_path.read_text(encoding="utf-8")) if recount_flags_path.exists() else None
    party_check_path = outdir / f"{args.name}.party_check.json"
    party_check = json.loads(party_check_path.read_text(encoding="utf-8")) if party_check_path.exists() else None

    headers = build_size_headers(order_form, stage_a)
    image_path = find_source_image(Path(args.images), args.name)
    if image_path is None:
        print(f"Warning: no source image found for '{args.name}' in {args.images}/")

    html_out = render_html(order_form, stage_a, headers, image_path, args.name, brandlist, recount_flags, party_check)
    out_path = outdir / f"{args.name}.review.html"
    out_path.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
