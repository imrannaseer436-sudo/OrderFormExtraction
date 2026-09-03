#!/usr/bin/env python3
"""
extract_claude.py -- Claude API extraction pipeline (v1, cost-optimized).

Deliberately NOT a port of extract_ollama.py's v3-v6 architecture. That
pipeline's multi-stage design (schema-constrained structure call -> CV-based
row cropping -> per-row OCR/VLM reads -> regex parsing) exists entirely to
work around a local 7B model's unreliable spatial grounding -- it hallucinated
header rows as data, drifted columns, and confused repeated item names (see
CLAUDE.md's "Known issues" / v3-v6 history). Claude's vision and instruction-
following don't have that failure profile, so none of that scaffolding is
needed here: this is ONE schema-constrained call per image, reading the whole
form (structure + all quantities) in one shot. Fewer calls is also directly
the cheapest architecture, which matters since this pipeline bills real money
per image instead of running on local hardware.

Cost design (see the "why" for each in comments below):
  1. Message Batches API for any folder of images (50% off every token) --
     this pipeline already has a human review step downstream, so nothing
     needs a synchronous response.
  2. Prompt caching on the system prompt (domain rules) -- identical on every
     call, so image #2 onward reads it at ~10% of first-call cost.
  3. Structured outputs (output_config.format) instead of a prefill/regex
     parsing stage -- guarantees valid JSON up front, no Stage C needed.
  4. Sonnet 5 as the default model (see DEFAULT_MODEL below) -- Opus 5 is
     available via --model for forms where Sonnet's accuracy isn't enough.
  5. Images are only ever downscaled, never upscaled -- sending more pixels
     than the model's high-res cap doesn't improve accuracy (Claude would
     downscale server-side and bill the same either way), so a client-side
     resize before upload just avoids wasted upload bytes; it is NOT the
     token-cost lever people often assume it is. See _prepare_image().
  6. Item-name/style suggestions and letter-size resolution (brandlist_match.py)
     against the product catalog run entirely locally -- a DB query plus
     string matching, no extra Claude calls -- rather than as a second API
     pass or a bigger, catalog-stuffed prompt.

Not yet done, deliberately -- would need real accuracy data first:
  - Lowering `effort` or disabling thinking. Sonnet 5 / Opus 5 both run
    adaptive thinking by default at effort="high", which costs output
    tokens but plausibly helps on exactly the failure mode this pipeline
    cares about (careful column-by-column reading of a dense handwritten
    grid). Tune `output_config` below once there's a real accuracy/cost
    comparison against a few sample forms -- don't guess at a lower
    setting without one.
  - Self-consistency voting (calling Stage-equivalent N times and flagging
    disagreement) if single-call accuracy turns out insufficient on the
    hardest forms. Multiplies cost by N, so only worth it if the schema-
    constrained single call demonstrably isn't enough.

Prerequisite:
    pip install -r requirements-claude.txt
    Add ANTHROPIC_API_KEY=sk-ant-... to .env (or export it) -- see .env,
    which already holds the DB credentials for this project; add the key
    as a new line rather than replacing anything in that file.

Usage:
    python3 extract_claude.py "Images/sample 5.jpeg"
    python3 extract_claude.py Images/ --outdir extracted_claude
    python3 extract_claude.py Images/ --model claude-opus-5
    python3 extract_claude.py Images/ --live   # skip batching, e.g. for a quick 2-image test

Output per image (written to --outdir, default extracted_claude/, kept
separate from extracted/ so this pipeline's output never overwrites the
Ollama pipeline's output for the same sample images):
    extracted_claude/<name>.json        -- final structured data (OrderForm shape)
    extracted_claude/<name>.raw.json    -- full raw model output before conversion (debugging)
    extracted_claude/<name>.stageA.json -- {seller_name, size_headers} only, written purely so
                                            generate_review.py (built for the Ollama pipeline)
                                            lays out columns in the form's real left-to-right
                                            order -- works unmodified against this pipeline's
                                            output too.
    extracted_claude/<name>.brandlist.json -- per-item catalog match/suggestion notes from
                                            brandlist_match.py (skipped if --no-brandlist-check,
                                            or if the DB wasn't reachable this run).
    extracted_claude/review.csv         -- flattened, one row per (item, size)

Every call (live or batch) also appends one row to a persistent usage log
(default ./usage_log.csv, override with --usage-log) -- a running record of
what's actually been spent, independent of --outdir, so cost across many
runs/sessions/models can be tallied and judged for plausibility rather than
trusted from a single run's console output.
"""

import argparse
import base64
import concurrent.futures
import csv
import io
import json
import os
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rapidfuzz import fuzz, utils

import brandlist_match
from schema import OrderForm, OrderItem

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}

# Intro pricing ($2/$10 per MTok, vs $5/$25 for Opus 5) through 2026-08-31,
# and strong out-of-the-box coding/agentic/vision quality -- the right
# default for a cost-sensitive batch pipeline. Override with --model
# claude-opus-5 for forms Sonnet 5 gets wrong.
DEFAULT_MODEL = "claude-sonnet-5"

# Claude Opus 5 / Sonnet 5's high-res vision cap. Images already at or under
# this are sent unmodified; nothing larger is ever needed, and Claude would
# downscale a larger image server-side anyway (billing the downscaled size),
# so resizing here saves upload bytes only, not API cost -- see module docstring.
MAX_LONG_EDGE = 2576

MAX_TOKENS = 32000  # confirmed necessary, not just a guess: 16000 truncated mid-JSON
                     # (stop_reason="max_tokens") on sample 3.jpeg -- a 20-item dense
                     # table under the newer, more detailed prompt (letter-sizes reported
                     # instead of dropped) generates more output than the old prompt did.
                     # Above the ~16k non-streaming safe threshold, so extract_one_live()
                     # uses client.messages.stream() + get_final_message() instead of a
                     # plain create() call -- see claude-api skill notes on why.

RECOUNT_MAX_TOKENS = 32000  # only used by the whole-table fallback call now (see
                            # _recount_quantities_live) -- confirmed necessary, not just a guess
                            # (2026-08-07): a 16000 budget truncated mid-response on sample 4's
                            # 12-row recount when all rows still went through one combined call
                            # (out=16000 exactly, stop_reason "max_tokens") -- thinking + one full
                            # RowQuantityReading per image adds up fast across a dozen images in
                            # one call. Streams via client.messages.stream(), same as the main
                            # call, since this is above the SDK's non-streaming safe threshold.

RECOUNT_ROW_MAX_TOKENS = 8000  # per-row call budget, used by the primary (per-row) recount path
                                # added 2026-08-08. Each call now only has to think about ONE row's
                                # column alignment, not synthesize N rows' worth of reasoning in a
                                # single continuous thought stream the way the old one-call-for-all-
                                # rows approach did (that approach was confirmed, via usage_log.csv,
                                # to be regularly hitting the 32000 cap on forms with ~12+ rows,
                                # which was both the dominant source of per-image latency -- pushing
                                # total time from ~2-2.5min to 8-10min -- and a source of silently
                                # unrecounted rows whenever the call ran out of budget partway
                                # through). Kept well under RECOUNT_MAX_TOKENS's 32000 on purpose;
                                # raise only with real evidence a row is still truncating (a
                                # truncated row degrades gracefully to "kept the original reading
                                # for this row", per _recount_one_row, so this is a real fallback,
                                # not a silent failure).

RECOUNT_MAX_WORKERS = 6  # bounds concurrent in-flight per-row recount calls per image -- enough to
                          # meaningfully parallelize a dozen-row form's recount pass (the dominant
                          # per-image latency cost) without an unbounded burst of simultaneous
                          # requests against the API.

MIN_ROW_HEIGHT_FRAC = 0.008  # a per-item row_top_frac/row_bottom_frac span narrower than this
                              # (roughly the height of one text line on a typical photo) is almost
                              # certainly a bad model estimate, not a real row -- see _validate_row_fracs.

# Row-depth drift-trend calibration (2026-08-08) -- see _fit_drift_trend's docstring for the full
# rationale (adapted from the Ollama pipeline's v6.1 self-calibration, CLAUDE.md). Same activation
# guard values v6.1 settled on after its first, too-strict attempt (spread 0.15) silently never
# activated on a real form -- reused directly here rather than re-discovering the same lesson.
MIN_CALIBRATION_ANCHORS = 3
MIN_CALIBRATION_Y_SPREAD = 0.08

BATCH_POLL_SECONDS = 20

# Current list pricing per million tokens (input, output), USD -- for the
# rough cost estimate printed after each call. Verify against
# platform.claude.com/pricing before relying on this for real accounting;
# it's a convenience readout, not a billing source of truth.
PRICING_PER_MTOK = {
    "claude-sonnet-5": (2.00, 10.00),  # intro pricing through 2026-08-31 ($3/$15 standard after)
    "claude-opus-5": (5.00, 25.00),
}
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute (default) ephemeral TTL

# Same safety net as schema.py's ItemStub -- ported rather than imported, since
# this file's LLM-facing schema (quantities as a list of pairs, not a dict --
# see ExtractedItem below) is intentionally different from schema.py's
# FormMeta/ItemStub, which was Stage-A-only. A model this capable should
# rarely need it, but it's a free correctness backstop, not a network call.
KNOWN_STYLE_CODES = {"IE", "OE", "RN", "RNS"}
_TRAILING_CODE_RE = re.compile(r"^(?P<name>.+?)\s+(?P<code>[A-Z]{2,4})$")
_DATE_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}")

# A real size header on these forms is always a plain number ("45", "105")
# -- never a running-total column. Confirmed real, independent of
# schema.py's own FormMeta.size_headers validator (2026-09-01): a live
# call here returned `size_headers` with the form's own trailing "Total
# Dozen" column included as a 14th entry ("Total"), which then corrupted
# _hybrid_ocr_quantities' column-spacing math downstream the same way it
# did for the other pipeline -- confirmed by inspecting the raw stageA
# output directly, not assumed. Unlike schema.py's version, this doesn't
# need a letter-size allowance: a letter-coded size here is reported
# per-cell in QuantityPair.size (see the LETTER SIZES prompt rule), never
# as an entry in the shared size_headers list itself.
_NUMERIC_HEADER_RE = re.compile(r"^\d{1,3}$")


SYSTEM_PROMPT_TEMPLATE = """You are transcribing a garment order form photo into structured data. These \
forms mix handwriting, printed text, and typed text, with a dense size/quantity table. Reproduce \
exactly what is written on the page; never guess, infer a pattern, or extrapolate a sequence for \
a value you cannot read confidently -- leave it out instead.

FORM LAYOUT -- IDENTIFY THIS FIRST, IT CHANGES HOW SEVERAL RULES BELOW APPLY: this business receives \
order forms in more than one physical layout. Check which one this specific page actually is before \
reading further -- several rules below are written separately for each, clearly labeled; apply only \
the ones for the layout actually in front of you, and never mix the two:
  - GRID LAYOUT: one printed table with a single row of size headers printed once across the top, \
and every item as its own row below it, all sharing that same header row. Most forms use this layout.
  - FREE-FORM LAYOUT: no single shared table -- each item has its own size numbers handwritten \
locally, directly above or beside that item's own quantities, and a different item elsewhere on the \
same page can use a completely different set of sizes. Common on handwritten notebook-style pages.

SELLER vs BUYER: report seller_name and party_name exactly as printed/handwritten -- do not try to \
judge which one is the "real" buyer for Essa's own records; that is resolved separately, downstream \
of this extraction, against Essa's own buyer database. seller_name is the printed letterhead/logo at \
the top of the page. party_name is the name handwritten in the "Party Name" box further down the \
page (e.g. "Murugan Tex"). Most forms are Essa Garments' OWN pre-printed order pad -- on those, \
seller_name will read some form of "ESSA"/"ESSA GARMENTS", and party_name is already the correct \
buyer; never output "Essa"/"Essa Garments" (or any variant of it) as party_name even if it appears \
elsewhere on the page (stamp, signature area, etc.) -- it is never the buyer. Some forms instead use \
a DIFFERENT business's own pre-printed order pad (their own company name/logo printed at the top, \
not Essa's) -- seller_name will NOT be Essa in that case. Still report both fields exactly as \
printed/handwritten either way; do not guess, override, or swap one field's value for the other.

ITEM NAME vs STYLE CODE: "Particulars" (item name) and "Style" are separate columns. If \
Particulars says "Trend Trunk" and Style says "IE", then item="Trend Trunk" and type="IE" -- \
never "Trend Trunk IE" as the item name. Known Style-column values from this business's product \
catalog include: {style_codes}. Recognize one of these wherever it actually appears on the row -- \
including if handwriting drifted into the Particulars column, or into a size column -- and treat \
it as the style/type, not as part of the item name and not as a quantity. This list may not be \
exhaustive; a short (2-5 letter) all-caps token that isn't a plausible item-name word is likely a \
style code even if it's not on the list.

DITTO MARKS (—"—, -"-, or similar) apply only within the column they are written in -- \
"same as the row above, in THIS column," never "same as the row above in every column":
  - Ditto in Particulars: write out the FULL inherited item name from the row above, with this \
row's own word appended if there is one (e.g. row above "Fairlady Print", this row '—"— \
Plain' -> item="Fairlady Plain"). Never write just the modifier word alone.
  - Ditto in Particulars while Style has an ACTUAL new value on the same row: the item name still \
comes from the ditto inheritance above; Style's new value is read independently as this row's \
type. E.g. row above item="MYNA" type="IE"; next row Particulars='—"—' Style="OE" -> \
item="MYNA", type="OE". Never mistake a Style column value for the item name.

SIZE HEADERS: GRID LAYOUT -- report every size column header across the top of the shared table, \
left to right, exactly as printed, in size_headers. Many grid forms print a SECOND row of numbers \
just beneath the real headers (an age/chest-size equivalent) -- that second row is a column label, \
never put it in size_headers, and never read quantities from it for any item. Do NOT include a \
trailing running-total column (e.g. "Total", "Total Dozen", "Grand Total") in size_headers either -- \
that is a summary column, not a size, even though it sits at the end of the same header row. \
FREE-FORM LAYOUT -- there is no single shared header row to report; leave size_headers as an empty \
list rather than forcing one item's local sizes into a page-wide list that doesn't really exist on \
this page.

QUANTITIES: report every size/quantity pair exactly as written. GRID LAYOUT -- go through the size \
columns left to right under the shared header row for each item's row. FREE-FORM LAYOUT -- read the \
size numbers written locally for this item and the quantities written below/beside them. Either way, \
watch for two ways a pair can deviate from the simple case, both real and both common on these forms:
  - SIZE LABEL OVERRIDE (GRID LAYOUT): sometimes a cell doesn't hold a plain quantity under its \
printed header at all -- the writer has instead written an actual SIZE label inside or near that \
cell (overriding the printed header, because it doesn't apply to this particular row/product) with \
the QUANTITY written below or beside it, essentially as its own small size:quantity pair squeezed \
into the row. When you see this, report the size actually written there -- not the printed header \
above it -- paired with the quantity associated with it.
  - LETTER SIZES: a size is sometimes written as a standard clothing letter size (0, XS, S, M, L, \
XL, XXL, 2XL, 3XL, 4XL) instead of a plain number, when the row's real product doesn't use this \
form's numeric sizing at all. When this happens, report the "size" field as the letter exactly as \
written (e.g. "M") -- do NOT try to convert it to a number yourself, and do NOT skip it just because \
it isn't numeric. A separate step with the full product catalog resolves which number that letter \
means for this specific product.
  Otherwise: omit blank cells and cells marked "x" or "-" entirely -- do not report them as 0, and \
do not invent a value for a cell you can't read confidently. A fraction like "60/10" is not a \
whole number -- omit it rather than rounding or guessing. The default unit on these forms is \
dozens; report the number exactly as written, do not convert units.

MULTI-LINE QUANTITY BLOCKS (FREE-FORM LAYOUT, but check for this on any layout): an item's \
size/quantity pairs sometimes continue onto a SECOND line directly below the first, still part of \
the SAME item -- e.g. one line reads "50 60 65 70" over "10 15 10 5", and the line right below it \
reads "80 85 90" over "15 5 5" for that same item, before any new item name/heading appears. Always \
check for a second line of numbers directly under the first before moving to the next item -- if \
there is one and no new item name separates it from the line above, it belongs to the current item; \
include ALL of its pairs too, not just the first line's.

SELF-CHECK (any layout): some forms print or circle a genuine per-row TOTAL near each row -- e.g. a \
"Total Dozen" column -- meant to equal the sum of that row's own quantities. Only report a number \
like this as row_total when you're confident that's what it means; if your first reading of the \
row's quantities doesn't sum to it, recount before answering (you likely misread a digit, shifted a \
column, or missed a mark obscured by other ink), and if it still doesn't match, report your best \
reading anyway rather than forcing an artificial match. But NOT every circled/printed number near a \
row is this kind of total -- some forms instead print a NUMBER OF BUNDLES/PACKS per row, a \
shipping/packing count that has no reason to equal the row's quantity sum at all. If you can't tell \
which one a given number is, leave row_total empty rather than reporting a number that was never \
meant to match -- and never adjust a quantity reading just to force a match against a number you're \
not sure is even a total.

ROWS OUTSIDE THE MAIN RULED GRID (GRID LAYOUT): sometimes the last item is squeezed below the \
table's ruled lines, sharing space with printed footer text ("Stock Entry By", "Checked By", \
"Signature") -- its name may be written in the margin and its quantity marks may overlap or sit \
right next to that footer text. Read it as carefully as any other row; do not drop marks just \
because they are near printed text, and do not skip the row entirely.

order_date: normalize to DD/MM/YYYY, assuming 20xx for 2-digit years. This field must contain \
ONLY the final normalized date -- never an explanation or the original as-written text alongside it.

notes: any page-level handwritten notes (e.g. "Old Rate Supply Only"), and anything illegible or \
unusual not tied to a specific row's quantities. Empty list if there are none.

table_top_frac / table_bottom_frac: the fraction (0.0 at the very top of the image, 1.0 at the very \
bottom) of the image's height where the item data starts and ends -- never the letterhead/party-info \
box above it. GRID LAYOUT -- table_top_frac is the top edge of the HEADER row (the size numbers). \
FREE-FORM LAYOUT -- there is no header row, so table_top_frac is the top edge of the FIRST item's \
own data (its name or its own local size numbers, whichever comes first). Either way, \
table_bottom_frac is the bottom edge of the LAST item's data -- include it even if it is squeezed \
into an irregular space near the bottom (e.g. overlapping "Stock Entry By" / "Checked By" / \
"Signature" footer text in GRID LAYOUT, or written in a page margin in FREE-FORM LAYOUT), since that \
is still real order data.

row_top_frac / row_bottom_frac (per item): the same 0.0-1.0 image-height fraction convention, but for \
THIS ONE item's own row -- from the top of its data (where its Particulars text/quantities begin) to \
the bottom (just before the next item's data begins, or the bottom of the item data for the last \
item). If an item's name wraps onto two lines (common on GRID LAYOUT forms, e.g. "1. B.3825" on one \
line and "SHORT SET RNBS" on the next), both lines and this row's quantities are still ONE row -- \
row_top_frac starts at the first line, not the second. The same applies if the item's QUANTITIES \
wrap onto a second line (see MULTI-LINE QUANTITY BLOCKS above) -- row_bottom_frac must extend past \
BOTH lines of numbers, not just the first, or the crop built from these fractions will cut the \
second line off entirely. Be as precise as you can; these fractions are used to crop each row out \
individually, so a boundary that bleeds into a neighboring row will make that crop less useful, not more."""


def build_system_prompt(style_codes: list[str]) -> str:
    codes = style_codes or sorted(KNOWN_STYLE_CODES)
    return SYSTEM_PROMPT_TEMPLATE.format(style_codes=", ".join(codes))

USER_PROMPT = "Extract this garment order form photo per the schema and rules in the system prompt."


class QuantityPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    size: str = Field(description="Size column header exactly as printed on the form (e.g. '45', '90').")
    quantity: int = Field(description="The quantity written for this size. Only include cells with an actual legible number -- see the QUANTITIES rule.")


class ExtractedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item: str = Field(description="Product/article name exactly as written (Particulars column), with ditto marks expanded per the DITTO MARKS rule.")
    type: str = Field(description="Style/variant code from the Style column (e.g. IE, OE, RN, RNS). Empty string if there is no separate Style column or it's blank for this row.")
    quantities: List[QuantityPair] = Field(description="One entry per size column with a legible quantity for this row. Empty list if the row has no readable quantities.")
    row_total: str = Field(description="This row's own printed or circled running total (e.g. a 'Total Dozen' column, or a circled number in the margin), exactly as written, regardless of what it's labeled or where on the row it appears. Empty string if this form has no such total, or it's blank for this row.")
    row_top_frac: float = Field(description="0.0-1.0 fraction of image height where THIS item's own row starts. See row_top_frac/row_bottom_frac rule.")
    row_bottom_frac: float = Field(description="0.0-1.0 fraction of image height where THIS item's own row ends.")

    @model_validator(mode="after")
    def _split_trailing_style_code(self) -> "ExtractedItem":
        if not self.type and self.item in KNOWN_STYLE_CODES:
            self.type = self.item
            self.item = ""
            return self
        m = _TRAILING_CODE_RE.match(self.item)
        if not m:
            return self
        code = m.group("code")
        if not self.type and code in KNOWN_STYLE_CODES:
            self.item = m.group("name")
            self.type = code
        elif self.type and code == self.type:
            self.item = m.group("name")
        return self


class ExtractedForm(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seller_name: str = Field(description="Letterhead/seller company name printed at the top of the form -- NOT the buyer.")
    party_name: str = Field(description="Buyer/customer name, handwritten in the Party Name box. Empty string if not present.")
    order_no: str = Field(description="Order form number if present, else empty string.")
    order_date: str = Field(description="Date normalized to DD/MM/YYYY, assuming 20xx for 2-digit years. ONLY the final value.")
    size_headers: List[str] = Field(description="GRID LAYOUT: every size column header across the top of the shared table, left to right, exactly as printed. FREE-FORM LAYOUT (no single shared table): empty list.")
    items: List[ExtractedItem] = Field(description="Every product/article row, top to bottom, in order.")
    notes: List[str] = Field(description="Page-level handwritten notes and anything illegible/unusual not tied to one row's quantities.")
    table_top_frac: float = Field(description="0.0-1.0 fraction of image height where the item data starts -- GRID LAYOUT: the header row; FREE-FORM LAYOUT: the first item's own data. See table_top_frac/table_bottom_frac rule.")
    table_bottom_frac: float = Field(description="0.0-1.0 fraction of image height where the last item's data ends, including any row squeezed into an irregular space (e.g. outside the main ruled grid in GRID LAYOUT, or a page margin in FREE-FORM LAYOUT).")

    @field_validator("order_date")
    @classmethod
    def _clean_order_date(cls, v: str) -> str:
        if not v:
            return v
        m = _DATE_RE.search(v)
        return m.group(0) if m else v

    @field_validator("size_headers")
    @classmethod
    def _drop_non_numeric_headers(cls, v: List[str]) -> List[str]:
        return [h for h in v if _NUMERIC_HEADER_RE.match(h)]

    @model_validator(mode="after")
    def _forward_fill_ditto_item_names(self) -> "ExtractedForm":
        last_item = ""
        for stub in self.items:
            if stub.item:
                last_item = stub.item
            elif last_item:
                stub.item = last_item
        return self


# --- Row-quantity recount pass -----------------------------------------
#
# Confirmed real bug (2026-08-07, sample 4.jpeg, cross-checked against the
# printed "Total Dozen" per-row totals and the business's own catalog via
# brandlist_match.py): the main call above gets the COUNT of filled cells
# in a row right, but on dense tally-mark rows (many narrow columns, most
# filled with repeated "1" strokes rather than distinct numbers) it can
# attribute the whole block to the wrong columns -- a column shift. Because
# the shift doesn't change the row's sum, the existing Total-Dozen
# checksum can't catch it, and brandlist_match's catalog cross-check only
# catches it when the shifted sizes happen to fall outside that specific
# product's known catalog sizes -- confirmed it missed most of the shifted
# rows on sample 4, since many of this form's rows share similar,
# overlapping catalog size ranges.
#
# Fix: a second, focused pass per image, sent ONLY a tight crop of the
# table region (header through last row, enlarged) -- removing the
# letterhead/party-info clutter the first call also had to look at, and
# forcing the model to explicitly commit to first_size/last_size anchors
# and a Total-Dozen self-check per row, instead of freely emitting
# size:quantity pairs. Only merged back into the item when the row's own
# item_seen text plausibly matches what the main call already extracted
# for that position -- if the crop's row boundaries drifted (e.g. grid
# detection isn't used here; table_top_frac/table_bottom_frac come from
# the main call's own estimate, which is coarse), the mismatch is
# detectable and the original full-page reading is kept instead of
# silently trusting a misaligned crop. Live-mode only so far (see
# extract_one_live) -- not yet wired into extract_batch(), which would
# need a second batch round-trip; noted as a follow-up in CLAUDE.md rather
# than built partially/untested here.
#
# 2026-08-08: this pass is now N SEPARATE, PARALLEL calls (one per row),
# not one call bundling every row's crop together -- see
# _recount_quantities_live / _recount_one_row / RECOUNT_ROW_MAX_TOKENS
# below for why (the old bundled-call design was confirmed, via
# usage_log.csv, to be the dominant source of per-image latency on dense
# forms, and to silently lose recount coverage for every row after a
# mid-call truncation).

RECOUNT_SYSTEM_PROMPT = """You are re-reading ONLY the size/quantity grid of a garment order form. You have been \
given one cropped, enlarged image showing this form's size/quantity grid -- isolated from the letterhead/party-info \
clutter the rest of the page has -- so you can focus purely on precise column alignment without distraction. Usually \
this is a single item row with the table header stitched directly above it, isolated from every other row, so \
nothing else competes for your attention; occasionally (labeled accordingly) it is the whole table at once instead. \
The accompanying label tells you which row(s) the image covers -- read it before the image.

WHY THIS SECOND PASS EXISTS: these tables often have 10+ narrow columns, and rows are frequently filled with \
repetitive tally-mark strokes (e.g. many handwritten "1"s in a row) rather than distinct numbers -- it is easy to \
correctly count HOW MANY cells are filled but misjudge WHICH column they start under, shifting every value in the \
row one column left or right of where it truly belongs. This pass exists specifically to get that alignment right.

SIZE HEADERS -- READ THE TOP LINE ONLY: many GRID-LAYOUT forms (one shared printed table) print TWO stacked \
numbers in each header cell, e.g. "45" directly above a smaller "18" -- the top line is the real size header \
(matches the size_headers this form uses everywhere else); the smaller number below it is an unrelated \
age/chest-size equivalent, printed as a label, never as data. Use ONLY the top-line number for first_size/ \
last_size and every size in quantities. If the numbers you are about to report look like a different, \
smaller/denser sequence than the item list's own known sizes (e.g. reporting "18, 20, 22..." instead of \
"45, 50, 55..."), you have read the wrong line -- go back to the top line. This doesn't apply to a crop from a \
FREE-FORM LAYOUT page (no single shared table) -- there each item has only its own local size numbers directly \
above its own quantities, with no second decoy line underneath.

MULTI-LINE QUANTITY BLOCKS: an item's size/quantity pairs sometimes continue onto a SECOND line directly below \
the first, still part of the SAME item -- e.g. one line reads "50 60 65 70" over "10 15 10 5", and the line \
right below it reads "80 85 90" over "15 5 5" for that same item. This is common on FREE-FORM LAYOUT crops, but \
check for it regardless of layout. If the crop you were given includes a second line of numbers directly under \
the first, with no new item name/heading separating it, include ALL of its pairs too -- not just the first \
line's -- even if the main extraction's item list only led you to expect one line here.

COLUMN ALIGNMENT: for each row, count grid columns from that row's OWN left edge (starting at the first printed \
size header), not from memory of a previous row or from where the row above's marks fell. Report first_size and \
last_size explicitly -- the size headers of the leftmost and rightmost marked columns in this row -- then list \
every marked cell as size:quantity pairs. On these forms, marked cells are virtually always ONE CONTIGUOUS block \
with no unmarked gaps in the middle -- if your reading has an isolated blank cell surrounded by marked cells, \
recheck it before finalizing; you likely misread a faint mark as blank.

SELF-CHECK: if this row has a printed/circled running total (e.g. a "Total Dozen" column) that you're confident \
is meant as a sum of this row's own quantities, read it and report it as total_dozen -- the sum of this row's \
quantities should equal it. If your first reading doesn't match, recount before answering: either you shifted a \
column, or missed a mark obscured by other ink (a row squeezed near footer/signature text at the bottom of the \
table is especially prone to this -- read it extra carefully rather than dropping unclear marks). If it still \
doesn't match after recounting, report your best reading anyway rather than forcing an artificial match. Some \
forms instead print a NUMBER OF BUNDLES/PACKS per row (a shipping/packing count, not a quantity total) -- if \
you can't tell which kind of number this is, leave total_dozen empty rather than reporting one that was never \
meant to match.

Also report item_seen -- the item/particulars name written on this specific row, exactly as you see it in the \
crop -- purely so a mismatch against the expected row can be caught; it is never used as the final item name.

Otherwise, the same rules as before apply: omit blank cells and cells marked "x"/"-" entirely (never as 0); a \
size written as a standard clothing letter (S, M, L, XL, ...) instead of a plain number should be reported as \
that letter in the size field, exempt from the contiguous-block expectation above; a cell where the writer wrote \
an actual size label overriding the printed header should use that written size, not the printed header."""


class RowQuantityReading(BaseModel):
    model_config = ConfigDict(extra="forbid")
    item_seen: str = Field(description="The item/particulars name written on this row, exactly as seen in the crop -- for alignment verification only, never used as the final item name.")
    first_size: str = Field(description="Size header of the leftmost marked column in this row. Empty string if the row has no quantities.")
    last_size: str = Field(description="Size header of the rightmost marked column in this row. Empty string if none.")
    quantities: List[QuantityPair] = Field(description="size:quantity for every marked cell in this row, left to right. Every size should fall between first_size and last_size in printed header order, unless it's a SIZE LABEL OVERRIDE or LETTER SIZES cell (exempt from that constraint).")
    total_dozen: str = Field(description="This row's printed/circled running total, exactly as written. Empty string if this form has no such column or it's blank here.")


class QuantityRecount(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: List[RowQuantityReading] = Field(description="One entry per item row, in the same top-to-bottom order as the item list given to you.")


def _prepare_image(image_path: Path) -> tuple[bytes, str]:
    """Load image bytes, resizing only if it exceeds the model's high-res cap
    -- see MAX_LONG_EDGE comment for why this doesn't change billed tokens."""
    original = image_path.read_bytes()
    with Image.open(io.BytesIO(original)) as img:
        long_edge = max(img.size)
        if long_edge <= MAX_LONG_EDGE:
            return original, MEDIA_TYPES.get(image_path.suffix.lower(), "image/jpeg")
        scale = MAX_LONG_EDGE / long_edge
        new_size = (round(img.width * scale), round(img.height * scale))
        resized = img.convert("RGB").resize(new_size, Image.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), "image/jpeg"


def _upscale_toward_cap(crop: Image.Image, max_scale: float = 3.0) -> Image.Image:
    """Upscale (never downscale beyond MAX_LONG_EDGE) toward the model's
    resolution cap -- deliberate, unlike _prepare_image's downscale-only
    policy: the whole point of a recount crop is more pixels per column
    than the full-page image gave the main call, not just less clutter."""
    long_edge = max(crop.size)
    scale = min(max_scale, MAX_LONG_EDGE / long_edge) if long_edge > 0 else 1.0
    if abs(scale - 1.0) <= 0.05:
        return crop
    return crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS)


def _image_to_jpeg_bytes(img: Image.Image) -> tuple[bytes, str]:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"


def _prepare_table_crop(original_bytes: bytes, top_frac: float, bottom_frac: float) -> tuple[bytes, str]:
    """Fallback crop: the WHOLE table (every row together), used only when
    per-item row_top_frac/row_bottom_frac don't validate (see
    _validate_row_fracs) -- still strictly better than the full page (no
    letterhead/party-box clutter, upscaled), but doesn't isolate individual
    rows from each other the way _build_row_crops does."""
    margin = (bottom_frac - top_frac) * 0.03
    top_frac = max(0.0, top_frac - margin)
    bottom_frac = min(1.0, bottom_frac + margin)
    with Image.open(io.BytesIO(original_bytes)) as img:
        w, h = img.size
        crop = img.crop((0, int(top_frac * h), w, int(bottom_frac * h)))
        return _image_to_jpeg_bytes(_upscale_toward_cap(crop))


def _validate_row_fracs(items: List[ExtractedItem], table_top: float, table_bottom: float) -> bool:
    """True if every item's row_top_frac/row_bottom_frac forms a plausible,
    non-degenerate, top-to-bottom-ordered partition of the table -- guards
    against building N crops from a bad model estimate (the OLD Ollama
    pipeline's v4b attempt at model-reported per-row bounding boxes failed
    exactly this way: "boxes came back a few percent of image height tall
    (far too thin) and shifted" -- see CLAUDE.md). A small tolerance on
    ordering absorbs minor estimation noise between adjacent rows without
    accepting genuinely nonsensical output."""
    if not items:
        return False
    prev_bottom = table_top
    for it in items:
        top, bottom = it.row_top_frac, it.row_bottom_frac
        if not (0.0 <= top < bottom <= 1.0):
            return False
        if bottom - top < MIN_ROW_HEIGHT_FRAC:
            return False
        if top < prev_bottom - 0.01:
            return False
        prev_bottom = bottom
    return prev_bottom <= table_bottom + 0.02


def _build_row_crops(original_bytes: bytes, table_top_frac: float, items: List[ExtractedItem]) -> list[tuple[bytes, str]]:
    """One (image_bytes, media_type) crop per item: the table header band
    (table_top_frac through the first item's own row_top_frac) stitched
    above that item's own row band, enlarged -- same header-plus-row idea
    as the Ollama pipeline's build_header_crop/build_row_crop (grid.py),
    but driven by Claude's own row_top_frac/row_bottom_frac instead of
    CV-detected boundaries, since grid.py's automatic row detector failed
    to find sample 4's real rows (item 12's irregular footer-overlap row
    breaks its uniform-row-height assumption).

    GRID LAYOUT (shared header row) is the case this was built for -- there,
    table_top_frac (top of the header row) and items[0].row_top_frac (where
    the first item's own data starts) are genuinely different points, with
    real header content between them. On a FREE-FORM LAYOUT page there is no
    shared header row -- per the prompt, table_top_frac there IS the first
    item's own data start, so the two fracs collapse to the same point and
    there'd be nothing real to stitch. Detected here (header band shorter
    than MIN_ROW_HEIGHT_FRAC) rather than stitching a near-empty sliver:
    each item's own row crop already contains that item's own local header
    numbers (they're written directly above its own quantities, within its
    own row_top_frac/row_bottom_frac span), so skipping the stitch is the
    semantically correct behavior for that layout, not just a defensive
    fallback."""
    with Image.open(io.BytesIO(original_bytes)) as img:
        img = img.convert("RGB")
        w, h = img.size
        header_bottom_frac = items[0].row_top_frac
        has_real_header_band = (header_bottom_frac - table_top_frac) >= MIN_ROW_HEIGHT_FRAC
        header_crop = img.crop((0, int(table_top_frac * h), w, int(header_bottom_frac * h))) if has_real_header_band else None

        crops = []
        for it in items:
            margin = (it.row_bottom_frac - it.row_top_frac) * 0.08
            row_top = max(header_bottom_frac if has_real_header_band else table_top_frac, it.row_top_frac - margin)
            row_bottom = min(1.0, it.row_bottom_frac + margin)
            row_crop = img.crop((0, int(row_top * h), w, int(row_bottom * h)))

            if header_crop is not None:
                stitched = Image.new("RGB", (w, header_crop.height + row_crop.height), "white")
                stitched.paste(header_crop, (0, 0))
                stitched.paste(row_crop, (0, header_crop.height))
                crops.append(_image_to_jpeg_bytes(_upscale_toward_cap(stitched)))
            else:
                crops.append(_image_to_jpeg_bytes(_upscale_toward_cap(row_crop)))
        return crops


def _build_messages(image_bytes: bytes, media_type: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(image_bytes).decode("ascii")}},
            {"type": "text", "text": USER_PROMPT},
        ],
    }]


def _build_recount_messages(crops: list[tuple[bytes, str, str]]) -> list[dict]:
    """crops: one (image_bytes, media_type, label) per image to send --
    either N per-row crops (label = "Row i: <item> (Style: <type>)") or a
    single whole-table fallback crop (label = the full item list as text)."""
    intro = (
        f"Here {'is' if len(crops) == 1 else 'are'} {len(crops)} cropped image(s), each labeled. "
        "Read the quantities per the rules in the system prompt, in the order given."
    )
    content: list[dict] = [{"type": "text", "text": intro}]
    for crop_bytes, media_type, label in crops:
        content.append({"type": "text", "text": label})
        content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": base64.standard_b64encode(crop_bytes).decode("ascii")}})
    return [{"role": "user", "content": content}]


def _catalog_match_for(item_name: str, brandlist_available: bool) -> "brandlist_match.MatchCandidate | None":
    """The same 'good enough to discriminate between two already-extracted
    candidate readings' bar used throughout _apply_recount, factored out
    since 2026-08-08's changes now call it from more than one place.
    Intentionally lighter than brandlist_match's own auto-apply gate (see
    the module note there) -- nothing here identifies which PRODUCT a row
    is, only whether a small set of already-plausible readings survive a
    catalog-size check."""
    if not brandlist_available:
        return None
    try:
        match = brandlist_match.find_best_match(item_name)
    except Exception:
        return None
    if not match or not match.sizes:
        return None
    unique_code_match = match.code_narrowed and match.narrowed_pool_size == 1
    if match.score >= brandlist_match.SUGGEST_SCORE_THRESHOLD or unique_code_match:
        return match
    return None


def _parse_int_or_none(s: str) -> int | None:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _apply_offset(quantities: dict[str, int], offset: int, size_headers: list[str]) -> dict[str, int]:
    """Relabels every numeric-size key in `quantities` by `offset` header
    positions (per size_headers' printed left-to-right order) -- the actual
    correction once brandlist_match.resolve_shift_offset has confirmed which
    offset is right. Non-numeric sizes (letter sizes, SIZE LABEL OVERRIDE
    overrides) pass through unchanged, since a header-column offset doesn't
    apply to them."""
    header_order = [h for h in size_headers if h.isdigit()]
    index_of = {h: i for i, h in enumerate(header_order)}
    shifted: dict[str, int] = {}
    for s, q in quantities.items():
        if s in index_of:
            new_index = index_of[s] + offset
            if 0 <= new_index < len(header_order):
                shifted[header_order[new_index]] = q
        else:
            shifted[s] = q
    return shifted


def _fit_drift_trend(anchors: list[tuple[float, int]]) -> "tuple[float, float] | None":
    """Least-squares line (slope, intercept) through (row_top_frac,
    catalog-confirmed header-column offset) points -- the same idea as the
    Ollama pipeline's v6.1 drift self-calibration (CLAUDE.md), adapted to
    this pipeline's units: that one regresses PIXEL drift against row depth
    from OCR-measured digit positions; this one regresses HEADER-COLUMN
    offset against row depth from brandlist_match.resolve_shift_offset's
    catalog-confirmed readings, since Claude never reports raw pixel
    coordinates -- there's nothing else to regress against here. No numpy
    dependency (not declared in requirements-claude.txt) -- this is a
    two-line closed-form fit, not worth adding one for.

    Mirrors v6.1's activation guard too, reusing the exact same lesson: its
    first attempt used a too-strict spread threshold (0.15) and silently
    never activated on a real form, confirmed only because the output came
    back byte-identical to the uncorrected version; fixed by lowering it to
    0.08 and confirming activation on a real run. MIN_CALIBRATION_ANCHORS/
    MIN_CALIBRATION_Y_SPREAD reuse that same 0.08 value rather than
    re-discovering the lesson. Returns None (don't trust a trend) when the
    guard isn't met."""
    if len(anchors) < MIN_CALIBRATION_ANCHORS:
        return None
    xs = [x for x, _ in anchors]
    if max(xs) - min(xs) < MIN_CALIBRATION_Y_SPREAD:
        return None
    n = len(anchors)
    mean_x = sum(xs) / n
    mean_y = sum(y for _, y in anchors) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return None
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in anchors)
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _apply_recount(extracted: "ExtractedForm", recount: "QuantityRecount", brandlist_available: bool = False) -> tuple[list[str], list[dict]]:
    """Merges recount.rows back into extracted.items IN PLACE, only where the
    row's own item_seen text plausibly matches the item already at that
    position -- protects against a misaligned crop (e.g. table_top_frac was
    off) silently overwriting a correct original reading with a wrong one --
    AND only where every reported digit size is one of this form's actual
    size_headers.

    That second check exists because of a confirmed real failure (2026-08-07,
    sample 4.jpeg): the recount call read the form's SECONDARY header line
    (a smaller age/chest-size-equivalent row printed just below the real
    size headers -- the main call's own prompt already warns about this,
    but the recount call's separate system prompt hadn't, since a second
    API call doesn't inherit the first call's system prompt) for nearly
    every row, producing a plausible-looking but entirely wrong sequence
    (e.g. "18, 20, 22, ..." instead of "45, 50, 55, ..."). The item_seen
    alignment check alone did NOT catch this -- the item name was read
    correctly, only the size axis was wrong -- so this is a second,
    independent guard, not a replacement for the first.

    2026-08-08: previously, when the main call and the recount call agreed
    exactly, that agreement alone was trusted with NO further check -- but
    both calls read the same photo with the same underlying perspective, so
    a shared column-drift bias (confirmed real on sample 7's rows 6-7, see
    CLAUDE.md) can make them agree on the SAME wrong answer. Every row with
    a trustworthy catalog match (via _catalog_match_for) is now checked with
    brandlist_match.resolve_shift_offset even when main and recount agree --
    grounded in that row's own catalog data, not a guess -- and corrected in
    place when the offset is unambiguous. Rows with no catalog match of
    their own borrow a row-depth drift TREND fitted from the rows that do
    (_fit_drift_trend) -- but only ever as an explicit flag on an already-
    unresolved disagreement, never a silent auto-correction, since that
    signal is indirect (borrowed from other rows, not this one).

    Returns (notes, flags): notes are human-readable, for extracted.notes /
    the review page's top-level notes field. flags is a list aligned 1:1
    with extracted.items -- one dict per row, always present (default
    {"status": "no_recount"} for a row the recount never actually reached,
    e.g. a row-count mismatch against recount.rows) -- added 2026-08-08 so
    generate_review.py can highlight the SPECIFIC quantity cells worth a
    second look, not just bury the signal in a paragraph of free text at
    the top of the page. See _write_debug_artifacts /
    <name>.recount_flags.json for where this gets persisted."""
    notes: list[str] = []
    flags: list[dict] = [{"status": "no_recount"} for _ in extracted.items]
    if len(recount.rows) != len(extracted.items):
        notes.append(
            f"Row-crop quantity recount returned {len(recount.rows)} rows but the form has "
            f"{len(extracted.items)} items -- row counts didn't line up, unmatched rows kept their "
            f"original (unverified) quantities."
        )
    known_headers = set(extracted.size_headers)
    size_headers = extracted.size_headers

    calibration_anchors: list[tuple[float, int]] = []
    pending: list[dict] = []

    for idx, (item, row) in enumerate(zip(extracted.items, recount.rows)):
        expected, seen = item.item.strip(), row.item_seen.strip()
        aligned = not expected or not seen or fuzz.WRatio(expected, seen, processor=utils.default_process) >= 55
        if not aligned:
            note = f"Row-crop recount for '{expected}' saw '{seen}' on that row instead -- alignment looked unreliable, kept the original full-page quantities for this row."
            notes.append(note)
            flags[idx] = {"status": "unverified", "sizes": [qp.size for qp in item.quantities], "note": note}
            continue
        if not row.quantities and item.quantities:
            note = f"Row-crop recount found no quantities for '{expected}' -- kept the original full-page reading."
            notes.append(note)
            flags[idx] = {"status": "unverified", "sizes": [qp.size for qp in item.quantities], "note": note}
            continue

        digit_sizes = [qp.size for qp in row.quantities if qp.size.isdigit()]
        unknown_sizes = known_headers and [s for s in digit_sizes if s not in known_headers]
        if unknown_sizes:
            note = (
                f"Row-crop recount for '{expected}' reported sizes {sorted(set(unknown_sizes))} that aren't "
                f"among this form's real size headers {sorted(known_headers, key=lambda x: int(x) if x.isdigit() else 0)} -- "
                f"likely read the wrong header line, kept the original full-page quantities for this row."
            )
            notes.append(note)
            flags[idx] = {"status": "unverified", "sizes": [qp.size for qp in item.quantities], "note": note}
            continue

        # Agree-or-flag: the main call and this recount are two genuinely
        # independent reads of the same row (different crop, different
        # prompt). If they agree exactly, that agreement itself is real
        # evidence of correctness -- UNLESS a trustworthy catalog match for
        # this specific row says otherwise (see module note above on the
        # shared-bias gap). If they DISAGREE -- whether a column-shift
        # (different keys) or a value misread (same keys, different
        # quantity) -- don't silently prefer one; that's exactly how a wrong
        # number ships unnoticed. Total Dozen (if printed), then a catalog
        # check, then (as a flag only) a row-depth drift trend can break the
        # tie; otherwise both readings go into the note so a human can
        # compare them against the photo directly.
        original_map = {qp.size: qp.quantity for qp in item.quantities}
        recount_map = {qp.size: qp.quantity for qp in row.quantities}
        match = _catalog_match_for(item.item, brandlist_available)

        orig_offset = brandlist_match.resolve_shift_offset(original_map, match.sizes, size_headers) if match else None
        if orig_offset is not None:
            calibration_anchors.append((item.row_top_frac, orig_offset))

        if original_map == recount_map:
            if match and orig_offset not in (None, 0):
                corrected = _apply_offset(original_map, orig_offset, size_headers)
                item.quantities = [QuantityPair(size=s, quantity=q) for s, q in corrected.items()]
                note = (
                    f"'{expected}': main reading and row-crop recount agreed, but neither fits "
                    f"{match.bname}'s catalog sizes directly -- shifting by {orig_offset:+d} column(s) "
                    f"uniquely does. Both reads likely shared the same column-drift bias; auto-corrected "
                    f"from the catalog check, not from agreement alone -- verify this row against the photo."
                )
                notes.append(note)
                flags[idx] = {"status": "auto_corrected", "sizes": sorted(corrected.keys(), key=lambda s: int(s) if s.isdigit() else 0), "note": note}
            else:
                item.quantities = row.quantities
                # Agreement alone is real evidence, but not proof -- if this
                # row also carries its own printed/circled total (row_total
                # on the main read, total_dozen on the recount -- either
                # source, whichever is present), a mismatch there is WORTH
                # a note. But NOT necessarily an error: some forms print a
                # per-row number of bundles/packs instead of a quantity
                # total (confirmed real concern, 2026-08-08 -- the prompt
                # now tells the model to leave row_total/total_dozen empty
                # when it can't tell which kind of number it is, but a
                # bundle count could still slip through if it looks
                # plausible). So this is deliberately a soft "unverified"
                # flag, not "unresolved" -- there's no actual disagreement
                # between the two independent reads, only an unconfirmed
                # secondary signal that might not even apply to this form.
                printed_total = _parse_int_or_none(row.total_dozen) or _parse_int_or_none(item.row_total)
                agreed_sum = sum(original_map.values())
                if printed_total is not None and agreed_sum != printed_total:
                    note = (
                        f"'{expected}': main reading and row-crop recount agreed on a total of "
                        f"{agreed_sum}, but this row's own printed/circled total says {printed_total}. "
                        f"This could mean a misread cell -- or this number might not be a quantity "
                        f"total at all on this form (e.g. a bundle/pack count instead). Kept the agreed "
                        f"reading either way; worth a glance against the photo if unsure."
                    )
                    notes.append(note)
                    flags[idx] = {"status": "unverified", "sizes": sorted(original_map.keys(), key=lambda s: int(s) if s.isdigit() else 0), "note": note}
                else:
                    flags[idx] = {"status": "ok"}
            continue

        recount_offset = brandlist_match.resolve_shift_offset(recount_map, match.sizes, size_headers) if match else None
        if recount_offset is not None:
            calibration_anchors.append((item.row_top_frac, recount_offset))

        pending.append(dict(
            idx=idx, item=item, expected=expected, original_map=original_map, recount_map=recount_map,
            row=row, match=match, orig_offset=orig_offset, recount_offset=recount_offset,
        ))

    # Every disagreement gets Total Dozen, then the catalog-offset check --
    # both grounded in this specific row's own evidence. Only a row with
    # NEITHER (no catalog match of its own, Total Dozen absent/tied) falls
    # through to the row-depth trend fitted from every OTHER row's
    # catalog-confirmed offset above -- and even then, only as an explicit
    # note, never an auto-apply, since that's borrowed evidence, not this
    # row's own.
    trend = _fit_drift_trend(calibration_anchors)

    for p in pending:
        idx, item, expected = p["idx"], p["item"], p["expected"]
        original_map, recount_map, row, match = p["original_map"], p["recount_map"], p["row"], p["match"]
        orig_offset, recount_offset = p["orig_offset"], p["recount_offset"]

        chosen_map, chosen_label, resolved = original_map, "main (default, unresolved)", False

        # Prefer the recount's own total_dozen reading, but fall back to the
        # main call's row_total when the recount didn't capture one (e.g.
        # whole-table fallback mode, or this specific row's recount call
        # failed) -- either is the same printed/circled total on the page,
        # just read by a different call.
        printed_total = _parse_int_or_none(row.total_dozen) or _parse_int_or_none(item.row_total)
        if printed_total is not None:
            orig_sum, recount_sum = sum(original_map.values()), sum(recount_map.values())
            if recount_sum == printed_total and orig_sum != printed_total:
                chosen_map, chosen_label, resolved = recount_map, "recount (matched this row's Total Dozen)", True
            elif orig_sum == printed_total and recount_sum != printed_total:
                chosen_map, chosen_label, resolved = original_map, "main (matched this row's Total Dozen)", True

        # Catalog tiebreak, using resolve_shift_offset's unambiguous-offset
        # test rather than a plain "is this a subset of the catalog's sizes"
        # check -- stricter than the old subset check (confirmed necessary:
        # this form's rows often share overlapping catalog size ranges, so a
        # plain subset check can find BOTH candidates "valid" and fail to
        # discriminate at all; requiring the offset to be the unique fit
        # avoids treating that ambiguity as a resolution).
        if not resolved and match:
            if orig_offset == 0 and recount_offset != 0:
                chosen_map, chosen_label, resolved = original_map, f"main (matches {match.bname}'s catalog sizes)", True
            elif recount_offset == 0 and orig_offset != 0:
                chosen_map, chosen_label, resolved = recount_map, f"recount (matches {match.bname}'s catalog sizes)", True

        trend_hint = ""
        if not resolved and not match and trend is not None:
            slope, intercept = trend
            predicted = round(slope * item.row_top_frac + intercept)
            if predicted != 0:
                trend_hint = (
                    f" This form's other catalog-confirmed rows suggest a {predicted:+d}-column drift is "
                    f"plausible around this row's depth in the table -- not applied automatically (no "
                    f"catalog match for this specific item to confirm it directly), but worth checking first."
                )

        all_sizes = sorted(set(original_map) | set(recount_map), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
        diffs = "; ".join(
            f"{s}: main={original_map.get(s, '-')} vs recount={recount_map.get(s, '-')}"
            for s in all_sizes if original_map.get(s) != recount_map.get(s)
        )
        item.quantities = [QuantityPair(size=s, quantity=q) for s, q in chosen_map.items()]
        confidence = "resolved" if resolved else "UNRESOLVED"
        note = (
            f"NEEDS REVIEW ({confidence}) -- '{expected}': main reading and row-crop recount disagree ({diffs}). "
            f"Kept the {chosen_label} reading; verify this row against the photo.{trend_hint}"
        )
        notes.append(note)
        differing_sizes = [s for s in all_sizes if original_map.get(s) != recount_map.get(s)]
        flags[idx] = {"status": "resolved" if resolved else "unresolved", "sizes": differing_sizes, "note": note}

    return notes, flags


_EMPTY_ROW_READING_KWARGS = dict(item_seen="", first_size="", last_size="", quantities=[], total_dozen="")


def _recount_one_row(client: anthropic.Anthropic, model: str, crop: tuple[bytes, str], label: str) -> tuple["RowQuantityReading", "str | None", "dict | None"]:
    """One recount call scoped to a SINGLE row's header+row crop -- the unit
    of work _recount_quantities_live now parallelizes across all of a form's
    rows (added 2026-08-08, replacing the old single call that bundled every
    row's crop into one message; see RECOUNT_ROW_MAX_TOKENS above for why).

    Returns (reading, error_note, usage_breakdown). On truncation/refusal/any
    exception, returns an empty reading rather than raising -- a single bad
    row degrades to "kept the original reading for this row" in
    _apply_recount, instead of that row (or, as under the old single-call
    design, EVERY row after it in the same call) losing recount coverage
    entirely."""
    crop_bytes, media_type = crop
    params = dict(
        model=model,
        max_tokens=RECOUNT_ROW_MAX_TOKENS,
        system=[{"type": "text", "text": RECOUNT_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=_build_recount_messages([(crop_bytes, media_type, label)]),
        # effort="medium" was tried on the (then single, whole-form) recount
        # call on 2026-08-07 to fight max_tokens truncation -- REVERTED the
        # same day: a real re-run at medium effort undercounted multiple rows
        # (e.g. B.4749 dropped from its catalog-confirmed 8 filled cells to
        # 3), a genuine accuracy regression, not just a cost tradeoff. Left
        # at the default (adaptive thinking, high effort) here too, on
        # purpose -- don't re-try lower effort without new evidence.
        output_config={"format": {"type": "json_schema", "schema": RowQuantityReading.model_json_schema()}},
    )
    try:
        with client.messages.stream(**params) as stream:
            message = stream.get_final_message()
    except Exception as exc:
        return RowQuantityReading(**_EMPTY_ROW_READING_KWARGS), str(exc), None

    breakdown = _usage_breakdown(model, message.usage) if message.usage is not None else None

    if message.stop_reason == "max_tokens":
        return RowQuantityReading(**_EMPTY_ROW_READING_KWARGS), "truncated at max_tokens", breakdown
    if message.stop_reason == "refusal":
        return RowQuantityReading(**_EMPTY_ROW_READING_KWARGS), "declined (safety refusal)", breakdown

    text = next(b.text for b in message.content if b.type == "text")
    return RowQuantityReading.model_validate_json(text), None, breakdown


def _recount_quantities_live(client: anthropic.Anthropic, model: str, image_path: Path, extracted: "ExtractedForm", usage_log: Path, brandlist_available: bool = False) -> tuple[list[str], list[dict]]:
    """Second pass: re-reads only the size/quantity grid, forcing explicit
    column anchors and a Total-Dozen self-check -- see the module-level
    comment above RECOUNT_SYSTEM_PROMPT for why this exists. Prefers true
    per-row crops (_build_row_crops, using each item's own row_top_frac/
    row_bottom_frac from the main call) since a whole-table crop was
    confirmed (2026-08-07, sample 4.jpeg) to still leave rows far from the
    header vulnerable to the same column-drift failure mode, just with more
    pixels. Falls back to a single whole-table crop if the per-row fractions
    don't validate (see _validate_row_fracs).

    Per-row crops are read via N SEPARATE, PARALLEL calls (_recount_one_row),
    not one call bundling every row -- changed 2026-08-08 after usage_log.csv
    showed the old single bundled call regularly hitting its 32000-token cap
    on 12+ row forms (thinking + a full RowQuantityReading per row, compounded
    across every row in one continuous call), which was simultaneously the
    dominant source of per-image latency (~5-7 of the ~8-10 total minutes) AND
    a silent-accuracy risk: truncation partway through meant every row after
    the cutoff point lost recount coverage entirely, with no indication which
    ones. Splitting into parallel single-row calls fixes both: each call only
    needs to think about one row (a much smaller, boundeder budget --
    RECOUNT_ROW_MAX_TOKENS), and a bad/truncated row no longer affects any
    other row's result.

    Returns (notes, flags) -- see _apply_recount's docstring for what flags
    is; here it's just threaded through so extract_one_live can persist it
    as a debug artifact. Mutates extracted.items' quantities in place via
    _apply_recount(). Never raises for a bad/implausible crop region --
    that's "skip this pass," not a hard failure, since the main call's
    quantities are still usable."""
    top, bottom = extracted.table_top_frac, extracted.table_bottom_frac
    if not (0.0 <= top < bottom <= 1.0):
        no_recount_flags = [{"status": "no_recount", "note": "table_top_frac/table_bottom_frac from the main read looked implausible"} for _ in extracted.items]
        return [f"Row-crop quantity recount skipped: table_top_frac/table_bottom_frac from the main read looked implausible ({top}, {bottom})."], no_recount_flags

    original_bytes = image_path.read_bytes()
    headers_line = (
        f"This form's real size headers, left to right, are exactly: {', '.join(extracted.size_headers)} -- "
        f"if a header crop shows a second, smaller line of numbers below these, that second line is NOT one "
        f"of these headers and must be ignored, per the system prompt."
    ) if extracted.size_headers else ""

    per_row_mode = _validate_row_fracs(extracted.items, top, bottom)

    if not per_row_mode:
        crop_bytes, media_type = _prepare_table_crop(original_bytes, top, bottom)
        item_list = "\n".join(
            f"{i + 1}. {it.item}" + (f" (Style: {it.type})" if it.type else "")
            for i, it in enumerate(extracted.items)
        )
        label = f"The item rows, top to bottom, are:\n{item_list}" + (f"\n{headers_line}" if headers_line else "")
        params = dict(
            model=model,
            max_tokens=RECOUNT_MAX_TOKENS,
            system=[{"type": "text", "text": RECOUNT_SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=_build_recount_messages([(crop_bytes, media_type, label)]),
            output_config={"format": {"type": "json_schema", "schema": QuantityRecount.model_json_schema()}},
        )
        with client.messages.stream(**params) as stream:
            message = stream.get_final_message()

        if message.usage is not None:
            breakdown = _usage_breakdown(model, message.usage)
            print(f"  usage (recount, whole-table): {_format_usage(breakdown)}")
            _log_usage(usage_log, image_path.name, model, "live-recount-whole-table", breakdown)

        if message.stop_reason == "max_tokens":
            truncated_flags = [{"status": "no_recount", "note": "whole-table recount call truncated at max_tokens"} for _ in extracted.items]
            return ["Row-crop quantity recount truncated at max_tokens -- kept the original full-page quantities for every row."], truncated_flags
        if message.stop_reason == "refusal":
            refused_flags = [{"status": "no_recount", "note": "whole-table recount call declined (safety refusal)"} for _ in extracted.items]
            return ["Row-crop quantity recount declined (safety refusal) -- kept the original full-page quantities for every row."], refused_flags

        text = next(b.text for b in message.content if b.type == "text")
        recount = QuantityRecount.model_validate_json(text)
        notes, flags = _apply_recount(extracted, recount, brandlist_available)
        notes.append("Row-crop quantity recount used the whole-table fallback crop (per-item row_top_frac/row_bottom_frac from the main call didn't validate) -- less isolation between rows than the intended per-row crops.")
        return notes, flags

    row_crops = _build_row_crops(original_bytes, top, extracted.items)
    labels = [
        f"Row {i + 1}: {it.item}" + (f" (Style: {it.type})" if it.type else "") + (f"\n{headers_line}" if headers_line else "")
        for i, it in enumerate(extracted.items)
    ]

    readings: list = [None] * len(extracted.items)
    errors: list = [None] * len(extracted.items)
    breakdowns: list = []
    max_workers = min(RECOUNT_MAX_WORKERS, len(row_crops))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_recount_one_row, client, model, crop, label): i
            for i, (crop, label) in enumerate(zip(row_crops, labels))
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            reading, err, breakdown = fut.result()
            readings[i] = reading
            errors[i] = err
            if breakdown is not None:
                breakdowns.append(breakdown)

    if breakdowns:
        total_breakdown = {
            "input_tokens": sum(b["input_tokens"] for b in breakdowns),
            "cache_read_tokens": sum(b["cache_read_tokens"] for b in breakdowns),
            "cache_write_tokens": sum(b["cache_write_tokens"] for b in breakdowns),
            "output_tokens": sum(b["output_tokens"] for b in breakdowns),
            "estimated_cost_usd": sum(b["estimated_cost_usd"] for b in breakdowns if b["estimated_cost_usd"] is not None) or None,
        }
        print(f"  usage (recount, per-row x{len(breakdowns)}): {_format_usage(total_breakdown)}")
        _log_usage(usage_log, image_path.name, model, f"live-recount-per-row-x{len(breakdowns)}", total_breakdown)

    recount = QuantityRecount(rows=readings)
    notes, flags = _apply_recount(extracted, recount, brandlist_available)
    for i, err in enumerate(errors):
        if err:
            expected = extracted.items[i].item
            note = f"Row-crop recount call for '{expected}' failed ({err}) -- kept the original full-page reading for this row."
            notes.append(note)
            flags[i] = {"status": "unverified", "sizes": [qp.size for qp in extracted.items[i].quantities], "note": note}
    return notes, flags


def _request_params(model: str, image_bytes: bytes, media_type: str, system_prompt: str) -> dict:
    return dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
        messages=_build_messages(image_bytes, media_type),
        output_config={"format": {"type": "json_schema", "schema": ExtractedForm.model_json_schema()}},
    )


def _usage_breakdown(model: str, usage, batch: bool = False) -> dict:
    """Raw token counts plus an estimated USD cost (None if `model` isn't in
    PRICING_PER_MTOK) -- shared by the console printout and the usage log so
    the two can never disagree."""
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    input_tok = usage.input_tokens
    output_tok = usage.output_tokens

    cost = None
    prices = PRICING_PER_MTOK.get(model)
    if prices is not None:
        in_price, out_price = prices
        cost = (
            input_tok * in_price
            + cache_read * in_price * CACHE_READ_MULTIPLIER
            + cache_write * in_price * CACHE_WRITE_MULTIPLIER
            + output_tok * out_price
        ) / 1_000_000
        if batch:
            cost *= 0.5  # Message Batches API: 50% off every token above

    return {
        "input_tokens": input_tok,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "output_tokens": output_tok,
        "estimated_cost_usd": cost,
    }


def _format_usage(breakdown: dict, batch: bool = False) -> str:
    cost = breakdown["estimated_cost_usd"]
    cost_str = f"~${cost:.4f}" if cost is not None else "cost unknown (model not in PRICING_PER_MTOK)"
    tag = " [batch]" if batch else ""
    return (
        f"{cost_str}{tag} (in={breakdown['input_tokens']} cache_read={breakdown['cache_read_tokens']} "
        f"cache_write={breakdown['cache_write_tokens']} out={breakdown['output_tokens']})"
    )


def _log_usage(log_path: Path, image_name: str, model: str, mode: str, breakdown: dict) -> None:
    is_new = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "timestamp_utc", "image", "model", "mode",
                "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens",
                "estimated_cost_usd",
            ])
        cost = breakdown["estimated_cost_usd"]
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            image_name, model, mode,
            breakdown["input_tokens"], breakdown["cache_read_tokens"],
            breakdown["cache_write_tokens"], breakdown["output_tokens"],
            f"{cost:.6f}" if cost is not None else "",
        ])


def _parse_message(message) -> ExtractedForm:
    if message.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request (safety refusal) -- check the image/system prompt.")
    if message.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Response truncated at max_tokens ({MAX_TOKENS}) before the JSON finished -- raise "
            f"MAX_TOKENS in extract_claude.py further, this form's table is denser than that budget covers."
        )
    text = next(b.text for b in message.content if b.type == "text")
    return ExtractedForm.model_validate_json(text)


def _to_order_form(extracted: ExtractedForm, source_name: str) -> OrderForm:
    items = []
    for it in extracted.items:
        # gt=0 isn't representable in the JSON schema we send (structured
        # outputs doesn't support numeric constraints), so filter defensively
        # here instead -- same "no non-positive quantities" rule the Ollama
        # pipeline's Stage C enforces.
        quantities = {qp.size: qp.quantity for qp in it.quantities if qp.quantity > 0}
        items.append(OrderItem(item=it.item, type=it.type, quantities=quantities))
    return OrderForm(
        party_name=extracted.party_name,
        order_no=extracted.order_no,
        order_date=extracted.order_date,
        items=items,
        notes=extracted.notes,
        source_file=source_name,
    )


def _write_debug_artifacts(outdir: Path, stem: str, extracted: ExtractedForm, recount_flags: list[dict] | None = None) -> None:
    (outdir / f"{stem}.raw.json").write_text(
        json.dumps(extracted.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / f"{stem}.stageA.json").write_text(
        json.dumps({"seller_name": extracted.seller_name, "size_headers": extracted.size_headers}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if recount_flags is not None:
        # One entry per item, aligned by index with the final OrderForm's
        # items list -- see _apply_recount's docstring for what each status
        # means. generate_review.py reads this (2026-08-08) to highlight the
        # SPECIFIC quantity cells worth a second look, instead of leaving
        # that signal buried in a paragraph of free text at the top of the
        # review page.
        (outdir / f"{stem}.recount_flags.json").write_text(
            json.dumps(recount_flags, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def extract_one_live(client: anthropic.Anthropic, model: str, image_path: Path, outdir: Path, usage_log: Path, system_prompt: str, brandlist_available: bool = False) -> OrderForm:
    image_bytes, media_type = _prepare_image(image_path)
    # Streaming, not a plain create() call -- required once max_tokens is
    # this high, to avoid the SDK's own non-streaming HTTP-timeout guard.
    with client.messages.stream(**_request_params(model, image_bytes, media_type, system_prompt)) as stream:
        message = stream.get_final_message()
    # Log usage BEFORE parsing -- the call already happened and was billed
    # regardless of whether _parse_message() below succeeds. Logging after
    # it (the original order) meant a truncated/malformed response's real
    # cost never made it into usage_log.csv, since the exception skipped
    # straight past the logging code -- confirmed missing exactly this way
    # for the max_tokens truncation hit earlier.
    if message.usage is not None:
        breakdown = _usage_breakdown(model, message.usage)
        print(f"  usage: {_format_usage(breakdown)}")
        _log_usage(usage_log, image_path.name, model, "live", breakdown)
    extracted = _parse_message(message)

    recount_flags: list[dict] = []
    if extracted.items:
        try:
            notes, recount_flags = _recount_quantities_live(client, model, image_path, extracted, usage_log, brandlist_available)
            extracted.notes.extend(notes)
        except Exception as exc:
            extracted.notes.append(f"Row-crop quantity recount failed ({exc}) -- kept the original full-page quantities for every row.")
            recount_flags = [{"status": "no_recount", "note": f"recount pass raised an exception: {exc}"} for _ in extracted.items]

    _write_debug_artifacts(outdir, image_path.stem, extracted, recount_flags)
    return _to_order_form(extracted, image_path.name)


def extract_batch(client: anthropic.Anthropic, model: str, image_paths: list[Path], outdir: Path, usage_log: Path, system_prompt: str) -> dict[str, OrderForm]:
    # NOTE: does NOT run the row-crop quantity recount pass (see
    # _recount_quantities_live / RECOUNT_SYSTEM_PROMPT above) -- that pass
    # needs each image's table_top_frac/table_bottom_frac from ITS OWN
    # first-call result before it can build the second call, which would
    # mean a second batch round-trip (submit, wait, poll) per run rather
    # than the current single one. Not built until that's worth the added
    # wall-clock time for a real batch run -- live mode (extract_one_live)
    # is where this has been implemented and verified so far.
    #
    # custom_id must be alnum/underscore/hyphen only -- image stems like
    # "sample 5" (a space) aren't valid, so use an index and map back.
    id_to_path: dict[str, Path] = {}
    requests = []
    for i, p in enumerate(image_paths):
        custom_id = f"img{i:04d}"
        id_to_path[custom_id] = p
        image_bytes, media_type = _prepare_image(p)
        requests.append(Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(**_request_params(model, image_bytes, media_type, system_prompt)),
        ))

    batch = client.messages.batches.create(requests=requests)
    print(f"Submitted batch {batch.id} ({len(requests)} images, 50% batch discount + prompt caching apply) -- polling...", flush=True)

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(f"  ... processing={c.processing} succeeded={c.succeeded} errored={c.errored}", flush=True)
        time.sleep(BATCH_POLL_SECONDS)

    forms: dict[str, OrderForm] = {}
    total_cost = 0.0
    for result in client.messages.batches.results(batch.id):
        path = id_to_path[result.custom_id]
        if result.result.type != "succeeded":
            # No message/usage on a transport-level failure (errored/canceled/
            # expired) -- nothing billable to log here, unlike the case below.
            print(f"  FAILED [{path.name}]: {result.result.type}", file=sys.stderr)
            continue

        # Log usage BEFORE parsing -- a "succeeded" batch result was still
        # billed even if our client-side JSON parsing then fails on it (e.g.
        # truncated at max_tokens). See extract_one_live() for the same fix.
        usage = getattr(result.result.message, "usage", None)
        if usage is not None:
            breakdown = _usage_breakdown(model, usage, batch=True)
            _log_usage(usage_log, path.name, model, "batch", breakdown)
            if breakdown["estimated_cost_usd"] is not None:
                total_cost += breakdown["estimated_cost_usd"]

        try:
            extracted = _parse_message(result.result.message)
        except Exception as exc:
            print(f"  FAILED [{path.name}]: {exc}", file=sys.stderr)
            continue
        _write_debug_artifacts(outdir, path.stem, extracted)
        forms[path.stem] = _to_order_form(extracted, path.name)
    if total_cost:
        print(f"  batch usage: ~${total_cost:.4f} total (50% batch discount applied)")
    return forms


def flatten_for_review(form: OrderForm) -> list[dict]:
    rows = []
    notes = "; ".join(form.notes)

    if not form.items:
        rows.append({
            "source_file": form.source_file, "party_name": form.party_name,
            "order_no": form.order_no, "order_date": form.order_date,
            "item": "", "type": "", "size": "", "quantity": "", "notes": notes,
        })
        return rows

    for it in form.items:
        if not it.quantities:
            rows.append({
                "source_file": form.source_file, "party_name": form.party_name,
                "order_no": form.order_no, "order_date": form.order_date,
                "item": it.item, "type": it.type, "size": "", "quantity": "", "notes": notes,
            })
            continue
        for size, qty in it.quantities.items():
            rows.append({
                "source_file": form.source_file,
                "party_name": form.party_name,
                "order_no": form.order_no,
                "order_date": form.order_date,
                "item": it.item,
                "type": it.type,
                "size": size,
                "quantity": qty,
                "notes": notes,
            })
    return rows


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Extract structured data from order form photos via the Claude API (one schema-constrained call per image).")
    parser.add_argument("input", help="Path to a single image, or a folder of images. Quote paths containing spaces.")
    parser.add_argument("--outdir", default="extracted_claude", help="Output directory (default: ./extracted_claude)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Claude model ID (default: {DEFAULT_MODEL}; try claude-opus-5 for forms Sonnet 5 gets wrong)")
    parser.add_argument("--live", action="store_true", help="For a folder, process images with sequential live calls instead of the Message Batches API (loses the 50%% batch discount and the queue wait -- useful for a quick test on a couple of images)")
    parser.add_argument("--usage-log", default="usage_log.csv", help="CSV file every call's token usage/cost is appended to, across all runs (default: ./usage_log.csv)")
    parser.add_argument("--no-brandlist-check", action="store_true", help="Skip the local (free, no API cost) cross-check against the brandlist product catalog -- item-name/style suggestions and letter-size resolution. On by default; use this if the DB isn't reachable and you don't want the warning.")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        print("Error: ANTHROPIC_API_KEY is not set. Add it to .env (as a new line -- don't touch the existing DB credentials there) or export it before running.", file=sys.stderr)
        sys.exit(1)

    style_codes: list[str] = []
    brandlist_available = False
    if not args.no_brandlist_check:
        try:
            style_codes = brandlist_match.known_style_codes()
            brandlist_available = True
        except Exception as exc:
            print(f"Warning: couldn't reach the brandlist DB ({exc}) -- using the built-in style-code list and skipping the item/size cross-check for this run.", file=sys.stderr)
    system_prompt = build_system_prompt(style_codes)

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        image_paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    elif input_path.is_file():
        image_paths = [input_path]
    else:
        print(f"Error: '{input_path}' not found. If the path has spaces, wrap it in quotes.", file=sys.stderr)
        sys.exit(1)

    if not image_paths:
        print(f"No images found in {input_path}.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()
    usage_log = Path(args.usage_log)
    forms: dict[str, OrderForm] = {}

    if input_path.is_dir() and not args.live:
        forms = extract_batch(client, args.model, image_paths, outdir, usage_log, system_prompt)
    else:
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Extracting {img_path.name} ...", flush=True)
            try:
                forms[img_path.stem] = extract_one_live(client, args.model, img_path, outdir, usage_log, system_prompt, brandlist_available)
            except Exception as exc:
                print(f"  FAILED: {exc}", file=sys.stderr)

    all_review_rows = []
    for img_path in image_paths:
        form = forms.get(img_path.stem)
        if form is None:
            continue
        if brandlist_available:
            stage_a_path = outdir / f"{img_path.stem}.stageA.json"
            stage_a = json.loads(stage_a_path.read_text(encoding="utf-8")) if stage_a_path.exists() else {}
            size_headers = stage_a.get("size_headers", [])
            annotations = brandlist_match.annotate_and_resolve(form, size_headers)
            (outdir / f"{img_path.stem}.brandlist.json").write_text(
                json.dumps(annotations, indent=2, ensure_ascii=False)
            )
            # Cross-checks seller_name/party_name against Essa's own buyer
            # master -- catches the case where a form uses a DIFFERENT
            # business's own pre-printed order pad (seller_name isn't Essa),
            # where the handwritten "Party Name" field may just be that
            # business's own downstream customer rather than the real buyer
            # for Essa's records. See brandlist_match.resolve_party_name()
            # for the confirmed real case this is built from (sample 8.jpeg).
            party_check = brandlist_match.resolve_party_name(stage_a.get("seller_name", ""), form.party_name)
            if party_check is not None:
                (outdir / f"{img_path.stem}.party_check.json").write_text(
                    json.dumps(party_check, indent=2, ensure_ascii=False)
                )
        json_out = outdir / f"{img_path.stem}.json"
        json_out.write_text(json.dumps(form.model_dump(mode="json", exclude={"source_file"}), indent=2, ensure_ascii=False))
        n_qty = sum(len(it.quantities) for it in form.items)
        print(f"  -> {json_out.name}  ({len(form.items)} items, {n_qty} size/qty cells, {len(form.notes)} notes)")
        all_review_rows.extend(flatten_for_review(form))

    if all_review_rows:
        review_csv = outdir / "review.csv"
        fieldnames = list(all_review_rows[0].keys())
        with review_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_review_rows)
        print(f"\nWrote {review_csv} ({len(all_review_rows)} rows).")

    if usage_log.exists():
        with usage_log.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        grand_total = sum(float(r["estimated_cost_usd"]) for r in rows if r["estimated_cost_usd"])
        print(f"\n{usage_log}: {len(rows)} calls logged, ~${grand_total:.4f} total across every run so far.")


if __name__ == "__main__":
    main()
