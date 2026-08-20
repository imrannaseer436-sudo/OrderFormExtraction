#!/usr/bin/env python3
"""
extract_ollama_cloud.py -- Ollama Cloud comparison test (additive, experimental).

Tests a hosted open-weight vision model (via Ollama Cloud) as an alternative
to extract_claude.py's Claude API calls, for COMPARISON PURPOSES ONLY -- not
a replacement for extract_claude.py, which remains the recommended pipeline
(see CLAUDE.md). This file does not modify extract_claude.py or any other
existing file: it imports that file's already-tuned Pydantic schemas
(ExtractedForm, ExtractedItem, QuantityPair, QuantityRecount,
RowQuantityReading) and prompt text (SYSTEM_PROMPT_TEMPLATE via
build_system_prompt, RECOUNT_SYSTEM_PROMPT) verbatim, and reuses its
crop/validate/merge helpers (_prepare_image, _prepare_table_crop,
_build_row_crops, _validate_row_fracs, _apply_recount, _to_order_form,
_write_debug_artifacts, flatten_for_review) rather than reimplementing that
logic. Only the model-call sites differ: this file talks to Ollama's chat()
API instead of Anthropic's messages API. The overall architecture (one
schema-constrained main call -> per-row-crop recount pass, run as N parallel
calls -> agree-or-flag merge via _apply_recount) is unchanged from
extract_claude.py -- see that file's module docstring and CLAUDE.md for the
full rationale.

MODEL CHOICE -- confirmed by direct testing (2026-08-11), not assumed:
the originally-targeted qwen3-vl:235b-cloud is RETIRED on Ollama Cloud (a
live call returns HTTP 410, "retired at 2026-06-16"). Of the plausible
large-model substitutes listed by the Cloud API, most require a paid
Ollama plan this project's API key doesn't have (qwen3.5:397b, glm-5.1,
glm-5.2, kimi-k3 all returned HTTP 403 "requires a subscription");
nemotron-3-super explicitly rejects image input (HTTP 400 "this model does
not support image input"); minimax-m3 accepts an images param but answered
a direct color-identification sanity check incorrectly (a solid red test
square) -- accepted, but not trustworthy. gemma4:31b is the only model that
was BOTH accessible on this account's plan AND answered that same sanity
check correctly ("Red"). DEFAULT_MODEL below reflects that real result, not
the model named in the original task -- override with --model once/if a
paid plan unlocks one of the gated candidates.

STRUCTURED-OUTPUT RELIABILITY -- confirmed by direct testing, a real
difference from both extract_claude.py's Claude calls and the local Ollama
pipeline's qwen2.5vl calls (extract_ollama.py): passing format=<json
schema> to Ollama Cloud's gemma4:31b does NOT strictly constrain the
output the way Anthropic's structured outputs or local qwen2.5vl's grammar
constraint do. A real call against sample 5.jpeg, with format= set to
ExtractedForm's schema, came back (a) wrapped in ```json ... ``` markdown
fences, (b) missing a required field (order_no) entirely, (c) carrying an
extra top-level key not in the schema at all ("layout"), and (d) using
JSON null for non-nullable string fields (row_total) instead of "". None
of this is a formatting nicety Pydantic's default lax coercion papers
over -- model_validate_json() raised on the raw text (invalid JSON, due to
the fences) and model_validate() on the fence-stripped dict raised 13
separate validation errors. _strip_json_fences() and _normalize_for_schema()
below exist specifically to repair this before validation runs, so the
same field validators/model validators ExtractedForm and friends already
have (ditto forward-fill, trailing style-code split, order_date cleanup)
still get a fair shot at running on real field values instead of erroring
out first on a missing key or a stray null.

Usage:
    python3 extract_ollama_cloud.py "Images/sample 5.jpeg"
    python3 extract_ollama_cloud.py Images/ --outdir extracted_ollama_cloud
    python3 extract_ollama_cloud.py Images/ --model gemma4:31b

Prerequisite: OLLAMA_API_KEY=... in .env (an Ollama Cloud API key from
ollama.com/settings/keys) -- added as a new line, same convention as this
project's existing ANTHROPIC_API_KEY/DB credentials in .env.

Output per image (written to --outdir, default extracted_ollama_cloud/, kept
separate from every other pipeline's output directory):
    <name>.json               -- final structured data (OrderForm shape)
    <name>.raw.json            -- full parsed model output before conversion (debugging)
    <name>.stageA.json         -- {seller_name, size_headers} only, for generate_review.py
    <name>.recount_flags.json  -- per-row agree-or-flag status, see _apply_recount in extract_claude.py
    <name>.brandlist.json      -- per-item catalog match/suggestion notes (skipped if --no-brandlist-check)
    review.csv                 -- flattened, one row per (item, size)

Every call (main or per-row/whole-table recount) appends one row to a
persistent usage log, kept entirely separate from extract_claude.py's own
usage_log.csv so this experiment's numbers never mix with the real Claude
pipeline's cost/usage record (default ./usage_log_ollama_cloud.csv, override
with --usage-log). Columns mirror usage_log.csv's shape where they apply
(input_tokens <- prompt_eval_count, output_tokens <- eval_count;
cache_read_tokens/cache_write_tokens/estimated_cost_usd are left blank --
Ollama Cloud doesn't report prompt caching the way Anthropic does, and this
file doesn't have a verified per-model price to estimate a cost from), plus
two columns extract_claude.py's log doesn't have: duration_seconds (real
wall-clock time for that specific call, timed locally with
time.perf_counter() -- CLAUDE.md notes this was never measured for any
pipeline before now) and error (non-empty when a call failed or came back
unparseable, logged immediately rather than only on success -- the same
"log usage before parsing" fix extract_claude.py made after losing a
truncated call's real cost).
"""

import argparse
import concurrent.futures
import csv
import io
import json
import os
import re
import sys
import time
import typing
from typing import List
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ollama
from dotenv import load_dotenv
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

import brandlist_match
from grid import iter_row_boundary_candidates_auto
from ocr_cell_read import get_ocr, item_alignment_ok, ocr_row, _try_digit_correct, _monotonic_assign
from preprocess_for_vlm import preprocess_for_vlm
from extract_ollama import build_header_crop, build_row_crop
from schema import OrderForm
from extract_claude import (
    IMAGE_EXTENSIONS,
    MAX_LONG_EDGE,
    ExtractedForm,
    ExtractedItem,
    QuantityPair,
    QuantityRecount,
    RowQuantityReading,
    RECOUNT_MAX_WORKERS,
    RECOUNT_SYSTEM_PROMPT,
    USER_PROMPT,
    _EMPTY_ROW_READING_KWARGS,
    _apply_recount,
    _build_row_crops,
    _parse_int_or_none,
    _prepare_image,
    _prepare_table_crop,
    _to_order_form,
    _validate_row_fracs,
    _write_debug_artifacts,
    build_system_prompt,
    flatten_for_review,
)

# See module docstring's "MODEL CHOICE" section for why this isn't
# qwen3-vl:235b-cloud (the originally-targeted model, confirmed retired).
DEFAULT_MODEL = "gemma4:31b"
OLLAMA_CLOUD_HOST = "https://ollama.com"

# Analogous to extract_claude.py's MAX_TOKENS/RECOUNT_ROW_MAX_TOKENS, passed
# as options.num_predict. A real single-image main call against sample
# 5.jpeg used eval_count=2938 of this budget (see CLAUDE.md's dated section
# for this test) -- generous headroom kept here since gemma4:31b's real
# per-form output size on a denser form hasn't been characterized yet.
MAX_TOKENS = 32000
RECOUNT_ROW_MAX_TOKENS = 8000
RECOUNT_MAX_TOKENS = 32000  # whole-table fallback call only, mirrors extract_claude.py

# qwen3.5:397b (and possibly other Cloud models) run extended internal
# "thinking" by default even at temperature=0, confirmed by direct testing
# 2026-08-13: main-call duration on sample 5.jpeg dropped from 254s to 53s
# with this off, output token count fell from 13760 to 2231, and accuracy
# did not regress (12/14 rows exact vs 11/14 with thinking on -- see
# CLAUDE.md's dated entry). NOT a universal win, though: kimi-k2.6 does the
# opposite -- confirmed the same day that think=False makes it return a
# schema-valid response with every quantity deterministically 0 (same
# eval_count across repeated calls, not the random ~1-in-5 flake seen on
# other models), while its default (thinking on) produces a real, non-zero
# reading. THINK is a module-level default overridable per run via
# --think/--no-think (see main()) since the right setting is model-specific,
# not something this file can assume once for every model.
THINK = False

# Confirmed 2026-08-13: mistral-large-3:675b was the clear best Ollama Cloud
# model of five tried -- fast (~30-55s), reliable (no flake observed across
# 5 real trials, unlike qwen3.5:397b's ~1-in-5 rate), and the only one that
# extracted real quantity data on a free-form-layout form (sample 2.jpeg)
# where both gemma4:31b and qwen3.5:397b returned zero quantities twice
# each. This addendum targets confirmed, repeated failure patterns from
# direct testing; it is appended to the shared build_system_prompt() text
# (imported from extract_claude.py, unmodified) only when the model name
# contains "mistral" -- see main() -- so it never touches the prompt any
# other model, or the real extract_claude.py pipeline, actually uses.
#
# Two originally-included bullets were tested and DROPPED after real
# re-runs showed no benefit -- kept out deliberately, not just omitted:
# a COLUMN ALIGNMENT instruction (re-derive header position fresh per row)
# produced a wash on sample 5.jpeg -- some rows' shift improved, others got
# WORSE, net value-accuracy unchanged (66/69 before and after) -- this
# looks like a real positional-grounding limitation prompting can't fix,
# consistent with this project's own history on the local Ollama model. A
# QUANTITY VALUE ACCURACY instruction (aimed at sample 4.jpeg's undercounting,
# where a real value of 5-8 consistently came back as 1) was tried in two
# different framings (first as "count individual tally strokes," corrected
# after the user clarified that framing was wrong -- these forms don't
# necessarily use tally marks at all, whatever mark type is present the
# model is just undercounting it -- then reworded as generic "don't default
# to a low placeholder") and produced byte-identical output both times on
# the same ground-truth row. Confirmed a real model ceiling, not a wording
# problem -- not worth spending prompt tokens on every call.
#
# Two new bullets added 2026-08-14 after a real sample 2.jpeg (free-form,
# no printed grid) run: a wrapped-continuation-line rule (mistral split an
# item's overflow line into a fake second item) and a no-fabrication rule
# (mistral separately reported one whole item, "BLOOMER PRINT" a second
# time, with quantities that don't correspond to anything on the page at
# all). Unlike the two dropped bullets above, these are structural/
# instruction-following errors, not raw pixel-reading limits, so they're a
# more plausible candidate for a real prompt fix -- not yet confirmed by a
# re-run.


class MistralExtractedForm(BaseModel):
    """Mistral-only variant of extract_claude.py's ExtractedForm -- NOT a
    subclass, a fully parallel definition, because Pydantic appends a
    subclass's own new fields after its parent's inherited ones when
    building the JSON schema, and grammar-constrained decoding generates
    JSON properties in schema-declared order. date_present must be
    generated BEFORE order_date for the presence-gating below to mean
    anything -- committing to "is there a real date here at all" before
    the model has already started generating date-shaped text is the
    whole point (mistral was confirmed, across 3 separate tests including
    with --think, to reproduce the identical fabricated date on
    sample 2.jpeg regardless of prompt wording alone -- see CLAUDE.md's
    dated section on this). Every other field is copied verbatim from
    ExtractedForm; only order_date's description and the new date_present
    field differ. Deliberately kept in this file, not extract_claude.py --
    per this project's established principle, mistral-specific prompt/
    schema engineering stays out of the file shared with the real Claude
    pipeline."""
    model_config = ConfigDict(extra="forbid")
    seller_name: str = Field(description="Letterhead/seller company name printed at the top of the form -- NOT the buyer.")
    party_name: str = Field(description="Buyer/customer name, handwritten in the Party Name box. Empty string if not present.")
    order_no: str = Field(description="Order form number if present, else empty string.")
    date_present: bool = Field(description="True only if an actual calendar date (a day, month, and year -- however abbreviated, e.g. '30/3/26') is written or printed anywhere on the page. False if there is no such date -- a pre-printed day-of-week label alone (e.g. a diary/planner page's \"MONDAY\") is NOT a date and does not count; decide this before you consider what order_date should be.")
    order_date: str = Field(description="Date normalized to DD/MM/YYYY, assuming 20xx for 2-digit years. ONLY the final value. Must be an empty string whenever date_present is false -- never invent or estimate a date just to fill this field.")
    size_headers: List[str] = Field(description="GRID LAYOUT: every size column header across the top of the shared table, left to right, exactly as printed. FREE-FORM LAYOUT (no single shared table): empty list.")
    items: List[ExtractedItem] = Field(description="Every product/article row, top to bottom, in order.")
    notes: List[str] = Field(description="Page-level handwritten notes and anything illegible/unusual not tied to one row's quantities.")
    table_top_frac: float = Field(description="0.0-1.0 fraction of image height where the item data starts -- GRID LAYOUT: the header row; FREE-FORM LAYOUT: the first item's own data. See table_top_frac/table_bottom_frac rule.")
    table_bottom_frac: float = Field(description="0.0-1.0 fraction of image height where the last item's data ends, including any row squeezed into an irregular space (e.g. outside the main ruled grid in GRID LAYOUT, or a page margin in FREE-FORM LAYOUT).")

    @model_validator(mode="after")
    def _gate_order_date_on_presence(self) -> "MistralExtractedForm":
        # Defense in depth, per the task spec: discard order_date if
        # date_present is false regardless of what string the model
        # actually returned there -- don't rely on it honoring the
        # instruction above perfectly.
        if not self.date_present:
            self.order_date = ""
        return self


def _mistral_form_to_extracted_form(m: MistralExtractedForm) -> ExtractedForm:
    """Converts a parsed MistralExtractedForm into a regular ExtractedForm
    -- date_present is consumed here (already applied via the gating
    validator above) and dropped, so every function downstream of the main
    call keeps working with the exact same ExtractedForm shape it already
    expects, unaware this schema swap ever happened. Constructing a fresh
    ExtractedForm also runs ITS OWN validators (_clean_order_date,
    _forward_fill_ditto_item_names) on the way in, same as any other
    ExtractedForm."""
    return ExtractedForm(
        seller_name=m.seller_name,
        party_name=m.party_name,
        order_no=m.order_no,
        order_date=m.order_date,
        size_headers=m.size_headers,
        items=m.items,
        notes=m.notes,
        table_top_frac=m.table_top_frac,
        table_bottom_frac=m.table_bottom_frac,
    )


MISTRAL_PROMPT_ADDENDUM_BASE = """

MODEL-SPECIFIC GUIDANCE FOR YOU SPECIFICALLY (confirmed by direct testing against this exact \
model on real forms -- these are real, repeated failure patterns, not hypothetical warnings):

- ITEM NAME vs STYLE CODE SPLIT: on a real test form, you split multi-word item names incorrectly \
-- e.g. "BLOOMER PLAIN" came back as item="BLOOMER" type="PLAIN IE" instead of item="BLOOMER PLAIN" \
type="IE". A word describing color, pattern, or fit (Plain, Print, White, Adults, etc.) is part of \
the item name, not the style code, UNLESS that exact word also appears in this business's known \
style-code list above -- when in doubt, keep it in the item name.

- DUPLICATE VALUES: on a real test form, you occasionally reported the same size:quantity pair \
twice within a single row's output. Read each header column exactly once, left to right, and do \
not repeat a pair you have already reported for that row.

- COMPLETENESS NEAR THE BOTTOM OF THE PAGE: on a real test form, the last 1-2 items returned an \
empty quantities list even though their data was genuinely visible in the photo. A row being near \
the bottom edge of the image is not a reason to give it less attention -- read it as carefully as \
the first row on the page.

- WRAPPED/CONTINUATION LINES: on a free-form (no printed grid) form, a line of size:quantity pairs \
with no new item name or bullet marker before it is a CONTINUATION of the item above it, not a new \
item -- e.g. a form had "BABYCARE DRAWER" with 5 pairs on one line, then 3 more pairs on the very \
next line with no new item name above them; those 3 pairs belong to BABYCARE DRAWER too, not a new, \
separately-named item. Only start a new item when you see actual new item-name text.

- DO NOT INVENT ITEMS: on that same form, you reported an item ("BLOOMER PRINT", a second time) with \
quantities that do not correspond to any handwritten block actually on the page -- a fabricated \
duplicate, not a misread of real content. Only report an item if you can point to its own distinct \
handwritten name and quantity line; never generate an additional entry by pattern-completing from a \
similar nearby item.

- DATE PRESENCE: on a real test form (a diary/planner page with no calendar date anywhere on it, only \
a pre-printed day-of-week label like "MONDAY"), you fabricated a plausible-looking date instead of \
recognizing none was present -- and reproduced the exact same fabricated date across repeated separate \
attempts, which means this is a default completion pattern, not a one-off misread. Decide date_present \
FIRST, before thinking about what order_date should say: is there an actual day+month+year written or \
printed anywhere on this page? A day-of-week label alone does not count. If date_present is false, \
order_date must be an empty string -- do not estimate, guess, or reuse a date from anywhere else in \
your reasoning just to have something to put there.

"""

# An "ASTERISK (*) CELLS ARE A RATIO MARKER" bullet was tried here
# 2026-08-14, targeting sample 10.jpeg (a printed/typed, non-ESSA form
# using "*" instead of a digit, with a "PLEASE DISPATCH AS PER RATIO"
# note) -- REVERTED after a real re-run showed no benefit: the severe
# column-position shift already present (values correct, but attached to
# a header 8-10 positions away from the real one -- worse than the usual
# 1-position drift seen elsewhere) was still just as wrong afterward (a
# different, still-badly-wrong shift, not improved), AND the model didn't
# even follow the explicit instruction to add a per-cell note for each
# "*" it saw -- zero such notes appeared in the output. Consistent with
# this file's established pattern: column-position problems have not
# responded to prompting in any attempt tried so far this session. "*"
# handling on this form remains unsolved; do not re-attempt via prompting
# without a genuinely new hypothesis.

# A "DO NOT LET VALUES BLEED INTO THE ADJACENT ROW" bullet was tried here
# 2026-08-14, targeting a real row-bleed bug found on sample 8.jpeg (a
# dense, 19-column non-ESSA form) -- REVERTED after a real re-run made
# things worse, not better: the bleed didn't stop, it relocated to
# different (still wrong) columns, a previously-clean row (20-20 RNS)
# regressed with a new spurious value, and a new hallucinated cell
# appeared on Fairlady Plain (120:6, nothing there on the real page).
# Cell count went up (63 -> 70) purely from added wrong/spurious values,
# not fixes. Consistent with this file's established pattern for
# column-drift prompt attempts (4/4 failed, one caused a regression) --
# row-bleed looks like the same underlying spatial-grounding limitation
# on a different axis, not an instruction-following gap. Do not re-add
# this bullet without a genuinely new hypothesis, same standard as the
# 105/110 case in CLAUDE.md.

# Separate from the base addendum above because it's built dynamically from
# a real DB query (brandlist_match.known_numeric_sizes()), not a static
# string -- confirmed by direct testing (2026-08-13) that a prose-only
# version of this same instruction ("sizes sometimes go up to 110, 115, or
# 120") was NOT enough: mistral still never reported anything past the last
# PRINTED header on sample 5.jpeg's Fairlady rows even with that sentence in
# the prompt. Giving it the actual, real, complete numeric size vocabulary
# this business's catalog uses -- not an abstract claim -- is the next thing
# to try. Falls back to the same static phrasing (no concrete list) if the
# DB isn't reachable, so this file still runs standalone per its own
# design (see get_client()'s docstring-equivalent comment on --no-brandlist-check).
def _sizes_past_header_bullet(known_sizes: list[int]) -> str:
    if known_sizes:
        sizes_str = ", ".join(str(s) for s in known_sizes)
        catalog_clause = (
            f"This business's REAL, COMPLETE numeric size vocabulary, across its entire product "
            f"catalog, is exactly: {sizes_str}. Any of these can legitimately appear on a row, "
            f"even ones well past this specific form's printed header row."
        )
    else:
        catalog_clause = (
            "This business's real product catalog frequently includes sizes larger than this "
            "form's printed grid shows -- the printed header row commonly stops around 105, but "
            "real catalog sizes for many products go up to 110, 115, or 120."
        )
    return f"""

- THE PRINTED GRID IS A GENERIC TEMPLATE, NOT A HARD BOUNDARY: this form's printed size headers \
are a standard template sized for this business's MOST COMMON products -- but not every product on \
the form actually fits that template, and confirmed by direct testing, you consistently handle this \
the same wrong way: you force the row's data to fit the printed grid regardless, instead of noticing \
the row doesn't really belong to it. Real people filling out this form do one of two specific things \
when a product doesn't fit the printed template -- watch for BOTH, on every single row, before \
assuming the printed headers apply as-is:
  (a) EXTRA COLUMNS PAST THE LAST HEADER: {catalog_clause} When a product's real sizes run larger \
than the printed grid, the writer adds one or more EXTRA handwritten size:quantity pairs to the \
RIGHT of the last printed header (e.g. a handwritten "110" with its own quantity beside or below \
it) -- confirmed missed entirely, every single time, on every real test form checked so far.
  (b) LETTER SIZES INSTEAD OF THE GRID ENTIRELY: when a product's real sizes (often a kids' product) \
don't correspond to the printed numeric grid AT ALL -- not even loosely -- the writer abandons the \
printed headers for that row completely and writes standard clothing letter sizes (S, M, L, XL, \
XXL, etc.) instead, each with its own quantity. Confirmed by direct testing: you consistently \
misread this exact situation as if the row still used the printed numeric headers -- reporting \
plain numbers under the nearest printed columns -- instead of recognizing that this row's letters \
are not numbers at all. If a row's marks don't look like the digits 0-9, or don't align cleanly \
under any printed header, they are very likely letter sizes -- report them as the letters actually \
written (S, M, L, etc.), per the LETTER SIZES rule above, rather than forcing them into the nearest \
numeric column.
For every row: first check whether it plausibly fits the printed grid at all (letters where you'd \
expect digits, or marks past the last printed column, are both signs it does not) before defaulting \
to "assign each mark to its nearest printed header," which is the wrong default for a row like this.

- CODES EMBEDDED IN ITEM NAMES: a product code inside an item name can mix letters and digits \
together, e.g. "K4532" -- confirmed by direct testing, you sometimes drop the leading letter and \
report only the digits (e.g. "K4532" -> "64532"), which is a different, nonexistent code and breaks \
downstream catalog matching entirely. Transcribe an alphanumeric code exactly as written, letters \
and digits both -- never drop or reinterpret a letter within a code as if it were part of a number.

- ON FREE-FORM PAGES WITH NO PRINTED GRID AT ALL: some forms (a personal order notebook, not this \
business's own printed pad) have no printed header row anywhere -- every item writes its own size \
list fresh, by hand, with no shared table to anchor against. On a page like this there is NO implied \
ceiling on which sizes can appear, not even loosely -- do not let a size range that looks "typical" \
(e.g. because most other items on the same page stop around 100) make you stop reading a line early. \
If an item's handwritten size:quantity pairs wrap onto a second line below the first, keep reading \
that second line in full and all the way to its own right edge, even if the resulting largest size \
(e.g. 110) is larger than anything you've seen elsewhere on this specific page."""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def get_client() -> ollama.Client:
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print(
            "Error: OLLAMA_API_KEY is not set. Add it to .env (as a new line -- don't touch the "
            "existing DB/Anthropic credentials there) or export it before running. Get a key from "
            "ollama.com/settings/keys.",
            file=sys.stderr,
        )
        sys.exit(1)
    return ollama.Client(host=OLLAMA_CLOUD_HOST, headers={"Authorization": f"Bearer {api_key}"})


def _strip_json_fences(text: str) -> str:
    """Ollama Cloud's gemma4:31b wraps its JSON output in ```json ... ```
    markdown fences even with format=<schema> set (confirmed by a real
    call, see module docstring) -- Claude's structured outputs and the
    local Ollama pipeline's qwen2.5vl never do this. Strips the fence if
    present; returns the text unchanged otherwise, so a future model/
    response that doesn't need this still parses fine."""
    m = _FENCE_RE.search(text)
    return m.group(1) if m else text


def _normalize_value(val, annotation):
    """One field's raw model output, coerced toward `annotation`'s shape --
    see _normalize_for_schema for why this exists at all."""
    origin = typing.get_origin(annotation)
    if origin is list:
        args = typing.get_args(annotation)
        item_type = args[0] if args else str
        if not isinstance(val, list):
            return []
        out = []
        for v in val:
            try:
                out.append(_normalize_value(v, item_type))
            except Exception:
                continue  # one malformed list entry shouldn't sink the whole field
        return out
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _normalize_for_schema(val if isinstance(val, dict) else {}, annotation)
    if annotation is str:
        if val is None:
            return ""
        return val if isinstance(val, str) else str(val)
    if annotation is float:
        if val is None:
            return 0.0
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0
    if annotation is int:
        if val is None:
            return 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0
    return val


def _normalize_for_schema(data: dict, model_cls: type[BaseModel]) -> dict:
    """Best-effort repair of a dict against `model_cls`'s fields before
    validation -- confirmed necessary (see module docstring): a real
    gemma4:31b response was missing a required field entirely, used null
    for a non-nullable string field, and carried an extra top-level key
    the schema doesn't define. This walks `model_cls.model_fields` (so
    unknown extra keys are silently dropped, never passed through) and
    supplies a type-appropriate default ("" / 0.0 / 0 / [] / a recursively
    normalized nested model) for anything missing or null, rather than
    letting Pydantic raise on the first offending field. Deliberately
    generic (driven by type annotations, not hardcoded per-field) so it
    works unchanged for ExtractedForm, ExtractedItem, QuantityPair,
    QuantityRecount, and RowQuantityReading alike -- all five of
    extract_claude.py's schemas this file reuses."""
    if not isinstance(data, dict):
        data = {}
    out = {}
    for name, info in model_cls.model_fields.items():
        out[name] = _normalize_value(data.get(name), info.annotation)
    return out


# Confirmed 2026-08-14, prompted by the user noticing sample 10.jpeg's
# column drift was unusually severe compared to every ESSA-family form
# tested this session: QuantityPair.size's field description literally
# reads "Size column header exactly as printed on the form (e.g. '45',
# '90')." -- and since this schema (imported unmodified from
# extract_claude.py, shared with the Claude pipeline) is passed as
# format= on EVERY Ollama Cloud call via model_json_schema(), that exact
# ESSA-specific example text (45 is ESSA's own first real header column)
# is baked into every single request this file has ever sent, main or
# recount, regardless of --no-recount. A concrete, previously-untested
# hypothesis: this could be quietly anchoring the model toward expecting
# a 45-105-shaped grid even on forms whose real grid starts somewhere
# else entirely (sample 10.jpeg's real grid starts at 25, with genuinely
# variable per-row start columns). Neutralized here, LOCALLY, rather than
# editing QuantityPair's description in extract_claude.py directly --
# that class is shared with the already-confirmed-working Claude
# pipeline, and this project's own practice is to not risk a shared,
# proven component on an unverified change; a local schema-dict patch
# applied only to this file's own Ollama Cloud calls carries none of
# that risk.
_ESSA_ANCHORED_EXAMPLE = "Size column header exactly as printed on the form (e.g. '45', '90')."
_NEUTRAL_EXAMPLE = "Size column header exactly as printed on the form -- read the real header digits for THIS form, whatever they are; do not assume any particular starting number or range."


def _neutralize_schema_examples(schema: dict) -> dict:
    """Recursively replaces _ESSA_ANCHORED_EXAMPLE with _NEUTRAL_EXAMPLE
    anywhere it appears in a JSON schema dict (top level or nested under
    $defs/properties/items) -- see the comment above for why this exists.
    Generic dict/list walk, not hardcoded to one field's position, so it
    still works if QuantityPair's place in the schema tree ever changes."""
    if isinstance(schema, dict):
        return {
            k: (_NEUTRAL_EXAMPLE if k == "description" and v == _ESSA_ANCHORED_EXAMPLE else _neutralize_schema_examples(v))
            for k, v in schema.items()
        }
    if isinstance(schema, list):
        return [_neutralize_schema_examples(v) for v in schema]
    return schema


def _call_schema(
    client: ollama.Client,
    model: str,
    system_prompt: str,
    user_content: str,
    images: list[bytes],
    schema_model: type[BaseModel],
    max_tokens: int,
) -> tuple[BaseModel | None, str | None, dict]:
    """One Ollama Cloud chat call constrained to `schema_model`'s JSON
    schema, with fence-stripping + normalization (see above) before
    validation. Returns (parsed_or_None, error_or_None, usage) --
    usage always has prompt_eval_count/eval_count/duration_seconds when
    the call itself completed, even if parsing then failed, so a bad
    response's real (billed-against-plan-quota) cost is still logged --
    same rationale as extract_claude.py's "log usage before parsing" fix."""
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": user_content, "images": images})

    t0 = time.perf_counter()
    try:
        resp = client.chat(
            model=model,
            messages=messages,
            format=_neutralize_schema_examples(schema_model.model_json_schema()),
            options={"temperature": 0, "num_predict": max_tokens},
            think=THINK,
        )
    except Exception as exc:
        return None, str(exc), {"prompt_eval_count": None, "eval_count": None, "duration_seconds": time.perf_counter() - t0}

    duration = time.perf_counter() - t0
    usage = {
        "prompt_eval_count": resp.prompt_eval_count,
        "eval_count": resp.eval_count,
        "duration_seconds": duration,
    }

    if resp.done_reason not in ("stop", None):
        return None, f"generation stopped early (done_reason={resp.done_reason})", usage

    text = _strip_json_fences(resp.message.content or "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON from model: {exc}", usage

    normalized = _normalize_for_schema(data, schema_model)
    try:
        parsed = schema_model.model_validate(normalized)
    except ValidationError as exc:
        return None, f"schema validation failed after normalization: {exc}", usage
    return parsed, None, usage


def _log_usage(log_path: Path, image_name: str, model: str, mode: str, usage: dict, error: str | None = None) -> None:
    is_new = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow([
                "timestamp_utc", "image", "model", "mode",
                "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens",
                "estimated_cost_usd", "duration_seconds", "error",
            ])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            image_name, model, mode,
            usage.get("prompt_eval_count") if usage.get("prompt_eval_count") is not None else "",
            "", "",
            usage.get("eval_count") if usage.get("eval_count") is not None else "",
            "",
            f"{usage['duration_seconds']:.3f}" if usage.get("duration_seconds") is not None else "",
            error or "",
        ])


# Plain-text row recount, added 2026-08-14 to replace the schema-constrained
# call above (_recount_one_row, kept below for reference/whole-table fallback
# use) after confirming the JSON-schema recount call fails for EVERY Ollama
# Cloud model tested so far, not just one: qwen3.5:397b (0/14 valid responses,
# 2026-08-13), kimi-k2.6 (deterministic all-zero), and mistral-large-3:675b
# (12/14 invalid-JSON on sample 5.jpeg, then 7/10 invalid-JSON + 2 empty +
# 1 misaligned on sample 8.jpeg -- both 2026-08-14). A targeted test isolated
# WHY: it's not that row-cropping fails to isolate a row (a manually-built
# crop of exactly the two rows in a confirmed row-bleed case, sent with a
# simple plain-text ask instead of the strict schema, correctly separated
# both rows AND fixed the bleed -- see CLAUDE.md's 2026-08-14 entry) -- it's
# that these models can't reliably produce schema-constrained JSON on this
# call. Same idea, same crop, just a plain "Item seen: / Quantities: / Total:"
# text format instead of forced JSON, parsed with a regex (mirrors how the
# ORIGINAL local Ollama pipeline's own VLM fallback, stage_b_row in
# extract_ollama.py, already avoids this exact class of failure).
ROW_RECOUNT_TEXT_PROMPT = """This crop shows a printed size-header row, then one handwritten item row \
below it (a thin sliver of the row above or below may also be visible -- ignore those, read only the \
row matching the item given below).

{label}

Reply in EXACTLY this plain-text format and nothing else -- no explanation, no markdown:
Item seen: <the item/particulars name exactly as written on THIS row>
Quantities: <size:qty, size:qty, ... for every legible marked cell on this row, left to right; leave \
blank if this row has no quantities>
Total: <this row's own printed or circled running total, exactly as written; leave blank if none>"""

_ROW_TEXT_PAIR_RE = re.compile(r"(\d+)\s*:\s*(\d+)")


def _parse_row_text_reading(text: str) -> RowQuantityReading:
    """Parses the plain-text ROW_RECOUNT_TEXT_PROMPT response into a
    RowQuantityReading, so it flows into the existing _apply_recount merge
    logic (imported from extract_claude.py, unmodified) exactly like a
    schema-constrained response would. first_size/last_size are derived
    from the parsed quantities themselves (first/last in reading order)
    rather than asked for separately -- one less thing for the model to
    get right, and they're only used for alignment sanity-checking anyway."""
    item_seen, total_dozen = "", ""
    quantities: list[QuantityPair] = []
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("item seen:"):
            item_seen = line.split(":", 1)[1].strip()
        elif low.startswith("quantities:"):
            rest = line.split(":", 1)[1]
            quantities = [QuantityPair(size=s, quantity=int(q)) for s, q in _ROW_TEXT_PAIR_RE.findall(rest)]
        elif low.startswith("total:"):
            total_dozen = line.split(":", 1)[1].strip()
    first_size = quantities[0].size if quantities else ""
    last_size = quantities[-1].size if quantities else ""
    return RowQuantityReading(item_seen=item_seen, first_size=first_size, last_size=last_size, quantities=quantities, total_dozen=total_dozen)


def _recount_one_row_text(client: ollama.Client, model: str, crop_bytes: bytes, label: str) -> tuple[RowQuantityReading, str | None, dict]:
    """Text-format counterpart to _recount_one_row below -- same crop, same
    per-row parallelization, no format= schema constraint. See the module
    comment above ROW_RECOUNT_TEXT_PROMPT for why this replaced the
    schema-constrained call as the default recount path."""
    content = ROW_RECOUNT_TEXT_PROMPT.format(label=label)
    t0 = time.perf_counter()
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": content, "images": [crop_bytes]}],
            options={"temperature": 0, "num_predict": RECOUNT_ROW_MAX_TOKENS},
            think=THINK,
        )
    except Exception as exc:
        usage = {"prompt_eval_count": None, "eval_count": None, "duration_seconds": time.perf_counter() - t0}
        return RowQuantityReading(**_EMPTY_ROW_READING_KWARGS), str(exc), usage

    usage = {
        "prompt_eval_count": resp.prompt_eval_count,
        "eval_count": resp.eval_count,
        "duration_seconds": time.perf_counter() - t0,
    }
    if resp.done_reason not in ("stop", None):
        return RowQuantityReading(**_EMPTY_ROW_READING_KWARGS), f"generation stopped early (done_reason={resp.done_reason})", usage
    try:
        reading = _parse_row_text_reading(resp.message.content or "")
    except Exception as exc:
        return RowQuantityReading(**_EMPTY_ROW_READING_KWARGS), f"failed to parse text response: {exc}", usage
    return reading, None, usage


def _recount_one_row(client: ollama.Client, model: str, crop_bytes: bytes, label: str) -> tuple[RowQuantityReading, str | None, dict]:
    """Single-row recount call -- the unit of work parallelized across a
    form's rows in _recount_quantities, mirroring extract_claude.py's
    _recount_one_row (ThreadPoolExecutor over one call per row) rather than
    one call bundling every row's crop, for the same reason: a bad/slow row
    shouldn't cost every other row its recount coverage.

    Superseded by _recount_one_row_text above as of 2026-08-14 (see its
    module comment) -- kept here unused-by-default rather than deleted, in
    case a future Ollama Cloud model turns out to comply with schema-
    constrained output where every model tested so far has not."""
    content = f"{label}\n\nRead this row's quantities per the rules in the system prompt."
    parsed, error, usage = _call_schema(client, model, RECOUNT_SYSTEM_PROMPT, content, [crop_bytes], RowQuantityReading, RECOUNT_ROW_MAX_TOKENS)
    if error:
        return RowQuantityReading(**_EMPTY_ROW_READING_KWARGS), error, usage
    return parsed, None, usage


def _exif_corrected_bytes(image_path: Path) -> bytes:
    """Read image_path's raw bytes, applying EXIF orientation correction if
    present. Confirmed necessary 2026-08-17: a user-supplied photo carried
    an EXIF orientation tag (value 8, "rotate 90") while its raw pixel data
    was left untouched -- photo viewers respect that tag and display it
    upright, but neither Ollama Cloud's API nor a plain Image.open() apply
    it, so the raw sideways pixels were what actually got sent/processed.
    On the main VLM call this caused a severe failure (2 of 14 items
    dropped, item names/quantities cross-attached to the wrong rows -- see
    CLAUDE.md's dated section); for this file's CV+OCR path it would be
    just as bad, since grid.py's row detection assumes an upright image.
    A plain photo with no orientation tag (confirmed the common case for
    every untouched sample image in this project's Images/ folder) is
    returned completely unchanged -- this is a no-op for the common case,
    not just for the one image that exposed the bug."""
    original = image_path.read_bytes()
    with Image.open(io.BytesIO(original)) as img:
        if img.getexif().get(0x0112, 1) == 1:  # 0x0112 = 274 = Orientation tag
            return original
        corrected = ImageOps.exif_transpose(img).convert("RGB")
    buf = io.BytesIO()
    corrected.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _prepare_image_exif_safe(image_path: Path) -> tuple[bytes, str]:
    """EXIF-corrected counterpart to extract_claude.py's _prepare_image
    (imported unmodified elsewhere in this file, used only as this
    function's fast path) -- same downscale-only-past-MAX_LONG_EDGE policy,
    but starting from orientation-corrected pixels so the long-edge cap is
    computed against the image's real displayed dimensions, not its raw
    sensor dimensions (they differ for a 90/270-rotated photo). Not folded
    into extract_claude.py itself, per this file's established
    additive-only convention (see MISTRAL_PROMPT_ADDENDUM_BASE's own
    comment for the same reasoning applied elsewhere) -- this only ever
    changes behavior for images _prepare_image already reads wrong."""
    with Image.open(image_path) as img:
        if img.getexif().get(0x0112, 1) == 1:
            return _prepare_image(image_path)
        corrected = ImageOps.exif_transpose(img).convert("RGB")
    if max(corrected.size) <= MAX_LONG_EDGE:
        buf = io.BytesIO()
        corrected.save(buf, format="JPEG", quality=95)
        return buf.getvalue(), "image/jpeg"
    scale = MAX_LONG_EDGE / max(corrected.size)
    new_size = (round(corrected.width * scale), round(corrected.height * scale))
    resized = corrected.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"


def _prepare_image_for_mistral(image_path: Path) -> tuple[bytes, str]:
    """_prepare_image_exif_safe + automated deskew/contrast normalization
    (preprocess_for_vlm.py) -- mistral-only, called from extract_one() only
    when "mistral" is in the model name, per this file's established
    pattern for model-specific behavior. See preprocess_for_vlm.py's module
    docstring for why this exists (a real row-bleed bug on sample 8.jpeg,
    previously only fixable by a manually deskewed image) and CLAUDE.md's
    dated section for whether it's confirmed to actually reproduce that
    fix -- not assumed safe just because it runs without error."""
    image_bytes, media_type = _prepare_image_exif_safe(image_path)
    return preprocess_for_vlm(image_bytes), media_type


# Comfortably under PaddleOCR's own internal max_side_limit (~4000px,
# confirmed via its own "Resized image size ... exceeds max_side_limit of
# 4000" warning text). Every OCR-facing crop in this module is built at
# this width, regardless of what upscale factor row-boundary DETECTION
# needed to succeed -- see _rescale_for_ocr's docstring for why the two
# can't just share one image.
_OCR_TARGET_WIDTH = 3800


def _rescale_for_ocr(original_bytes: bytes, skew: float, target_width: int = _OCR_TARGET_WIDTH) -> Image.Image:
    """Renders original_bytes (native resolution) deskewed by `skew` and
    resized to a controlled, safe width for OCR -- deliberately decoupled
    from whatever scale grid.py's row-boundary detection needed to succeed
    (see iter_row_boundary_candidates_auto, which can go up to 4x).
    Confirmed necessary 2026-08-17: a stitched header+row crop is always
    the FULL image width, so building it from a 4x-upscaled working image
    reliably exceeds PaddleOCR's own ~4000px cap, and PaddleOCR then
    downscales it internally -- but confirmed (via direct debug output on
    a real form) that ratio ends up inconsistent between the once-per-
    candidate standalone header check and the per-row stitched check that
    reuses the same header pixels: most rows found only 12-17 of 18 real
    headers in their own stitched crop despite the identical header pixels
    reading 18/18 in isolation, and ocr_row()'s strict per-row header match
    then discarded the row entirely as a failed header read -- so only 1 of
    14 otherwise-correctly-aligned rows actually produced a quantity
    reading. Rendering every OCR-facing crop from a single, explicitly
    controlled width removes PaddleOCR's own inconsistent internal resize
    from the picture, rather than trying to predict or tune around it."""
    img = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    if skew:
        img = img.rotate(skew, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    if img.width == target_width:
        return img
    scale = target_width / img.width
    return img.resize((target_width, round(img.height * scale)), Image.LANCZOS)


_HYBRID_ROW_LINE_RE = re.compile(r"ROW\s+(\d+)\s*:(.*)", re.IGNORECASE)


def _split_merged_qty_token(text: str, box: list[float]) -> list[tuple[int, float]] | None:
    """Re-segments a single OCR token that's almost certainly two adjacent
    quantity numbers PaddleOCR fused into one detection (e.g. two
    handwritten "30"s next to each other read as one token "3030") into two
    separate (qty, x_center) candidates, instead of the qty > 500 case being
    dropped outright (2026-08-17's documented gap -- "not recovered/split
    here", see CLAUDE.md's hybrid-quantities section).

    Tries every internal split point and keeps it ONLY if EXACTLY ONE split
    point yields two valid (1-500, no spurious leading zero) integers --
    checked at every position rather than just the middle, since a merge of
    two differently-sized numbers (e.g. "5"+"30" -> "530") won't split
    evenly. Requiring uniqueness matters because a token like "1010" splits
    validly at two different points (10|10 and 101|0 -- the latter rejected
    by the leading-zero-on-neither-side check, but "255" would split at both
    2|55 and 25|5 with nothing to prefer one over the other) -- an ambiguous
    split is worse than no value at all, matching the qty > 500 case's own
    existing "leave it out rather than invent a value" principle. On a
    unique split, the box's pixel width is divided proportionally by
    character position to estimate each half's own x-center, so downstream
    row/column assignment (which keys entirely off x-position) still works
    on the recovered values."""
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


def _split_merged_header_token(text: str, box: list[float], size_headers: list[str]) -> list[tuple[str, float]] | None:
    """Re-segments a single OCR token that fuses two or more adjacent SIZE
    HEADER labels into one detection (e.g. "100" and "105" read as one
    "100105" token) into separate (header, x_center) entries -- the same
    merged-detection phenomenon already handled for quantity marks
    (_split_merged_qty_token), confirmed by direct testing 2026-08-19 to
    also hit header labels: on `sample 12-scanned.jpg`, PaddleOCR read the
    adjacent "100"/"105" headers as one token, so neither ever entered
    header_x under an exact `t in size_headers` match -- silently losing
    two whole columns' worth of coordinate ground truth, which then caused
    every real mark meant for those columns to collide with a neighboring
    header (row 0/1's large sum-mismatch flags were traced directly to
    this).

    Unlike the quantity-token splitter (which accepts any numeric split),
    this only accepts a partition into SUBSTRINGS THAT ARE THEMSELVES KNOWN
    HEADERS from size_headers -- a much tighter constraint than "looks like
    two plausible numbers," since headers are a small, known, fixed set
    (confirmed via testing this correctly finds the unique split
    "100"+"105" for "100105" while not being fooled by unrelated digit
    strings). Returns None when the token isn't a clean, UNIQUE partition
    into 2+ known headers -- an ambiguous or no-match token is left alone
    rather than guessed at."""
    headers_set = set(size_headers)
    if not text.isdigit() or text in headers_set:
        return None
    n = len(text)
    partitions: list[list[str]] = []

    def backtrack(start: int, current: list[str]) -> None:
        if start == n:
            if len(current) >= 2:
                partitions.append(list(current))
            return
        for end in range(start + 1, n + 1):
            piece = text[start:end]
            if piece in headers_set:
                current.append(piece)
                backtrack(end, current)
                current.pop()

    backtrack(0, [])
    if len(partitions) != 1:
        return None
    parts = partitions[0]
    x0, x1 = box[0], box[2]
    width = x1 - x0
    result: list[tuple[str, float]] = []
    pos = 0
    for part in parts:
        seg_x0 = x0 + width * (pos / n)
        seg_x1 = x0 + width * ((pos + len(part)) / n)
        result.append((part, (seg_x0 + seg_x1) / 2))
        pos += len(part)
    return result


def _flag_hybrid_total_mismatch(quantities: dict[str, int], printed_total: int | None) -> str | None:
    """Checks a hybrid-pass row's quantities against its own printed running
    total (e.g. "Total Dozen" / "TOTAL") and returns a human-readable flag
    string when they disagree, else None. Flag-only, deliberately NOT an
    auto-correction -- see the note below for why a total-based
    auto-correction (originally suggested in CLAUDE.md's 2026-08-19 "Future
    suggestions" #3) turned out to be a dead end, confirmed by direct
    testing before shipping it, not assumed from the idea's description
    alone:

    Re-assigning a row's reported quantities to different header KEYS while
    keeping the same VALUES can never change their sum -- shifting
    {45:5,50:30,55:30,60:4} to {50:5,55:30,60:30,65:4} still sums to 69
    either way (confirmed directly: sum() before and after are identical).
    So a "try shifting by an offset until the sum matches" search (mirroring
    brandlist_match.detect_column_shift's approach, which works there
    because it's a set-MEMBERSHIP check against catalog sizes, not a sum
    check) can only ever fire when the row's sum ALREADY matches the printed
    total pre-shift -- meaning it can never fire on the actual target case (a
    genuine sum mismatch), regardless of which offset range is searched.
    This exactly contradicts -- and was written without checking against --
    this same file's own much earlier, already-confirmed finding in the
    "Column-shift detection" section (2026-08-06): "a shift relabels values,
    it doesn't change their total." The 2026-08-19 suggestion re-introduced
    a claim the project had already disproven; this flag-only version is
    the corrected form of that suggestion.

    A genuine sum mismatch on this hybrid pass most often means a real value
    went undetected entirely (the already-documented "18% missing, tail
    shifts into the gap" pattern) -- not recoverable by relabeling the
    values that WERE detected, since the fix requires a value that was never
    read in the first place. Surfacing it for human review (matching this
    project's established "flag, don't auto-correct" pattern for
    lower-confidence signals, e.g. brandlist_match's
    sizes_outside_catalog_range) is the honest thing this signal can do."""
    if not printed_total or printed_total <= 0 or not quantities:
        return None
    total = sum(quantities.values())
    if total == printed_total:
        return None
    return f"sums to {total}, but this row's own printed total is {printed_total} -- likely a missed or extra value, not auto-corrected"


def _hybrid_ocr_quantities(client: ollama.Client, model: str, image_path: Path, extracted: ExtractedForm, outdir: Path | None = None) -> tuple[dict[int, dict[str, int]], list[str]]:
    """Reads quantities by giving mistral REAL, OCR-measured coordinates
    (not its own self-report) for headers and quantity marks, then asking
    it to group each mark with its nearest header by x-position -- a
    single OCR pass on the whole image (no per-row crops, no grid.py row-
    boundary detection), so this sidesteps both root causes behind the
    2026-08-17 --cv-quantities revert (PaddleOCR's inconsistent internal
    resize on stitched crops; grid.py's row-boundary detection getting
    confused by an extra letterhead row).

    THE CORE MECHANISM IS CONFIRMED STRONG IN ISOLATION, NOT YET AT THE
    SAME LEVEL END-TO-END -- be precise about which claim is which. In a
    hand-curated test (real OCR coordinates fed in manually, not through
    this function), mistral got 47 of 48 marks exactly right (~98%),
    including full-row exact matches on rows that had been off by 8-9
    header positions under every VLM-only approach tried this session --
    confirming that when GIVEN real coordinates instead of asked to
    generate its own, it reliably does the nearest-neighbor matching
    rather than defaulting to a plausible-looking guess (see CLAUDE.md's
    dated section for the full breakdown). But wiring this into the real
    pipeline (this function) surfaced two additional, real bugs the
    isolated test never exercised, both confirmed and fixed the same day:
    (1) this form's DECOY sub-header row(s) (an age/chest-equivalent
    number, plus separately letter clothing sizes, both sitting directly
    below the real size headers) were initially accepted as candidate
    quantity marks, corrupting row 1 -- fixed by detecting and skipping any
    y-band below the headers whose detections cover most header columns
    (a decoy row's signature; real data is sparse by comparison), not by
    trusting extracted.table_top_frac (confirmed WRONG for this purpose --
    it marks the TOP of the header, not the bottom of the last decoy
    line). (2) row assignment via extracted.items[i]'s own
    row_top_frac/row_bottom_frac (even just as a "nearest center"
    comparison, not a strict range test) still misassigned row 1, because
    that row's own center estimate was itself biased early enough that the
    NEXT row's center was numerically closer -- fixed by clustering
    OCR-detected marks into rows by y-GAP (data-driven, not model-
    reported) and matching clusters to items via the SAME order-preserving
    DP already proven for digit-to-header assignment in ocr_cell_read.py
    (_monotonic_assign), rather than trusting either fraction directly.
    Even after both fixes, a real end-to-end run's per-row completeness
    still fell short of the isolated test's number -- not yet root-caused,
    see CLAUDE.md.

    Returns (quantities_by_index, notes). Only rows where OCR found
    candidate marks AND the model's response parsed cleanly are included;
    every other row keeps the model's own main-call reading, same
    fallback philosophy as this file's other optional passes."""
    size_headers = extracted.size_headers
    n_items = len(extracted.items)
    if n_items == 0 or not size_headers:
        return {}, []
    if not _validate_row_fracs(extracted.items, extracted.table_top_frac, extracted.table_bottom_frac):
        return {}, ["Hybrid OCR+VLM quantity read: skipped -- row_top_frac/row_bottom_frac from the "
                     "main read didn't form a plausible partition."]

    original_bytes = _exif_corrected_bytes(image_path)
    image = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    width, height = image.size
    arr = np.array(image)[:, :, ::-1]
    ocr = get_ocr()
    result = list(ocr.predict(arr))
    if not result:
        return {}, ["Hybrid OCR+VLM quantity read: OCR found no text at all on this image."]
    res = result[0]
    texts, boxes, scores = res["rec_texts"], res["rec_boxes"].tolist(), res["rec_scores"]

    header_x: dict[str, float] = {}
    header_y_max = 0.0
    header_y_min = float("inf")
    for t, b, s in zip(texts, boxes, scores):
        if t in size_headers and s >= 0.5 and t not in header_x:
            header_x[t] = (b[0] + b[2]) / 2
            header_y_max = max(header_y_max, b[3])
            header_y_min = min(header_y_min, b[1])

    # Second pass: recover headers PaddleOCR fused into one token (e.g.
    # "100"+"105" read as one "100105" token) via _split_merged_header_token
    # -- confirmed real and necessary 2026-08-19 (see that function's
    # docstring). Bounded to the header row's own y-band (with a small
    # margin) rather than scanned across the whole page, so this can't
    # accidentally treat two adjacent QUANTITY digits deep in the table as
    # a merged header -- only text sitting where the real headers already
    # are is considered.
    if header_x:
        band_margin = max(header_y_max - header_y_min, 15.0) * 0.5
        for t, b, s in zip(texts, boxes, scores):
            if t in header_x or s < 0.5:
                continue
            yc = (b[1] + b[3]) / 2
            if not (header_y_min - band_margin <= yc <= header_y_max + band_margin):
                continue
            split = _split_merged_header_token(t, b, size_headers)
            if split is None:
                continue
            for h, hx in split:
                if h not in header_x:
                    header_x[h] = hx
                    header_y_max = max(header_y_max, b[3])
                    header_y_min = min(header_y_min, b[1])

    if len(header_x) < len(size_headers) * 0.6:
        return {}, [f"Hybrid OCR+VLM quantity read: only found {len(header_x)}/{len(size_headers)} "
                     f"headers via OCR -- skipped (likely a free-form page with no shared header row)."]

    header_xs_sorted = sorted(header_x.values())
    spacing = (header_xs_sorted[-1] - header_xs_sorted[0]) / max(len(header_xs_sorted) - 1, 1)
    x_floor = header_xs_sorted[0] - spacing * 0.6
    x_ceiling = header_xs_sorted[-1] + spacing * 0.6  # excludes a trailing Total/Rate column

    # y_floor: NOT just header_y_max (the bottom of the matched size_headers
    # row). Confirmed necessary 2026-08-17 via a real end-to-end run on
    # sample 12-scanned.jpg, whose DECOY sub-header row(s) (an age/chest
    # equivalent number, and separately clothing letter sizes, both sitting
    # directly below each real size header) sit below header_y_max and were
    # wrongly accepted as candidate quantity marks, corrupting the first
    # row with values like 14, 16, 18... lifted straight from the decoy
    # row. A first attempt used extracted.table_top_frac as an extra floor,
    # reasoning it was the main call's own judgment of where real data
    # starts past both header lines -- confirmed WRONG by direct
    # measurement: table_top_frac is defined as the TOP of the header (not
    # the bottom of the last decoy sub-header), and per-item row_top_frac
    # was *also* checked and found to sit right at the decoy row's own top
    # edge, not past it -- consistent with this project's already-
    # documented finding elsewhere that these self-reported fractions are
    # unreliable on dense forms. Neither VLM self-report was trustworthy
    # here, so this detects the decoy band directly from the OCR data
    # itself instead: a REAL data row is sparse (a few marks under a few
    # columns), while a decoy sub-header row has near-complete coverage
    # (one token under almost every column, since it labels every header).
    # Scans y-bands below header_y_max; any band whose detections cover at
    # least half the header x-positions is treated as another decoy
    # sub-header line (this form has two stacked on top of each other) and
    # the floor is pushed past it, repeating until a low-coverage (real)
    # band is found or the search window is exhausted.
    def _band_header_coverage(y_lo: float, y_hi: float) -> tuple[int, float]:
        xs_in_band = [(b[0] + b[2]) / 2 for _t, b, sc in zip(texts, boxes, scores) if sc >= 0.5 and y_lo < (b[1] + b[3]) / 2 <= y_hi]
        if not xs_in_band:
            return 0, y_hi
        covered = sum(1 for hx in header_xs_sorted if any(abs(hx - x) <= spacing * 0.4 for x in xs_in_band))
        return covered, max(b[3] for _t, b, sc in zip(texts, boxes, scores) if sc >= 0.5 and y_lo < (b[1] + b[3]) / 2 <= y_hi)

    y_floor_px = header_y_max
    band_height = max(header_y_max - min(b[1] for t, b, s in zip(texts, boxes, scores) if t in header_x), 15.0)
    for _ in range(3):  # at most 3 stacked decoy sub-header lines
        covered, band_bottom = _band_header_coverage(y_floor_px, y_floor_px + band_height * 1.3)
        if covered < len(header_xs_sorted) * 0.5:
            break
        y_floor_px = band_bottom

    candidates: list[tuple[int, float, float]] = []  # (qty, x_center, y_frac)
    for t, b, s in zip(texts, boxes, scores):
        if b[1] <= y_floor_px or s < 0.5:
            continue
        xc = (b[0] + b[2]) / 2
        if not (x_floor <= xc <= x_ceiling):
            continue
        qty = int(t) if t.isdigit() else _try_digit_correct(t)
        if qty is None or qty <= 0:
            continue
        if qty > 500:
            # A single OCR token this large on a per-size quantity cell is
            # almost certainly two adjacent numbers PaddleOCR merged into
            # one detection (confirmed on this same run: "30" + "30"
            # merged into one "3030" token) rather than a real quantity.
            # Try to recover both real values via _split_merged_qty_token
            # (2026-08-19 fix) instead of dropping them outright; only a
            # token with no safe unique split still gets dropped, per this
            # project's "leave it out rather than invent a value" principle.
            split = _split_merged_qty_token(t, b)
            if split is None:
                continue
            y_frac = (b[1] + b[3]) / 2 / height
            for sub_qty, sub_xc in split:
                candidates.append((sub_qty, sub_xc, y_frac))
            continue
        candidates.append((qty, xc, (b[1] + b[3]) / 2 / height))

    if not candidates:
        return {}, ["Hybrid OCR+VLM quantity read: OCR found no quantity-shaped marks in the table area."]

    # Cluster candidates into rows by y-GAP (data-driven), then match each
    # cluster to its nearest item by row_top_frac/row_bottom_frac midpoint
    # -- NOT by testing whether a candidate falls inside an item's claimed
    # [row_top_frac, row_bottom_frac] range directly. Confirmed necessary
    # 2026-08-17: that direct-containment approach silently misassigned
    # every row by one, because item 0's own row_bottom_frac (0.5, i.e.
    # 313px) fell just BEFORE its real data actually starts (316px) -- the
    # same self-reported-fraction unreliability already documented
    # elsewhere in this project, now confirmed to affect row boundaries
    # too, not just row-crop building. Using the fractions only as a
    # coarse "which item is this cluster nearest to" comparison (not a
    # strict boundary test) is far more tolerant of that kind of small,
    # systematic error.
    sorted_candidates = sorted(candidates, key=lambda c: c[2])
    y_clusters: list[list[tuple[int, float, float]]] = []
    gap_threshold_frac = 20.0 / height
    for cand in sorted_candidates:
        if y_clusters and cand[2] - y_clusters[-1][-1][2] <= gap_threshold_frac:
            y_clusters[-1].append(cand)
        else:
            y_clusters.append([cand])

    # Match clusters to items via the SAME order-preserving DP already
    # proven for digit-to-header assignment in ocr_cell_read.py
    # (_monotonic_assign), rather than independent nearest-center matching
    # per cluster. Confirmed necessary 2026-08-17: independent nearest-
    # center matching still got row 1 wrong even after fixing the decoy-
    # row contamination above -- item 0's own row_top_frac/row_bottom_frac
    # center was itself biased early enough that item 1's center was
    # numerically CLOSER to row 1's real data cluster than item 0's own
    # center was, so the naive per-cluster nearest match picked the wrong
    # item. A joint, order-preserving assignment (this DP's whole point,
    # per its own docstring) is far less likely to let one biased estimate
    # pull an assignment away from its true position independent of every
    # other row's evidence.
    item_centers = [(it.row_top_frac + it.row_bottom_frac) / 2 for it in extracted.items]
    row_candidates: dict[int, list[tuple[int, float]]] = {}
    if 0 < len(y_clusters) <= len(item_centers):
        cluster_ys = [sum(c[2] for c in cluster) / len(cluster) for cluster in y_clusters]
        assignment, _avg_cost = _monotonic_assign(cluster_ys, item_centers)
        for cluster_idx, item_idx in assignment.items():
            row_candidates[item_idx] = [(c[0], c[1]) for c in y_clusters[cluster_idx]]

    if not row_candidates:
        return {}, ["Hybrid OCR+VLM quantity read: no OCR-detected marks matched to any row."]

    # Debug artifact (2026-08-19) so the still-open "missing values" gap
    # (isolated-test 98% vs. real end-to-end runs falling short, per
    # CLAUDE.md) can be root-caused directly against real per-row candidate
    # data instead of re-guessing from output alone -- records every OCR
    # detection considered in the table area (before row-clustering) plus
    # the final per-row candidate list actually sent to the model, so a
    # specific missing mark can be traced to "OCR never detected it",
    # "dropped by the score/x/y filters above", or "detected and sent, but
    # the model's response for that row just didn't include it".
    if outdir is not None:
        try:
            debug = {
                "headers": {h: x for h, x in header_x.items()},
                "y_floor_px": y_floor_px,
                "x_floor": x_floor,
                "x_ceiling": x_ceiling,
                "all_candidates": [{"qty": q, "x": x, "y_frac": yf} for q, x, yf in candidates],
                "row_candidates": {
                    str(i): {"item": extracted.items[i].item, "marks": [{"qty": q, "x": x} for q, x in marks]}
                    for i, marks in row_candidates.items()
                },
            }
            (outdir / f"{image_path.stem}.hybrid_debug.json").write_text(json.dumps(debug, indent=2, ensure_ascii=False))
        except Exception:
            pass  # debug-only, never let this block the real extraction

    headers_text = "\n".join(f'  "{h}" at x={x:.1f}' for h, x in sorted(header_x.items(), key=lambda kv: kv[1]))
    rows_text = ""
    for i, marks in row_candidates.items():
        marks_text = ", ".join(f'"{q}" at x={x:.1f}' for q, x in marks)
        rows_text += f"\nROW {i} ({extracted.items[i].item}): {marks_text}"

    system_prompt = f"""You are given an order form image, plus REAL, ALREADY-MEASURED pixel coordinates \
(from OCR, not your own estimate) for every size header and every handwritten quantity mark on the \
table. These coordinates are accurate ground truth -- trust them over your own visual impression of \
where a mark "should" belong. For each quantity mark, GROUP it with whichever header's x-coordinate \
is numerically closest to that mark's own x-coordinate -- this is arithmetic on the numbers given, \
not a fresh guess.

SIZE HEADERS (text, x-position in pixels, image is {width}px wide):
{headers_text}

ROW QUANTITY MARKS (text, x-position in pixels), one row per line:
{rows_text}

Reply in EXACTLY this plain-text format, one line per row, and nothing else -- no explanation, no markdown:
ROW <n>: <size>:<quantity>, <size>:<quantity>, ...
using the header text nearest each mark's given x-position."""

    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Group the quantities as instructed.", "images": [original_bytes]}],
            options={"temperature": 0, "num_predict": 4000},
            think=THINK,
        )
    except Exception as exc:
        return {}, [f"Hybrid OCR+VLM quantity read failed ({exc}) -- kept the model's own quantities for every row."]

    quantities_by_index: dict[int, dict[str, int]] = {}
    for line in resp.message.content.splitlines():
        m = _HYBRID_ROW_LINE_RE.match(line.strip())
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in row_candidates:
            continue
        pairs = _ROW_TEXT_PAIR_RE.findall(m.group(2))
        if pairs:
            quantities_by_index[idx] = {size: int(qty) for size, qty in pairs}

    # Total-checksum flagging (2026-08-19) -- see _flag_hybrid_total_mismatch's
    # own docstring for why this flags rather than auto-corrects: a
    # shift-based auto-correction against the printed total was tried first
    # and confirmed mathematically incapable of ever firing (relabeling a
    # row's header keys can't change its sum), so this only surfaces a
    # mismatch for human review instead.
    mismatch_notes: list[str] = []
    for idx, qty in quantities_by_index.items():
        printed_total = _parse_int_or_none(extracted.items[idx].row_total)
        flag = _flag_hybrid_total_mismatch(qty, printed_total)
        if flag is not None:
            mismatch_notes.append(f"Hybrid OCR+VLM: row {idx} ({extracted.items[idx].item}) {flag}.")

    used = len(quantities_by_index)
    note = (f"Hybrid OCR+VLM quantity read: used for {used}/{len(row_candidates)} rows with OCR-detected "
            f"marks ({n_items - len(row_candidates)} row(s) had no OCR marks in range and kept the "
            f"model's own reading).")
    return quantities_by_index, [note] + mismatch_notes


def _headers_found_lenient(header_crop_bytes: bytes, size_headers: list[str], score_threshold: float = 0.5) -> int:
    """Substring-based counterpart to ocr_cell_read.count_headers_found,
    used only by _select_cv_row_boundaries below. count_headers_found
    requires each header to be its OWN separately-recognized OCR token --
    confirmed by direct testing this fails outright on sample 8.jpeg: its
    header row's numbers sit close enough together that PaddleOCR reads the
    whole row as one merged blob ('354045 50 55 6065 7075 7880859095100105110')
    instead of separate tokens, so count_headers_found scored a genuinely
    correct candidate (visually confirmed against the photo) as 0/19.
    Checking substring containment against the concatenation of every
    recognized text in the crop is strictly more permissive than an exact
    token match -- anything that passes count_headers_found also passes
    this -- so it's a safe, broader replacement scoped to this new
    integration, without touching the shared function the already-proven
    local Ollama pipeline depends on."""
    image = Image.open(io.BytesIO(header_crop_bytes)).convert("RGB")
    arr = np.array(image)[:, :, ::-1]
    ocr = get_ocr()
    result = list(ocr.predict(arr))
    if not result:
        return 0
    res = result[0]
    blob = " ".join(t for t, s in zip(res["rec_texts"], res["rec_scores"]) if s >= score_threshold)
    return sum(1 for h in size_headers if h in blob)


def _select_cv_row_boundaries(image_bytes: bytes, size_headers: list[str], expected_items: list[str]):
    """CV-based alternative to trusting the main call's self-reported
    row_top_frac/row_bottom_frac for building recount row-crops -- added
    2026-08-14 after confirming those self-reported fractions are
    unreliable on dense forms (a suspiciously uniform even split, not real
    measurements, on sample 8.jpeg -- see CLAUDE.md), which caused a real
    regression: a misaligned crop's data got silently accepted via the
    catalog tiebreak and marked "resolved" even though it was actually a
    neighboring row's data.

    Ports the candidate-selection loop already proven in extract_ollama.py's
    extract_one() (grid.py's iter_row_boundary_candidates -> validate each
    -> keep the best-scoring candidate, require >=70% of headers found and
    >=50% of rows aligned to accept at all) -- reused via direct import
    (build_header_crop, build_row_crop from extract_ollama.py; grid.py and
    ocr_cell_read.py directly) rather than reimplemented, so a future fix
    to that logic doesn't have to be duplicated here. One piece NOT reused
    as-is: the header-presence check uses _headers_found_lenient (below)
    instead of ocr_cell_read.count_headers_found, since the latter's exact-
    token-match requirement failed on this form -- see that function's
    docstring.

    Returns (image, boundaries, header_crop) for the best candidate, or
    None if no candidate clears the acceptance bar -- callers should fall
    back to the existing row_top_frac/row_bottom_frac-based crops in that
    case (e.g. a free-form page with no ruled grid at all, which this
    function is not expected to find anything on)."""
    n_items = len(expected_items)
    if n_items == 0:
        return None
    best_good_count = -1
    best_candidate = None
    # iter_row_boundary_candidates_auto (not the plain iterator) -- confirmed
    # 2026-08-17 that a low-resolution or tightly-cropped image can return
    # ZERO row-boundary candidates outright at native resolution (see
    # grid.py's docstring); this was silently leaving _select_cv_row_boundaries
    # returning None on such images, falling back to the less reliable
    # self-reported row_top_frac/row_bottom_frac this function exists to
    # avoid trusting in the first place.
    for cand_boundaries, cand_skew, _working_bytes, _scale in iter_row_boundary_candidates_auto(image_bytes, n_items):
        # See _rescale_for_ocr's docstring -- OCR-facing crops always come
        # from a controlled, safe-width rendering, never directly from
        # _working_bytes (whatever scale detection needed, up to 4x).
        candidate_image = _rescale_for_ocr(image_bytes, cand_skew)
        candidate_header_crop, header_crop_bytes = build_header_crop(candidate_image, cand_boundaries)
        headers_found = _headers_found_lenient(header_crop_bytes, size_headers)
        if size_headers and headers_found < len(size_headers) * 0.7:
            continue

        bad_count = 0
        for idx in range(n_items):
            row_bytes, header_h_px = build_row_crop(candidate_image, cand_boundaries, idx, candidate_header_crop)
            _, _, item_name_ocr, _ = ocr_row(row_bytes, header_h_px, size_headers, score_threshold=0.5)
            if not item_alignment_ok(item_name_ocr, expected_items[idx]):
                bad_count += 1

        good_count = n_items - bad_count
        if good_count < n_items * 0.5:
            continue
        if good_count > best_good_count:
            best_good_count = good_count
            best_candidate = (candidate_image, cand_boundaries, candidate_header_crop)
        if good_count == n_items:
            break
    return best_candidate


def _recount_quantities(
    client: ollama.Client,
    model: str,
    image_path: Path,
    extracted: ExtractedForm,
    usage_log: Path,
    brandlist_available: bool = False,
) -> tuple[list[str], list[dict]]:
    """Second pass: re-reads only the size/quantity grid, mirroring
    extract_claude.py's _recount_quantities_live -- same per-row-crop
    strategy (falling back to a single whole-table crop when per-item
    row_top_frac/row_bottom_frac don't validate), same N-parallel-calls
    design, same _apply_recount merge (imported, not reimplemented). Only
    the actual model-call mechanics differ (Ollama chat() instead of
    Anthropic messages())."""
    top, bottom = extracted.table_top_frac, extracted.table_bottom_frac
    if not (0.0 <= top < bottom <= 1.0):
        no_recount_flags = [{"status": "no_recount", "note": "table_top_frac/table_bottom_frac from the main read looked implausible"} for _ in extracted.items]
        return [f"Row-crop quantity recount skipped: table_top_frac/table_bottom_frac from the main read looked implausible ({top}, {bottom})."], no_recount_flags

    original_bytes = _exif_corrected_bytes(image_path)
    headers_line = (
        f"This form's real size headers, left to right, are exactly: {', '.join(extracted.size_headers)} -- "
        f"if a header crop shows a second, smaller line of numbers below these, that second line is NOT one "
        f"of these headers and must be ignored, per the system prompt."
    ) if extracted.size_headers else ""

    # Try CV-detected row boundaries (grid.py, validated via ocr_cell_read.py
    # -- see _select_cv_row_boundaries) before falling back to the main
    # call's own self-reported row_top_frac/row_bottom_frac. Added
    # 2026-08-14: those self-reported fractions were confirmed unreliable
    # on a dense form (an even split, not real measurements), which caused
    # a real regression (a misaligned crop's data silently accepted as
    # "resolved"). CV boundaries are real ruled-line measurements, not a
    # model guess, and confirmed by direct visual check to correctly find
    # sample 8.jpeg's rows where the model's own estimate did not.
    cv_candidate = _select_cv_row_boundaries(original_bytes, extracted.size_headers, [it.item for it in extracted.items])
    if cv_candidate is not None:
        cv_image, cv_boundaries, cv_header_crop = cv_candidate
        row_crops = [build_row_crop(cv_image, cv_boundaries, i, cv_header_crop) for i in range(len(extracted.items))]
        labels = [
            f"Row {i + 1}: {it.item}" + (f" (Style: {it.type})" if it.type else "") + (f"\n{headers_line}" if headers_line else "")
            for i, it in enumerate(extracted.items)
        ]
        crop_source_note = "CV-detected row boundaries (grid.py), not the main call's self-reported row_top_frac/row_bottom_frac."
    else:
        per_row_mode = _validate_row_fracs(extracted.items, top, bottom)
        row_crops = None
        crop_source_note = None

    if cv_candidate is None and not per_row_mode:
        crop_bytes, _media_type = _prepare_table_crop(original_bytes, top, bottom)
        item_list = "\n".join(
            f"{i + 1}. {it.item}" + (f" (Style: {it.type})" if it.type else "")
            for i, it in enumerate(extracted.items)
        )
        label = f"The item rows, top to bottom, are:\n{item_list}" + (f"\n{headers_line}" if headers_line else "")
        parsed, error, usage = _call_schema(client, model, RECOUNT_SYSTEM_PROMPT, label, [crop_bytes], QuantityRecount, RECOUNT_MAX_TOKENS)
        print(f"  usage (recount, whole-table): prompt={usage.get('prompt_eval_count')} eval={usage.get('eval_count')} {usage.get('duration_seconds', 0):.1f}s" + (f" ERROR: {error}" if error else ""))
        _log_usage(usage_log, image_path.name, model, "recount-whole-table", usage, error)
        if error:
            flags = [{"status": "no_recount", "note": f"whole-table recount call failed: {error}"} for _ in extracted.items]
            return [f"Row-crop quantity recount failed ({error}) -- kept the original full-page quantities for every row."], flags
        notes, flags = _apply_recount(extracted, parsed, brandlist_available)
        notes.append("Row-crop quantity recount used the whole-table fallback crop (per-item row_top_frac/row_bottom_frac from the main call didn't validate) -- less isolation between rows than the intended per-row crops.")
        return notes, flags

    if row_crops is None:  # cv_candidate was None but per_row_mode validated -- the pre-2026-08-14 fallback path
        row_crops = _build_row_crops(original_bytes, top, extracted.items)
        labels = [
            f"Row {i + 1}: {it.item}" + (f" (Style: {it.type})" if it.type else "") + (f"\n{headers_line}" if headers_line else "")
            for i, it in enumerate(extracted.items)
        ]
        crop_source_note = "the main call's own self-reported row_top_frac/row_bottom_frac (CV-detected boundaries were not found for this form)."

    readings: list = [None] * len(extracted.items)
    errors: list = [None] * len(extracted.items)
    usages: list[dict] = []
    max_workers = min(RECOUNT_MAX_WORKERS, len(row_crops))
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_recount_one_row_text, client, model, crop[0], label): i
            for i, (crop, label) in enumerate(zip(row_crops, labels))
        }
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            reading, err, usage = fut.result()
            readings[i] = reading
            errors[i] = err
            usages.append(usage)
    wall_time = time.perf_counter() - t0

    total_usage = {
        "prompt_eval_count": sum(u["prompt_eval_count"] for u in usages if u.get("prompt_eval_count") is not None) or None,
        "eval_count": sum(u["eval_count"] for u in usages if u.get("eval_count") is not None) or None,
        "duration_seconds": wall_time,  # true parallel wall time, not summed per-call time
    }
    print(f"  usage (recount, per-row x{len(readings)}): prompt={total_usage['prompt_eval_count']} eval={total_usage['eval_count']} {wall_time:.1f}s wall (parallel, {max_workers} workers)")
    _log_usage(usage_log, image_path.name, model, f"recount-per-row-x{len(readings)}", total_usage, None)

    recount = QuantityRecount(rows=readings)
    notes, flags = _apply_recount(extracted, recount, brandlist_available)
    notes.append(f"Row-crop quantity recount used {crop_source_note}")
    for i, err in enumerate(errors):
        if err:
            expected = extracted.items[i].item
            note = f"Row-crop recount call for '{expected}' failed ({err}) -- kept the original full-page reading for this row."
            notes.append(note)
            flags[i] = {"status": "unverified", "sizes": [qp.size for qp in extracted.items[i].quantities], "note": note}
    return notes, flags


MAIN_CALL_MAX_RETRIES = 1  # extra attempts beyond the first, only on the zero-quantities flake below


def extract_one(client: ollama.Client, model: str, image_path: Path, outdir: Path, usage_log: Path, system_prompt: str, brandlist_available: bool = False, do_recount: bool = True, do_preprocess: bool = False, do_hybrid: bool = False) -> OrderForm:
    t_image_start = time.perf_counter()
    # mistral-only, and opt-in (do_preprocess), NOT automatic on every
    # mistral call -- automated deskew + contrast normalization (see
    # preprocess_for_vlm.py) confirmed 2026-08-17 to fix its target case
    # (a real row-bleed bug on sample 8.jpeg that prompting alone made
    # worse) but ALSO confirmed, via direct A/B and regression testing the
    # same day, to cause real damage elsewhere: sample 7.jpeg (previously
    # 5/7 rows exact) regressed on nearly every row -- one row gained a
    # spurious extra value, another (previously an independently-confirmed
    # exact match) picked up both a column shift and a new digit misread,
    # and one row came back severely wrong. sample 5.jpeg was a rough wash
    # (some rows better, some worse, no net gain). Same pattern already
    # seen once this session (the sample 12 column-shift bullets that
    # regressed sample 5) -- a fix earning its target case back does not
    # mean it's safe as a blanket default. See CLAUDE.md's dated section
    # for the full real numbers.
    if "mistral" in model.lower() and do_preprocess:
        image_bytes, _media_type = _prepare_image_for_mistral(image_path)
    else:
        image_bytes, _media_type = _prepare_image_exif_safe(image_path)

    # mistral-only: date_present-gated schema (see MistralExtractedForm's
    # docstring) instead of the shared ExtractedForm -- confirmed necessary
    # 2026-08-17, since plain prompt wording alone (tried 3 separate times,
    # including with --think) never stopped mistral fabricating a date on a
    # page with none present. Converted back to a plain ExtractedForm right
    # after parsing, so every line below this is unaware the schema ever
    # changed.
    main_schema_cls = MistralExtractedForm if "mistral" in model.lower() else ExtractedForm

    extracted = None
    error = None
    for attempt in range(MAIN_CALL_MAX_RETRIES + 1):
        parsed, error, usage = _call_schema(client, model, system_prompt, USER_PROMPT, [image_bytes], main_schema_cls, MAX_TOKENS)
        print(f"  usage (main, attempt {attempt + 1}): prompt={usage.get('prompt_eval_count')} eval={usage.get('eval_count')} {usage.get('duration_seconds', 0):.1f}s" + (f" ERROR: {error}" if error else ""))
        _log_usage(usage_log, image_path.name, model, f"main-attempt{attempt + 1}", usage, error)
        if error:
            continue
        if main_schema_cls is MistralExtractedForm:
            parsed = _mistral_form_to_extracted_form(parsed)
        n_pairs = sum(len(it.quantities) for it in parsed.items)
        total_value = sum(qp.quantity for it in parsed.items for qp in it.quantities)
        if parsed.items and total_value == 0:
            # Confirmed real flake (2026-08-13, see CLAUDE.md), two different shapes seen:
            # (a) quantities: [] entirely, or (b) size headers present but every quantity
            # literally 0 (e.g. {"size": "80", "quantity": 0} for every cell) -- checking
            # len(quantities) alone misses shape (b), since the pairs exist, just with a
            # 0 value in every one. Roughly 1-in-5 calls on this model. Retrying is cheap
            # (one more ~45s call) and resolved it every time observed in testing.
            print(f"  main call returned {len(parsed.items)} items, {n_pairs} quantity pairs, but every value is 0 -- retrying (known flake)")
            extracted = parsed  # keep as a fallback in case every retry also comes back empty
            continue
        extracted = parsed
        error = None
        break

    if extracted is None:
        raise RuntimeError(f"Main extraction call failed after {MAIN_CALL_MAX_RETRIES + 1} attempt(s): {error}")

    if do_hybrid and extracted.items:
        try:
            hybrid_quantities, hybrid_notes = _hybrid_ocr_quantities(client, model, image_path, extracted, outdir)
            for i, qty in hybrid_quantities.items():
                extracted.items[i].quantities = [QuantityPair(size=size, quantity=q) for size, q in qty.items()]
            extracted.notes.extend(hybrid_notes)
        except Exception as exc:
            extracted.notes.append(f"Hybrid OCR+VLM quantity read failed ({exc}) -- kept the model's own quantities for every row.")

    recount_flags: list[dict] = []
    if do_recount and extracted.items:
        try:
            notes, recount_flags = _recount_quantities(client, model, image_path, extracted, usage_log, brandlist_available)
            extracted.notes.extend(notes)
        except Exception as exc:
            extracted.notes.append(f"Row-crop quantity recount failed ({exc}) -- kept the original full-page quantities for every row.")
            recount_flags = [{"status": "no_recount", "note": f"recount pass raised an exception: {exc}"} for _ in extracted.items]
    elif not do_recount:
        extracted.notes.append("Row-crop quantity recount skipped (--no-recount).")

    _write_debug_artifacts(outdir, image_path.stem, extracted, recount_flags)
    total_duration = time.perf_counter() - t_image_start
    _log_usage(usage_log, image_path.name, model, "total", {"prompt_eval_count": None, "eval_count": None, "duration_seconds": total_duration})
    print(f"  total wall time for {image_path.name}: {total_duration:.1f}s")
    return _to_order_form(extracted, image_path.name)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Extract structured data from order form photos via Ollama Cloud (comparison test against extract_claude.py -- see module docstring).")
    parser.add_argument("input", help="Path to a single image, or a folder of images. Quote paths containing spaces.")
    parser.add_argument("--outdir", default="extracted_ollama_cloud", help="Output directory (default: ./extracted_ollama_cloud)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama Cloud model tag (default: {DEFAULT_MODEL} -- see module docstring for why qwen3-vl:235b-cloud, the originally-targeted model, isn't the default)")
    parser.add_argument("--usage-log", default="usage_log_ollama_cloud.csv", help="CSV file every call's token usage/timing is appended to (default: ./usage_log_ollama_cloud.csv, kept separate from extract_claude.py's usage_log.csv)")
    parser.add_argument("--no-brandlist-check", action="store_true", help="Skip the local (free, no API cost) cross-check against the brandlist product catalog.")
    parser.add_argument("--no-recount", action="store_true", help="Skip the per-row recount pass. Confirmed 2026-08-13: on qwen3.5:397b the recount call failed on every row in every trial (truncation or invalid JSON), so it added ~30-90s of latency with zero corroboration -- skipping it is how the under-2.5-min timing was achieved. The 2026-08-14 switch to a plain-text row format (see ROW_RECOUNT_TEXT_PROMPT) fixed that specific JSON-compliance failure, but a real, separate regression was found the same day on a dense 19-column form (sample 8.jpeg): a misaligned row-crop's data got silently accepted via the catalog tiebreak and marked 'resolved' (i.e. trustworthy) when it was actually a neighboring row's data -- worse than the original bleed error, since it reads as high-confidence. Recommend keeping --no-recount as the default choice until row-crop alignment reliability (row_top_frac/row_bottom_frac from the main call) is independently improved, not just the output format.")
    parser.add_argument("--think", action="store_true", help="Leave the model's internal 'thinking' mode on (Ollama Cloud default) instead of forcing it off. Confirmed 2026-08-13: the right setting is model-specific -- qwen3.5:397b is faster AND accurate with thinking off, but kimi-k2.6 returns deterministically all-zero quantities with thinking off and needs it on to produce a real reading. Try --no-think first (this file's default); if a model comes back schema-valid but all-zero, retry with --think before concluding the model can't do the task.")
    parser.add_argument("--no-think", dest="think", action="store_false", help="Force thinking off (default behavior already -- explicit flag for clarity/scripting).")
    parser.add_argument("--preprocess-image", action="store_true", help="Apply automated deskew + CLAHE contrast normalization (preprocess_for_vlm.py) before sending the image to mistral. Confirmed 2026-08-17 to fix a real row-bleed bug on sample 8.jpeg (values from one row contaminating the row below it), but ALSO confirmed via direct A/B and regression testing the same day to damage other forms -- sample 7.jpeg regressed on nearly every row (a previously exact-match row picked up both a shift and a new digit error), sample 5.jpeg was a wash. Off by default for exactly that reason -- turn this on only for a specific image you know has a row-bleed problem, not as a general-purpose quality improvement. See CLAUDE.md's dated section for the full numbers.")
    parser.add_argument("--hybrid-quantities", action="store_true", help="Re-read quantities by giving mistral REAL OCR-measured coordinates (not its own self-report) for headers and marks, then asking it to group each mark with its nearest header by x-position -- one whole-image OCR pass, no per-row crops. The core mechanism is confirmed strong in isolation (47 of 48 hand-fed marks exactly correct on sample 12-scanned.jpg), but full pipeline integration surfaced real bugs (a decoy sub-header row contaminating candidates; unreliable row_top_frac/row_bottom_frac misassigning rows) -- two are fixed, but end-to-end completeness on this same form still falls short of the isolated result. Experimental, off by default -- see CLAUDE.md's dated section for the honest current state, not just the best-case number.")
    parser.set_defaults(think=None)
    args = parser.parse_args()

    global THINK
    if args.think is not None:
        THINK = args.think

    client = get_client()

    style_codes: list[str] = []
    known_sizes: list[int] = []
    brandlist_available = False
    if not args.no_brandlist_check:
        try:
            style_codes = brandlist_match.known_style_codes()
            known_sizes = brandlist_match.known_numeric_sizes()
            brandlist_available = True
        except Exception as exc:
            print(f"Warning: couldn't reach the brandlist DB ({exc}) -- using the built-in style-code list and skipping the item/size cross-check for this run.", file=sys.stderr)
    system_prompt = build_system_prompt(style_codes)
    if "mistral" in args.model.lower():
        system_prompt += MISTRAL_PROMPT_ADDENDUM_BASE + _sizes_past_header_bullet(known_sizes)

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

    usage_log = Path(args.usage_log)
    forms: dict[str, OrderForm] = {}

    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Extracting {img_path.name} ...", flush=True)
        try:
            forms[img_path.stem] = extract_one(client, args.model, img_path, outdir, usage_log, system_prompt, brandlist_available, do_recount=not args.no_recount, do_preprocess=args.preprocess_image, do_hybrid=args.hybrid_quantities)
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
        print(f"\n{usage_log}: {len(rows)} calls logged across every run so far.")


if __name__ == "__main__":
    main()
