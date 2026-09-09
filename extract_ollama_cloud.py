#!/usr/bin/env python3
"""
extract_ollama_cloud.py -- production pipeline: Ollama Cloud + mistral-large-3.

Reads structured order-form data from a photo via a hosted vision model on
Ollama Cloud. This is the pipeline this project runs in production. It
started as a comparison test against extract_claude.py (the Claude API
pipeline, kept in the repo only because this file imports its schemas/
prompts/helpers -- see below) and became the production choice once
mistral-large-3:675b was shown to match Claude's accuracy on this project's
test forms at a fraction of the cost. Full model-comparison history (why
gemma4:31b/qwen3.5:397b/kimi/glm were tried and set aside) lives in
HISTORY.md, not repeated here.

This file imports extract_claude.py's already-tuned Pydantic schemas
(ExtractedForm, ExtractedItem, QuantityPair, QuantityRecount,
RowQuantityReading) and prompt text (SYSTEM_PROMPT_TEMPLATE via
build_system_prompt, RECOUNT_SYSTEM_PROMPT) verbatim, and reuses its
crop/validate/merge helpers (_prepare_image, _prepare_table_crop,
_build_row_crops, _validate_row_fracs, _apply_recount, _to_order_form,
_write_debug_artifacts, flatten_for_review) rather than reimplementing that
logic -- extract_claude.py is a required dependency of this file, even
though its own Claude API calls are not part of the production path. Only
the model-call sites differ: this file talks to Ollama's chat() API instead
of Anthropic's messages API.

DEFAULT MODEL: mistral-large-3:675b. Confirmed via extensive real-run
testing (see HISTORY.md) to be the most reliable Ollama Cloud vision model
for this task -- fast (~30-70s/image), zero observed flakes across dozens
of runs, and its remaining accuracy gaps (column-position drift, missed
overflow columns, row misattribution on dense tables) are corrected by the
--hybrid-quantities pass below rather than needing a different model.

STRUCTURED-OUTPUT RELIABILITY -- confirmed by direct testing, a real
difference from both extract_claude.py's Claude calls and the local Ollama
pipeline's qwen2.5vl calls (extract_ollama.py): passing format=<json
schema> to Ollama Cloud models does NOT always strictly constrain output the
way Anthropic's structured outputs or local qwen2.5vl's grammar constraint
do (confirmed on gemma4:31b, an earlier default -- see HISTORY.md). A
response can come back wrapped in ```json ... ``` markdown fences, missing
a required field, carrying an extra undeclared key, or using JSON null for
a non-nullable string field. _strip_json_fences() and _normalize_for_schema()
below exist specifically to repair this before validation runs, so the
same field validators/model validators ExtractedForm and friends already
have (ditto forward-fill, trailing style-code split, order_date cleanup)
still get a fair shot at running on real field values instead of erroring
out first on a missing key or a stray null. mistral-large-3:675b (the
production default) has not needed this repair path in practice, but it's
kept as a safety net for any model.

Usage:
    python3 extract_ollama_cloud.py "Images/sample 5.jpeg"
    python3 extract_ollama_cloud.py Images/ --outdir extracted_ollama_cloud
    python3 extract_ollama_cloud.py Images/ --model mistral-large-3:675b

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
    <name>.party_check.json    -- buyer-name cross-check against the DB (only written when a match is found)
    <name>.hybrid_debug.json   -- OCR candidate/header data behind --hybrid-quantities (only when it runs)
    review.csv                 -- flattened, one row per (item, size)

Every call (main or per-row/whole-table recount) appends one row to a
persistent usage log, kept entirely separate from extract_claude.py's own
usage_log.csv so this pipeline's numbers never mix with the Claude
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
import base64
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

import httpx
import numpy as np
import ollama
from dotenv import load_dotenv
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

import brandlist_match
import template_learning
from chandra_parser import CHANDRA_MODEL  # constant only -- call_chandra/parse_chandra_output are imported lazily in extract_one(), see there
from grid import iter_row_boundary_candidates_auto
from ocr_cell_read import get_ocr, item_alignment_ok, ocr_row, _try_digit_correct, _monotonic_assign, _split_merged_qty_token
from preprocess_for_vlm import preprocess_for_vlm
from extract_ollama import build_header_crop, build_row_crop
from schema import OrderForm
from extract_claude import (
    IMAGE_EXTENSIONS,
    KNOWN_STYLE_CODES,
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
    _TRAILING_CODE_RE,
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

# Production default. See module docstring / HISTORY.md for the model
# comparison this was chosen from.
DEFAULT_MODEL = "mistral-large-3:675b"
OLLAMA_CLOUD_HOST = "https://ollama.com"

# Analogous to extract_claude.py's MAX_TOKENS/RECOUNT_ROW_MAX_TOKENS, passed
# as options.num_predict. mistral-large-3:675b's real per-form output on
# this project's forms runs roughly 1500-4000 output tokens (see
# usage_log_ollama_cloud.csv) -- kept at 32000 anyway as headroom for a
# denser form than any tested so far, since a truncated call wastes the
# whole call's cost/latency for nothing.
MAX_TOKENS = 32000
RECOUNT_ROW_MAX_TOKENS = 8000
RECOUNT_MAX_TOKENS = 32000  # whole-table fallback call only, mirrors extract_claude.py

# Confirmed necessary 2026-09-08: the `ollama` package's Client defaults to
# NO request timeout at all (`httpx.Timeout(timeout=None)`, confirmed by
# direct inspection -- `ollama.Client()._client.timeout`) unless one is
# passed at construction. A genuinely stuck call (the local Ollama service
# wedged, GPU contention between this pipeline's own local models --
# chandra and LightOnOCR-2 share the same GPU, an explicitly documented
# untested concern -- or a dropped connection) would then block the calling
# thread FOREVER: no exception, no return, so none of this file's existing
# try/except error handling ever runs. In the review app specifically, that
# thread is the single-worker `_EXECUTOR` (app/pipeline.py) -- a hang there
# leaves a page's `status` stuck at "running" forever with nothing for the
# UI to show an error for, and blocks every other page/session queued
# behind it too. Applied to every `ollama.Client()` this file (and
# hybrid_quantities_lighton.py) constructs, both local and cloud, so a
# stuck call surfaces as a real, caught exception (same as any other
# call failure already handled) instead of hanging silently. 300s is
# generous headroom above every real call time measured on this
# project's forms so far (worst case ~130s, a dense form's chandra main
# call plus LightOnOCR-2 hybrid combined) -- meant to catch a genuine
# hang, not a slow-but-working call.
OLLAMA_REQUEST_TIMEOUT = 300.0

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


class MistralExtractedItem(BaseModel):
    """Mistral-only variant of extract_claude.py's ExtractedItem -- NOT a
    subclass, for the same schema-property-ordering reason
    MistralExtractedForm isn't a subclass of ExtractedForm (see that
    class's own docstring). struck_out must be generated BEFORE quantities/
    row_total for the gating below to mean anything -- committing to "is
    this row cancelled" before the model has already started generating
    quantity-shaped text for it, mirroring date_present's own placement
    rule exactly.

    Confirmed necessary 2026-09-01: sample 3-scanned.jpg has one row
    ("Super Boy 3/4") struck through top to bottom in the photo -- no real
    marks anywhere in it, and its own printed total is blank, the only
    blank total on this 20-row form -- yet the main call reported a full,
    plausible-looking set of quantities for it anyway, borrowed from a
    neighboring row via the same cluster-competition mechanism
    _realign_row_clusters_by_total's own docstring documents fixing for a
    different case. A blank row_total is used as a MODEL-AGNOSTIC proxy for
    "possibly void" in _hybrid_ocr_quantities (works for every model,
    including gemma4:31b, which actually produced the reported bug), but
    it's only a proxy -- a row's total could be blank for other reasons
    (illegible, smudged) without the row being void at all. Asking mistral
    directly whether a row is struck through is a stronger, more direct
    signal where it's available -- the same "recognize an annotation a
    model can see but OCR/coordinates can't" principle already used for
    the date-presence case."""
    model_config = ConfigDict(extra="forbid")
    item: str = Field(description="Product/article name exactly as written (Particulars column), with ditto marks expanded per the DITTO MARKS rule.")
    type: str = Field(description="Style/variant code from the Style column (e.g. IE, OE, RN, RNS). Empty string if there is no separate Style column or it's blank for this row.")
    struck_out: bool = Field(description="True if this entire row has a line drawn through it by hand (e.g. a single stroke across the item name and/or its quantity cells) -- a cancellation mark, not normal handwriting. False for every ordinary row, even one with a blank or illegible cell.")
    letter_sizes: bool = Field(description="True if this row's quantities are written using standard clothing letter sizes (S, M, L, XL, XXL, etc.) instead of aligning to this form's printed numeric size-column headers -- each letter usually stacked directly above its own quantity digit, in a cramped two-line cell. False for an ordinary row using the printed numeric grid, even one with blank or illegible cells. Decide this BEFORE reading quantities -- if true, still report each visible letter+quantity pair you can read in quantities using the letter as the size (per the LETTER SIZES rule), but do not force any of them into the nearest printed numeric column.")
    quantities: List[QuantityPair] = Field(description="One entry per size column with a legible quantity for this row. Empty list if the row has no readable quantities, or if struck_out is true.")
    row_total: str = Field(description="This row's own printed or circled running total (e.g. a 'Total Dozen' column, or a circled number in the margin), exactly as written, regardless of what it's labeled or where on the row it appears. Empty string if this form has no such total, it's blank for this row, or struck_out is true.")
    row_top_frac: float = Field(description="0.0-1.0 fraction of image height where THIS item's own row starts. See row_top_frac/row_bottom_frac rule.")
    row_bottom_frac: float = Field(description="0.0-1.0 fraction of image height where THIS item's own row ends.")

    @model_validator(mode="after")
    def _gate_on_struck_out(self) -> "MistralExtractedItem":
        # Defense in depth, same pattern as MistralExtractedForm's own
        # date-presence gate -- don't rely on the model honoring the
        # "must be empty" instructions above perfectly.
        if self.struck_out:
            self.quantities = []
            self.row_total = ""
        return self

    @model_validator(mode="after")
    def _split_trailing_style_code(self) -> "MistralExtractedItem":
        # Identical to ExtractedItem's own validator of the same name --
        # duplicated rather than imported/shared, so this class stays
        # self-contained (its whole point is to be a drop-in stand-in for
        # ExtractedItem during parsing, converted away immediately after --
        # see _mistral_form_to_extracted_form).
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
    items: List[MistralExtractedItem] = Field(description="Every product/article row, top to bottom, in order.")
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
    validator above) and dropped, and each MistralExtractedItem is
    converted into a plain ExtractedItem the same way (struck_out is
    consumed and dropped; quantities/row_total were already forced empty
    by MistralExtractedItem's own gating validator if struck_out was true,
    so this conversion doesn't need to repeat that check). So every
    function downstream of the main call -- including
    _hybrid_ocr_quantities's own, separate, model-agnostic blank-total
    handling in _realign_row_clusters_by_total -- keeps working with the
    exact same ExtractedForm/ExtractedItem shape it already expects,
    unaware either schema swap ever happened. Constructing fresh objects
    also runs THEIR OWN validators (_clean_order_date,
    _forward_fill_ditto_item_names, ExtractedItem's own
    _split_trailing_style_code) on the way in, same as any other
    ExtractedForm."""
    return ExtractedForm(
        seller_name=m.seller_name,
        party_name=m.party_name,
        order_no=m.order_no,
        order_date=m.order_date,
        size_headers=m.size_headers,
        items=[
            ExtractedItem(
                item=it.item,
                type=it.type,
                quantities=it.quantities,
                row_total=it.row_total,
                row_top_frac=it.row_top_frac,
                row_bottom_frac=it.row_bottom_frac,
            )
            for it in m.items
        ],
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

- STRUCK-OUT / CANCELLED ROWS: on a real test form, one row had a line drawn by hand through the \
entire row (the item name and every quantity cell) -- a cancellation, with no real handwritten \
quantities anywhere in it -- yet you reported a full set of plausible-looking quantities for it \
anyway. Decide struck_out FIRST for each row, before reading its quantities: does this row have an \
actual pen/pencil line drawn through it, separate from the row's own printed ruling and separate from \
normal handwriting? If struck_out is true, quantities must be an empty list and row_total must be an \
empty string -- do not read, guess, or borrow numbers from a neighboring row just to have something to \
put there. A row that is merely blank (no handwriting at all, but also no line through it) is NOT \
struck_out -- leave struck_out false and quantities empty for that case instead.

"""

# A "LETTER-SIZE ROWS: SET letter_sizes=true FROM THE CELL'S SHAPE, EVEN IF
# A LETTER IS TOO FAINT TO READ CONFIDENTLY" bullet was tried here
# 2026-09-02, targeting sample 3-scanned.jpg's MM K4532 row (see CLAUDE.md's
# dated section) -- REVERTED after 3 consecutive real live runs (2 before
# this bullet existed, 1 after) all came back letter_sizes=false for that
# exact row, no change. A genuinely different angle from the two general
# LETTER SIZES bullets already in this prompt (framed around the cell's
# STRUCTURAL SHAPE -- two stacked lines -- rather than asking the model to
# read the letter text itself), but it made no measurable difference. Per
# this project's own established practice (see the 105/110 case and
# sample 12's column-shift bullets elsewhere in CLAUDE.md), this is one real
# attempt at a new hypothesis, not yet the 3-4 tries that earlier cases
# needed before being called a settled ceiling -- but not worth spending
# more live-call budget re-wording without a genuinely different mechanism.
# The crop-based recovery mechanism itself (_recover_letter_size_row_digits,
# _infer_letter_columns) is confirmed working correctly WHEN the flag fires
# (tested with it supplied synthetically) -- the model just isn't setting it
# on this specific row in practice, a real, separate, still-open gap.

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


def get_client(model: str = DEFAULT_MODEL) -> ollama.Client:
    """A chandra model runs on the LOCAL Ollama service (see
    chandra_parser.py), not Ollama Cloud -- no API key needed, and this
    project's other local models (LightOnOCR-2, qwen2.5vl) already use
    the same default `ollama.Client()` (localhost:11434) unauthenticated.
    Every other model still goes through Ollama Cloud as before.
    `timeout=OLLAMA_REQUEST_TIMEOUT` on both -- see that constant's own
    comment for why this isn't optional."""
    if "chandra" in model.lower():
        return ollama.Client(timeout=OLLAMA_REQUEST_TIMEOUT)
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        print(
            "Error: OLLAMA_API_KEY is not set. Add it to .env (as a new line -- don't touch the "
            "existing DB/Anthropic credentials there) or export it before running. Get a key from "
            "ollama.com/settings/keys.",
            file=sys.stderr,
        )
        sys.exit(1)
    return ollama.Client(host=OLLAMA_CLOUD_HOST, headers={"Authorization": f"Bearer {api_key}"}, timeout=OLLAMA_REQUEST_TIMEOUT)


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
    if annotation is bool:
        # Confirmed necessary 2026-08-20: this generic normalizer predates
        # MistralExtractedForm.date_present (the first bool field any reused
        # schema has ever had), so a null date_present sailed straight
        # through to Pydantic and hard-failed BOTH retry attempts on a real
        # run (sample 13-scanned.jpg) instead of being repaired like every
        # other type already handled above. False is the safe default here
        # specifically -- it's also date_present's own documented meaning
        # ("no date"), so an unparseable value degrades to the same
        # no-fabricated-date outcome as a confident False, never a
        # fabricated True.
        if isinstance(val, bool):
            return val
        return False
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


class _RawChatResponse:
    """Minimal stand-in for ollama._types.ChatResponse, built from a raw
    HTTP JSON body -- just enough attribute access (.done_reason,
    .message.content, .prompt_eval_count, .eval_count) for _call_schema's
    own code below to treat it identically to a real SDK response, without
    needing to know which path produced it."""
    class _Message:
        def __init__(self, content: str):
            self.content = content

    def __init__(self, data: dict):
        self.done_reason = data.get("done_reason")
        self.message = self._Message(data.get("message", {}).get("content", ""))
        self.prompt_eval_count = data.get("prompt_eval_count")
        self.eval_count = data.get("eval_count")


def _chat_bypassing_sdk_validation(client: ollama.Client, model: str, messages: list[dict], format_schema: dict, options: dict, think: str, timeout: float = 280.0) -> "_RawChatResponse":
    """Raw HTTP call to Ollama Cloud's /api/chat, bypassing the ollama
    Python package's OWN client-side Pydantic validation of the `think`
    field -- confirmed 2026-09-02: that field is typed as
    `bool | Literal['low','medium','high']` in the installed SDK version,
    so passing think='max' (glm-5.3-flash's own documented top reasoning
    tier, per ollama.com/library/glm-5.3-flash) raises a ValidationError
    before any request is even sent. A direct, unvalidated HTTP call to
    the same endpoint with think='max' in the JSON body returned a normal
    200 response -- the Ollama Cloud API itself accepts it fine; this is
    purely an outdated client-library type hint, not a server limitation.
    Reuses the already-authenticated client's own base_url/auth header
    (client._client is the underlying httpx.Client `ollama.Client` already
    built in get_client()) rather than re-deriving credentials here."""
    # The SDK base64-encodes raw image bytes before serializing a request
    # (see ollama._types.Image.serialize_model) -- bypassing the SDK means
    # doing that encoding step ourselves; every other message field passes
    # through unchanged.
    wire_messages = []
    for msg in messages:
        wire_msg = dict(msg)
        if "images" in wire_msg:
            wire_msg["images"] = [base64.b64encode(img).decode() for img in wire_msg["images"]]
        wire_messages.append(wire_msg)

    resp = client._client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": wire_messages,
            "format": format_schema,
            "options": options,
            "think": think,
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return _RawChatResponse(resp.json())


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
        if THINK == "max":
            # 'max' is glm-5.3-flash's own documented top reasoning tier,
            # but the installed ollama SDK's think field only accepts
            # bool | 'low' | 'medium' | 'high' -- see
            # _chat_bypassing_sdk_validation's own docstring for the real
            # HTTP call that confirmed the SERVER accepts 'max' fine.
            resp = _chat_bypassing_sdk_validation(
                client, model, messages,
                _neutralize_schema_examples(schema_model.model_json_schema()),
                {"temperature": 0, "num_predict": max_tokens},
                THINK,
            )
        else:
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

# Fraction of the VLM's own filled-cell count that the OCR pass must actually
# detect before its reading is allowed to correct anything. Below this, the
# whole hybrid stage stands down for the page -- see the long comment at the
# guard's own site in _hybrid_ocr_quantities for the measurements behind it.
# Healthy pages in this project's test set land at 102-108%; the one page that
# defeats OCR detection lands at 22%.
MIN_OCR_COVERAGE = 0.60


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


def _reconcile_hybrid_with_vlm(vlm_map: dict[str, int], hybrid_map: dict[str, int], printed_total: int | None, size_headers: set[str]) -> tuple[dict[str, int], dict | None]:
    """Reconciles hybrid OCR's reading for a row against the VLM's OWN
    main-call reading for that same row, instead of _hybrid_ocr_quantities's
    caller unconditionally overwriting one with the other (confirmed the
    wrong shape 2026-09-02 -- see CLAUDE.md's "WHY hybrid and the VLM need to
    complement each other" section). hybrid (pixel-grounded OCR) is kept as
    the default in every case -- the only thing this function ever does is
    (a) arbitrate a pure VALUE disagreement on an otherwise-IDENTICAL column
    set via this row's own printed total, when that arbitration carries no
    positional ambiguity, or (b) surface a flag when the two sources
    disagree about which COLUMNS even hold this row's data, without ever
    acting on it.

    This is deliberately more conservative than an earlier version of this
    function tried and shipped the same day: that version also treated a
    DIFFERENT column set as reconcilable (preferring whichever of
    hybrid-alone / vlm-alone / a union matched the printed total), on the
    theory that a hybrid-missing cell the VLM caught (CLAUDE.md's "row 4"
    evidence) is real, recoverable signal. Two separate, real, live-run
    regressions disproved that a same-call self-reported row_total is
    trustworthy enough to arbitrate a COLUMN disagreement: sample
    5-scanned.jpg's "Bloomers Plain"/"F.G-3025" rows (the model read an
    entirely different header block that also happened to foot its own
    row_total) and sample 12-scanned.jpg's "TA22 Short" row (a genuine
    2-column shift that still shared 4 of 7 columns with the correct
    reading, so a "some overlap" guard alone didn't catch it either -- and
    critically, hybrid's OWN correct reading did NOT match this row's
    printed total, ruling out "does hybrid already match" as a safe
    discriminator too). In both cases vlm_map's quantities and its own
    row_total came from the SAME model call, so "vlm's reading sums to its
    own total" is nearly tautological (self-consistency), not independent
    corroboration -- unlike hybrid matching the total, which genuinely is
    independent evidence, since hybrid is a different mechanism entirely.
    A pure value conflict on an IDENTICAL column set has no such positional
    ambiguity to exploit (both sources already agree on which columns
    matter), so the total safely arbitrates only that narrower case.

    Net effect: a hybrid-missing real mark (row 4's shape) is no longer
    auto-recovered -- it's flagged instead, so a human notices hybrid's
    blind spot without risking a column-shifted VLM reading silently
    overwriting an already-correct row elsewhere. Matches this project's
    own established "a model being honestly wrong beats a model being
    confidently wrong" principle (see CLAUDE.md's "Design decisions worth
    preserving") applied to hybrid's own blind spots, not just the VLM's.

    Returns (merged_map, flag) -- flag is None when there's nothing here
    worth a reviewer's attention (the caller still runs its own
    _flag_hybrid_total_mismatch checksum check on the merged map in that
    case, unchanged); otherwise the same {"status","sizes","note"} shape
    generate_review.py already renders for hybrid/recount flags."""
    if not hybrid_map or not vlm_map or not set(hybrid_map) <= size_headers:
        # Empty hybrid (voided/struck-out row, or hybrid found nothing): keep
        # as-is, nothing to reconcile against. Empty vlm_map: no independent
        # signal to check hybrid against either. hybrid_map using a key
        # outside this form's own printed size_headers means it already
        # resolved a DIFFERENT, more-authoritative axis (a letter-coded row
        # via the letter-size-sharing mechanism, or a numeric overflow column
        # via _split_merged_header_token) -- vlm_map is reading the wrong
        # axis entirely there, and cell-merging against it would corrupt an
        # already-hard-won fix rather than improve it. Trust hybrid wholesale
        # in every one of these cases, exactly as before this function existed.
        return hybrid_map, None
    if vlm_map == hybrid_map:
        return hybrid_map, None  # exact agreement -- the strongest case, nothing to do

    hybrid_keys, vlm_keys = set(hybrid_map), set(vlm_map)
    sort_key = lambda s: int(s) if s.isdigit() else 0

    if hybrid_keys != vlm_keys:
        # Different COLUMN sets, not just different values on shared ones --
        # a positional disagreement about which columns hold this row's
        # data, not a "which digit is right" question. See this function's
        # own docstring for why a same-call self-reported total can't safely
        # arbitrate this shape. Always keep hybrid; still worth flagging the
        # divergence so a reviewer can glance at the photo.
        note = (f"model read this row under different sizes ({sorted(vlm_keys, key=sort_key)}) than "
                f"pixel-OCR did ({sorted(hybrid_keys, key=sort_key)}) -- kept the OCR reading.")
        return hybrid_map, {"status": "unverified", "sizes": sorted(hybrid_keys, key=sort_key), "note": note}

    # Identical column set, differing value(s) on >=1 cell -- the
    # "smoothed outlier" shape (rows 3/8/11/14 on sample 3-scanned.jpg,
    # think=high: the VLM smooths a genuine outlier digit into the row's
    # dominant value). No positional ambiguity here, so the printed total
    # can safely arbitrate which source's digit is right.
    conflicts = sorted((s for s in hybrid_keys if vlm_map[s] != hybrid_map[s]), key=sort_key)
    diffs = "model and pixel-OCR disagree on size(s) " + "; ".join(
        f"{s}: model={vlm_map[s]} vs OCR={hybrid_map[s]}" for s in conflicts
    ) + "."
    if printed_total is not None:
        hybrid_sum, vlm_sum = sum(hybrid_map.values()), sum(vlm_map.values())
        if hybrid_sum == printed_total and vlm_sum != printed_total:
            return hybrid_map, {"status": "resolved", "sizes": conflicts, "note": f"{diffs} Kept pixel-grounded OCR (matches this row's own printed total)."}
        if vlm_sum == printed_total and hybrid_sum != printed_total:
            return vlm_map, {"status": "resolved", "sizes": conflicts, "note": f"{diffs} Kept the AI model's own reading (matches this row's own printed total)."}
    return hybrid_map, {"status": "unresolved", "sizes": conflicts, "note": diffs}


def _row_order_plausible(items: list[ExtractedItem]) -> bool:
    """Looser stand-in for _validate_row_fracs, used only by
    _hybrid_ocr_quantities's row-to-item matching below. There, row_top_frac/
    row_bottom_frac are only ever used as a coarse top-to-bottom ORDERING
    hint for the order-preserving DP that matches y-clustered OCR marks to
    items -- never to crop pixels or define a strict containment range, the
    way _validate_row_fracs's other caller (the recount row-crop builder,
    which genuinely needs hard [0,1] bounds) uses them.

    Confirmed necessary 2026-08-20: _validate_row_fracs hard-rejects the
    WHOLE form the moment even one item's fraction is out of [0,1] range --
    exactly what happened on sample 13-scanned.jpg (20 items), where the
    model's row fractions are a mechanically uniform sequence (every item
    exactly 0.03-0.04 tall, incrementing in lockstep from 0.38 to 1.03) that
    overshoots 1.0 by 0.03 on the LAST item alone. That overshoot means the
    fractions aren't genuine pixel measurements, but the sequence's ORDER is
    still almost certainly right (a formulaic partition can't reorder
    items) -- which is all this function's actual use of them requires.
    Confirmed the strict gate was costing real accuracy, not just being
    over-cautious: with the hybrid pass skipped, several of this form's real
    rows read badly wrong from mistral's raw main-call reading alone (row 2
    and row 5 both undercounted by more than half against the form's own
    printed BOXES total) -- see CLAUDE.md."""
    prev_center = -1.0
    for it in items:
        top, bottom = it.row_top_frac, it.row_bottom_frac
        if bottom <= top:
            # Degenerate (zero-height) item -- carries no ordering info to
            # violate, so it can't actually break monotonic order; skipped
            # rather than treated as a hard failure. Confirmed necessary
            # 2026-08-22 on sample 3-scanned.jpg: the model reported
            # row_top_frac == row_bottom_frac == 1.0 for the LAST two items
            # (MM LOOPER 4289, NIVI KNOT RNBS), which used to veto the
            # ENTIRE hybrid pass for all 20 rows before OCR even ran, even
            # though the other 18 items' fractions were perfectly valid.
            # _recover_degenerate_item_centers/_find_item_name_y further
            # down already exist specifically to reconstruct a real
            # position for exactly this case (see their own docstrings,
            # 2026-08-21) -- but they never got a chance to run, since this
            # gate rejected the whole image first.
            continue
        center = (top + bottom) / 2
        if center < prev_center - 0.005:
            return False
        prev_center = center
    return True


def _find_item_name_y(item_name: str, texts: list[str], boxes: list[list[float]], scores: list[float],
                       x_floor: float, y_floor_px: float) -> float | None:
    """Searches the item-name column (x < x_floor) for OCR text that
    CONTENT-matches this specific item's name -- a distinctive 3+-digit
    numeric code (e.g. "4532") or a 3+-letter alpha word (e.g. "LOOPER"),
    extracted from the name itself, not a fixed vocabulary. This is the
    same "exact match on a known value" principle _hybrid_ocr_quantities
    already uses for HEADER detection (columns) -- match on real, known
    CONTENT, not a generic position guess -- just applied to item names
    instead of header numbers. 2-letter/short tokens (e.g. "MM", a prefix
    shared by several items in the same group on a real form) are
    deliberately excluded so a non-distinguishing common prefix can't
    produce an ambiguous match across several different rows.

    Returns the y-center (in px) of the matched text, or None if no
    distinctive token exists in the name, no match was found, or the
    matches found are too spread out to trust (more than one real
    row-height apart -- suggests an accidental/ambiguous match, e.g. the
    same digit code appearing elsewhere on the page, rather than one real
    row's own text)."""
    alpha_words = {w.upper() for w in re.findall(r"[A-Za-z]{3,}", item_name)}
    digit_codes = set(re.findall(r"\d{3,}", item_name))
    if not alpha_words and not digit_codes:
        return None

    candidate_ys = []
    for t, b, s in zip(texts, boxes, scores):
        if s < 0.5 or (b[0] + b[2]) / 2 >= x_floor or b[1] <= y_floor_px:
            continue
        tt = t.upper()
        if any(w in tt for w in alpha_words) or any(d in t for d in digit_codes):
            candidate_ys.append((b[1] + b[3]) / 2)
    if not candidate_ys:
        return None
    candidate_ys.sort()
    if candidate_ys[-1] - candidate_ys[0] > 60.0:  # roughly a real form's own row height -- see below
        return None  # scattered across more than one row's worth of height -- ambiguous, don't trust it
    return sum(candidate_ys) / len(candidate_ys)


def _recover_degenerate_item_centers(
    centers: list[float], extracted: ExtractedForm,
    texts: list[str], boxes: list[list[float]], scores: list[float],
    x_floor: float, y_floor_px: float, height: int,
) -> list[float]:
    """Rescues item row-center fractions for items whose VLM-self-reported
    row_top_frac/row_bottom_frac are DEGENERATE -- literally identical to a
    sibling item's, giving the row-clustering zero positional signal to
    differentiate them. Confirmed real and reproducible on
    sample 3-scanned.jpg (two independent live runs, same signature both
    times, 2026-08-21): mistral's main call reported the exact same
    (0.98, 1.0) fraction for 4 consecutive items (MM K4532 through Nivi
    Brick).

    THREE designs were tried against this same real data before this one --
    see CLAUDE.md's 2026-08-21 sections for the full writeups, kept here as
    a short summary so a future change doesn't re-try any of them:
    1. Matching OCR text-density clusters (position only, no content check)
       to items via the same order-preserving DP used elsewhere in this
       file. Failed because the bias wasn't confined to the exactly-tied
       items -- the DP's GLOBAL joint cost-minimization "spent" a
       degenerate item's real cluster satisfying a nearby, non-degenerate-
       but-still-biased neighbor instead.
    2. A single scalar offset between a naive equal-share (bottom-top)/n
       uniform division and the model's own non-degenerate centers.
       Confirmed NOT constant (grows with index) -- an early, narrow
       calibration window just happened to look stable by coincidence.
    3. The same equal-share idea reworked to WEIGHT each item by its own
       real height instead of assuming identical 1/n slices (supporting
       rows of different sizes, the same principle as the column-drift
       fix). A real, partial improvement (2 of 4 degenerate rows became
       correct or close), but still an ESTIMATE, not real evidence -- it
       has no way to be more accurate than roughly "somewhere in the
       neighborhood," because it never looks at the image again.

    This version does what attempt 1 SHOULD have done: use real OCR
    coordinates directly, the same way column drift is fixed elsewhere in
    this file, by matching on CONTENT (a header's own exact string) rather
    than raw position. Attempt 1 only used OCR text for generic position
    CLUSTERING (gap-based grouping, no idea which cluster was which item),
    which is why an order-preserving DP was needed at all -- and why it was
    vulnerable to the DP donating a cluster to the wrong (nearby, biased)
    item. This version instead searches, PER ITEM, for OCR text that
    directly CONTAINS a distinctive piece of THAT item's own name (a
    3+-digit code or 3+-letter word, via _find_item_name_y) -- no
    competition between items, no clustering, no DP: either this specific
    item's own name is found in the image or it isn't. Confirmed real and
    unambiguous on this exact case: "MM K4532" -> the digit code "4532"
    matches only one place on the page; "MM Looper 4289" -> "4289"; "Nivi
    Brick RNBS" -> "NIVI" -- each resolves to the item's true row position
    exactly (0.854, 0.883/n-a, 0.917, 0.950 as fractions -- matching this
    file's own photo-verified ground truth for this group precisely,
    confirmed via a real re-run, not assumed from the design alone).

    Falls back to attempt 3's weighted-division ESTIMATE only for an item
    whose name has no distinctive token to search for, or whose search
    comes up empty/ambiguous -- strictly better than attempt 3 alone, since
    real evidence is now used everywhere it's available and the estimate is
    only relied on as a last resort, not as the primary mechanism.

    2026-08-21, same day, later: content-matching is now tried for EVERY
    item, not just ones that were part of an exact tie -- confirmed
    necessary via a real re-run: a fresh live draw on the same form (main
    calls are non-deterministic, per this file's own extensive history)
    came back with items 16/17 (MM K4532 / MM K 3674) showing the SAME
    "identical quantities" symptom as the original degenerate-collapse bug,
    but this time with row_top_frac/row_bottom_frac that were DISTINCT
    (0.89-0.92 vs 0.92-0.95) -- not tied, so the exact-tie gate correctly
    did nothing, yet the underlying misattribution was the same. This
    confirms what the offset-calibration investigation (attempt 2, above)
    already found: the bias in this table's tail isn't limited to the
    exactly-collapsed items, it's just usually not severe enough to
    literally tie two items together. Only the WEIGHTED-DIVISION fallback
    stays gated to degenerate items (it's an estimate with no real
    evidence behind it, appropriate only as a last resort for a total
    information loss) -- content-matching itself is real, verifiable
    evidence regardless of whether the model's own fraction happened to
    collide with a neighbor's or not, so restricting it to only the
    collided case was leaving accuracy on the table for no safety reason."""
    n = len(centers)
    degenerate = [False] * n
    for i in range(n - 1):
        if abs(centers[i + 1] - centers[i]) < 1e-6:
            degenerate[i] = degenerate[i + 1] = True

    new_centers = list(centers)
    still_degenerate = list(degenerate)
    for k in range(n):
        y = _find_item_name_y(extracted.items[k].item, texts, boxes, scores, x_floor, y_floor_px)
        if y is not None:
            new_centers[k] = y / height
            still_degenerate[k] = False

    if not any(still_degenerate):
        return new_centers

    top, bottom = extracted.table_top_frac, extracted.table_bottom_frac
    if not any(still_degenerate) or not (0.0 <= top < bottom <= 1.0) or n == 0:
        return new_centers

    trusted_heights = sorted(
        extracted.items[k].row_bottom_frac - extracted.items[k].row_top_frac
        for k in range(n) if not degenerate[k] and extracted.items[k].row_bottom_frac > extracted.items[k].row_top_frac
    )
    if len(trusted_heights) < 3:
        return new_centers
    typical_height = trusted_heights[len(trusted_heights) // 2]  # median, robust to a couple of outlier rows

    weights = [
        (extracted.items[k].row_bottom_frac - extracted.items[k].row_top_frac)
        if not degenerate[k] and extracted.items[k].row_bottom_frac > extracted.items[k].row_top_frac
        else typical_height
        for k in range(n)
    ]
    total_weight = sum(weights)
    if total_weight <= 0:
        return new_centers
    scale = (bottom - top) / total_weight

    cursor = top
    for k in range(n):
        span = weights[k] * scale
        if still_degenerate[k]:
            new_centers[k] = cursor + span / 2
        cursor += span
    return new_centers


def _realign_row_clusters_by_total(
    y_clusters: list[list[tuple[int, float, float]]],
    item_centers: list[float],
    items: list[ExtractedItem],
) -> tuple[dict[int, int], set[int]]:
    """Order-preserving assignment of OCR row-clusters (real, pixel-grounded
    quantity marks -- one cluster per physically detected row) to items.
    Same job as _monotonic_assign(cluster_ys, item_centers), and reduces to
    exactly that when no printed totals are available, but additionally
    prefers whichever valid assignment maximizes how many clusters' own
    summed quantity matches the assigned item's own printed row_total,
    using y-proximity only as a tiebreaker. Also deprioritizes (never
    strictly excludes) VOIDABLE items -- see below -- from receiving a
    cluster at all, so a row with no real data of its own can't cannibalize
    a real neighbor's.

    Why plain y-proximity isn't enough: confirmed directly on
    sample 3-scanned.jpg (2026-09-01 row-bleed diagnosis -- see CLAUDE.md)
    that when even ONE physical row's marks go completely undetected by OCR
    (too faint, or a cramped stacked letter-over-digit cell like MM K4532's
    own row), _monotonic_assign has no way to know a cluster is MISSING --
    it just gives every subsequent item the next row's cluster instead,
    cascading an off-by-one substitution through most of the table.
    item_centers can't rescue this either: they're frequently a formulaic,
    evenly-spaced guess rather than a real per-row measurement (see
    _row_order_plausible's docstring), so they don't reliably tell the DP
    WHERE the gap is -- confirmed on the same run that even
    _recover_degenerate_item_centers's own content-matching (a real,
    independent signal) doesn't always land for every item (e.g. "MM
    K4532"'s own distinctive code, "4532", didn't resolve this run), so a
    second, independent signal is worth having rather than relying on that
    alone.

    The row's own printed running total is that second signal: confirmed
    on the same real run that it stays correctly read for its OWN row even
    when the quantities read for that row are actually the next row's data
    -- i.e. the main call loses track of which row's DIGITS it's reading
    well before it loses track of which row's TOTAL it's reading. Distance
    stays the tiebreaker (not thrown away) for the common case where no
    printed total disambiguates -- including every row whose total is
    blank/illegible, where this is identical to _monotonic_assign.

    Deliberately NOT the already-disproven "shift a row's values under
    different header KEYS to match its total" idea (see
    _flag_hybrid_total_mismatch's docstring) -- relabeling can't change a
    row's sum, since the values are unchanged. This instead chooses WHICH
    CLUSTER of real marks -- a genuinely different multiset of values, with
    a genuinely different sum -- gets assigned to an item in the first
    place, which total-matching can actually detect and correct.

    VOIDABLE ITEMS -- confirmed necessary 2026-09-01, a DIFFERENT bug from
    the one above, found by the user directly on the same form:
    sample 3-scanned.jpg's "Super Boy 3/4" row is struck through top to
    bottom in the photo -- no real marks anywhere in it -- and its own
    row_total is blank (the only blank total on this 20-row form). This
    total-matching fix alone doesn't help a row like that at all: with no
    total to check against, it competes for a cluster on pure y-distance
    exactly like _monotonic_assign always did, and happily absorbs a real
    neighbor's marks. An item is treated as voidable when its own row_total
    doesn't parse to a number AND fewer than half the table's rows are in
    that same state -- the second condition matters because a blank total
    is completely normal on some form layouts (this only means something
    when it's the rare exception, not the rule for this specific form).
    Voidable items are deprioritized, not excluded outright -- if every
    non-voidable item is already satisfied and a real cluster is still
    left over, a voidable item can still receive one (a blank total isn't
    proof of a void row, just a proxy for it), but never at a non-voidable
    item's expense. The caller uses the returned voidable set to force
    genuinely-unmatched voidable items to explicit empty quantities rather
    than falling back to the main call's own (frequently also-wrong) guess
    for that row.

    Standard three-key (match_count, -voidable_assigned_count,
    -total_distance) lexicographic DP, O(len(y_clusters) *
    len(item_centers)) -- both are ~20 for every form in this project's
    test set, so this is cheap. Every cluster is assigned (same contract as
    _monotonic_assign); the caller is already guaranteed len(y_clusters) <=
    len(item_centers)."""
    cluster_sums = [sum(q for q, _x, _y in cluster) for cluster in y_clusters]
    cluster_ys = [sum(c[2] for c in cluster) / len(cluster) for cluster in y_clusters]
    item_totals = [_parse_int_or_none(it.row_total) for it in items]

    n = len(item_centers)
    blank_total_indices = {j for j, t in enumerate(item_totals) if t is None}
    voidable = blank_total_indices if len(blank_total_indices) < n * 0.5 else set()

    m = len(cluster_ys)
    NEG = (-1, -(m + 1), float("-inf"))
    # dp[i][j]: best (match_count, -voidable_assigned_count, -total_distance)
    # assigning clusters[0:i] using only items[0:j] (order-preserving, i.e.
    # cluster i-1 if placed must land on some item < j). dp[0][*] =
    # (0, 0, 0.0) -- no clusters placed yet is always trivially achievable.
    dp: list[list[tuple[int, int, float]]] = [[(0, 0, 0.0)] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = NEG  # a cluster with zero items available to hold it
        for j in range(1, n + 1):
            skip = dp[i][j - 1]  # item j-1 holds nothing
            take = NEG
            if dp[i - 1][j - 1] != NEG:
                match = 1 if item_totals[j - 1] is not None and cluster_sums[i - 1] == item_totals[j - 1] else 0
                voided = 1 if (j - 1) in voidable else 0
                dist = abs(cluster_ys[i - 1] - item_centers[j - 1])
                prev_count, prev_void, prev_neg_dist = dp[i - 1][j - 1]
                take = (prev_count + match, prev_void - voided, prev_neg_dist - dist)  # cluster i-1 -> item j-1
            dp[i][j] = max(skip, take)

    assignment: dict[int, int] = {}
    i, j = m, n
    while i > 0:
        if j > 0 and dp[i][j] == dp[i][j - 1]:
            j -= 1
            continue
        assignment[i - 1] = j - 1
        i -= 1
        j -= 1
    return assignment, voidable


STANDARD_LETTER_SIZE_ORDER = ["XS", "S", "M", "L", "XL", "XXL", "2XL", "3XL", "4XL"]


def _recover_letter_size_row_digits(
    original_bytes: bytes, header_xs: list[float], anchor_label_y: float,
    spacing: float, width: int, height: int, band_height: float, upscale: int = 8,
) -> dict[float, int]:
    """Per-COLUMN targeted, upscaled re-OCR of ONE row's stacked letter-size
    cells -- used only for a row mistral has directly flagged
    (MistralExtractedItem.letter_sizes) as using letter clothing sizes
    instead of the printed numeric grid.

    Root cause this exists for, confirmed via direct inspection of
    sample 3-scanned.jpg's MM K4532 row: the handwritten letter label and its
    own quantity digit are stacked so tightly (a cramped two-line cell, no
    ruled line between them) that PaddleOCR's whole-page detection pass
    sometimes merges both lines into ONE box, and the recognizer reads the
    merged crop as garbage ('#6', 'y', '2') instead of two legible
    characters -- confirmed for 3 of this row's 5 columns (headers 45/50/55),
    while the other 2 (60/65, whose lines happened to have enough visual gap)
    were detected as two clean separate boxes each ("XL"/"6", "xxC"/"6").

    TWO prior designs were tried and failed before this one, both real
    attempts confirmed via saved crop images, not assumed:
    1. One wide crop spanning every header column at once (mirroring the
       shape of the row itself) -- failed outright regardless of height
       tried (34px, 55px, 73px): PaddleOCR's detector returned ONE giant box
       spanning almost the whole strip, and recognition on that box came back
       either garbage ('6666', one blob) or completely empty (score 0.0) --
       too much heterogeneous content (5 separate label+digit cells, mostly
       blank paper past the group) for the detector to segment correctly.
    2. A narrow per-column crop bounded to just the label+digit cell's own
       known height -- still failed on several columns: a ruled horizontal
       line (the row's own bottom boundary) frequently cuts straight through
       the handwritten digit at that fixed vertical position, confirmed by
       saving and visually inspecting the crop directly.

    This version crops ONLY the digit's own sub-line (skips the label line
    entirely -- this function never tries to read the letter text itself,
    see below) in a NARROW single-column band, then sweeps a small range of
    vertical offsets and takes the highest-scoring literal digit found
    anywhere in the sweep -- confirmed via direct testing this reliably
    dodges the ruled-line-through-the-digit problem, since at least one
    offset in the sweep always lands the digit cleanly between ruled lines.
    Deliberately requires t.isdigit() exactly (no _try_digit_correct
    letter-lookalike fallback, unlike the whole-page pass) -- confirmed
    necessary: a stray 'L' character (bleeding in from a neighboring cell at
    one offset) scored 0.93 and would have been "corrected" to the digit 1
    by that fallback, a real wrong-value risk this search's much larger
    number of attempts (multiple offsets x multiple columns) makes more
    likely to hit than the whole-page pass's single attempt per token ever
    was.

    Deliberately does NOT try to re-read the LETTER text itself -- recognizing
    S/M/L/XL/XXL reliably in a cramped handwritten cell is a strictly harder
    problem than recovering a single quantity digit at an already-known
    x-position (this form's own printed numeric headers give exact, reliable
    column anchors to crop and search around, whether or not this row
    actually uses their numeric MEANING). The caller maps each recovered
    column to a letter by ORDINAL (left-to-right) position relative to
    whichever letters the whole-page pass DID manage to read cleanly (e.g.
    "XL"/"XXL" here) -- see _infer_letter_columns.

    Returns {header_x_position: quantity} for every column where a confident
    digit was found -- the caller is responsible for merging this with
    whatever the whole-page pass already found and mapping to letters."""
    image = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    ocr = get_ocr()
    recovered: dict[float, int] = {}
    for hx in header_xs:
        left_px = max(0, int(hx - spacing * 0.5))
        right_px = min(width, int(hx + spacing * 0.5))
        if right_px - left_px < 10:
            continue
        best: tuple[str, float] | None = None
        # Sweep vertical offsets covering roughly one row's worth of
        # displacement below the label line -- confirmed via direct testing
        # to be the range needed to dodge the ruled line at whichever exact
        # position it happens to cross a given column's digit (varies per
        # column on this real form, not a fixed offset).
        for dy in range(-16, 26, 2):
            top_px = max(0, int(anchor_label_y + band_height * 0.35 + dy))
            bottom_px = min(height, int(anchor_label_y + band_height * 1.55 + dy))
            if bottom_px - top_px < 6:
                continue
            crop = image.crop((left_px, top_px, right_px, bottom_px))
            crop = crop.resize((crop.width * upscale, crop.height * upscale), Image.LANCZOS)
            arr = np.array(crop)[:, :, ::-1]
            result = list(ocr.predict(arr))
            if not result:
                continue
            for t, s in zip(result[0]["rec_texts"], result[0]["rec_scores"]):
                if s < 0.85 or not t.isdigit():
                    continue
                if best is None or s > best[1]:
                    best = (t, s)
        if best is not None:
            qty = int(best[0])
            if 0 < qty <= 500:
                recovered[hx] = qty
    return recovered


def _infer_letter_columns(group_header_xs: list[float], known: dict[str, float], tolerance: float) -> dict[str, float] | None:
    """Given the sorted x-positions of every header column a letter-size
    group spans and at least one CONFIRMED letter->x anchor (from real OCR
    text, e.g. "XL" read cleanly at one column), infers the letter for
    EVERY column in the group by ordinal (left-to-right) offset from
    STANDARD_LETTER_SIZE_ORDER -- deliberately does not guess a starting
    letter with zero anchor, and refuses (returns None) if multiple anchors
    disagree on the offset, or if the inferred range would run off either
    end of the standard list. Confirmed real via direct OCR + pixel
    measurement on sample 3-scanned.jpg: MM K4532's 5 columns, anchored by
    a cleanly-read "XL" at the 4th (of 5) column, infer to
    S/M/L/XL/XXL -- exactly the label sequence confirmed present in the
    photo."""
    if not known:
        return None
    group_sorted = sorted(group_header_xs)
    anchors = []
    for letter, x in known.items():
        if letter not in STANDARD_LETTER_SIZE_ORDER:
            continue
        pos = min(range(len(group_sorted)), key=lambda i: abs(group_sorted[i] - x))
        if abs(group_sorted[pos] - x) > tolerance:
            continue
        anchors.append((pos, STANDARD_LETTER_SIZE_ORDER.index(letter)))
    if not anchors:
        return None
    col_pos, order_idx = anchors[0]
    offset = order_idx - col_pos
    if offset < 0 or offset + len(group_sorted) > len(STANDARD_LETTER_SIZE_ORDER):
        return None
    for col_pos2, order_idx2 in anchors[1:]:
        if order_idx2 - col_pos2 != offset:
            return None  # anchors disagree -- don't trust this group
    return {STANDARD_LETTER_SIZE_ORDER[offset + i]: x for i, x in enumerate(group_sorted)}


def _hybrid_ocr_quantities(image_path: Path, extracted: ExtractedForm, outdir: Path | None = None, letter_size_hints: set[int] = frozenset()) -> tuple[dict[int, dict[str, int]], dict[int, dict]]:
    """Reads quantities from REAL OCR-measured coordinates for headers and
    quantity marks, then groups each mark with its nearest header by
    x-position -- a single OCR pass on the whole image (no per-row crops, no
    grid.py row-boundary detection), so this sidesteps both root causes
    behind the 2026-08-17 --cv-quantities revert (PaddleOCR's inconsistent
    internal resize on stitched crops; grid.py's row-boundary detection
    getting confused by an extra letterhead row).

    2026-08-20: the mark-to-header grouping itself is now done by CODE
    (order-preserving DP, _monotonic_assign), not by asking mistral to
    reproduce the arithmetic in a text reply -- no model call happens in
    this function at all any more. See the grouping code below for why:
    a real run found the VLM silently dropping a correctly-detected mark
    from its own reply despite being handed exact coordinates for
    everything. The historical note below (isolated-test accuracy of the
    VLM-grouping approach) is kept for context on why this function exists
    in this shape, not as a description of current behavior.

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

    Returns (quantities_by_index, flags). flags is aligned by item index,
    same {"status", "sizes", "note"} shape as extract_claude.py's own
    recount flags (see _apply_recount) -- 2026-08-22: this function used to
    return a flat list of bookkeeping/diagnostic strings appended straight
    into extracted.notes, which just piled up as an unstructured wall of
    text at the top of the review page (nothing pointed at which cell was
    actually suspect). Whole-pipeline skip/failure reasons below are now
    printed to the console instead (operationally useful, not something a
    reviewer checking cells against a photo needs to read); only a genuine
    per-row signal (the total-checksum mismatch) becomes a flag, so it
    renders as a highlighted cell via generate_review.py's existing
    recount-flag mechanism instead of prose. Rows where OCR found candidate
    marks AND the code-side header assignment was confident (avg distance
    under ASSIGN_COST_LIMIT) are included in quantities_by_index; every
    other row keeps the model's own main-call reading, same fallback
    philosophy as this file's other optional passes -- EXCEPT a voidable
    row (see _realign_row_clusters_by_total's own docstring) that ended up
    with no cluster, which is included as an explicit empty dict instead of
    falling back, since that fallback is exactly what was showing
    fabricated quantities for a struck-through row in the first place."""
    size_headers = extracted.size_headers
    n_items = len(extracted.items)
    if n_items == 0 or not size_headers:
        return {}, {}
    if not _row_order_plausible(extracted.items):
        print("  hybrid OCR quantity read: skipped -- row_top_frac/row_bottom_frac from the main read wasn't even monotonically ordered top-to-bottom.")
        return {}, {}

    original_bytes = _exif_corrected_bytes(image_path)
    image = Image.open(io.BytesIO(original_bytes)).convert("RGB")
    width, height = image.size
    arr = np.array(image)[:, :, ::-1]
    ocr = get_ocr()
    result = list(ocr.predict(arr))
    if not result:
        print("  hybrid OCR+VLM quantity read: OCR found no text at all on this image.")
        return {}, {}
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
        print(f"  hybrid OCR+VLM quantity read: only found {len(header_x)}/{len(size_headers)} headers via OCR -- skipped (likely a free-form page with no shared header row).")
        return {}, {}

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
    # Restricted to the header/quantity column x-range (x_floor..x_ceiling) --
    # confirmed necessary 2026-09-02 on sample 3-scanned.jpg: an unrelated,
    # tall OCR box from the Style/Particulars column (x well left of
    # x_floor, e.g. a garbled partial read of that row's own style text,
    # "ens" at x=251) landed inside this y-window and its bottom edge alone
    # inflated the computed band_bottom (a fake "ens" box reaching y=382,
    # deeper than the real decoy header row's own bottom of ~364) --
    # pushing y_floor_px 18px past where item 0's real quantity marks
    # actually start (y=365), so every one of that row's marks (whose box
    # TOPS sat just below the true decoy row but above this inflated floor)
    # got excluded from candidates entirely, and the row fell back to the
    # main call's own wrong reading (column-shifted AND missing 2 values).
    # The "covered" count already implicitly ignored text outside header
    # range (no header sits near x=251, so it never counted toward
    # coverage), but the bottoms computation had no such filter -- unifying
    # both to the same x-range closes that gap.
    def _band_header_coverage(y_lo: float, y_hi: float) -> tuple[int, float]:
        in_band = [b for _t, b, sc in zip(texts, boxes, scores)
                   if sc >= 0.5 and y_lo < (b[1] + b[3]) / 2 <= y_hi and x_floor <= (b[0] + b[2]) / 2 <= x_ceiling]
        if not in_band:
            return 0, y_hi
        xs_in_band = [(b[0] + b[2]) / 2 for b in in_band]
        covered = sum(1 for hx in header_xs_sorted if any(abs(hx - x) <= spacing * 0.4 for x in xs_in_band))
        return covered, max(b[3] for b in in_band)

    # Coverage threshold raised from 0.5 to 0.85, 2026-09-02 -- confirmed
    # necessary once the x-range fix above stopped an unrelated Style-column
    # token from artificially inflating band_bottom: with that contamination
    # gone, sample 3-scanned.jpg's item 0 (a genuinely WIDE real row, 8 of
    # 13 header columns = 62% coverage) started tripping this SAME >=50%
    # test on its own -- pushed past not once but three times (0.62, 0.62,
    # 0.62 across the range(3) cap), landing y_floor_px 90+px past its own
    # real data. A real printed decoy sub-header row (this form's own
    # "18/20/22.../42" line, and every other decoy row confirmed elsewhere
    # in this file) is a full-width table LABEL row by construction --
    # measured directly across all 4 of this project's core regression
    # forms, every confirmed genuine decoy row showed 1.00 (complete)
    # coverage, while sample 3's false-positive wide DATA row topped out at
    # 0.62 -- a wide, comfortable gap. 0.85 sits safely between the two,
    # confirmed via that same real measurement, not chosen blind.
    y_floor_px = header_y_max
    band_height = max(header_y_max - min(b[1] for t, b, s in zip(texts, boxes, scores) if t in header_x), 15.0)
    for _ in range(3):  # at most 3 stacked decoy sub-header lines
        covered, band_bottom = _band_header_coverage(y_floor_px, y_floor_px + band_height * 1.3)
        if covered < len(header_xs_sorted) * 0.85:
            break
        y_floor_px = band_bottom

    candidates: list[tuple[int, float, float]] = []  # (qty, x_center, y_frac)
    for t, b, s in zip(texts, boxes, scores):
        # Center, not box TOP, against y_floor_px -- confirmed necessary
        # 2026-09-02 on sample 3-scanned.jpg: a real quantity mark's own box
        # top can land exactly ON y_floor_px when a row starts immediately
        # after the decoy row with no visual gap (a tall handwritten-digit
        # box straddling that boundary), silently dropping the one mark
        # whose top happened to tie the floor (box [646,364,674,393] vs.
        # y_floor_px=364 -- excluded by a strict "<=" even though its
        # center, 378.5, is comfortably inside the real data row). Using the
        # center matches how every other position check in this function
        # already treats a box (decoy-band coverage, y_frac, row-clustering).
        if (b[1] + b[3]) / 2 <= y_floor_px or s < 0.5:
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

    # Supplementary low-confidence recovery pass, 2026-08-20, RESCOPED
    # 2026-09-03 to a targeted per-box re-OCR instead of a second whole-page
    # OCR call. Originally: CLAHE-boost the WHOLE image, then re-run OCR
    # over it entirely, to add extra low-score digit candidates the primary
    # (score>=0.5) pass above missed. Measured cost (real timing, not
    # estimated): ~2-4s per image, unconditionally, on every single image --
    # and across this project's 3 core regression forms, it recovered a real
    # value on exactly 1 of 3 (sample 12-scanned's "ESSA Premium" row, a
    # faint "3" PaddleOCR misread as the CJK character for "two", '二',
    # score 0.25) and 0 new candidates on the other 2, so most of that cost
    # was pure overhead. Confirmed directly (by inspecting the primary
    # pass's own raw detections) that the ONE real case was already picked
    # up as a bounding box by the primary pass -- just misrecognized, not
    # undetected -- so re-scanning only the SPECIFIC low-confidence boxes
    # the primary pass already found in the data band (rather than the
    # entire page) preserves the same recovery mechanism at a fraction of
    # the cost: a form with nothing ambiguous there (the common case) now
    # costs one Python loop over already-in-memory data, not a second OCR
    # inference pass. Confirmed via the same 3 regression forms after this
    # change: sample 12-scanned still recovers the exact same digit,
    # sample 5/13-scanned are now near-zero-cost here (0-1 tiny crop
    # attempts instead of a full-page pass) with byte-identical output.
    #
    # Dash/blank marks on this form are sometimes read by PaddleOCR as the
    # CJK character for "one" ('一', a single horizontal stroke visually
    # close to a handwritten dash) instead of a literal "-", and -- since
    # that's a genuine ambiguity in the mark's shape, not a CLAHE artifact --
    # confirmed present on the UNMODIFIED image too: the same physical dash,
    # in the same column, reads as '一' at usable confidence (0.97, 0.60,
    # 0.43) on three rows, but as a spurious DIGIT-shaped misread ('2' at
    # 0.33) on a fourth. Excluding any low-score digit candidate that shares
    # a column with an independently-detected '一'/'-' elsewhere on this
    # form (computed from the PRIMARY pass, same stable coordinate space as
    # y_floor_px) stops that misread from being recovered as a fake
    # quantity by the lowered floor below.
    dash_xs = [(b[0] + b[2]) / 2 for t, b, s in zip(texts, boxes, scores)
               if b[1] > y_floor_px and s >= 0.3 and t in ("一", "-")]

    LOW_SCORE_FLOOR = 0.2
    CROP_UPSCALE = 3
    # Same data-band bounds the primary pass itself uses (x_floor..x_ceiling,
    # below y_floor_px) -- a box already accepted as a candidate above
    # (score>=0.5, digit or digit-lookalike) needs no re-examination; only a
    # genuinely uncertain (score<0.5) detection is worth the extra look, and
    # only if it isn't already a recognized blank/dash marker (recovering a
    # "digit" out of a known-blank cell would be a false positive, not a
    # recovery).
    ambiguous_boxes = [
        (t, b, s) for t, b, s in zip(texts, boxes, scores)
        if s < 0.5 and t not in ("一", "-", "")
        and (b[1] + b[3]) / 2 > y_floor_px
        and x_floor <= (b[0] + b[2]) / 2 <= x_ceiling
    ]
    _clahe_new_candidates = 0
    if ambiguous_boxes:
        # CLAHE needs whole-page context to be effective, confirmed by direct
        # testing (2026-09-03): applying it to a small crop in isolation
        # (this block's first design) computes local contrast statistics
        # over far too small an area (CLAHE's 8x8 tile grid divides a tiny
        # crop into ~10px tiles) and produced a WORSE reading than doing
        # nothing (the confirmed sample-12-scanned recovery case read as a
        # DIFFERENT wrong character, '二' -> '了', not fixed) -- so CLAHE
        # itself still runs once on the whole page (~0.5-0.9s, unavoidable
        # for correct results), but the expensive step this optimization
        # actually targets, OCR's text detection+recognition, now only runs
        # on small crops of the already-enhanced page instead of the whole
        # thing. Row spacing on a dense form can be under 30px, so the crop
        # around each ambiguous box is kept TIGHT (confirmed necessary: a
        # generous ~1.5x-box-height pad bridged into the next row's own
        # digits and matched the wrong one) -- just enough margin for
        # PaddleOCR's detector to segment the character cleanly, not enough
        # to pull in a neighboring row or column.
        try:
            clahe_bytes = preprocess_for_vlm(original_bytes)
            clahe_image = Image.open(io.BytesIO(clahe_bytes)).convert("RGB")
        except Exception:
            clahe_image = None
        if clahe_image is not None:
            for t, b, s in ambiguous_boxes:
                box_h, box_w = b[3] - b[1], b[2] - b[0]
                pad_y, pad_x = max(6.0, box_h * 0.2), max(6.0, box_w * 0.4)
                left, top = max(0, int(b[0] - pad_x)), max(0, int(b[1] - pad_y))
                right, bottom = min(width, int(b[2] + pad_x)), min(height, int(b[3] + pad_y))
                if right - left < 6 or bottom - top < 6:
                    continue
                target_cx, target_cy = (b[0] + b[2]) / 2 - left, (b[1] + b[3]) / 2 - top
                try:
                    crop = clahe_image.crop((left, top, right, bottom))
                    crop = crop.resize((crop.width * CROP_UPSCALE, crop.height * CROP_UPSCALE), Image.LANCZOS)
                    arr = np.array(crop)[:, :, ::-1]
                    crop_result = list(ocr.predict(arr))
                except Exception:
                    continue
                if not crop_result:
                    continue
                # Pick whichever detection sits nearest the original
                # ambiguous box's own position (in this crop's local
                # coordinates), not just the highest-scoring digit found
                # anywhere in the crop -- confirmed necessary: even a tight
                # crop can contain more than one detection, and the
                # highest-scoring one isn't always the one at the target
                # position.
                best: tuple[str, float, float] | None = None  # (text, score, dist)
                for ct, cb, cs in zip(crop_result[0]["rec_texts"], crop_result[0]["rec_boxes"].tolist(), crop_result[0]["rec_scores"]):
                    if not (LOW_SCORE_FLOOR <= cs < 1.0 and ct.isdigit()):
                        continue
                    ccx, ccy = (cb[0] + cb[2]) / 2 / CROP_UPSCALE, (cb[1] + cb[3]) / 2 / CROP_UPSCALE
                    dist = ((ccx - target_cx) ** 2 + (ccy - target_cy) ** 2) ** 0.5
                    if best is None or dist < best[2]:
                        best = (ct, cs, dist)
                if best is None or best[2] > max(box_h, box_w):
                    continue
                qty = int(best[0])
                if qty <= 0 or qty > 500:
                    continue
                xc, yc = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if any(abs(xc - dx) <= spacing * 0.4 for dx in dash_xs):
                    continue
                # Skip if this is very likely the SAME physical mark the
                # primary pass already found (possibly at a different
                # score/box) -- only meant to add marks the primary pass
                # missed entirely.
                if any(abs(xc - cx) <= 15 and abs(yc - cy * height) <= 15 for _q, cx, cy in candidates):
                    continue
                candidates.append((qty, xc, yc / height))
                _clahe_new_candidates += 1
        print(f"  faint-digit recovery: checked {len(ambiguous_boxes)} low-confidence detection(s), recovered {_clahe_new_candidates} new candidate(s).")

    # Duplicate-header-label exclusion, 2026-08-20 -- confirmed necessary on
    # sample 5-scanned.jpg: Fairlady Print's handwritten "105" size-label
    # (written above its own quantity, for the unprinted-extra-column
    # override documented elsewhere in this file) got detected as its OWN
    # candidate quantity mark, sitting almost exactly at the "105" header's
    # own x-position (1px off, in the confirmed case), corrupting Image
    # FCD's row with a spurious {"105": 105} cell. Only excludes a candidate
    # when BOTH its value equals some header's own number AND it sits
    # tightly at that SAME header's own x -- a normal quantity (a small
    # count near some column) never matches on value, so this can't misfire
    # on real data; the rare case of a genuine quantity coincidentally equal
    # to its own column's header number gets dropped rather than kept,
    # matching this project's established "leave it out rather than risk a
    # wrong value" default.
    DUP_LABEL_TOLERANCE_PX = 15.0
    candidates = [c for c in candidates
                  if not (str(c[0]) in header_x and abs(c[1] - header_x[str(c[0])]) <= DUP_LABEL_TOLERANCE_PX)]

    # Recognize a genuinely NEW, unprinted overflow column (e.g. "110" on
    # sample 5.jpeg's Fairlady rows -- a real, DB-confirmed valid size for
    # this business's products, past this form's own last printed header)
    # from its stacked label-over-quantity SHAPE, not by asking any model to
    # notice it. The same idea (known_numeric_sizes(), the business's real
    # catalog size vocabulary) was already tried as VLM PROMPT grounding
    # three separate ways in the 2026-08-13/17 sessions and failed
    # identically every time -- mistral never once reported a value past
    # the printed header even with the real catalog list spelled out (see
    # CLAUDE.md's "settled vision-attention ceiling" verdict). But mistral
    # isn't involved in reading these tokens in the hybrid pipeline at all
    # -- PaddleOCR already detects both the label and quantity text
    # correctly (confirmed directly on this same form's "105" case above)
    # -- so this is a code-side pattern match, not the vision problem that
    # kept failing.
    #
    # Deliberately narrow, per the lesson from the SAME DAY's adaptive-
    # threshold regression (a fix tuned to one form's failure broke a
    # different form): only scans the small margin PAST x_ceiling (which
    # normally excludes the trailing Total/Rate column entirely), and only
    # accepts a token there when a SECOND token stacks directly below it at
    # nearly the same x -- a lone Total/Rate number has no such stacked
    # companion, so this can't misfire on that column into treating a
    # circled total as a new size header.
    # Computed here (earlier than its other use further below, for row-
    # clustering) so the overflow-column density guard just below can use it
    # too -- a plain row-index -> expected-y-center lookup, doesn't depend on
    # anything computed in between.
    item_centers = [(it.row_top_frac + it.row_bottom_frac) / 2 for it in extracted.items]
    item_centers = _recover_degenerate_item_centers(
        item_centers, extracted, texts, boxes, scores, x_floor, y_floor_px, height
    )

    try:
        known_sizes = set(brandlist_match.known_numeric_sizes())
    except Exception:
        known_sizes = set()
    if known_sizes:
        overflow_margin_px = spacing * 1.2
        new_overflow_headers: dict[str, tuple[float, float]] = {}  # label_val -> (xc, label_box_bottom)
        for t, b, s in zip(texts, boxes, scores):
            if s < 0.5 or not t.isdigit() or t in header_x or b[1] <= y_floor_px:
                continue
            xc = (b[0] + b[2]) / 2
            if not (x_ceiling < xc <= x_ceiling + overflow_margin_px):
                continue
            label_val = int(t)
            if label_val not in known_sizes:
                continue
            yc = (b[1] + b[3]) / 2
            companion = next(
                ((t2, b2) for t2, b2, s2 in zip(texts, boxes, scores)
                 if s2 >= 0.5 and t2.isdigit()
                 and abs((b2[0] + b2[2]) / 2 - xc) <= spacing * 0.3
                 and 0 < (b2[1] + b2[3]) / 2 - yc <= band_height * 1.5),
                None,
            )
            if companion is None:
                continue

            # Reject a column that looks like a densely-populated PRINTED
            # per-row column (e.g. a "Total Dozen"/"Grand Total" tally
            # sitting just past the last real size header) rather than a
            # rare handwritten override label -- confirmed necessary
            # 2026-08-21 on sample 3-scanned.jpg: several per-row printed
            # total VALUES (30, 35, ...) individually happen to be valid
            # known_numeric_sizes() members AND sit close enough together
            # (same column region, one per row) to also satisfy the
            # label+companion shape above, corrupting header detection with
            # fake "35"/"30" size columns -- which then merged 5 unrelated
            # rows' candidates into one cluster (too many marks for any real
            # row) and silently dropped them all. A genuine handwritten
            # override label (confirmed on sample 5.jpeg's "110") is written
            # ONCE and only ever has detections on the few rows that
            # actually share it. Reuses the exact same "sparse real data vs.
            # dense printed row" signal already used for decoy-sub-header-
            # row detection above, just applied along the other axis: count
            # how many of the table's OWN rows (via item_centers, tolerant
            # of this column's natural position drift, not a tight pixel
            # window) have ANY digit nearby this x -- a real printed column
            # hits most rows, a rare override hits only the few sharing it.
            rows_with_nearby_digit = sum(
                1 for ic in item_centers
                if any(s3 >= 0.5 and t3.isdigit()
                       and abs((b3[1] + b3[3]) / 2 - ic * height) <= band_height
                       and abs((b3[0] + b3[2]) / 2 - xc) <= spacing
                       for t3, b3, s3 in zip(texts, boxes, scores))
            )
            if item_centers and rows_with_nearby_digit > len(item_centers) * 0.5:
                continue

            t2, b2 = companion
            qty2 = int(t2)
            if qty2 <= 0 or qty2 > 500:
                continue
            xc2 = (b2[0] + b2[2]) / 2
            yc2 = (b2[1] + b2[3]) / 2
            header_x[str(label_val)] = xc
            new_overflow_headers[str(label_val)] = (xc, b[3])
            candidates.append((qty2, xc2, yc2 / height))

        # A handwritten overflow-column label is often written ONCE and
        # implicitly shared by every row below it, not re-written per row --
        # confirmed directly on sample 5-scanned.jpg: Fairlady Print and
        # Fairlady Plain sit in consecutive rows sharing a single "105"/"110"
        # label pair written only above Fairlady Print (the first of the
        # two); Fairlady Plain has its own quantity in that same column but
        # no label of its own, so the label+immediate-companion scan above
        # only ever recovers ONE row's value per overflow column, silently
        # missing every other row that shares it (this is also the user's
        # own suspicion about sample 3.jpeg's last 4 rows sharing one label
        # above the first of the four -- the same pattern, not yet tested
        # there). Once a new overflow column's x-position is confirmed, treat
        # it like any other known header column for the REST of the table:
        # scan the same narrow x-band, across every row below y_floor_px, for
        # additional digit-shaped candidates the label-adjacency check above
        # can't reach because they have no label of their own. Deliberately
        # tight x-tolerance (spacing*0.3, same as the label-companion match
        # above) so this can't sweep in an unrelated column's marks.
        for label_val, (xc, label_bottom) in new_overflow_headers.items():
            for t, b, s in zip(texts, boxes, scores):
                # Strictly below the LABEL's own box (not just y_floor_px,
                # the printed header row's bottom) -- otherwise this scan
                # re-detects the overflow label token itself (e.g. "110",
                # itself a plausible-looking small quantity, <=500) as a
                # spurious candidate. Confirmed necessary: an earlier version
                # using only the y_floor_px bound double-counted the label as
                # a quantity in Fairlady Print's own row before this guard
                # was added.
                if s < 0.5 or b[1] <= label_bottom:
                    continue
                xc2 = (b[0] + b[2]) / 2
                if abs(xc2 - xc) > spacing * 0.3:
                    continue
                qty = int(t) if t.isdigit() else _try_digit_correct(t)
                if qty is None or qty <= 0 or qty > 500:
                    continue
                yc2 = (b[1] + b[3]) / 2
                if any(abs(xc2 - cx) <= 15 and abs(yc2 - cy * height) <= 15 for _q, cx, cy in candidates):
                    continue
                candidates.append((qty, xc2, yc2 / height))

    if not candidates:
        print("  hybrid OCR+VLM quantity read: OCR found no quantity-shaped marks in the table area.")
        return {}, {}

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
    # systematic error. (item_centers itself is computed earlier above, so
    # the overflow-column density guard can reuse it too.)

    # Gap threshold for row-clustering, ADAPTIVE to this form's own row
    # density, not a fixed 20px constant -- confirmed necessary 2026-08-20
    # on sample 13-scanned.jpg (a dense, machine-typed 20-row grid, ~20.6px/
    # row average): the old fixed 20px threshold couldn't tell a gap BETWEEN
    # two rows from a gap WITHIN one row on a form this tightly spaced, and
    # collapsed all 138 detected marks across the entire table into ONE
    # cluster (matched to a single item, every other item left with no
    # marks at all -- 0/20 rows helped). item_centers's absolute positions
    # aren't trustworthy (see _row_order_plausible above -- this form's own
    # sequence is a mechanically uniform partition, not real measurement),
    # but the SPACING between consecutive centers is a reasonable density
    # estimate regardless: a formulaic table_height/n_items division still
    # roughly reflects true row density even when the absolute positions it
    # is anchored to are off. Floored so a degenerate or wildly off
    # item_centers spacing can't produce a near-zero threshold (splits one
    # row's own digits into fake separate rows).
    #
    # Capped at the OLD fixed 20px value, not something larger -- confirmed
    # necessary 2026-08-20 after a real regression on sample 5-scanned.jpg
    # (an ESSA-family form, sparser row spacing than sample 13): an earlier
    # version of this cap (60px) let the threshold grow past 20px on this
    # form, and a stray handwritten size-label mark (Fairlady Print's
    # unprinted "105" override, sitting between Image FCD's and Fairlady
    # Print's real marks) bridged two rows' clusters into one 11-mark blob
    # -- at the OLD 20px threshold the second of the two gaps involved
    # (26.5px) would NOT have merged, keeping the rows separate. So this
    # threshold may only ever SHRINK below 20px (for a denser form like
    # sample 13 that genuinely needs it), never grow past what was already
    # proven safe on the handwritten ESSA-family forms.
    if len(item_centers) >= 2:
        avg_spacing_frac = (max(item_centers) - min(item_centers)) / (len(item_centers) - 1)
    else:
        avg_spacing_frac = 20.0 / height
    gap_threshold_frac = max(min(avg_spacing_frac * 0.6, 20.0 / height), 8.0 / height)

    sorted_candidates = sorted(candidates, key=lambda c: c[2])
    y_clusters: list[list[tuple[int, float, float]]] = []
    for cand in sorted_candidates:
        if y_clusters and cand[2] - y_clusters[-1][-1][2] <= gap_threshold_frac:
            y_clusters[-1].append(cand)
        else:
            y_clusters.append([cand])

    # Match clusters to items via a joint, order-preserving assignment
    # rather than independent nearest-center matching per cluster.
    # Confirmed necessary 2026-08-17: independent nearest-center matching
    # still got row 1 wrong even after fixing the decoy-row contamination
    # above -- item 0's own row_top_frac/row_bottom_frac center was itself
    # biased early enough that item 1's center was numerically CLOSER to
    # row 1's real data cluster than item 0's own center was, so the naive
    # per-cluster nearest match picked the wrong item.
    #
    # _realign_row_clusters_by_total (not plain _monotonic_assign) --
    # confirmed necessary on sample 3-scanned.jpg: pure y-proximity has no
    # way to tell "a row's cluster is genuinely missing" apart from "this
    # row's own center estimate is just a little off," so when even one
    # physical row's marks go undetected (see that function's own
    # docstring), every item after it silently inherits the next row's
    # cluster instead -- a cascading off-by-one that plain _monotonic_assign
    # (still used as-is for the column/header axis just below, which has no
    # equivalent "missing row" failure mode) can't distinguish from a
    # correct fit. See that function's docstring for why this uses each
    # row's own printed total as a second, independent signal rather than
    # trying to fix item_centers itself.
    row_candidates: dict[int, list[tuple[int, float]]] = {}
    voidable: set[int] = set()
    if 0 < len(y_clusters) <= len(item_centers):
        assignment, voidable = _realign_row_clusters_by_total(y_clusters, item_centers, extracted.items)
        for cluster_idx, item_idx in assignment.items():
            row_candidates[item_idx] = [(c[0], c[1]) for c in y_clusters[cluster_idx]]

    if not row_candidates:
        print("  hybrid OCR+VLM quantity read: no OCR-detected marks matched to any row.")
        return {}, {}

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
                # 2026-09-02: the VLM's own pre-reconciliation reading, snapshotted
                # here specifically because extract_one() mutates
                # extracted.items[i].quantities IN PLACE with this function's own
                # output before <name>.raw.json is written -- so raw.json never
                # reflects the VLM-alone reading, and this is the only place that
                # ever does. Needed to audit/replay _reconcile_hybrid_with_vlm
                # without spending a fresh live call every time.
                "vlm_quantities": {
                    str(i): {qp.size: qp.quantity for qp in it.quantities}
                    for i, it in enumerate(extracted.items)
                },
            }
            (outdir / f"{image_path.stem}.hybrid_debug.json").write_text(json.dumps(debug, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass  # debug-only, never let this block the real extraction

    # ---- detection-coverage guard -------------------------------------
    #
    # Everything below assumes OCR actually SAW this page's handwriting, and
    # that a disagreement with the VLM therefore means the VLM drifted. On a
    # page where PaddleOCR simply can't detect the marks, that assumption
    # inverts: _reconcile_hybrid_with_vlm keeps the OCR reading whenever the
    # two disagree on which columns are filled, so near-total blindness gets
    # treated as near-total authority and wipes out a good VLM reading.
    #
    # Confirmed on sample 4-scanned.jpg (2026-09-03), a form whose cells are
    # almost all a single "1" -- a lone vertical stroke, the hardest possible
    # glyph for a text DETECTOR (not recognizer) to find at all. OCR found 13
    # marks on a page with ~60 filled cells, and two of those 13 were
    # adjacent 1s merged into "111". The result: 8 of 12 rows overwritten
    # with one-or-two-cell readings, "111" quantities, and row totals that
    # contradicted the form's own printed totals. Measured against those
    # printed totals, on a fresh CLI run:
    #     hybrid ON  ->  2/11 rows correct
    #     hybrid OFF ->  7/11 rows correct
    #
    # Detection coverage separates the two situations with a wide margin --
    # measured over every form in this project's test set:
    #     sample 5-scanned   71 marks / 66 VLM cells = 108%
    #     sample 13-scanned 138 marks / 135 VLM cells = 102%
    #     sample 4-scanned   13 marks /  60 VLM cells =  22%
    # (healthy pages come out at or slightly above 100%, since OCR also picks
    # up marks outside the VLM's own reading.) MIN_OCR_COVERAGE sits far
    # below every healthy figure and far above the broken one.
    #
    # Standing down is the whole correction, not a partial one: a page this
    # sparse gives no per-row basis for deciding WHICH rows to trust, so the
    # VLM's reading stands everywhere and every non-empty row is flagged for
    # a human instead.
    vlm_cell_count = sum(len(it.quantities) for it in extracted.items)
    if vlm_cell_count and len(candidates) < vlm_cell_count * MIN_OCR_COVERAGE:
        print(f"  hybrid OCR+VLM quantity read: OCR detected only {len(candidates)} marks for "
              f"{vlm_cell_count} quantity cells ({len(candidates) / vlm_cell_count:.0%} coverage) -- "
              f"too little of this page's handwriting was found to correct anything with. "
              f"Kept the model's own reading for every row and flagged them for review.")
        stand_down_flags = {
            i: {
                "status": "unverified",
                "sizes": [],
                "note": (f"Hybrid OCR+VLM: the OCR pass found only {len(candidates)} handwritten marks on a page "
                         f"with {vlm_cell_count} filled cells, so it could not verify anything. This row is the "
                         f"model's own reading, unchecked -- compare it against the photo."),
            }
            for i, it in enumerate(extracted.items) if it.quantities
        }
        return {}, stand_down_flags

    # Group each row's marks to headers by CODE, not by asking the VLM to
    # reproduce the arithmetic in text -- confirmed necessary 2026-08-20.
    # The prior VLM-grouping step was already given exact ground-truth
    # coordinates for both headers and marks (nothing left to "read" from
    # the image), yet a real run on sample 12-scanned.jpg still silently
    # dropped one correctly-detected mark from its own text reply (Ladies
    # Drawers: 4 real marks given, only 3 came back) -- a pure text-
    # generation failure on a task that's just nearest-neighbor matching on
    # numbers already in hand. Reusing the same order-preserving DP
    # (_monotonic_assign) already proven for digit-to-header assignment in
    # ocr_cell_read.py removes this failure mode entirely for any row whose
    # marks were genuinely detected, and is strictly more predictable than
    # a model call for arithmetic it's already been handed pre-computed.
    header_items_sorted = sorted(header_x.items(), key=lambda kv: kv[1])
    header_xs_sorted_list = [hx for _, hx in header_items_sorted]
    ASSIGN_COST_LIMIT = spacing * 0.6  # a loose fit means at least one mark landed on the wrong header -- don't trust the row

    quantities_by_index: dict[int, dict[str, int]] = {}
    for idx, marks in row_candidates.items():
        marks_sorted = sorted(marks, key=lambda m: m[1])
        mark_xs = [x for _, x in marks_sorted]
        if len(mark_xs) > len(header_xs_sorted_list):
            continue  # more marks than headers can't be a real reading -- OCR noise
        assignment, avg_cost = _monotonic_assign(mark_xs, header_xs_sorted_list)
        if avg_cost > ASSIGN_COST_LIMIT:
            continue
        quantities_by_index[idx] = {
            header_items_sorted[header_idx][0]: marks_sorted[mark_idx][0]
            for mark_idx, header_idx in assignment.items()
        }

    # Voidable items (see _realign_row_clusters_by_total's own docstring)
    # that still didn't end up with a cluster -- the overwhelmingly common
    # outcome, since deprioritization means one only gets assigned when a
    # real cluster is left over with nowhere better to go -- are forced to
    # explicit empty quantities here, rather than left absent. "Absent"
    # would mean _hybrid_ocr_quantities returns nothing for this index, and
    # the caller's merge (extract_one()) falls back to the main call's own
    # reading for it -- exactly the bug this exists to fix, since that
    # fallback is what showed fabricated quantities for a struck-through
    # row with a blank printed total in the first place.
    for idx in voidable:
        if idx not in quantities_by_index:
            quantities_by_index[idx] = {}

    # Letter-coded size columns shared across a group of rows -- 2026-08-21.
    # Structurally the SAME "label written once, shared downward" pattern as
    # the numeric overflow-column fix above, confirmed via direct OCR +
    # pixel measurement on sample 3-scanned.jpg: MM K4532's row has S/M/L/
    # XL/XXL labels stacked directly above its own quantities, in the SAME 5
    # columns as this form's own printed 45/50/55/60/65 headers -- NOT a new
    # overflow column past x_ceiling (unlike the numeric case above), these
    # letter labels sit INSIDE the normal header range, overlapping columns
    # real numeric rows elsewhere on the same form also legitimately use
    # (e.g. B-509's row). Confirmed real: 3 rows below MM K4532 (no labels of
    # their own) each sum EXACTLY to their own printed Total Dozen once their
    # already-assigned "45/50/55/60/65" quantities are re-read as
    # S/M/L/XL/XXL instead. Because these columns overlap real numeric
    # headers used elsewhere, propagation must be scoped tightly -- only
    # CONSECUTIVE rows immediately following an explicit label row, and only
    # while a row's own already-assigned candidate x's stay entirely within
    # the label columns (a genuine wider numeric row breaks containment and
    # stops the group). Deliberately does not resolve the letter to a real
    # catalog number here -- returns letter-coded keys (e.g. "XL") the same
    # way extract_claude.py's own LETTER SIZES prompt rule does, so
    # brandlist_match.annotate_and_resolve() (already wired into main()'s
    # post-processing) converts them once the row's product match is known,
    # with no new resolution logic needed here.
    LETTER_SIZE_TOKENS = {"XS", "S", "M", "L", "XL", "XXL", "2XL", "3XL", "4XL"}
    letter_labels: list[tuple[str, float, float]] = []  # (letter, x, y) -- label's own position
    for t, b, sc in zip(texts, boxes, scores):
        tt = t.strip().upper()
        if sc < 0.5 or tt not in LETTER_SIZE_TOKENS or b[1] <= y_floor_px:
            continue
        xc = (b[0] + b[2]) / 2
        if not (x_floor <= xc <= x_ceiling):
            continue
        yc = (b[1] + b[3]) / 2
        companion = next(
            ((t2, b2) for t2, b2, s2 in zip(texts, boxes, scores)
             if s2 >= 0.5 and t2.isdigit()
             and abs((b2[0] + b2[2]) / 2 - xc) <= spacing * 0.3
             and 0 < (b2[1] + b2[3]) / 2 - yc <= band_height * 1.5),
            None,
        )
        if companion is None:
            continue
        letter_labels.append((tt, xc, yc))

    if letter_labels:
        by_anchor: dict[int, dict[str, float]] = {}
        anchor_label_top_y: dict[int, float] = {}  # anchor_idx -> topmost confirmed label's own y-center
        # Row-height tolerance for preferring a VLM-hinted anchor below --
        # reuses the same avg_spacing_frac already computed above for the
        # row-clustering gap threshold (this form's own typical row height
        # as a fraction of image height), not a fresh constant.
        _hint_tolerance_frac = avg_spacing_frac * 1.2
        for letter, xc, yc in letter_labels:
            yf = yc / height
            naive_idx = min(range(len(item_centers)), key=lambda i: abs(item_centers[i] - yf))
            anchor_idx = naive_idx
            # Prefer a VLM-hinted row over the naive nearest-center pick, but
            # ONLY when the hinted row is still within about one row's own
            # height of this token's real y -- confirmed necessary
            # 2026-09-02 on sample 3-scanned.jpg: an unrestricted "always
            # prefer the hint" version dragged in unrelated letter-shaped OCR
            # noise from far-away rows (a couple of misread "5"s elsewhere on
            # the page) onto the one hinted anchor just because it was the
            # only allowed candidate. Gating by distance keeps the real win
            # -- a genuine letter token ("XL") sitting inside MM K4532's own,
            # content-matching-CONFIRMED row got misassigned to item 10
            # instead under plain nearest-center matching, because item 10's
            # own center is a weighted-division ESTIMATE (no distinctive
            # digit-code/word for _find_item_name_y to content-match against,
            # unlike 15-19's real matches) that happened to land numerically
            # closer to this token's y than item 16's true position -- the
            # same "an estimate is not real evidence" gap already documented
            # for _recover_degenerate_item_centers's own fallback path, here
            # corrupting anchor selection rather than row-clustering.
            if letter_size_hints:
                hinted = [i for i in letter_size_hints if i < len(item_centers)]
                if hinted:
                    hint_idx = min(hinted, key=lambda i: abs(item_centers[i] - yf))
                    if abs(item_centers[hint_idx] - yf) <= _hint_tolerance_frac:
                        anchor_idx = hint_idx
            by_anchor.setdefault(anchor_idx, {})[letter] = xc
            anchor_label_top_y[anchor_idx] = min(anchor_label_top_y.get(anchor_idx, yc), yc)

        col_tolerance = spacing * 0.4

        # VLM-guided digit recovery for a row mistral has directly flagged
        # as letter-sized (MistralExtractedItem.letter_sizes) -- confirmed
        # real and necessary on sample 3-scanned.jpg's MM K4532 row: the
        # whole-page OCR pass above only cleanly read 2 of its 5 letter+
        # digit column pairs ("XL"/"6", "xxC"/"6"); the other 3 (S/M/L)
        # were merged by PaddleOCR's detector into unreadable single boxes
        # (see _recover_letter_size_row_digits's own docstring for the two
        # prior crop designs that failed before landing on the current
        # per-column, y-offset-swept approach). Only runs for a row BOTH
        # flagged by the model AND already anchoring at least one confirmed
        # letter (by_anchor) -- with zero confirmed letters there is no
        # known label y-position to crop around, and no safe ordinal anchor
        # to infer the rest from (see _infer_letter_columns) either, so such
        # a row is left as-is rather than guessed at.
        for anchor_idx in sorted(letter_size_hints & by_anchor.keys()):
            if anchor_idx >= len(item_centers) or anchor_idx not in anchor_label_top_y:
                continue
            # Search only the columns spanning from the table's own left
            # edge to a few columns past the rightmost CONFIRMED letter --
            # confirmed necessary 2026-09-02 on sample 3-scanned.jpg: this
            # form has 13 header columns total, but a letter-size group only
            # ever occupies a handful of them (5, in the confirmed case) --
            # searching every column would mean 7-8 wasted per-column sweeps
            # over blank paper for no benefit.
            search_ceiling = min(x_ceiling, max(by_anchor[anchor_idx].values()) + spacing * 3)
            search_headers = [hx for hx in header_xs_sorted if x_floor <= hx <= search_ceiling]
            recovered = _recover_letter_size_row_digits(
                original_bytes, search_headers, anchor_label_top_y[anchor_idx], spacing, width, height, band_height,
            )
            if not recovered:
                continue
            existing = row_candidates.setdefault(anchor_idx, [])
            existing_xs = [x for _, x in existing]
            for hx, qty in recovered.items():
                if any(abs(hx - ex) <= 15 for ex in existing_xs):
                    continue  # already found by the whole-page pass
                existing.append((qty, hx))
                existing_xs.append(hx)

            # Safety-net numeric re-assignment, even if letter-inference
            # below can't run -- a recovered value should never be silently
            # lost.
            marks_sorted = sorted(existing, key=lambda mrk: mrk[1])
            mark_xs = [x for _, x in marks_sorted]
            if len(mark_xs) <= len(header_xs_sorted_list):
                assignment, avg_cost = _monotonic_assign(mark_xs, header_xs_sorted_list)
                if avg_cost <= ASSIGN_COST_LIMIT:
                    quantities_by_index[anchor_idx] = {
                        header_items_sorted[hi][0]: marks_sorted[mi][0] for mi, hi in assignment.items()
                    }

            group_header_xs: list[float] = []
            for _, x in existing:
                nearest_h, nearest_x = min(header_x.items(), key=lambda kv: abs(kv[1] - x))
                if abs(nearest_x - x) <= col_tolerance and nearest_x not in group_header_xs:
                    group_header_xs.append(nearest_x)
            inferred = _infer_letter_columns(group_header_xs, by_anchor[anchor_idx], col_tolerance)
            if inferred is not None:
                by_anchor[anchor_idx] = inferred

        for anchor_idx, letter_columns in by_anchor.items():
            # Walk forward from the anchor row while every row's own
            # already-assigned candidate x's stay entirely within the label
            # columns -- a genuine numeric row (wider, or using different
            # columns) breaks containment and ends the group. Requiring the
            # group to reach at least 2 rows (anchor + 1) is the corroborating
            # signal a lone, possibly OCR-noisy label token can't provide by
            # itself -- real handwritten letter labels on this form's photos
            # are noisy enough that requiring 2+ DISTINCT letters (a cleaner
            # bar) turned out too strict to even fire on the confirmed real
            # case (only "XL" OCR'd cleanly enough to exact-match; "XXL" read
            # as "xxC", S/M/L unreadable at low score) -- multi-row
            # containment is the more reliable signal here instead.
            group_indices: list[int] = []
            idx = anchor_idx
            while idx in row_candidates:
                xs = [x for _, x in row_candidates[idx]]
                if not xs or not all(any(abs(x - lx) <= col_tolerance for lx in letter_columns.values()) for x in xs):
                    break
                group_indices.append(idx)
                idx += 1
            if len(group_indices) < 2:
                continue
            for gidx in group_indices:
                relabeled: dict[str, int] = {}
                for qty, x in sorted(row_candidates[gidx], key=lambda m: m[1]):
                    nearest_letter = min(letter_columns, key=lambda l: abs(letter_columns[l] - x))
                    if abs(letter_columns[nearest_letter] - x) > col_tolerance:
                        continue
                    relabeled[nearest_letter] = qty
                if relabeled:
                    quantities_by_index[gidx] = relabeled

    used = len(quantities_by_index)
    voided = sum(1 for idx in voidable if idx in quantities_by_index)
    # skipped_low_confidence counts only real row_candidates entries that
    # didn't make it into quantities_by_index -- voided items are counted
    # separately since they're a deliberate, confident empty result, not a
    # low-confidence skip, and weren't in row_candidates to begin with.
    skipped_low_confidence = sum(1 for idx in row_candidates if idx not in quantities_by_index)
    print(f"  hybrid OCR quantity read: used for {used}/{n_items} rows "
          f"({n_items - len(row_candidates) - voided} row(s) had no OCR-detected marks in range, "
          f"{skipped_low_confidence} row(s) had marks but the header assignment was too "
          f"low-confidence to trust -- both kept the model's own reading instead; "
          f"{voided} row(s) forced to empty as void/struck-out).")

    # Total-checksum flagging (2026-08-19, converted to a per-cell flag
    # 2026-08-22) -- see _flag_hybrid_total_mismatch's own docstring for why
    # this flags rather than auto-corrects: a shift-based auto-correction
    # against the printed total was tried first and confirmed mathematically
    # incapable of ever firing (relabeling a row's header keys can't change
    # its sum), so this only surfaces a mismatch for human review instead.
    # Flags the WHOLE row (every size this row's hybrid read reported) since
    # a sum mismatch alone doesn't pin down which single cell is wrong.
    #
    # Severity is scaled by the SIZE of the mismatch, not treated as one
    # binary signal -- confirmed necessary 2026-08-22 on a real
    # sample 3-scanned.jpg run: 16 of 20 rows got flagged this way in one
    # pass, with diffs ranging from -1 to +35, all colored identically
    # ("unresolved", red, "most important to check"). Per this project's
    # own repeatedly-confirmed history (e.g. the "FASTASTIC COLLAR" and
    # "MM Looper 4289" cases elsewhere in this file/CLAUDE.md), a small
    # (+-2) gap is almost always the MODEL misreading its own single-digit
    # printed total, not a real missing/extra quantity -- burying that
    # near-certainly-fine majority in the same red bucket as a genuine
    # 22-unit gap defeats the entire point of flagging (helping a reviewer
    # prioritize). Small gaps downgrade to "unverified" (gray, worth a
    # glance) instead.
    # 2026-09-02: each row is now reconciled against the VLM's OWN main-call
    # reading for that row (_reconcile_hybrid_with_vlm) before the checksum
    # check below runs -- previously this loop trusted hybrid's map outright,
    # a hierarchy CLAUDE.md's own dated findings confirmed was the wrong
    # shape (hybrid and the VLM fail in each other's strengths, not the same
    # places). extracted.items[idx].quantities is still the VLM's untouched
    # reading at this point -- the caller (extract_one()) doesn't overwrite it
    # until after this function returns.
    size_headers_set = set(extracted.size_headers)
    flags: dict[int, dict] = {}
    for idx, qty in quantities_by_index.items():
        printed_total = _parse_int_or_none(extracted.items[idx].row_total)
        vlm_map = {qp.size: qp.quantity for qp in extracted.items[idx].quantities}
        merged, conflict_flag = _reconcile_hybrid_with_vlm(vlm_map, qty, printed_total, size_headers_set)
        quantities_by_index[idx] = merged
        if conflict_flag is not None:
            # A genuine source disagreement is the more specific signal --
            # don't also run the total-mismatch check on top of it.
            flags[idx] = conflict_flag
            continue
        flag_text = _flag_hybrid_total_mismatch(merged, printed_total)
        if flag_text is not None:
            diff = sum(merged.values()) - printed_total
            status = "unresolved" if abs(diff) > 2 else "unverified"
            flags[idx] = {
                "status": status,
                "sizes": sorted(merged.keys(), key=lambda s: int(s) if s.isdigit() else 0),
                "note": f"Hybrid OCR+VLM: {flag_text}.",
            }
    return quantities_by_index, flags


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
            _, _, item_name_ocr, _, _ = ocr_row(row_bytes, header_h_px, size_headers, score_threshold=0.5)
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


def _normalize_fused_size_headers(extracted: ExtractedForm) -> None:
    """Repairs a real, non-deterministic main-call quirk (root-caused
    2026-08-22 on sample 3-scanned.jpg, previously just documented as an
    unexplained "merged-header" flake -- see CLAUDE.md's 2026-08-21 note):
    on some draws mistral reports extracted.size_headers as "REAL/DECOY"
    fused strings (e.g. "45/18") instead of the real printed header alone.
    This ESSA-family form prints a SECOND header row directly below the
    real one -- an age/chest-equivalent number the model is otherwise told
    to ignore (see this file's/CLAUDE.md's "Header-row hallucination"
    history) -- and on this flake, instead of ignoring it, the model
    concatenates it onto the real header with a "/".

    This is NOT the OCR-side merged-token bug _split_merged_header_token
    fixes (that's PaddleOCR fusing two adjacent PRINTED digits into one
    detection); this fusion happens in the model's own structured JSON
    output, upstream of OCR entirely -- so OCR can never match "45/18"
    against a page that only prints "45", the header-count guard
    (correctly) treats that as "not enough real headers found", and the
    ENTIRE hybrid OCR pass silently skips itself for the whole image on
    every draw this hits, not just the affected columns.

    Confirmed via a real raw.json (sample 3-scanned.jpg, 2026-08-22): item
    quantities are never affected, only size_headers itself, and the FIRST
    segment is always the genuine printed header -- it exactly reproduces
    this form's real 45-105 header row, and every one of those values is a
    real, valid size in this business's own catalog
    (brandlist_match.known_numeric_sizes()), unlike most of the second
    segment's values (18/20/22/24/26/32/34/36/38/42 are NOT valid catalog
    sizes -- only 30/40 coincidentally are, so catalog membership alone
    can't cleanly discriminate the two segments for every column, but
    segment ORDER can: the model always reads the real header first, top
    to bottom, then appends the decoy row's value after it)."""
    fixed = []
    changed = False
    for h in extracted.size_headers:
        if "/" in h:
            first = h.split("/", 1)[0].strip()
            if first.isdigit():
                fixed.append(first)
                changed = True
                continue
        fixed.append(h)
    if changed:
        print(f"  main call reported fused header/decoy size_headers ({extracted.size_headers}) -- repaired to {fixed}.")
        extracted.size_headers = fixed


MAIN_CALL_MAX_RETRIES = 1  # extra attempts beyond the first, only on the zero-quantities flake below


def extract_one(client: ollama.Client, model: str, image_path: Path, outdir: Path, usage_log: Path, system_prompt: str, brandlist_available: bool = False, do_recount: bool = False, do_preprocess: bool = False, do_hybrid: bool = True, use_lighton_hybrid: bool | None = None) -> OrderForm:
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
    # VLM-confirmed letter-size row indices (mistral-only, see
    # MistralExtractedItem.letter_sizes) -- captured BEFORE
    # _mistral_form_to_extracted_form discards the mistral-only schema, and
    # threaded into _hybrid_ocr_quantities below so it can target its
    # crop-based digit-recovery pass (_recover_letter_size_row_digits) only
    # at rows the model itself has directly flagged, rather than relying
    # purely on blind whole-page OCR text matching for a rare, hard-to-read
    # S/M/L label token (see that function's docstring for why the whole-
    # page pass alone isn't enough on a cramped stacked label+digit cell).
    letter_size_hints: set[int] = set()
    # struck_out_hints: mistral's own direct "this row is crossed out"
    # judgment (see MistralExtractedItem.struck_out), threaded through to
    # lighton_hybrid_quantities the same way letter_size_hints is -- a
    # stronger, more direct signal than the blank-quantities/blank-total
    # proxy _hybrid_ocr_quantities falls back to for model-agnostic
    # struck-out detection (see _realign_row_clusters_by_total's own
    # docstring on why that proxy exists at all). Needed here specifically
    # because that proxy produced a real false positive during testing:
    # a row where mistral's main call simply flaked and returned empty
    # quantities/total for ONE run (a documented, known non-determinism,
    # not an actual strike-through) got its otherwise-correct LightOnOCR-2
    # reading discarded, even though PaddleOCR's own independent pass on
    # the exact same (non-struck-out) row found the same real values.
    struck_out_hints: set[int] = set()
    if "chandra" in model.lower():
        # Native-format path (see chandra_parser.py's own module docstring
        # for the full rationale): Chandra gets no schema and no
        # MistralExtractedForm-style flags at all, so letter_size_hints/
        # struck_out_hints stay empty here -- confirmed acceptable by
        # direct testing (HISTORY.md, 2026-09-07): Chandra's own reading
        # already comes back with empty quantities for a genuine
        # struck-out row with no special mechanism needed. Lazy import,
        # same reason hybrid_quantities_lighton's own import is lazy
        # inside this function: paid only the first time a run actually
        # takes this branch.
        from chandra_parser import call_chandra, parse_chandra_output
        for attempt in range(MAIN_CALL_MAX_RETRIES + 1):
            raw_text, error, usage = call_chandra(client, model, image_bytes)
            print(f"  usage (main, attempt {attempt + 1}): prompt={usage.get('prompt_eval_count')} eval={usage.get('eval_count')} {usage.get('duration_seconds', 0):.1f}s" + (f" ERROR: {error}" if error else ""))
            _log_usage(usage_log, image_path.name, model, f"main-attempt{attempt + 1}", usage, error)
            if error:
                continue
            try:
                extracted = parse_chandra_output(raw_text)
            except Exception as exc:
                error = f"failed to parse Chandra's native-format output: {exc}"
                continue
            break
    else:
        for attempt in range(MAIN_CALL_MAX_RETRIES + 1):
            parsed, error, usage = _call_schema(client, model, system_prompt, USER_PROMPT, [image_bytes], main_schema_cls, MAX_TOKENS)
            print(f"  usage (main, attempt {attempt + 1}): prompt={usage.get('prompt_eval_count')} eval={usage.get('eval_count')} {usage.get('duration_seconds', 0):.1f}s" + (f" ERROR: {error}" if error else ""))
            _log_usage(usage_log, image_path.name, model, f"main-attempt{attempt + 1}", usage, error)
            if error:
                continue
            if main_schema_cls is MistralExtractedForm:
                letter_size_hints = {i for i, it in enumerate(parsed.items) if it.letter_sizes}
                struck_out_hints = {i for i, it in enumerate(parsed.items) if it.struck_out}
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

    _normalize_fused_size_headers(extracted)

    # Per-template correction memory (template_learning.py): a signature
    # match here means a human has already reviewed and corrected at least
    # one earlier form with this exact seller_name + size_headers layout.
    # Looked up once, used twice below -- immediately, to pick a hybrid-
    # quantities backend this template has historically needed less
    # correction under (only when the caller expressed no preference of its
    # own); and again just before this image's own debug artifacts are
    # written, to reapply whatever specific item-name/column-shift
    # corrections a human already made on this template before. Matching
    # only happens here, AFTER the main call, deliberately -- see
    # template_learning.py's module docstring for why (seller_name/
    # size_headers don't exist before it runs).
    template_match = template_learning.lookup(extracted.seller_name, extracted.size_headers)
    if use_lighton_hybrid is None:
        preferred = template_learning.preferred_backend(template_match)
        use_lighton_hybrid = (preferred != "paddleocr")

    # 2026-08-22: bookkeeping/diagnostic text from the hybrid and recount
    # passes used to be dumped straight into extracted.notes -- a growing
    # wall of prose at the top of the review page that didn't point at any
    # specific cell. Genuinely suspect rows are now captured as structured
    # flags instead (same {"status", "sizes", "note"} shape
    # extract_claude.py's own recount pass already uses), so
    # generate_review.py highlights the actual doubtful cells directly;
    # routine status messages ("used the whole-table fallback crop",
    # "skipped (--no-recount)") go to the console instead, since they're
    # operational information for whoever ran the script, not something a
    # reviewer checking cells against a photo needs to read. extracted.notes
    # is left for genuine page-level handwritten remarks only (its
    # originally-intended purpose, per schema.py's own field description).
    hybrid_flags: dict[int, dict] = {}
    if do_hybrid and extracted.items:
        try:
            # use_lighton_hybrid (default ON as of 2026-09-05 -- see
            # --no-lighton-hybrid to fall back to the previous default):
            # swaps PaddleOCR+grid.py's pixel-position DP for a second VLM
            # call (maternion/LightOnOCR-2:1b, local Ollama) that reads its
            # own assembled table back out -- see hybrid_quantities_lighton.py's
            # module docstring for the full rationale and known gaps (an
            # imported local, not a top-level import, so this file has no
            # import-time dependency on it -- only paid the first time a run
            # actually reaches this branch). Compared against the previous
            # PaddleOCR default on identical mistral output via
            # compare_hybrid_backends.py: near-total row-for-row agreement on
            # 3 of 4 test forms (two independent mechanisms landing on the
            # same correction is real cross-validation, not coincidence) and
            # a still-being-verified divergence on the one form where
            # PaddleOCR's own coverage guard already stood down entirely
            # (sample 4-scanned, where PaddleOCR contributed nothing anyway).
            # Made default ON before that last form's rows were individually
            # re-verified against the photo -- if you're reading this while
            # investigating a regression, that's the first thing to check.
            if use_lighton_hybrid:
                from hybrid_quantities_lighton import lighton_hybrid_quantities
                hybrid_quantities, hybrid_flags = lighton_hybrid_quantities(image_path, extracted, outdir, struck_out_hints, letter_size_hints)
            else:
                hybrid_quantities, hybrid_flags = _hybrid_ocr_quantities(image_path, extracted, outdir, letter_size_hints)
            for i, qty in hybrid_quantities.items():
                extracted.items[i].quantities = [QuantityPair(size=size, quantity=q) for size, q in qty.items()]
        except Exception as exc:
            print(f"  hybrid OCR+VLM quantity read failed ({exc}) -- kept the model's own quantities for every row.")

    recount_flags: list[dict] = [hybrid_flags.get(i, {"status": "no_recount"}) for i in range(len(extracted.items))]
    if do_recount and extracted.items:
        try:
            pass_notes, recount_flags_from_pass = _recount_quantities(client, model, image_path, extracted, usage_log, brandlist_available)
            for pass_note in pass_notes:
                print(f"  recount: {pass_note}")
            # Recount's own flag wins over a hybrid total-mismatch flag for
            # the same row only when it actually has something to say --
            # "no_recount"/"ok" would otherwise silently erase a real hybrid
            # finding for a row recount didn't cover.
            for i, flag in enumerate(recount_flags_from_pass):
                if i < len(recount_flags) and flag.get("status") not in (None, "no_recount", "ok"):
                    recount_flags[i] = flag
        except Exception as exc:
            print(f"  row-crop quantity recount failed ({exc}) -- kept the original full-page quantities for every row.")

    corrections_applied = template_learning.apply_corrections(template_match, extracted) if template_match else []
    for corr in corrections_applied:
        i = corr["row_index"]
        if i < len(recount_flags):
            recount_flags[i] = _merge_flag(recount_flags[i], "template_corrected", corr["sizes"], corr["note"])
    backend_used = ("lighton" if use_lighton_hybrid else "paddleocr") if do_hybrid else None
    template_learning.write_debug_artifact(
        outdir, image_path.stem, extracted.seller_name, extracted.size_headers, template_match,
        backend_used=backend_used, model_used=model, corrections_applied=corrections_applied,
    )

    _write_debug_artifacts(outdir, image_path.stem, extracted, recount_flags)
    total_duration = time.perf_counter() - t_image_start
    _log_usage(usage_log, image_path.name, model, "total", {"prompt_eval_count": None, "eval_count": None, "duration_seconds": total_duration})
    print(f"  total wall time for {image_path.name}: {total_duration:.1f}s")
    return _to_order_form(extracted, image_path.name)


_FLAG_SEVERITY = {"no_recount": 0, "ok": 0, "unverified": 1, "resolved": 2, "auto_corrected": 2, "template_corrected": 2, "unresolved": 3}


def _merge_flag(base: dict | None, status: str, sizes: list[str], note: str) -> dict:
    """Combines a new catalog-derived flag with whatever flag (if any)
    already covers this row -- keeps the higher-severity status (so a real
    hybrid total-mismatch, "unresolved", isn't silently downgraded by a
    lower-severity catalog note), but always unions in the newly-flagged
    sizes and appends the new note, so a reviewer sees every reason a row
    is worth a second look, not just the first one found."""
    base = base or {}
    base_status = base.get("status", "no_recount")
    new_status = status if _FLAG_SEVERITY.get(status, 0) > _FLAG_SEVERITY.get(base_status, 0) else base_status
    merged_sizes = sorted(set(sizes) | set(base.get("sizes", [])), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
    notes = [n for n in [base.get("note"), note] if n]
    return {"status": new_status, "sizes": merged_sizes, "note": " | ".join(notes)}


def _merge_brandlist_flags_into_file(recount_flags_path: Path, annotations: list[dict]) -> None:
    """Folds brandlist_match.py's own catalog-cross-check signals
    (likely_column_shift, sizes_outside_catalog_range) into the same
    per-row flag file generate_review.py already renders as highlighted
    cells -- these are just as much a "this quantity looks suspicious"
    signal as the hybrid total-mismatch check, and pinning down the EXACT
    size(s) involved (rather than flagging a whole row, the total-mismatch
    check's only option) is a real quality improvement: a reviewer can look
    at one or two cells instead of re-checking an entire row against the
    photo. likely_column_shift additionally carries a concrete suggested
    correction (from detect_column_shift), so it's treated as higher
    confidence ("resolved" severity, matching the same status
    generate_review.py already uses for "disagreed but auto-resolved") than
    a bare out-of-range flag with no specific fix in hand ("unverified").
    Only touches the file this run's own extract_one() already wrote via
    _write_debug_artifacts -- silently does nothing if it's missing (e.g.
    an image with zero items)."""
    if not recount_flags_path.exists():
        return
    flags: list[dict] = json.loads(recount_flags_path.read_text(encoding="utf-8"))
    changed = False
    for idx, note in enumerate(annotations):
        if idx >= len(flags):
            continue
        shift = note.get("likely_column_shift")
        if shift:
            sizes = list(shift["suggested_correction"].keys())
            # ASCII "->" not a unicode arrow -- write_text() calls in this
            # file don't pass encoding="utf-8" (a pre-existing gap across
            # every debug-artifact writer, not just this one), so they fall
            # back to the platform default codec; confirmed by direct
            # testing this raises UnicodeEncodeError under Windows' cp1252
            # default the moment a non-ASCII character shows up in a note.
            correction = ", ".join(f"{k}->{v}" for k, v in shift["suggested_correction"].items())
            flags[idx] = _merge_flag(flags[idx], "resolved", sizes,
                f"Catalog check: likely column shift (offset {shift['offset']:+d}) -- try {correction}.")
            changed = True
        elif note.get("sizes_outside_catalog_range"):
            sizes = [str(s) for s in note["sizes_outside_catalog_range"]]
            flags[idx] = _merge_flag(flags[idx], "unverified", sizes,
                f"Catalog check: size(s) {', '.join(sizes)} aren't in this product's known catalog sizes.")
            changed = True
    if changed:
        recount_flags_path.write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Extract structured data from order form photos via Ollama Cloud (production pipeline, default model mistral-large-3:675b -- see module docstring).")
    parser.add_argument("input", help="Path to a single image, or a folder of images. Quote paths containing spaces.")
    parser.add_argument("--outdir", default="extracted_ollama_cloud", help="Output directory (default: ./extracted_ollama_cloud)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama Cloud model tag (default: {DEFAULT_MODEL} -- see HISTORY.md for the model comparison this was chosen from), or 'chandra' as a shorthand for {CHANDRA_MODEL} (local Ollama, free, no API key -- see chandra_parser.py and HISTORY.md's 2026-09-07 evaluation for what this trades off against the default: comparable accuracy on this project's test forms, but 2-3x slower than mistral's own main call alone, needs `ollama pull {CHANDRA_MODEL}` first, and doesn't have mistral's struck_out/date_present/letter_sizes flags -- chandra_parser.py's own reading already handles the first two correctly without them).")
    parser.add_argument("--usage-log", default="usage_log_ollama_cloud.csv", help="CSV file every call's token usage/timing is appended to (default: ./usage_log_ollama_cloud.csv, kept separate from extract_claude.py's usage_log.csv)")
    parser.add_argument("--no-brandlist-check", action="store_true", help="Skip the local (free, no API cost) cross-check against the brandlist product catalog.")
    parser.add_argument("--recount", action="store_true", help="Run the per-row recount pass (default: off). Row-crop alignment (row_top_frac/row_bottom_frac from the main call) has a documented reliability gap -- a misaligned crop can get silently accepted as 'resolved' when it actually holds a neighboring row's data, which reads as higher-confidence than doing nothing. Off by default for that reason; see HISTORY.md for the specific case this caused. --hybrid-quantities is the recommended way to independently verify quantities instead.")
    parser.add_argument("--think", action="store_true", help="Leave the model's internal 'thinking' mode on (Ollama Cloud default) instead of forcing it off. Confirmed 2026-08-13: the right setting is model-specific -- qwen3.5:397b is faster AND accurate with thinking off, but kimi-k2.6 returns deterministically all-zero quantities with thinking off and needs it on to produce a real reading. Try --no-think first (this file's default); if a model comes back schema-valid but all-zero, retry with --think before concluding the model can't do the task.")
    parser.add_argument("--no-think", dest="think", action="store_false", help="Force thinking off (default behavior already -- explicit flag for clarity/scripting).")
    parser.add_argument("--think-effort", choices=["low", "medium", "high", "max"], default=None,
                         help="Pass a string reasoning-effort level to Ollama's think= parameter instead of a bare bool. Confirmed necessary 2026-09-02 for glm-5.3-flash: that model's own reasoning is ALWAYS ON (per its ollama.com model page -- 'effort tunable per request across low, high, and max levels'), so a plain --think/--no-think bool has no effect on it at all. Confirmed via real runs on sample 3-scanned.jpg: 'low' avoids the truncation (~2900 tokens, ~13s) but flakes to zero quantities often; 'medium' -- not a real tier this model's own vocabulary recognizes at all (only low/high/max) -- fails identically to the bare bool (32000-token truncation, both attempts); 'high' avoided both problems on that form (20/20 rows exact vs. mistral-large-3:675b's own reading) but STILL flaked empty 2 of 3 attempts on a different, free-form page (sample 2.jpeg) -- 'high' is not actually a fix for the flake rate, it just happened to succeed on the first try on the one form it was first tested against. 'max', the model's own documented top tier: the installed `ollama` package's own ChatRequest normally validates think as bool | Literal['low','medium','high'] via Pydantic and rejects 'max' client-side before any request is sent -- confirmed via a raw HTTP call that the Ollama Cloud API itself accepts 'max' fine, so _call_schema bypasses the SDK's own chat() method (see _chat_bypassing_sdk_validation) specifically for this value. The bypass mechanism itself works, but 'max' is WORSE than 'high' in practice: tried on sample 5-scanned.jpg (the easiest form in this project's test set) and it burned the entire 32000-token ceiling in 273.6s without ever finishing (done_reason=length) -- 'max' triggers even more verbose reasoning than this pipeline's current MAX_TOKENS budget can accommodate, so its actual accuracy has never been observed. Not recommended; 'high' remains the best working setting despite its own flake rate. Takes precedence over --think/--no-think when set. Not yet confirmed for any other model.")
    parser.add_argument("--preprocess-image", action="store_true", help="Apply automated deskew + CLAHE contrast normalization (preprocess_for_vlm.py) before sending the image to mistral. Confirmed to fix a real row-bleed bug on one hard form, but also confirmed via A/B testing to damage other forms (a previously exact-match row picked up a shift and a new digit error). Off by default for that reason -- turn this on only for a specific image you know has a row-bleed problem, not as a general-purpose quality improvement. See HISTORY.md for the full numbers.")
    parser.add_argument("--no-hybrid-quantities", action="store_true", help="Skip the OCR-grounded quantity re-read (default: on). Re-reads quantities from REAL OCR-measured coordinates (not the VLM's self-report) for headers and marks, then groups each mark with its nearest header by x-position -- one whole-image OCR pass, no per-row crops, header-grouping done deterministically in code (order-preserving DP), not by a model call. Extensively tested and regression-checked (see HISTORY.md): corrects mistral's column-position drift, recovers handwritten overflow columns past the printed grid, resolves letter-coded sizes (S/M/L/XL/XXL), and reconciles its own reading against the VLM's per-row reading rather than unconditionally overriding it. This is the main accuracy lever for the production pipeline -- only disable it to isolate a bug or compare raw VLM output.")
    parser.add_argument("--no-lighton-hybrid", action="store_true", help="Fall back to the previous default (PaddleOCR + grid.py's pixel-position DP, extract_ollama_cloud._hybrid_ocr_quantities) instead of hybrid_quantities_lighton.py's LightOnOCR-2-based quantity re-read, which became the default 2026-09-05 (has no effect if --no-hybrid-quantities is also set, since neither backend runs then). See hybrid_quantities_lighton.py's module docstring and compare_hybrid_backends.py for what justified the switch and what's still open -- most notably, sample 4-scanned's rows haven't all been individually re-verified against the photo yet. Needs maternion/LightOnOCR-2:1b pulled in local Ollama (`ollama pull maternion/LightOnOCR-2:1b`) unless this flag is passed.")
    parser.set_defaults(think=None)
    args = parser.parse_args()

    global THINK
    if args.think is not None:
        THINK = args.think
    if args.think_effort is not None:
        THINK = args.think_effort

    if args.model.lower() == "chandra":
        args.model = CHANDRA_MODEL

    client = get_client(args.model)

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
            forms[img_path.stem] = extract_one(client, args.model, img_path, outdir, usage_log, system_prompt, brandlist_available, do_recount=args.recount, do_preprocess=args.preprocess_image, do_hybrid=not args.no_hybrid_quantities, use_lighton_hybrid=(False if args.no_lighton_hybrid else None))
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
                json.dumps(annotations, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            _merge_brandlist_flags_into_file(outdir / f"{img_path.stem}.recount_flags.json", annotations)
            party_check = brandlist_match.resolve_party_name(stage_a.get("seller_name", ""), form.party_name)
            if party_check is not None:
                (outdir / f"{img_path.stem}.party_check.json").write_text(
                    json.dumps(party_check, indent=2, ensure_ascii=False), encoding="utf-8"
                )
        json_out = outdir / f"{img_path.stem}.json"
        json_out.write_text(json.dumps(form.model_dump(mode="json", exclude={"source_file"}), indent=2, ensure_ascii=False), encoding="utf-8")
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
