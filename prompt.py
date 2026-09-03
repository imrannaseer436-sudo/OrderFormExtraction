"""
V3 prompting strategy — three stages instead of two:

STAGE A (constrained, image): extract just the STRUCTURE — party info,
size headers, and the list of item rows (name + style code). No
quantities. This is grammar-constrained (format=FormMeta schema) because
it has worked reliably in every earlier attempt — the failures were
always in the quantities, never in this metadata.

STAGE B (unconstrained, image, once PER ROW): for each item identified in
stage A, a separate call asks the model to focus on ONLY that one row and
read across its size columns. Isolating one row at a time drastically
reduces how much positional state the model has to track per call, which
is the fix for the column-drift problem seen with whole-table extraction.

STAGE B parsing: done in plain Python regex, not another model call —
the per-row output format is constrained enough (comma-separated
"size:value" pairs) that a model call to reformat it would just be another
place for errors to creep in.
"""

from schema import FormMeta

STAGE_A_PROMPT = """You are transcribing the STRUCTURE of a garment order form photo — not the \
quantities yet, just what's there.

IMPORTANT — two things people commonly misread on these forms:

A) SELLER vs BUYER. The top of the page (usually a logo/letterhead, e.g. "ESSA GARMENTS PRIVATE \
LIMITED" with its own printed address) is the SELLER. The buyer/party's own name and city are \
handwritten in the "Party Name" box further down — use THAT for party_name/party_city, never the \
letterhead's printed address.

B) DITTO MARKS. A ditto mark (—"—, -"-, or similar) applies only within the column it's written in \
— it means "same value as the row above, in THIS column," never "same as the row above in every \
column." Two cases:
  - Ditto mark in the Particulars column: item name is inherited from the row above, with this \
row's word added if there is one. E.g. if the row above is "Fairlady Print" and the next row is \
'—"— Plain', the full item name for that row is "Fairlady Plain" — write out the FULL inherited \
name, never just the modifier word alone.
  - Ditto mark in Particulars while the Style column has an ACTUAL new value (not a ditto mark) on \
that same row: the item name still comes from the ditto inheritance above, and the Style column's \
new value is read independently as that row's type. E.g. row above is "MYNA" / Style "IE"; next row \
is '—"—' / Style "OE" → item="MYNA", type="OE". Never mistake the Style column's value for the item \
name just because Particulars was a ditto mark.

C) ITEM NAME vs STYLE CODE. The "Particulars" column (item name) and the "Style" column are two \
separate columns — never merge them. If Particulars says "Trend Trunk" and Style says "IE", then \
item must be exactly "Trend Trunk" and type must be "IE". Do NOT write "Trend Trunk IE" as the item \
name. The item name is exactly what's written in the Particulars column, nothing appended from \
another column.

D) WRAPPED LINES ON FREE-FORM PAGES. Some order forms have no printed grid at all — just handwritten \
item names with their own size:quantity list written directly below or beside each one, with no \
shared header row across the page. On a page like this, a line of size:quantity pairs (or a plain \
list of numbers) that has NO new item name or bullet point directly above it is a CONTINUATION of \
the item above, not a new item — do not create a separate item for it.

Extract:
- seller_name, party_name, party_city, order_no.
- order_date: normalize to DD/MM/YYYY, assuming 20xx for 2-digit years. This field must contain \
ONLY the final normalized date, e.g. "30/03/2026" — never any explanation, working, or the original \
as-written text alongside it.
- size_headers: every size column header across the top of the table, left to right, exactly as \
printed. Ignore any second header row underneath giving an age/chest equivalent — only the primary \
size code goes here. Do NOT include a trailing running-total column (e.g. "Total", "Total Dozen", \
"Grand Total") — that is a summary column, not a size. If this page has NO single shared header row \
at all (e.g. a free-form handwritten page where each item has its own local list of sizes written \
directly with it, per rule D), leave size_headers as an empty list rather than guessing one.
- items: every single product/article row, top to bottom, with its item name (ditto marks expanded \
per rule B, style code excluded per rule C) and style code if there's a separate Style column. A \
style code is always alphabetic (e.g. RN, RNS, OE, IE), never a number — the second header row's \
age/chest-equivalent numbers sit close to the Style column on some forms and must NEVER be read as \
that row's style code. If the Style column is blank or too faint/illegible to read for a row, leave \
type as an empty string rather than guessing a nearby number. Do NOT include quantities here.
- notes: any page-level handwritten notes (e.g. "Old Rate Supply Only"), and anything illegible/\
unusual not tied to a specific row's quantities."""


ROW_PROMPT_TEMPLATE_FULL = """Look at this order form image again. Focus ONLY on the row for:

    ITEM: {item}
    STYLE: {style}

First, find "{item}" written in the leftmost "Particulars" column. Do this every time, even if you \
just read a different row — do not assume this row is directly below the last one you read.

Once you've found it, ignore every other row, INCLUDING the header rows above the table (the size \
numbers themselves, and any smaller second row of numbers just below them, e.g. an age/chest-size \
equivalent — those are column labels, not data for this or any item). Only read the row of numbers \
that sits on the same horizontal line as "{item}" in the Particulars column.

The size column headers across the top of the table, in left-to-right order, are:

    {headers}

For THIS ROW ONLY, go through those headers one at a time, left to right, and state what is \
written directly beneath each one, on this item's line. Many rows are blank for the first several \
columns and only have numbers starting partway across — go column by column and don't skip any, so \
you don't lose your place.

{format_instructions}"""


# Used ONLY when the WHOLE image has no printed grid at all (grid.py found
# zero row candidates for the entire page) -- e.g. a free-form handwritten
# notebook page where each item has its own local size:quantity list
# written directly with it, not lined up under one shared row of column
# headers. Deliberately a SEPARATE template from ROW_PROMPT_TEMPLATE_FULL,
# not a conditional clause added to it -- confirmed by a real regression
# (2026-09-01): adding a "maybe there's no shared header, ignore the list
# above" escape hatch to the shared per-row-fallback template made an
# ALREADY-uncertain row (one already flagged as a row-alignment failure on
# an otherwise-normal grid form) measurably worse -- the model took the
# escape hatch on a form that DOES have a real shared header row, and
# hallucinated a full-range reading instead. A model weak enough to need
# that escape hatch is also too weak to reliably judge when it applies;
# only route here when the CALLER already knows (from grid.py finding no
# candidates at all) that no shared header row exists.
ROW_PROMPT_TEMPLATE_FREEFORM = """Look at this order form image. This page has NO single shared \
header row across the top of a table -- it's a free-form handwritten page where each item has its \
own local size:quantity list written directly with it (e.g. sizes on one line and their quantities \
on the line below, or size/quantity pairs written inline like "60/10, 65/15").

Find this item on the page:

    ITEM: {item}
    STYLE: {style}

Read whatever size:quantity pairs are written directly with THIS item -- not with any other item on \
the page. If this item's own list continues onto a second line with no new item name above it, \
include that continuation too. Read the item's FULL local list even if it runs longer than a \
typical range for this page — there's no implied ceiling on how many sizes one item can have.

{format_instructions}"""


ROW_PROMPT_TEMPLATE_CROP = """This image has two parts stacked vertically, cropped from a garment \
order form:

- TOP part: the column header — the size numbers (45, 50, 55, ...) that label each column.
- BOTTOM part: a single row of item data, possibly with a thin sliver of a neighboring row visible \
at its very top or bottom edge.

The bottom part's row is for:

    ITEM: {item}
    STYLE: {style}

Use the TOP part purely to know which column is which — do not read data from it. If more than one \
row's worth of content is visible in the BOTTOM part, find "{item}" in its leftmost "Particulars" \
column first, then read only the row of numbers on that same horizontal line; ignore any sliver of \
another row visible at the very top/bottom edge.

The size column headers, in left-to-right order (matching the columns shown in the TOP part), are:

    {headers}

For THIS ROW ONLY, go through those headers one at a time, left to right, and state what is \
written directly beneath each one, on this item's line. Many rows are blank for the first several \
columns and only have numbers starting partway across — go column by column and don't skip any, so \
you don't lose your place.

{format_instructions}"""


FORMAT_INSTRUCTIONS = """Respond with ONLY a single line in this exact format, one pair per header, \
in the same order as the headers above, separated by commas:

{example}

Use "blank" for empty cells, or the actual value (a number, "x", or "-") for filled ones. Do not \
add any explanation, just the one line of size:value pairs."""


def build_stage_a_prompt() -> str:
    return STAGE_A_PROMPT


def build_row_prompt(item: str, style: str, headers: list[str], cropped: bool = False, freeform: bool = False) -> str:
    # On a free-form page there's no real headers list to build an example
    # from (see ROW_PROMPT_TEMPLATE_FREEFORM) -- a generic illustrative
    # example still shows the model the expected size:value output shape.
    example = ", ".join(f"{h}:value" for h in headers) if headers else "45:10, 50:15, 55:blank"
    format_instructions = FORMAT_INSTRUCTIONS.format(example=example)
    if freeform:
        template = ROW_PROMPT_TEMPLATE_FREEFORM
    elif cropped:
        template = ROW_PROMPT_TEMPLATE_CROP
    else:
        template = ROW_PROMPT_TEMPLATE_FULL
    return template.format(item=item, style=style or "(none)", headers=", ".join(headers), format_instructions=format_instructions)


# Exposed for extract_ollama.py
STAGE_A_SCHEMA = FormMeta.model_json_schema()
