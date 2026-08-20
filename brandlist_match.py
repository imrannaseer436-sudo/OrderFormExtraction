"""
brandlist_match.py -- local, zero-API-cost cross-check of extract_claude.py's
output against the business's own product catalog (SQL Server `brandlist`
table). Everything here is a DB query plus string matching in Python; none of
it calls the Claude API, so none of it adds to the per-image cost.

Two independent things this buys:

1. Item-name/style suggestions. Fuzzy-match each extracted item against the
   catalog's real product names, since name alone is ambiguous (see
   CLAUDE.md's "Database cross-check" section -- e.g. "Bloomer Plain" could
   be the ladies' or kids' line; only the size range actually ordered tells
   them apart). Surfaced as a suggestion (written to <name>.brandlist.json,
   read by generate_review.py) for a human to confirm -- name identity is
   the genuinely ambiguous part, so this never silently overwrites item.item.
   A blank `type` IS filled in automatically, but only when the matched
   product has exactly one style in the catalog (no ambiguity to resolve).

2. Letter-size resolution. Some rows are written using standard clothing
   letter sizes (S/M/L/XL/XXL/...) instead of this form's printed numeric
   grid, because the product's real sizing doesn't correspond to the form's
   generic 45-105 range at all -- confirmed directly against the catalog on
   sample3.jpeg's "MM K 4532 B FULL PANT SET" row: its real sizes are
   {35, 40, 45, 50, 55}, a kids range with no relation to what's printed on
   the form. The letter->number mapping is ambiguous on its own (M means 85
   for an adult line, 40 for a kids line) -- resolved here by checking which
   mapping's value is actually one of the matched product's own sizes in the
   catalog, never assumed from the letter alone. Applied automatically (not
   just suggested) because once the right product is confidently identified,
   this is mechanical unit conversion, not an identity guess.

3. Column-shift detection. A row's values can all be attributed one column
   left/right of where they truly are (confirmed on sample3.jpeg: "B-4749
   Full Pant" reported {50,55,60,65,70,75,80,85}, but the catalog's only
   valid sizes for that exact product are {55,60,65,70,75,80,85,90} --
   every value one column left of truth). The row's SUM is unaffected by a
   shift (same numbers, different labels), so the Total-Dozen-checksum
   approach can't catch this -- but shifting the reported sizes by a small
   offset within the form's own printed header order and checking whether
   that now lands exactly inside the matched product's catalog sizes can.
   Only flagged (`likely_column_shift`), never auto-applied -- unlike letter
   resolution, this rewrites already-plausible-looking numbers, and a rare
   coincidental match is a real (if small) risk worth a human's eyes on.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import pyodbc
from rapidfuzz import fuzz, process, utils

from schema import OrderForm

# Numeric product codes (3+ digits, e.g. "4532", "3674") are the single most
# distinctive signal in these item names -- confirmed against the real
# catalog: whole-name fuzzy matching alone picked the WRONG product for
# "MM K4532" (matched a same-brand, same-word-shape, differently-numbered
# item) because generic shared words ("MM", "FULL PANT SET") outweighed the
# actual distinguishing digits. See find_best_match().
_DIGIT_RUN_RE = re.compile(r"\d{3,}")

# Garment letter-size -> numeric size. Two mappings because the same letter
# means a different number depending on the product line -- which one
# applies is resolved per item against that item's OWN catalog sizes (see
# resolve_letter_size), never assumed from the letter alone.
SIZE_LETTER_MAP_ADULT = {"XS": 75, "S": 80, "M": 85, "L": 90, "XL": 95, "XXL": 100, "2XL": 100, "3XL": 105, "4XL": 110}
SIZE_LETTER_MAP_KIDS = {"0": 25, "XS": 30, "S": 35, "M": 40, "L": 45, "XL": 50, "XXL": 55, "2XL": 55}

# Deliberately conservative -- a wrong auto-applied conversion is worse than
# an unresolved one flagged for review (a human resolves that in seconds; a
# silently-wrong conversion could ship in an order). Tune only against real
# precision/recall data, not a guess.
#
# Score alone is NOT a reliable confidence signal on this catalog -- tested
# directly: a correct code-narrowed match ("MM K4532" -> the one catalog name
# containing "4532") scored only 85.5 because of an extra middle word, while
# a WRONG whole-catalog word-overlap match ("MM Loop 4289", whose "4289"
# doesn't exist anywhere in the catalog) also scored 85.5 from generic shared
# words alone. Whether the digit code actually narrowed the candidate pool is
# the stronger signal, so it's the primary gate; score is secondary.
SUGGEST_SCORE_THRESHOLD = 70.0       # below this, don't even surface a suggestion -- too weak to be useful
AUTO_APPLY_UNNARROWED_SCORE = 95.0   # no digit code to narrow by (or its code isn't in the catalog at all):
                                      # require a near-exact whole-name match before auto-applying anything
AUTO_APPLY_NARROWED_SCORE = 70.0     # digit code narrowed the pool to a small set: the code match itself is
                                      # strong evidence, so a much lower text score is still trustworthy
AUTO_APPLY_NARROWED_MAX_POOL = 2     # ... but only when narrowing actually left few enough candidates that
                                      # the remaining text score is picking between near-duplicates, not guessing

# A digit code narrowing the pool to EXACTLY ONE catalog product is stronger
# evidence than the general narrowed-pool case above -- there is no other
# catalog product this could be, full stop, so a low text score there means
# the surrounding words (item name / style abbreviation) were misread, not
# that the product identity is actually in doubt. Confirmed on a real case
# (2026-08-07): "CREBO B4756 F/PANT" scored only 45.2-51.6 against its one
# code-4756 candidate "B 4756 SHORTS", entirely because Claude misread the
# form's "H/PANT" (half pant -- a real style match to "SHORTS") as "F/PANT"
# (full pant) -- confirmed against the actual handwriting, which matches row
# 11's "H/PANT" abbreviation exactly. The uniqueness of the code match, not
# the text score, is what should carry this. Still keeps a low floor (not
# zero) as a guard against a coincidental 3+ digit substring shared with an
# otherwise-unrelated item name.
AUTO_APPLY_UNIQUE_CODE_SCORE = 35.0


@dataclass
class MatchCandidate:
    bname: str
    styles: set[str]
    sizes: set[int]
    score: float
    code_narrowed: bool       # True if item_name had a digit code AND at least one catalog name shared it
    narrowed_pool_size: int   # size of the pool actually searched (post-narrowing, if narrowing succeeded)
    unmatched_code: str = ""  # set when item_name had a digit code but NO catalog name contains it at all


def _has_unique_code_match(match: "MatchCandidate") -> bool:
    return match.code_narrowed and match.narrowed_pool_size == 1


def _is_trustworthy(match: "MatchCandidate") -> bool:
    """Gate for auto-applying a change (type fill / letter-size resolution)
    rather than just suggesting it -- see the threshold comments above for
    why code-narrowing, not score, is the primary signal."""
    if _has_unique_code_match(match):
        return match.score >= AUTO_APPLY_UNIQUE_CODE_SCORE
    if match.code_narrowed and match.narrowed_pool_size <= AUTO_APPLY_NARROWED_MAX_POOL:
        return match.score >= AUTO_APPLY_NARROWED_SCORE
    return match.score >= AUTO_APPLY_UNNARROWED_SCORE


def _connect() -> pyodbc.Connection:
    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={os.environ['SERVER']};DATABASE={os.environ['DB']};"
        f"UID={os.environ['USER']};PWD={os.environ['PASSWORD']};TrustServerCertificate=yes"
    )
    return pyodbc.connect(conn_str, timeout=10)


@lru_cache(maxsize=1)
def _catalog() -> dict[str, dict]:
    """{bname: {"styles": {...}, "sizes": {...}}} for every isprimary=1 row.
    Loaded once per process (61K rows -- trivial to hold in memory) and
    reused across every item/image in a run instead of re-querying per item."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT bname, bstyle, bsize FROM brandlist WHERE isprimary = 1")
        catalog: dict[str, dict] = {}
        for bname, bstyle, bsize in cur.fetchall():
            entry = catalog.setdefault(bname, {"styles": set(), "sizes": set()})
            if bstyle:
                entry["styles"].add(bstyle)
            if bsize is not None:
                entry["sizes"].add(int(bsize))
        return catalog
    finally:
        conn.close()


@lru_cache(maxsize=1)
def known_style_codes() -> list[str]:
    """Every distinct Style value seen in the catalog -- for the system
    prompt, so Claude can recognize a style code wherever it lands on a row
    (not just a small hardcoded whitelist)."""
    styles: set[str] = set()
    for entry in _catalog().values():
        styles.update(entry["styles"])
    return sorted(s for s in styles if s)


@lru_cache(maxsize=1)
def known_numeric_sizes() -> list[int]:
    """Every distinct numeric bsize seen anywhere in the catalog (across all
    products, not per-item) -- built for extract_ollama_cloud.py's
    mistral-large-3:675b prompt addendum: confirmed by direct testing
    (2026-08-13) that mistral never once reported a handwritten size beyond
    a form's last PRINTED header (e.g. "110" on sample 5.jpeg's Fairlady
    rows), even after being told in prose that this happens sometimes. This
    gives the model the concrete, real size vocabulary this business
    actually uses (not a per-product lookup, which would need a second pass
    -- this is the general "sizes bigger than 105 are real and expected"
    signal, grounded in real data instead of an abstract claim)."""
    sizes: set[int] = set()
    for entry in _catalog().values():
        sizes.update(entry["sizes"])
    return sorted(sizes)


def find_best_match(item_name: str) -> MatchCandidate | None:
    catalog = _catalog()
    if not item_name or not catalog:
        return None

    # If the item name carries a numeric product code, narrow the candidate
    # pool to catalog names containing that same digit run BEFORE fuzzy
    # ranking -- see _DIGIT_RUN_RE comment for why this matters.
    codes = _DIGIT_RUN_RE.findall(item_name)
    pool = list(catalog.keys())
    code_narrowed = False
    unmatched_code = ""
    if codes:
        narrowed = [name for name in catalog if any(code in name for code in codes)]
        if narrowed:
            pool = narrowed
            code_narrowed = True
        else:
            unmatched_code = codes[0]  # a code was written but doesn't exist anywhere in the catalog

    # processor=default_process case-folds and strips punctuation before
    # scoring -- without it, "full"/"FULL" and "-"/" " differences alone
    # were enough to tank an otherwise-exact match's score by 50+ points
    # (confirmed: "B-4749 full Pant" vs "B 4749 FULL PANT SET" scored 44.4
    # raw, 95.0 with this processor).
    result = process.extractOne(item_name, pool, scorer=fuzz.WRatio, processor=utils.default_process)
    if result is None:
        return None
    bname, score, _ = result
    entry = catalog[bname]
    return MatchCandidate(
        bname=bname, styles=entry["styles"], sizes=entry["sizes"], score=score,
        code_narrowed=code_narrowed, narrowed_pool_size=len(pool), unmatched_code=unmatched_code,
    )


def detect_column_shift(quantities: dict[str, int], catalog_sizes: set[int], size_headers: list[str]) -> dict | None:
    """Checks whether this row's reported numeric sizes, shifted by a small
    consistent offset within the form's own printed header order, would land
    exactly inside the matched product's known catalog sizes -- the
    signature of a column-shift misread. See module docstring point 3 for
    the confirmed real case this is built from. Returns
    {"offset": int, "suggested_correction": {reported_size: corrected_size}}
    on a match, else None (including when the reported sizes are already
    all valid -- nothing to detect)."""
    numeric_sizes = sorted(int(s) for s in quantities if s.isdigit())
    header_order = [int(h) for h in size_headers if h.isdigit()]
    if not numeric_sizes or not catalog_sizes or not header_order:
        return None
    if set(numeric_sizes) <= catalog_sizes:
        return None  # already all valid -- no shift to find

    index_of = {h: i for i, h in enumerate(header_order)}
    if not all(s in index_of for s in numeric_sizes):
        return None  # a reported size isn't even a real header on this form -- a different problem

    for offset in (1, -1, 2, -2):
        shifted: dict[str, int] = {}
        in_bounds = True
        for s in numeric_sizes:
            new_index = index_of[s] + offset
            if not (0 <= new_index < len(header_order)):
                in_bounds = False
                break
            shifted[str(s)] = header_order[new_index]
        if in_bounds and set(shifted.values()) <= catalog_sizes:
            return {"offset": offset, "suggested_correction": shifted}
    return None


def resolve_shift_offset(quantities: dict[str, int], catalog_sizes: set[int], size_headers: list[str]) -> int | None:
    """Like detect_column_shift, but answers a different question: "does
    THIS reading, as a whole, sit at a confirmed, UNAMBIGUOUS position
    relative to the form's headers?" -- checked at every offset in -2..2
    INCLUDING 0 (detect_column_shift skips 0 entirely, since it only exists
    to flag an already-invalid reading). That matters because most rows on
    a real form are NOT shifted, and "this reading needs no correction" is
    itself a useful, checkable fact -- not just the absence of one.

    Returns the offset ONLY when it is the UNIQUE one (among -2..2) that
    lands every reported size inside catalog_sizes -- if two different
    offsets (e.g. 0 and +1) both happen to fit, that means catalog_sizes is
    too broad/overlapping to discriminate for this specific product, and
    returning either would be a guess dressed up as a fact. Returns None in
    that case, or when no offset fits at all.

    Built for extract_claude.py's _apply_recount(): a per-row, catalog-
    grounded offset measurement, used both to resolve individual
    disagreements (like detect_column_shift, but not restricted to already-
    invalid readings) and, unlike detect_column_shift, to catch the case
    where the main call and the recount call AGREE but share the same
    column-drift bias -- agreement alone was previously trusted with no
    check at all; this closes that gap wherever a trustworthy catalog match
    exists for the row."""
    numeric_sizes = sorted(int(s) for s in quantities if s.isdigit())
    header_order = [int(h) for h in size_headers if h.isdigit()]
    if not numeric_sizes or not catalog_sizes or not header_order:
        return None
    index_of = {h: i for i, h in enumerate(header_order)}
    if not all(s in index_of for s in numeric_sizes):
        return None

    valid_offsets = []
    for offset in (-2, -1, 0, 1, 2):
        shifted_vals = set()
        in_bounds = True
        for s in numeric_sizes:
            new_index = index_of[s] + offset
            if not (0 <= new_index < len(header_order)):
                in_bounds = False
                break
            shifted_vals.add(header_order[new_index])
        if in_bounds and shifted_vals <= catalog_sizes:
            valid_offsets.append(offset)
    return valid_offsets[0] if len(valid_offsets) == 1 else None


def resolve_letter_size(letter: str, db_sizes: set[int]) -> int | None:
    """Which numeric size a garment letter size means for THIS product,
    determined by checking which mapping's value the product's own catalog
    sizes actually contain -- see module docstring. None if neither mapping's
    value is in db_sizes, or if both are (genuinely ambiguous; the adult/kids
    ranges shouldn't overlap in practice, but don't guess if they somehow do)."""
    letter = letter.strip().upper()
    adult_val = SIZE_LETTER_MAP_ADULT.get(letter)
    kids_val = SIZE_LETTER_MAP_KIDS.get(letter)
    adult_hit = adult_val if adult_val in db_sizes else None
    kids_hit = kids_val if kids_val in db_sizes else None
    if adult_hit is not None and kids_hit is not None and adult_hit != kids_hit:
        return None
    return adult_hit if adult_hit is not None else kids_hit


# Confirmed real (2026-08-08, sample 8.jpeg -- see module docstring point 4
# below): fuzzy name matching alone caps out around 85.5 for two UNRELATED
# names that just happen to share generic words (the exact same ceiling
# already independently confirmed for item-name matching above) -- so a real
# identity match needs to clear that ceiling with a safety margin, not just
# beat it narrowly.
BUYER_MATCH_THRESHOLD = 88.0


@dataclass
class PartyMatchCandidate:
    buyer_name: str
    city: str
    score: float


@lru_cache(maxsize=1)
def _buyers() -> list[tuple[str, str, str]]:
    """(BuyerName, City, search_text) for every active buyer -- search_text
    combines name + city since real buyer identity on these forms is often
    only disambiguated by city (confirmed: multiple "SHARDA SALES" buyer
    records exist in different cities/branches, only distinguishable by
    city). Loaded once per process, same pattern as _catalog()."""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT BuyerName, City FROM buyer WHERE IsActive = 1")
        return [(bname, city or "", f"{bname} {city or ''}".strip()) for bname, city in cur.fetchall()]
    finally:
        conn.close()


def find_best_buyer_match(candidate_name: str) -> PartyMatchCandidate | None:
    """Fuzzy-matches candidate_name against the real buyer master list --
    used by resolve_party_name() to check whether a form's seller_name or
    party_name is a REAL registered Essa buyer, not just plausible-looking
    text. None if candidate_name is empty or the buyer list is unavailable."""
    buyers = _buyers()
    if not candidate_name or not buyers:
        return None
    pool = [b[2] for b in buyers]
    result = process.extractOne(candidate_name, pool, scorer=fuzz.WRatio, processor=utils.default_process)
    if result is None:
        return None
    _, score, idx = result
    bname, city, _ = buyers[idx]
    return PartyMatchCandidate(buyer_name=bname, city=city, score=score)


def resolve_party_name(seller_name: str, party_name: str) -> dict | None:
    """Cross-checks a form's TWO candidate party names -- the printed
    letterhead (seller_name) and the handwritten "Party Name" box
    (party_name) -- against the real buyer master list, since which one is
    the actual Essa-facing buyer depends on the form's own template, not on
    a fixed field-position rule:

    - Essa Garments' OWN pre-printed order pads always have "Essa"/"Essa
      Garments" as the letterhead and the real buyer handwritten in Party
      Name -- the existing prompt rule already gets this right, so this
      check is skipped entirely when seller_name reads as Essa (the common
      case; no point flagging every normal form).
    - A DIFFERENT business's own pre-printed order pad can have this
      reversed -- confirmed real (2026-08-08, sample 8.jpeg): letterhead
      "M/s. Sharda Sales Solapur" scores 95.0 against a real buyer record
      ("SHARDA SALES, SE" / city "SOLAPUR"), while the handwritten "Party
      Name" field "Rajkumar & Co." scores only 85.5 against ANY buyer
      record -- the same generic word-overlap ceiling confirmed for
      unrelated item names, i.e. no real match at all. "Sharda Sales
      Solapur" is Essa's actual registered buyer here; "Rajkumar & Co." is
      almost certainly Sharda Sales' own downstream customer, irrelevant to
      Essa's own records.

    Never changes party_name itself -- returns match info so the review page
    can flag this for a human to confirm, the same "flag, don't silently
    override" principle as the item-level checks above. Returns None when
    there's nothing worth flagging: seller_name is Essa (the normal case),
    or party_name already matches a real buyer and the letterhead doesn't
    (the expected, unproblematic outcome for a non-Essa-letterhead form
    whose Party Name box is still correct -- nothing to fix), or neither
    candidate is a strong enough signal to say anything useful."""
    if "essa" in (seller_name or "").lower():
        return None

    seller_match = find_best_buyer_match(seller_name)
    party_match = find_best_buyer_match(party_name)
    seller_hit = seller_match is not None and seller_match.score >= BUYER_MATCH_THRESHOLD
    party_hit = party_match is not None and party_match.score >= BUYER_MATCH_THRESHOLD

    if party_hit and not seller_hit:
        return None  # party_name is already a real buyer -- nothing to flag
    if seller_hit and party_hit and seller_match.buyer_name == party_match.buyer_name:
        return None  # both candidates agree on the same real buyer -- confirmed, nothing to flag

    result = {"seller_name": seller_name, "party_name": party_name}
    if seller_match:
        result["seller_name_match"] = {"buyer_name": seller_match.buyer_name, "city": seller_match.city, "score": round(seller_match.score, 1)}
    if party_match:
        result["party_name_match"] = {"buyer_name": party_match.buyer_name, "city": party_match.city, "score": round(party_match.score, 1)}

    if seller_hit and party_hit:
        # different real buyers -- genuinely ambiguous, no single suggestion.
        result["note"] = (
            f"Both the letterhead ('{seller_name}') and the handwritten Party Name ('{party_name}') "
            f"match DIFFERENT registered buyers ({seller_match.buyer_name}, {seller_match.city} vs "
            f"{party_match.buyer_name}, {party_match.city}) -- verify which one is the real buyer for "
            f"this order before uploading."
        )
    elif seller_hit:
        result["suggested_party_name"] = seller_match.buyer_name
        # Confirmed real (2026-08-08, sample 8.jpeg): the buyer table has
        # multiple near-duplicate records for the same real-world business
        # (branch/agent-code variants like "(S)", "(AS)", "(SE)"), and
        # IsActive doesn't reliably distinguish them from closed ones --
        # e.g. "M/S SHARDA SALES (Z CLOSE)" is IsActive=1 while several
        # non-"(CLOSE)" variants are IsActive=0. Don't assert the top fuzzy
        # match as definitely correct when its own name suggests it's a
        # closed/inactive account -- flag that explicitly so a human picks
        # the right branch record instead of the closed one.
        closed_hint = (
            " (NOTE: this specific buyer record's name suggests it may be closed/inactive -- "
            "check for an active variant of the same business in the buyer list.)"
            if "close" in seller_match.buyer_name.lower() else ""
        )
        result["note"] = (
            f"This form's letterhead ('{seller_name}') matches a real registered buyer "
            f"({seller_match.buyer_name}, {seller_match.city}), but the handwritten Party Name "
            f"('{party_name}') doesn't match any buyer -- this looks like a DIFFERENT business's own "
            f"order pad, not Essa's. The letterhead business is likely the real buyer for Essa's "
            f"records; '{party_name}' may just be that business's own downstream customer.{closed_hint}"
        )
    else:
        # neither matched -- still worth a heads-up (this is a non-Essa
        # letterhead, so the usual "Party Name box is always right" rule is
        # already less certain here), but no suggestion to offer.
        result["note"] = (
            f"This form's letterhead ('{seller_name}') isn't Essa, and neither it nor the handwritten "
            f"Party Name ('{party_name}') matched a registered buyer -- this buyer may not be in the "
            f"database yet, or either name may be misread. Verify against the photo before uploading."
        )
    return result


def annotate_and_resolve(form: OrderForm, size_headers: list[str] | None = None) -> list[dict]:
    """Runs all three checks above against every item in `form`, IN PLACE,
    gated by _is_trustworthy() (code-narrowing first, score second -- see
    the threshold comments above):
    - quantities gets letter-size keys converted to numeric ones;
    - a blank `type` gets filled in when the matched product has exactly one
      catalog style (no ambiguity to resolve);
    - item.item is never changed -- name identity is the ambiguous part, so
      that's returned as a suggestion only, at the lower SUGGEST_SCORE_THRESHOLD;
    - a likely column-shift is flagged (never auto-corrected) if `size_headers`
      is provided -- pass the form's own printed header order (e.g. from the
      matching <name>.stageA.json) to enable this check; omit it to skip.
    Returns one annotation dict per item (same order as form.items), meant to
    be written to <name>.brandlist.json and read by generate_review.py."""
    size_headers = size_headers or []
    annotations = []
    for item in form.items:
        match = find_best_match(item.item)
        note: dict = {"item": item.item, "type": item.type, "match": None}
        if match is None:
            annotations.append(note)
            continue
        if match.unmatched_code:
            note["code_not_in_catalog"] = match.unmatched_code

        note["match"] = {
            "bname": match.bname, "styles": sorted(match.styles),
            "sizes": sorted(match.sizes), "score": round(match.score, 1),
            "code_narrowed": match.code_narrowed,
            # narrowed_pool_size, not just code_narrowed, is what
            # generate_review.py needs to reproduce _has_unique_code_match's
            # display bypass below -- see the 2026-08-08 fix there for why
            # this field was missing and hiding real matches on the review
            # page (CRERO B4756 H/PANT, LOOPER ZB3965 H/PANT: both genuine
            # unique-code matches with a low text score, confirmed via a
            # live catalog query).
            "narrowed_pool_size": match.narrowed_pool_size,
        }
        # A unique-code match bypasses the general suggest floor too, not
        # just the auto-apply one -- see AUTO_APPLY_UNIQUE_CODE_SCORE: a
        # score below 70 here is normally "too weak to even mention," but
        # when the code has already narrowed the field to the ONE catalog
        # product it could possibly be, that low score is telling you the
        # surrounding text was misread, not that the identity is in doubt.
        if match.score < SUGGEST_SCORE_THRESHOLD and not _has_unique_code_match(match):
            annotations.append(note)
            continue  # too weak to even surface as a suggestion

        trustworthy = _is_trustworthy(match)

        if trustworthy and not item.type and len(match.styles) == 1:
            item.type = next(iter(match.styles))
            note["type_filled_from_catalog"] = item.type
        elif item.type and match.styles and item.type not in match.styles:
            note["style_mismatch"] = sorted(match.styles)

        # A letter can resolve to a numeric size that's ALSO already present
        # as its own key (e.g. a printed-header "45" cell was read directly,
        # and a different cell's letter "L" also resolves to 45 for this
        # product) -- checked against the pre-resolution snapshot so this
        # can't happen via plain dict.update() silently keeping whichever
        # key iteration happened to process last. Flagged instead of guessed.
        original_numeric_keys = {s for s in item.quantities if s.isdigit()}
        resolved: dict[str, int] = {}
        unresolved: list[str] = []
        conflicts: dict[str, str] = {}
        for size in list(item.quantities.keys()):
            if size.isdigit():
                continue
            qty = item.quantities[size]
            numeric = resolve_letter_size(size, match.sizes) if trustworthy else None
            if numeric is None:
                unresolved.append(size)
                continue
            numeric_str = str(numeric)
            if numeric_str in original_numeric_keys or numeric_str in resolved:
                conflicts[size] = numeric_str
                unresolved.append(size)
                continue
            resolved[numeric_str] = qty
            del item.quantities[size]
        if resolved:
            item.quantities.update(resolved)
            note["resolved_sizes"] = resolved
        if unresolved:
            note["unresolved_sizes"] = unresolved
        if conflicts:
            note["size_conflicts"] = conflicts

        # Plausibility check, not a correction -- a column-shift misread (a
        # whole row's values attributed one column left/right of where they
        # really are) doesn't change the row's sum, so the Total Dozen
        # checksum can't catch it; comparing reported sizes against the
        # matched product's own known catalog sizes can. Confirmed on a real
        # case: "B-4749 Full Pant" (95.0 score, code-narrowed -- as
        # trustworthy a match as this pipeline produces) has catalog sizes
        # {55,60,65,70,75,80,85,90} only, but a misread reported size "50"
        # (not a valid size for this product at all) and was missing "90"
        # (which the row does have) -- classic one-column-left shift.
        if trustworthy and match.sizes:
            numeric_sizes = {int(s) for s in item.quantities if s.isdigit()}
            out_of_range = sorted(numeric_sizes - match.sizes)
            if out_of_range:
                note["sizes_outside_catalog_range"] = out_of_range
                # Cross-reference against the form's own printed header order
                # to pin down not just THAT it's suspect but the likely exact
                # shift -- see detect_column_shift(). Still only a flag, not
                # an auto-correction: this rewrites already-plausible-looking
                # numbers, unlike letter-size resolution's unit conversion.
                shift = detect_column_shift(item.quantities, match.sizes, size_headers)
                if shift:
                    note["likely_column_shift"] = shift

        annotations.append(note)
    return annotations
