"""
template_learning.py -- per-template correction memory.

The review app already writes every human correction to
ui_sessions/<sid>/submitted_<orderno>.json, pairing what the pipeline
produced with exactly what a human fixed it to -- but until this module,
nothing read that data back INTO a future extraction. This module closes
that loop for a narrow, deliberately-scoped case: when a newly-uploaded
photo's seller_name + size_headers exactly match a template a human has
already corrected before, reapply the SAME corrections automatically
instead of making a reviewer retype them.

Two architectural constraints, both decided with the user rather than
assumed, shape everything below:

1. MATCH TIMING. seller_name/size_headers (the two signals that identify
   "which template this is") only exist AFTER extract_ollama_cloud.py's one
   main VLM call finishes -- there is no cheap pre-call signal in this
   pipeline, and adding one (an image fingerprint, or a second cheap model
   call) was explicitly declined. So matching happens strictly AFTER the
   main call, using its own real output. That rules out few-shot prompt
   injection into that same image's main call (it already ran) and rules
   out ever choosing WHICH MODEL handles the main call by template -- both
   would need to know the template before the call that would use that
   knowledge. What's still possible, entirely after the main call but
   inside the same extract_one() invocation: which hybrid-quantities
   BACKEND to use (decided by code that runs after the main call), and
   deterministic corrections to the main call's own item/quantity reading
   (also applied after it returns). See extract_ollama_cloud.py's
   extract_one() for exactly where both of those hook in.

2. WHAT A MATCH DOES. The point of this module is that a correction a
   human already made once for a given template should not need to be made
   again -- not just surfaced as a hint. So a match auto-applies, subject to
   the one non-negotiable rule every other auto-* mechanism in this
   codebase already follows (brandlist cross-check, hybrid quantities,
   catalog auto-correction): NEVER SILENT. Every applied correction is
   folded into the same recount_flags mechanism generate_review.py and the
   review app already render as a highlighted, hoverable, one-click-to-undo
   cell (see _FLAG_SEVERITY / _merge_flag in extract_ollama_cloud.py) under
   its own status, "template_corrected".

WHAT'S SAFE TO LEARN AS A REUSABLE VALUE, AND WHAT ISN'T. The same
pre-printed order form is reused across many different orders -- only the
handwritten party name and quantities differ order to order. So:

  - Item-name misreads ARE template-stable (the printed word doesn't
    change) and are stored/reapplied as a direct text substitution.
  - Column-shift corrections are template-stable as a POSITION, never as a
    value -- raw quantity numbers are handwritten and differ every order,
    so only the header-relative OFFSET a human's correction implies is ever
    stored, replaying the exact same logic
    app/static/app.js's shiftRowHorizontally() uses for the manual version
    of this fix.
  - Actual quantity values and the buyer/party name are NEVER stored as
    reusable corrections at all -- a different value every time is not a
    mistake to learn from.

MATCHING IS FUZZY AT LOOKUP TIME, EXACT AT STORAGE TIME. Rules are keyed by
the literal raw text that triggered them (no fuzzy merging when storing --
two distinct misreads stay two distinct rules). At lookup time, a freshly
extracted string is compared against stored triggers with rapidfuzz at a
threshold well above this codebase's general suggestion threshold
(brandlist_match.SUGGEST_SCORE_THRESHOLD = 70) rather than requiring a
byte-exact match -- a VLM re-misreading the same printed word across
separate calls is not guaranteed to sample an identical string every time,
and exact-only matching would silently under-fire on exactly the case this
module exists for. The template SIGNATURE itself (which template this is)
stays exact -- only within-template trigger matching is fuzzy.

STORAGE: one small JSON file per template signature under template_memory/
(gitignored -- real business data about this company's own order-form
templates, same reasoning as ui_sessions/), mirroring this repo's existing
per-key JSON sidecar convention rather than a database. No dependency this
project doesn't already have: hashlib/json (stdlib) for storage, rapidfuzz
(already used the same way in brandlist_match.py) for matching.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, utils as fz_utils

from extract_claude import QuantityPair

REPO_ROOT = Path(__file__).resolve().parent
STORE_DIR = REPO_ROOT / "template_memory"

# Well above brandlist_match.SUGGEST_SCORE_THRESHOLD (70) -- a wrong trigger
# match here doesn't just surface a suggestion, it silently rewrites a
# reading, so the bar for "confident enough to auto-apply" is deliberately
# much higher than "confident enough to mention."
TRIGGER_MATCH_THRESHOLD = 92.0

# Candidate header-index offsets tested when detecting whether a reviewer's
# correction is explained by a pure column shift. 0 is excluded -- a "fit"
# at 0 means the quantities didn't actually change, which is handled before
# this is ever called.
_SHIFT_CANDIDATE_OFFSETS = (-3, -2, -1, 1, 2, 3)


def _normalize(text: str) -> str:
    """Punctuation-insensitive casefold. A copy of app/pipeline.py's own
    _normalize_name(), not an import -- app/pipeline.py already imports
    extract_ollama_cloud.py (which imports this module), so importing back
    the other way would create a cycle. One line; kept in sync by hand."""
    return " ".join("".join(c if c.isalnum() else " " for c in (text or "")).split()).casefold()


def _signature(seller_name: str, size_headers: list[str]) -> str | None:
    """None when there's nothing stable to sign on -- a free-form page (no
    seller_name reliably read, or no shared header row at all) has no
    consistent layout for a future upload to match against."""
    if not (seller_name or "").strip() or not size_headers:
        return None
    raw = _normalize(seller_name) + "||" + "|".join(size_headers)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _store_path(signature: str) -> Path:
    return STORE_DIR / f"{signature}.json"


def _empty_template(signature: str, seller_name: str, size_headers: list[str]) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "signature": signature,
        "seller_name": seller_name,
        "size_headers": list(size_headers),
        "first_seen": now,
        "last_seen": now,
        "occurrences": 0,
        "item_substitutions": [],
        "column_shifts": [],
        "backend_stats": {},
        "model_stats": {},
    }


def _load(signature: str) -> dict | None:
    try:
        return json.loads(_store_path(signature).read_text(encoding="utf-8"))
    except Exception:
        return None


def _save(template: dict) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    _store_path(template["signature"]).write_text(
        json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------- lookup

@dataclass
class TemplateMatch:
    """Read-only snapshot handed to extract_one() -- what's known about this
    template signature at the moment of lookup. Only "active" rules are
    exposed; a rule can be hand-retired (status != "active") by editing its
    JSON file without deleting its history."""
    signature: str
    seller_name: str
    size_headers: list[str]
    occurrences: int
    item_substitutions: list[dict]
    column_shifts: list[dict]
    backend_stats: dict
    model_stats: dict


def lookup(seller_name: str, size_headers: list[str]) -> TemplateMatch | None:
    """Never raises -- first-ever occurrence, a missing file, or a corrupt
    one all just mean no template help this round, the same fail-soft
    posture app/pipeline.py's own _read_json() already uses for every other
    pipeline sidecar file."""
    try:
        signature = _signature(seller_name, size_headers)
        if signature is None:
            return None
        template = _load(signature)
        if template is None:
            return None
        return TemplateMatch(
            signature=signature,
            seller_name=template.get("seller_name", seller_name),
            size_headers=template.get("size_headers", size_headers),
            occurrences=template.get("occurrences", 0),
            item_substitutions=[r for r in template.get("item_substitutions", []) if r.get("status") == "active"],
            column_shifts=[r for r in template.get("column_shifts", []) if r.get("status") == "active"],
            backend_stats=template.get("backend_stats", {}),
            model_stats=template.get("model_stats", {}),
        )
    except Exception:
        traceback.print_exc()
        return None


def preferred_backend(match: TemplateMatch | None) -> str | None:
    """Only returns a preference when BOTH backends have at least one
    recorded sample for this template AND one's average correction rate is
    STRICTLY lower -- a single sample is not enough to prefer one backend
    over an alternative this template has never actually been tried
    against. Not cached on the template file: derived live from
    backend_stats on every call so it can never drift from the stats that
    justify it."""
    if match is None:
        return None
    usable = {name: s for name, s in (match.backend_stats or {}).items() if s.get("uses", 0) >= 1}
    if len(usable) < 2:
        return None
    best_name = min(usable, key=lambda n: usable[n]["avg_correction_rate"])
    others = [s["avg_correction_rate"] for name, s in usable.items() if name != best_name]
    if usable[best_name]["avg_correction_rate"] < min(others):
        return best_name
    return None


# ------------------------------------------------------------- apply (read)

def _best_fuzzy_trigger(text: str, candidates: list[dict], key: str) -> dict | None:
    if not text:
        return None
    best, best_score = None, 0.0
    for cand in candidates:
        trigger = cand.get(key, "")
        if not trigger:
            continue
        score = fuzz.ratio(text, trigger, processor=fz_utils.default_process)
        if score > best_score:
            best, best_score = cand, score
    return best if best_score >= TRIGGER_MATCH_THRESHOLD else None


def _shift_quantities(quantities: dict[str, int], headers: list[str], offset: int) -> dict[str, int]:
    """Replays app/static/app.js's shiftRowHorizontally() exactly: each
    size moves `offset` positions within the form's own header order,
    left to right; anything pushed off either edge is dropped. A size not
    on the header grid at all (e.g. a handwritten overflow column) stays
    put, same as that UI action."""
    result: dict[str, int] = {}
    for size, qty in quantities.items():
        at = headers.index(size) if size in headers else -1
        if at == -1:
            result[size] = qty
            continue
        to = at + offset
        if 0 <= to < len(headers):
            result[headers[to]] = qty
    return result


def apply_corrections(match: TemplateMatch | None, extracted) -> list[dict]:
    """Mutates extracted.items[i].item / .quantities in place for every row
    an active rule's trigger fuzzy-matches, and records that the rule fired
    (times_applied) on this template's stored file. Returns what it
    applied -- consumed twice by the caller: to merge a "template_corrected"
    flag into recount_flags (extract_ollama_cloud.py owns _merge_flag, not
    this module, so the merge itself happens at the call site), and by the
    write path on the NEXT submission to recover each row's true original
    reading (see record_corrections below)."""
    applied: list[dict] = []
    if match is None or not extracted.items:
        return applied

    headers = extracted.size_headers
    for i, item in enumerate(extracted.items):
        sub = _best_fuzzy_trigger(item.item, match.item_substitutions, "raw")
        if sub is not None and sub["corrected"] != item.item:
            observed_raw = item.item
            item.item = sub["corrected"]
            applied.append({
                "row_index": i, "kind": "item_substitution",
                "rule_raw": sub["raw"], "raw": observed_raw, "corrected": sub["corrected"],
                "sizes": [],
                "note": (f"Template memory: item name corrected from a previously confirmed "
                         f"reading of this exact form ('{observed_raw}' -> '{sub['corrected']}')."),
            })

        # Matched against the item name AFTER any substitution above, so a
        # shift rule keyed on the corrected name still finds this row.
        shift = _best_fuzzy_trigger(item.item, match.column_shifts, "item_trigger_display")
        if shift is not None and headers:
            offset = shift["offset"]
            raw_qty = {qp.size: qp.quantity for qp in item.quantities}
            shifted = _shift_quantities(raw_qty, headers, offset)
            if shifted != raw_qty:
                item.quantities = [QuantityPair(size=s, quantity=q) for s, q in shifted.items()]
                sizes = sorted(set(raw_qty) | set(shifted), key=lambda s: (0, int(s)) if s.isdigit() else (1, s))
                applied.append({
                    "row_index": i, "kind": "column_shift",
                    "rule_trigger": shift["item_trigger"], "rule_trigger_display": shift["item_trigger_display"],
                    "raw_quantities": raw_qty, "offset": offset, "sizes": sizes,
                    "note": (f"Template memory: quantities shifted {offset:+d} column(s) to match a "
                             f"previously confirmed reading of this exact form."),
                })

    if applied and match is not None:
        template = _load(match.signature)
        if template is not None:
            fired_subs = {c["rule_raw"] for c in applied if c["kind"] == "item_substitution"}
            fired_shifts = {c["rule_trigger"] for c in applied if c["kind"] == "column_shift"}
            for rule in template.get("item_substitutions", []):
                if rule["raw"] in fired_subs:
                    rule["times_applied"] = rule.get("times_applied", 0) + 1
            for rule in template.get("column_shifts", []):
                if rule["item_trigger"] in fired_shifts:
                    rule["times_applied"] = rule.get("times_applied", 0) + 1
            _save(template)

    return applied


def write_debug_artifact(outdir: Path, stem: str, seller_name: str, size_headers: list[str],
                          match: TemplateMatch | None, backend_used: str | None, model_used: str | None,
                          corrections_applied: list[dict]) -> None:
    """Writes <stem>.template_learning.json unconditionally, including on a
    miss (matched: false) -- useful on its own for a developer confirming
    two photos hash to the same signature. Also the artifact
    app/pipeline.py's _rebuild_payload() reads back into session.payload,
    which is what lets the write path (record_corrections) recover each
    row's true pre-correction reading on the next submission."""
    data = {
        "signature": _signature(seller_name, size_headers),
        "matched": match is not None,
        "occurrences": match.occurrences if match else 0,
        "backend_used": backend_used,
        "model_used": model_used,
        "corrections_applied": corrections_applied,
    }
    try:
        (outdir / f"{stem}.template_learning.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        traceback.print_exc()


# ------------------------------------------------------------- record (write)

def _find_substitution(template: dict, raw: str) -> dict | None:
    return next((r for r in template["item_substitutions"] if r["raw"] == raw), None)


def _confirm_substitution(rule: dict) -> None:
    rule["times_confirmed"] = rule.get("times_confirmed", 0) + 1
    rule["last_confirmed"] = datetime.now().isoformat(timespec="seconds")


def _upsert_substitution(template: dict, raw: str, corrected: str) -> None:
    if not raw or not corrected or raw == corrected:
        return
    now = datetime.now().isoformat(timespec="seconds")
    rule = _find_substitution(template, raw)
    if rule is None:
        template["item_substitutions"].append({
            "raw": raw, "corrected": corrected, "status": "active",
            "times_applied": 0, "times_confirmed": 0, "times_contradicted": 0,
            "first_seen": now, "last_confirmed": now,
        })
    elif rule["corrected"] != corrected:
        # A human just overrode either the raw reading or this rule's own
        # prior correction with something else -- the new value replaces
        # the old one on the SAME rule rather than creating a second,
        # confusing entry for the identical raw trigger.
        rule["corrected"] = corrected
        rule["times_contradicted"] = rule.get("times_contradicted", 0) + 1
        rule["times_confirmed"] = 0
        rule["last_confirmed"] = now


def _find_shift(template: dict, trigger_key: str) -> dict | None:
    return next((r for r in template["column_shifts"] if r["item_trigger"] == trigger_key), None)


def _confirm_shift(rule: dict) -> None:
    rule["times_confirmed"] = rule.get("times_confirmed", 0) + 1
    rule["last_confirmed"] = datetime.now().isoformat(timespec="seconds")


def _upsert_shift(template: dict, trigger_display: str, offset: int) -> None:
    if not trigger_display:
        return
    trigger_key = _normalize(trigger_display)
    now = datetime.now().isoformat(timespec="seconds")
    rule = _find_shift(template, trigger_key)
    if rule is None:
        template["column_shifts"].append({
            "item_trigger": trigger_key, "item_trigger_display": trigger_display, "offset": offset,
            "status": "active", "times_applied": 0, "times_confirmed": 0, "times_contradicted": 0,
            "first_seen": now, "last_confirmed": now,
        })
    elif rule["offset"] != offset:
        rule["offset"] = offset
        rule["times_contradicted"] = rule.get("times_contradicted", 0) + 1
        rule["times_confirmed"] = 0
        rule["last_confirmed"] = now


def _detect_shift_offset(raw_qty: dict[str, int], submitted_qty: dict[str, int], headers: list[str]) -> int | None:
    """The write-path mirror of _shift_quantities: is the reviewer's
    correction fully explained by moving every size a constant number of
    header positions? Only returns an offset when it is the UNIQUE one (of
    _SHIFT_CANDIDATE_OFFSETS) that reproduces submitted_qty exactly --
    mirrors brandlist_match.py's resolve_shift_offset() conservatism (only
    trust a shift finding when no other offset also fits). An ordinary
    freeform quantity edit (not a pure shift) correctly returns None --
    too ambiguous to learn from safely."""
    if not headers or not raw_qty or not submitted_qty:
        return None
    fits = [off for off in _SHIFT_CANDIDATE_OFFSETS if _shift_quantities(raw_qty, headers, off) == submitted_qty]
    return fits[0] if len(fits) == 1 else None


def _fold_rate(stats: dict, name: str, rate: float) -> None:
    """Plain running mean, not a decaying average -- real per-template
    sample counts (a handful of orders) are too small for anything fancier
    than that to mean something."""
    entry = stats.setdefault(name, {"uses": 0, "avg_correction_rate": 0.0})
    entry["avg_correction_rate"] = (entry["avg_correction_rate"] * entry["uses"] + rate) / (entry["uses"] + 1)
    entry["uses"] += 1


def record_corrections(payload: dict, submitted: dict) -> None:
    """The write-path entry point -- called once per successful order
    submission from app/pipeline.py's record_submission(). Diffs each
    page's pre-correction reading (payload["pages"][i], including the
    template_match artifact extract_one() wrote for it) against what the
    reviewer actually submitted, and folds the result into that page's
    template file. Never raises: a bug here must not threaten an
    already-DB-committed order."""
    try:
        _record_corrections(payload, submitted)
    except Exception:
        traceback.print_exc()


def _record_corrections(payload: dict, submitted: dict) -> None:
    submitted_rows = submitted.get("rows") or []
    by_uid = {r["uid"]: r for r in submitted_rows if r.get("uid")}

    for page in payload.get("pages") or []:
        seller_name = page.get("seller_name") or ""
        headers = page.get("raw_size_headers") or []
        signature = _signature(seller_name, headers)
        if signature is None:
            continue  # free-form page, or no seller_name read -- nothing stable to sign on

        template = _load(signature) or _empty_template(signature, seller_name, headers)
        template["last_seen"] = datetime.now().isoformat(timespec="seconds")
        template["occurrences"] = template.get("occurrences", 0) + 1

        page_match = page.get("template_match") or {}
        fired_by_row = {c["row_index"]: c for c in (page_match.get("corrections_applied") or [])}
        backend_used = page_match.get("backend_used")
        model_used = page_match.get("model_used")

        rows_with_qty = 0
        rows_with_diff = 0

        for row in page.get("rows") or []:
            uid = row.get("uid") or ""
            if not uid or "#new" in uid:
                continue  # human-added row -- not a correction of an existing read
            submitted_row = by_uid.get(uid)
            if submitted_row is None:
                continue  # deleted, or zeroed out and dropped by buildSubmitBody()

            fired = fired_by_row.get(row.get("source_index"))
            shown_item = row.get("item") or ""
            shown_qty = {str(k): int(v) for k, v in (row.get("quantities") or {}).items()}
            if shown_qty:
                rows_with_qty += 1

            # -- item name --
            sub_fired = fired if fired and fired["kind"] == "item_substitution" else None
            trigger_raw = sub_fired["rule_raw"] if sub_fired else shown_item
            submitted_item = (submitted_row.get("item") or "").strip()
            if submitted_item and submitted_item != shown_item:
                rows_with_diff += 1
                _upsert_substitution(template, trigger_raw, submitted_item)
            elif submitted_item and sub_fired:
                rule = _find_substitution(template, trigger_raw)
                if rule is not None:
                    _confirm_substitution(rule)

            # -- quantities --
            shift_fired = fired if fired and fired["kind"] == "column_shift" else None
            trigger_display = shift_fired["rule_trigger_display"] if shift_fired else (submitted_item or shown_item)
            raw_qty = shift_fired["raw_quantities"] if shift_fired else shown_qty
            submitted_qty = {str(k): int(v) for k, v in (submitted_row.get("quantities") or {}).items()}
            if submitted_qty != shown_qty:
                offset = _detect_shift_offset(raw_qty, submitted_qty, headers)
                if offset is not None:
                    rows_with_diff += 1
                    _upsert_shift(template, trigger_display, offset)
                # else: not a pure shift -- too ambiguous to learn from, and
                # not evidence against an existing rule either (this row may
                # have BOTH a shift and an unrelated handwriting fix).
            elif shift_fired:
                rule = _find_shift(template, _normalize(trigger_display))
                if rule is not None:
                    _confirm_shift(rule)

        if rows_with_qty:
            rate = rows_with_diff / rows_with_qty
            if backend_used:
                _fold_rate(template.setdefault("backend_stats", {}), backend_used, rate)
            if model_used:
                _fold_rate(template.setdefault("model_stats", {}), model_used, rate)

        _save(template)
