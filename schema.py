"""
Two schemas now:

- FormMeta: stage-A output. Just structure — item names, style codes,
  size headers, party info, notes. No quantities. This is the part that
  has worked reliably in every attempt so far (party_name/date/notes were
  always right even when quantities were badly misaligned), so it stays
  as one schema-constrained call.

- OrderForm / OrderItem: the FINAL output shape you asked for. Same as
  before. Now assembled in Python (extract_ollama.py) from FormMeta +
  per-row quantity parsing, rather than asking the model to produce the
  whole thing in one shot.
"""

from __future__ import annotations

import re
from typing import Dict, List

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# Safety net for when the model merges the Style column into the item name
# anyway despite the prompt (e.g. "Trend Trunk IE" instead of item="Trend
# Trunk", type="IE"). Deliberately a whitelist, not a generic "strip
# trailing caps" regex — some item names legitimately end in an all-caps
# abbreviation (e.g. "Image FCD"), which must NOT be split.
KNOWN_STYLE_CODES = {"IE", "OE", "RN", "RNS"}
_TRAILING_CODE_RE = re.compile(r"^(?P<name>.+?)\s+(?P<code>[A-Z]{2,4})$")

# A real size header on these forms is either a plain number ("45", "105")
# or, more rarely, a clothing letter-size code -- never a running-total
# column label. Confirmed necessary 2026-09-01: a live Stage A call
# included the form's own trailing "Total Dozen" column as a 14th
# size_headers entry ("Total"), which corrupted every row's column-spacing
# math downstream (an empty read on one row, a fabricated "Total" quantity
# key on another, systematic +1 column shifts elsewhere) -- size_headers
# had no validation at all before this. LETTER_SIZE_CODES matches the
# vocabulary ocr_cell_read.py's stacked-label detection also uses, so a
# genuine letter-coded header (rare, but real -- see CLAUDE.md's sample 3
# MM K4532 case) isn't dropped here just because it isn't numeric.
LETTER_SIZE_CODES = {"XS", "S", "M", "L", "XL", "XXL", "2XL", "3XL", "4XL"}
_NUMERIC_HEADER_RE = re.compile(r"^\d{1,3}$")

# Safety net for when the model leaks its normalization reasoning into the
# order_date field (e.g. "30/13/26 (normalize to DD/MM/YYYY: 30/01/2026)")
# despite the prompt asking for only the final value. Keeps the first
# date-like token — the model's direct reading of the form — and drops
# everything else.
_DATE_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4}")


class ItemStub(BaseModel):
    item: str = Field(description="Product/article name exactly as written. If the row uses a ditto mark (—\"—, -\"-), write out the FULL inherited name from the row above, not just the modifier word.")
    type: str = Field(default="", description="Style/variant code from the Style column (IE, OE, RN, RNS, etc). Empty string if not present.")

    @model_validator(mode="after")
    def _split_trailing_style_code(self) -> "ItemStub":
        # ditto row where the model wrote the Style column's value into
        # item instead of leaving item blank — leave item empty so
        # FormMeta._forward_fill_ditto_item_names can carry the real name
        # forward from the row above.
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
            # model filled `type` correctly but ALSO left the code stuck on `item`
            self.item = m.group("name")
        return self

    @model_validator(mode="after")
    def _clear_numeric_type(self) -> "ItemStub":
        # Safety net for a confirmed real Stage A hallucination (2026-09-01,
        # sample 3.jpeg): on a form where a secondary age/chest-equivalent
        # header row sits close to a mostly blank/faint Style column, the
        # model can read that header row's own number sequence as if it
        # were each item's style code (e.g. every row in order getting
        # "18", "20", "22", ... -- the header sequence itself, not real
        # per-row data). A real style code on these forms is always
        # alphabetic (RN, RNS, OE, IE, etc.), never a bare number, so
        # dropping a purely-numeric `type` is safe and can't discard a
        # genuine style code.
        if self.type.isdigit():
            self.type = ""
        return self


class FormMeta(BaseModel):
    seller_name: str = Field(default="", description="Letterhead/seller company name, NOT the buyer.")
    party_name: str = Field(default="", description="Buyer/customer name, handwritten in the Party Name box.")
    party_city: str = Field(default="", description="Buyer's city, handwritten in the Party Name box. NOT the seller's printed address city.")
    order_no: str = Field(default="", description="Order form number if present, else empty string.")
    order_date: str = Field(default="", description="Date normalized to DD/MM/YYYY, assuming 20xx for 2-digit years.")
    size_headers: List[str] = Field(default_factory=list, description="Every size column header across the top of the table, in left-to-right order, exactly as printed (e.g. ['45','50','55',...]).")
    items: List[ItemStub] = Field(default_factory=list, min_length=1, description="Every product/article row, top to bottom, in order.")
    notes: List[str] = Field(default_factory=list, description="Page-level handwritten notes (e.g. 'Old Rate Supply Only'), and anything illegible/unusual not tied to a specific row's quantities.")

    @field_validator("order_date")
    @classmethod
    def _clean_order_date(cls, v: str) -> str:
        if not v:
            return v
        m = _DATE_RE.search(v)
        return m.group(0) if m else v

    @field_validator("size_headers")
    @classmethod
    def _drop_non_size_headers(cls, v: list[str]) -> list[str]:
        return [h for h in v if _NUMERIC_HEADER_RE.match(h) or h.upper() in LETTER_SIZE_CODES]

    @model_validator(mode="after")
    def _forward_fill_ditto_item_names(self) -> "FormMeta":
        # A ditto mark in Particulars (item name blank/empty) means "same
        # item as the row above" — safety net for when the model reads the
        # Style column's value on that row but leaves item blank instead of
        # carrying the name forward itself.
        #
        # A ditto+modifier schema field (ditto: bool, composing the full
        # name from last_item + modifier word here instead of trusting the
        # model to write it out inline) was tried and reverted
        # (2026-08-03): the model never reliably set the flag (still came
        # back bare "Plain" instead of "Fairlady Plain"), and the prompt
        # rewording needed for it caused an unrelated regression elsewhere
        # ("Fairlady Print" misread as "Fairly Print"). Net negative, not
        # worth the added schema complexity -- reverted to this simpler,
        # already-working blank-item forward-fill only. The bare-modifier-
        # word case (ditto + word, e.g. "Plain") remains unfixed.
        last_item = ""
        for stub in self.items:
            if stub.item:
                last_item = stub.item
            elif last_item:
                stub.item = last_item
        return self


class OrderItem(BaseModel):
    item: str
    type: str = ""
    quantities: Dict[str, int] = Field(default_factory=dict)


class OrderForm(BaseModel):
    party_name: str = ""
    order_no: str = ""
    order_date: str = ""
    items: List[OrderItem] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    source_file: str = ""


def validate_meta(raw: dict) -> tuple[FormMeta | None, list[str]]:
    try:
        return FormMeta(**raw), []
    except ValidationError as exc:
        errors = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()]
        return None, errors
