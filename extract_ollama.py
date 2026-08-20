#!/usr/bin/env python3
"""
extract_ollama.py — v6: per-row crop (v5) read via OCR instead of the VLM.

v3 (per-row prompting on the full image) fixed whole-table column drift but
still let the model misread the header row as data, or bleed one row's
numbers into another, since it still had to *find* the right row inside the
full image every call. A v4 attempt tried fixing that by having the VLM
itself estimate row positions to crop around — tested twice (even-split,
then per-row bounding boxes) and both made things WORSE, because the
model's spatial grounding turned out to be unreliable whether reading
cells directly or estimating boxes.

v5 replaced the VLM position guess with deterministic classical CV
(grid.py): row-boundary detection via density-peak analysis, confirmed
against a real form. That fixed *cropping* -- but a fresh end-to-end run
(2026-08-03, see CLAUDE.md) showed the 7B VLM still misread cells even
given a clean, correctly-isolated row: hallucinating the form's second
header row as data in blank columns, and shifting real values into the
wrong column. Two rounds of prompt tweaking (v3, v5) both hit this same
ceiling, so v6 stops asking the VLM to read quantities at all.

v6 reads each row+header crop with PaddleOCR instead (ocr_cell_read.py):
OCR only reports text where ink exists (no hallucination of blank cells),
and every recognized digit's own pixel position is matched to the nearest
header column via an order-preserving assignment (tolerant of the mild
per-row perspective drift a photographed page has). Falls back to the VLM
per-row call for: (a) whole images where grid detection didn't find clean
row boundaries at all, and (b) individual rows where OCR fails to even
read the header labels on its own crop.

Prerequisite:
    ollama pull qwen2.5vl:7b
    ollama serve   (usually already running as a background service)
    PaddleOCR model weights download on first run (needs internet once)

Usage:
    python3 extract_ollama.py "images/sample 5.jpeg"
    python3 extract_ollama.py images/ --outdir extracted
    python3 extract_ollama.py images/ --model qwen2.5vl:32b

Cost per image: 1 VLM call (stage A) + N OCR reads (one per item row,
local, no model call) -- plus a VLM row call for any row/image that falls
back per above.

Output per image:
    extracted/<name>.json           -- final structured data
    extracted/<name>.stageA.json    -- stage-A structure-only output (debugging)
    extracted/<name>.rowcrops/      -- the header+row crop OCR'd for each row, if grid
                                        detection succeeded (debugging: open these to see exactly
                                        what was read; absent for an image that fell back to v3)
    extracted/<name>.rows.txt       -- every row's raw OCR tokens (position + confidence) or VLM
                                        response (debugging: if one row looks wrong, check it here)
    extracted/review.csv            -- flattened, one row per (item, size)
"""

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

from ollama import chat, ResponseError
from PIL import Image

from grid import iter_row_boundary_candidates
from ocr_cell_read import read_rows, count_headers_found, ocr_row, item_alignment_ok
from schema import FormMeta, OrderForm, OrderItem, validate_meta
from prompt import build_stage_a_prompt, build_row_prompt, STAGE_A_SCHEMA

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_RETRIES = 2
# size:value pair, e.g. "80:50" or "70:x" or "45:blank" — tolerant of extra spaces
PAIR_RE = re.compile(r"(\d+)\s*:\s*([A-Za-z0-9\-]+)")


def stage_a(model: str, image_bytes: bytes) -> FormMeta:
    last_error = None
    raw = ""
    for _ in range(MAX_RETRIES + 1):
        try:
            response = chat(
                model=model,
                messages=[{"role": "user", "content": build_stage_a_prompt(), "images": [image_bytes]}],
                format=STAGE_A_SCHEMA,
                options={"temperature": 0},
            )
        except ResponseError as exc:
            raise RuntimeError(
                f"Ollama request failed: {exc}. Is 'ollama serve' running, and have you "
                f"pulled the model with `ollama pull {model}`?"
            ) from exc

        raw = response.message.content
        meta, errors = validate_meta(json.loads(raw)) if _is_json(raw) else (None, ["not valid JSON"])
        if meta is not None:
            return meta
        last_error = errors

    raise RuntimeError(f"Stage A failed after {MAX_RETRIES + 1} attempts: {last_error}\nLast output:\n{raw}")


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def _crop_band(image: Image.Image, top_f: float, bottom_f: float) -> Image.Image:
    width, height = image.size
    y_top = max(0, int(top_f * height))
    y_bottom = min(height, int(bottom_f * height))
    if y_bottom <= y_top:
        y_bottom = min(height, y_top + 1)
    return image.crop((0, y_top, width, y_bottom))


def build_header_crop(image: Image.Image, boundaries: list[float]) -> tuple[Image.Image, bytes]:
    """The header band alone (both PIL image and PNG bytes) -- split out
    from build_row_crop so extract_one() can sanity-check it (see
    ocr_cell_read.count_headers_found) once per image, before building N
    row crops around a position that might be wrong."""
    avg_row_h_f = (boundaries[-1] - boundaries[0]) / (len(boundaries) - 1)
    header_crop = _crop_band(image, max(0.0, boundaries[0] - avg_row_h_f * 1.3), boundaries[0])
    buf = io.BytesIO()
    header_crop.save(buf, format="PNG")
    return header_crop, buf.getvalue()


def build_row_crop(
    image: Image.Image, boundaries: list[float], index: int, header_crop: Image.Image, margin_frac: float = 0.15
) -> tuple[bytes, int]:
    """Header band stitched above this row's band, both from CV-detected
    boundaries (real ruled lines, not a guess) — gives Stage B the header's
    size numbers for column labeling without showing it every other row.
    Returns (png_bytes, header_height_px) -- the height is needed by the
    OCR path (ocr_cell_read.read_row_cells) to know where the header band
    ends and the actual row data starts. `header_crop` comes from
    build_header_crop() -- computed once per image, not once per row."""
    width, _ = image.size
    row_top_f, row_bottom_f = boundaries[index], boundaries[index + 1]
    margin_f = (row_bottom_f - row_top_f) * margin_frac
    row_crop = _crop_band(image, row_top_f - margin_f, row_bottom_f + margin_f)

    stitched = Image.new(header_crop.mode, (width, header_crop.height + row_crop.height), "white")
    stitched.paste(header_crop, (0, 0))
    stitched.paste(row_crop, (0, header_crop.height))

    buf = io.BytesIO()
    stitched.save(buf, format="PNG")
    return buf.getvalue(), header_crop.height


def stage_b_row(model: str, image_bytes: bytes, item: str, style: str, headers: list[str], cropped: bool = False) -> tuple[dict[str, int], str]:
    """One call focused on a single row (a header+row crop when grid
    detection succeeded, otherwise the full image). Returns {size: qty} for filled cells only."""
    response = chat(
        model=model,
        messages=[{"role": "user", "content": build_row_prompt(item, style, headers, cropped), "images": [image_bytes]}],
        options={"temperature": 0},
    )
    raw_line = response.message.content.strip()

    pairs = PAIR_RE.findall(raw_line)
    if not pairs:
        # Model sometimes drops the "size:" labels and just lists values
        # positionally (seen mainly on header+row crops) -- if the token
        # count matches the header count, zip them back up rather than
        # discarding what may be a correctly-read row.
        tokens = [t.strip() for t in raw_line.split(",")]
        if len(tokens) == len(headers):
            pairs = list(zip(headers, tokens))

    quantities: dict[str, int] = {}
    for size, value in pairs:
        value = value.lower()
        if value in ("blank", "x", "-"):
            continue
        try:
            qty = int(value)
        except ValueError:
            continue  # e.g. a fraction like "60/10" slipped through — not representable as int
        if qty <= 0:
            continue  # garment quantities can't be negative/zero; "-12" is a misformatted blank marker, not a real value
        quantities[size] = qty
    return quantities, raw_line


def extract_one(model: str, image_path: Path, outdir: Path) -> OrderForm:
    image_bytes = image_path.read_bytes()

    meta = stage_a(model, image_bytes)
    (outdir / f"{image_path.stem}.stageA.json").write_text(
        json.dumps(meta.model_dump(mode="json"), indent=2, ensure_ascii=False)
    )

    n_items = len(meta.items)
    image = None
    boundaries = None
    header_crop = None
    crop_dir = None
    bad_row_indices: set[int] = set()

    # grid.py tries several row-detection strategies; validate EACH
    # candidate independently and keep the best-scoring one, rather than
    # trusting the first "clean" result grid.py returns. Two checks, not
    # one -- both confirmed necessary on real forms, not just theoretical:
    #  1. Does the header crop actually contain the real header text?
    #     (Catches: a whole run of lines found in entirely the wrong part
    #     of the page, e.g. the letterhead/party-info-box area.)
    #  2. Does EVERY row's own OCR'd item name plausibly match what Stage A
    #     expects there? (Catches: the header position is right but
    #     individual row boundaries still drifted mid-table -- a
    #     missing/extra candidate line shifts every later row a position
    #     or two. The header check alone can't see this, and confirmed a
    #     SAMPLE of rows isn't reliable either: the drift was sparse/
    #     localized to a couple of rows, not uniform, so a small sample
    #     could randomly miss or hit them. Checking every row is only
    #     expensive in absolute terms -- in practice at most one or two
    #     candidates ever pass the header check, so this is a handful of
    #     extra OCR calls, not N_CANDIDATES x N_ROWS.)
    # Rows that fail check 2 are tracked individually (bad_row_indices)
    # rather than rejecting the whole candidate -- confirmed a candidate
    # can be right for 16 of 18 rows, and using OCR for those 16 while
    # falling back to the VLM for the 2 bad ones beats discarding all 18.
    # Every row's OCR call here is cached (row index -> ocr_row() result)
    # and, if this candidate wins, handed to read_rows() below instead of
    # being re-OCR'd from scratch -- confirmed (2026-08-04) that a single
    # OCR call costs ~14s on this machine's CPU-only PaddleOCR build, so
    # re-reading the same crop pixels a second time was pure waste (and,
    # separately, a full-image VLM fallback call costs ~87s -- also
    # confirmed -- which is why recovering rows to the OCR path at all,
    # via the header-collision fix above, matters far more for wall-clock
    # time than this caching does; this just avoids doubling what's left).
    best_good_count = -1
    best_candidate = None
    best_bad_indices: set[int] = set()
    best_row_cache: dict[int, tuple] = {}
    for cand_boundaries, cand_skew in iter_row_boundary_candidates(image_bytes, n_items):
        candidate_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        candidate_image = candidate_image.rotate(cand_skew, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        candidate_header_crop, header_crop_bytes = build_header_crop(candidate_image, cand_boundaries)
        headers_found = count_headers_found(header_crop_bytes, meta.size_headers)
        if headers_found < len(meta.size_headers) * 0.7:
            continue  # header position itself is already implausible

        cand_bad_indices = set()
        cand_row_cache: dict[int, tuple] = {}
        for idx in range(n_items):
            row_bytes, header_h_px = build_row_crop(candidate_image, cand_boundaries, idx, candidate_header_crop)
            result = ocr_row(row_bytes, header_h_px, meta.size_headers, score_threshold=0.5)
            cand_row_cache[idx] = result
            _, _, item_name_ocr, _ = result
            if not item_alignment_ok(item_name_ocr, meta.items[idx].item):
                cand_bad_indices.add(idx)

        good_count = n_items - len(cand_bad_indices)
        if good_count < n_items * 0.5:
            continue  # too much of this candidate is wrong to be worth using at all
        if good_count > best_good_count:
            best_good_count = good_count
            best_candidate = (candidate_image, cand_boundaries, candidate_header_crop)
            best_bad_indices = cand_bad_indices
            best_row_cache = cand_row_cache
        if good_count == n_items:
            break  # already perfect -- no remaining candidate config can beat this, stop paying for more OCR calls

    if best_candidate is not None:
        image, boundaries, header_crop = best_candidate
        bad_row_indices = best_bad_indices
        crop_dir = outdir / f"{image_path.stem}.rowcrops"
        crop_dir.mkdir(parents=True, exist_ok=True)

    row_log_lines = []
    items: list[OrderItem] = []

    if image is not None:
        # Built and OCR'd together (not per-row independently) -- drift
        # self-calibration in ocr_cell_read.read_rows() needs every row's
        # initial fit before it can correct any of them.
        row_inputs: list[tuple[bytes, int, float]] = []
        for i in range(n_items):
            row_bytes, header_h_px = build_row_crop(image, boundaries, i, header_crop)
            safe_name = re.sub(r'[<>:"/\\|?*]', "_", meta.items[i].item or "blank")
            (crop_dir / f"{i:02d}_{safe_name}.png").write_bytes(row_bytes)
            row_mid = (boundaries[i] + boundaries[i + 1]) / 2
            row_inputs.append((row_bytes, header_h_px, row_mid))

        ocr_results = read_rows(row_inputs, meta.size_headers, precomputed=best_row_cache)

        for i, stub in enumerate(meta.items):
            quantities, raw_line, item_name_ocr = ocr_results[i]
            needs_full_fallback = i in bad_row_indices or (not quantities and raw_line.startswith("(only found"))
            if needs_full_fallback:
                # Either OCR couldn't even read the header band on this
                # specific crop, or this row was flagged during grid
                # selection as one where the crop's own content doesn't
                # match the expected item (individual row boundary drift
                # within an otherwise-good candidate) -- either way, fall
                # back to the VLM on the FULL original image, not the same
                # crop OCR/the alignment check already found suspect
                # (retrying a possibly-broken crop with a different reader
                # doesn't help; confirmed this used to just reproduce the
                # same failure).
                reason = "row alignment check failed" if i in bad_row_indices else "OCR header read failed"
                quantities, raw_line = stage_b_row(model, image_bytes, stub.item, stub.type, meta.size_headers, cropped=False)
                raw_line = f"[{reason}, fell back to VLM on full image] {raw_line}"

            item_name = stub.item
            # Cross-check: Stage A (VLM) confirmed to occasionally misread
            # short alphanumeric item codes (e.g. "F.G-3005" -> "F.G1-3005")
            # even though it reads natural-language item names correctly.
            # Scoped to names containing a digit so ditto-composed names
            # ("Fairlady Plain" etc, never digits on this form) are untouched.
            if item_name_ocr and re.search(r"\d", item_name) and item_name_ocr.strip().lower() != item_name.strip().lower():
                raw_line = f"[item name overridden: Stage A read {item_name!r}, OCR crop read {item_name_ocr!r}] {raw_line}"
                item_name = item_name_ocr.strip()

            row_log_lines.append(f"{item_name} | {stub.type} -> {raw_line}")
            items.append(OrderItem(item=item_name, type=stub.type, quantities=quantities))
    else:
        for stub in meta.items:
            quantities, raw_line = stage_b_row(model, image_bytes, stub.item, stub.type, meta.size_headers, cropped=False)
            row_log_lines.append(f"{stub.item} | {stub.type} -> {raw_line}")
            items.append(OrderItem(item=stub.item, type=stub.type, quantities=quantities))

    (outdir / f"{image_path.stem}.rows.txt").write_text("\n".join(row_log_lines), encoding="utf-8")

    form = OrderForm(
        party_name=meta.party_name,
        order_no=meta.order_no,
        order_date=meta.order_date,
        items=items,
        notes=meta.notes,
        source_file=image_path.name,
    )
    return form


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
    parser = argparse.ArgumentParser(description="Extract structured data from order form photos via local Ollama + PaddleOCR (v6, per-row).")
    parser.add_argument("input", help="Path to a single image, or a folder of images. Quote paths containing spaces.")
    parser.add_argument("--outdir", default="extracted", help="Output directory (default: ./extracted)")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="Ollama model tag (default: qwen2.5vl:7b)")
    args = parser.parse_args()

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

    all_review_rows = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Extracting {img_path.name} ...", flush=True)
        try:
            form = extract_one(args.model, img_path, outdir)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            continue

        json_out = outdir / f"{img_path.stem}.json"
        json_out.write_text(json.dumps(form.model_dump(mode="json", exclude={"source_file"}), indent=2, ensure_ascii=False))

        n_qty = sum(len(it.quantities) for it in form.items)
        print(f"  -> {json_out.name}  ({len(form.items)} items, {n_qty} size/qty cells, "
              f"{len(form.notes)} notes)")

        all_review_rows.extend(flatten_for_review(form))

    if all_review_rows:
        review_csv = outdir / "review.csv"
        fieldnames = list(all_review_rows[0].keys())
        with review_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_review_rows)
        print(f"\nWrote {review_csv} ({len(all_review_rows)} rows).")


if __name__ == "__main__":
    main()
