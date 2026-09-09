"""
Standalone head-to-head: today's production hybrid-quantities backend
(PaddleOCR + grid.py's order-preserving DP, via
extract_ollama_cloud._hybrid_ocr_quantities) vs. the experimental
LightOnOCR-2 alternative (hybrid_quantities_lighton.lighton_hybrid_quantities)
-- run against the IDENTICAL mistral main-call output, so any difference
in the two reports is the quantity-reading backend, not sampling
variance between two separate mistral calls.

Does NOT touch extract_one(), extract_ollama_cloud.py's CLI, or the
review app -- neither backend's output is written back into any
production file. This exists purely so LightOnOCR-2 can be evaluated
against real production behavior before earning a switch, per the
project's "don't break the almost-working PaddleOCR pipeline" rule.

Usage:
    .venv\\Scripts\\python.exe compare_hybrid_backends.py
    .venv\\Scripts\\python.exe compare_hybrid_backends.py "Images/sample 5-scanned.jpg"
    .venv\\Scripts\\python.exe compare_hybrid_backends.py --outdir compare_out Images/*.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

import extract_ollama_cloud as eoc
from hybrid_quantities_lighton import lighton_hybrid_quantities

DEFAULT_IMAGES = [
    Path("Images/sample 5-scanned.jpg"),
    Path("Images/sample 12-scanned.jpg"),
    Path("Images/sample 13-scanned.jpg"),
    Path("Images/sample 4-scanned.jpg"),
]


def run_main_call(client, model: str, image_path: Path, system_prompt: str, usage_log: Path):
    """Replicates only the main-call portion of extract_one() (not its
    hybrid/recount/brandlist stages) -- gets one ExtractedForm skeleton
    both backends below will be run against, so neither result is
    confounded by a second, separately-sampled mistral call."""
    if "mistral" in model.lower() and False:  # preprocess-image not used for this comparison
        image_bytes, _media_type = eoc._prepare_image_for_mistral(image_path)
    else:
        image_bytes, _media_type = eoc._prepare_image_exif_safe(image_path)

    main_schema_cls = eoc.MistralExtractedForm if "mistral" in model.lower() else eoc.ExtractedForm
    extracted = None
    error = None
    letter_size_hints: set[int] = set()
    struck_out_hints: set[int] = set()
    for attempt in range(eoc.MAIN_CALL_MAX_RETRIES + 1):
        parsed, error, usage = eoc._call_schema(client, model, system_prompt, eoc.USER_PROMPT, [image_bytes], main_schema_cls, eoc.MAX_TOKENS)
        print(f"  usage (main, attempt {attempt + 1}): prompt={usage.get('prompt_eval_count')} eval={usage.get('eval_count')} {usage.get('duration_seconds', 0):.1f}s" + (f" ERROR: {error}" if error else ""))
        eoc._log_usage(usage_log, image_path.name, model, f"main-attempt{attempt + 1}", usage, error)
        if error:
            continue
        if main_schema_cls is eoc.MistralExtractedForm:
            letter_size_hints = {i for i, it in enumerate(parsed.items) if it.letter_sizes}
            struck_out_hints = {i for i, it in enumerate(parsed.items) if it.struck_out}
            parsed = eoc._mistral_form_to_extracted_form(parsed)
        total_value = sum(qp.quantity for it in parsed.items for qp in it.quantities)
        if parsed.items and total_value == 0:
            extracted = parsed
            continue
        extracted = parsed
        error = None
        break

    if extracted is None:
        raise RuntimeError(f"Main extraction call failed: {error}")
    eoc._normalize_fused_size_headers(extracted)
    return extracted, letter_size_hints, struck_out_hints


def fmt_map(m: dict[str, int]) -> str:
    if not m:
        return "{}"
    return "{" + ", ".join(f"{k}:{v}" for k, v in sorted(m.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 0)) + "}"


def compare_one(client, model: str, image_path: Path, system_prompt: str, outdir: Path, usage_log: Path) -> dict:
    print(f"\n=== {image_path.name} ===")
    extracted, letter_size_hints, struck_out_hints = run_main_call(client, model, image_path, system_prompt, usage_log)
    n_items = len(extracted.items)
    print(f"  main call: {n_items} items, size_headers={extracted.size_headers}")

    vlm_raw = {i: {qp.size: qp.quantity for qp in it.quantities} for i, it in enumerate(extracted.items)}

    extracted_paddle = extracted.model_copy(deep=True)
    paddle_qty, paddle_flags = eoc._hybrid_ocr_quantities(image_path, extracted_paddle, outdir, letter_size_hints)

    extracted_lighton = extracted.model_copy(deep=True)
    lighton_qty, lighton_flags = lighton_hybrid_quantities(image_path, extracted_lighton, outdir, struck_out_hints, letter_size_hints)

    rows = []
    for i, item in enumerate(extracted.items):
        p = paddle_qty.get(i, vlm_raw[i])
        l = lighton_qty.get(i, vlm_raw[i])
        agree = p == l
        rows.append(dict(
            idx=i, item=item.item, style=item.type, row_total=item.row_total,
            vlm_raw=vlm_raw[i], paddle=p, paddle_flag=paddle_flags.get(i),
            lighton=l, lighton_flag=lighton_flags.get(i), agree=agree,
        ))
        marker = "==" if agree else "!="
        print(f"  [{i:2d}] {item.item[:28]:28s} vlm={fmt_map(vlm_raw[i]):40s} paddle={fmt_map(p):40s} {marker} lighton={fmt_map(l)}")
        if paddle_flags.get(i):
            print(f"        paddle flag:   {paddle_flags[i].get('note')}")
        if lighton_flags.get(i):
            print(f"        lighton flag:  {lighton_flags[i].get('note')}")

    n_agree = sum(1 for r in rows if r["agree"])
    print(f"  agreement: {n_agree}/{n_items} rows identical between the two backends")
    return dict(image=image_path.name, n_items=n_items, n_agree=n_agree, rows=rows)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", help="Image path(s) to compare (default: the 4 regression test forms)")
    parser.add_argument("--outdir", default="compare_hybrid_out", help="Where debug artifacts (.hybrid_debug.json, .lighton_debug.json) land")
    parser.add_argument("--model", default=eoc.DEFAULT_MODEL)
    parser.add_argument("--usage-log", default="usage_log_compare_hybrid.csv")
    args = parser.parse_args()

    client = eoc.get_client()
    style_codes: list[str] = []
    known_sizes: list[int] = []
    try:
        style_codes = eoc.brandlist_match.known_style_codes()
        known_sizes = eoc.brandlist_match.known_numeric_sizes()
    except Exception as exc:
        print(f"Warning: brandlist DB unavailable ({exc}) -- using built-in style codes only.", file=sys.stderr)
    system_prompt = eoc.build_system_prompt(style_codes)
    if "mistral" in args.model.lower():
        system_prompt += eoc.MISTRAL_PROMPT_ADDENDUM_BASE + eoc._sizes_past_header_bullet(known_sizes)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    usage_log = Path(args.usage_log)

    images = [Path(p) for p in args.images] if args.images else DEFAULT_IMAGES
    reports = []
    for img in images:
        if not img.exists():
            print(f"skip: {img} not found", file=sys.stderr)
            continue
        try:
            reports.append(compare_one(client, args.model, img, system_prompt, outdir, usage_log))
        except Exception as exc:
            print(f"FAILED on {img.name}: {exc}", file=sys.stderr)

    (outdir / "comparison_report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {outdir / 'comparison_report.json'}")


if __name__ == "__main__":
    main()
