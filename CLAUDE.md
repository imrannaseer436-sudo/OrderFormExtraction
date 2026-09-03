# OrderFormV2 — Order Form Extraction Pipeline

## Goal

Automate data entry from photos of garment order forms (mixed
handwritten/printed/typed, dense size-quantity tables) into clean
structured JSON for DB upload, with a human review step before upload.

Target output shape (`OrderForm` in [schema.py](schema.py)):

```json
{
  "party_name": "M.K Enterprises",
  "order_no": "",
  "order_date": "30/03/2026",
  "items": [
    {"item": "Trend Trunk", "type": "IE", "quantities": {"90": 68, "95": 30, "100": 50}}
  ]
}
```

## Production pipeline: Ollama Cloud + mistral-large-3:675b

**[extract_ollama_cloud.py](extract_ollama_cloud.py) is the production
entry point.** It sends each order-form photo to `mistral-large-3:675b`
on Ollama Cloud for one schema-constrained extraction call, then corrects
that model's known weak spot (column/row positional drift on dense
tables) with a deterministic, OCR-grounded quantity re-read, then
cross-checks item names/sizes against the real product catalog. This
replaced an earlier local Ollama + PaddleOCR pipeline and a
pay-per-token Claude API pipeline — both are still in the repo (see
"Other pipelines kept in this repo" below) but neither is the production
path. **Full narrative of how this pipeline reached its current design —
every model tried, every bug found, every fix that was tried and
reverted — is in [HISTORY.md](HISTORY.md).** This file only documents
what's true *now*.

### Pipeline stages

1. **Main call** (1 per image, schema-constrained via `format=`): reads
   party info, order no./date, size headers, and every item row
   (name, style/type, quantities, per-row printed total if present) in
   one shot. Mistral-specific schema extensions
   (`MistralExtractedForm`/`MistralExtractedItem`) add three fields the
   base Claude-pipeline schema doesn't need: `date_present` (stops a
   diary-page "MONDAY" label from being fabricated into a fake date),
   `struck_out` (a cancelled row's quantities are zeroed rather than
   guessed), and `letter_sizes` (flags a row using clothing letter sizes
   S/M/L/XL/XXL instead of the form's printed numeric grid).
2. **Hybrid OCR quantity correction** (`--hybrid-quantities`, **on by
   default**): a real PaddleOCR pass over the image measures the actual
   pixel position of every printed header and every handwritten mark,
   then assigns marks to headers via an order-preserving DP (not
   independent nearest-neighbor — a photographed page's perspective
   drift can otherwise tie-break a digit onto the wrong column). This is
   the main accuracy lever for this pipeline: it corrects mistral's
   column-position drift, recovers handwritten columns past the printed
   grid (e.g. a form running out of headers and writing `105`/`110` by
   hand), resolves letter-coded sizes to real catalog numbers, and fixes
   row-to-item misattribution on dense tables via row content-matching
   and printed-total-guided cluster reassignment. It reconciles its own
   reading against the VLM's own per-row reading rather than
   unconditionally overriding it (see `_reconcile_hybrid_with_vlm`) —
   exact agreement ships silently; a value-only conflict on the same
   column set is arbitrated by the row's own printed total; anything
   else is flagged for human review rather than guessed. Only runs when
   the image has a real shared header row (`size_headers` from the main
   call) — a free-form page with no printed grid skips this stage
   entirely and the VLM's own reading stands.
3. **Brandlist DB cross-check** (`--no-brandlist-check` to skip): a free,
   local lookup against the real SQL Server product catalog
   (`brandlist_match.py`). Surfaces fuzzy item-name/style suggestions,
   auto-fills a blank style code when the matched product has exactly
   one, resolves letter sizes to real catalog numbers once a product
   match is trustworthy, flags (never auto-corrects) a likely
   column-shift or an out-of-catalog-range size, and cross-checks
   `party_name` against the real buyer table for non-ESSA form
   templates whose letterhead/handwritten-name roles are reversed from
   the usual convention.
4. **Human review** (manual step, not auto-run):
   [generate_review.py](generate_review.py) builds a self-contained HTML
   page per image — the source photo next to an editable item×size grid
   laid out like the physical form, with every hybrid/brandlist flag
   rendered as an inline note — so a reviewer checks output against the
   photo directly rather than a flat JSON dump, and can export corrected
   JSON from the page itself.

### Setup

```powershell
py -3.11 -m venv .venv          # paddlepaddle has no 3.13+ wheels yet
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.env` (not committed — see `.gitignore`):

```
OLLAMA_API_KEY=...      # required — ollama.com/settings/keys
SERVER=...               # optional — enables brandlist DB cross-check
DB=...                    # (all four required together, or all omitted)
USER=...
PASSWORD=...
ANTHROPIC_API_KEY=...     # only needed to run extract_claude.py directly
```

If the DB vars are absent, `extract_ollama_cloud.py` prints a warning and
runs without the catalog cross-check (item names/sizes go unverified,
but the pipeline still produces output) — it does not fail outright.

### Usage

```powershell
.venv\Scripts\python.exe extract_ollama_cloud.py "Images\sample 5.jpeg"
.venv\Scripts\python.exe extract_ollama_cloud.py Images\ --outdir extracted_ollama_cloud

# human review page, once JSON output exists (not auto-generated):
.venv\Scripts\python.exe generate_review.py "sample 5" --outdir extracted_ollama_cloud --images Images
```

### CLI defaults (as of the 2026-09-03 production cutover)

| flag | default | notes |
|---|---|---|
| `--model` | `mistral-large-3:675b` | see HISTORY.md for the model comparison this was chosen from |
| `--hybrid-quantities` | **on** (`--no-hybrid-quantities` to disable) | the main accuracy lever; disable only to isolate a bug or compare raw VLM output |
| `--recount` | **off** (opt-in) | a documented row-crop-alignment reliability gap can make a misaligned recount look *more* confident than doing nothing; hybrid-quantities is the recommended way to independently verify a row instead |
| `--preprocess-image` | off (opt-in) | deskew+CLAHE; fixed a real row-bleed bug on one hard form but measurably damaged two others in A/B testing — only turn on for a specific image known to have that problem |
| `--no-brandlist-check` | off (check runs) | skip if DB access is unavailable/slow |
| `--think` / `--think-effort` | off / unset | only relevant for non-mistral models (see HISTORY.md); mistral doesn't need either |

### Output per image (written to `--outdir`)

- `<name>.json` — final `OrderForm`-shaped output, the DB-upload target
- `<name>.raw.json` — full parsed model output before conversion (debugging)
- `<name>.stageA.json` — `{seller_name, size_headers}`, used by `generate_review.py`
- `<name>.recount_flags.json` — per-row agree-or-flag status (hybrid + recount + brandlist flags merged)
- `<name>.brandlist.json` — per-item catalog match/suggestion notes (skipped if `--no-brandlist-check`)
- `<name>.party_check.json` — buyer-name cross-check against the DB (only written when a candidate match is found)
- `<name>.hybrid_debug.json` — every OCR-detected candidate + final per-row assignment (only when hybrid runs)
- `review.csv` — flattened, one row per (item × size), across every image in the run

### Known limitations (current — see HISTORY.md for how each was investigated)

- **Column-position drift is mostly, not completely, corrected.** Hybrid
  quantities fixes the large majority of cases on this project's test
  forms; a residual 1-2 position shift or a single missing/extra cell
  can still occur, especially on very dense or unusually-laid-out
  tables. Always spot-check via the review page before DB upload.
- **`struck_out`/`date_present`/`letter_sizes` are mistral-only schema
  fields.** Running a different `--model` loses these three specific
  corrections (a cancelled row's quantities won't be zeroed, etc.) —
  there's a model-agnostic fallback for struck-out rows via the row's
  own printed total, but it only fires when a total is present and
  parses.
- **Non-ESSA form templates aren't all handled.** This pipeline is
  tuned against the business's own ESSA-style printed grid forms. Other
  surveyed templates (`*` as a quantity marker, dual stacked headers
  where both rows are real, brace-grouped cells spanning several
  columns, free-form notebook pages with no shared header at all) range
  from "handled" (free-form pages — the VLM's own reading is used
  directly) to "not yet built for" (`*` markers, brace grouping) — see
  HISTORY.md's "Multi-form-type survey" section for the inventory.
- **Recount is off by default because of a real, documented failure
  mode**, not just unproven — see the CLI defaults table above.
- **No automatic retry-with-different-model.** If mistral's main call
  fails validation twice in a row (`MAIN_CALL_MAX_RETRIES = 1`), that
  image is skipped with an error printed to stderr, not silently
  retried with a fallback model.

## Other pipelines kept in this repo

Both of these remain because **`extract_ollama_cloud.py` directly
imports code from both** — they are load-bearing dependencies, not dead
weight, even though neither is the production entry point:

- **[extract_claude.py](extract_claude.py)** — a Claude API pipeline
  (one schema-constrained call via Anthropic's structured outputs). More
  accurate on some hard cases (letter sizes, ditto-mark composition, an
  unprinted overflow column) on the first try, with no
  hybrid-OCR-style correction machinery needed — but priced per-token,
  which ruled it out for routine production volume (see HISTORY.md's
  cost figures). `extract_ollama_cloud.py` imports its Pydantic schemas,
  prompt text, and crop/validate/merge helpers verbatim rather than
  duplicating them. Still useful directly (`--live` flag) as a
  higher-accuracy option for a form the production pipeline flags as
  low-confidence, or for spot-checking a new form template.
- **[extract_ollama.py](extract_ollama.py)** — the original local
  Ollama (`qwen2.5vl:7b`) + PaddleOCR pipeline ("v6.2" in HISTORY.md).
  Superseded because a local 7B model's spatial grounding wasn't
  reliable enough even with per-row crops — but `extract_ollama_cloud.py`
  imports its row/header crop-building helpers (`build_header_crop`,
  `build_row_crop`) directly, and its own imports
  ([grid.py](grid.py), [ocr_cell_read.py](ocr_cell_read.py),
  [schema.py](schema.py), [prompt.py](prompt.py)) must all still import
  cleanly for that to work.

[brandlist_match.py](brandlist_match.py) and
[generate_review.py](generate_review.py) are shared by both cloud
pipelines and documented above; [grid.py](grid.py) (CV row-boundary
detection) and [ocr_cell_read.py](ocr_cell_read.py) (PaddleOCR cell
reading) back both the local pipeline and `--hybrid-quantities`;
[preprocess_for_vlm.py](preprocess_for_vlm.py) is the opt-in
deskew/contrast step.

## Test fixtures

[Images/](Images/) holds every sample form used throughout this
project's development, including a `-scanned` variant of several
(flatbed-scanned rather than phone-photographed) used for the current
regression set. `sample 5-scanned.jpg`, `sample 12-scanned.jpg`, and
`sample 13-scanned.jpg` are the three forms every change to
`extract_ollama_cloud.py` should be regression-checked against before
being trusted (see HISTORY.md for the specific known-good cell counts/
checksums each one should reproduce) — `sample 2.jpeg` (free-form, no
printed grid) is the fourth, checked for structural safety (the hybrid
pass must no-op on it, not error).

## Not yet built

- `generate_review.py` isn't wired into `extract_ollama_cloud.py`'s main
  loop — the review page is a manual second step per image.
- No automated regression test suite — "regression-checked" throughout
  HISTORY.md means a real CLI run compared by hand against known-good
  cell counts/checksums, not an automated test that fails CI.
- SKU/item-name normalization and quantity/unit normalization beyond the
  brandlist cross-check.
