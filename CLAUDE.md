# OrderFormV2 — Order Form Extraction Pipeline

## Goal

Automate data entry from photos of garment order forms (mixed
handwritten/printed/typed, dense size-quantity tables) into clean
structured JSON, reviewed by a human against the photo, and written
straight into the order database (`OrderMaster` / `OrderDetails`).

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
2. **Hybrid quantity correction** (`--hybrid-quantities`, **on by
   default**): an independent, second re-read of every row's quantities,
   reconciled against mistral's own main-call reading rather than
   unconditionally overriding it (see `_reconcile_hybrid_with_vlm`) —
   exact agreement ships silently; a value-only conflict on the same
   column set is arbitrated by the row's own printed total; anything
   else is flagged for human review rather than guessed. Only runs when
   the image has a real shared header row (`size_headers` from the main
   call) — a free-form page with no printed grid skips this stage
   entirely and the VLM's own reading stands.

   **As of 2026-09-05, [hybrid_quantities_lighton.py](hybrid_quantities_lighton.py)
   (a second VLM call, `maternion/LightOnOCR-2:1b` local Ollama, reading
   its own assembled table back out) is the default backend** —
   `--no-lighton-hybrid` reverts to the original PaddleOCR + `grid.py`
   pixel-position/DP backend (`_hybrid_ocr_quantities` in
   [extract_ollama_cloud.py](extract_ollama_cloud.py), still fully
   maintained, not a fallback of last resort). Both backends produce the
   same `{size_header: quantity}` shape and go through the same
   reconciliation; see [compare_hybrid_backends.py](compare_hybrid_backends.py)
   for the evidence the switch was based on and
   [hybrid_quantities_lighton.py](hybrid_quantities_lighton.py)'s own
   module docstring for the full list of quirks it does and doesn't
   handle (SIZE LABEL OVERRIDE / handwritten overflow columns, struck-out
   rows via mistral's direct flag, two-printed-line items, stacked
   dual-numbering headers). One quirk it does NOT close: a row using
   clothing **letter sizes** (S/M/L/XL/XXL stacked over a digit, e.g.
   `sample 3-scanned.jpg`'s "MM K4532") — LightOnOCR-2 cannot be prompted
   into preserving the label instead of guessing a plain digit under the
   nearest numeric column (confirmed: an explicit prompt instruction for
   exactly this made no difference on a real re-run), and mistral itself
   practically never sets `letter_sizes=True` for this specific cramped
   shape either (confirmed via 5 fresh live calls, all `False` — a
   separate, independently-documented gap, see MISTRAL_PROMPT_ADDENDUM_BASE's
   own comment in extract_ollama_cloud.py). A fix exists
   (reuse PaddleOCR's own already-tuned `_recover_letter_size_row_digits`
   pixel-level recovery, gated on a code-derived "this row's total
   doesn't add up" signal instead of mistral's unreliable flag) but was
   reverted the same day it was built: that signal isn't actually narrow
   — a large sum mismatch correlates with ordinary column-shift misreads
   just as often as with a real letter-size row, so it fired PaddleOCR's
   full ~20-40s whole-page pass on unrelated rows of other forms too,
   undoing the latency benefit of switching backends. The idea (kept
   alive, not dropped — a cheaper, genuinely targeted letter-detection-
   only pre-check before ever committing to the expensive path) is
   pending further testing. On the hardest test form
   (`sample 4-scanned.jpg`, see "Detection-coverage guard" below),
   LightOnOCR-2's own reading currently agrees with the PaddleOCR backend
   on very few rows — this form has NOT been shown to be handled better
   under the new default, only differently; the PaddleOCR backend's
   coverage guard (below) has no LightOnOCR-2 equivalent yet.
3. **Brandlist DB cross-check** (`--no-brandlist-check` to skip): a free,
   local lookup against the real SQL Server product catalog
   (`brandlist_match.py`). Surfaces fuzzy item-name/style suggestions,
   auto-fills a blank style code when the matched product has exactly
   one, resolves letter sizes to real catalog numbers once a product
   match is trustworthy, flags (never auto-corrects) a likely
   column-shift or an out-of-catalog-range size, and cross-checks
   `party_name` against the real buyer table for non-ESSA form
   templates whose letterhead/handwritten-name roles are reversed from
   the usual convention. This cross-check is also what catches a
   **fabricated `party_name`** — confirmed live on `sample 12-scanned.jpg`
   (a non-ESSA "To"/"From" order pad, no labeled "Party Name" box at
   all): mistral invented a plausible-sounding but entirely fictitious
   buyer name ("Fruitshop"/"Fruitwala"/"Fruitful" across repeated calls —
   nothing on the page is fruit-related) rather than reading the real
   handwritten name in the "From" field, because the SELLER vs BUYER
   prompt rule only described where to look on ESSA's own "Party Name"
   box layout. **Fixed 2026-09-05**: `extract_claude.py`'s shared prompt
   now explicitly covers the To/From layout (read the "From" field) and
   forbids inventing a name when none is legible. The party-name
   cross-check above is still the safety net for whatever this prompt
   fix doesn't fully resolve on other templates — treat a flagged
   mismatch as "verify against the photo," not as proof of a fabrication
   specifically.
4. **Human review and DB upload** — see "Review app" below. The
   browser app in [app/](app/) is the production review path: it runs
   stages 1-3 per uploaded photo, shows the result beside the photo for
   correction, and writes the finished order to `OrderMaster` /
   `OrderDetails`. [generate_review.py](generate_review.py) remains as a
   standalone, no-server alternative that exports corrected JSON instead
   of uploading — useful for checking one form offline, but it is not
   the path orders go through.

### Setup

```powershell
py -3.11 -m venv .venv          # paddlepaddle has no 3.13+ wheels yet
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.env` (not committed — see `.gitignore`):

```
OLLAMA_API_KEY=...      # required — ollama.com/settings/keys
SERVER=...               # required for the review app; optional for the CLI,
DB=...                    # where it only enables the brandlist cross-check.
USER=...                   # (all four required together, or all omitted)
PASSWORD=...
ANTHROPIC_API_KEY=...     # only needed to run extract_claude.py directly
```

If the DB vars are absent, `extract_ollama_cloud.py` prints a warning and
runs without the catalog cross-check (item names/sizes go unverified,
but the pipeline still produces output) — it does not fail outright. The
review app degrades rather than failing too: it still extracts and lets a
reviewer edit, but shows a "no database" banner and can neither bind
products nor upload, since both need the catalog.

### Usage

```powershell
# the review app — upload, correct, upload to OrderMaster/OrderDetails:
.venv\Scripts\python.exe run_ui.py

# the pipeline on its own (batch extraction, debugging a form, no DB upload):
.venv\Scripts\python.exe extract_ollama_cloud.py "Images\sample 5.jpeg"
.venv\Scripts\python.exe extract_ollama_cloud.py Images\ --outdir extracted_ollama_cloud

# alternative model (local, free, slower -- see "Alternative model: Chandra" below):
.venv\Scripts\python.exe extract_ollama_cloud.py "Images\sample 5.jpeg" --model chandra

# standalone HTML review page for one already-extracted image (not auto-generated):
.venv\Scripts\python.exe generate_review.py "sample 5" --outdir extracted_ollama_cloud --images Images
```

### CLI defaults (as of the 2026-09-03 production cutover)

| flag | default | notes |
|---|---|---|
| `--model` | `mistral-large-3:675b` | see HISTORY.md for the model comparison this was chosen from; `chandra` is a shorthand for the local, free `datalab-to/chandra-ocr-2` alternative -- see "Alternative model: Chandra" below |
| `--hybrid-quantities` | **on** (`--no-hybrid-quantities` to disable) | the main accuracy lever; disable only to isolate a bug or compare raw VLM output |
| `--no-lighton-hybrid` | **LightOnOCR-2 is the default backend as of 2026-09-05** (pass `--no-lighton-hybrid` for the PaddleOCR+grid.py backend) | see "Hybrid quantity correction" above for what this changes and its one open gap (letter-size rows). Needs `maternion/LightOnOCR-2:1b` pulled in local Ollama unless this flag is passed. |
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
- `<name>.template_learning.json` — per-template signature match + any corrections auto-applied from a prior order on this same template (see "Per-template correction memory" below); written on every run, `matched: false` on a template's first occurrence
- `review.csv` — flattened, one row per (item × size), across every image in the run

### Detection-coverage guard (2026-09-03)

The hybrid pass assumes OCR actually *saw* the page's handwriting, so a
disagreement with the VLM means the VLM drifted. On a page where
PaddleOCR can't detect the marks at all, that assumption inverts:
`_reconcile_hybrid_with_vlm` keeps the OCR reading whenever the two
disagree about which columns are filled, so near-total blindness gets
treated as near-total authority.

Found on `sample 4-scanned.jpg`, a form whose cells are almost all a
single `1` — a lone vertical stroke, the hardest possible glyph for a
text *detector* (as opposed to recognizer) to find. OCR located 13 marks
on a page with ~60 filled cells, and two of those 13 were adjacent 1s
merged into `111`. Eight of twelve rows were overwritten with
one-or-two-cell readings. Scored against the form's own printed row
totals, on fresh CLI runs:

| | rows matching the printed total |
|---|---|
| hybrid on, before the guard | 2/11 |
| `--no-hybrid-quantities` | 7/11 – 10/11 (varies by run) |
| hybrid on, with the guard | **10/11** (twice, stable) |

Detection coverage separates the two situations with a wide margin, so
that's the gate (`MIN_OCR_COVERAGE = 0.60`):

| form | OCR marks | VLM cells | coverage |
|---|---|---|---|
| `sample 5-scanned` | 71 | 66 | 108% |
| `sample 13-scanned` | 138 | 135 | 102% |
| `sample 4-scanned` | 13 | 60 | **22%** |

Healthy pages land at or slightly above 100% (OCR also picks up marks
outside the VLM's own reading). Below the gate the stage stands down for
the **whole page** rather than per row: a page this sparse gives no basis
for deciding which rows to trust, so the VLM's reading stands everywhere
and every non-empty row is flagged `unverified` for a human. Regression-
checked against the other three forms — all unchanged (14/14, 10/10 and
20/20 rows still hybrid-corrected; `sample 2.jpeg` still no-ops).

**This does not make `sample 4`-style forms accurate, it makes them
honest.** Every row comes back flagged as unverified, because nothing
automated checked it. That page needs the review UI, not trust.

### Known limitations (current — see HISTORY.md for how each was investigated)

- **A form of mostly-`1` cells defeats OCR detection entirely.** See the
  coverage guard above (still accurate for the PaddleOCR backend). The
  guard prevents the damage but recovers nothing: those pages get the
  VLM's unverified reading, and the printed row totals (shown in the
  review UI beside each computed total) are the only cross-check
  available. **2026-09-05:** LightOnOCR-2 (evaluated as a PaddleOCR
  replacement starting 2026-09-04) became the default hybrid-quantities
  backend — see "Hybrid quantity correction" above. It does NOT have an
  equivalent coverage guard, and on this exact form
  (`sample 4-scanned.jpg`) its own reading currently agrees with the
  PaddleOCR backend's on very few rows out of 12 — this form has not
  been shown to be handled better under the new default, only
  differently and not yet with confidence either way.
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
  parses. `struck_out` itself isn't perfectly reliable even on mistral —
  confirmed on a real run of `sample 13-scanned.jpg`'s "B 4457 COLLAR"
  row: mistral reported `struck_out=True` on a row later confirmed, by
  zooming into the photo, to have no strike-through at all. The
  LightOnOCR-2 hybrid backend (see above) treats this as a genuine
  conflict rather than trusting the flag blindly when its own reading
  still finds a substantial quantity set — flags it for a human instead
  of silently deciding either way.
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

## Review app: `run_ui.py` + [app/](app/)

**This is how orders actually get entered.** A local FastAPI server
([app/server.py](app/server.py)) serving a single-page browser UI
([app/static/](app/static/)). It takes the photos, runs them through
`extract_ollama_cloud.extract_one` (the same function the CLI calls — no
stage is reimplemented, so the UI can't drift from the pipeline
HISTORY.md's accuracy work was done against), shows the result beside the
photo for correction, and writes the finished order to the database. An
"Extraction mode" dropdown on the upload screen ("Cloud" -- the
production default, mistral -- or "Local" -- `chandra`, see "Alternative
model: Chandra" above; the model names themselves are deliberately not
shown in the UI, only in this doc) is stored per session (`POST
/api/sessions`'s `model` field) and fixed for every page in that order —
mixing models within one order isn't supported, same as this app's
existing one-buyer-per-order rule.

```powershell
.venv\Scripts\python.exe run_ui.py       # binds 0.0.0.0; prints a LAN URL too
```

Binding `0.0.0.0` is deliberate: an operator opens the LAN URL on a phone,
photographs a form, uploads it from the camera, then reviews it on a
desktop where the grid and the photo fit side by side. Several photos can
belong to one order — they're reviewed together and become one
`OrderMaster` row.

### What it writes

| table | column | source |
|---|---|---|
| `OrderMaster` | `OrderNo` | `MAX(OrderNo) + 1`, taken under `(UPDLOCK, HOLDLOCK)` |
| | `HeadID` | `BUYER.BuyerID` from the buyer picker |
| | `RefNo` / `RefDt` | the form's own order no. and date (editable) |
| | `OrderDt` | entry date, defaults to today |
| | `CompanyID` / `UserID` | the topbar company/operator pickers |
| | `IsClose`, `OrderSource`, `OnlineOrderId`, `WebUserId` | `0`, `'Local'`, `0`, `0` |
| `OrderDetails` | `DetID` | `brandlist.detid` for one (product, style, size) |
| | `Qty`, `RSno`, `Sno` | quantity, item-row serial, running serial |

`OrderNo` is **not** an identity column in this schema (checked against
`INFORMATION_SCHEMA`, and there are no stored procedures that insert
here) — the existing application allocates it itself, so this does too,
serialized by the range lock rather than racing.

### The binding model (the part that matters)

`OrderDetails.DetID` identifies a **(product, style, size)** triple, not a
product. So every reviewed row must be bound to one catalog product+style
group (`bsid` = `bid` + `bstyle`) before it can be uploaded, and only for
sizes that group actually has. A size the reviewer added that the bound
product has no `detid` for **cannot be inserted at all** —
`db.resolve_lines()` reports it as a blocking problem rather than dropping
it, because a silently-dropped size is an order shipped short.

Two routes auto-bind a row; everything else is left for the picker, which
is pre-ranked so the right product is usually the first option:

1. **brandlist_match.py's own trust rules** (score / product-code
   narrowing) say the name match is trustworthy and it resolves to
   exactly one style group.
2. **The row's own style code and ordered sizes single out exactly one
   candidate**, among candidates whose *name* scores within 8 points of
   the best. This is the disambiguator brandlist_match.py's module
   docstring identifies as the real one — "Bloomer Plain" is ambiguous
   between the ladies' and kids' lines, and only the size range ordered
   tells them apart. The name band is what keeps it honest: without it,
   size compatibility alone bound "Bloomers Print" to a *Plain* product
   (confirmed on a real sample 5 run). The pool for this test is 40 deep,
   not the 8 shown, because "exactly one possibility" only means
   something over a pool wide enough to contain the alternatives — with a
   pool of 8, the catalog's four LADIES PRINT variants were all outside
   it.

A binding whose product can't cover the sizes the row ordered is **kept,
but not marked "auto"** — it shows a "check sizes" tag, and the offending
cells turn red naming the exact size. Dropping such a binding was tried
and is worse: it replaces a message that points at one bad cell (usually
a column shift, repairable in one click) with a bare "not linked to a
product" the reviewer has to diagnose from scratch. What gets dropped is
the *confidence*, not the information. Real examples this catches:
"BLOOMER PRINT" → BLOOMER KIDS PRINT (catalog 40-75) for a row ordering
80/85/90; eight rows of `sample 13-scanned` each overflowing their
product's range by exactly one size, which is the fingerprint of a
whole-page column shift.

Measured across the current test forms: 8/14 rows auto-bind on
`sample 5-scanned`, 14/20 on `sample 13-scanned`, 11/12 on `sample 2`
(the free-form page). The rows that don't bind are genuinely ambiguous
(white vs. colour vs. gym vest; plain vs. white trunks) with the correct
product ranked first or second in the picker.

### Search ranking

[app/db.py](app/db.py) caches the catalog (~6.2k product+style groups) and
the buyer list (~9.4k) in memory and fuzzy-matches in process, which is
what makes search-as-you-type both instant (~40ms) and *fuzzy* — a SQL
`LIKE` can't find the right product from a half-remembered name. Two
measured findings shape it:

- **`token_set_ratio`, not `WRatio`.** Tested against the item names this
  project's history records as hard: WRatio saturates at ~86 for almost
  everything on this catalog (its `partial_ratio` component rewards any
  short shared fragment), so "Image FCD" ranked four BABYCARE products
  above every FCD product and "F.G-3025" ranked five unrelated "2.0 G ..."
  products above "F.G - 3025 TRUNKS". Same generic-word-overlap ceiling
  brandlist_match.py documents hitting. `token_set_ratio` put the right
  product first in every one of those cases.
- **Product codes get their own narrowing pass first.** A 3+ digit code
  ("4532") is the most distinctive token on this catalog and no text
  scorer ranks it correctly on its own — confirmed on "MM K4532", whose
  one true match was crowded out of a 60-candidate shortlist by generic
  word overlap. Same finding `brandlist_match.find_best_match` documents.

**`BUYER.IsActive` is inverted from its name in this database** — `1`
means *closed*. Confirmed against live data (2026-09-03): 1996 of the
last 2000 orders went to `IsActive = 0` buyers (882 distinct), 4 to
`IsActive = 1` buyers, and the `IsActive = 1` records with any order
history are retired accounts, several with "(CLOSE)" in the name itself.
Both `app/db.py` and `brandlist_match._buyers()` now filter
`IsActive = 0`; **brandlist_match.py had the inverted filter until this
was found**, so its party-name cross-check had been matching against a
pool that excluded nearly every buyer in daily use.

### Repairs the UI offers

Each maps to a documented pipeline failure mode, so the fix is one click
rather than re-typing a row:

- **← / →** move one row's quantities one size column, within the page's
  own printed header order (not by numeric size — a photographed page's
  columns aren't evenly spaced). (A whole-page version of this, plus a
  whole-page vertical shift, existed until 2026-09-07 — removed at the
  user's request; the horizontal one had been the fix for the systematic
  single-column overflow seen on `sample 13-scanned`, so that specific
  case now needs the per-row action repeated down the page instead of
  one click.)
- **Shift quantities ↑ / ↓ from this row** cascades every row below, for a
  block attributed one row too high or low. Item names and product
  bindings stay put — it's the quantities that drifted.
- **Add this product's sizes as columns** — for the case where a product's
  real sizes have no relation to the form's printed grid at all (the kids'
  line sized 35-55 on a form printed 45-105, the `MM K 4532` case above).
  The pipeline has nowhere to put those numbers, so the columns have to
  exist before the quantities can be typed.
- Add/remove any size column, add/insert/delete/clear any row, edit any
  item name or quantity, re-link any product, remove or retry a page.
- **One line per row.** The item cell used to be two lines (name input
  plus the bound catalog product) with note paragraphs under them, so a
  row was ~58px where a quantity cell needs ~30px — on a 20-row form that
  is more than a screenful of scrolling spent on text nobody reads during
  the quantity-checking pass. Now:
  - Notes are **single-glyph markers** (`⚠` flag, `ℹ` catalog note, `⌫`
    struck-out) with the full text in a tooltip.
  - The bound product is always shown on a **second line** under the item
    name. (An earlier version made this opt-in via a "Show products"
    toolbar toggle — removed 2026-09-07 at the user's request, since
    knowing what a row is actually bound to shouldn't require a click
    a reviewer has to remember to make.)
  - What stays on the one line is only what demands action: a red
    **link** chip on an unbound row, an amber **sizes** chip when the
    bound product doesn't come in a size the row ordered. Both open the
    picker. A correctly-bound row shows nothing — absence is the "fine"
    signal, which keeps the grid quiet and the exceptions loud.
  - The row's coloured left edge and its highlighted cells are unchanged,
    so nothing that needed attention stopped looking like it.
- The form's own printed row total is shown beside the computed total and
  highlighted when they disagree — the strongest independent check a
  reviewer has. It's read from `<name>.raw.json`, the only artifact that
  still carries `row_total`; the final `OrderForm` shape has no field for
  it.
- Frozen size-header row and frozen item-name column, so neither scrolls
  out of view on a 20-row × 20-size table. This is why the table is its
  own scroll region: `position: sticky` sticks to the nearest scrolling
  ancestor, so wrapping the grid in an `overflow-x` container (or putting
  `overflow: hidden` on the card) silently disables it.
- A **second, mirrored horizontal scrollbar pinned above the grid**
  (`buildTableRegion`). Consequence of the point above: the grid's own
  horizontal scrollbar is at the bottom of a viewport-tall scroll region,
  so reaching the far-right size columns would otherwise mean scrolling
  to the last row first. The mirror stays under the card header wherever
  you are in the rows, and hides itself when the table already fits.
- **Columns sized so the whole grid usually fits without scrolling at
  all.** On a 20-size form every pixel of size-column width costs 20
  pixels of scrolling. The grid is `table-layout: fixed` with every width
  declared once in a `<colgroup>` (`col.c-idx` / `c-item` / `c-size` /
  `c-total` / `c-printed` / `c-actions`); a 20-size form comes to
  24 + 200 + 20×34 + 42 + 42 + 74 = **1062px**, down from ~1590px of
  declared width and considerably more in practice.

  Three things had to change together, and the first two are why setting
  `min-width` alone did nothing:

  1. **`table-layout: fixed`.** Under the default auto layout every
     cell's max-content sets its column, and a `width: 100%` on an
     `<input>` counts as `auto` for that calculation — so each quantity
     cell was contributing the browser's default input width (~170px) and
     the `min-width` on the header was never the binding constraint.
  2. **`min-width: 0` on `.bound-name`.** A flex item defaults to
     `min-width: auto`, i.e. it won't shrink below min-content, and for
     `white-space: nowrap` text that's the entire string — so the product
     label was forcing the item column as wide as the longest catalog
     name and the `text-overflow: ellipsis` never fired.
  3. The remove-column `✕` is absolutely positioned rather than
     `visibility: hidden`, which still reserves layout space — about 14px
     in *every* size column.

  Note the trade: under fixed layout an over-long value ellipsizes or
  wraps rather than widening its column. That is the intended behaviour
  here (the full text is in a `title` tooltip, and the picker shows full
  names), but it is a real change from "nothing is ever truncated".
- **The photo pane collapses** (`H`, or the chevron in its toolbar),
  giving the grid the full window once a page has been eyeballed and only
  quantities are being typed. A narrow rail stays behind to bring it
  back, and the choice is remembered.
- **A progress bar under the topbar** whenever an upload is in flight or
  any page is still being read, plus a live elapsed-seconds counter on
  the page being read. Without them a reviewer who has scrolled away from
  the page in progress sees a completely static screen for the ~1 minute
  extraction takes.

### Safety properties

- Nothing reaches SQL Server before the "Upload to database" click. That
  click validates server-side first — `POST /api/sessions/{sid}/validate`
  runs the exact same `resolve_lines()` the upload runs, so "Check" can
  never promise something the upload then rejects — then inserts master
  and details in one transaction.
- `(OrderNo, DetID)` is `OrderDetails`' primary key, so the same SKU twice
  in one order is not insertable — confirmed by direct test against the
  live table. `resolve_lines()` sums those and reports the merge rather
  than failing the upload, which makes that merge load-bearing, not a
  convenience.
- Pages naming different buyers raise a banner rather than being silently
  resolved: every page in a session becomes one order, so that almost
  always means two unrelated forms were uploaded together.
- Edits are local until upload, kept in `localStorage`, and a page whose
  rows the reviewer has taken over is never overwritten by a later poll —
  so extraction of page 3 finishing can't revert edits to page 1.
- Every upload writes `ui_sessions/<sid>/submitted_<orderno>.json`,
  pairing what the model produced with exactly what the human corrected it
  to. That's the audit trail, and it's also the raw material for
  prioritizing the next round of accuracy work against real production
  failures rather than this repo's fixed sample set.

### Files

- [app/server.py](app/server.py) — routes only, no logic
- [app/pipeline.py](app/pipeline.py) — sessions, background extraction,
  payload build, product/buyer pre-binding
- [app/db.py](app/db.py) — catalog/buyer/operator lookups, `resolve_lines`,
  the transactional insert
- [app/static/](app/static/) — `index.html`, `app.js`, `styles.css`; no
  build step, no framework, no CDN
- `ui_sessions/<sid>/` — per-session working data: uploaded photos, the
  pipeline's own output files for them, `review_payload.json`, and the
  `submitted_<orderno>.json` audit record. Gitignored.

### Known gaps

- **Sessions live in memory.** Restarting the server loses in-progress
  reviews (the browser keeps its edits in `localStorage`, but the session
  is gone and it falls back to the upload screen). Already-uploaded orders
  are unaffected — they're in the database.
- **No automated test suite for the app either.** What exists is a jsdom
  smoke test and a rollback-only insert dry-run, both run by hand during
  development, not in CI.
- **One extraction at a time, process-wide.** Two reviewers uploading
  simultaneously queue behind each other rather than running in parallel.
  A stuck extraction (see below) used to mean queueing FOREVER, not just
  waiting a turn.
- **Every Ollama call (local or Cloud) has a 300s request timeout**
  (`OLLAMA_REQUEST_TIMEOUT`, extract_ollama_cloud.py). Confirmed
  necessary 2026-09-08: the `ollama` package's own default client has NO
  timeout at all, so a genuinely stuck call (a wedged local Ollama
  service, GPU contention between this pipeline's own local models) used
  to block the review app's single-worker executor forever — no
  exception, so a page's status stayed "running" with nothing for the UI
  to show, and every later page/session queued behind it indefinitely.
  Now such a call fails loudly after 5 minutes into the same error
  handling every other failure already goes through — plenty of headroom
  above every real call time measured so far (worst case ~130s).

## Per-template correction memory ([template_learning.py](template_learning.py))

**Built 2026-09-09** — was CLAUDE.md's own "Not yet built" idea since
2026-09-08; this section replaces that entry the same way Chandra's own
entry replaced its "not yet built" note once it shipped. The review app
already writes every human correction to
`ui_sessions/<sid>/submitted_<orderno>.json` (see "Review app" above); this
module is what reads that data back INTO a future extraction, so a
correction a reviewer already made once for a given order-form template
doesn't need to be made again the next time that same template is
uploaded.

Two decisions shape the design, both made deliberately rather than
assumed:

1. **Matching happens strictly AFTER the main call**, using its own real
   `seller_name`/`size_headers` output — never a pre-call image fingerprint
   or a second cheap model call, both of which were considered and
   declined. This rules out few-shot prompt injection into that same
   image's own main call (it already ran by the time a match is known) and
   rules out choosing WHICH MODEL handles the main call by template — both
   would need to know the template before the call that would use that
   knowledge. What's still possible, entirely after the main call but
   inside the same `extract_one()` invocation: which hybrid-quantities
   *backend* to use (decided by code that already runs after the main
   call), and deterministic corrections to the main call's own item/
   quantity reading.
2. **A match auto-applies — it does not just surface a hint.** The whole
   point is that a reviewer shouldn't have to retype the same fix twice.
   Every application is still visibly flagged and one click to undo, via a
   new flag status, `template_corrected`, wired through exactly like the
   existing `auto_corrected` status (`_FLAG_SEVERITY`/`_merge_flag` in
   extract_ollama_cloud.py, `_friendly_recount_summary()`/CSS in
   generate_review.py, `friendlyFlag()` in app/static/app.js) — never a
   silent rewrite.

**What's learned, and what deliberately isn't.** The same pre-printed
order form is reused across many different orders — only the handwritten
party name and quantities differ order to order. So item-name misreads
(the printed word doesn't change) are stored and reapplied as a direct
text substitution, and column-shift corrections are stored as a
header-relative POSITION only, replaying `app/static/app.js`'s
`shiftRowHorizontally()` logic exactly — never as raw quantity values,
which are handwritten and genuinely differ every order. Actual quantities
and the buyer/party name are never stored as reusable corrections at all.
Trigger matching is fuzzy at lookup time (`rapidfuzz`, threshold 92 — well
above `brandlist_match.SUGGEST_SCORE_THRESHOLD`'s 70) since a VLM
re-misreading the same printed word twice isn't guaranteed to sample an
identical string both times, but exact at storage time — two distinct
misreads stay two distinct rules. A single contradicting human correction
replaces a stored rule immediately rather than requiring a fresh streak of
disagreement first, deliberately more responsive than this codebase's
usual evidence-gating (e.g. `MIN_OCR_COVERAGE`, the `AUTO_APPLY_*`
thresholds) — being wrong here just falls back to today's fully-manual
baseline, never worse.

**Storage**: one small JSON file per template signature (normalized
`seller_name` + `size_headers`, sha256-hashed) under `template_memory/`
(gitignored, same reasoning as `ui_sessions/` — real business data about
this company's own order-form templates, not source). Also accumulates
per-backend/per-model correction-rate stats, which `preferred_backend()`
uses to route `--hybrid-quantities`' backend choice (lighton vs
paddleocr) once a template has ≥1 sample under each — an explicit
`--no-lighton-hybrid` (CLI) or an explicit `use_lighton_hybrid` argument
always overrides this routing; it only fills in when the caller expressed
no preference.

**Confirmed via a real run (2026-09-09)**: extracting
`sample 5-scanned.jpg` a first time produced `matched: false`, as
expected. A simulated reviewer correction (an item-name fix on "Trend
Trunk" and a +1 column shift on "Exoda Trunk") was recorded via
`record_corrections()`, then the exact same image was re-extracted through
the real production pipeline a second time: both corrections were
auto-applied (`Trend Trunk` → `Trend Trunk V2`; Exoda Trunk's quantities
shifted 80→85/85→90/etc.), both rows carried `status: template_corrected`
with a specific note, and the review app rendered a page-level "📋 Seen
before: this template has been reviewed once before, 2 corrections
applied automatically from past orders" note (verified in a real headless
browser, not just the JSON). One of the two rows also already carried an
unrelated hybrid-OCR disagreement flag from this same run — its note came
back correctly showing BOTH reasons, joined by " | ", which is what
motivated fixing `_friendly_recount_summary()`/`friendlyFlag()`'s prefix
check from `.startswith()` to a substring check while this was built: a
template correction is always merged in LAST, so it's never the first
segment `_merge_flag()` assembled once another mechanism already flagged
the same row, and `.startswith()` would have silently dropped it from the
tooltip.

**Known limitations**:
- A template's first-ever occurrence gets no benefit — nothing is stored
  yet to match against. Benefit starts on the 2nd occurrence.
- No few-shot prompt injection, and no automatic choice of which *model*
  handles the main call — both structurally require knowing the template
  before the main call runs, which decision 1 above rules out. Per-model
  correction-rate stats are still collected (`model_stats` in each
  template file) so this is cheap to revisit if pre-call matching is ever
  built.
- Within one multi-page review session, a later page does not benefit
  from an earlier page's corrections in that same session — corrections
  are only captured at submit time, after the whole session has been
  reviewed. By design, not a bug.
- Trigger matching is fuzzy-but-bounded (threshold 92), not exact — worth
  revisiting the constant if real `template_memory/` data ever shows a
  false hit.

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

## Alternative model: Chandra (`--model chandra`)

`datalab-to/chandra-ocr-2` (local, free, via
`hf.co/mradermacher/chandra-ocr-2-GGUF:Q5_K_M`) was evaluated 2026-09-07
with strong, comprehensively-verified accuracy (see HISTORY.md's own
dated section) but, at the time, nothing from that evaluation was wired
into this repo. **It now is** — pass `--model chandra` to
[extract_ollama_cloud.py](extract_ollama_cloud.py) (shorthand for the
full tag above), or pick "Local" from the "Extraction mode" dropdown on
the review app's upload screen (the model name itself isn't shown
there, deliberately — see "Review app" above) — and it runs through the
exact same `extract_one()` pipeline (hybrid quantities, brandlist cross-check,
review app) every other model does, with no stage reimplemented.

**Why it needs its own code path, not just a different `--model` value
on the existing one**: confirmed by direct testing that forcing Chandra
through this project's schema-constrained call (the same `format=`
mechanism mistral goes through) badly degrades its output — it was
trained to emit its own bbox-annotated-HTML format, not an arbitrary
externally-imposed JSON shape (see HISTORY.md). So Chandra gets a plain
unconstrained prompt instead ("Extract this order form as markdown,
preserving the table structure exactly"), and
[chandra_parser.py](chandra_parser.py) converts its native HTML/bbox
output into this project's `ExtractedForm` shape — dynamically counting
leading non-size header columns per form, concatenating a table Chandra
sometimes splits across two `<div>` blocks, expanding ditto marks
(`— " — Plain` → the row above's name with its own modifier word
replaced), and reading a To/From booking-agent template's real buyer out
of the "From" field the same way [extract_claude.py](extract_claude.py)'s
shared prompt already does for the other pipelines. Runs on the
**local** Ollama service (`get_client()` returns an unauthenticated
`ollama.Client()` for any model name containing "chandra"), not Ollama
Cloud — needs `ollama pull hf.co/mradermacher/chandra-ocr-2-GGUF:Q5_K_M`
first, no `OLLAMA_API_KEY` involved.

Real end-to-end runs (CLI and, separately, through the review app's own
HTTP API) against all 5 of this project's regression forms —
`sample 5/3/13/4-scanned.jpg` and the free-form `sample 2.jpeg` —
confirmed clean, crash-free extraction through the full pipeline
(hybrid quantities via both backends, brandlist cross-check, product
auto-binding), including two real Chandra-side quirks the parser had to
be built around, not just the ones the original evaluation already
found: on `sample 13-scanned.jpg`, Chandra's own body rows carry one
more cell in the size-column region than its own header row lists on
every row (`chandra_parser.py` reads leading/trailing cells by position
from each end of the row rather than a fixed offset, so this drops at
most one — empty, on every row checked — middle cell instead of
shifting every column including `row_total`), and a `GRAND TOTAL`
merged-cell footer row (3 `<td>`, only the first one actually merged)
needed a broader footer-row detector than the single-merged-cell case
the first evaluation's regression forms happened to show.

**Traded off against the production default, not a strict upgrade**:
mistral's `struck_out`/`date_present`/`letter_sizes` flags don't exist
for this model at all (Chandra gets no schema, so there's nothing to set
them on) — confirmed acceptable for the first two (Chandra's own
unprompted reading already leaves a genuinely struck-out row's
quantities empty, and never fabricates a date for a pre-printed
day-of-week label), but letter-size rows (`MM K4532`) remain exactly as
unsolved as on every other model in this project. And per the original
evaluation, Chandra is still meaningfully slower than mistral's entire
production pipeline on denser forms, not just its own main call — real
wall-clock times from these regression runs: ~90s (`sample 5-scanned`),
~128s (`sample 13-scanned`, 20 size columns), ~68s (`sample 4-scanned`),
~76s (`sample 12-scanned`), ~38s (`sample 2.jpeg`, free-form). No
retry-across-models exists either — see "Known limitations" above.

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
  loop — that standalone review page is still a manual second step per
  image. It is no longer the review path, though: the browser app in
  [app/](app/) is (see "Review app" above), and it runs the pipeline
  itself rather than reading its output files after the fact.
- No automated regression test suite — "regression-checked" throughout
  HISTORY.md means a real CLI run compared by hand against known-good
  cell counts/checksums, not an automated test that fails CI. The review
  app is in the same position (see its own "Known gaps").
- SKU/item-name normalization and quantity/unit normalization beyond the
  brandlist cross-check.
- **Few-shot prompt injection and automatic main-call model routing**,
  the more ambitious half of what was sketched here as "per-template
  learning from human corrections" before 2026-09-09 — see "Per-template
  correction memory" above for what was actually built instead, and
  exactly why those two specific pieces are out of scope (not just
  deferred) under the match-after-the-main-call constraint that design
  was built to.
