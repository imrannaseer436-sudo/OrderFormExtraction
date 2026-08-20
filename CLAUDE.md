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

## What actually lives in this repo

Only the **Ollama / Qwen2.5-VL + PaddleOCR local pipeline** is implemented here, at
[extract_ollama.py](extract_ollama.py), [schema.py](schema.py), [prompt.py](prompt.py),
[grid.py](grid.py), [ocr_cell_read.py](ocr_cell_read.py). This is "v6" (per-row
crop, read via OCR rather than the VLM) — see [README.md](README.md) for the
full version history (v1 single-call, v2 two-stage) and why each earlier
version was abandoned, and "Pipeline architecture" below for how v3→v6
evolved.

**Runs in a Python 3.11 venv, not the system Python**: this repo's
`.venv/` (`py -3.11 -m venv .venv`) is where `paddlepaddle` actually has
wheels — a newer system Python (3.13+) will fail to install it. Always
use `.venv\Scripts\python.exe`, not a bare `py`/`python` call.

Two other approaches were explored in a separate sandbox environment and are
**not present in this repo**:
- A Claude API pipeline using `tool_choice` for guaranteed schema-valid
  output — logic-tested only, never run against a real API key. Likely the
  most accurate option if local accuracy plateaus; would need to be ported
  in if pursued.
- Qwen2.5-VL via raw HuggingFace/transformers — abandoned outright since the
  actual runtime is Ollama, not raw transformers.

## Pipeline architecture (current, v6)

Run from `extract_one()` in [extract_ollama.py](extract_ollama.py):

1. **Stage A** — one schema-constrained Ollama call (`format=FormMeta` JSON
   schema). Extracts structure only: seller/party info, order no/date,
   size column headers, and the ordered list of item rows (name + style
   code, no quantities). This has been the reliable part in every version
   so far — though see Known issues for two confirmed-wrong style codes.
2. **Row localization** — [grid.py](grid.py): deskew + density-peak row-boundary
   detection (classical CV, no model call). See "v5" history below.
3. **Stage B** — reads each row's quantities. Two paths:
   - **Primary (v6): OCR**, via [ocr_cell_read.py](ocr_cell_read.py) — PaddleOCR
     reads the header+row crop directly; each detected digit is matched to
     its nearest header column via an order-preserving DP assignment (not
     independent nearest-neighbor, so photographed perspective drift can't
     tie-break a digit onto the wrong header alone). No VLM call, no
     hallucination risk (a blank cell just produces no OCR detection).
   - **Fallback: VLM**, `stage_b_row()` — one unconstrained Ollama call
     focused on a single row, used when grid detection fails for a whole
     image, or (rare) when OCR can't even read its own crop's header
     labels. See "v3/v4/v5" history below for why VLM-only reading was
     replaced.
4. **Stage C** — plain Python regex parsing (`PAIR_RE` in
   [extract_ollama.py](extract_ollama.py)) of the VLM fallback's
   `size:value, size:value, ...` response into `{size: qty}`. Only used on
   the VLM fallback path — OCR's assignment happens directly in
   `ocr_cell_read.py`, no separate parse step.

## Design decisions worth preserving

- **No self-reported confidence scores.** The 7B model was badly
  miscalibrated — confidently wrong on quantities. A simpler schema it can
  actually fill in correctly beats a richer one it fills in dishonestly.
- **Domain rules are baked into the prompts**, not handled in post-processing:
  - Seller letterhead vs. handwritten buyer "Party Name" box — don't
    confuse the two for `party_name`/`party_city`.
  - Ditto-mark row-name inheritance (`—"—`, `-"-`): write out the full
    inherited item name, not just the modifier word.
  - Blank vs. zero: empty cells are `"blank"`, not `0`, and Stage C drops
    `blank`/`x`/`-` rather than coercing them to `0`.
  - Dozens as the default unit; fraction quantities (e.g. `"60/10"`) are
    read but currently un-representable as `int` and get silently dropped
    in Stage C (`except ValueError: continue`) — see Known issues.
- **Debug artifacts are saved alongside every output** specifically so a
  wrong answer can be traced to the stage that caused it:
  - `extracted/<name>.stageA.json` — check first if item names/headers are
    wrong.
  - `extracted/<name>.rows.txt` — raw per-row model output, one line per
    item (`item | style -> raw text`); check this if a specific item's
    quantities look wrong in the final JSON.
  - `extracted/review.csv` — flattened one row per (item × size), meant for
    human review before DB upload.

## Setup / usage

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5vl:7b        # ollama serve must be running
.venv\Scripts\python.exe extract_ollama.py "Images\sample 5.jpeg"
.venv\Scripts\python.exe extract_ollama.py Images\ --outdir extracted
.venv\Scripts\python.exe extract_ollama.py Images\ --model qwen2.5vl:32b   # if hardware supports it
```

PaddleOCR's inference backend needed `device="cpu", enable_mkldnn=False`
on this machine — the default (oneDNN-accelerated) path raised
`NotImplementedError` in `paddle`'s text-detection model. That's set in
`ocr_cell_read.get_ocr()`; if paddlepaddle is upgraded and this is
revisited, confirm the default path actually works before removing it.

## Known issues — status after fixing against `sample 5.jpeg`

The three issues below were found by comparing `extracted/sample 5.*`
against the actual source image ([Images/sample 5.jpeg](Images/sample%205.jpeg)), then verified
by re-running the pipeline after each fix (not just inferred from
symptoms).

**Fixed and confirmed (structure/metadata, in [prompt.py](prompt.py) + [schema.py](schema.py)):**

1. **`order_date` leaking model reasoning into the field** — was
   `"30/13/26 (normalize to DD/MM/YYYY: 30/01/2026)"` instead of
   `"30/03/2026"` (the correct value, per the handwritten `30/3/26` on the
   form). Fixed via a stricter Stage A prompt instruction plus a
   `field_validator` on `FormMeta.order_date` that keeps only the first
   date-like token as a safety net. Confirmed correct (`"30/03/2026"`)
   across three re-runs.
2. **Style/type code stuck inside the item name** — e.g.
   `{"item": "Trend Trunk IE", "type": ""}` instead of
   `{"item": "Trend Trunk", "type": "IE"}`. Fixed via an explicit Stage A
   prompt rule plus a whitelist-based `model_validator` on `ItemStub`
   (only splits a trailing token if it's a known style code — `IE`, `OE`,
   `RN`, `RNS` — so item names that legitimately end in caps, e.g.
   `"Image FCD"`, are left alone). Confirmed correct across re-runs.
3. **Ditto-marked rows where the Style column changes** (e.g. row below
   `MYNA`/`IE` is a ditto mark in Particulars but `OE` in Style) — this
   surfaced *while* fixing #2: the model first mistook the Style column's
   new value for the item name (`item: "OE"`), then after a prompt fix
   started leaving `item` blank instead. Fixed with a `FormMeta`-level
   validator that forward-fills a blank item name from the previous row —
   matching the documented ditto-mark semantics. Confirmed: both `MYNA`/`OE`
   and `Classy W. Vest`/`RNS` now resolve correctly.
   - Side note: this also means Stage A's `party_city` field is
     unreliable (returned the seller's printed address city, not the
     buyer's handwritten one, in one run) — harmless today since
     `party_city` never makes it into the final `OrderForm` output, but
     worth knowing if that field is ever surfaced later.
   - A self-inflicted regression also slipped in here: splitting the
     `order_date` bullet out of the Stage A prompt's `Extract:` list
     accidentally deleted the adjacent notes instruction, so `notes` came
     back empty (`[]`) on every run after the date fix. Restored as its
     own bullet in [prompt.py](prompt.py) — **not yet re-verified against a real image**,
     since re-running is manual from here (see below).

Also fixed in this pass, independent of the above: **negative quantities
leaking through Stage C parsing** (e.g. `"-30"`) — the blank-marker check
only matched the literal tokens `blank`/`x`/`-`, so a malformed `"-30"`
token parsed as `int("-30") == -30` and got kept as a real quantity.
`stage_b_row()` in [extract_ollama.py](extract_ollama.py) now drops any parsed value `<= 0`.

**Still open — not fixable by more prompt tweaking, needs a structural
change:**

4. **Row quantities themselves are frequently wrong**, not just
   mislabeled. This is more serious than originally scoped and was found
   by checking Stage B's raw output against the image directly. Confirmed
   failure modes, reproduced across multiple re-runs with different prompt
   wording each time:
   - **Header-row hallucination**: for `Trend Trunk` and `Fairlady Print`
     (real handwritten data: `Trend Trunk` = `80:50, 85:90, 90:68, 95:30,
     100:50`), Stage B instead returns a suspiciously clean arithmetic
     sequence (`...30, 32, 34, 36, 38, 40, 42`) that matches the form's
     *second header row* (the "18/20/22/.../42" chest-equivalent row Stage
     A is explicitly told to ignore for headers). Stage B has no
     equivalent protection, and explicit warnings added to the row prompt
     did not stop it.
   - **Column-shifted but real values**: `Exoda Trunk` and `Image FCD` get
     genuinely-present numbers from the image, but attached to the wrong
     size column (e.g. real values `36, 122, 126, 34, 42` — correctly
     belonging under `80/85/90/95/100` — come back under `50/55/60/65/70`).
   - **Item-name-as-anchor, confirmed twice now**: rows that share an item
     name but differ only in `type` (ditto rows — `MYNA`/`IE` then
     `MYNA`/`OE`; `Classy W. Vest`/`RN` then `Classy W. Vest`/`RNS`) come
     back with **identical** quantities for both rows in the latest run —
     the model re-finds the first occurrence of "MYNA" in the image and
     re-reads that same row both times, ignoring the style difference that
     should point it at the second occurrence instead. This means Stage
     B's accuracy is partly downstream of Stage A's item-list accuracy,
     compounding errors across stages, and is worse whenever a name
     repeats (which ditto rows guarantee).

   **A cropping-based fix was attempted and reverted** (2026-08-01): the
   idea was to stop Stage B from ever seeing the header row or other item
   rows by cropping a tight image band around just the target row before
   sending it. Two variants were built and tested against this same real
   form:
   - *v4a — even-split crop, no header in the crop*: table_top/table_bottom
     fractions from one grounding call, row bands evenly divided between
     them. Row positioning was actually decent (see `Trend Trunk`'s crop —
     it *did* isolate the correct row), but with no header numbers visible
     in the crop, the model lost the `size:value` labeling convention on
     several rows and just dumped raw unlabeled numbers instead — which
     the regex parser can't read. Net result: **filled cells dropped from
     113 to 52** (mostly parse failures, not read failures).
   - *v4b — header band stitched above each row, per-row bounding boxes
     from the grounding call instead of even-split*: asking the model for
     an individual `y_top`/`y_bottom` per item (14 boxes in one call)
     turned out even less reliable than reading the quantities directly —
     boxes came back a few percent of image height tall (far too thin) and
     shifted, so several crops showed the wrong row's content entirely.
     Net result: **filled cells dropped to 24**, several rows showing
     another row's real data under the wrong item.

   Conclusion: this local 7B model's spatial/positional grounding isn't
   reliable enough for this task, whether it's asked to read dense cells
   directly *or* to output bounding boxes — cropping just relocated the
   same unreliable-positioning problem earlier in the pipeline rather than
   fixing it. Reverted back to v3 (full image, no crop) in
   [extract_ollama.py](extract_ollama.py)/[prompt.py](prompt.py)/[schema.py](schema.py); this is the best-performing
   version confirmed so far (113/~150 total size/qty cells filled on
   `sample 5.jpeg`, with the two failure modes above still present on a
   handful of rows).

   Options not yet tried, in rough order of effort: `qwen2.5vl:32b` (if
   hardware allows — likely the next thing to try, per [README.md](README.md)'s original
   prediction), self-consistency voting (call Stage B 2-3x per row, flag
   disagreement for human review instead of guessing), or hybrid routing
   hard rows/forms to the Claude API pipeline (see "What actually lives in
   this repo" above — not present in this repo, would need porting in).

## v5 history — CV-based row detection (superseded by v6 below)

The two v4 attempts above both failed because the *VLM* was asked to guess
row positions and its spatial grounding wasn't reliable. v5's idea: stop
asking the model, measure the row positions instead — the ruled table
lines are real, physical lines on the page. **[grid.py](grid.py)**:

- Deskews the image (small rotation, estimated via Hough transform on the
  binarized page).
- Finds row-boundary y-positions via **relative density-peak detection**
  on the row-wise dark-pixel profile (not a strict "contiguous line across
  X% of the width" morphology test — that approach was tried first and
  missed 5 of 14 rows because the photographed page has mild perspective
  *warp*, not just rotation, which breaks real ruled lines into short
  fragments a strict contiguity test rejects; peak detection tolerates
  fragmentation and still found them).
- Confirmed by drawing detected boundaries back on `sample 5.jpeg` and
  eyeballing them: **all 15 boundaries (14 rows) landed exactly on the
  true ruled lines**. Column (vertical line) detection was also tried
  (density-peaks on the *full page*) and stayed too sparse/unreliable to
  trust at the time — revisited and solved differently in v6 below.

Row detection was solid, but a fresh end-to-end run (2026-08-03) proved
the remaining bottleneck was never the crop — it was the VLM's reading.
Even given a clean, correctly-isolated header+row crop, `qwen2.5vl:7b`
still: (a) hallucinated the form's second (chest-equivalent) header row
as data in blank columns — e.g. `MYNA`/IE came back as
`65:26,70:28,75:30,80:32,...,105:42`, the *exact* second-header-row
sequence, no real values at all; (b) shifted real values into the wrong
column — e.g. `Fairlady Plain`'s real `80:45` came back keyed as `45:45`
(the value reused as the key); (c) on `Classy W. Vest`/RN, inverted
key/value entirely (`6:blank, 15:blank, ...`), which a parser can safely
detect and drop to `{}` but can't safely recover, since the model doesn't
preserve left-to-right column order when it does this. Two rounds of
prompt-wording changes (full-image v3, cropped v5) both hit this same
ceiling — confirmed this is not a prompt problem.

## v6 — read cells via OCR instead of the VLM (current)

**[ocr_cell_read.py](ocr_cell_read.py)**: PaddleOCR reads each header+row
crop directly instead of asking the VLM to. Rationale: a non-generative
recognizer can't hallucinate a blank cell into a plausible sequence — it
just detects no text there. Column assignment: match each detected
header's text to the known `size_headers` list (exact string match, so
the visually-similar second header row never collides — its values are
numerically disjoint from the real headers on this form), then assign
each recognized digit to its nearest header column via an
**order-preserving DP** (`_monotonic_assign` in `ocr_cell_read.py`), not
independent nearest-neighbor per digit — this matters because a
photographed (not scanned) page has mild perspective drift that shifts a
column's true pixel position further from the header's fixed position the
deeper a row sits in the table (measured directly on `sample 5.jpeg`: ~0px
drift at the top row, growing to ~50-58px at the bottom row) — a DP that
looks at the whole row jointly is far less likely to get one ambiguous
digit tie-broken onto the wrong header than per-digit argmin is. A
Total-Dozen-column exclusion (`x_cutoff`) keeps the trailing running-total
column from being misread as a 14th size value. Falls back to the VLM
(`stage_b_row`) for: a whole image where `grid.py` finds no clean row
boundaries at all (unchanged from v5), or an individual row-crop where OCR
can't even read its own header labels (rare).

An interior-gap penalty in the DP (bias toward assigning digits to
*contiguous* headers, since real data on this form's rows is almost
always one contiguous block) was tried and tuned against `sample 5.jpeg`:
it fixed 2 rows but pulled an already-correct anchor away from its true
header on 2 others, chasing contiguity at the cost of individual
distance. Reverted to the plain distance-minimizing DP (`gap_penalty=0`)
as the safer, more predictable default.

**Fresh end-to-end run confirmed row-by-row against the source image
(2026-08-03)**:
- **3 of 14 rows exact**: `Trend Trunk`, `Exoda Trunk`, `Image FCD`.
- **Most others correct except 1 missing value** — a faint handwritten
  digit OCR'd as a letter (`"50"→"so"`, `"10"→"LO"`) and dropped by the
  digit-only filter, not a wrong value.
- **Some rows (deeper in the table) have their last 1-2 values shifted
  one column right** — the residual perspective drift the DP doesn't
  fully correct for the very last columns on rows furthest from the
  header (`Bloomer Print`, `Bloomer Plain`, `Classy W. Vest`/RN & RNS,
  `F.G-3005`, `F.G-3025`).
- **One spurious value**: `Classy W. Vest`/RNS picked up a fake
  `"45": 21` — OCR misread part of the handwritten item-name/ditto text
  in the Particulars column as a digit; the digit filter doesn't yet
  exclude that x-range.
- **Zero hallucinated sequences** — the core v3/v5 failure mode is gone.
- Cross-checking OCR's raw read against Stage A also caught **two
  Stage A item-name errors**, confirmed against the row crops directly:
  `F.G-3005` and `F.G-3025` were misread by the VLM as `F.G1-3005` and
  `F.G1-302S` (spurious inserted "1", and "3025"→"302S"). OCR read both
  correctly. Unrelated to quantities — a Stage A prompt issue, not yet
  fixed.

**Perspective-correction attempts, tried and ruled out (2026-08-03)** —
both aimed at fixing the residual column drift above at the source rather
than tolerating it:
1. Fit the two outer boundary lines of the size-column block (near
   col-45's left edge and col-105's right edge) via Hough transform across
   the whole table height, then use them to locally rescale header x
   positions per row. Result: **under-corrected** — predicted only ~11px
   of drift for the bottom row where the real drift is ~50-58px. Not
   pursued further.
2. A `cv2`-based "document scanner" approach (Canny edges → largest
   4-sided contour → `getPerspectiveTransform`) was proposed and tested
   directly against `sample 5.jpeg`, with and without `cv2.RETR_EXTERNAL`.
   **Failed outright, not just imprecisely**: the true page boundary never
   forms one complete closed Canny contour in this photo (page edge
   contrast against the background isn't clean enough), so the algorithm's
   "largest 4-sided contour" latched onto the printed **"LORRY / BOOKING
   STATION / COURIER" box** on the form instead of the page — a small,
   fully-enclosed, high-contrast rectangle that out-competes the real page
   edge. Produced a 617×187px crop of that empty box, discarding the
   entire order table. This is a known failure mode of that
   Canny+contour "4-point scanner" pattern on real (not studio-lit demo)
   photos, not a tunable-parameter issue — confirmed by testing, not
   inferred.

**Decision (2026-08-03): keep the v6 pipeline as-is.** Both perspective
fixes above were dead ends on this real form; chasing a third is
diminishing returns given the current failure modes (1-column drift,
occasional missed digit, one spurious value) are all small and
individually catchable in `review.csv` — a much safer failure profile
than v3/v5's hallucination. If revisited, the more promising untried paths
are: manual 4-corner picking (a human clicks the page corners once per
photo, reusing the same warp math — trades automation for reliability) or
fitting the page's own black border bars (visible on left/right edges of
these photos) instead of Canny+contours for whole-page rectification.

Not yet addressed (at that point): the `Classy W. Vest`/RN key/value-inversion
case; the Stage A style-code misreads; `Fairlady Plain`/`Bloomer Plain`
returning bare `"Plain"`.

## v6.1 — drift self-calibration + item-name cross-check (2026-08-03, later same day)

Picked up two of the three items above, plus the column-drift issue v6
left unsolved:

**Drift self-calibration, implemented and confirmed working.** Added to
`ocr_cell_read.read_rows()`: every row's *uncorrected* DP fit is computed
first; rows whose average per-digit distance is under
`_CONFIDENT_AVG_COST_PX` (12px) are trusted as genuine, undrifted
measurements, and their (row depth, residual) pairs are fit with a linear
regression, then applied as a per-row header-position correction before
re-running the DP on every row. First attempt used a spread threshold
(`_MIN_CALIBRATION_Y_SPREAD = 0.15`) that was too strict — on
`sample 5.jpeg` only 4 rows ever qualified as confident, and they span
just 0.10 of the table, so calibration silently never activated (**the
fresh run's output was byte-identical to the uncorrected version** — this
is why the spec says confirm against a real run, not just check the code
runs without error). Lowered to `0.08` and **confirmed via fresh run**:
`Bloomer Print`, `Bloomer Plain`, `Classy W. Vest`/RN, `F.G-3005`, and
`F.G-3025` — every row that previously had a value shifted one column
right — are now **exactly correct**. This is the self-calibration approach
outlined as the next untried idea in the v6 section above; it worked
where the two independent-geometry attempts (Hough boundary fit,
Canny+contour) both failed, because it uses evidence from the same
image's own confidently-correct rows instead of re-deriving the page's
geometry from scratch.

**Item-name cross-check for style codes, implemented and confirmed
working.** `ocr_cell_read._ocr_row()` also returns the leftmost
sufficiently-long non-digit text token in each row's data band
(`item_name_ocr` — item name is always the leftmost column on this form,
so no explicit exclusion of the Style column's text is needed).
`extract_ollama.py` overrides Stage A's item name with this OCR reading
when Stage A's name contains a digit and differs from the OCR read —
scoped to digit-containing names specifically so natural-language item
names (which never contain digits on this form) are untouched. Confirmed
fixed: `F.G-3005`/`F.G-3025` (Stage A still misreads these as
`F.G1-3005`/`F.G1-302S` every run; the override catches and corrects it
every time, not just once by chance — verified across two separate runs).

**Ditto+modifier composition (`"Plain"` → `"Fairlady Plain"`), tried and
reverted.** Added a `ditto: bool` field to `ItemStub` (schema.py) so the
model would flag ditto rows explicitly rather than being asked to compose
the full name itself, with composition (`last_item + " " + modifier`)
done in `FormMeta._forward_fill_ditto_item_names` instead of trusting the
model's own attempt at the same rule. **Did not work**: confirmed via
fresh run that the model never set `ditto=true` for these rows — it kept
returning bare `"Plain"` with `ditto=false`, so the flag was simply
unused and composition never triggered. Worse, the prompt rewording this
required caused a new regression, confirmed by direct before/after
comparison: `"Fairlady Print"` (not even a ditto row) came back misspelled
as `"Fairly Print"` — a side effect of shifting the model's attention via
the surrounding prompt edit, not a targeted change. **Reverted** schema.py
and prompt.py to the prior wording; `Fairlady Plain`/`Bloomer Plain` are
still unfixed, same as before this session, but the regression is gone
too. Lesson: this rule needs a fix that doesn't touch the shared Stage A
prompt at all (e.g. an OCR-based signal, since OCR already reads the
Particulars-column text and a true ditto mark contributes little/no
recognizable text there) — not another Stage A prompt/schema iteration,
since two separate attempts at that (this one, and the original ditto
rule in the "Known issues" section above) have now each partially failed.

**Confirmed still open after v6.1**: `Fairlady Plain`/`Bloomer Plain`
bare-`"Plain"` naming (unchanged, see above); 3 rows each missing one digit
where OCR misreads a handwritten number as a letter (`"S"`, `"LO"`,
`"so"`); one spurious value (`Classy W. Vest`/RNS showed a fake
`"45": 21` from OCR misreading a stray mark near the item name as a
digit). Correction: the `Classy W. Vest`/RN key/value-inversion issue from
the v6 section above turned out not to be live in this run's output —
that row went through OCR successfully (no VLM fallback triggered), so
the inversion (a VLM-only failure mode) never applied here. Documented in
v6 as a residual *risk* for whichever row/image ever needs the VLM
fallback, not as an active bug in this output — a mistake to double-check
before listing it as current, made once mid-session (caught when the user
asked for a fresh list and it was cross-checked against the actual file
again).

## v6.2 — recover misread digits + exclude the item-name region (2026-08-03, later same day)

Two more fixes, both confirmed via fresh run, both in `ocr_cell_read._ocr_row`:

- **Digit-lookalike recovery** (`_try_digit_correct`): a small, deliberately
  conservative substitution table (`O`/`o`→0, `S`/`s`→5, `L`/`l`/`I`→1) for
  short (1-3 char) tokens that failed the plain `isdigit()` check but sit
  inside the plausible size-column x-range. Recovers cases where OCR read
  a handwritten digit as a similar-looking letter instead of dropping them
  silently. Confirmed fixed, all previously-missing: `Fairlady Print`
  `95:5` (was OCR'd as `"S"`), `Fairlady Plain` `105:10` (was `"LO"`),
  `Exoda Bloomer Plain` `90:50` (was `"so"`).
- **`x_floor`**: mirrors the existing `x_cutoff` (which excludes the
  trailing Total-Dozen column) on the left side, excluding the
  item-name/Style/ditto-mark region from ever being considered as a digit
  candidate. Confirmed fixed: `Classy W. Vest`/RNS's spurious `"45": 21`
  (a misread ditto-mark/stray-ink in the Particulars column) is gone.

**Result: every quantity across all 14 rows of `sample 5.jpeg` now matches
the source image exactly**, confirmed by a real end-to-end run (not
inferred). The only remaining known issue is `Fairlady Plain`/
`Bloomer Plain` returning bare `"Plain"` (ditto-name composition) —
explicitly left unfixed this round (two prior attempts at this specific
rule have each had problems; see v6.1's revert above).

## Review UI — built 2026-08-03

**[generate_review.py](generate_review.py)**: generates a self-contained HTML
review page per image (`extracted/<name>.review.html`) — the source photo
embedded alongside an editable item x size grid, laid out to match the
physical form (Particulars/Style down the left, size columns across the
top) so a human can visually cross-check output against the photo directly,
not as a flat list. Quantity/item-name/style cells are editable inputs; an
"Export corrected JSON" button reconstructs the `OrderForm` JSON from
whatever is currently in the page (including edits) and downloads it.
Fully self-contained (image embedded as a base64 data URI) — works by
double-clicking the file, no server needed, same as `review.csv` today.

Usage: `python3 generate_review.py "sample 5"` (reads
`extracted/sample 5.json` + `.stageA.json`, finds the source image in
`Images/`, writes `extracted/sample 5.review.html`). Not yet wired into
`extract_ollama.py`'s main loop — currently a separate step.

Verified 2026-08-03 via a headless-Chrome + Playwright script (not just a
static screenshot): edited a quantity cell, an item name, and the party
name, clicked Export, and confirmed the downloaded JSON reflected all
three edits correctly with no console errors.

## Database cross-check — explored 2026-08-03, not yet integrated

The business's SQL Server DB (connection in `.env` — `SERVER`/`DB`/`USER`/
`PASSWORD`, not committed) has a `brandlist` table: one row per (product,
style, size) SKU (`bname`, `bstyle`, `bsize`, `detid`, filtered by
`isprimary=1`), and `ordermaster`/`orderdetails` (the eventual upload
target — `orderdetails.DetID` is the FK into `brandlist.detid`, so each
final line item needs to resolve to one specific SKU row, not just a name).

**Confirmed by direct query, not assumed:** cross-checking extracted item
names against `brandlist` is a real signal but not a clean exact-match
lookup — the handwritten form uses shorthand the catalog doesn't. E.g.
`"Fairlady Plain"` (the ditto-composed name) matched exactly, confirming
that composition; but `"Bloomer Plain"` had no match at all — the real
catalog entry is `"BLOOMER LADIES PLAIN"`, and disambiguating "Ladies" vs
"Kids" (both exist for many product lines, e.g. `BLOOMER KIDS PLAIN` sizes
40-75 vs `BLOOMER LADIES PLAIN` sizes 80-110) requires checking which size
range the row's *actual ordered sizes* fall into — the form's own text
never says "Ladies"/"Kids", only the size range implies it. So a real
matching feature needs: name-similarity search + size-range confirmation,
not name matching alone.

**Not yet built**: the actual lookup/suggestion feature. Given the
matching is inherently fuzzy/ambiguous for many items (not just ditto
rows), any implementation should surface suggested matches for human
confirmation (e.g. as a column in the review UI above) rather than
silently auto-applying a best guess, except where a match is exact and
unambiguous.

## IN PROGRESS — generalizing grid.py to a second form (paused, resume here)

Tested the pipeline against a second real form, `Images/sample 3.jpeg` —
denser (18 items vs 14), lower-resolution (998×1138 vs 1448×1490), messier
handwriting, mostly single-digit quantities. Confirmed by real run: it
broke badly, in two independent ways.

**1. `grid.py`'s row detection found the wrong region entirely.** The
density-peak method picked a "clean" run of 19 evenly-spaced lines in the
letterhead/party-info-box area (candidate boundaries starting at y-fraction
0.05) instead of the real item table (which visually starts around 0.28-0.30,
similar proportionally to `sample 5`'s 0.32). Every row's header crop then
showed the letterhead, OCR failed identically on every row, and the
existing per-row VLM fallback reused the same wrong crop instead of the
real image — reproducing the old pre-OCR hallucination failure modes
(empty rows, row-bleed, arithmetic-sequence fabrication) across the board.
Also separately confirmed: Stage A's `type` field came back as the
secondary header row's sequence (`18,20,22,...50`) for every item on this
form — a second, independent Stage A failure not yet investigated.

**Fixes implemented this session (grid.py, ocr_cell_read.py,
extract_ollama.py), verified partially, not yet end-to-end confirmed:**

- `grid.py`: replaced the single fixed-parameter row-detection strategy
  with `iter_row_boundary_candidates()`, trying several (profile, threshold)
  combinations -- the original total-dark-pixel-count profile, plus a new
  "longest contiguous dark run after tolerantly closing small gaps" profile
  (`_row_profile_longest_run`) that can tell a real ruled line apart from
  generic text-line density when total pixel count alone can't. Confirmed
  by parameter sweep: no single configuration works for both test forms —
  settings that succeed on one fail outright on the other — so the caller
  must try several and validate, not trust one calibrated threshold.
- `ocr_cell_read.count_headers_found()`: OCRs a candidate's header crop and
  counts how many real headers are found -- catches the "wrong region
  entirely" failure (confirmed: 0/13 for the bad candidate, 12-13/13 for
  the real table).
- `ocr_cell_read.row_alignment_ok()`: OCRs a candidate's specific row crop
  and forgiving-word-matches it against the item Stage A expects there --
  catches drift WITHIN an otherwise-correct candidate (confirmed: header
  check alone missed this. On `sample 3`'s only header-passing candidate,
  individual rows further down had drifted -- row 17's crop showed row
  14's content). First version of this check had a bug (confirmed by
  testing): it defaulted to "pass" whenever OCR found no 3+-letter word,
  which let a genuinely wrong row through when the WRONG row's item
  happened to also be a short code with no long word (`"MM K4532"`) --
  fixed to only default-pass when OCR found *no text at all* or the
  *expected* name has no checkable word, not just because the OCR'd text
  didn't have one.
- `extract_ollama.py`: tries every candidate from `iter_row_boundary_candidates`,
  scores each by (header check, then per-row alignment checked for
  **every** row, not a sample -- confirmed a 3-point and even a 5-point
  sample can miss sparse/localized drift, since the bad rows aren't spread
  evenly through the table). Keeps the candidate with the most aligned
  rows (must clear 50% to be used at all), and tracks *which specific
  rows* failed alignment (`bad_row_indices`) so only those individual
  rows fall back to the VLM on the full image, rather than discarding an
  otherwise-good candidate wholesale or keeping known-wrong OCR output
  for those rows.

**Confirmed so far (offline, reusing cached `stageA.json`/images, no
Ollama calls needed for this part):**
- `sample 5` regression check: the new multi-candidate logic still picks
  the exact same boundaries as before (`0.3228, 0.3584, ...`), via the
  original "sum" profile strategy, unchanged. Its 2 rows checked so far
  (samples, before switching to all-rows) passed alignment.
- `sample 3`: the "longest_run" (close_width=20, prominence_frac=0.15)
  strategy is the only one that passes the header check (12/13), boundaries
  starting at 0.2794 -- much closer to the real table than the original
  strategy's 0.05.

**Not yet confirmed -- pick up here on resume:**
1. The final all-rows alignment check (checking every one of `sample 3`'s
   18 rows, not a sample) was implemented but the offline verification run
   was interrupted before completing -- rerun it to get the actual
   `bad_row_indices` list for `sample 3`'s winning candidate, and to
   reconfirm `sample 5` still passes on *every* row (not just the 3-5
   sampled before this last edit).
2. A real end-to-end Ollama run on `sample 3.jpeg` has NOT been done since
   these fixes -- `extracted/sample 3.json` on disk is from BEFORE this
   session's grid/alignment fixes and does not reflect them. Re-run
   `extract_ollama.py "Images/sample 3.jpeg"` fresh, then verify quantities
   against the actual image (not assumed from the row-crop spot checks
   above).
3. Regenerate `extracted/sample 3.review.html` after that fresh run and
   re-check it visually -- the version currently on disk is also stale
   (from before these fixes).
4. Stage A's `type`-field corruption on `sample 3` (every item got the
   secondary-header sequence `18,20,...50` instead of a real style code)
   is a separate, not-yet-investigated bug -- unrelated to the grid/OCR
   work above, still needs its own root-cause look.
5. Performance note: the multi-candidate + all-rows-alignment check adds
   real overhead (up to ~6 candidates x up to N rows of OCR calls) when
   the first/original strategy doesn't immediately win outright -- fine
   for a fallback path, but worth knowing if this ever needs to run on a
   large batch of images routinely.

## Claude API pipeline — built 2026-08-05, not yet run against a real key

**[extract_claude.py](extract_claude.py)**: a second, independent extraction
pipeline alongside the Ollama one above — not a port of its v3-v6
architecture. That architecture (schema-constrained structure call → CV row
cropping → per-row OCR/VLM reads → regex parsing) exists specifically to
work around a local 7B model's unreliable spatial grounding; Claude's vision
and instruction-following don't have that failure profile, so none of the
row-isolation scaffolding is used here. Instead: **one schema-constrained
call per image**, reading structure and all quantities in a single shot via
`output_config.format` against a Pydantic schema (`ExtractedForm` in
`extract_claude.py` — a different shape from `schema.py`'s `FormMeta`,
since structured outputs doesn't support the `Dict[str, int]` quantities
shape `OrderForm` uses: `additionalProperties` must be `false` for every
object, so quantities are a `List[{size, quantity}]` on the wire and
converted to the final `Dict[str, int]` shape after parsing). The domain
rules that were split across `prompt.py`'s Stage A/B prompts (seller vs
buyer, ditto-mark inheritance in both its forms, item-name/style-code
separation, blank-vs-zero) are consolidated into one system prompt. The two
Pydantic safety-net validators proven in `schema.py` (trailing style-code
split, ditto blank-item forward-fill) are ported onto the new schema rather
than imported, since the shapes differ.

**Cost design**, in priority order (see `extract_claude.py`'s module
docstring for the reasoning behind each):
1. **Message Batches API** (50% off every token) for any folder input —
   default behavior, since this pipeline already has a human review step
   downstream and nothing needs a synchronous response. `--live` opts out
   for quick tests on a couple of images.
2. **Prompt caching** (`cache_control: ephemeral` on the system prompt) —
   identical system prompt on every call, so image #2 onward in a run reads
   it at ~10% cost. Batches API supports caching too, so this stacks with #1.
3. **Structured outputs** instead of a prefill/regex parsing stage —
   guarantees valid JSON, so there's no Stage-C-equivalent post-processing.
4. **Sonnet 5 as the default model** (`DEFAULT_MODEL` in the script) —
   intro pricing ($2/$10 per MTok through 2026-08-31) vs Opus 5's $5/$25;
   `--model claude-opus-5` available for forms Sonnet 5 gets wrong.
5. Images are downscaled only if they exceed the model's high-res cap
   (2576px long edge) — this saves upload bytes, **not** API cost (Claude
   would downscale server-side and bill the same either way); documented
   as a deliberate non-lever in the script so it isn't mistaken for one
   later.

**Explicitly deferred, not forgotten** (see module docstring): tuning
`effort` down or disabling thinking, and self-consistency voting for hard
rows. Both need real accuracy data against a few sample forms first —
guessing a cost-saving setting without that risks silently degrading the
one thing this pipeline exists to get right.

**Verified so far (2026-08-05), offline only — no live API call made yet,
since no `ANTHROPIC_API_KEY` is configured in this environment:**
- `ExtractedForm.model_json_schema()` produces a structured-outputs-valid
  schema: `additionalProperties: false` and all fields in `required` at
  every level (top level + both nested `$defs`), confirmed by direct
  inspection.
- The ported validators behave identically to `schema.py`'s originals on
  the same test cases: `order_date` reasoning-leak stripped to
  `"30/13/26"`, `"Trend Trunk IE"`/type="" split to item="Trend Trunk"
  type="IE", a ditto row's blank item name forward-filled from the row
  above, and a negative quantity (`-5`) filtered out during the
  `ExtractedForm` → `OrderForm` conversion.
- `_prepare_image()` runs correctly against a real sample image
  (`Images/sample 5.jpeg`, under the 2576px cap, passed through
  unmodified) and `_request_params()` / batch `Request` construction both
  build valid request objects (checked via the SDK's own types, not just
  assumed).
- CLI argument parsing and the missing-`ANTHROPIC_API_KEY` error path both
  behave as intended.

## v1 — first real run, confirmed 2026-08-05

`ANTHROPIC_API_KEY` added to `.env`; ran live (`--live`, single image, no
batching) against both test forms with the default model (`claude-sonnet-5`).
Checked against the source images directly, not just schema validity.

**`sample 5.jpeg`: 13 of 14 rows match the OCR pipeline's already-verified
output exactly**, confirmed cell-by-cell. The one apparent mismatch (`MYNA`/
OE's `100` column: Claude said `13`, the stored Ollama-pipeline output said
`3`) was resolved by looking at the actual photo directly — it clearly
shows `13`. So the OCR pipeline's `sample 5.jpeg` output (and the v6.2
"every quantity matches exactly" claim above) was itself wrong on this one
cell; Claude's read is correct. Two issues the Ollama pipeline never
resolved after multiple documented attempts were fixed on the first try:
- **Ditto-name composition** (`"—"— Plain"` → `"Fairlady Plain"` /
  `"Bloomer Plain"`) — the Ollama pipeline tried this twice (a `ditto: bool`
  schema field, then leaving it to Stage A's own prompt) and reverted both;
  Claude does it correctly with no special-casing.
- **`F.G-3005` / `F.G-3025`** read correctly with no OCR cross-check
  needed (the Ollama pipeline required `item_name_ocr` override logic
  specifically to catch its own `F.G1-3005` misread).

`notes` also came back far richer than the Ollama pipeline's Stage A ever
produced — the circled GST letter, ambiguous small numbers near the
Fairlady Print row, every `x`-marked cell — none of which Stage A's prompt
asked for at that granularity.

**`sample 3.jpeg`: dramatically better than every Ollama-pipeline attempt**
(see the "IN PROGRESS" section above — that pipeline never got past garbled
item names, corrupted `type` fields, and misaligned row crops on this
image). Verified via the form's own printed "Total Dozen" column as a
built-in checksum: summing Claude's per-size quantities against that
printed total matched exactly on **18 of 20 rows**. The two rows that don't
sum to the printed total (`XUV Half Shirt`, `MM K4532`) are ones where some
cells are letter-coded (S/M/L/XL/XXL) instead of numeric — Claude correctly
flagged and omitted those non-numeric cells per the "leave it out rather
than invent a value" rule instead of guessing, which is why the checksum
comes up short there; this is the schema's documented limitation (only
`Dict[str, int]` is representable), not a misread. No trace of the Ollama
pipeline's `type`-field corruption bug (every item's `type` coming back as
the secondary-header sequence) — every `type` value is a real style code or
correctly blank. Party name (`"A.T Distributors, HYDERABAD"`) also reads as
a far more plausible real name than the Ollama pipeline's `"A.T
Dibbawudas"`.

**Cost, this run:** `sample 3.jpeg` (harder, 20 items) cost ~$0.1383 live
with a cold cache (`in=1507 cache_read=0 cache_write=2269 out=12959` on
Sonnet 5) — no batch discount and no cache reuse since it was the first/
only call in that process. Real batch runs (default behavior for a folder
input) would be ~50% cheaper on top of that, plus cache reads at ~10% cost
for every image after the first in the same run. This number was read off
the console print, not persisted anywhere — see the next entry.

**Usage logging added (2026-08-06):** every call (live or batch) now
appends one row — timestamp, image, model, mode, all four token counts,
estimated USD cost — to a persistent `usage_log.csv` (default, override
with `--usage-log`), independent of `--outdir`. `main()` prints a running
grand total across every logged call at the end of each run. This exists
specifically so cost can be tracked and judged for plausibility across
many runs/sessions, not just read off one run's console output (which is
exactly the gap that lost `sample 5.jpeg`'s first-run cost figure — that
call predated this feature and its exact usage was never captured).
Verified offline with mocked usage objects (matches the real `sample
3.jpeg` cost above exactly, confirming the formula didn't drift during the
refactor) — not yet exercised by an actual live run since adding it.

**Bug found and fixed (2026-08-06, later same day): usage was logged
*after* parsing, not after the call.** `extract_one_live()` called
`_parse_message(message)` before the usage-logging block — so when a call
failed *after* it was made (the `MAX_TOKENS` truncation bug above is
exactly this case), the exception skipped past the logging code entirely
and that call's real, billed cost was never written to `usage_log.csv`.
Same bug existed in `extract_batch()`'s loop (a batch result can report
transport-level `"succeeded"` yet still fail our client-side JSON parsing
on a truncated response). Fixed in both: usage is now logged immediately
after the response comes back, before `_parse_message()` runs, so a
failed/truncated call is tracked exactly like a successful one going
forward. The one call this bug actually affected (the truncated `sample 3`
run right before this fix) has its cost reconstructed by hand as a
one-time baseline — see [[project-orderform-claude-pipeline]]'s "Running
call/cost tally" section — since it's not recoverable from the log itself.

**Not yet done:**
- `sample 3.jpeg`'s two letter-coded rows and the exact reading on a
  couple of borderline item names (`"fca 4131"`, `"Loop 82 4211"`) haven't
  been zoomed-in/pixel-verified — worth a closer look before trusting them
  completely, though the checksum match on every other row is strong
  indirect evidence the read is generally sound.
- No batch-mode run yet (only `--live` single-image calls so far) — the
  batching code path (`extract_batch()`) is exercised by the offline
  `Request`-construction check above but not by a real batch submit/poll/
  results cycle.
- Only tested against Sonnet 5 (the default) — Opus 5 not yet compared for
  accuracy on either form.

## Prompt hardening + local brandlist cross-check — built 2026-08-06

Three accuracy gaps identified from real usage (not hypothetical): (1) item
names need cross-checking against the actual product catalog, since Claude
can misread a digit or a whole name; (2) some cells deviate from the
printed grid entirely — the writer puts an override SIZE label in the cell
with the QUANTITY below/beside it, confirmed by zooming into
`sample 3.jpeg`'s `XUV Half Short`/`MM K4532` rows; (3) some products use
standard clothing letter sizes (S/M/L/XL/XXL) instead of this form's
numeric grid at all, because the product's real sizing doesn't match the
generic 45-105 range — confirmed directly against the DB: `MM K 4532 B
FULL PANT SET` (style `RNS`) has real catalog sizes `{35,40,45,50,55}`, a
kids range with no relation to what's printed on the form.

**Fix for (2) and (3): prompt-only, zero extra cost.** `SYSTEM_PROMPT_TEMPLATE`
in `extract_claude.py` now has explicit rules: read a cell's actual
handwritten size label when one overrides the printed header (FREEFORM
PLACEMENT), and report clothing letter sizes verbatim in the `size` field
rather than guessing a numeric conversion (LETTER SIZES) — a later local
step resolves the real number. Same one call, same image, just better
instructions; the only cost is a few hundred more (cached) prompt tokens.

**Fix for (1) and the letter→number resolution: [brandlist_match.py](brandlist_match.py),
entirely local, zero API calls.** Connects to the same SQL Server DB noted
in "Database cross-check" above (`brandlist`, `isprimary=1`, confirmed
61,752 rows / 8,840 distinct product names / 24 distinct styles / 20
distinct sizes via `pyodbc`, already available in this environment). Two
things, both gated by a trust check, not blind score thresholds:
- **Item-name/style suggestions** via `rapidfuzz` fuzzy matching — surfaced
  in `<name>.brandlist.json` and rendered as a note under each row in
  `generate_review.py`'s page. `item.item` is never auto-changed (name
  identity is the genuinely ambiguous part); a blank `type` IS auto-filled
  when the matched product has exactly one catalog style.
- **Letter-size resolution** — `S`/`M`/`L`/etc. converted to the real
  numeric size by checking which of two mappings (adult: 75→XS...110→4XL;
  kids: 25→0...55→XXL, both supplied by the business) actually appears in
  the *matched product's own* catalog sizes, never assumed from the letter
  alone. Applied automatically once a match is trustworthy, since it's
  mechanical unit conversion at that point, not an identity guess.
- The trust gate (`_is_trustworthy`) uses whether a numeric product code
  in the item name narrowed the candidate pool as the *primary* signal,
  not the raw fuzzy score — confirmed necessary by testing: a correct
  code-narrowed match (`"MM K4532"` → the one catalog name containing
  `4532`) scored only 85.5 due to an extra middle word, while a *wrong*
  whole-catalog match (`"MM Loop 4289"`, whose `"4289"` doesn't exist
  anywhere in the catalog) also scored 85.5 from generic word overlap
  alone. Score-only gating would have trusted both or neither; the
  narrowing signal correctly separates them.
- A separate, real bug caught by testing: `rapidfuzz.fuzz.WRatio` without
  `processor=utils.default_process` is case-sensitive, so `"full"` vs
  `"FULL"` alone tanked an exact match's score from 95.0 to 44.4. Confirmed
  via direct side-by-side scoring before/after adding the processor arg.
- Also caught: a letter size can resolve to a numeric size that's *already*
  present as its own key (e.g. a printed-header `"45"` cell was read
  directly, and a different cell's letter `"L"` also resolves to 45 for
  that product). Fixed to detect the collision against a pre-resolution
  snapshot and flag it (`size_conflicts` in the annotation) rather than
  let a plain `dict.update()` silently pick whichever key was processed
  last — confirmed via a synthetic collision test that the original `45`
  reading survives untouched.

**Verified so far, offline (reusing the already-paid-for `sample 3.jpeg`
extraction) — no new Claude API call made for any of this:** DB
connectivity confirmed live; the numeric-code-priority + case-normalization
matching fixes confirmed against real catalog data (18 of 20 sample-3 items
now get a plausible match, several at 90+ score); `type_filled_from_catalog`
confirmed firing correctly for `MM K4532` → `RNS` (previously blank);
`code_not_in_catalog` confirmed firing for `Loop 82 4211` / `MM Loop 4289`
(neither digit code exists anywhere in the 61K-row catalog — likely a
Claude misread or a genuine catalog gap, flagged either way); letter-size
resolution and the collision-detection fix both confirmed via synthetic
test cases built from the real `MM K 4532` catalog entry, since the saved
`sample 3.json` predates the prompt change and never captured letter-coded
cells in the first place. `generate_review.py` confirmed rendering all of
this as `.db-note` text under each row.

**First real re-run attempt (2026-08-06) found a genuine bug, not yet a
clean result:** the user ran `extract_claude.py "Images/sample 3.jpeg"`
with the new prompt and hit a truncated-JSON `pydantic.ValidationError`
(`EOF while parsing a value`) — `stop_reason` was `"max_tokens"`. The old
`MAX_TOKENS=16000` wasn't enough: the new prompt reports letter-coded
cells instead of dropping them, so a dense 20-item form like `sample 3`
now generates more output than before. Fixed: `MAX_TOKENS` raised to
32000, `extract_one_live()` switched from a plain `create()` call to
`client.messages.stream()` + `get_final_message()` (required once
`max_tokens` exceeds the SDK's ~16k non-streaming safe threshold), and
`_parse_message()` now raises a clear error naming the real cause
(`stop_reason == "max_tokens"`) instead of surfacing a cryptic pydantic
JSON-parse error. Verified offline: request construction at the higher
`max_tokens` still builds correctly for both the live (streaming) and
batch (non-streaming, unaffected by the same guard) code paths.

## v2 — confirmed working end-to-end, 2026-08-06 (same day, after the MAX_TOKENS fix)

Re-ran `extract_claude.py "Images/sample 3.jpeg"` after the truncation fix.
Completed successfully this time. Checked `<name>.raw.json` (what Claude
actually returned) directly:

- **`MM K4532`** now reports letter-coded sizes verbatim exactly as
  instructed: `{"S":6, "M":6, "L":6, "XL":6, "XXL":6}` in the raw output —
  matching what the zoomed image showed. `brandlist_match` then resolved
  every one of them via the kids mapping to `{"35":6, "40":6, "45":6,
  "50":6, "55":6}` (the confirmed real catalog sizes for `MM K 4532 B FULL
  PANT SET`), auto-filled `type` from blank to `RNS`, and the row now sums
  to **30 — exactly matching the form's printed Total Dozen** for that row.
- **`XUV Half Short`** (the other letter-coded row) now sums to **37**,
  also an exact match to its printed total — up from 19 in the pre-fix run,
  where the letter cells were simply dropped.
- **`MM K 3674 T-Shirt`** sums to 30, also matching.
- Every other numeric row still checks out against its printed Total Dozen,
  same as the v1 run.

**Confirmed: both mechanisms (prompt-only letter-size passthrough,
brandlist-based resolution) work together correctly on a real call, closing
the gap flagged after the truncation bug.**

**Caveat, not a regression:** a few borderline item names shifted between
the v1 and v2 runs on genuinely hard handwriting -- `"Loop 82 4211"` (v1,
digit code not in catalog) became `"Loop 82 Cn 4214"` (v2, also not in
catalog), and `"Salvo Baby Full Pant"` became `"Salvo Boy Full Pant"`.
Normal model variance (not deterministic sampling) on the hardest-to-read
names, not something either prompt change caused -- these specific names
still aren't fully pinned down and would benefit from a human glance in the
review page.

## Column-shift detection — built 2026-08-06, same day

The user spotted a real column mismatch in the v2 output: `B-4749 Full
Pant` and `B-4748 Full Pant` both had every value attributed one column
LEFT of where it actually is (confirmed by zooming into the source photo
at high resolution and checksumming against the printed Total Dozen —
e.g. `B-4749`'s real values are `55:5,60:8,65:8,70:8,75:8,80:8,85:8,90:5`,
not the reported `50:5,55:8,...,85:5`). The row's *sum* is identical
either way, which is exactly why the Total-Dozen-checksum method used
throughout this doc couldn't catch it on its own — a shift relabels
values, it doesn't change their total.

**Correction (2026-08-14): the "real values" above are wrong.** Directly
viewing `Images/sample 4.jpeg` confirms `B.4749`'s row is 8 single marks
(each worth `1`) under `55,60,65,70,75,80,85,90`, matching the row's own
circled Total Dozen of `8` (8 × 1 = 8, not the `55:5,...,90:5` sum of
58 stated above, which was never actually checked against the photo
carefully enough despite the "confirmed by zooming in" claim). Per the
user, this holds for every row on this entire form — every marked cell
on `sample 4.jpeg` is a single-unit mark, full stop, not a varied
multi-digit quantity. This was a real, consequential error: this row was
used as the trusted ground-truth anchor for judging every model's
`sample 4.jpeg` accuracy in every session below through 2026-08-13 — see
the dedicated correction section near the end of this file for how that
changes those conclusions.

**Explicitly rejected: hand-patching the output JSON.** First instinct was
to just correct the two rows directly in `sample 3.json` — the user
stopped this immediately ("I don't want you to fix it directly in the
output, I want it fixed directly when we run it"), correctly redirecting
toward a pipeline-level fix rather than a one-off data patch.

**The actual fix — `brandlist_match.py`'s third mechanism, entirely
local/free:** `detect_column_shift()` cross-references a row's reported
sizes against the form's own printed header order (`extracted.size_headers`,
threaded through from `<name>.stageA.json`) AND the matched product's known
catalog sizes together. If shifting the reported sizes by a small offset
(±1, ±2 header positions) makes them land exactly inside the catalog's
valid range, that's flagged as `likely_column_shift` with the *exact*
suggested correction — not just "something's off" but "here's what it
probably should be". Confirmed catching the real bug: on `B-4749`, it
correctly found offset `+1` and reproduced the same correction independently
verified by hand.

**Deliberately a flag, never an auto-correction** — unlike letter-size
resolution (a mechanical unit conversion once the product is known), this
rewrites already-plausible-looking numbers, and testing surfaced a real
false-positive risk: the same run also flagged `B 5109 3/4 Set`, which had
already been independently verified correct against the photo earlier in
this session. The catalog simply doesn't list size `45` for that specific
product (a catalog completeness gap, not a misread) — mathematically
indistinguishable from a real shift using this signal alone. `generate_review.py`
renders both `likely_column_shift` (⚠⚠, with the specific suggested fix)
and the weaker `sizes_outside_catalog_range` (⚠, no shift found) as
`.db-note` text, worded to prompt a photo check rather than assert an error.

**Not yet done:** a fresh end-to-end run to see whether this specific
misread (or others like it) still occurs with the current prompt — this
fix is a detection/flagging layer on top of whatever Claude returns, not a
prompt change, so it doesn't reduce how often shifts happen, only how
visible they are when they do. A prompt-side mitigation (e.g. reminding
the model that cramped/overlapping handwriting near the Style column
shouldn't shift its sense of where the size columns begin) was considered
but not added — unverified without another paid run, and the project's
history (ditto-mark composition, attempted twice, reverted twice) argues
for not layering an unverified prompt change on top of an already-large
change set without testing it in isolation first.

## Row-crop quantity recount — built 2026-08-07, partially verified

The user reported `sample 4.jpeg`'s size→qty mismatch as unacceptably high
and, critically, supplied ground truth the pipeline itself couldn't derive:
only rows 1, 2, 3, and 8 of 12 were actually correct. This matters because
the existing checksum-based confidence signal (row sum vs. the printed
"Total Dozen") had been wrongly reassuring — a column shift doesn't change
a row's sum, so 11/12 rows appeared to "check out" by that signal alone
while most were actually column-shifted. Confirmed directly: row 5
(`B.4749 F/PANT SET`) was flagged by `brandlist_match.py`'s existing
`likely_column_shift` (reported `50–85`, catalog's real sizes `55–90`,
offset +1) — but most of the other wrong rows were NOT flagged, because
either the item's catalog match scored below the trust threshold (rows
9–11), or the shifted sizes happened to still fall inside an overlapping
catalog range for a similar product (this form's rows mostly span similar
7-8 contiguous columns, so a 1-column shift often doesn't leave the
catalog's valid range at all). So the catalog cross-check alone was never
going to catch most of this.

**Root cause, confirmed by inspecting the actual photo:** most rows on
this form are dense tally marks (repeated handwritten "1" strokes across
13 narrow columns) rather than distinct multi-digit numbers — there's
nothing content-based to anchor a column to, so it's a pure visual
counting task across a wide row on a full-page image, and it's easy to
get the COUNT right while still misjudging WHICH column the block starts
under. `sample 5.jpeg`'s earlier "13 of 14 rows match" result (see the v1
section above) used mostly distinct multi-digit numbers, not
tally-of-ones — this form's failure mode is much more acute specifically
because the data is repetitive tally marks.

**Fix implemented in `extract_claude.py`:** a second, focused Claude call
per image (live mode only so far — see below), sent a tight crop of just
the item table (using new `table_top_frac`/`table_bottom_frac` fields the
main call now also returns), enlarged toward the model's resolution cap.
This removes the letterhead/party-box clutter the main call also had to
look at, and the new schema (`QuantityRecount`/`RowQuantityReading`)
forces the model to explicitly commit to `first_size`/`last_size` anchors
per row and self-check against the row's own printed Total Dozen, instead
of freely emitting size:quantity pairs. Results are only merged back into
an item when its `item_seen` text (also returned per row, for alignment
checking only) plausibly matches the item already at that position — a
misaligned crop is detectable this way rather than silently trusting it.
Also added: a prompt rule (both calls) telling the model not to drop
quantity marks for an item whose row is squeezed outside the main ruled
grid, overlapping printed footer text (`Stock Entry By`/`Checked
By`/`Signature`) — see below for why this matters.

**Verified so far, real run against `sample 4.jpeg` (2026-08-07):**
- **Row 5 is now exactly correct** — the one row with independent ground
  truth (the product catalog): reads `55–90`, an exact match to
  `B 4749 FULL PANT SET`'s real catalog sizes, and `brandlist_match.py`
  now reports zero flags for it (previously `likely_column_shift`,
  offset +1). This is real, confirmed evidence the mechanism works, not
  just a plausible theory.
- **Not a full fix, confirmed by the same run:** row 6 (`SPRIT F/PANT
  SET`) changed from `50–85` to `60–95` — still NOT fully inside its
  catalog range (`55–90`; `95` is invalid) despite the recount pass.
  `brandlist_match.py` didn't flag this either, for a separate, pre-existing
  reason: this item's catalog match scores 91.4, below
  `AUTO_APPLY_UNNARROWED_SCORE` (95) since it isn't code-narrowed, so the
  out-of-range/shift check is skipped entirely for it (see
  `_is_trustworthy` — a real gap, not a bug in the new code, but worth
  knowing: the flag-based safety net has blind spots exactly where the
  recount pass needs it most).
- **Row 12 (`MYD D.7 3/4TH PANT`, the footer-overlap row) produced a new,
  suspicious reading**: `50, 55, 90, 95, 100:4` — a single cell with
  quantity 4 (every other cell on this entire form is a single tally
  mark, i.e. always 1) and a 35-point gap with no marks in between,
  breaking the "contiguous block" pattern real data on this form has
  followed everywhere else. Sums to 8, matching the printed Total Dozen,
  but the shape looks more like a misread than a real fix — this
  remains the hardest row on the form (footer text overlapping the
  quantity marks) and needs a human look, not an automated one.
- Rows 4, 7, 9 also changed from the original run but have no independent
  ground truth (no trustworthy catalog match, or match sizes wide enough
  that both old and new readings fit) — **not yet confirmed correct or
  wrong**, unlike row 5.
- Cost impact, confirmed from the same run: recount call added
  ~$0.05 to sample 4's ~$0.11 main-call cost (in=5084 cache_write=1909
  out=3886 for the recount vs in=1543 cache_write=3476 out=10167 for the
  main call) — roughly +50% per image, not the doubling a naive second
  full-page call would cost, since the crop is smaller and the recount
  output has no item names/notes to generate.
- **Not wired into `extract_batch()`** — the recount pass needs each
  image's own `table_top_frac`/`table_bottom_frac` from its first-call
  result before building the second call, which would mean a second
  batch submit/poll round-trip per run. Only `extract_one_live()` has
  this today; batch-mode users won't get the recount pass until that's
  built.

**Honest bottom line:** this is a real, evidence-backed improvement (one
independently-confirmed fix, a plausible mechanism, real behavior change
on most previously-wrong rows) but not a proven complete fix — rows 6 and
12 are known-still-suspicious in this exact run, and rows 4/7/9 are
unverified either way. Needs the user's own ground truth (they have it for
this real order) on the remaining rows, and ideally a couple more real
runs on other ESSA-form samples, before trusting this as "solved" per this
project's own verify-before-trusting practice.

## True per-row crops + agree-or-flag gating — built and verified 2026-08-07, same day

The user's bar, stated directly: "I want size -> qty to be 100%, then only
this automation process can be implemented." Two rounds of follow-up work
this same day, both confirmed via real runs, not just reasoning:

**1. True per-row crops (not just the whole-table crop above).** The
whole-table-crop recount above left rows 10 and 11 byte-identical to the
un-recounted original — proof it wasn't really isolating rows, just
giving the same multi-row image more pixels. Rebuilt: the main call now
also returns `row_top_frac`/`row_bottom_frac` per item (cheap -- same
call), validated for a plausible monotonic partition
(`_validate_row_fracs`), then used to build one true header+row crop per
item (`_build_row_crops`) sent as N images in the recount call, falling
back to the old whole-table crop if the per-item fractions don't
validate.

**Bug caught before it shipped, not after:** the first real test of this
came back with nearly every row reading the form's SECONDARY header line
(18,20,22...) instead of the real one (45,50,55...) -- the main call's
prompt already warns against this, but the recount call has its own,
separate system prompt that never inherited the warning. Fixed the prompt
AND added a hard validator (`_apply_recount` now rejects any row whose
reported digit sizes aren't in `extracted.size_headers`) so this class of
bug can't silently corrupt output again even in a different form. This is
exactly the kind of near-miss the project's verify-before-trusting habit
exists to catch -- confirmed via a real re-run after the fix, not assumed
fixed from the code change alone.

After the fix, confirmed real wins on `sample 4.jpeg`: row 6 (`SPRIT`) now
reads `55-90`, exactly its catalog range (was `60-95`, invalid, under the
whole-table crop); row 12 (footer-overlap row) now reads a clean
contiguous `45-80` summing to its Total Dozen (was a suspicious
`100:4`-anomaly reading before). Row 5 stayed correct. Cost: recount rose
from ~$0.05 (whole-table) to ~$0.16-0.19 (per-row, N images in one call).

**Found on `sample 7.jpeg` immediately after, and it's important:** the
user caught (correctly, on visual inspection) that rows 6-7 of that
sample (`Fair lady Plain`/`Fair lady Round`, the LAST two rows on that
form) were shifted one column right (reported `80-105`, real `75-100`) --
**even with true per-row isolation.** Diagnosis: this isn't the
"counting across a wide row" failure per-row-crops fix -- it's
**perspective drift**, the same camera-angle warp the old Ollama/
PaddleOCR pipeline had to build explicit self-calibration for (v6.1
above), which grows with row depth and isn't corrected just by cropping a
row in isolation, since the stitched-in header crop is still the
top-of-table header's pixel geometry, unchanged. Per-row crops fix
"lost count of which column while scanning a 13-column row on a full
page"; they do NOT fix "the columns have physically drifted by the time
you're 6 rows down."

**2. Agree-or-flag gating**, built in response to the user's actual bar
("size -> qty to be 100%") -- reframed as: 100% of what ships is verified,
not "the model is never wrong." `_apply_recount` now compares the main
call's reading against the recount's reading PER ROW:
- Exact agreement -> kept, no flag (real corroborating evidence from two
  independent reads -- different crop, different prompt).
- Disagreement -> NOT silently resolved by preferring one; Total Dozen
  (when printed) breaks the tie if exactly one side matches it, else the
  main reading is kept as a default but the row is flagged either way,
  with both full readings and the specific differing cells spelled out in
  a `"NEEDS REVIEW -- ..."` note.

**Confirmed via real re-runs, both samples:**
- `sample 7.jpeg`: this run, main and recount agreed everywhere,
  INCLUDING the corrected `75-100` for rows 6-7 -- genuine independent
  corroboration of the fix, not a lucky pick (no flags fired).
- `sample 4.jpeg`: 7 of 12 rows flagged in one run. Real value misreads
  caught and auto-corrected via the Total Dozen tiebreak (row 1: a `1`
  the main call misread as `4`; row 2: a `1` misread as `9`). Row 4 (
  unstable across every earlier test) is now honestly flagged instead of
  silently guessed -- both readings sum to the same total, so there's no
  automatic tiebreak, and it correctly stays uncertain rather than being
  arbitrarily resolved. Rows 5-7 flagged on a 1-column edge disagreement;
  notably the recount's answer for `B.4749` here was the SAME one already
  catalog-confirmed correct, but Total-Dozen-only tiebreaking couldn't
  see that and defaulted to keeping the (wrong) main reading -- still
  flagged for a human either way, but this shows the tiebreak should also
  consult `brandlist_match.py`'s catalog check, not just Total Dozen --
  not yet wired in, since that check currently runs in a separate
  post-processing pass in `main()`, after `extract_one_live()` returns.

**Catalog tiebreak wired into `_apply_recount`, same day, confirmed
working:** `brandlist_match.find_best_match()` (already loaded once via
`main()`'s `known_style_codes()` call, so this is a free in-memory lookup,
zero extra API cost) now runs as a second-priority tiebreak when Total
Dozen doesn't resolve a disagreement. Verified on a fresh sample 4 run:
`B.4749`'s disagreement was auto-resolved to the catalog-correct `55-90`
reading, closing exactly the gap identified above. `brandlist_available`
threaded through from `main()` down through `extract_one_live()` ->
`_recount_quantities_live()` -> `_apply_recount()`.

**Effort reduction on the recount call, tried and reverted (2026-08-07,
same day):** to fight the occasional `RECOUNT_MAX_TOKENS` truncation (a
12-image recount call once hit exactly 32000 output tokens and wasted its
~$0.35 cost with zero rows applied -- Sonnet 5 runs adaptive thinking at
`high` effort by default even with no `thinking` param set, and that's
genuinely unpredictable across runs), set `output_config.effort: "medium"`
on the recount call only. **Reverted after one real re-run**: row 5
(`B.4749`), the one row with independent catalog ground truth, dropped
from its confirmed-correct 8 filled cells down to 3 -- a real accuracy
regression, not just a cost tradeoff, and it got auto-"resolved" against
the wrong Total Dozen in the process. Truncation is rare enough to just
eat as an occasional wasted call (the existing max_tokens guard already
falls back safely to the original reading rather than shipping a
truncated one) -- not worth trading row-level accuracy for. Left at the
default (adaptive thinking, high effort, no explicit `effort` override)
on purpose. Don't re-try lower effort here without new evidence it
doesn't cost accuracy.

**Not solved by any of this, confirmed as still-open:**
- Perspective drift on late-table rows (the sample 7 rows 6-7 failure
  mode) is not actively corrected, only sometimes caught by disagreement
  -- if both calls share the same drift bias, they can still agree on a
  wrong answer. A real fix would need something like the Ollama
  pipeline's drift self-calibration, not yet ported.
- The tiebreak only consults Total Dozen, not the catalog -- a real,
  identified gap (see the `B.4749` case above).
- Batch mode still doesn't run any of this (recount is live-mode only).
- Item-name/style-code splitting (e.g. "F/PANT" mistaken for a style
  code) is a pre-existing, unrelated Stage-1 issue that occasionally
  causes the alignment check to reject a correct recount -- not addressed
  in this pass.

## Multi-form-type survey — 2026-08-07, noted for later, not acted on

At the user's request: surveyed all of `Images/` (13 samples) since the
pipeline so far has only been tuned/tested against the business's own ESSA
printed order form (samples 3, 4, 5, 7). Confirmed by direct inspection —
real diversity, not a hypothetical concern:

- **Free-form handwritten notebook pages, no printed grid at all**
  (samples 1, 2, 6) — party-side notes with size headers written locally
  per item/block rather than one shared table (e.g. sample 1: "80 85 90
  95 100" written once above a quantity line, repeated per item; sample 6:
  item + style + a size RANGE written as text, e.g. "RNS-45 TO 80", plus a
  bundle count, no per-size quantities at all in some rows).
- **Other businesses' own printed order-form templates** (samples 8, 10,
  12), each with different conventions from ESSA's: sample 10 uses `*` as
  a quantity marker in many cells instead of a digit (meaning unclear —
  possibly "per standard ratio"); sample 12 double-labels size headers
  with both a number and a letter size on the same column (e.g. `80/S`,
  `85/M`) as two stacked header rows that both apply, unlike ESSA's second
  header row (which is a decoy to ignore).
- **"Excel-maintained" dense grid templates** (samples 9, 13) — cleaner
  and more tabular than ESSA's handwritten-into-printed-grid form, but
  with their own new conventions: sample 9 has curly braces (`{`/`}`)
  drawn across several columns, apparently grouping one written quantity
  across a combined size range instead of one cell = one size; sample 13
  has a column-wise "GRAND TOTAL" footer row (summing down each size
  column) instead of ESSA's per-row "Total Dozen" — a different, untested
  cross-check signal.

**Decision: stay focused on ESSA-style forms (3, 4, 5, 7) for now** — per
the user, this is the template to "make click" first, with explicit
acknowledgment that all of the above will need handling eventually.
Nothing built yet for the other templates; no prompt rules added for `*`
markers, brace-grouped cells, dual-label headers, free-form notebook
layout, or column-footer totals — noted here so this doesn't get
rediscovered from scratch later.

**Scope note:** the very next day's session (below) ran real calls against
`sample 2.jpeg` and `sample 8.jpeg` anyway — a de facto reversal of "stay
focused on ESSA-style forms," though nothing in that session's code
comments explicitly calls it out as one.

## Per-row parallel recount + drift-trend calibration + buyer cross-check — built 2026-08-08

Three changes, all in `extract_claude.py`/`brandlist_match.py`/
`generate_review.py`, none yet reflected anywhere else in this file before
now. Verified 2026-08-11 by re-reading the code and the real output
artifacts under `extracted_claude_test7/` (a scratch outdir from this
session, distinct from `extracted_claude/`, which still holds only
pre-08-08 output) — not just inferred from the diff.

**1. Recount calls split from one bundled call into N parallel per-row
calls — confirmed working via real runs.** The per-row-crop recount path
(added 2026-08-07) originally sent all of a form's row crops in a single
`QuantityRecount` call. `usage_log.csv` shows that call regularly hit the
32000-token cap on forms with 12+ rows (three 2026-08-07 `sample 4.jpeg`
`live-recount-per-row` rows show `output_tokens` at exactly `32000` or
`31689`) — silent truncation, with no record of which rows past the
cutoff simply never got recounted. Fixed by splitting into one call per
row (`_recount_one_row()`), fired concurrently via
`concurrent.futures.ThreadPoolExecutor` (new `RECOUNT_MAX_WORKERS = 6`),
each with its own `RECOUNT_ROW_MAX_TOKENS = 8000` budget — an order of
magnitude harder to exhaust than one call synthesizing every row's
reasoning together. The old single-call path is kept, unchanged, as the
whole-table fallback for forms whose per-item `row_top_frac`/
`row_bottom_frac` don't validate (e.g. free-form layouts like
`sample 2.jpeg`, which used it both times it was run on 08-08).
Confirmed via real runs, mode string includes the row count:
`sample 4.jpeg`'s `live-recount-per-row-x12` (12 rows, in=14006
cache_read=12288 cache_write=12288 out=11633, ~$0.178) and
`sample 8.jpeg`'s `live-recount-per-row-x10` (in=7644 cache_read=17248
cache_write=7392 out=14575, ~$0.183) both completed with no row hitting
its 8000-token cap — a real, log-visible fix, not just a plausible-sounding
one. Note: no comment in the code explicitly states "confirmed via fresh
run" for this specific change (unlike most other fixes in this file) —
the confirmation here is from re-inspecting the real `usage_log.csv` rows
and `recount_flags.json` output directly during this doc update, not from
the original session's own notes.

**2. Row-depth drift-trend calibration (`_fit_drift_trend`) — implemented,
reuses v6.1's proven guard values, but NOT yet confirmed active on any
real run.** Directly aimed at the open gap flagged at the end of the
"agree-or-flag gating" section above: two independent calls can still
share the same camera-angle perspective bias and agree on the same wrong
column (confirmed real on `sample 7.jpeg` rows 6-7). Mirrors the Ollama
pipeline's v6.1 self-calibration (same idea — regress a per-row
correction against row depth from confidently-measured anchor rows) but
regresses a **catalog-confirmed header-column offset** (an integer from
`brandlist_match.resolve_shift_offset()`) instead of v6.1's raw pixel
drift, since Claude doesn't report pixel coordinates. Deliberately reuses
v6.1's exact activation guards (`MIN_CALIBRATION_ANCHORS = 3`,
`MIN_CALIBRATION_Y_SPREAD = 0.08`) rather than re-discovering v6.1's own
lesson that a stricter `0.15` spread threshold silently never activates
on a real form. The fitted trend is used only as a flag-text hint on a
disagreement that has no catalog match of its own to resolve it directly
— never an auto-correction. **Checked directly against the three real
08-08 runs (`sample 2`, `sample 4`, `sample 8` `.recount_flags.json`
files) as part of this doc update: zero rows anywhere show
`"status": "auto_corrected"` and zero notes contain the trend-hint text**
— meaning none of these runs had ≥3 catalog-confirmed anchor rows with
enough row-depth spread to activate it. Logically sound and reuses a
mechanism already proven in a different pipeline, but per this project's
own verify-before-trusting standard, this should be treated as **not yet
demonstrated to work**, only implemented — needs a form with more
catalog-matched rows spread across more of the table depth to actually
exercise it.

**3. Buyer-table cross-check for non-ESSA forms (`resolve_party_name`) —
confirmed working on `sample 8.jpeg`.** Testing against a non-ESSA
template surfaced a new failure mode the existing party-name logic never
anticipated: on ESSA's own pad, the printed letterhead is always Essa and
the handwritten "Party Name" box is always the real buyer. `sample 8.jpeg`
is a *different* business's own printed order pad — its letterhead reads
`"M/s. Sharda Sales Solapur"` and its handwritten Party Name reads
`"Rajkumar & Co."`, inverting that assumption entirely. New
`resolve_party_name()` cross-checks *both* fields against the real
`buyer` DB table (new `BUYER_MATCH_THRESHOLD = 88.0`, chosen because
generic word-overlap between two unrelated names caps out around 85.5 —
the same ceiling already confirmed for unrelated item names in the
brandlist item-matching work). Confirmed via the real persisted
`sample 8.party_check.json`: the letterhead scores 95.0 against real buyer
`"M/S SHARDA SALES (Z CLOSE)"` (Solapur — a real match), while
`"Rajkumar & Co."` scores only 85.5 against its best buyer candidate (an
unrelated name — not a real match, correctly below the threshold). Never
overwrites `party_name` automatically; returns a suggestion +
human-readable note surfaced in the review page. Also caught, on this
same form, that the buyer table has near-duplicate closed/active variants
of the same business and `IsActive` doesn't reliably distinguish them
(the matched record itself is a `(Z CLOSE)` variant) — flagged as a
`closed_hint` note rather than silently trusted. Unlike the recount pass,
this check runs from `main()`'s post-processing loop for both live- and
batch-extracted forms, not just live mode.

**Paired fix: `generate_review.py` was silently hiding real low-score
matches.** The review page's match-display logic used a plain score
threshold, but `brandlist_match.py`'s own matching logic (built
2026-08-06/07) already had a documented bypass for unique-code matches —
a match can have a legitimately low *text* score (surrounding words
misread) while still being a certain identity match purely because a
digit product code narrows the catalog to exactly one candidate (e.g.
`"CREBO B4756 F/PANT"` scoring only 45.2–51.6 against its one real
candidate, per the 2026-08-07 `AUTO_APPLY_UNIQUE_CODE_SCORE` note). The
review page never implemented that same bypass, so genuinely-correct
unique-code matches were invisible to the reviewer — confirmed concretely
on `sample 4.jpeg`: `'CRERO B4756 H/PANT'` (score 51.6) and `'LOOPER
ZB3965 H/PANT'` (score 69.4) would not have rendered any note pre-fix.
Fixed by exporting a new `narrowed_pool_size` field from
`brandlist_match.annotate_and_resolve()` and having the review page's
display gate check `code_narrowed and narrowed_pool_size == 1` as an
alternate pass condition, mirroring the matcher's own internal rule
instead of re-deriving a separate one. Also added: per-row and per-cell
recount-flag rendering (`_friendly_recount_summary()` translates a raw
`unresolved`/`resolved`/`auto_corrected`/`unverified` status into a
plain-language sentence for a non-engineer reviewer; flagged cells get
their own highlighted CSS class + tooltip instead of the signal being
buried in a paragraph at the top of the page).

**Confirmed-still-open on `sample 8.jpeg` specifically, not fixed by any
of the above:** several rows (`'Calm'`/RN, `'20-20'`) have main and
recount readings that disagree completely (not just a column shift, e.g.
sizes 80-105 vs 45-95) and stay `unresolved` — no signal here was strong
enough to pick a side. The form's printed "total" column also doesn't
appear to be a per-row quantity sum the way ESSA's Total Dozen is (e.g.
`Fairlady Plain`: both calls agree on 134, printed value is `8`) —
flagged `unverified` with an explicit note that this number may be a
different kind of count entirely (e.g. a bundle/pack count) rather than
assumed to be a checksum failure. Both are genuinely new-template
problems, not a regression in anything built for ESSA forms.

**Not yet done:** no wall-clock timing is logged anywhere
(`usage_log.csv` has no duration column), so the latency-improvement
reasoning behind change #1 is not itself a stored, re-checkable number —
treat it as plausible from the token-budget math and the log's own
timestamps, not as a formally measured benchmark. Drift-trend calibration
(#2) needs a real activating run before it can be called confirmed. This
session's own output was written to a scratch `extracted_claude_test7/`
outdir rather than the project's real `extracted_claude/` — that
directory still reflects pre-08-08 output only.

## Not yet built

- SKU/item-name normalization step beyond the brandlist cross-check idea
  above (e.g. reconciling spelling/casing inconsistencies across forms).
- Quantity/unit normalization step.
- Wiring `generate_review.py` into `extract_ollama.py`'s main loop so the
  review page is generated automatically per image, not as a separate step.

These are deliberately deferred until extraction accuracy — including the
known issues above — is solid.

## Ollama Cloud comparison test — 2026-08-11

Built `extract_ollama_cloud.py`, a new, additive file testing a hosted
open-weight vision model (via Ollama Cloud) as a comparison point against
`extract_claude.py`'s Claude API calls -- **not a replacement**;
`extract_claude.py` remains the recommended pipeline. Does not modify
`extract_claude.py` or any other existing file: imports its Pydantic
schemas (`ExtractedForm`, `ExtractedItem`, `QuantityPair`, `QuantityRecount`,
`RowQuantityReading`) and prompt text (`build_system_prompt`,
`RECOUNT_SYSTEM_PROMPT`) verbatim, and reuses its crop/validate/merge
helpers (`_prepare_image`, `_prepare_table_crop`, `_build_row_crops`,
`_validate_row_fracs`, `_apply_recount`, `_to_order_form`,
`_write_debug_artifacts`, `flatten_for_review`) rather than reimplementing
that logic -- only the model-call sites differ (Ollama `chat()` instead of
Anthropic `messages()`), and the per-row recount pass is parallelized via
`ThreadPoolExecutor` the same way. Logs to its own `usage_log_ollama_cloud.csv`
(kept separate from the real pipeline's `usage_log.csv`), with a
`duration_seconds` column timed locally via `time.perf_counter()` around
each call -- real wall-clock timing, a gap CLAUDE.md had flagged as never
having been measured for any pipeline before now.

**Model choice — confirmed by direct testing, not the model named in the
original task.** `qwen3-vl:235b-cloud` is retired: a live call returned
HTTP 410, `"qwen3-vl:235b was retired at 2026-06-16 00:00:00 -0700 PDT"`.
Checked the Cloud API's current model list and tested each plausible
large-model substitute directly with a solid-red-square sanity image
(asked each to name the color) rather than guessing from the model name:
`qwen3.5:397b`, `glm-5.1`, `glm-5.2`, `kimi-k3` all returned HTTP 403
`"this model requires a subscription"` (this account's API key is on the
free tier); `nemotron-3-super` returned HTTP 400 `"this model does not
support image input"`; `minimax-m3` accepted the `images` param but
answered the color-identification test wrong (`"Gray"` for a solid red
square) -- accepted, but not trustworthy. `gemma4:31b` was the only model
both accessible on this account's plan and correct on the same test
(`"Red"`). User confirmed proceeding with `gemma4:31b` as `DEFAULT_MODEL`
(overridable via `--model` if a paid plan unlocks one of the gated
candidates later).

**Structured-output non-compliance — confirmed by direct testing, a real
difference from both extract_claude.py's Claude calls and the local Ollama
pipeline's qwen2.5vl calls.** Passing `format=<json schema>` to Ollama
Cloud's `gemma4:31b` does NOT strictly constrain output the way Anthropic's
structured outputs or local qwen2.5vl's grammar constraint do (confirmed
working for the latter via `extract_ollama.py`'s existing
`format=FormMeta.model_json_schema()` usage). A real call against
`sample 5.jpeg` came back (a) wrapped in ```` ```json ... ``` ```` markdown
fences despite `format=` being set, (b) missing a required field
(`order_no`) entirely, (c) carrying an extra top-level key the schema
doesn't define at all (`"layout"`), and (d) using JSON `null` for a
non-nullable string field (`row_total`) instead of `""`.
`ExtractedForm.model_validate_json()` raised on the raw text (invalid JSON,
from the fences); `model_validate()` on the fence-stripped dict raised 13
separate validation errors. Fixed with two new, generic (not
per-field-hardcoded) helpers in `extract_ollama_cloud.py`:
`_strip_json_fences()` and `_normalize_for_schema()` (walks a Pydantic
model's `model_fields` by type annotation, supplying `""`/`0.0`/`0`/`[]`/a
recursively-normalized nested model for anything missing or null, and
silently dropping unrecognized keys) -- repairs the response *before*
validation runs, so `ExtractedForm`'s own field/model validators (ditto
forward-fill, trailing style-code split, order_date cleanup) still get a
fair shot at running on real values. Confirmed working end-to-end across
all three real runs below.

**Non-determinism at temperature=0 — confirmed by direct testing, a real
difference from local Ollama (`extract_ollama.py`'s qwen2.5vl calls are
deterministic at `temperature=0`) and from Claude.** The first real
`extract_one()` CLI run against `sample 5.jpeg` came back schema-valid but
with **zero quantity cells across all 14 items**
(`prompt=2895 eval=2868, 26.0s, done_reason=stop` -- a clean, non-error
response, just empty). A follow-up isolated test of 4 repeated identical
main calls against the same image (same prompt, same `temperature: 0`)
returned real, populated quantities (68-71 cells) on all 4 -- so the empty
result was a real, reproducible-risk flake (roughly 1 in 5 calls in this
sample), not a code bug: confirmed by inspecting the raw pre-normalization
JSON text from a fresh call at the time, which had genuine populated
`quantities` arrays, ruling out `_normalize_for_schema` as the cause.
Re-ran `sample 5.jpeg` for the results below; every pipeline run should be
treated as one draw from a distribution that occasionally comes back
empty, not a single deterministic result the way the other two pipelines
in this project behave.

**Real per-image results, all three via the full pipeline (main call +
per-row recount + `_apply_recount` merge), checked against this project's
own already-established ground truth, not assumed:**

- **`sample 5.jpeg`** (14 items, 70 size/qty cells; 30.7s main + 31.5s
  recount = 62.4s total; main call `prompt=2895 eval=2960`, recount
  `prompt=21603 eval=2165` across 14 parallel per-row calls) — compared
  cell-by-cell against `extracted/sample 5.json` (the confirmed-correct
  Ollama v6.2 output, with the one known correction from the Claude v1
  section: MYNA/OE's `100` column is `13`, not `3`). **7 of 14 rows exact
  match**: Trend Trunk, Exoda Trunk, Image FCD, Fairlady Print, Bloomer
  Plain, F.G-3005, F.G-3025. **3 rows show a clean, consistent +1
  header-column shift with otherwise-correct values** (Exoda Bloomer
  Plain, Classy W. Vest/RN, Classy W. Vest/RNS -- e.g. Classy W. Vest/RN
  reported `85:6,90:15,95:15,100:5,105:5`, which shifted left by one
  column reproduces the true `80:6,85:15,90:15,95:5,100:5` exactly) — the
  same late-table perspective-drift failure mode already documented
  extensively in this file for both the Ollama OCR pipeline (v6/v6.1) and
  the Claude pipeline (`sample 7.jpeg` rows 6-7, "True per-row crops"
  section above). **1 row (MYNA/OE) combines that same +1 shift with a
  wrong final value** (`105:3` where shift-corrected + true is `100:13` --
  notably the exact same wrong digit, `3` not `13`, the old Ollama pipeline
  had before Claude's v1 correction). **2 rows have a single wrong cell,
  no shift** (Fairlady Plain: `105:50` vs true `10`; MYNA/IE: `85:20` vs
  true `30`). **1 row (Bloomer Print) has a spurious extra cell** (6
  reported pairs including `105:20` where the true row has 5 pairs ending
  `100:20`). Two real wins over the Ollama pipeline's own documented
  history: ditto-name composition worked correctly with no special-casing
  (`"Fairlady Plain"`, `"Bloomer Plain"` -- the Ollama pipeline tried and
  reverted this twice, see v6.1 above) and `F.G-3005`/`F.G-3025` read
  correctly with no OCR cross-check needed (the Ollama pipeline required
  `item_name_ocr` override logic specifically for this). Two real misses:
  `order_date` read as `"30/06/2026"` (true: `"30/03/2026"` -- the month
  digit misread) and `party_name` read as `"M. K. Enterpriss"` (true:
  `"M.K. Enterprises"` -- spelling/spacing errors on both words).

  **Correction, same day, caught by the user spotting it directly in the
  photo (not by re-running anything):** the "ground truth" used for the
  comparison above (`extracted/sample 5.json`, the old Ollama pipeline's
  output) is itself incomplete on `Fairlady Print` and `Fairlady Plain` --
  both rows have a genuine handwritten `SIZE LABEL OVERRIDE` past the
  printed `105` column: the writer wrote `105` and `110` by hand (`110`
  is not a printed header anywhere on this form) with `10` written under
  each, for both rows. Confirmed by cropping and directly viewing that
  region of `Images/sample 5.jpeg`. The old Ollama pipeline's OCR-based
  column assignment only maps digits to *known printed* headers, so it
  had nowhere to put a handwritten `110` and silently dropped it -- this
  was never caught before because CLAUDE.md's v6.2 "every quantity
  matches" claim (see above) was itself checked against this same
  incomplete reading, not the photo directly, for this specific cell.
  Reclassifying: `Fairlady Print` is **not** an exact match for
  gemma4:31b -- it's also missing the `110:10` cell. `Fairlady Plain`'s
  error is worse than first stated: gemma's `105:50` is wrong on *two*
  counts, not one -- the printed `105` column's real value is `10` (not
  `50`), and the handwritten `110:10` override is missing entirely.
  Re-ran `extract_claude.py` fresh on `sample 5.jpeg` (live, current
  prompt, `extracted_claude_recheck/`) specifically to check whether
  Claude's current pipeline -- not the stale 2026-08-06 saved file, which
  turned out to also miss this cell -- catches it: confirmed it does,
  `Fairlady Print` now reads `{'85': 40, '90': 50, '95': 5, '100': 15,
  '105': 10, '110': 10}` and `Fairlady Plain` reads `{'80': 45, '85': 105,
  '90': 130, '95': 30, '100': 50, '105': 10, '110': 10}`, both exactly
  matching the photo. This is a real, confirmed gap between the two
  pipelines on a case neither this file's prior "ground truth" nor
  gemma4:31b's own reading got right: Claude's current prompt handles an
  unlabeled handwritten column with no printed header at all; gemma4:31b
  does not (it doesn't invent a wrong value here, it just omits the cell
  -- a safer failure than a hallucination, but still a miss). Worth
  double-checking whether any *other* row on this form (or the other two
  test images) has a similar unprinted handwritten column that's been
  silently missed by every pipeline's "ground truth" so far -- not done
  in this session.

- **`sample 4.jpeg`** (12 items, 109 size/qty cells; 27.7s main + 30.6s
  recount = 58.5s total) — the tally-mark-dominated form (CLAUDE.md's
  hardest documented quantity-reading case). **Confirmed decisive
  failure**: every single one of the 109 reported quantity values is
  exactly `1`, across every item, every column. Checked against the one
  row with independently-established, catalog-confirmed ground truth from
  the "Column-shift detection" section above -- `B.4749 F/PANT SET`'s real
  values are `55:5,60:8,65:8,70:8,75:8,80:8,85:8,90:5` (8 cells, sizes
  55-90) -- gemma4:31b instead reported 10 cells of all-`1`s spanning
  `45-90` (wrong range AND wrong per-cell values on every cell). Also
  checked via an internal, model-independent consistency signal: for 9 of
  the 12 rows, gemma4:31b's own reported quantities don't even sum to its
  own reported `row_total` (the row's printed Total Dozen) -- e.g.
  `B.3825 SHORT SET` reported `row_total: "8"` but its own 10 quantity
  cells (each `1`) sum to `10`. This reads as the model detecting *that* a
  cell has a tally mark but not attempting to *count* how many strokes are
  in it -- a materially different and more severe failure than anything
  documented for the Claude pipeline on this same form (which, even in its
  hardest/earliest attempts on `sample 4.jpeg`, returned varied real
  values, never a uniform "1 everywhere" pattern).

- **`sample 7.jpeg`** (7 items, 37 size/qty cells; 15.1s main + 19.3s
  recount = 34.5s total) — the perspective-drift case (`sample 7.jpeg`
  rows 6-7 were the form where the Claude pipeline's "true per-row crops"
  fix was proven, see above). Compared against `extracted_claude_test5/sample
  7.json` (Claude's confirmed-correct run, main and recount agreeing on
  every row including the corrected `75-100` range). **5 of 7 rows exact
  quantity match**: Exoda Plain/Plam, Essa Trend, Classy Vest, Coold Vest,
  Image Trunk. **Critically, gemma4:31b independently got the
  historically-hard row right on the first try**: `Fair Lady Plain`
  reported `75:15,80:15,85:35,90:35,95:15,100:15`, an exact match to
  Claude's hard-won corrected reading for that same row (the range this
  entire fix effort was built around, see "True per-row crops" above,
  originally misread as `80-105`). The 7th row (`"Fair Lady Print"` vs
  Claude's `"Fair Lady Round"` -- itself a genuine handwriting ambiguity,
  not resolved here) matches on 5 of 6 cells, differing only on
  `85: 35` (gemma) vs `85: 30` (Claude) -- not independently resolved
  against the photo in this session, flagged here as an open discrepancy
  rather than assumed to favor either pipeline. Item names/style codes
  read differently in three places (`"Exoda Plain"` vs `"Exoda Plam"`,
  `"Image Trunk (OE)"`/type `ICO` vs `"Image Trunk (OB)"`/type `IE`) --
  genuinely ambiguous handwriting neither pipeline can be assumed correct
  on without the photo.

**Recount-pass reliability — confirmed real limitation, distinct from the
structured-output and non-determinism issues above.** Across all three
runs, the per-row recount almost never corroborated or corrected the main
reading -- most rows logged `"alignment looked unreliable, kept the
original full-page quantities for this row"` or `"found no quantities"`.
Traced to a real cause, not assumed: saved and visually inspected one
row's stitched header+row crop directly (`sample 5.jpeg`'s `Trend Trunk`
row) -- confirmed the crop was built correctly from
`_build_row_crops`/`_validate_row_fracs` (both imported unmodified from
`extract_claude.py`), but the underlying `row_top_frac`/`row_bottom_frac`
values gemma4:31b itself reported in the main call don't reliably point at
that row's real content. `_validate_row_fracs` only checks that the
fractions form a plausible, monotonic, non-degenerate partition -- it
passed on every run here -- but a monotonic partition can still be
uniformly offset from the real rows, which is exactly what the recount's
own `item_seen` readings showed: e.g. on `sample 5.jpeg`, the crop
generated for item index 2 (`Image FCD`) consistently showed item index
0's content (`Trend Trunk`) instead, a 2-row offset; on `sample 4.jpeg`
the offset varies row to row rather than following one fixed N, consistent
with imprecise self-localization rather than a fixed indexing bug. This
means `_apply_recount`'s existing alignment safety net (imported
unmodified, and confirmed still doing its job: no misaligned crop's
content ever got merged into the wrong row) protected against corruption
correctly, but also means the recount pass provided little real
corroborating value against gemma4:31b specifically -- a different root
cause than the shared-perspective-drift-bias gap already documented for
the Claude pipeline (where main and recount *do* align but can still share
the same wrong column bias).

**Letter-size / brandlist DB cross-check test, `sample 3.jpeg` -- confirmed
real gap, at the user's direct request.** This is the form with the
documented `MM K4532` row (real product `MM K 4532 B FULL PANT SET`, style
`RNS`, real catalog sizes `{35,40,45,50,55}`, no relation to this form's
own printed 45-90 numeric grid -- see "Prompt hardening + local brandlist
cross-check" above) that Claude's pipeline resolves correctly via the
`LETTER SIZES` prompt rule (report the letter verbatim, e.g. `S`/`M`/`L`,
then `brandlist_match.py` converts it to the real number locally, for
free, once the product match is known). Ran `extract_ollama_cloud.py`
against it (20 items, 234.3s total -- 48.0s main + 186.1s recount across
20 parallel per-row calls, the longest of the four forms tested, and the
sparsest result: only 43 size/qty cells filled across 20 rows, 13 rows
entirely empty). **`MM K4532` specifically: gemma4:31b
never reported a single letter size anywhere on this entire form** --
confirmed by grepping every row's raw `quantities` in
`sample 3.raw.json`: every reported size across all 20 rows is a plain
digit from the form's own printed header row, including `MM K4532`, which
came back `{'50': 6, '55': 6, '60': 6, '65': 6, '70': 6}` -- the same cell
count and the same value (6) per cell as Claude's confirmed-correct
resolved reading, but attached to the wrong axis entirely (this form's own
printed numeric headers) instead of the product's real catalog sizes.
`XUV Half Short`, the other known letter-coded row from this form's
history, came back completely empty (no quantities read at all). Because
gemma never emitted a letter token, `brandlist_match.py`'s letter-size
resolution never had anything to convert -- it's a local, deterministic,
model-independent conversion step, and it works identically regardless of
which VLM produced the input, but only once the input actually contains a
letter. The existing safety net did still do its job on the *symptom*: the
brandlist cross-check (`sample 3.brandlist.json`) correctly found the
unique, code-narrowed catalog match (`score: 85.5, code_narrowed: true,
narrowed_pool_size: 1`) and flagged `sizes_outside_catalog_range: [60, 65,
70]` -- 3 of the 5 reported sizes fall outside the product's real
`{35,...,55}` range -- so this row would surface as suspicious in the
review page rather than silently ship. It also flagged
`style_mismatch: ["RNS"]` against gemma's reported type `"RNBS"` (an extra
inserted letter versus the real style code). **Net: a real, confirmed
capability gap specific to gemma4:31b** -- not a bug in the reused prompt
(the `LETTER SIZES` rule is the same text Claude successfully follows) or
in `brandlist_match.py` (unmodified, and its downstream flagging worked
correctly on the bad input it was given) -- gemma4:31b did not act on the
instruction to recognize and report a letter-coded size on this real test.
One nuance worth being precise about: `MM K4532`'s adopted `{'50':
6,...}` reading came from the *recount* call, not necessarily the main
call -- see the root-cause finding immediately below, which found the
main call returned empty for literally every row on this run. Whether
main would independently have made the same letter-size miss is unknown
from this run alone; only the recount call's behavior on this row is
directly confirmed.

**Root cause of `sample 3.jpeg`'s 13-empty-row result, traced through
`sample 3.json`'s own merge notes (2026-08-11, later same session) --
found on request when asked "why is sample 3 a failure," and it is NOT
the same failure mode as `sample 4.jpeg`'s.** `sample 4.jpeg`'s failure
was a genuine counting error (every cell read as `1`, confirmed wrong
against catalog ground truth). `sample 3.jpeg`'s is different: **every
single one of this run's 20 rows shows `main=-` (empty) in
`_apply_recount`'s disagreement notes** -- meaning the main call's own
`quantities` were empty for the entire form on this draw, the same
non-determinism already documented above (the ~1-in-5-calls empty-response
flake first seen on `sample 5.jpeg`), just landing on every row of one
form in a single draw rather than one row. The recount pass (also
gemma4:31b, but a separate, focused per-row call) then did real,
verifiable work: **7 of 20 rows were recovered with confirmed-correct
values**, resolved via an exact match against each row's own printed
Total Dozen (`B-509`, `Super Boxer`, `Elf force`, `MM K4532`, `MM k 3674`,
`MM Looper 4289`, `Nivi Brick`) -- an independent, trustworthy signal that
doesn't depend on main being non-empty. But `_apply_recount`'s existing
disagreement tiebreak (imported unmodified from `extract_claude.py`, see
that function's own docstring) defaults to *keeping main* when neither
Total Dozen nor the catalog resolves a disagreement -- a reasonable
default when main is normally the more reliable read (true for Claude,
which this logic was designed and tuned against), but actively harmful
here: it discarded real, plausible recount data on rows where nothing
could independently confirm it. Confirmed concretely on `XUV Half Short`:
the recount call correctly read letter sizes this time
(`L:13, M:11, S:9, XL:13, XXL:13`) -- direct evidence gemma4:31b *can*
follow the `LETTER SIZES` rule -- but this reading was thrown away in
favor of main's empty result, because no Total Dozen or catalog match
existed to break the tie in recount's favor. The same pattern explains
`Looper x 2 4232`, `C41131`, and `Tech Tron` (recount had real values,
discarded for the same reason). A further 5 rows (`B Gentle`, `B-511`,
`Looper`, `B5102`, `Salma Perry`) failed for the already-documented,
separate reason (recount row-crop misalignment from gemma4:31b's
imprecise self-reported row boundaries -- see "Recount-pass reliability"
above). Two rows (`Loop Square`, `Feather Box`) are a genuine unexplained
gap: main and recount *independently agreed* on empty, despite printed
Total Dozen values of `30` and `37` -- not accounted for by anything
above. **Net assessment: this form's poor result is mostly attributable
to one especially unlucky empty-main-call draw compounding with a
tiebreak default that assumes main is trustworthy -- an assumption this
whole comparison test has shown doesn't hold for gemma4:31b -- rather
than a form-specific reading failure the way `sample 4.jpeg`'s was.** A
retry (a fresh call, hoping for a non-empty main draw) was not attempted
in this session; whether this specific form fares much better on a
second draw is unconfirmed.

**Timing summary (real numbers from `usage_log_ollama_cloud.csv`, all
four runs, single image at a time, no batching):**

| image | main call | recount (per-row, parallel) | total |
|---|---|---|---|
| sample 5.jpeg (14 rows) | 30.7s | 31.5s (14 calls, 6 workers) | 62.4s |
| sample 4.jpeg (12 rows) | 27.7s | 30.6s (12 calls, 6 workers) | 58.5s |
| sample 7.jpeg (7 rows) | 15.1s | 19.3s (7 calls, 6 workers) | 34.5s |
| sample 3.jpeg (20 rows) | 48.0s | 186.1s (20 calls, 6 workers) | 234.3s |

`sample 3.jpeg`'s recount pass took roughly 6x `sample 7.jpeg`'s despite
under 3x the row count -- with a fixed 6-worker pool, 20 rows means a
third batch of calls queues behind the first two (vs one batch for
`sample 7.jpeg`'s 7 rows), so wall time doesn't scale linearly with row
count once the row count exceeds `RECOUNT_MAX_WORKERS`.

No cost figure is reported -- Ollama Cloud's free-tier usage isn't
metered per-token the way Claude's API is, so `estimated_cost_usd` is left
blank in the log rather than guessed. No free-tier session-limit error was
encountered during this test (16 calls total across the four runs, well
under whatever the free tier's cap is); if one is hit in a future run, the
exact error text should be captured here per the original task's request,
but nothing to report yet.

**Not yet done:**
- No batch-mode equivalent -- `extract_ollama_cloud.py` only has a
  live-call loop (matching `extract_claude.py`'s `--live` path), no
  attempt was made to find/use an Ollama Cloud batch API equivalent.
- Only tested against `gemma4:31b`. If a paid Ollama plan is added later,
  the gated candidates (`qwen3.5:397b`, `glm-5.1`, `glm-5.2`, `kimi-k3`)
  are worth the same three-image comparison, especially given
  `sample 4.jpeg`'s tally-mark failure -- unknown whether a larger/newer
  model does better on that specific failure mode or shares it.
  `qwen3.5:397b` in particular, as a lineage successor to the local
  pipeline's `qwen2.5vl`, would be the most direct point of comparison.
- The single open cell-level discrepancy on `sample 7.jpeg`'s 7th row
  (`85: 35` vs Claude's `85: 30`) wasn't resolved against the source photo.
- The non-determinism rate (~1-in-5 observed in one small sample of 5
  calls) isn't precisely characterized -- worth revisiting if this pipeline
  is used more than as a one-off comparison.
- No run yet on any image beyond these four (three matching
  `extract_claude.py`'s original validation set, plus `sample 3.jpeg` for
  the letter-size test above) -- in particular, nothing tried yet on the
  free-form/non-ESSA-template forms noted in the "Multi-form-type survey"
  section above.
- `sample 3.jpeg`'s dismal fill rate (43 cells across 20 rows, 13 rows
  entirely empty) has a confirmed root cause now (see "Root cause of
  sample 3.jpeg's 13-empty-row result" above -- an empty-main-call draw
  compounding with the disagreement tiebreak's main-preferring default),
  but the FIX for it (e.g. having `_apply_recount`'s tiebreak prefer a
  non-empty recount reading over an empty main reading even without a
  Total Dozen/catalog match) hasn't been built -- would require either
  changing shared logic in `extract_claude.py` (out of scope for this
  additive-only file, and risky to tune against Claude's very different
  reliability profile without breaking that pipeline) or a local override
  in `extract_ollama_cloud.py` that post-processes `_apply_recount`'s
  output. Not attempted this session. A second, fresh run of
  `sample 3.jpeg` (hoping for a non-empty main draw) also wasn't tried --
  unknown how much of this result is specific to this one unlucky draw
  versus a more persistent problem with this particular form.
- The `"type"` field values on `sample 3.jpeg` (`"full Pant"`,
  `"3/4 set"`, `"RNBS"`, etc.) weren't checked against the photo -- unclear
  whether these are genuinely what's printed in this form's Style/type
  column or another gemma4:31b misread, beyond the one confirmed
  `style_mismatch` (`"RNBS"` vs the catalog's real `"RNS"` for `MM K4532`).
- Whether any *other* row across all four test images has a similar
  unprinted handwritten column (like `sample 5.jpeg`'s `Fairlady
  Print`/`Fairlady Plain` `110` override, found only because the user
  spotted it directly in the photo) hasn't been systematically checked --
  the discovery process there was manual, not a repeatable check.

## qwen3.5:397b access + think=False speed fix — 2026-08-13

User subscribed to Ollama Pro. `qwen3.5:397b` -- one of the four models
that returned HTTP 403 "requires a subscription" during the 2026-08-11
comparison test above -- is now listed by `client.list()` and reachable.
Passed the same solid-red-square sanity check used to vet every other
candidate that day.

**First full-pipeline run (default settings: thinking on, recount on)
against `sample 5.jpeg` -- confirmed the best single-model accuracy of any
Ollama Cloud model tried, but confirmed unusably slow.** 721.8s (12 min)
total: 254.2s/13760 output tokens for the main call, then 467.4s for a
recount pass that **failed on all 14 of 14 rows** (a mix of
`done_reason=length` truncation and invalid/empty JSON) -- zero
corroboration for the cost. Checked the main call's own reading
cell-by-cell against the established `sample 5.jpeg` ground truth
(`extracted/sample 5.json`, with the two known corrections already
documented above -- `MYNA`/OE's `100:13`, and the handwritten `110`
override on both Fairlady rows every pipeline but Claude's rechecked run
misses): **11 of 14 rows exact** (modulo the shared `110` gap, which this
run also missed, same as every non-Claude pipeline so far), one row
(`MYNA`/IE) missing two cells entirely, one row (`Image FCD`) with a
spurious extra cell (`105:105`) absent from ground truth, and two rows
(`F.G-3005`/`F.G-3025`) with cosmetic item-name misreads (a stray space;
`F.G1-3025`, the exact same misread pattern the original local VLM
pipeline had before its OCR cross-check override).

**User's real constraint, stated directly: needs a result in under
2-2.5 minutes, not 12.** Root-caused the slowness rather than assuming
model size was the ceiling: `qwen3.5:397b` runs extended internal
"thinking" by default even at `temperature=0` (`ollama.Client.chat()`
exposes a `think` bool the earlier 2026-08-11 test never set). Confirmed
by direct testing:
- A trivial single-question call (no schema, no image complexity) dropped
  from thinking-mode's multi-minute territory to **11s** with `think=False`.
- The real schema-constrained main call dropped from 254.2s/13760 output
  tokens to **~40-56s / ~1850-2300 output tokens** across six separate
  trials on `sample 5.jpeg`, with accuracy holding or improving -- one
  `think=False` run scored **12 of 14 rows exact**, and the `Image FCD`
  spurious-cell issue from the thinking-mode run didn't recur.
- The recount pass's 0%-success rate did not improve with `think=False`
  either (the one variant tested, the whole-table fallback, still failed
  with invalid JSON) -- confirmed recount has provided **zero observed
  corroboration value for this model under any configuration tested**, only
  added 30-90s+ of latency. `--no-recount` CLI flag added to
  `extract_ollama_cloud.py` so it can be skipped outright; existing
  behavior (recount on) is left as the default since other models haven't
  shown this same 0%-success pattern.

**A real non-determinism flake, confirmed in two different shapes across
the six `think=False` trials (~1-in-5, roughly matching the
already-documented gemma4:31b flake rate from 2026-08-11 -- likely a
general Ollama Cloud characteristic, not something `think=False`
specifically caused, though the sample size here is small):**
(a) `quantities: []` entirely for every item, and (b) a **new** shape not
seen before -- correct item names and correct size-header keys, but every
single quantity value literally `0` (e.g. `{"size": "80", "quantity": 0}`
for every cell on the form). Shape (b) matters because a naive
"is the quantities list empty" check misses it entirely -- confirmed the
hard way: the first version of the retry check below (`len(quantities) ==
0`) shipped, was tested, and silently let a real all-zero-value response
straight through to the final output as `"quantities": {}` for every item,
with no error and no retry attempted, until the raw model output was
inspected directly and the actual per-pair values (not just pair count)
were checked.

**Fixes implemented in `extract_ollama_cloud.py`, all confirmed via real
runs:**
- `THINK = False` module constant, threaded into `_call_schema()`'s
  `client.chat()` call as `think=THINK`. Comment on the constant records
  the before/after numbers above so a future change to this default isn't
  made blind.
- `--no-recount` CLI flag, wired through `main()` -> `extract_one()`.
- Retry-on-flake in `extract_one()` (`MAIN_CALL_MAX_RETRIES = 1`): checks
  the actual **summed quantity values** across the whole response (not
  list length, per the shape-(b) bug above), retries the main call once if
  every value came back `0`. Confirmed working end-to-end: one real run hit
  the flake on attempt 1 (56.5s, all-zero), retried automatically, and
  succeeded on attempt 2 (46.8s, 70 cells) -- **103.3s total, still
  comfortably inside the user's 2.5-minute budget even in this
  worst-observed case.**

**Confirmed final numbers, `think=False` + `--no-recount`, `sample
5.jpeg`:**

| scenario | time | quality |
|---|---|---|
| typical (no flake) | ~40-56s | 12-13/14 rows exact |
| flake on attempt 1, recovered by retry | 103.3s | recovers to full quality |
| default settings (thinking on, recount on) | 721.8s | 11/14 rows exact, recount 0% functional |

A clean post-fix run scored **13 of 14 rows exact**: one cosmetic
item-name misread (`"Bloomer Print"` -> `"Bloomer Paint"`, quantities
unaffected), and one specific cell that came back wrong **identically
across at least two separate runs** (`MYNA`/IE size `85`: read as `20`
both times, true value `30`, per the established ground truth) -- flagged
as possibly genuinely ambiguous handwriting on this specific digit rather
than random per-call noise, since it didn't vary between runs the way the
flake above does.

**Not yet done:**
- Only tested against `sample 5.jpeg` -- `sample 4`/`sample 7`/`sample 3`
  (the harder forms from the 2026-08-11 comparison) haven't been re-run
  under the `think=False` + `--no-recount` configuration.
- No second retry attempt -- a double-flake (~4% chance at the observed
  ~1-in-5 single-flake rate, not itself confirmed by a large sample) would
  exceed the 2.5-minute budget uncaught. Deliberately left at one retry
  rather than guessing a second retry is worth the added worst-case
  latency; revisit if a double-flake is actually observed.
- Recount remains fully non-functional for this model under every
  configuration tried (thinking on/off, per-row/whole-table) -- would need
  its own investigation (e.g. `think=False` specifically on the per-row
  path, never tried before recount was disabled outright; or a larger
  `RECOUNT_ROW_MAX_TOKENS`) before it could add real corroboration value
  here the way it does for Claude.
- `THINK = False` is a single hardcoded module constant applied to every
  model this file calls, not scoped per-model -- fine while `qwen3.5:397b`
  is the only model characterized this deeply, but `gemma4:31b`'s own
  thinking behavior (if it has one) hasn't been tested under this same
  toggle.
- The scratch output directories from this session's testing
  (`extracted_ollama_cloud_qwen*/`, `usage_log_ollama_cloud_qwen*.csv`)
  were intended to be removed after this write-up, per the user's request
  to keep only what's necessary for the project -- this section is the
  persisted record of those runs' findings, not the raw JSON.
  **Correction (2026-08-14): that removal never actually happened in this
  session** -- the qwen scratch dirs/logs were still on disk, alongside
  equivalent untracked scratch output from every other model tested this
  day (gemma, kimi, mistral x9 variants). Actually deleted in the cleanup
  pass below. `extracted_ollama_cloud/` (the default outdir,
  `gemma4:31b`-only) is unaffected.

## qwen3.5:397b on the harder forms — same day, later: verdict reversed

Followed up on the "not yet done" item directly above: ran `sample
4.jpeg` (tally marks -- gemma4:31b's worst documented failure, every cell
read as `1`) and `sample 3.jpeg` (letter sizes -- gemma4:31b's other
worst failure, never once emitted a letter size) through the
`think=False` + `--no-recount` config that looked strong on `sample
5.jpeg`. **Result: qwen3.5:397b is not a reliable upgrade over
gemma4:31b -- it wins clearly on `sample 5.jpeg` specifically, but fails
in new, form-specific ways on both of the harder forms, and on `sample
4.jpeg` the failure is arguably worse than gemma4:31b's.**

**`sample 4.jpeg`: complete failure under both configurations tested.**
- `think=False`: **4 consecutive attempts across two separate script
  invocations (each retrying once internally) all came back with real
  item names/style codes but every single quantity value literally `0`**
  -- the exact flake documented above, but happening every time on this
  specific image rather than the ~1-in-5 rate seen on `sample 5.jpeg`.
  95.2s and 103.7s wall time for the two invocations, both producing a
  fully empty, useless result despite the retry logic working exactly as
  designed.
  - Compare to gemma4:31b's own documented failure on this same image
    (see the 2026-08-11 section above): gemma at least returned
    schema-valid, wrong data (every cell read as `1`) -- something a
    human reviewer or a checksum could visibly catch as implausible.
    qwen's all-zero response is arguably a *worse* failure mode for a
    production pipeline: it's the same shape as a legitimately empty row,
    so nothing downstream flags it as suspicious.
- `think=True` (diagnostic, to check whether disabling thinking was
  *causing* the failure specifically on this harder image): **the call
  never returned.** Force-stopped after roughly 1.5 hours with zero
  output written -- not a slow success, a genuine hang, confirmed by the
  complete absence of the script's own output file that only gets written
  after a successful response. This is qualitatively different from
  `sample 5.jpeg`'s worst documented case (721.8s, slow but completed) --
  worth knowing that thinking mode isn't just "slow" on a hard-enough
  image, it can fail to terminate at all.
- Net: `sample 4.jpeg` looks like a genuine blind spot for this model
  independent of the thinking setting, not something the speed fix
  caused or could fix.

**`sample 3.jpeg`: technically completed (138.7s after one retry, right
at the edge of the 2.5-minute budget for this denser 20-item form), but
with a severe, different data-quality regression, confirmed in the raw
model output before any Pydantic validator ran:**
- **Every one of the 20 item names came back as an empty string** --
  not a post-processing bug (checked `sample 3.raw.json` directly).
  Style/type codes survived for most rows (`RN`, `RNS`, `RNBS`), but item
  names did not, for any row, at all.
- **`row_total` also came back blank on every row** -- this breaks the
  Total-Dozen-checksum verification method this project has relied on
  throughout (see the 2026-08-06/07 sections above) for exactly this kind
  of form, since there's nothing to sum against.
- **Two rows are byte-identical duplicates of each other** (rows 5 and 6,
  both `45:2,50:5,55:5,60:5,65:5,70:5,75:5,80:3`) -- plausibly the same
  row read twice rather than two distinct rows, though with no item names
  this can't be confirmed against the photo without manual cross-referencing.
- **The one row that plausibly is the known `MM K4532` letter-size test
  case came back on a mixed/wrong axis**: `45:6, 50:6, 55:6` read as
  plain printed-header numbers, with only `XL:6, XXL:6` read as letters --
  compare to Claude's confirmed-correct `{"S":6, "M":6, "L":6, "XL":6,
  "XXL":6}` (all five as letters, resolved locally by
  `brandlist_match.py` to the real catalog sizes). Worse, since the item
  name is blank, this row can't even be confirmed as `MM K4532` with
  confidence -- the identification is circumstantial (matching cell
  values only), not verified.
- Net: this is a different failure mode from `sample 4.jpeg`'s (a
  completed, schema-valid response, not a flake or hang), but equally
  disqualifying for real use -- no item names means the brandlist
  cross-check, letter-size resolution, and column-shift detection this
  project built specifically for cases like this (see the 2026-08-06
  sections above) have nothing to key off of.

**Revised bottom line, superseding the "clear upgrade" framing from
earlier in this session:** qwen3.5:397b's `sample 5.jpeg` result was
real and reproducible, not a fluke -- but it does not generalize to the
two forms this project already knows are hard. The honest comparison is
now: gemma4:31b fails predictably and visibly (implausible-looking wrong
data, or a wrong axis) on its hard cases; qwen3.5:397b fails
unpredictably and less visibly (empty-looking data, hangs, or silently
missing the one field -- item name -- everything else depends on) on
its own, different hard cases. Neither model has been shown reliable
across the three most demanding forms in this project's own test set.
This materially changes the "is Ollama Pro worth it" answer from earlier
in this session: not proven for this pipeline's harder forms, only for
its easiest one.

## Full Ollama Cloud model survey + mistral-large-3:675b discovered — same day, later

Checked every model on the account's Ollama Pro plan (`client.list()`, 18
models) rather than assuming only qwen3.5:397b was worth trying. Ran the
same solid-red-square vision sanity check against every untested one, with
a hard 90s per-model timeout after the earlier qwen3.5:397b hang:

- **10 of 18 don't support image input at all**, confirmed by HTTP 400
  "this model does not support image input": `glm-5.1`, `glm-5.2`,
  `deepseek-v4-flash:preview`, `deepseek-v4-flash:0731`, `deepseek-v4-pro`,
  `nemotron-3-nano:30b`, `nemotron-3-ultra`, `nemotron-3-super` (already
  known), `gpt-oss:20b`, `gpt-oss:120b` -- disqualified outright, no point
  testing further for this task.
- **`kimi-k3` needs extra paid usage** beyond the Pro plan ("extra usage
  balance is empty") -- not tested further.
- **3 passed the vision check**: `mistral-large-3:675b`, `kimi-k2.6`,
  `kimi-k2.7-code`.

**`kimi-k2.6` tested and confirmed worse than qwen3.5:397b on every
axis.** `think=False` (this file's default) produced a deterministic
all-zero read on `sample 5.jpeg` -- 4/4 attempts, byte-identical
`eval_count` across separate calls, not a random flake. `think=True`
(the only setting that ever produced real data) succeeded only 1 of 3
times, taking 118-260s per attempt even when it worked. Net: slower and
less reliable than qwen3.5:397b in every dimension tested, with no
observed advantage anywhere. Not pursued further.

**`mistral-large-3:675b` tested and confirmed the strongest Ollama Cloud
model found this session, across every form tried:**
- `sample 5.jpeg`: fast (~40-55s), reliable (5/5 first-attempt successes
  across every trial run this session, zero flakes observed for this
  model specifically), and -- once the actual evaluation criterion was
  corrected mid-session to "did it capture the real quantity VALUES,
  since column-label offset can be fixed by hand" (the user's own explicit
  framing) -- **95.7% of true quantity values captured (66/69), 8 of 14
  rows with zero errors at all.** Its real, confirmed weakness is a
  column-header-label shift that grows with row depth (0 header-positions
  off on row 1, up to 7 by row 11-14) -- the raw digits are read right,
  they just get attached to the wrong size column on later rows.
- `sample 2.jpeg` (free-form layout, no shared header row -- see the
  "Multi-form-type survey" section above): **the only model of three
  tested that extracted any real quantity data at all.** Both gemma4:31b
  and qwen3.5:397b returned zero quantities across two attempts each on
  this same image; mistral got 47 cells across 12 of 14 rows on the first
  attempt, including correctly identifying an out-of-context handwritten
  quote on the page as a note rather than folding it into item data (both
  other models also read this quote accurately, in an unrelated
  confirmation that item-name/note-level reading is solid across models
  here -- only quantity extraction diverges this sharply).
- `sample 4.jpeg` (tally marks): **fails the same way every model tested
  this session fails** -- checked against the one row with independent
  catalog-confirmed ground truth (`B.4749`, real values
  `55:5,60:8,65:8,70:8,75:8,80:8,85:8,90:5`): mistral read it as
  `45:1,50:1,55:1,60:1,65:1,70:1` -- wrong header range AND every real
  multi-stroke count flattened to 1. Confirmed this is a shared,
  model-agnostic ceiling (see below), not something a different model
  choice fixes.

**Column-shift drift-calibration, investigated and found not currently
buildable safely.** Given the shift correlates with row depth (though not
perfectly linearly -- a plain linear fit only predicted 5/14 rows
correctly; an isotonic/step fit predicted 13/14 correctly when given
every row's true shift as training data), the natural next step was
porting the Ollama pipeline's v6.1 self-calibration idea: use
catalog-confirmed "anchor" rows to fit the trend blind, without ground
truth. This did not pan out on real data: the existing `_is_trustworthy()`
gate (built for auto-applying letter-size/type corrections, a
higher-stakes action) left only 4 of 14 rows usable as anchors, all
clustered in the shallow half of the table with zero coverage where the
shift is largest. Widening the anchor-selection gate to
`resolve_shift_offset`'s own "is there a unique fitting offset" logic
(safer in principle, since it requires uniqueness) still produced a
confirmed WRONG anchor on `Fairlady Plain` (resolved offset 1, true
offset 2) -- wide, non-code-narrowed catalog ranges can make a wrong
offset look uniquely valid by coincidence, the same false-positive risk
already documented for the `B 5109` case in the 2026-08-06 section above.
**Decision: did not ship this.** A wrong auto-correction is worse than
mistral's raw output, which at least fails in a way a human glancing at
the review page's `.db-note` flags might catch.

## mistral-large-3:675b prompt tuning — same day, later

At the user's request ("adapt and improvise... make the model understand
the needs clearly"), built `MISTRAL_PROMPT_ADDENDUM_BASE` in
`extract_ollama_cloud.py` -- extra system-prompt text appended only when
`"mistral" in model.lower()`, so it never touches the prompt any other
model or `extract_claude.py` itself uses. Five distinct fixes were
attempted, each verified (or rejected) against real re-runs, not assumed
from the wording alone:

**Confirmed working, kept:**
- **Item name vs style code splitting** -- mistral was splitting
  "BLOOMER PLAIN" into item="BLOOMER" type="PLAIN IE" instead of
  item="BLOOMER PLAIN" type="IE". Confirmed fixed on a real re-run of
  `sample 2.jpeg`: also recovered a previously-empty row
  ("CYCLING SHORTS"/"IE ADULTS" -> "CYCLING SHORTS ADULTS"/"IE", which
  went from 0 quantity cells to 6 real ones).
- **Row completeness near the bottom of the page** -- confirmed partial
  improvement on the same `sample 2.jpeg` re-run (1 of 2 previously-empty
  trailing rows recovered, not both).

**Confirmed NOT working after real attempts, deliberately dropped rather
than left in "just in case" (they cost prompt tokens on every call for
zero measured benefit):**
- **Column alignment** ("re-derive header position fresh per row"):
  produced a wash on `sample 5.jpeg` -- some rows' shift improved,
  others got WORSE, net value-accuracy unchanged (66/69 before and
  after).
- **Quantity-value undercounting** (the `sample 4.jpeg` tally-mark
  issue): tried in two different framings -- first as "count individual
  tally strokes" (this framing was WRONG per the user's own domain
  knowledge: these forms don't necessarily use countable tally strokes
  at all, whatever mark type is present the model is just undercounting
  it), then reworded as a generic "don't default to a low placeholder."
  **Both produced byte-identical output** on the same ground-truth row
  (`B.4749`, still all `1`s). Confirmed a real model ceiling, not a
  wording problem.
- **Sizes past the last printed header** (the confirmed `110`-column miss
  on `sample 5.jpeg`'s Fairlady rows) and **letter-coded rows instead of
  the printed grid** (the confirmed `MM K4532` miss on `sample 3.jpeg`,
  same failure qwen3.5:397b had on this exact row): tried THREE
  distinct approaches --
  1. a prose-only mention ("sizes sometimes go up to 110, 115, or 120"),
  2. the same claim grounded in a real DB query
     (`brandlist_match.known_numeric_sizes()`, new function added
     specifically for this -- injects the business's actual complete
     numeric size vocabulary, `[25, 30, ..., 120]`, into the prompt
     instead of an abstract claim),
  3. a unified "the printed grid is a generic template, not a hard
     boundary" framing naming both failure modes as the same underlying
     human behavior (writers working around a template that doesn't fit
     a specific product), plus a related fix for item names with
     embedded alphanumeric codes (`"K4532"` losing its `K`).

  **All three produced identical failures** on both target rows, checked
  via real re-runs each time -- `Fairlady Print`/`Fairlady Plain` never
  once reported a `110` size across any attempt; `MM K4532` never once
  reported a letter size across any attempt (though attempt 3 did fix
  the `K`-dropping issue on the item name itself, confirmed via
  `brandlist_match.py`'s `code_not_in_catalog` field disappearing).

**Recount tested for mistral for the first time this session (every
earlier mistral test ran with `--no-recount`, inherited from qwen/kimi's
documented failures without ever giving mistral its own shot).**
Confirmed equally broken: 12 of 14 per-row recount calls on
`sample 5.jpeg` failed outright (`invalid JSON from model`), the other 2
hit misaligned row-crops (one call literally read `MYNA`'s content while
labeled `Fairlady Print`). Zero real corroboration, same structured-output
compliance failure already documented for qwen/kimi's recount calls, not
specific to the two target rows.

**Conclusion, stated directly to the user after four different genuinely
distinct approaches (3 prompt framings + porting Claude's own recount
mechanism) all failed identically on both target rows: this is not a
prompt-engineering gap.** Claude solves both cases using the exact same
single-call architecture and the exact same underlying prompt rules
mistral already has -- its advantage here is intrinsic visual attention,
not a technique available to port over. Stopped iterating on these two
specific cases via prompting; the addendum now ships with only the two
confirmed-working fixes plus the confirmed-broken-but-harmless size-list
injection (kept as-is at the point this was written; not yet trimmed).

## olmOCR2 (local, community GGUF) tested as a structurally different approach — same day, later

Given mistral's two remaining failures were confirmed to be genuine
visual-attention limits rather than prompt-fixable, tried a category
change instead of another model swap: a document-OCR-*specialized* model
instead of a general chat VLM, on the theory (backed by this project's
own history -- the local pipeline's v6 switch from a general VLM to
PaddleOCR + DP column assignment, which fully solved an analogous
column-drift problem) that a model trained for faithful, exhaustive
transcription wouldn't have the "force everything into the expected
template" bias every general VLM tested this session has shown.

**Model**: `richardyoung/olmocr2:7b-q8` -- a community GGUF build (not an
official Ollama Cloud model; confirmed via `client.list()` it isn't
hosted there, and via web search it's only available to `ollama pull`
locally) of Ai2's real `olmOCR-2-7B-1025`, itself a Qwen2.5-VL-7B-Instruct
fine-tune trained on the olmOCR dataset with RL refinement specifically
for hard cases like tables. Runs through the LOCAL Ollama install (same
mechanism as `extract_ollama.py`'s existing `qwen2.5vl:7b`), not the
Cloud API `extract_ollama_cloud.py` is built around -- a different
resource tradeoff (local compute/time, not Pro quota). Pulled successfully
(9.5GB, `ollama pull richardyoung/olmocr2:7b-q8` -- note the bare
`richardyoung/olmocr2` tag does NOT resolve, confirmed by a real failed
pull attempt first).

**Real, confirmed limitation: very slow locally** -- every full-page call
this session took 7-9 minutes (419-545s), regardless of prompt wording.
This is a genuinely different cost axis from every Cloud model tested
(seconds to ~2 min on Ollama Pro quota vs minutes of local wall-clock
time), not yet weighed against the accuracy gains below.

**First finding, confirmed by testing not assumed: output was silently
truncated by the default context window, not the output-length cap.**
Raising `num_predict` alone (32000 -> 4096, oddly a *decrease* here since
the default was already high) did not fix a response cutting off at the
same point twice in a row (`done_reason=length`, ~1250-1280 tokens both
times) -- raising `num_ctx` to 16384 alongside it did (`done_reason=stop`,
1953 tokens, complete 14-row table). The image itself consumes a large
share of a vision model's default context before any output token is
generated; Ollama's default `num_ctx` was too small for this model on a
full-page image. Worth remembering for any future local-Ollama vision
model test in this project.

**`sample 5.jpeg`, checked precisely (a manual first pass mis-counted
table cells by eye and was wrong twice -- redone with an actual HTML
parser both times before trusting any number here): 11 of 14 rows exact
match against established ground truth**, once a parsing bug was found
and fixed -- the raw HTML table's column position doesn't align 1:1 with
the header row (the item-name cell absorbs the Style column without
actually removing a slot), producing a CONSTANT, position-independent
cell offset, confirmed different between two separate calls on the same
image (+1 shift in the first truncated run, +3 in the complete run) --
not the row-depth-growing drift mistral has. Fixed by auto-calibrating
the shift per-run against `Trend Trunk`'s already-known-correct values
rather than hardcoding a constant (a hardcoded shift from the first run
would have been silently wrong on the second). This is a genuinely easier
class of error to correct than mistral's drift, precisely because it's
constant rather than growing -- one calibration point fixes the whole
table.
- Of the 3 remaining wrong rows: 2 are the exact same specific hard
  cells every model this session has struggled with (`MYNA`/IE's `85`
  misread as `20` not `30`; `MYNA`/OE's `100` misread as `3`, matching
  the *original local pipeline's* pre-Claude-correction error, not a
  new mistake). The 3rd (`Fairlady Print`'s `105` cell) is a genuinely
  new, olmOCR2-specific miss (`108` instead of `10`) -- but notably, raw
  signal for the `110`-column override (which every general VLM tested
  this session missed completely, on every attempt) DID appear in the
  transcription, landing in the `Total Dozen` table slot instead of a
  new column. Not yet correctly parsed out, but present -- a
  structurally different, more promising failure mode than "never looks
  there at all."
- Review page generated and verified: `extracted_olmocr2/sample 5.review.html`.

**`sample 3.jpeg`, tested with an explicit prompt addition asking for
extra-column and letter-size handling -- did NOT transfer, and overall
quality was notably worse than `sample 5.jpeg`, not just neutral on the
targeted issues:**
- **Letter-size instruction had zero effect**, confirmed directly: `MM
  K4532` still returned 7 plain numeric values (`6,6,6,6,6,6,6`) instead
  of letter-coded sizes, identical in kind to every other model's failure
  on this row. Specialized OCR models appear to be less prompt-steerable
  than general chat models, consistent with them being trained hard for
  one transcription behavior -- confirmed by this one data point, not
  yet a broad conclusion.
- **Item name preserved correctly this time** (`"MM K4532"`, not
  `"MM 64532"` -- the `K`-dropping bug mistral had did not reproduce
  here), a genuine plus, though not yet confirmed as a repeatable
  improvement vs one good run.
- **17 of 20 rows failed this form's own Total Dozen checksum**
  (previously confirmed a real, trustworthy per-row signal for this
  specific form -- Claude matched it on 18/20 rows in the original
  2026-08-05 test) -- checked precisely with the same validated HTML
  parser, not the first-pass regex (which agreed with the parser here,
  for once, but was cross-checked anyway given the parser's track record
  of catching real bugs the regex missed on `sample 5.jpeg`). Spot-checked
  one mismatched row (`B-4749 Full Pant Set`, real values
  `55:5,60:8,65:8,70:8,75:8,80:8,85:8,90:5`) against the mismatch: olmOCR2
  got the first value and the repeated middle value right but ran the
  repeated `8` two positions too long and dropped the trailing `5` --
  an over-counting error, a different failure shape than `sample 5`'s
  clean alignment.
- **A genuinely useful, real, independent signal found despite the wrong
  read**: `MM K4532`'s own transcribed Total Dozen (`30`) exactly matches
  `5 letters x 6 each = 30`, the true answer -- even though the 7-value
  numeric breakdown above it is wrong, the checksum alone flags the row
  as suspicious automatically, without needing to solve why the letters
  weren't read. This is a real, actionable finding independent of whether
  olmOCR2's raw quantity-reading gets fixed on this form.

**Not yet done:**
- The `Fairlady Print`/`110`-column raw signal (landing in Total Dozen
  instead of a new column) hasn't been parsed out into a correct value --
  worth a targeted fix given the signal is confirmed present, unlike
  every general VLM's total miss on this exact case.
- `sample 4.jpeg` (tally marks) not yet tested with olmOCR2 at all.
- Whether `sample 3.jpeg`'s worse result is form-specific (denser, 20
  rows vs 14) or a real regression from the added prompt instructions
  hasn't been isolated -- the two changed at the same time (different
  image AND different prompt), so this hasn't been cleanly separated yet.
- No repeat runs yet to check olmOCR2's own consistency across identical
  calls (mistral, qwen, and kimi all showed real run-to-run variance this
  session; olmOCR2's has only been sampled once per image so far).
- 7-9 minute local runtime per full-page call is a real, unresolved cost
  if this were ever wired into a real pipeline rather than used for
  one-off comparison testing.

## Repo cleanup — 2026-08-14

Removed every one-off scratch artifact from the Ollama Cloud model
comparison testing (2026-08-11 to 2026-08-13) whose findings are already
fully written up in prose above, per this project's established pattern
(see the "scratch output directories" note in the qwen3.5:397b section
above -- this pass is what that note originally claimed had already
happened):

- 12 `extracted_ollama_cloud_*_test*/` / `*_final/` / `*_tuned*/` /
  `*_v2/` / `*_v3/` / `*_recount/` comparison-run folders (gemma, kimi,
  mistral x9 variants, qwen x2) and their matching
  `usage_log_ollama_cloud_*.csv` siblings.
- Loose `scratch_olmocr2_*.txt` / `scratch_parse_olmocr2.py` debug dumps
  from the olmOCR2 session, left in the project root instead of the
  scratchpad temp dir.
- `__pycache__`.

**Kept, deliberately:** `extracted_olmocr2/` (that investigation still
has open TODOs per the section above -- keep the artifact until it's
closed out), `extracted/` (Ollama local v6.2 pipeline, confirmed-good
`sample 5` baseline), `extracted_claude/` (Claude pipeline output),
`extracted_ollama_cloud/` (the real default outdir for `gemma4:31b`, not
a scratch variant), and the unsuffixed `usage_log.csv` /
`usage_log_ollama_cloud.csv`.

## Correction: `sample 4.jpeg`'s ground truth was wrong — 2026-08-14

**The real structure of this form, confirmed directly by viewing
`Images/sample 4.jpeg` and by the user directly: every marked cell on
`sample 4.jpeg` is a single-unit mark worth `1`, throughout the entire
form, no exceptions.** The circled Total Dozen at the end of each row is
just the count of marked cells (row 5, `B.4749`: 8 marks under
`55,60,65,70,75,80,85,90`, Total Dozen `8`). This is not a "dense tally
marks that need counting" form — there is nothing to count, every mark
already means exactly 1. **The real, and only, difficulty on this form is
which columns are marked** (position/count of marks), not what value goes
in a marked cell.

This invalidates the ground truth used to judge model accuracy on this
form in every session from 2026-08-06 through 2026-08-13 (`B.4749`'s
"real values" were stated as `55:5,60:8,65:8,70:8,75:8,80:8,85:8,90:5`,
summing to 58 — not 8, and not all-`1`s — see the correction inline in
the "Column-shift detection" section above for where this first went
wrong). Specific conclusions built on that wrong ground truth that should
now be read skeptically, not as confirmed findings:

- **The "Row-crop quantity recount" section's `sample 4.jpeg` framing is
  partly right, partly wrong**: it correctly describes the form as
  "repetitive tally marks" / "tally-of-ones" where the real difficulty is
  "misjudging WHICH column the block starts under," not counting — that
  part holds up. But it still treats the wrong `55:5,60:8,...` figures as
  trustworthy ground truth alongside that framing, an internal
  inconsistency that should have been the tell.
- **The "Full Ollama Cloud model survey" section's verdict on
  `mistral-large-3:675b`** ("wrong header range AND every real
  multi-stroke count flattened to 1") is likely wrong on the second half:
  mistral's reported values (`45:1,50:1,55:1,60:1,65:1,70:1`) were
  probably *correct* on value (real value is 1) — its actual bug is a
  wrong column range/count (6 cells at 45–70 instead of 8 at 55–90), the
  same column-drift category already documented elsewhere in this file,
  not a distinct counting failure.
- **The "Root cause" analysis in that same section** (gemma4:31b:
  "every single one of the 109 reported quantity values is exactly `1`" —
  framed as "the model detecting *that* a cell has a tally mark but not
  attempting to *count* how many strokes are in it") is likely backwards:
  reading every value as `1` was probably correct; the real, still-valid
  evidence of a bug is the *row_total mismatch* (9 of 12 rows didn't sum
  to their own reported total) — that's genuine evidence of wrong
  column count/position, just not evidence of a counting failure.
- **The "mistral-large-3:675b prompt tuning" section's "Quantity-value
  undercounting... Confirmed a real model ceiling" conclusion** should be
  read as unconfirmed. That section already recorded the user correcting
  the "count individual tally strokes" framing as wrong at the time, but
  the deeper implication — that the row is genuinely supposed to be
  all-`1`s, so mistral's all-`1`s output may have been right on value all
  along — wasn't drawn out, and the "real model ceiling" verdict was kept.
- The `qwen3.5:397b` sections' comparisons against this same row inherit
  the same wrong reference point.

**What's still true and unaffected:** the column-shift/drift problem
itself is real and well-documented across multiple forms and pipelines
(this file's v6.1 self-calibration work, the Claude "True per-row crops"
section, `brandlist_match.py`'s `detect_column_shift`) — this correction
doesn't undo that, it just means `sample 4.jpeg`'s specific failures
should be filed under that same bucket instead of a separate "can't count
tally marks" bucket that, per this correction, may not actually exist as
a distinct failure mode.

**Not yet done:** no model has been re-run against `sample 4.jpeg` with
this corrected understanding — the specific accuracy verdicts above are
reasoned from the corrected ground truth, not re-confirmed by a fresh
run. Worth doing before trusting any of this file's older `sample 4.jpeg`
model comparisons as still-accurate, per this project's own
verify-before-trusting practice.

## mistral-large-3:675b on `sample 2.jpeg` (free-form page) reaches 100% — 2026-08-14

Ran `mistral-large-3:675b` and, separately, `richardyoung/olmocr2:7b-q8`
(local, one-off script, not part of any standing pipeline) against
`sample 2.jpeg` -- a free-form handwritten notebook page (no printed
grid, per the "Multi-form-type survey" section), never tested against
this specific model pair before. Real page content, confirmed by direct
photo inspection: **12 items, 58 size/qty cells** (not 14 or 13 as first
counted -- two items wrap onto a second handwritten line with no new
item-name/bullet before the continuation, easy to miscount without
zooming in).

**First mistral run (14 items, 57 cells) had two apparent bugs, one of
which turned out not to be a bug at all -- caught only by zooming in
directly, a real example of this project's own "verify before trusting"
lesson applying to the *investigator*, not just the model:**
- `BABYCARE DRAWER`'s wrapped second line (`70:25,75:30,80:30`, no new
  item name above it) was split into a fake second item mislabeled
  `BABYCARE JETTY` -- a genuine bug.
- `BLOOMER PRINT` appeared twice, the second occurrence
  (`80:15,85:5,90:5`) initially assumed to be a hallucinated duplicate
  with fabricated values. **This assumption was wrong.** Zooming into the
  photo directly confirmed `BLOOMER PRINT` genuinely wraps onto a second,
  fainter, smaller-written line with exactly those values -- mistral read
  it correctly the whole time; the "hallucination" was mis-diagnosed by
  not looking closely enough at the source image before concluding.

**Two prompt fixes added to `MISTRAL_PROMPT_ADDENDUM_BASE`
(`extract_ollama_cloud.py`), confirmed by real re-runs:**

1. **WRAPPED/CONTINUATION LINES bullet** -- explicitly states that a line
   of size:quantity pairs with no new item name/bullet above it belongs
   to the item above, not a new item. Re-run confirmed fixed: item count
   dropped from 14 to the correct 12, `BABYCARE DRAWER` now holds all 8
   pairs in one item, no more fake `BABYCARE JETTY` split.
2. **ON FREE-FORM PAGES WITH NO PRINTED GRID AT ALL bullet** -- the
   existing "sizes past the last header" guidance
   (`_sizes_past_header_bullet`, built 2026-08-13) turned out to be
   written entirely in terms of "this form's printed grid" / "the printed
   header row" -- language that doesn't even apply to a page like this
   one, which has no printed grid anywhere. Added a distinct bullet for
   the no-grid case: there is no implied size ceiling on a free-form page,
   don't stop reading a wrapped line early just because the range looks
   "typical" for the page. Confirmed fixed by a real re-run: `CYCLING
   SHORTS ADULTS` now correctly includes the trailing `110:10` pair (an
   unprinted-overflow value every prior run of every model, on every form,
   had missed up to this point in the project).

**Result after both fixes: a real, fully clean run.** 12/12 items, 58/58
cells, every value matching the photo exactly, confirmed cell-by-cell
(`extracted_ollama_cloud_mistral_sample2_v3/sample 2.json`). This is the
first 100%-accurate result recorded anywhere in this file for any
Ollama Cloud model on any form. A third, separate issue was caught in the
same run but not yet fixed: `order_date` was reported as `"10/06/2024"`
despite there being no date anywhere on the page (only a pre-printed
"MONDAY" diary-page label) -- a fabrication, unrelated to the two fixes
above.

**olmOCR2: the same wrapped-line fix did not transfer, and the
before/after comparison is confounded by real non-determinism.** Only one
run each before and after the prompt change, so nothing here should be
read as a controlled result. The wrapped-line bug reproduced identically
in both runs (still splits `BABYCARE DRAWER`'s continuation into a fake
`BABYCARE JETTY`). Between the two runs, olmOCR2 also: caught the
`110:10` overflow value on one run (a genuine capability, but unclear if
prompt-caused or incidental), newly dropped `TREND PLAIN` entirely on the
second run, and newly misattributed `BLOOMER PRINT`'s wrapped values onto
the following item `BLOOMER PLAIN` (corrupting a previously-correct row)
on the second run. Its "IE" style code also read inconsistently across
runs (`JE`, `OE`, and correctly `IE`, all observed across the two runs on
different rows) -- confirming this specific character confusion is noisy
per-call, not a fixed systematic bug, and not something either prompt
change touched. **Not yet done:** repeat olmOCR2 runs (3-5x, matching this
project's own established practice for characterizing model variance)
before drawing any real conclusion about whether its wrapped-line bug is
prompt-fixable at all.

**Not yet done:** the `order_date` fabrication; testing whether the two
new prompt bullets help or hurt on the ESSA-template forms
(`sample 4`/`sample 5`/`sample 3`) -- they were added generically but
only verified against this one free-form page so far; cleaning up the
`extracted_ollama_cloud_mistral_sample2*` scratch outdirs from this
session's iteration (v1/v2/v3) once this write-up is trusted, per the
2026-08-14 "Repo cleanup" section's own precedent.

## `sample 12-scanned.jpg` (dual-header form) + severe column-shift prompt
## engineering — 2026-08-17

First-ever run against `sample 12-scanned.jpg`, one of the "Other
businesses' own printed order-form templates" named in the "Multi-form-
type survey" section above -- this one double-labels every size column
with both a numeric size (35-110) and an age/chest-equivalent number
(14-44), both meaningful, unlike ESSA's decoy second header row. Also has
a printed per-row TOTAL column, usable as a checksum the same way ESSA's
Total Dozen is.

**Structural fields, confirmed wrong against the photo:** `order_no` blank
(form prints `No. 11792`); `order_date` `"27/12/2024"` (real is `27/3/26`
-- day right, month/year wrong); `party_name` fabricated (`"Fruitful"`,
unrelated to the real handwritten `"Rajasthan Readymade, Chittorgarh"`).
Not addressed this session -- see the note at the very end of this
section on why the equivalent `order_date` fix was reverted from scope
here entirely.

**Quantities: real values almost all correct, but every row column-
shifted, and by a much larger and more erratic margin than any ESSA-
family form tested in this project.** Verified against the photo directly
using the form's own printed TOTAL column as a checksum for every row
(all 10 rows' real values sum to their printed total, confirming the
reading). Real per-row starting column-index ranged 0-11 (real data
mostly clustered far right, columns 10-11 of 17, for 6 of 10 rows); the
model's reported starting index was 0-1 for EVERY row regardless of where
the real data was -- e.g. a row whose real data starts at column-index 10
was reported starting at column-index 1. This is not the same failure as
the small (1-3 position), roughly row-depth-correlated drift documented
elsewhere in this file for ESSA forms -- confirmed by direct measurement
that the model was not locating each row's real horizontal position at
all, just defaulting to a typical-looking early column regardless of
truth.

**Prompt engineering, three real attempts, all scoped to
`MISTRAL_PROMPT_ADDENDUM_BASE` only** (kept out of `extract_claude.py`
deliberately, per direct user instruction: mistral-specific prompt
engineering belongs in the Ollama Cloud file only, not the shared
Claude-pipeline schema/prompt -- the earlier `order_date` fix from this
same session was reverted out of `extract_claude.py` for this same
reason, see below):

1. **A "do not default to an early column" bullet + a "row-total self-
   check is mandatory" bullet**, both citing the exact sample-12 failures
   as worked examples. Confirmed via a real re-run: a real, if modest,
   improvement -- reported starting column-index moved from clustering at
   0-1 to clustering at 2-3 (still wrong, but less wrong), and value
   capture rose slightly (41/45 -> 42/45 real quantity values correctly
   read, recovering one previously-dropped trailing value). But the
   row-total self-check bullet had **zero measurable effect** despite
   citing this exact row's exact arithmetic failure (64 vs a printed 74)
   as a worked example -- the identical wrong sum (64) reproduced in the
   fixed run. Real, unrelated run-to-run variance was also observed in
   several item names and `party_name` between the two runs (e.g.
   `"Cold Packed"` -> `"Cotton Lycra Pink"`), a confound worth naming
   directly rather than silently attributing all change to the prompt
   edit.
2. **A more mechanical rewrite of the same bullet** -- instead of "count
   carefully," explicitly instructing a per-column yes/no pass across all
   17 headers before writing any quantity. Confirmed via a real re-run:
   **this made column position WORSE, not better** -- reported starting
   index regressed back to 0 on most rows, even below attempt 1's result.
   Value capture was unchanged (still 42/45). Reverted immediately back to
   attempt 1's wording, which is the version currently in the codebase.

**Reverted entirely, same session, after a real regression was found on
`sample 5.jpeg`.** Attempt 1's two bullets (`DO NOT DEFAULT TO AN EARLY
COLUMN...` / `ROW-TOTAL SELF-CHECK IS MANDATORY...`) gave a real, modest
improvement on `sample 12` in isolation, but `MISTRAL_PROMPT_ADDENDUM_BASE`
applies to every mistral call regardless of which form is being read --
so the natural next check (already flagged above as "not yet done") was
whether these bullets help or hurt the ESSA-family forms this project
treats as the real production template. **They hurt it, confirmed via a
direct A/B re-run of `sample 5.jpeg` with and without the two bullets,
checked cell-by-cell against the photo:**
- 3 rows (`Exoda Bloomer Plain`, `Bloomer Print`, `Bloomer Plain`) got
  MUCH worse -- their pre-existing column shift (already wrong, -3
  positions) grew to -7 positions with the bullets added, a real,
  substantial regression, not noise.
- 1 row (`Exoda Trunk`) went from an exact match (zero shift) without the
  bullets to a -1 shift with them -- a row this form had never gotten
  wrong before.
- 2 rows (`Classy W. Vest` RN/RNS) improved slightly (-6 -> -5 positions)
  -- a real but much smaller benefit than the damage above.
- The remaining 8 rows were identical either way.
Net: 4 rows clearly worse, 2 rows very slightly better, 8 unchanged. A bad
trade for the form this project has repeatedly called the one to "make
click" first. **Both bullets were removed from `MISTRAL_PROMPT_ADDENDUM_BASE`
entirely** -- confirmed via the same A/B test that removing them restores
`sample 5.jpeg`'s original (better) behavior. `sample 12`'s column-shift
problem is back to fully unaddressed, same as before this session's
attempts.

**Why this happened, most likely:** the bullets were grounded in
`sample 12`'s specific failure shape (defaults to near-left regardless of
true position, needs a push toward respecting far-right data). ESSA forms
have a different, smaller, and apparently *row-depth-correlated* drift
already documented elsewhere in this file (worse for rows further down
the page) -- pushing the model to "count deliberately from the left,
don't assume data starts earlier than it does" seems to have amplified
that existing drift on later rows specifically (rows 8-10, mid-to-late in
a 14-row form) rather than correcting it, since ESSA rows also have long
blank leading runs (columns 45-75 empty on every single row of this form)
that superficially resemble `sample 12`'s trigger condition. **This is
direct, concrete evidence that a column-position fix tuned to one form's
specific failure shape does not safely generalize to a different form's
version of "the same" problem** -- not a hypothesis, a measured result.
This reads as the same kind of ceiling this project has documented
repeatedly for mistral's column-position grounding on other forms (the
`105`/`110` stacked-label case, `sample 8`'s row-bleed case, `sample 10`'s
severe drift) -- prompt wording alone moves the needle a little on the ONE
form it was tuned against, and can move it backward on others, regardless
of how specifically the instruction is worded or how mechanically it's
phrased.

**If revisited, the next real lever is structural, not another wording
attempt**: this project's Claude-pipeline recount mechanism
(`RowQuantityReading` in `extract_claude.py`) already proved, on a
different hard case, that forcing an explicit `first_size`/`last_size`
commitment in the SCHEMA itself (not just the prompt) works better than
asking a model to "read carefully" in free-form. The same idea could be
built as a mistral-only schema variant local to `extract_ollama_cloud.py`
(keeping it separate from `extract_claude.py`, per the same "keep
mistral-specific work in this file only" principle established this
session) -- not attempted yet, a materially bigger task than a prompt
edit, not done this session given the two prompt-only attempts above were
the requested scope.

**Note on `extract_claude.py`:** earlier in this same session, an
`order_date` fabrication fix (`sample 2.jpeg`'s diary-page "MONDAY"
mislabeled as a real date -- see below) was initially implemented in the
SHARED `extract_claude.py` schema/prompt, since it affects any model, not
just mistral. Per direct user instruction, this was reverted out of
`extract_claude.py` entirely -- the user's stated principle: prompt
engineering aimed at getting the best out of mistral specifically belongs
in `extract_ollama_cloud.py`'s own addendum mechanism (already proven by
the `sample 2.jpeg` wrapped-line/free-form-ceiling fixes), not in the file
shared with the real (if currently unaffordable-for-production) Claude
pipeline. `extract_claude.py` is back to its exact original wording as of
this session; the `order_date` root cause (diary page, no real date,
schema had no "empty if absent" carve-out) remains documented below but
unfixed in code, pending a mistral-scoped version of the same idea if
revisited.

## `order_date` fabrication on `sample 2.jpeg` -- root cause found, prompt
## fix attempted and confirmed NOT sufficient — 2026-08-17

**Root cause, confirmed by direct inspection:** `sample 2.jpeg` is a page
from a diary/day-planner with **"MONDAY" pre-printed** in the corner (a
day-of-week label, not a date) -- the page has no calendar date anywhere
on it at all. `ExtractedForm.order_date` (`extract_claude.py`) was a
required string field with no "leave blank if absent" instruction, unlike
`party_name`/`order_no` which both explicitly say "empty string if not
present" -- so under structured-output constraints the model had no
sanctioned way to report "no date" and fabricated one instead. The
specific fabricated value is a strong tell that this isn't random noise:
**June 10, 2024 (the fabricated `"10/06/2024"`) was itself a Monday** --
the model appears to have generated a date consistent with the one real
cue on the page rather than picking arbitrarily.

**Fix attempted, then reverted out of the shared file:** first implemented
in `extract_claude.py` (both the system-prompt text and
`ExtractedForm.order_date`'s field description) -- added the same "empty
string if not present" carve-out `party_name`/`order_no` already have,
plus an explicit rule not to treat a pre-printed day-of-week label as a
date. Confirmed the fix text was actually present in the assembled system
prompt (`build_system_prompt()`), so this wasn't a case of the edit
silently not taking effect.

**Confirmed NOT sufficient, three separate real tests, before the revert:**
`sample 2.jpeg` still returned the exact same fabricated `"10/06/2024"`
(a) on the first run with the fix applied, (b) on an identical immediate
re-run (ruling out run-to-run non-determinism -- this is a stubborn,
repeatable choice, not noise), and (c) with `--think` (reasoning mode)
enabled, which has fixed comparable instruction-following gaps for other
models earlier in this project (e.g. kimi-k2.6). All three produced
byte-identical output. Item/quantity accuracy was unaffected (58/58 cells,
matching the confirmed-good baseline) -- the fix was safe, just not
effective for this specific case.

**Reverted, per direct user instruction, same session:** the user's
stated principle is to keep `extract_claude.py` (the real Claude-pipeline
schema/prompt) and `extract_ollama_cloud.py`'s mistral-specific prompt
engineering separate -- the `sample 2.jpeg` 100% result itself came from
mistral-only addendum bullets, not a shared-file edit, and that's the
established pattern to keep following. `extract_claude.py` is back to its
exact original wording; this fix does not currently exist anywhere in the
codebase, mistral-scoped or otherwise. **Not done this session**: porting
an equivalent (already-proven-ineffective-as-worded) instruction into
`MISTRAL_PROMPT_ADDENDUM_BASE` -- given the exact wording already failed
three ways, a mistral-only copy of the same text would need a genuinely
different angle to be worth adding, not attempted here since this
session's prompt-engineering effort went to the column-shift problem in
the section above instead.

**Not yet done:** a genuinely different hypothesis, if this is revisited
-- e.g. explicitly listing "a day-of-week label alone" as a worked
non-example, framed differently than the reverted attempt; or testing
whether a different model handles this case better. Per this project's
own "stop after several distinct failed attempts" practice (see the
`105`/`110` stacked-label case's four failed tries earlier in this file),
this is close to but not yet past that threshold -- three tries, not four
-- so not formally closed out as a
settled ceiling yet, but should not be re-attempted with a similar
"add an instruction" framing without a genuinely new angle.

## Attempt to generalize the "sample 10 solved" CV+OCR result into the
## pipeline — 2026-08-17: partial success, kept; quantity-reading pass
## reverted, not kept

Following up on the 2026-08-14 "sample 10 solved" result (`sample10_cv_full.py`,
a one-off script with hardcoded pixel anchors for one specific image),
attempted to generalize it into `extract_ollama_cloud.py` proper: read
quantities on any grid-form image via `grid.py`'s CV row-boundary
detection + `ocr_cell_read.py`'s OCR/DP column assignment (the same
mechanism the local Ollama pipeline uses), instead of hand-tuning anchors
per image.

**Kept, confirmed working, still in the codebase:**
- **`grid.py`: `iter_row_boundary_candidates_auto()`** -- wraps the
  existing `iter_row_boundary_candidates` with an automatic upscale retry
  (native, then 2x, then 4x), since a low-resolution/tightly-cropped image
  can return ZERO candidates outright at native resolution (confirmed:
  `sample 10 crop.jpg` at ~24px/row). Deliberately does NOT stop at the
  first scale that finds *any* candidate -- confirmed necessary after a
  real bug: on `sample 10.jpeg` (the raw, uncropped photo, which has an
  extra DATE/letterhead row above the real column-header row), 2x upscale
  found a geometrically-uniform run of lines that was actually WRONG (it
  locked onto the boundary above that extra row instead of the real
  header), and every strategy at that scale agreed on the same wrong
  answer. Stopping there would have given up before trying 4x, which (on
  the properly-cropped image) finds the right answer. Now yields
  candidates across every scale and lets the caller's own validation
  decide, same as the existing multi-strategy design.
- **EXIF-orientation auto-correction** (`_exif_corrected_bytes`,
  `_prepare_image_exif_safe` in `extract_ollama_cloud.py`) -- confirmed
  necessary via this session's earlier finding (see the "manually-upscaled
  image" sections above): a photo can carry an EXIF orientation tag while
  its raw pixels stay sideways, which neither Ollama Cloud's API nor a
  plain `Image.open()` corrects for. Now applied automatically to the main
  call's image and the recount pass's raw bytes -- no-op for any image
  without an orientation tag (confirmed the common case for this project's
  own sample images).
- **`_rescale_for_ocr()`** -- renders any OCR-facing crop at a single,
  controlled width (3800px, safely under PaddleOCR's own ~4000px internal
  cap) regardless of what scale row-boundary detection needed, instead of
  building crops directly from whatever (possibly 4x-upscaled) image
  detection succeeded on. A real, partial fix -- see below for what it
  didn't fully solve.
- **`_select_cv_row_boundaries()`** (used by the recount pass, itself off
  by default) now uses the auto-upscale iterator and the safe-width
  rescale -- a strict improvement to that function's own documented "still
  open" row-crop-alignment gap, independent of the reverted work below,
  since recount reads rows via a VLM text call, not OCR digit assignment.

**Reverted, NOT kept:** a new `_read_quantities_cv_ocr()` function and a
`--cv-quantities` CLI flag that would use the above to override the
model's own quantities with CV+OCR-derived ones. Confirmed on a real
end-to-end run (`sample 10 crop.jpg`, `mistral-large-3:675b`) that
candidate *selection* worked perfectly -- 18/18 headers found, 14/14 rows
correctly aligned to their expected item, using the general auto-upscale
code path, no hardcoded anchors -- but the actual digit-reading step
(`ocr_row()` re-detecting the header row's exact text within each row's
own stitched crop) failed on 12 of 14 rows, even though the SAME header
pixels read correctly (18/18) in isolation. Root cause traced partway to
PaddleOCR's own internal preprocessing (`resize_image_type0` in
`paddlex`'s `text_detection/processors.py`) treating a very
wide/short header-plus-row strip inconsistently depending on exactly what
image is passed in -- the `_rescale_for_ocr` fix above reduced but did not
eliminate this. Net effect before reverting: `--cv-quantities` delivered a
real reading for only 1 of 14 rows, with ~5 minutes of added latency
(candidate search + PaddleOCR calls) for almost no benefit -- worse than
not having the feature at all. Removed per the project's "keep only
what's working" standard rather than leaving a costly, non-functional
option in the CLI.

**If revisited:** the real blocker is `ocr_row()` re-OCRing the header
band fresh for every row instead of OCRing it once and reusing those
x-positions -- a more invasive change to `ocr_cell_read.py`, which is
shared with the already-proven local Ollama pipeline and would need its
own careful regression check (e.g. re-confirm `sample 5.jpeg` still gets
an exact match) before trusting it, not attempted this session given time
already spent. The raw `sample 10.jpeg`'s separate DATE-row/header-row
boundary ambiguity (independent of resolution) is also still unsolved.

## A 4th attempt at `sample 5.jpeg`'s `105`/`110` case, tried and reverted — 2026-08-14, same day

Zooming into `sample 5.jpeg`'s Fairlady rows at high resolution (prompted
by the user asking to "be flexible on finding sizes") revealed a real
mechanism the three 2026-08-13 attempts never described: the writer
doesn't add a column past the page edge -- there's no room. Instead they
**subdivide the existing last printed column ("Total Dozen") into two
tiny stacked size/quantity sub-cells** (a small underlined "105" or "110"
directly above its own quantity) for just that one row, while every other
row still uses that same column as one ordinary total. The user further
pointed out this same "label stacked above quantity, underlined" shape is
also how `sample 3.jpeg`'s `MM K4532` letter sizes are written (confirmed
by zooming in: `S` over `6`, `M` over `6`, etc.) -- a genuine, visually-
confirmed unifying convention across three previously-separate failure
cases (this one, letter sizes, and `sample 2.jpeg`'s whole free-form
layout), not three unrelated special cases.

**Tried: replaced the two-bullet EXTRA-COLUMNS/LETTER-SIZES guidance with
one unified "look for a stacked label-over-quantity shape anywhere on the
row" rule, grounded in this real mechanism.** Result, confirmed by a real
re-run and checked cell-by-cell against the photo: **a genuine accuracy
regression, not just a non-fix.** Scored by real quantity VALUES captured
(this project's own established evaluation method, since column-label
offset is separately correctable) against the photo: **62/72 (~86%)**,
down from the already-confirmed 2026-08-13 baseline of 66/69 (~95.7%) for
mistral on this exact form. New errors appeared that were not part of
this model's previously-documented failure modes: `Exoda Trunk` and
`Image FCD` each dropped a leading digit (`122`→`12`, `126`→`26`,
`77`→`7`), and `MYNA`/`OE`'s last cell reverted to the old, already-
known-wrong `3` instead of the Claude-confirmed-correct `13`. The target
case itself was still not fixed either -- both Fairlady rows were still
missing their `110` (and now also `105` on one row) entirely.

**Reverted immediately** back to the original two-bullet text (unchanged
from 2026-08-13). Confirmed via a follow-up re-run that this restores
accuracy to ~93% (66/71 by the same value-only method) -- not exactly
69/69's prior figure (small counting differences, plus one already-known
ambiguous smudged cell on `MYNA` reading a new, different wrong value
this run: `120`, neither of the two previously-debated `20`/`30` guesses)
but clearly back in the same range as the known-good baseline, confirming
the regression really was caused by that specific prompt text and not
something else.

**This is the 4th independently-tried, independently-failed prompt
approach to this exact case** (3 from 2026-08-13's "mistral-large-3
prompt tuning" section, plus this one) -- and the first of the four to
actively make other rows worse while failing to fix the target. Per this
project's own established practice (see the `code-review`-adjacent lesson
throughout this file: stop iterating on a specific case once multiple
genuinely distinct hypotheses have all failed the same way), **this
should be treated as a settled vision-attention ceiling for
`mistral-large-3:675b` on this specific case, not re-attempted again
without a genuinely new, evidence-backed hypothesis** -- a 5th rewording
of "look harder" is very unlikely to succeed where four have failed,
including one grounded in an accurate, zoomed-in description of the real
mechanism. Today's two OTHER new prompt bullets (wrapped-line-continuation,
free-form-page size ceiling, both added in the `sample 2.jpeg` section
above) are unrelated to this regression and remain confirmed wins.

## mistral-large-3:675b on `sample 8.jpeg` (first run) + a row-bleed fix attempt, reverted — 2026-08-14

First-ever mistral run against `sample 8.jpeg` (previously only tested
with Claude, 2026-08-08 -- see "Buyer-table cross-check for non-ESSA
forms"). This form is denser than any ESSA-template form tested so far:
19 printed size columns (`35`-`120`) versus ESSA's ~13, narrower cells.
Checked cell-by-cell against the photo (zoomed crops, not just the
full-page view):

**Confirmed real errors, several distinct failure shapes:**
- **Tail-column shift** (same family as the already-documented
  row-depth drift, but here appearing near the right edge of individual
  wide rows): `Fairlady Plain`'s real `100:25,105:blank,110:4` came back
  `100:25,105:4` (the `110` value relabeled onto `105`, blank swallowed).
  `Fairlady Print`'s real `100:22,105:1,110:4` came back `100:22,105:14`
  -- two separate real values fused into one fabricated `14`, `110` lost.
  `Drawers Plain` and `Super Point Drawers` showed the same pattern at
  their own tails.
- **Row-bleed** (a new failure class, not seen on `sample 2.jpeg`):
  `Casual RNS`'s real row is blank past `65:10` -- but the run attached
  `70:3,75:6` to it, which are actually `Salma Top casual`'s own real
  trailing `78:3,80:3` values, bled into the row above. `Salma Top
  casual`'s own middle values came back scrambled as a result.
  `Calm RNS` showed a related 2-column shift with one value dropped.
- **Plain digit misreads**, unrelated to position: `Trend Plain`'s `80`
  column read as `8`, confirmed by zooming in it's `6`. `Super Point
  Drawers`'s `50` column read as `35`, real is `39`.
- **Looked correct**: `Calm RN` (verified cells), `20-20 RNS` (5/5
  exact), party name (`Rajkumar & Co.`), order no. (`2052`). `order_date`
  not verifiable -- the handwritten date is smudged in the photo itself.

**Row-bleed fix attempted, reverted after making things worse.** Added a
new bullet ("DO NOT LET VALUES BLEED INTO THE ADJACENT ROW", grounded in
the exact `Casual RNS`/`Salma Top casual` case above) to
`MISTRAL_PROMPT_ADDENDUM_BASE`. Real re-run result: **not fixed, and
actively worse** -- the bleed didn't stop, it relocated to different
(still wrong) columns on both rows, a previously-clean row (`20-20 RNS`)
regressed with a new spurious `80:2` value, and a new hallucinated cell
appeared on `Fairlady Plain` (`120:6`, nothing there on the real page).
Reported cell count rose from 63 to 70 purely from added wrong/spurious
values, not real fixes. **Reverted immediately**; a follow-up re-run
confirmed cell count back near baseline (66, consistent with this
model's normal run-to-run variance, not the inflated/hallucinating 70).

**Read together with the `105`/`110` case above, this is now a
consistent pattern across two independent spatial-grounding problems on
two different forms**: column-position drift (4/4 prompt attempts
failed, one regressed) and now row-boundary confusion (1/1 attempt
failed, also regressed). Both look like the same underlying
visual-grounding ceiling on different axes, not instruction-following
gaps -- prompting is very unlikely to fix either without a genuinely new
angle neither of us has found yet. Not yet done: no review page/CLAUDE.md
entry exists yet confirming whether `sample 8.jpeg`'s tail-shift errors
respond to the *already-existing* (unchanged) "sizes past the last
header" bullet the way `sample 2.jpeg`'s did -- this form wasn't tested
before any of today's prompt changes, so there's no clean pre/post
baseline for that specific comparison the way `sample 5.jpeg` had one.

## Row-crops confirmed to fix the row-bleed bug -- but only when hand-aligned; the automated recount pipeline still isn't safe to default on — 2026-08-14, same day

Directly following the row-bleed finding above, tested whether isolating
just the two bled rows (`Casual RNS` / `Salma Top casual`) in their own
crop -- removing the neighboring row from the image entirely -- would
stop the contamination. **Confirmed yes, cleanly**, via a hand-built crop
(header band + exactly these two rows, precisely aligned by eye against
the photo) sent with a simple plain-text prompt instead of a JSON
schema: `Casual RNS` went from a bled, contaminated 8-value read down to
a clean 6-value read (5/6 correct by value, no bleed at all); `Salma Top
casual` went from a scrambled, incomplete read to all 9/9 real values
captured correctly. This isolated the real cause: it was never the
row-cropping *idea* that was broken, it was the recount call's forced
JSON schema, which mistral (like every other Ollama Cloud model tested
so far -- qwen3.5:397b, kimi-k2.6) can't reliably comply with.

**Built into the pipeline**: `_recount_one_row_text()` +
`ROW_RECOUNT_TEXT_PROMPT` + `_parse_row_text_reading()` in
`extract_ollama_cloud.py`, replacing the schema-constrained per-row
recount call (`_recount_one_row`, kept in the file unused-by-default) with
a plain "Item seen: / Quantities: / Total:" text format parsed by regex --
mirrors how the original local Ollama pipeline's own VLM fallback
(`stage_b_row`) already avoids this exact class of failure. One real bug
caught before it shipped: `QuantityPair` was used in the new parser but
never added to the `from extract_claude import (...)` block -- confirmed
by a `NameError` on the very first real run, fixed by adding the missing
import.

**Real end-to-end test on `sample 8.jpeg` (all 10 rows) after the fix,
mixed result -- do not treat this as solved:**
- The JSON-compliance problem really is fixed: zero "invalid JSON" or
  empty-response failures across all 10 rows (versus 7/10 with the old
  schema-constrained call).
- But most row-crops still failed to help, for a DIFFERENT,
  already-known reason: this dense form's main call reports
  `row_top_frac`/`row_bottom_frac` per item that don't reliably track the
  real row boundaries (several crops showed a visibly different item's
  name than the one they were labeled for -- `'Casual'`'s crop showed
  `'Trouser Plain'`, `'20-20'`'s showed `'Trend Plain'` -- and others
  landed on the form's secondary/decoy header line instead of the real
  one). `Casual RNS`/`Salma Top casual` themselves got no benefit in this
  automated run -- their crops were among the misaligned ones, so the
  bleed in the final output is unchanged from before.
- **A genuine new regression, more serious than a non-fix**: `_apply_recount`'s
  existing alignment/hard-validator safety nets (imported unmodified from
  `extract_claude.py`) caught most of the misaligned crops correctly and
  fell back to the main reading -- but not all of them. `Drawers Plain`'s
  recount reading was silently accepted and marked `"resolved"` (i.e.
  trustworthy, via the catalog-match tiebreak) even though it was
  actually **`Super Point Drawers`' real data** (`8,10,25,39,45,25,20,...,8`)
  from the row below -- confirmed by zooming into the photo directly. A
  `"resolved"` status reads as higher-confidence than the original
  bleed-contaminated-but-flagged output, making this a worse failure mode
  to ship than doing nothing, not just an unhelpful one.

**Decision: `--no-recount` remains the recommended default for this
file**, updated in the CLI help text accordingly. The text-format fix is
real and worth keeping in the code (confirmed working when the crop is
actually well-aligned), but it only fixes one of two independent
problems blocking recount from being trustworthy here -- the JSON-
schema-compliance failure (fixed) and the row-crop-alignment reliability
problem (still open, pre-existing, and not touched by today's work).
Turning recount on by default would trade a visible, flaggable bleed bug
for an occasional silent wrong-but-confident answer, which is a worse
trade for a pipeline whose whole design principle (per the "Design
decisions worth preserving" section at the top of this file) is that a
model being honestly wrong beats a model being confidently wrong.

**Not yet done:** fixing `row_top_frac`/`row_bottom_frac` reliability
itself (the actual remaining blocker) -- would need its own investigation,
possibly along the lines of the local Ollama pipeline's `grid.py`
CV-based row-boundary detection instead of trusting the model's own
self-reported fractions, though that approach was built for a different
pipeline and hasn't been ported here. Also not done: converting the
whole-table-crop recount fallback path (`_call_schema(...QuantityRecount...)`,
still JSON-schema-based) to the same plain-text approach -- only the
per-row path was converted, since that's the one this session's testing
actually exercised.

## Row-bleed solved -- but by a clean input image, not by any of the engineering above — 2026-08-14, same day

After the CV row-crop fix above ran into its own new problem (its header-
crop-height heuristic, `1.3x average row height` above the first row
boundary, overshoots onto the "Party Name" line instead of the real
numeric header row when tried against a differently-cropped input image
-- confirmed by direct testing: 0/17 headers found on every one of 6
candidates for `Images/sample 8 manual crop.jpg`, a user-supplied,
manually deskewed and tightly-cropped version of the same
`sample 8.jpeg` photo), the user's own next instruction cut through the
whole direction: skip row-crops entirely, just run the plain main call
against the cleaner image and see what mistral does unassisted.

**Result: the row-bleed problem that motivated all of today's row-crop
work is completely gone**, confirmed value-by-value against the photo
(`extract_ollama_cloud.py --no-recount`, no CV, no per-row crops, just
one main call on `sample 8 manual crop.jpg`):
- `Casual RNS`: `{40:2, 45:12, 50:15, 55:12, 60:12, 65:10}` -- exactly
  its own 6 real values, zero contamination from the row below (versus
  the original photo's runs, which consistently bled 2-4 extra values
  from `Salma Top casual` into this row every time).
- `Salma Top casual`: matches its own real value set exactly
  (`2,3,8,8,8,5,6,3,3`, 9/9), including its own trailing `3,3` that
  previously bled into the row above instead of staying here.
- Every other row also matched by value: `Trend Plain` 5/5 (including
  the `6` that the original photo consistently misread as `8`),
  `Super Point Drawers` 8/8 (including `39` read correctly instead of
  the original photo's `35` digit-misread), `Drawers Plain` 9/9,
  `Calm RNS` 4/4, `20-20 RNS` 5/5. Only real gap: `Fairlady Print`
  missing one small tail value (`105:1`).
- The one issue that DID survive: the already-documented, more benign
  column-label shift (values land under a header one position to the
  right of correct, e.g. real `78:3` reported as `80:3`) -- a
  fundamentally different and easier problem than row-bleed, since the
  values themselves are still right.

**Takeaway: the row-bleed bug was substantially an image-quality
problem, not a model-reasoning problem** -- perspective distortion,
skew, and background clutter in the original phone photo were making
adjacent rows genuinely harder for mistral to keep visually separate,
and a clean, deskewed, tightly-cropped input fixed it outright, with no
prompting, no row-crop engineering, and no recount pass needed at all.
This is a simpler and more effective fix than the CV row-boundary
integration built earlier today -- worth remembering before reaching for
algorithmic fixes on a hard row: check input image quality first.

**Not yet done:** no systematic test of whether manual deskew+crop helps
on OTHER hard cases in this file (the `sample 5.jpeg` `105`/`110` case,
`sample 4.jpeg`'s tally-adjacent form, `sample 3.jpeg`'s letter sizes) --
only tried on `sample 8` so far, and only via a user-supplied crop, not
an automated preprocessing step. If this generalizes, an automated
deskew+crop preprocessing pass (e.g. reusing `grid.py`'s own deskew
estimation, or the document-scanner approach already ruled out elsewhere
in this file for a *different* reason) could be a higher-leverage fix
than any of today's per-form prompt/CV engineering -- not yet built or
tested as a standalone preprocessing step.

## First run against `sample 10.jpeg` -- a new, harder form type, mostly unsolved — 2026-08-14, same day

First-ever run (mistral-large-3:675b) against `sample 10.jpeg`, one of
the not-yet-addressed templates named in the "Multi-form-type survey"
section above (`SARAIYA GARMENTS, DUNGARPUR` -- a clean, printed/typed
grid, not handwritten, using `*` as a quantity marker in many cells
instead of a digit, alongside a printed note "PLEASE DISPATCH AS PER
RATIO"). Checked against a precise zoomed read of the header + first 4
rows before trusting anything.

**Two real, confirmed problems, distinct from anything seen on the
ESSA-family forms tested earlier today:**

1. **Severe, inconsistent column-position shift.** `LOOP SQUARE M 4301
   SHORTS HALF`'s real values (`75:*, 80:3, 85:5, 90:5, 95:5, 100:3,
   105:*`) came back as `{"30":3, "35":5, "40":5, "45":5, "50":3}` --
   the VALUES are exactly right (`3,5,5,5,3`), but shifted 10 header
   positions left. The very next row, with the same real column range
   (`75-105`), shifted by a *different* amount (8 positions, not 10) --
   not a single consistent offset, unlike the mild, roughly-monotonic
   drift documented for ESSA forms elsewhere in this file. Plausible
   cause: unlike ESSA forms (most rows start around the same column),
   this form's rows genuinely start at very different columns row to
   row (`50` through `75`), which may be defeating whatever positional
   tracking works well enough on more uniform layouts.
2. **`*` cells silently dropped entirely** -- not reported as blank, not
   flagged, just absent, with no schema/prompt concept of this marker
   at all (this form's `*` convention was explicitly out of scope per
   the original survey).

**Fix attempted, reverted after no improvement:** added a bullet to
`MISTRAL_PROMPT_ADDENDUM_BASE` explaining the `*` convention, instructing
the model to treat it as a real occupied cell (not to skip over it when
counting columns) and to surface each one via a specific per-cell note.
Real re-run result: the severe position shift was still just as wrong
(a different wrong shift, not an improved one), and **zero of the
requested per-cell notes appeared at all** -- the model didn't act on the
instruction in either respect. Reverted. This is now the 3rd distinct
column-position-related prompt attempt to fail this session (after the
`105`/`110` "unified stacked-label" attempt and the row-bleed attempt on
`sample 8.jpeg`, both also reverted after real regressions/non-fixes),
reinforcing that this class of problem doesn't respond to prompt wording
for this model.

**Not yet tried:** the "clean input image" fix that solved `sample 8`'s
row-bleed problem outright (see the section above) hasn't been tested
here -- `sample 10.jpeg` is already a flat, well-lit photo of a printed
sheet (not a handwritten form with visible perspective/skew like
`sample 8.jpeg`'s original photo), so it's unclear whether the same fix
would even apply, but it hasn't been ruled out either. Also not tried:
whether Claude handles this form's severe position variance or `*`
convention any better (no API credits available in this session to
test) -- given Claude's confirmed track record resolving several
column-position cases this project's other models couldn't (letter
sizes, ditto-composition, the `sample 2.jpeg` wrapped-line case), it may
be worth checking there before investing further engineering effort into
the Ollama Cloud path for this specific form. **Net status: `sample
10.jpeg` remains substantially unsolved** -- item names/order
no./date/party-name-blank all read correctly, but quantities are not
trustworthy on this form yet.

## `sample 10 crop.jpg` (manual crop) + an ESSA-schema-anchor hypothesis, tested and ruled out — 2026-08-14, same day

**The manual crop helped, partially -- unlike `sample 8.jpeg`, not a
full fix.** Re-ran mistral against `Images/sample 10 crop.jpg` (a
straightened, tightly-cropped version of the same photo, user-supplied
like `sample 8`'s). Checked against a precise zoomed read of every row:
- `LOOP SQUARE M 4301 SHORTS HALF` is now exactly correct, zero shift
  (was 10 positions off on the original photo).
- The wildly inconsistent shift (8 positions on one row, 10 on the next)
  mostly resolved into a more consistent **-3 position shift** across
  most rows -- a real improvement in predictability, though still wrong.
- A NEW problem appeared that wasn't as prominent before: most rows now
  **undercount**, missing 1-2 real trailing values they used to capture
  (e.g. `MM KOKO FULL PANT` has 5 real values, this run got 3;
  `HUNTER 3/4TH PANT` has 7 real threes, this run got 5).
- `*` markers remain completely unaddressed in both the original and
  cropped runs -- silently dropped, no notes, no change.

Net: real but partial improvement, not a clean solve like `sample 8`.
Plausible reason for the difference: `sample 8`'s row-bleed was
specifically a perspective/skew problem between adjacent rows, which a
flat crop directly fixes; `sample 10`'s harder problem -- rows starting
at wildly different columns from each other, `*` markers, and no
Total/Cases checksum column to lean on -- isn't primarily an image-
quality problem, so a cleaner image only partially helps.

**A second, sharper hypothesis tested, then ruled out:** the user
noticed the column drift here was unusually severe compared to every
ESSA-family form tested this session and asked whether mistral might be
anchoring on ESSA's own conventions. Checked directly and confirmed real:
`QuantityPair.size`'s field description (imported unmodified from
`extract_claude.py`, shared with the Claude pipeline) literally reads
`"...e.g. '45', '90'."` -- and since this schema is passed as `format=`
on EVERY Ollama Cloud call via `model_json_schema()`, that exact
ESSA-specific example (45 is ESSA's own first real header column) is
baked into every request this file has ever sent, main or recount,
regardless of `--no-recount`. Patched out locally in
`extract_ollama_cloud.py` (`_neutralize_schema_examples()`, applied at
the `format=` call site) rather than editing the shared class in
`extract_claude.py` directly -- keeps the Claude pipeline's proven schema
completely untouched while testing the hypothesis in isolation.

**Result, confirmed by a real re-run: no measurable change.** Same `-3`
position shift pattern, same undercounting, output essentially identical
to the pre-patch run (differences within this model's normal run-to-run
noise, not a systematic improvement). So this specific anchor was NOT
the (or at least not the dominant) cause of `sample 10`'s drift, despite
being a real, well-grounded, previously-untested thing to check. Kept
the patch anyway -- it's a strictly more correct, form-agnostic
description regardless of whether it explains this case, and costs
nothing -- but the underlying drift on this form is still unexplained.

**A fifth hypothesis, tested and also ruled out: schema-independent
(plain-text) output**, the same technique that fixed the recount
mechanism's JSON-compliance failures earlier today. Built a one-off test
(system prompt unchanged, `format=` schema constraint removed entirely,
a plain "ITEM N / NAME: / TYPE: / QUANTITIES: / ITEM_NOTE:" text format
requested instead) and ran it against `sample 10 crop.jpg`. Result: not
better, arguably less consistent -- shifts now ranged from +2 to -6
header positions (both directions), versus the JSON-schema version's
fairly consistent -3, plus a mix of spurious extra values on early rows
and undercounting on later ones. So the JSON/grammar-constrained
decoding mechanism itself was not the cause either -- this rules out the
same class of fix that worked for the recount mechanism.

**Net status: `sample 10.jpeg`/`sample 10 crop.jpg` remain substantially
unsolved.** Five real, distinct hypotheses now tested and ruled out for
this form specifically (the `*`-handling prompt bullet, the manual crop
-- partial only, the ESSA-schema-anchor patch, schema-independent
output), on top of the three column-position hypotheses already ruled
out on other forms earlier today. **Claude is NOT a viable next step**
despite resolving analogous problems elsewhere in this project's
history -- confirmed by the user (2026-08-14): the Claude API is not
affordable for this project's real production workload, only ever used
here for one-off comparison testing (see the "Claude API pipeline"
section's own framing, which is stale on this point). The realistic
paths forward, if this form is revisited: try the other Ollama Cloud
models already characterized in this file (gemma4:31b, qwen3.5:397b,
kimi-k2.6) in case one handles this specific row-start-position
irregularity better despite being weaker generally, or accept this form
as a human-review-heavy case and lean on the review page's flagging
rather than chasing full automation.

## `sample 10.jpeg` solved -- CV row/column detection + PaddleOCR, bypassing the VLM entirely for quantities — 2026-08-14, same day

After six independent VLM-based hypotheses all failed on this form (see
above), tried a structurally different approach instead of another
prompt/schema variant: since `sample 10.jpeg` (unlike every ESSA-family
form) is a clean, machine-printed/typed grid rather than handwritten,
it's exactly the kind of input the ORIGINAL local Ollama pipeline's
CV+OCR machinery (`grid.py` + `ocr_cell_read.py`, already proven on ESSA
forms) was built for -- detect the real ruled lines directly instead of
asking any model to guess column position at all.

**Critical precondition, confirmed by direct testing: `grid.py`'s row
detection requires upscaling first.** At native resolution (`sample 10
crop.jpg` is 1142x375 for 15 row boundaries, ~24px/row), row-boundary
detection found ZERO candidates on both `sample 10.jpeg` and `sample 10
crop.jpg` -- not a bad result, a total failure. Root cause: `grid.py`'s
peak-detection parameters (`min_gap=20-25px`) were tuned against ESSA's
much larger source photos (~99px/row) and mechanically cannot distinguish
two row boundaries closer together than that gap. **Upscaling 4x fixed
this immediately** -- 3 candidates found, one (candidate 1) visually
confirmed exact against every real ruled line, header through the last
row. This precondition likely generalizes to any low-resolution or
tightly-cropped input, not just this one form -- worth remembering before
concluding CV detection "doesn't work" on a hard image without checking
resolution first.

**Column detection needed calibration, not just detection.**
`grid.py`'s `detect_column_boundaries()` found only 14 of the ~19 needed
peaks even after upscaling (some real gridlines are too faint/thin to
individually clear the prominence threshold) -- not enough for a full
uniform run. Fixed by anchoring on two confidently-detected gridlines and
interpolating the rest evenly between them, which is valid specifically
*because* this is a genuinely machine-printed, evenly-spaced table (an
assumption that would NOT hold on a handwritten/perspective-warped form).
First interpolation attempt was off by exactly one column-width in a
consistent direction across every single row -- confirmed by comparing
OCR'd values against the real photo (values all correct, positions all
shifted the same way) -- fixed by shifting the anchor pixels by one
column-width. The fact that the error was uniform across every row,
unlike ANY VLM attempt's inconsistent per-row drift, was itself the
signal that this was a fixable calibration bug, not a fundamental
limitation.

**Result: all 14 rows match the photo exactly**, confirmed cell-by-cell,
after this single global calibration (no per-row correction needed at
all). Two rows (`HUNTER 3/4TH PANT`, both `BASIC RIB NECK` rows) where
this CV+OCR read disagreed with an earlier manual zoom-read turned out to
be cases where the EARLIER MANUAL READ was wrong (undercounted by one
cell each) -- confirmed by re-zooming precisely; the CV+OCR result was
right both times, a reminder that manual verification isn't infallible
either and disagreements should be re-checked, not assumed to favor the
human reader.

**`*` ratio-marker cells: position always correct, symbol recognition
unreliable.** PaddleOCR doesn't reliably recognize the literal `*` glyph
on this form's font -- it's sometimes read as `.`, sometimes as a garbage
character, sometimes missed entirely. Handled by treating any short
(<=2 char) non-digit detection in a real column slot as a probable ratio
marker (surfaced as a note, e.g. `"MM KOKO FULL PANT, size 60: ratio
marker (*), not a quantity"`) rather than requiring an exact "*" match --
never invented as a fake number, matching this project's "leave it out
rather than invent a value" principle. Confirmed working for 4 of the
~20+ real `*` cells on this form; most weren't detected as anything at
all (silently absent, not misrepresented) -- a real, acknowledged gap,
not claimed as solved.

**Structural fields (item names, style/type, order no., date) reused
directly from mistral's already-accurate structural read** rather than
re-derived via CV+OCR -- those were never the problem on this form, only
quantities were. `party_name` stayed blank (mistral never found one on
this form across any run; not independently verified against the photo).

**Implementation status: a one-off scratch script
(`sample10_cv_full.py`), not yet a reusable pipeline component.** Row/
column boundaries are hardcoded pixel anchors specific to this exact
image, manually found and calibrated through this session's interactive
debugging -- not yet generalized into something that auto-detects good
anchors for a new image the way `grid.py`'s row-boundary candidate
selection already does. Output written to
`extracted_ollama_cloud_mistral_sample10_cvocr/sample 10 crop.json` +
`.review.html`.

**Applicability, assessed but not yet tested on other forms:** this
approach needs (a) a real ruled grid (not free-form pages like `sample
2`/`1`/`6`) and (b) ideally machine-printed cell content for OCR to read
cleanly without extra correction machinery -- `sample 9`/`13` ("Excel-
maintained" templates per the multi-form-type survey) are plausible
similar candidates, not yet checked. Handwritten-content forms with a
real grid but a non-ESSA template (`sample 8`) would likely need the
row/column detection step (already shown promising for `sample 8`'s rows
specifically) PLUS the ESSA pipeline's handwriting-specific OCR
correction machinery (drift self-calibration, digit-lookalike recovery,
x-range exclusion) ported over -- not yet attempted, and not expected to
be as clean a win as this typed-form case.

## `sample 10` revisited with a manually-upscaled image -- orientation, not resolution, was the real blocker — 2026-08-17

User supplied a manually-upscaled photo (`Images/sample 10 upscaled.jpg`,
750x2284) to test the earlier hypothesis that `grid.py`'s CV row-detection
needs more px/row (see "Did we add upscale to the pipeline?" discussion
this same session -- upscaling had never been wired into the standing
pipeline, only used ad hoc in the now-gone `sample10_cv_full.py` scratch
script). Ran `mistral-large-3:675b` (`--no-recount`, the file's own
recommended default) directly against it first.

**Result: catastrophic failure, worse than any prior `sample 10` attempt
in this file.** Only 12 of the 14 real items came back; the first two
(`LOOP SQUARE M 4301`, `LOOPER Z M 3959`) were missing entirely -- one of
them leaked into the `party_name` field instead. Item order was roughly
reversed relative to the real form. Worse than a simple miscount: item
names and quantities were cross-contaminated across rows -- e.g. the
output's `"BASIC RIB NECK WHITE"` row actually held `K 3676`/`MM HAPPY
GIRL SKIRT`'s real values, confirmed by viewing the source photo directly.

**Root cause, confirmed by testing, not assumed:** the supplied image was
captured/upscaled sideways (rotated 90°) -- the header text and every row
of text runs vertically, not horizontally. Rotated upright with PIL
(`Images/sample 10 upscaled rot.jpg`) and re-ran the identical model/flags.

**Result after rotation: 14 of 14 items correctly identified, in the
correct order, zero cross-row contamination.** Checked value-by-value
against the photo: **78 of 86 real quantity values captured correctly by
VALUE (~91%)**, using this project's own established evaluation method
(value capture, since column-label offset is separately correctable).
`HUNTER 3/4TH PANT` matched exactly, zero shift. Every other row had the
right values but landed 1-3 header positions off (the same
already-documented column-position-shift failure mode from every other
`sample 10` session, not a new problem), plus 1 dropped trailing value on
a few rows. `*` ratio-marker cells were almost entirely dropped, same
known gap as before.

**Takeaway: for this specific image, orientation was the dominant blocker,
not resolution.** The 750x2284 upscaled image was well above the px/row
threshold that was hypothesized to matter for `grid.py`'s CV detection --
but that hypothesis was never actually exercised here, because the VLM
path (`extract_ollama_cloud.py`, not `grid.py`) was used, and its failure
mode (reversed/scrambled reading order on sideways text) is unrelated to
CV row-boundary pixel density. **Not yet isolated**: whether the upscale
itself contributed anything on top of the rotation fix -- no low-resolution
upright version of this image was tested for comparison. Practical lesson
for next time: check image orientation before reaching for resolution as
the explanation when a VLM's item order looks scrambled.

**Follow-up same day: a 4x-upscaled version of the same photo
(`Images/sample 10 upscaled 4x.jpg`, 1500x4568, also supplied sideways,
rotated upright the same way -> `sample 10 upscaled 4x rot.jpg`) tested
whether more resolution helps mistral-large-3 once orientation is no
longer the confound. Result: a real but mixed improvement, not a clean
win.** Value recall rose from 78/86 (~90.7%, the 2x/750-wide version) to
**81/86 (~94.2%)** -- `G 4468`, `MM THANSIKA`, `MM HAPPY GIRL SKIRT`, and
`K 3676` each recovered a trailing value they'd dropped at lower
resolution. Column-position shift (values landing 1-3 header positions off)
was unchanged in magnitude -- more resolution didn't touch that failure
mode. But 4x also introduced a **new failure not seen at 2x: 3 fabricated
cells**, all a spurious `"110": 3` on `F.G. B 3662`, `LOOPER Z B 4230 DS`,
and `HUNTER 3/4TH PANT` -- all three rows are genuinely blank past `95` in
the photo. `HUNTER 3/4TH PANT` had been a clean zero-shift exact match at
2x; at 4x its real cells stayed exact but gained this one phantom addition.
Net: **more resolution traded a small recall gain for a new tail-column
hallucination** -- not a strictly-better result, and not yet enough data
(one image, one trial each) to say whether 2x or 4x is the better default
for this model/form combination.

## Mistral fixes — 2026-08-17

Two targeted, structural (not prompt-wording) fixes, built and validated
against the "Mistral-only findings" section above -- both scoped to
`extract_ollama_cloud.py` only, per this project's established principle
of keeping mistral-specific engineering out of the shared
`extract_claude.py` file.

### Fix 1: `date_present`-gated schema -- CONFIRMED WORKING, kept

**Problem this responds to:** mistral fabricated a plausible-looking date
(`"10/06/2024"`) on `sample 2.jpeg` (a diary page with no real date, only
a "MONDAY" label), and reproduced the *identical* fabrication across 3
separate prior tests including with `--think` -- a stubborn default
completion pattern, not a one-off misread.

**What was built:** `MistralExtractedForm` in `extract_ollama_cloud.py`
-- a fully parallel schema (not a subclass of `ExtractedForm`, since
Pydantic appends a subclass's new fields after its parent's inherited
ones, and `date_present` must be generated BEFORE `order_date` for the
gating to mean anything under constrained decoding, which emits JSON
properties in schema-declared order). `date_present: bool` sits
immediately before `order_date: str`; a `model_validator` forces
`order_date` to `""` whenever `date_present` is `False`, regardless of
what string the model actually put there (defense in depth, not reliance
on instruction-following alone). Used only when `"mistral" in
model.lower()`, in `extract_one()`'s main call; the parsed result is
converted back to a plain `ExtractedForm` immediately after parsing
(`_mistral_form_to_extracted_form`), so every function downstream is
unaware the schema ever changed. A matching `DATE PRESENCE` bullet was
also added to `MISTRAL_PROMPT_ADDENDUM_BASE` reinforcing the same rule in
prose.

**Confirmed working, real runs, not inferred:**
- `sample 2.jpeg` run 3 separate times: **`order_date: ""` on all 3
  runs**, zero fabrication -- a clean break from the previous 3-for-3
  fabrication rate with plain prompt wording alone. Item/cell counts held
  at 12 items / 58 cells on every run, matching the already-confirmed-good
  baseline (no regression from the schema change itself).
- Checked the model's own raw `date_present` value directly (not just the
  post-gating result) via an isolated `_call_schema` test: **`False`** --
  confirmed the model itself is now making the correct presence decision,
  not just being silently overridden by the defense-in-depth fallback
  every time.
- `sample 5.jpeg` (has a real date, `30/3/26`) run once: `order_date:
  "30/03/2026"` -- correctly preserved, no false negative. 14 items, 71
  cells, in line with prior good runs. A fix that traded fabrication for
  under-detection would not have been progress; confirmed it didn't.

**Not yet tested:** any other free-form page besides `sample 2.jpeg`;
whether the same `date_present` pattern would help on a GRID-layout form
that also lacks a date (none identified yet in this project's test set).

### Fix 2: automated deskew + CLAHE contrast normalization -- CONFIRMED WORKING on its target case, CONFIRMED to regress other forms, shipped OPT-IN ONLY, off by default

**Problem this responds to:** on `sample 8.jpeg`, mistral bled one row's
values into the row below it (`Casual RNS` picking up `Salma Top
casual`'s trailing values). A prompt-only fix made this WORSE (see "Row-
crops confirmed to fix the row-bleed bug" above). The bleed only actually
stopped when the user supplied a manually deskewed, cleanly-cropped
version of the same photo (see "Row-bleed solved" above) -- a manual,
per-image workaround, not a pipeline step.

**What was built:** `preprocess_for_vlm.py`, a new, fully standalone
module -- NOT imported by `grid.py` or `ocr_cell_read.py`, and neither of
those files touched at all, per the task's explicit scoping (that CV+OCR
path is already validated on `sample 5.jpeg` and untuned new
preprocessing could change its behavior in untested ways).
`preprocess_for_vlm(image_bytes)` reuses `grid.py`'s own
`_estimate_skew_deg` (imported read-only, not reimplemented) for
Hough-transform-based small-angle deskew, then applies CLAHE contrast
normalization on the L channel in LAB space (preserves color balance
while boosting local contrast for an unevenly-lit real photo). Visually
inspected against `sample 8.jpeg` before wiring it in: the output was
level and legible, no visible artifacts.

**Confirmed working on its target case:** re-ran `sample 8.jpeg` (the
raw, original, un-manicured photo -- not the manual crop) through the
mistral path with preprocessing applied. **The specific cross-row
contamination is gone**: `Casual RNS` no longer picks up any of `Salma
Top Casual`'s values (previously bled-in), and vice versa. Some separate,
more ordinary completeness/shift issues remain on both rows (one dropped
value each, a 1-position shift on `Casual RNS`) -- but the specific bug
this fix targeted, values crossing between rows, did not reproduce.

**Confirmed, via direct A/B and regression testing the same day, to
actively damage other forms -- this is the load-bearing finding, not a
footnote:**
- **`sample 7.jpeg`** (previously 5 of 7 rows exact, independently
  confirmed correct): re-ran with preprocessing on, checked cell-by-cell
  against the photo. Nearly every row regressed:
  - `Exoda Plam` gained a spurious extra value (`75:24`) with nothing
    corresponding to it in the real row.
  - `Essa Trend`, `Classy Vest`, `Coold Vest`, `Fair Lady Print` all
    shifted -1 column from their previously-correct position.
  - `Fair Lady Plain` -- a row this project had independently confirmed
    as an EXACT match on a prior run -- now shows both a -1 shift AND a
    new digit misread (`55` instead of the real `35`).
  - `Image Trunk` came back severely wrong: reported values (`98, 46, 72,
    88, 48, 16`) bear no resemblance to the real row (`16, 20, 20, 16,
    16`).
- **`sample 5.jpeg`**: a genuine, controlled A/B (same prompt, same
  model, only the image differs -- preprocessed vs. the plain
  EXIF-corrected original) showed a rough wash, not a clean win: 8 of 14
  rows byte-identical either way, 2 rows measurably BETTER with
  preprocessing (`Classy W. Vest` RN/RNS shift shrank), 3 rows measurably
  WORSE (`Exoda Trunk`, `Fairlady Print`, `Fairlady Plain` all gained an
  extra column of shift they didn't have without preprocessing), 1 row
  neutral (only the already-known-ambiguous MYNA digit changed, which
  varies every run regardless of any change made this session).

**Decision: kept as real, working code, but changed from automatic
(every mistral call) to an explicit opt-in flag, `--preprocess-image`,
off by default.** This mirrors the exact lesson from the `sample 12`
column-shift bullets earlier in this same file: a fix earning back its
own target case is not evidence it's safe as a blanket default, and this
project already reverted a different fix once this session for the same
reason (see the `sample 12` section's "Reverted entirely" note above).
Unlike that case, this fix has a real, narrow, identifiable use (a
specific image already known to have a row-bleed problem), so it's kept
available rather than deleted outright -- `extract_one(..., do_preprocess:
bool = False)`, wired through `--preprocess-image` in `main()`.

**Not yet tested:** whether deskew alone (without CLAHE) would keep the
`sample 8.jpeg` fix without the `sample 7.jpeg`/`sample 5.jpeg` damage --
the two transforms were built and tested together, not in isolation, so
it's not yet known which one (or whether it's the combination) is
responsible for the regressions. If revisited, isolating them is the
obvious next step before concluding the whole approach is unsafe.
Recount path (`_recount_quantities`) and the whole-table CV fallback
(`_select_cv_row_boundaries`) were deliberately left untouched by this
flag -- only the main call's image prep is affected, per the task's own
scoping to "wherever mistral currently receives raw images" as tested.

## Hybrid OCR-coordinates + VLM quantities — `--hybrid-quantities`, built
## and integrated 2026-08-17, real but incomplete win

Follows directly from the coordinate experiment above. Two things were
tested, in order, both against `sample 12-scanned.jpg`:

**1. Does the model ground itself if asked to GENERATE its own
coordinates?** No -- tested directly by asking mistral to self-report an
`x_fraction` per header and per quantity mark. Its header positions came
back as a suspiciously perfect uniform arithmetic sequence (`0.28, 0.32,
0.36, 0.40, 0.44...`, spaced ~0.03-0.04 apart across all 17 headers) --
not a real measurement, the same "clean arithmetic sequence" tell already
documented elsewhere in this file for hallucinated header data. Backing
out which header each mark's self-reported coordinate was nearest to
matched the true header on 0 of 10 rows, and several were **less
coherent** than the label-based guesses ever were -- e.g. `Good Mini
Assam`'s derived headers (`60, 73, 75, 85, 95`) aren't even a contiguous
block, where every real row on this form is. The model also broke
structured-output mode entirely for this task (returned markdown prose,
not JSON) -- a second signal this is a harder, less-confident task for it
than direct labeling.

**2. Does the model ground itself if given REAL coordinates (from OCR)
and asked only to do nearest-neighbor matching?** Yes, confirmed
strongly, twice, in isolated hand-curated tests: PaddleOCR's own detected
header and quantity-mark positions were fed to mistral as plain text
alongside the image, with instructions to group each mark with its
nearest header x-position. Result: **47 of 48 tested marks exactly
correct (~98%)**, including full-row exact matches on rows that had been
off by 8-9 header positions under every VLM-only approach tried this
session (`ESSA Premium RA`: `80:5, 85:30, 90:30, 95:6` exactly; `TA22
Short`: all 7 values including the trailing one every prompt-only attempt
had dropped). The one error was a single nearest-neighbor slip on one
mark (grouped with a header 26px away instead of one 9px away), not a
systematic bias -- a fundamentally different, more tractable failure
shape than "defaults to a wrong column regardless of truth." This
directly overturned an earlier assumption in this file that a model would
treat given coordinates the same unreliable way it treats its own
self-reported ones -- it doesn't; the failure is specifically in
*generating* a spatial judgment from scratch, not in doing arithmetic on
numbers it's handed.

**Built into the pipeline**: `_hybrid_ocr_quantities()` in
`extract_ollama_cloud.py`, gated behind `--hybrid-quantities` (off by
default, experimental). One whole-image PaddleOCR pass (no per-row crops,
no `grid.py` row-boundary detection) -- deliberately sidesteps both root
causes behind the earlier `--cv-quantities` revert (PaddleOCR's
inconsistent internal resize on stitched crops; `grid.py`'s row-boundary
detection getting confused by an extra letterhead row on some forms).
Detected header/mark coordinates are handed to mistral as plain text
alongside the image; response is parsed as plain text (`ROW n: size:qty,
...`), not JSON, matching the same schema-compliance finding from the
isolated test above.

**Real, end-to-end integration surfaced two further bugs the isolated
test never exercised -- both found and fixed the same day:**
1. **Decoy sub-header contamination.** `sample 12-scanned.jpg` has not
   one but two stacked decoy lines directly below the real size headers
   (an age/chest-equivalent number, and separately letter clothing
   sizes) -- their gap to the real first data row is only ~2px, too thin
   for any fixed pixel margin to reliably separate. A first fix attempt
   used `extracted.table_top_frac` as an extra exclusion floor, reasoning
   it was the main call's own judgment of where real data starts --
   **confirmed wrong by direct measurement**: `table_top_frac` is defined
   as the TOP of the header, not the bottom of the last decoy line, so it
   sat well above the decoy rows entirely and excluded nothing. Fixed by
   detecting the decoy band directly from the OCR data's own shape
   instead of trusting any VLM self-report: scan y-bands below the real
   header row, and treat any band whose detections cover at least half
   the header x-positions as another decoy sub-header line (real data
   rows are sparse by comparison) -- confirmed via direct measurement
   this correctly found both stacked decoy lines and pushed the floor to
   314px, 2px before the real data starts at 316px.
2. **Row misassignment via unreliable row_top_frac/row_bottom_frac.**
   Even after fix #1, row 1's data was still landing on the wrong item --
   traced to item 0's own `row_top_frac`/`row_bottom_frac` center being
   biased early enough that item 1's center was numerically CLOSER to row
   1's real data than item 0's own center was, so per-cluster nearest-
   center matching picked the wrong item. This is the same self-reported-
   fraction unreliability already documented elsewhere in this file for
   building row crops -- now confirmed to affect row *identification* via
   coordinate proximity too, not just crop boundaries. Fixed by reusing
   the SAME order-preserving DP already proven for digit-to-header
   assignment in `ocr_cell_read.py` (`_monotonic_assign`) to match
   y-clustered OCR mark groups to items jointly, instead of matching each
   cluster to its nearest item independently.

**Confirmed after both fixes: row 1 is no longer contaminated or
misassigned** (`ESSA Premium` now reads real header values like `80:5,
85:30`, not the decoy row's `14, 16, 18...`), and a full run assigns all
10/10 rows via the hybrid pass with zero rows falling back to the model's
own reading. **But per-row completeness in this real end-to-end run still
falls short of the isolated test's 98% figure** -- several rows now
return fewer values than their real cell count (e.g. `ESSA Premium`
returned 3 values where 5 are expected), not yet root-caused. Given the
scope of debugging already done this session, this was left as a known,
documented gap rather than chased further -- the CLI help text and the
function's own docstring were both corrected to state the isolated-test
number and the integration gap separately, specifically so this doesn't
get miscited later as "the hybrid pass gets 98% end-to-end," which is not
yet a confirmed claim.

**Not yet done:** root-causing the completeness gap between the isolated
test and the real pipeline run (candidate filtering, OCR detection
variance, or something else -- not yet isolated); any regression check on
ESSA-family forms (`sample 5.jpeg` was run once with `--hybrid-quantities`
and confirmed NOT to crash, 14 items / 67 cells, but no cell-by-cell
accuracy comparison was done -- unlike `--preprocess-image`, which got a
real regression check and failed it, this flag's regression risk on the
project's priority form template is genuinely unknown, not confirmed
either way); the merged-OCR-token case (`"3030"` from two adjacent "30"s
glued into one detection) is currently just dropped via a `qty > 500`
sanity cap rather than correctly re-split into two values.

## Fixed-band + visual-anchor experiments — 2026-08-18

Two more distinct approaches to mistral's column-position problem, tried
in a separate session (`experiment_fixed_bands.py`,
`experiment_visual_anchors.py`) and left with results generated but never
written up here -- captured now from the actual recorded numbers rather
than left to be rediscovered from scratch. Both tested on `sample 5.jpeg`
(71 real size/qty cells), using `RowQuantityReading`'s categorical
`first_size`/`last_size` + labeled-quantity shape (imported unmodified
from `extract_claude.py`), not continuous coordinates.

**Fixed bands: does isolating the model's view to a small, fixed,
deterministic vertical slice of the table (NOT VLM-reported boundaries --
manually computed pixel fractions) improve column assignment?** Tested
progressively smaller bands (whole table -> ~7 rows -> ~4 rows -> ~2
rows) plus a text-header variant. Real, measured result across every
config:

| config | value_recall | exact_cell_accuracy | exact cells |
|---|---|---|---|
| table_only (whole table, 1 call) | 0.69 | **0.549** | 39/71 |
| large_7row (3 bands) | 0.69 | 0.549 | 39/71 |
| medium_4row (6 bands) | **0.873** | 0.451 | 32/71 |
| small_2row (13 bands) | 0.831 | 0.225 | 16/71 |
| medium_4row_textheader | 0.761 | 0.07 | 5/71 |

**Confirmed, not just theorized: shrinking visual context monotonically
HURT column assignment (0.549 -> 0.451 -> 0.225) even though it modestly
helped raw digit recall (0.69 -> 0.873 peak).** Inspecting raw output
confirmed the mechanism: small crops didn't just add mild drift, they lost
track of which section of the header applied at all (e.g. `MYNA`'s real
`80-100` range reported as `55-75` in one band). This is the same
"narrower context helps counting, hurts position" pattern already
documented elsewhere in this file (the mechanical per-column-enumeration
attempt on `sample 12` had an analogous effect) -- another independent
confirmation that shrinking what the model sees is not a viable lever for
this specific weakness, whatever form the narrowing takes. `large_7row`'s
3 overlapping bands also showed 0/6 agreement on their overlapping rows --
a reliability red flag on top of the accuracy numbers.

**Visual anchors: if fixed bands don't help, does making column
boundaries explicit IN THE PIXELS (drawn directly on the full,
un-cropped image) help instead?** Bold vertical lines at every real
column boundary, alternating pale per-column tint, tested against a
plain-prompt control -- all on the SAME full-table image, single call.
Real, measured result:

| config | value_recall | exact_cell_accuracy | exact cells |
|---|---|---|---|
| lines + tint | 0.732 | 0.0 | 0/71 |
| lines only | 0.732 | 0.0 | 0/71 |
| tint only | 0.732 | 0.0 | 0/71 |
| control (plain prompt, no drawing) | 0.718 | 0.07 | 5/71 |

**Inconclusive on the anchors themselves, but a real, useful data point
regardless.** All four numbers, INCLUDING the control, are far worse than
`table_only`'s 0.549 from the fixed-bands experiment run the same
session on the same image -- meaning something about this particular
run's conditions (not necessarily the drawn annotations) produced a much
more severe column shift than usual. Checked the raw output directly
(`raw_control_plain_prompt.txt`): the model's read values are almost all
individually correct in sequence (e.g. `Trend Trunk`: `80:50, 85:90,
90:68, 95:30, 100:50` -- an exact match) but the WHOLE FORM is shifted by
a consistent -1 header position on nearly every other row (`Exoda Trunk`
reported starting at `75` instead of the true `80`), which zeroes out
`exact_cell_accuracy` almost completely even though the underlying read
is mostly right -- a real illustration of how unforgiving a strict
exact-position metric is against a uniform shift, not evidence the read
was actually garbage. Since the CONTROL (no visual annotation at all)
failed the same way as all three annotated variants, this run cannot
cleanly isolate whether the lines/tint helped, hurt, or did nothing --
confirmed only that whatever changed between this session's draw and the
fixed-bands session's `table_only` draw (both nominally the same
config) caused a much worse shift that session. **Not re-run to
disambiguate; treat as an open, not a closed, question.**

**Cleaned up 2026-08-19**: `_band_probe/`, `experiment_fixed_bands_out/`,
and `experiment_visual_anchors_out/` (calibration PNGs and raw per-call
output) removed per this project's established practice once findings
are captured in prose above -- the two experiment scripts themselves were
also removed, being one-off diagnostic tools specific to a single
completed comparison, not reusable pipeline components.

## Hybrid OCR+VLM quantities revisited — 2026-08-19: shift vs.
## completeness are different problems, olmOCR2 ruled out as a fix

Follow-up session to the `--hybrid-quantities` work above, using the
version already integrated (decoy-band detection + `_monotonic_assign`
row clustering, both already in the codebase).

**Precise breakdown of the final `sample 12-scanned.jpg` run, separating
"wrong column" from "missing entirely"** -- the earlier write-up above
only reported aggregate cell counts, which blurred these two distinct
failure modes together:

- **28 of 45 real values (62%) exactly correct** -- right value, right
  header, no shift.
- **8 of 45 (18%) present but shifted exactly -1 header position** --
  and specifically ALWAYS the later values in an otherwise-correct row,
  never the earlier ones (e.g. `ESSA Premium`: `80:5` and `85:30` both
  exactly right, but a `90:6` appears that is really the true `95:6`
  value one column early). Confirmed this is a small, consistent drift,
  NOT a recurrence of the severe 8-9-position defaulting-to-the-left
  failure this feature was built to fix -- that failure mode is
  confirmed gone from every row of this run.
- **9 of 45 (20%) missing entirely** -- not mislabeled, just absent from
  the output, consistent with the candidate-detection completeness gap
  already flagged (OCR score threshold, x/y boundary filtering, or the
  merged-token drop all plausible causes, not yet isolated).

**Root-cause hypothesis, not yet confirmed:** the front-correct/tail-
shifted pattern is consistent with the SAME row's mark set being
incomplete (one real digit undetected) rather than a separate shift bug
-- if the "middle" of a row's sequence goes missing, the surviving later
marks could plausibly land nearest to a different header than they would
in a complete set. Not verified against the actual per-row candidate
lists; the two problems (missing values, tail shift) may collapse into
one root cause once debugged, or may not.

**olmOCR2 tested as an alternative to PaddleOCR for the candidate-
detection step -- ruled out, confirmed worse, not better.** Given
PaddleOCR's detection completeness was flagged as the likely source of
the missing-values problem, tried swapping in `richardyoung/olmocr2:7b-q8`
(the local, document-OCR-specialized model already used elsewhere in this
project) as a full-page table transcriber, to see if a model actually
trained for document transcription would read the table more completely.
Real result, direct full-page call against `sample 12-scanned.jpg`
(194.5s, `done_reason=stop`): **9 of 10 rows came back completely blank**
-- no quantity marks transcribed at all, not misread, just absent. **Row
1 reproduced the exact same decoy-sub-header contamination bug** already
found and fixed in this file's own candidate-detection code -- olmOCR2
copied the form's second (age-equivalent) header row (`14, 16, 18, 20,
22...`) directly into the quantity cells for that row, the same failure
mode via a third, independent mechanism (after the local qwen2.5vl
pipeline years ago, and mistral's self-reported-coordinate experiment).
Only one row (`TA22 Short`, misread as `"JA22 SHORT"`) produced real
data, and even that was shifted -1 column. **Conclusion: this form's
doubled decoy-header structure is hard for every mechanism tried so far,
not specifically a PaddleOCR weakness** -- olmOCR2 is not a viable
candidate-detection replacement here, and this was not pursued further.

**Not yet done:** isolating the real cause of the missing-values gap by
inspecting the actual per-row OCR candidate lists (which real marks were
detected vs. dropped, and at which filtering stage); re-segmenting merged
OCR tokens (`"3030"`) instead of discarding them; any accuracy comparison
on ESSA-family forms with `--hybrid-quantities` enabled (only a
crash-safety check has been done, per the section above).

## Future suggestions to try — 2026-08-19

Concrete next steps, each grounded in something already confirmed in this
file rather than a fresh guess -- ranked roughly by how directly they
follow from evidence already in hand.

1. **Finish root-causing the hybrid pass's missing-values gap before
   anything else.** This is the most scoped, most likely-to-pay-off open
   item: pull the real per-row OCR candidate list for one of the
   incomplete `sample 12` rows and check, mark by mark, whether it was
   dropped by the `score >= 0.5` threshold, the x/y boundary filters, or
   never detected by PaddleOCR at all. Unlike every column-shift fix
   attempted on the model side, this is deterministic code -- a fix here
   is verifiably correct once found, not "maybe it behaves differently
   next run."

2. **Re-segment merged OCR tokens instead of discarding them.** The
   `"3030"` case (two adjacent "30"s glued into one detection) is
   currently just dropped via the `qty > 500` cap. A real fix -- split the
   box roughly in half by width and re-OCR each half, or use the box's
   aspect ratio directly to flag likely merges before they're even
   considered -- would recover real, currently-lost values rather than
   just failing safely on them.

3. **Feed the hybrid pass's own per-row Total Dozen / printed TOTAL
   checksum back in as a deterministic, code-level correction** for the
   remaining -1-position tail-shift, instead of relying on the model to
   self-correct (already confirmed unreliable -- mistral ignored an
   explicit self-check instruction even when shown its own exact failing
   arithmetic, see the `sample 12` column-shift section above). This
   reuses `brandlist_match.py`'s already-existing `detect_column_shift`
   logic, just applied to the hybrid pass's output instead of raw VLM
   output -- if shifting a row's reported headers by +1 makes it sum to
   the printed total, that's a much stronger, cheaper signal than asking
   the model to notice its own mistake.

4. **Validate `--hybrid-quantities` against the ESSA-family forms
   (`sample 5`, `sample 7`, `sample 4`) before trusting it anywhere near
   default-on.** Only a crash-safety check has been done so far. This
   project has already been burned once this session by skipping exactly
   this step (the `sample 12` column-shift bullets that quietly regressed
   `sample 5`) -- don't repeat that with a structurally different but
   equally new mechanism.

5. **A cleaner, controlled re-run of the visual-anchors idea.** The
   2026-08-18 result was genuinely inconclusive -- the control (no
   drawing at all) failed exactly as badly as every annotated variant,
   which means that session's severe shift wasn't isolated to the
   anchors themselves. Worth a fresh, back-to-back comparison (anchors vs.
   plain, same session, same model state) before concluding either way.

6. **A genuinely new combination worth trying, not yet attempted: bake
   the REAL OCR-detected coordinates directly into the image pixels**
   (small markers at each actual detected header and quantity-mark
   position) instead of sending them as a separate text list. The
   fixed-band experiment showed cropping hurts; the visual-anchor
   experiment (drawing synthetic column *boundary* lines) was
   inconclusive; the hybrid-coordinates work proved the model uses REAL
   coordinates reliably when reasoning about them as text. Combining the
   two -- real coordinates, but shown visually at their actual detected
   position rather than described in text -- hasn't been tried, and
   could plausibly be more reliable than either a text coordinate list or
   synthetic grid lines, since the model would be matching what it sees
   in the image to markers already sitting at the right place, not
   translating a text description into a spatial judgment at all.

7. **Test the hybrid mechanism on an easier form before pushing further
   on `sample 12`.** Every real test of `--hybrid-quantities` so far has
   been on the hardest form in this project's set (dual decoy header
   rows, 17 columns). The "Excel-maintained" typed-grid templates
   (`sample 9`/`13`, per the multi-form-type survey) are a plausible
   ceiling-check -- if the hybrid approach doesn't reach near-100% there
   too, that's evidence the remaining gap is more fundamental than "one
   hard form's decoy rows," worth knowing before investing further in
   `sample 12` specifically.

## Suggestions #1-3 implemented for `--hybrid-quantities` — 2026-08-19, same
## day: code written and unit-tested, NOT YET CONFIRMED against a real
## Ollama Cloud run (no access in this session)

Picked up suggestions #1 (root-cause tooling), #2 (re-segment merged OCR
tokens), and #3 (total-checksum signal) from the "Future suggestions"
list above, all scoped to `extract_ollama_cloud.py`. Per this project's
own verify-before-trusting standard, these should be read as
**implemented and logic-tested, not yet run end-to-end on a real form** --
no Ollama Cloud call was made this session.

**#2 -- merged-OCR-token re-segmentation (`_split_merged_qty_token`),
real fix.** Where the `qty > 500` sanity cap used to just drop a fused
detection (e.g. two handwritten "30"s read as one "3030" token), it now
tries every internal split point and accepts the split ONLY when exactly
one position yields two valid (1-500, no spurious leading zero) integers
-- an ambiguous token (e.g. "255", which splits validly as both `2|55`
and `25|5` with nothing to prefer) still returns `None` and is dropped,
same "leave it out rather than invent a value" principle as before. Each
recovered half gets its own proportional x-center from the original box
so downstream row/column assignment still works on it. Verified by direct
unit tests: `"3030"` -> two `30`s at the correct sub-positions, `"530"`
-> `5`+`30` (uneven split, not just even-halves), `"255"` -> correctly
refused (ambiguous), `"1010"` -> two `10`s. Not yet confirmed this
actually recovers real values on a real form -- only tested against
synthetic token/box inputs.

**#3 -- reused as originally worded, DISPROVEN before shipping, shipped
in a corrected flag-only form instead.** The original suggestion ("shift
a row's reported headers by an offset until the sum matches the printed
total, mirroring `brandlist_match.detect_column_shift`") was implemented
first, then tested directly before trusting it -- and confirmed
mathematically incapable of ever firing: **re-assigning a row's
quantities to different header KEYS while keeping the same VALUES cannot
change their sum** (confirmed directly: `sum({45:5,50:30,55:30,60:4})`
and `sum({50:5,55:30,60:30,65:4})` are both `69`). So a shift-search
against total can only "succeed" when the row's sum already matched the
total before any shift was tried -- i.e., never on the actual target case
(a genuine mismatch). This isn't a new finding; it's this same file's own
2026-08-06 "Column-shift detection" section restated ("a shift relabels
values, it doesn't change their total") -- the 2026-08-19 suggestion
re-introduced a claim the project had already disproven once, and it
went unchecked against that earlier section when written. **Replaced**
with `_flag_hybrid_total_mismatch()`: still uses each row's own printed
running total (via `ExtractedItem.row_total`, parsed with
`extract_claude._parse_int_or_none`), but only surfaces a note ("sums to
X, but this row's own printed total is Y -- likely a missed or extra
value, not auto-corrected") for human review, matching the project's
existing "flag, don't auto-correct" pattern for lower-confidence signals
(e.g. `brandlist_match.sizes_outside_catalog_range`) rather than
attempting a fix that can't work. A genuine mismatch here most likely
means a value went undetected entirely -- the already-documented
"18% missing, tail shifts into the gap" pattern -- which isn't fixable by
relabeling only the values that WERE detected, since the true fix needs
a value that was never read at all.

**#1 -- root-cause tooling, not a root cause itself.** Rather than
guessing at the missing-values gap (isolated test ~98% vs. real
end-to-end runs falling short, still unexplained), `_hybrid_ocr_quantities`
now writes `<outdir>/<name>.hybrid_debug.json` on every call: every
OCR-detected candidate mark considered in the table area (post score/x/y
filtering, pre row-clustering) plus the final per-row candidate list
actually sent to the model. This makes it possible to trace a specific
missing value to one of three places -- OCR never detected it, it was
dropped by the score/x/y-range filters, or it was detected and sent but
the model's own response for that row just omitted it -- instead of
re-guessing from final output alone. **This is instrumentation, not a
fix** -- the missing-values root cause itself is still open, needs a real
run with this debug file inspected against the source photo.

**Not yet done, next actual step:** run `extract_ollama_cloud.py
Images/"sample 12-scanned.jpg" --hybrid-quantities` for real, inspect the
new `.hybrid_debug.json` against the photo to finally root-cause the
missing-values gap, and separately confirm `_split_merged_qty_token` and
`_flag_hybrid_total_mismatch` behave as intended on real OCR output (not
just the synthetic unit tests above). Also still not done: suggestion #4
(regression-check against the ESSA-family forms) and #7 (test on an
easier form) from the list above -- neither attempted this session.

## Missing-values gap partially root-caused via `.hybrid_debug.json` -- a
## real fix, confirmed by a live re-run, same day

Ran `extract_ollama_cloud.py "Images/sample 12-scanned.jpg" --outdir
extracted_ollama_cloud_hybrid_test --model mistral-large-3:675b
--no-recount --hybrid-quantities` for real (an `OLLAMA_API_KEY` was
present in `.env` after all -- corrected an earlier assumption in this
session that no Cloud access was available). This is the first real,
end-to-end confirmation of anything in the "Suggestions #1-3 implemented"
section immediately above.

**Root cause of a real chunk of the missing-values gap, found directly
from the new `.hybrid_debug.json` artifact, not guessed:** for row 0
(`ESSA Premium`), OCR-detected candidates summed to only 41 against a
printed total of 74 -- roughly HALF the row's real value was gone. Cross-
referencing `header_x`'s keys against `extracted.size_headers` (from
`<name>.stageA.json`) showed the header dict was missing `100` and `105`
entirely, even though both are real headers on this form and OCR found
15 of the other 17 fine. Directly re-running raw PaddleOCR against this
image and inspecting every detected token near where those two headers
should sit found the cause: **PaddleOCR fused the adjacent "100" and
"105" header labels into one single detection, text `"100105"`** (box
738-807px, score 0.9999) -- since header matching does an exact `t in
size_headers` string comparison, neither individual header ever entered
`header_x`, silently losing two whole columns' worth of coordinate ground
truth. Any real mark meant for those columns then had no correct header
to be near, and got misassigned to (or silently overwrote, via the flat
`{size: qty}` dict) a neighboring column instead -- this is the exact
same "PaddleOCR merges two adjacent handwritten numbers into one
detection" phenomenon already documented and fixed for QUANTITY marks
via `_split_merged_qty_token`, just newly confirmed to also hit HEADER
labels.

**Fix: `_split_merged_header_token()`, wired in as a second detection
pass** over the header row's own y-band (bounded to the already-found
headers' y-range plus a small margin, specifically so this can't
misfire on two adjacent quantity digits deep in the table -- only text
sitting where real headers already are is considered). Unlike the
quantity-token splitter (any numeric split point is a candidate), this
only accepts a partition into substrings that are THEMSELVES members of
`size_headers` -- a much tighter, less ambiguous constraint since headers
are a small known set. Confirmed via direct unit test:
`_split_merged_header_token("100105", box, size_headers)` returns exactly
`[("100", 755.25), ("105", 789.75)]`, with the ambiguous/no-match cases
(an already-known header, an unrelated digit string) correctly returning
`None`.

**Confirmed via a real, fresh re-run (not just the unit test) that this
is a genuine improvement, not just plausible-sounding:**
- Header coverage: 15/17 -> **17/17** (both `100` and `105` now present
  in `header_x`).
- Rows flagged by `_flag_hybrid_total_mismatch` (the #3 flag-only fix
  above): **7 of 10 -> 3 of 10.** `ESSA Premium PNS`, `Goodlooking Mini
  Adult`, `Cold Packed`, `Cold RN White`, and `AR Ptd 4150` all went from
  flagged-mismatched to sum-matching-their-printed-total, and each now
  correctly carries a real `"100"` value it was missing before (e.g.
  `AR Ptd 4150`: `{'80':2,'85':12,'90':12,'95':4,'100':2}` where the
  `100:2` didn't exist in the output at all pre-fix).
- Total quantity cells extracted: **37 -> 43** across the same 10-row form.
- Row 0 itself (the row that surfaced this) improved from a 33-unit gap
  (41 vs. printed 74) to a 3-unit gap (71 vs. 74) -- not fully closed
  (still flagged, correctly, for human review), but the dominant chunk of
  its error is gone.

**Still open, confirmed genuinely separate issues (not touched by this
fix):** `TA22 Short` (headers 45-75, no 100/105 involved) still flagged,
sum 94 vs. printed total 81/79 (varies slightly run to run, since
`row_total` is itself a fresh VLM read each call) -- 7 distinct,
non-colliding candidate marks were found, all mapped to different real
headers, so this isn't a header-merge or collision case; more likely a
genuine digit misread on one or more marks, a separate residual problem
correctly left flagged rather than silently shipped. `Ladies Drawers`
similarly still short by 2 (6 vs. 8) with no header-merge explanation
found. Both are honestly surfaced via the total-mismatch flag rather than
resolved.

**Not yet done:** whether the SAME merged-header-token bug recurs on
other forms/runs (only checked on this one image, this one run -- header
merging is presumably sensitive to exact spacing/font, not guaranteed to
reproduce identically); root-causing `TA22 Short`'s and `Ladies
Drawers`'s remaining mismatches; confirming `_split_merged_qty_token`
(the quantity-token counterpart) actually fired on this same real run
(not explicitly checked this session -- the header-token bug turned out
to be the dominant, most legible finding, so this wasn't separately
isolated); suggestion #4 (regression-check against the ESSA-family forms)
and #7 (test on an easier form) from the list above -- neither attempted
this session. Scratch outdir `extracted_ollama_cloud_hybrid_test/` from
this session's two test runs removed after this write-up, per this
project's established cleanup practice.
