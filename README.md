# Order Form Extractor — Ollama v3 (per-row extraction)

## Why this version exists

v2's single call tried to read all 17 rows × 13 columns of your sample
form at once. Even after explicitly forcing "blank" markers for empty
cells, the model still drifted — e.g. it counted "14 size columns" when
there are only 13, and that single miscounted header shifted every
quantity in the row one column off. This is a genuine limit on how much
positional state a 7B model can reliably track in one pass, not something
prompt wording alone fixes.

## What changed

**Stage A** (1 call, schema-constrained): party info, size headers, and
the list of item rows (name + style code only — no numbers). This part
has been reliable in every attempt so far, so it stays a single call.

**Stage B** (1 call PER item row, unconstrained): for each row identified
in Stage A, a separate call shows the image again and asks the model to
focus on *only that one row* and read across its columns. Much less to
track per call = much less room for drift. The response format
(`45:blank, 50:blank, ..., 80:50, ...`) is simple enough to parse with a
regex in Python — no third "convert to JSON" model call needed, which
also removes a place where errors could creep in.

**Trade-off:** more calls = slower. A 17-row form now makes ~18 model
calls instead of 2. Worth it if accuracy improves — check the output
before deciding it's worth the wait.

## Setup / usage — unchanged

```powershell
py -m pip install -r requirements.txt
py .\extract_ollama.py "images/sample 5.jpeg"
py .\extract_ollama.py images\ --outdir extracted
```

## Output per image (new debugging files)

- `extracted/<name>.json` — final structured data, same shape as before
- `extracted/<name>.stageA.json` — structure-only output (party info,
  headers, item list). Check this first — if item names or headers are
  wrong here, the problem is upstream of the per-row step.
- `extracted/<name>.rows.txt` — **the raw response for every single row**,
  one line each (`item | style -> raw model output`). This is the most
  useful debugging file now: if one item's quantities look wrong in the
  final JSON, find its line here and see exactly what the model said for
  that row before parsing.
- `extracted/review.csv` — one row per (item × size), for review

## If this still isn't accurate enough

At this point, further prompt tweaking has diminishing returns. Next
options, roughly in order of effort:

1. **Check `.rows.txt` first** for whichever items are wrong — this
   tells you if the per-row read itself is wrong (still a vision
   limitation) or if the regex parser mis-parsed a correctly-read line
   (a bug I'd want to know about).
2. **Try `qwen2.5vl:32b`** if your hardware supports it — meaningfully
   better raw visual grounding than 7B on dense tables.
3. **Crop the image before sending it**, if your company's forms are
   consistently laid out (e.g. all ESSA forms have the same blank
   leading columns) — removing the empty columns entirely removes the
   counting problem by construction. I can help build this if useful.
4. **Hybrid routing**: send only the hardest forms (e.g. ones where
   Stage B disagrees with itself on retry, or forms flagged as dense
   handwriting) to the Claude API pipeline, and keep everything else on
   the free local path — a middle ground between cost and accuracy.
