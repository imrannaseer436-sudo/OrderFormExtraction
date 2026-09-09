# Order Form Extraction

Turns a photo of a handwritten/printed garment order form into
structured JSON (party name, order no./date, and a size→quantity table
per item), reviews it against the photo in a browser, and writes the
finished order straight into the order database.

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

## How it works

One photo goes to `mistral-large-3:675b` on [Ollama Cloud](https://ollama.com)
for a single structured extraction call. A second, local re-read of the
quantity table (a small vision model, or OCR) then cross-checks it
against what the cloud call reported (catches the column-drift errors a
vision model makes on dense tables), and a local product-catalog lookup
cross-checks item names, style codes, and sizes against the real
database. A fully local alternative also exists (`--model chandra`, or
the "Chandra" option in the review app) for when the cloud call isn't
wanted at all — slower, but needs no API key and nothing leaves the
machine; see CLAUDE.md's "Alternative model: Chandra" section.

See [CLAUDE.md](CLAUDE.md) for the full pipeline architecture, CLI
defaults, and known limitations, and [HISTORY.md](HISTORY.md) for the
complete development history (every model and approach that was tried
along the way).

## The review app

`run_ui.py` is how orders actually get entered. It serves a browser app
that takes the photos, runs them through the pipeline, and shows the
extracted grid beside the photo for a human to correct — product and
buyer pickers with search, one-click repairs for the pipeline's known
failure modes (a row read one column off, a block of rows attributed one
row off, a product whose real sizes aren't on the printed grid), and a
frozen size-header row and item column so nothing scrolls out of view.
When it's right, one button writes the `OrderMaster` / `OrderDetails`
rows. Nothing reaches the database before that click.

```powershell
.venv\Scripts\python.exe run_ui.py
```

It binds `0.0.0.0`, and prints both a `localhost` URL and a LAN URL — open
the LAN one on a phone to photograph a form and upload it directly, then
review it on a desktop. Several photos can belong to one order; they're
reviewed together and uploaded as a single order.

The CLI below is still the way to run the pipeline on its own — for
batch extraction, for debugging a form, or when there's no database to
upload to.

## Setup

Requires Python 3.11 (not 3.13+ — `paddlepaddle` has no newer wheels
yet). Use a dedicated virtualenv, not your system Python:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OLLAMA_API_KEY=...      # required -- ollama.com/settings/keys
SERVER=...               # required for the review app; optional for the CLI,
DB=...                    # where it only enables the catalog cross-check.
USER=...                   # (all four required together, or all omitted)
PASSWORD=...
```

## Usage

```powershell
# one image
.venv\Scripts\python.exe extract_ollama_cloud.py "Images\sample 5.jpeg"

# a folder of images, into one output directory
.venv\Scripts\python.exe extract_ollama_cloud.py Images\ --outdir extracted_ollama_cloud

# generate a human review page for one already-extracted image
.venv\Scripts\python.exe generate_review.py "sample 5" --outdir extracted_ollama_cloud --images Images
```

Each run writes `<name>.json` (the final structured output) plus several
debugging artifacts and a flattened `review.csv` into `--outdir`.
`generate_review.py` turns one of those into a standalone HTML page that
exports corrected JSON — useful for checking a single form offline, but
the review app above is what feeds the database.
