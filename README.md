# Order Form Extraction

Turns a photo of a handwritten/printed garment order form into
structured JSON (party name, order no./date, and a size→quantity table
per item), ready for a quick human review before it goes into the order
database.

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
for a single structured extraction call. A local OCR pass then re-checks
the quantity table against the image's own measured pixel positions
(catches the column-drift errors a vision model makes on dense tables),
and a local product-catalog lookup cross-checks item names, style codes,
and sizes against the real database. Nothing about this needs a GPU or a
local model — the only local computation is OCR and a SQL lookup.

See [CLAUDE.md](CLAUDE.md) for the full pipeline architecture, CLI
defaults, and known limitations, and [HISTORY.md](HISTORY.md) for the
complete development history (every model and approach that was tried
along the way).

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
SERVER=...               # optional -- enables the product-catalog cross-check
DB=...                    # (all four required together, or all omitted)
USER=...
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
debugging artifacts and a flattened `review.csv` into `--outdir`. Open
the generated `<name>.review.html` in a browser to check the extracted
grid against the source photo side by side, edit any wrong cell, and
export corrected JSON before it goes to the database.
