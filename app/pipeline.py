"""Session + extraction orchestration behind the review UI.

A *session* is one order: one or more uploaded photos that will become one
OrderMaster row. Each photo is run through the existing production pipeline
(`extract_ollama_cloud.extract_one`) exactly as the CLI runs it -- this
module deliberately calls that function rather than reimplementing any
stage, so the UI can never drift from the pipeline the accuracy work in
HISTORY.md was done against.

What this module adds on top is the part the CLI has no use for: turning
the pipeline's output files (<name>.json, .stageA.json, .recount_flags.json,
.brandlist.json, .party_check.json) into a single review payload the browser
can render, with every row pre-bound -- where the catalog match is
trustworthy -- to the concrete `bsid` product group that OrderDetails will
ultimately need.

Extraction runs on a background thread and the payload is rebuilt after each
image finishes, so a multi-page order is reviewable page by page instead of
only when the whole batch is done.
"""

from __future__ import annotations

import json
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from . import db

REPO_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_ROOT = REPO_ROOT / "ui_sessions"
DISPLAY_MAX_WIDTH = 2600          # plenty for reading handwriting at 3-4x zoom,
                                   # small enough that a phone photo loads fast
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# One extraction at a time, process-wide. Each image already runs a cloud VLM
# call plus a full PaddleOCR pass; running two concurrently competes for the
# same OCR model instance and gains nothing but memory pressure.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="extract")

_engine_lock = threading.Lock()
_engine: dict[str, Any] | None = None


# ------------------------------------------------------------------ engine

def get_engine(model: str | None = None) -> dict[str, Any]:
    """Lazily imports the pipeline and builds the pieces `extract_one` needs.

    Import is deferred to first use because extract_ollama_cloud pulls in
    paddleocr/paddlepaddle, which take several seconds and a lot of memory --
    the web server should start instantly and only pay that once someone
    actually uploads a form.
    """
    global _engine
    with _engine_lock:
        # "chandra" is a shorthand expanded to the real model tag below
        # (CHANDRA_MODEL) -- compared loosely here too, so a session
        # re-requesting "chandra" on every image doesn't rebuild the
        # engine (client + system prompt) from scratch each time.
        cached_matches = _engine is not None and (
            model is None
            or _engine["model"] == model
            or (model.lower() == "chandra" and "chandra" in _engine["model"].lower())
        )
        if cached_matches:
            return _engine

        import extract_ollama_cloud as pipe   # heavy: paddleocr, cv2, ollama
        import brandlist_match

        style_codes: list[str] = []
        known_sizes: list[int] = []
        brandlist_available = False
        try:
            style_codes = brandlist_match.known_style_codes()
            known_sizes = brandlist_match.known_numeric_sizes()
            brandlist_available = True
        except Exception:
            pass   # same degradation the CLI has: run without the catalog cross-check

        model = model or pipe.DEFAULT_MODEL
        if model.lower() == "chandra":
            model = pipe.CHANDRA_MODEL
        system_prompt = pipe.build_system_prompt(style_codes)
        if "mistral" in model.lower():
            system_prompt += pipe.MISTRAL_PROMPT_ADDENDUM_BASE + pipe._sizes_past_header_bullet(known_sizes)

        _engine = {
            "pipe": pipe,
            "brandlist_match": brandlist_match,
            "client": pipe.get_client(model),
            "model": model,
            "system_prompt": system_prompt,
            "brandlist_available": brandlist_available,
        }
        return _engine


# ----------------------------------------------------------------- session

@dataclass
class ImageJob:
    index: int
    name: str                       # stem used for every output file
    original_name: str
    path: Path
    display_path: Path
    status: str = "queued"          # queued | running | done | error
    message: str = ""
    started_at: str = ""
    finished_at: str = ""

    def as_json(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "original_name": self.original_name,
            "status": self.status,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class Session:
    sid: str
    created_at: str
    dir: Path
    images: list[ImageJob] = field(default_factory=list)
    status: str = "idle"            # idle | processing | ready | error
    payload: dict = field(default_factory=dict)
    submitted: dict | None = None
    model: str | None = None        # None -> pipe.DEFAULT_MODEL, see get_engine()
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def outdir(self) -> Path:
        return self.dir / "extracted"

    def as_json(self) -> dict:
        return {
            "sid": self.sid,
            "created_at": self.created_at,
            "status": self.status,
            "images": [i.as_json() for i in self.images],
            "payload": self.payload,
            "submitted": self.submitted,
            "model": self.model,
        }


_sessions: dict[str, Session] = {}
_sessions_lock = threading.Lock()


def create_session(model: str | None = None) -> Session:
    sid = uuid.uuid4().hex[:12]
    sdir = SESSIONS_ROOT / sid
    (sdir / "extracted").mkdir(parents=True, exist_ok=True)
    session = Session(sid=sid, created_at=datetime.now().isoformat(timespec="seconds"), dir=sdir, model=model or None)
    with _sessions_lock:
        _sessions[sid] = session
    return session


def get_session(sid: str) -> Session | None:
    with _sessions_lock:
        return _sessions.get(sid)


def _safe_stem(name: str, taken: set[str]) -> str:
    stem = "".join(c if (c.isalnum() or c in " -_") else "_" for c in Path(name).stem).strip() or "page"
    candidate, n = stem, 1
    while candidate in taken:
        n += 1
        candidate = f"{stem} ({n})"
    return candidate


def add_images(session: Session, uploads: list[tuple[str, bytes]]) -> list[ImageJob]:
    """Stores uploads and queues them for extraction. Returns the new jobs.

    A display copy is written alongside the original: EXIF-rotated (a phone
    photo is very often stored sideways with an orientation tag the pipeline
    honours but an <img> tag does not, which would otherwise show the
    reviewer a rotated form) and capped in width so the browser stays
    responsive while panning and zooming.
    """
    jobs: list[ImageJob] = []
    with session.lock:
        taken = {i.name for i in session.images}
        start_index = len(session.images)
        for offset, (filename, data) in enumerate(uploads):
            suffix = Path(filename).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue
            stem = _safe_stem(filename, taken)
            taken.add(stem)
            path = session.dir / f"{stem}{suffix}"
            path.write_bytes(data)
            display_path = session.dir / f"{stem}.display.jpg"
            _write_display_copy(path, display_path)
            job = ImageJob(index=start_index + offset, name=stem, original_name=filename,
                           path=path, display_path=display_path)
            session.images.append(job)
            jobs.append(job)
        if jobs:
            session.status = "processing"

    for job in jobs:
        _EXECUTOR.submit(_run_job, session.sid, job.name)
    return jobs


def _write_display_copy(src: Path, dest: Path) -> None:
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            if im.width > DISPLAY_MAX_WIDTH:
                ratio = DISPLAY_MAX_WIDTH / im.width
                im = im.resize((DISPLAY_MAX_WIDTH, int(im.height * ratio)), Image.LANCZOS)
            im.save(dest, "JPEG", quality=88, optimize=True)
    except Exception:
        # Unreadable/odd format -- fall back to serving the original bytes.
        shutil.copyfile(src, dest)


def remove_image(session: Session, name: str) -> bool:
    with session.lock:
        job = next((i for i in session.images if i.name == name), None)
        if job is None or job.status == "running":
            return False
        session.images = [i for i in session.images if i.name != name]
        for i, img in enumerate(session.images):
            img.index = i
    _rebuild_payload(session)
    return True


def retry_image(session: Session, name: str) -> bool:
    with session.lock:
        job = next((i for i in session.images if i.name == name), None)
        if job is None or job.status == "running":
            return False
        job.status = "queued"
        job.message = ""
        session.status = "processing"
    _EXECUTOR.submit(_run_job, session.sid, name)
    return True


# --------------------------------------------------------------- extraction

def _run_job(sid: str, name: str) -> None:
    session = get_session(sid)
    if session is None:
        return
    job = next((i for i in session.images if i.name == name), None)
    if job is None:
        return

    job.status = "running"
    job.message = "Reading the form…"
    job.started_at = datetime.now().isoformat(timespec="seconds")
    try:
        engine = get_engine(session.model)
        pipe = engine["pipe"]
        outdir = session.outdir
        outdir.mkdir(parents=True, exist_ok=True)

        form = pipe.extract_one(
            engine["client"], engine["model"], job.path, outdir,
            REPO_ROOT / "usage_log_ollama_cloud.csv", engine["system_prompt"],
            brandlist_available=engine["brandlist_available"],
            do_recount=False, do_preprocess=False, do_hybrid=True,
        )

        # Same post-processing the CLI's main loop does after extract_one --
        # the catalog cross-check and party-name check are separate steps
        # there, not part of extract_one itself.
        if engine["brandlist_available"]:
            bm = engine["brandlist_match"]
            stage_a_path = outdir / f"{name}.stageA.json"
            stage_a = json.loads(stage_a_path.read_text(encoding="utf-8")) if stage_a_path.exists() else {}
            annotations = bm.annotate_and_resolve(form, stage_a.get("size_headers", []))
            (outdir / f"{name}.brandlist.json").write_text(
                json.dumps(annotations, indent=2, ensure_ascii=False), encoding="utf-8")
            pipe._merge_brandlist_flags_into_file(outdir / f"{name}.recount_flags.json", annotations)
            party_check = bm.resolve_party_name(stage_a.get("seller_name", ""), form.party_name)
            if party_check is not None:
                (outdir / f"{name}.party_check.json").write_text(
                    json.dumps(party_check, indent=2, ensure_ascii=False), encoding="utf-8")

        (outdir / f"{name}.json").write_text(
            json.dumps(form.model_dump(mode="json", exclude={"source_file"}), indent=2, ensure_ascii=False),
            encoding="utf-8")

        job.status = "done"
        job.message = f"{len(form.items)} rows, {sum(len(i.quantities) for i in form.items)} filled cells"
    except Exception as exc:
        job.status = "error"
        job.message = str(exc) or exc.__class__.__name__
        traceback.print_exc()
    finally:
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        _rebuild_payload(session)
        with session.lock:
            if any(i.status in ("queued", "running") for i in session.images):
                session.status = "processing"
            elif session.images and all(i.status == "error" for i in session.images):
                session.status = "error"
            else:
                session.status = "ready"


# ------------------------------------------------------------- payload build

def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _rebuild_payload(session: Session) -> None:
    """Rebuilds the whole review payload from the pipeline's output files.

    Rebuilt wholesale (rather than patched) after every image so that
    removing a page, retrying one, or adding a page late all converge on the
    same result -- and so the payload always reflects the files on disk,
    which are the artifacts a developer would debug against.
    """
    pages: list[dict] = []
    party_candidates: list[dict] = []
    party_names: list[str] = []
    order_no, order_date, party_name = "", "", ""

    for job in list(session.images):
        if job.status != "done":
            pages.append({
                "name": job.name, "index": job.index, "status": job.status,
                "message": job.message, "headers": [], "rows": [], "notes": [],
            })
            continue

        outdir = session.outdir
        form = _read_json(outdir / f"{job.name}.json", {}) or {}
        stage_a = _read_json(outdir / f"{job.name}.stageA.json", {}) or {}
        flags = _read_json(outdir / f"{job.name}.recount_flags.json", []) or []
        notes_db = _read_json(outdir / f"{job.name}.brandlist.json", []) or []
        party_check = _read_json(outdir / f"{job.name}.party_check.json", None)

        # The form's own per-row printed total ("Total Dozen" column, or a
        # circled number in the margin) only survives in the raw dump -- the
        # final OrderForm shape has no field for it. It is the single best
        # independent check a reviewer has, so it is carried through and
        # shown beside each row's computed total. Index-aligned with
        # form["items"] by construction (_to_order_form maps 1:1).
        raw = _read_json(outdir / f"{job.name}.raw.json", {}) or {}
        raw_items = raw.get("items", [])

        # Per-template correction memory (template_learning.py): the
        # signature match/apply result extract_one() already computed and
        # wrote for this image. Threaded through unconditionally (None when
        # missing/on a free-form page) -- record_submission() below needs
        # the raw artifact to diff a future correction against this page's
        # TRUE pre-correction reading, and the small "template_learning"
        # summary is what the browser renders as the page-level "seen
        # before" note (see app/static/app.js's applyServerSession()).
        template_match = _read_json(outdir / f"{job.name}.template_learning.json", None)
        template_learning_summary = None
        if template_match and template_match.get("matched"):
            template_learning_summary = {
                "matched": True,
                "occurrences": template_match.get("occurrences", 0),
                "corrections_applied": len(template_match.get("corrections_applied") or []),
            }

        headers = _page_headers(form, stage_a)
        rows = []
        for idx, item in enumerate(form.get("items", [])):
            raw_item = raw_items[idx] if idx < len(raw_items) else {}
            rows.append(_build_row(
                page=job.name, index=idx, item=item,
                flag=flags[idx] if idx < len(flags) else None,
                db_note=notes_db[idx] if idx < len(notes_db) else None,
                printed_total=_parse_total(raw_item.get("row_total")),
                struck_out=bool(raw_item.get("struck_out")),
            ))

        pages.append({
            "name": job.name,
            "index": job.index,
            "status": "done",
            "message": job.message,
            "headers": headers,
            "rows": rows,
            "notes": form.get("notes", []),
            "seller_name": stage_a.get("seller_name", ""),
            "party_check": party_check,
            "raw_size_headers": stage_a.get("size_headers", []),
            "template_match": template_match,
            "template_learning": template_learning_summary,
        })

        order_no = order_no or (form.get("order_no") or "")
        order_date = order_date or (form.get("order_date") or "")
        if form.get("party_name"):
            party_names.append(form["party_name"])
            if not party_name:
                party_name = form["party_name"]
        if party_check and party_check.get("suggested_party_name"):
            party_candidates.append(party_check)

    buyer = _suggest_buyer(party_name, party_candidates)

    # Every page in a session becomes ONE order, so pages naming different
    # buyers almost certainly means two unrelated forms were uploaded
    # together. Never silently resolved -- picking one buyer would file the
    # other form's rows against the wrong account.
    distinct = {_normalize_name(n) for n in party_names if n.strip()}
    if len(distinct) > 1:
        shown = ", ".join(f"“{n}”" for n in dict.fromkeys(party_names))
        buyer["note"] = (
            f"These pages name different buyers ({shown}). They will all be uploaded as ONE order "
            f"for the buyer selected here — remove any page that belongs to a different order."
        ) + (f" {buyer['note']}" if buyer.get("note") else "")

    with session.lock:
        session.payload = {
            "order": {
                "party_name": party_name,
                "order_no": order_no,
                "order_date": order_date,
                "buyer": buyer,
            },
            "pages": pages,
        }
    try:
        (session.dir / "review_payload.json").write_text(
            json.dumps(session.payload, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass   # debugging convenience only; never fail a run over it


def _page_headers(form: dict, stage_a: dict) -> list[str]:
    """The page's size columns: the form's own printed headers first, then
    any size a row actually carries that the header row missed (a handwritten
    overflow column, or a letter size the catalog check couldn't resolve)."""
    headers = [str(h) for h in stage_a.get("size_headers", [])]
    seen = set(headers)
    extra: set[str] = set()
    for item in form.get("items", []):
        extra.update(str(k) for k in (item.get("quantities") or {}))
    for size in sorted(extra - seen, key=lambda s: (0, int(s)) if s.isdigit() else (1, s)):
        headers.append(size)
    return headers


def _parse_total(value) -> int | None:
    """The printed row total as the model read it -- a free-text field, so
    it can hold anything from "12" to "12 dz" to "". Only a clean integer is
    useful as a checksum; anything else is treated as absent rather than
    guessed at."""
    text = str(value or "").strip()
    if not text:
        return None
    digits = "".join(c for c in text if c.isdigit())
    if not digits or digits != text.replace(" ", ""):
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _build_row(page: str, index: int, item: dict, flag: dict | None, db_note: dict | None,
               printed_total: int | None = None, struck_out: bool = False) -> dict:
    quantities = {str(k): int(v) for k, v in (item.get("quantities") or {}).items()}
    binding = _bind_product(item.get("item", ""), item.get("type", ""), db_note, quantities)
    return {
        "uid": f"{page}#{index}",
        "page": page,
        "source_index": index,
        "item": item.get("item", ""),
        "type": item.get("type", ""),
        "quantities": quantities,
        "printed_total": printed_total,
        "struck_out": struck_out,
        "flag": flag if flag and flag.get("status") not in (None, "ok", "no_recount") else None,
        "db_note": db_note,
        "bsid": binding["bsid"],
        "product": binding["product"],
        "auto_bound": binding["auto_bound"],
        "size_mismatch": binding["size_mismatch"],
        "candidates": binding["candidates"],
    }


def _bind_product(item_name: str, style: str, db_note: dict | None,
                  quantities: dict[str, int] | None = None) -> dict:
    """Resolves an extracted item row to a concrete catalog product+style
    group (`bsid`), which is what OrderDetails.DetID is derived from.

    Two independent routes can auto-bind, and everything else is offered as
    a ranked shortlist for the picker instead. A wrong silent binding ships
    the wrong product; an unbound row is a two-click fix. So both routes
    require the evidence to be *unambiguous*, not merely best-available.

    Route 1 -- brandlist_match.py's own trust rules (score / product-code
    narrowing) already say the name match is trustworthy, and it resolves to
    exactly one style group.

    Route 2 -- the row's own style code and its ordered sizes single out
    exactly one candidate. This is the disambiguator brandlist_match.py's
    module docstring identifies as the real one: "Bloomer Plain" alone is
    ambiguous between the ladies' and kids' lines, and only the size range
    actually ordered tells them apart. So a candidate qualifies only if its
    style matches what's written in the Style column AND its catalog sizes
    cover every size this row actually has a quantity for -- and it binds
    only when exactly one candidate qualifies. Confirmed against sample 5's
    real rows: "MYNA [IE]" binds (only HOORULEEN MYNA covers 80-100 in IE),
    while "Bloomers Plain [IE]" correctly stays unbound (its name match,
    BLOOMER KIDS PLAIN, tops out at size 75 and can't be this row), and
    "F.G-3025 [OE]" stays unbound because the plain and white variants both
    qualify -- a distinction only a human should make.
    """
    result = {"bsid": "", "product": None, "auto_bound": False, "candidates": [], "size_mismatch": []}
    try:
        db.LOOKUPS.ensure()
    except db.DbUnavailable:
        return result

    candidates: list = []
    trustworthy = False

    match = (db_note or {}).get("match")
    if match:
        named = db.LOOKUPS.products_named(match["bname"])
        if style:
            styled = [p for p in named if p.bstyle.casefold() == style.casefold()]
            named = styled or named
        candidates.extend(named)
        try:
            import brandlist_match as bm
            score = float(match.get("score") or 0)
            unique_code = bool(match.get("code_narrowed")) and match.get("narrowed_pool_size") == 1
            trustworthy = (
                score >= bm.AUTO_APPLY_UNNARROWED_SCORE
                or (unique_code and score >= bm.AUTO_APPLY_UNIQUE_CODE_SCORE)
                or (bool(match.get("code_narrowed"))
                    and (match.get("narrowed_pool_size") or 99) <= bm.AUTO_APPLY_NARROWED_MAX_POOL
                    and score >= bm.AUTO_APPLY_NARROWED_SCORE)
            )
        except Exception:
            trustworthy = False

    # Always search by text too: when the catalog matcher found nothing (or
    # something weak), the reviewer still gets a useful shortlist instead of
    # an empty picker. The pool is deliberately much wider than the eight
    # candidates actually shown -- route 2 below asks "is there exactly one
    # possibility?", and that question is only meaningful over a pool wide
    # enough to contain the alternatives. Confirmed necessary: with a pool
    # of 8, "Bloomers Print" auto-bound to a *Plain* product, because the
    # catalog's four LADIES PRINT variants were all outside the shortlist.
    query = f"{item_name} {style}".strip()
    pool = list(candidates)
    if query:
        for prod in db.LOOKUPS.search_products(query, limit=40):
            if all(prod.bsid != c.bsid for c in pool):
                pool.append(prod)

    ordered_sizes = {str(s) for s, q in (quantities or {}).items() if int(q or 0) > 0}

    # Picker ordering. Name relevance still decides *which* products are
    # worth showing -- reordering the whole 40-deep pool by style/size would
    # push genuinely-relevant name matches off the list entirely. So take
    # the most relevant dozen, then promote, among those, the ones whose
    # style matches the Style column read off the form and whose catalog
    # sizes actually cover what this row ordered. Display-only; binding is
    # decided separately below.
    def display_rank(index_and_product):
        index, prod = index_and_product
        style_miss = bool(style) and prod.bstyle.casefold() != style.casefold()
        size_miss = bool(ordered_sizes) and not ordered_sizes <= set(prod.sizes)
        return (style_miss, size_miss, index)

    shortlist = [p for _i, p in sorted(enumerate(pool[:12]), key=display_rank)][:8]
    result["candidates"] = [p.as_json() for p in shortlist]

    chosen = None
    if trustworthy and candidates:
        exact = [p for p in candidates if p.bname.casefold() == (match["bname"] or "").casefold()]
        trusted_pool = exact or candidates
        styled = [p for p in trusted_pool if style and p.bstyle.casefold() == style.casefold()]
        if len(trusted_pool) == 1:
            chosen = trusted_pool[0]
        elif len(styled) == 1:
            chosen = styled[0]

    if chosen is None:
        chosen = _bind_by_style_and_sizes(pool, item_name, style, ordered_sizes)

    if chosen is not None:
        result["bsid"] = chosen.bsid
        result["product"] = chosen.as_json()
        # brandlist_match.py matches on name alone, so route 1 can land on a
        # product that cannot actually fulfil this row -- confirmed on real
        # runs: "BLOOMER PRINT" -> BLOOMER KIDS PRINT (sizes 40-75) for a row
        # ordering 80/85/90, and "B 3444 COLLAR" -> B 3444 S 8719 COLLAR
        # (45-85) for a row ordering up to 90.
        #
        # The binding is still kept, because it carries real information: the
        # UI can then say "this product doesn't come in size 90" and turn
        # that one cell red, which points at the actual defect (usually a
        # column shift, repairable with one click). Dropping the binding
        # would replace that with a bare "not linked to a product" and make
        # the reviewer rediscover the problem. What is dropped is the
        # *confidence*: the row is marked for checking rather than tagged
        # "auto", so nothing invites a reviewer to trust it unexamined.
        covered = not ordered_sizes or ordered_sizes <= set(chosen.sizes)
        result["auto_bound"] = covered
        result["size_mismatch"] = sorted(
            (ordered_sizes - set(chosen.sizes)),
            key=lambda s: (0, int(s)) if s.isdigit() else (1, s),
        )
        # Whatever got bound must be visible in the picker, even when the
        # size-aware display ordering would otherwise push it off the list.
        if all(c["bsid"] != chosen.bsid for c in result["candidates"]):
            result["candidates"].insert(0, chosen.as_json())
            del result["candidates"][8:]
    return result


# A candidate is only considered for route 2 if its name is as good a match
# as the best name in the pool, within this many points. The size range then
# breaks ties among equally-plausible names -- it must never override a name
# difference. Confirmed against real rows: without this band, "Bloomers
# Print" bound to "EXODA BLOOMER LADIES PLAIN" purely because it was the one
# size-compatible entry in reach; with it, the PRINT variants are the band
# and there are several, so the row correctly stays for a human.
_NAME_BAND = 8.0


def _bind_by_style_and_sizes(pool: list, item_name: str, style: str, ordered_sizes: set[str]):
    """Route 2: the row's own style code and ordered size range single out
    exactly one candidate among the equally well-named ones.

    Returns None whenever that is 0 or 2+ candidates -- an unbound row costs
    a reviewer two clicks, a wrongly-bound one ships the wrong product."""
    # A single size is too thin to identify a product line by, and a row that
    # thin deserves a human's eyes regardless.
    if not style or len(ordered_sizes) < 2 or not pool:
        return None

    # Same scorer the picker ranks with (db._SCORER) -- WRatio saturates on
    # this catalog and would make the band meaningless. See db.py's comment.
    from rapidfuzz import utils as fz_utils
    scored = [(p, db._SCORER(item_name, p.bname, processor=fz_utils.default_process)) for p in pool]
    best = max(score for _p, score in scored)
    try:
        import brandlist_match as bm
        floor = bm.SUGGEST_SCORE_THRESHOLD
    except Exception:
        floor = 70.0
    if best < floor:
        return None

    qualified = [
        p for p, score in scored
        if score >= best - _NAME_BAND
        and p.bstyle.casefold() == style.casefold()
        and ordered_sizes <= set(p.sizes)
    ]
    return qualified[0] if len(qualified) == 1 else None


def _normalize_name(text: str) -> str:
    """Punctuation-insensitive form for comparing a handwritten party name
    against a buyer record: "M. K. Enterprises" and "M K ENTERPRISES" are
    the same buyer, and the difference is entirely in how the form was
    written."""
    return " ".join("".join(c if c.isalnum() else " " for c in (text or "")).split()).casefold()


def _suggest_buyer(party_name: str, party_checks: list[dict]) -> dict:
    """Pre-selects the buyer when the name on the form identifies exactly one
    buyer record, and always returns a shortlist for the picker.

    The buyer is the single worst field to get silently wrong -- it decides
    whose order this is -- and this buyer list is full of near-identical
    names ("M K ENTERPRISES" / "M.M ENTERPRISES" / "M K ENTERPRISES (JP)",
    all scoring within a few points of each other). So pre-selection needs
    the match to be unambiguous, not just top-ranked.
    """
    out = {"buyer_id": None, "selected": None, "candidates": [], "note": ""}
    name = party_name.strip()
    for check in party_checks:
        if check.get("suggested_party_name"):
            name = check["suggested_party_name"]
            out["note"] = check.get("note", "")
            break
    if not name:
        return out
    try:
        matches = db.LOOKUPS.search_buyers(name, limit=8)
    except db.DbUnavailable:
        return out
    out["candidates"] = [b.as_json() for b in matches]
    if not matches:
        return out

    # An exact match once punctuation is ignored beats any fuzzy score --
    # and this is the common case, since the pipeline reads the name off the
    # form correctly far more often than not. Only unique matches count:
    # two buyers normalizing to the same name is precisely when a human
    # should choose.
    target = _normalize_name(name)
    exact = [b for b in matches if _normalize_name(b.name) == target]
    if len(exact) == 1:
        out["buyer_id"] = exact[0].buyer_id
        out["selected"] = exact[0].as_json()
        return out

    from rapidfuzz import fuzz, utils as fz_utils
    best = matches[0]
    score = fuzz.WRatio(name, best.name, processor=fz_utils.default_process)
    runner_up = max(
        (fuzz.WRatio(name, b.name, processor=fz_utils.default_process) for b in matches[1:]),
        default=0,
    )
    if score >= 90 and score - runner_up >= 8:
        out["buyer_id"] = best.buyer_id
        out["selected"] = best.as_json()
    return out


# --------------------------------------------------------------- submission

def record_submission(session: Session, order_no: int, submitted: dict, resolved_count: int) -> None:
    """Writes an audit/correction record next to the session's own output.

    Beyond the audit trail, this is the raw material for improving the
    pipeline: it pairs what the model produced (already on disk as
    <name>.json) with exactly what a human decided was correct, per row and
    per cell. Diffing the two across a few hundred real orders is how the
    next round of accuracy work gets prioritized against real production
    failures rather than against this repo's fixed sample set.
    """
    record = {
        "order_no": order_no,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "session": session.sid,
        "images": [i.as_json() for i in session.images],
        "detail_rows": resolved_count,
        "extracted": session.payload,
        "submitted": submitted,
    }
    try:
        (session.dir / f"submitted_{order_no}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        traceback.print_exc()

    # Per-template correction memory (template_learning.py): diffs this
    # page-by-page against session.payload (the pre-correction reading) and
    # folds any learnable correction into that template's own file, so a
    # future upload of the same template benefits. Imported lazily, same
    # reason get_engine() lazily imports extract_ollama_cloud -- keeps this
    # (paddleocr/anthropic-adjacent) import cost off the server's startup
    # path, paid only when an order is actually submitted. Never raises on
    # its own (see its own try/except), but kept out of the block above
    # regardless -- this must never affect whether the audit record itself
    # gets written.
    import template_learning
    template_learning.record_corrections(session.payload, submitted)
