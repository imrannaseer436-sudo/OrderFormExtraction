"""FastAPI app behind the order-form review UI.

Routes are deliberately thin: session/extraction logic lives in
pipeline.py, all SQL in db.py. Everything the browser needs is JSON except
the two image routes, which stream files.

Run it with ../run_ui.py (which binds 0.0.0.0 so a phone on the same
network can upload photos straight from its camera).
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from . import db, pipeline   # noqa: E402  -- must follow load_dotenv/sys.path setup

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Order Form Review", docs_url=None, redoc_url=None)


# ------------------------------------------------------------------ models

class OrderHeader(BaseModel):
    head_id: int | None = None
    ref_no: str = ""
    order_dt: str = ""
    ref_dt: str = ""
    company_id: int = 1
    user_id: int | None = None
    order_source: str = "Local"


class SubmitRow(BaseModel):
    bsid: str = ""
    item: str = ""
    quantities: dict[str, int] = Field(default_factory=dict)
    # Not read by db.resolve_lines() -- only by template_learning.py's write
    # path (app/pipeline.py's record_submission()), to pair this submitted
    # row back to the exact extracted row it corrects. "" for a row somehow
    # submitted without one (defensive only; the client always sends it).
    uid: str = ""


class SubmitBody(BaseModel):
    header: OrderHeader
    rows: list[SubmitRow] = Field(default_factory=list)


# ------------------------------------------------------------------ helpers

def _session_or_404(sid: str) -> pipeline.Session:
    session = pipeline.get_session(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="That review session no longer exists. Upload the form again.")
    return session


def _db_error(exc: db.DbUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc), "kind": "db_unavailable"})


# ------------------------------------------------------------------- routes

@app.get("/api/bootstrap")
def bootstrap():
    """Everything the UI needs before it can render: operator/company
    pick-lists and whether the catalog is reachable at all. The UI stays
    usable (review, edit, correct) with the DB down -- only product binding
    and the upload itself need it -- so this reports the failure instead of
    erroring out."""
    out: dict = {"db_ok": True, "db_error": "", "users": [], "companies": [],
                 "catalog_size": 0, "buyer_count": 0}
    try:
        db.LOOKUPS.load(force=False)
        out["users"] = db.LOOKUPS.users()
        out["companies"] = db.LOOKUPS.companies()
        counts = db.LOOKUPS.counts()
        out["catalog_size"] = counts["products"]
        out["buyer_count"] = counts["buyers"]
    except db.DbUnavailable as exc:
        out["db_ok"] = False
        out["db_error"] = str(exc)
    return out


@app.post("/api/lookups/refresh")
def refresh_lookups():
    try:
        db.LOOKUPS.load(force=True)
    except db.DbUnavailable as exc:
        return _db_error(exc)
    return {"ok": True, "loaded_at": db.LOOKUPS.loaded_at.isoformat(timespec="seconds")}


@app.get("/api/products")
def search_products(q: str = "", limit: int = 30):
    try:
        return {"results": [p.as_json() for p in db.LOOKUPS.search_products(q, limit=min(limit, 100))]}
    except db.DbUnavailable as exc:
        return _db_error(exc)


@app.get("/api/products/{bsid}")
def get_product(bsid: str):
    try:
        product = db.LOOKUPS.product(bsid)
    except db.DbUnavailable as exc:
        return _db_error(exc)
    if product is None:
        raise HTTPException(status_code=404, detail="No such product in the catalog.")
    return product.as_json()


@app.get("/api/buyers")
def search_buyers(q: str = "", limit: int = 30):
    try:
        return {"results": [b.as_json() for b in db.LOOKUPS.search_buyers(q, limit=min(limit, 100))]}
    except db.DbUnavailable as exc:
        return _db_error(exc)


@app.post("/api/sessions")
async def create_session(files: list[UploadFile] = File(default=[]), model: str = Form(default="")):
    session = pipeline.create_session(model=model or None)
    if files:
        uploads = [(f.filename or "page.jpg", await f.read()) for f in files]
        pipeline.add_images(session, uploads)
    return session.as_json()


@app.post("/api/sessions/{sid}/images")
async def add_images(sid: str, files: list[UploadFile] = File(...)):
    session = _session_or_404(sid)
    if session.submitted:
        raise HTTPException(status_code=409, detail="This order has already been uploaded. Start a new one.")
    uploads = [(f.filename or "page.jpg", await f.read()) for f in files]
    added = pipeline.add_images(session, uploads)
    if not added:
        raise HTTPException(status_code=400, detail="No usable image files in that upload.")
    return session.as_json()


@app.delete("/api/sessions/{sid}/images/{name}")
def delete_image(sid: str, name: str):
    session = _session_or_404(sid)
    if not pipeline.remove_image(session, name):
        raise HTTPException(status_code=409, detail="That page is still being read — wait for it to finish.")
    return session.as_json()


@app.post("/api/sessions/{sid}/images/{name}/retry")
def retry_image(sid: str, name: str):
    session = _session_or_404(sid)
    if not pipeline.retry_image(session, name):
        raise HTTPException(status_code=409, detail="That page is already being read.")
    return session.as_json()


@app.get("/api/sessions/{sid}")
def get_session(sid: str):
    return _session_or_404(sid).as_json()


@app.get("/api/sessions/{sid}/images/{name}/file")
def image_file(sid: str, name: str, original: bool = False):
    session = _session_or_404(sid)
    job = next((i for i in session.images if i.name == name), None)
    if job is None:
        raise HTTPException(status_code=404, detail="No such page in this session.")
    path = job.path if original else job.display_path
    if not path.exists():
        path = job.path
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/sessions/{sid}/validate")
def validate(sid: str, body: SubmitBody):
    """Dry run of the exact resolution the upload does, so the reviewer sees
    every blocking problem (unbound row, size the product doesn't come in,
    quantity out of range) while they can still fix it -- not as a failed
    insert."""
    _session_or_404(sid)
    try:
        result = db.resolve_lines([r.model_dump() for r in body.rows])
    except db.DbUnavailable as exc:
        return _db_error(exc)
    return {
        "ok": not result.problems and bool(result.lines),
        "problems": result.problems,
        "merged": result.merged,
        "detail_rows": len(result.lines),
        "total_qty": sum(l.qty for l in result.lines),
    }


@app.post("/api/sessions/{sid}/submit")
def submit(sid: str, body: SubmitBody):
    session = _session_or_404(sid)
    if session.submitted:
        return {"ok": True, "already": True, **session.submitted}

    header = body.header
    if not header.head_id:
        raise HTTPException(status_code=400, detail="Pick the buyer before uploading.")
    if not header.user_id:
        raise HTTPException(status_code=400, detail="Pick the operator before uploading.")

    try:
        result = db.resolve_lines([r.model_dump() for r in body.rows])
        if result.problems:
            return JSONResponse(status_code=422, content={
                "ok": False, "problems": result.problems, "merged": result.merged,
                "detail": "Some rows can't be uploaded yet.",
            })
        if not result.lines:
            raise HTTPException(status_code=400, detail="Nothing to upload — no row has any quantity.")

        order_no = db.insert_order(header.model_dump(), result.lines)
        summary = db.order_summary(order_no)
    except db.DbUnavailable as exc:
        return _db_error(exc)

    pipeline.record_submission(session, order_no, body.model_dump(), len(result.lines))
    session.submitted = {"order_no": order_no, "summary": summary, "merged": result.merged}
    return {"ok": True, **session.submitted}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
