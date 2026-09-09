"""SQL Server access for the review UI: the search-as-you-type lookups the
reviewer picks from, and the OrderMaster/OrderDetails insert that ends the
review.

Why this exists separately from brandlist_match.py: that module answers
"which catalog product does this extracted name probably mean?" for the
*pipeline*, and deliberately deals in names/sizes only. The UI needs
something different -- the actual `detid` primary keys OrderDetails is
written in terms of, plus buyer/operator/company lookups the pipeline never
touches. Both talk to the same database; neither imports the other.

The unit OrderDetails is written in is `brandlist.detid`, which identifies a
(product, style, size) triple -- NOT a product. So a reviewed item row can
only be inserted once it is bound to one catalog product+style group
(`bsid` = bid + bstyle), and only for sizes that group actually has. A size
the reviewer added by hand that the product has no `detid` for cannot be
inserted at all; resolve_lines() reports those rather than silently
dropping them, because a silently-dropped size is an order shipped short.
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable

import pyodbc
from rapidfuzz import fuzz, process, utils

# Catalog rows worth offering in a picker. isunused=1 rows are literally
# named "(UNUSED) ..." in this catalog (confirmed against the live table),
# and isprimary=0 rows are the non-canonical duplicates brandlist_match.py
# already filters out for the same reason.
_CATALOG_SQL = """
SELECT bid, bsid, bname, bstyle, bsize, detid
FROM brandlist
WHERE isprimary = 1 AND isunused = 0 AND bsize IS NOT NULL
"""

# IsActive = 0, not 1. This column's sense is inverted from its name here --
# confirmed against live data (2026-09-03): 1996 of the last 2000 orders went
# to IsActive = 0 buyers (882 distinct), 4 to IsActive = 1 buyers, and the
# IsActive = 1 records with order history are retired accounts, several with
# "(CLOSE)" in the name. Filtering the other way would hide from the picker
# nearly every buyer that actually places orders. Same fix applied to
# brandlist_match._buyers(), which had the same inverted filter.
_BUYERS_SQL = "SELECT BuyerID, BuyerCode, BuyerName, City, State FROM BUYER WHERE IsActive = 0"
_USERS_SQL = "SELECT UserID, UserName FROM Users ORDER BY UserName"
_COMPANIES_SQL = "SELECT CompanyID, CompanyName FROM Company ORDER BY CompanyID"

# smallint -- OrderDetails.Qty's actual column type. A quantity above this
# fails at the driver with an unhelpful error, so it's checked up front
# where the message can name the row.
QTY_MAX = 32767
REFNO_MAX_LEN = 20   # OrderMaster.RefNo varchar(20)

_DIGIT_RUN_RE = re.compile(r"\d{3,}")

# token_set_ratio, NOT WRatio. Measured directly against this catalog
# (2026-09-03) over the item names this project's history records as hard:
# WRatio saturates at ~86 for almost every product, because its
# partial_ratio component rewards any short shared fragment -- so
# "Image FCD" ranked four BABYCARE products above every FCD product,
# "F.G-3025" ranked five unrelated "2.0 G ..." products above
# "F.G - 3025 TRUNKS", and "Bloomers Print" put three BABYCARE items in the
# top four. This is the same generic-word-overlap ceiling brandlist_match.py
# documents hitting (its comment about two unrelated names both scoring
# 85.5). token_set_ratio put the right product first in every one of those
# cases. Product-code queries ("MM K4532") are handled by the narrowing pass
# in _rank(), not by the scorer -- no scorer got those right on text alone.
_SCORER = fuzz.token_set_ratio


class DbUnavailable(RuntimeError):
    """Raised when the DB credentials are missing or the server is
    unreachable. Callers turn this into a visible banner rather than a
    stack trace -- the pipeline itself still runs without a DB (see
    CLAUDE.md), it's only the catalog binding and the insert that can't."""


@dataclass
class Product:
    bid: int
    bsid: str
    bname: str
    bstyle: str
    sizes: dict[str, int] = field(default_factory=dict)   # size -> detid

    @property
    def label(self) -> str:
        return f"{self.bname} [{self.bstyle}]" if self.bstyle else self.bname

    def as_json(self) -> dict:
        return {
            "bid": self.bid,
            "bsid": self.bsid,
            "bname": self.bname,
            "bstyle": self.bstyle,
            "label": self.label,
            "sizes": self.sizes,
            "size_list": _sorted_sizes(self.sizes.keys()),
        }


@dataclass
class Buyer:
    buyer_id: int
    code: str
    name: str
    city: str
    state: str

    def as_json(self) -> dict:
        return {
            "buyer_id": self.buyer_id,
            "code": self.code,
            "name": self.name,
            "city": self.city,
            "state": self.state,
            "label": f"{self.name} — {self.city}" if self.city else self.name,
        }


def _sorted_sizes(sizes: Iterable[str]) -> list[str]:
    return sorted(sizes, key=lambda s: (0, int(s)) if str(s).isdigit() else (1, str(s)))


def parse_date(value) -> date | None:
    """Accepts what either end of this app actually produces: the browser's
    ISO date input (YYYY-MM-DD) and the form's own DD/MM/YYYY reading, which
    is what the pipeline writes into order_date. Returns a real date object
    rather than a string so the driver binds a date, not a locale-dependent
    string conversion."""
    if isinstance(value, (datetime, date)):
        return value if isinstance(value, date) and not isinstance(value, datetime) else value.date()
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def connection_string() -> str:
    missing = [k for k in ("SERVER", "DB", "USER", "PASSWORD") if not os.environ.get(k)]
    if missing:
        raise DbUnavailable(
            f"Missing database settings in .env: {', '.join(missing)}. "
            "Product/buyer lookup and order upload need all four."
        )
    return (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER={os.environ['SERVER']};DATABASE={os.environ['DB']};"
        f"UID={os.environ['USER']};PWD={os.environ['PASSWORD']};TrustServerCertificate=yes"
    )


def connect(timeout: int = 15) -> pyodbc.Connection:
    try:
        return pyodbc.connect(connection_string(), timeout=timeout)
    except pyodbc.Error as exc:
        raise DbUnavailable(f"Could not connect to SQL Server: {exc}") from exc


# ---------------------------------------------------------------- lookups

class _Lookups:
    """In-memory copy of the three pick-lists. ~6k product groups and ~2.9k
    buyers -- small enough to hold and fuzzy-search in process, which is
    what makes search-as-you-type feel instant and, more importantly, lets
    the picker be *fuzzy*: a reviewer typing "4532" or a half-remembered
    name gets the right product, which a SQL LIKE can't do."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._products: list[Product] = []
        self._by_bsid: dict[str, Product] = {}
        self._product_keys: list[str] = []
        self._buyers: list[Buyer] = []
        self._buyer_keys: list[str] = []
        self._users: list[dict] = []
        self._companies: list[dict] = []
        self._loaded_at: datetime | None = None
        self._error: str = ""

    # -- loading -------------------------------------------------------
    def load(self, force: bool = False) -> None:
        with self._lock:
            if self._loaded_at and not force:
                return
            conn = connect()
            try:
                cur = conn.cursor()

                groups: dict[str, Product] = {}
                cur.execute(_CATALOG_SQL)
                for bid, bsid, bname, bstyle, bsize, detid in cur.fetchall():
                    bname = (bname or "").strip()
                    if not bname:
                        continue
                    key = bsid or f"{bid}{bstyle or ''}"
                    prod = groups.get(key)
                    if prod is None:
                        prod = Product(bid=int(bid), bsid=key, bname=bname, bstyle=(bstyle or "").strip())
                        groups[key] = prod
                    prod.sizes[str(int(bsize))] = int(detid)

                self._products = sorted(groups.values(), key=lambda p: p.bname)
                self._by_bsid = {p.bsid: p for p in self._products}
                self._product_keys = [f"{p.bname} {p.bstyle}".strip() for p in self._products]

                cur.execute(_BUYERS_SQL)
                self._buyers = sorted(
                    (Buyer(int(r[0]), (r[1] or "").strip(), (r[2] or "").strip(),
                           (r[3] or "").strip(), (r[4] or "").strip())
                     for r in cur.fetchall() if (r[2] or "").strip()),
                    key=lambda b: b.name,
                )
                self._buyer_keys = [f"{b.name} {b.city}".strip() for b in self._buyers]

                cur.execute(_USERS_SQL)
                self._users = [{"user_id": int(r[0]), "name": (r[1] or "").strip()} for r in cur.fetchall()]

                cur.execute(_COMPANIES_SQL)
                self._companies = [{"company_id": int(r[0]), "name": (r[1] or "").strip()} for r in cur.fetchall()]

                self._loaded_at = datetime.now()
                self._error = ""
            finally:
                conn.close()

    def ensure(self) -> None:
        if self._loaded_at is None:
            self.load()

    @property
    def loaded_at(self) -> datetime | None:
        return self._loaded_at

    # -- queries -------------------------------------------------------
    def search_products(self, query: str, limit: int = 30) -> list[Product]:
        self.ensure()
        query = (query or "").strip()
        if not query:
            return self._products[:limit]
        return _rank(query, self._product_keys, self._products, limit)

    def product(self, bsid: str) -> Product | None:
        self.ensure()
        return self._by_bsid.get(bsid)

    def products_named(self, bname: str) -> list[Product]:
        """Every style group sharing one catalog product name -- used to turn
        brandlist_match.py's name-level suggestion into a concrete bsid."""
        self.ensure()
        target = (bname or "").strip().casefold()
        return [p for p in self._products if p.bname.casefold() == target]

    def search_buyers(self, query: str, limit: int = 30) -> list[Buyer]:
        self.ensure()
        query = (query or "").strip()
        if not query:
            return self._buyers[:limit]
        return _rank(query, self._buyer_keys, self._buyers, limit)

    def counts(self) -> dict[str, int]:
        self.ensure()
        return {"products": len(self._products), "buyers": len(self._buyers)}

    def users(self) -> list[dict]:
        self.ensure()
        return self._users

    def companies(self) -> list[dict]:
        self.ensure()
        return self._companies


def _rank(query: str, keys: list[str], rows: list, limit: int) -> list:
    """Fuzzy rank, with exact-ish evidence promoted over the text score.

    A plain WRatio ordering is not good enough for either of these lists.
    On the catalog, a 3+ digit product code ("4532") is far more
    distinctive than the words around it -- brandlist_match.find_best_match
    documents this from real failures, where whole-name fuzzy matching
    picked a same-brand, differently-numbered product over the exact code
    match. On the buyer list, generic shared words ("SALES", "HOSIERY",
    "HANDLOOM") tie hundreds of records at nearly the same score, and ties
    fall back to list order, i.e. alphabetical -- so typing an exact buyer
    name could return a page of unrelated "A ..." names.

    Sorting on (code hit, prefix match, substring match, score) fixes both:
    anything the query literally matches outranks anything it merely
    resembles, and the fuzzy score only orders what's left."""
    q = query.casefold()
    codes = set(_DIGIT_RUN_RE.findall(query))

    def rank_pool(pool_idx: list[int], want: int) -> list[int]:
        pool_keys = [keys[i] for i in pool_idx]
        scored = process.extract(
            query, pool_keys, scorer=_SCORER, processor=utils.default_process,
            limit=max(want * 4, 60), score_cutoff=40,
        )

        def sort_key(entry):
            key, score, _idx = entry
            low = key.casefold()
            return (low.startswith(q), q in low, score)

        scored.sort(key=sort_key, reverse=True)
        return [pool_idx[idx] for _key, _score, idx in scored]

    ordered: list[int] = []
    seen: set[int] = set()

    # Pass 1: only the records literally containing the query's product
    # code. Scoring the whole list first and hoping the code match survives
    # into the top N does not work -- confirmed here on "MM K4532", whose
    # one true catalog match ("MM K 4532 B FULL PANT SET") was crowded out
    # of a 60-candidate shortlist by generic word overlap alone.
    if codes:
        narrowed = [i for i, key in enumerate(keys) if any(c in key for c in codes)]
        for i in rank_pool(narrowed, limit):
            if i not in seen:
                seen.add(i)
                ordered.append(i)

    # Pass 2: the whole list, for everything the code didn't already catch
    # (and for queries with no code at all).
    if len(ordered) < limit:
        for i in rank_pool(list(range(len(keys))), limit):
            if i not in seen:
                seen.add(i)
                ordered.append(i)

    return [rows[i] for i in ordered[:limit]]


LOOKUPS = _Lookups()


# ---------------------------------------------------------------- insert

@dataclass
class ResolvedLine:
    rsno: int
    detid: int
    qty: int
    item_label: str
    size: str


@dataclass
class ResolveResult:
    lines: list[ResolvedLine]
    problems: list[dict]           # blocking: cannot be inserted as-is
    merged: list[dict]             # informational: duplicate detids summed


def resolve_lines(rows: list[dict]) -> ResolveResult:
    """Turns the reviewer's grid into OrderDetails rows.

    `rows` are the submitted item rows: {bsid, item, quantities: {size: qty}}
    in display order. Every problem found is collected and returned rather
    than raised on the first one -- a reviewer fixing five unbound rows one
    round-trip at a time is exactly the friction this UI exists to remove.
    """
    LOOKUPS.ensure()
    lines: list[ResolvedLine] = []
    problems: list[dict] = []
    merged: list[dict] = []
    # (OrderNo, DetID) is OrderDetails' primary key, so the same SKU landing
    # twice in one order is not insertable. It happens legitimately -- a
    # multi-page order repeating a product, or two rows bound to the same
    # product after a correction -- so the quantities are summed and the
    # merge is reported, rather than failing the whole upload.
    by_detid: dict[int, ResolvedLine] = {}

    for rsno, row in enumerate(rows, start=1):
        label = (row.get("item") or "").strip() or f"row {rsno}"
        bsid = (row.get("bsid") or "").strip()
        quantities = {str(k): v for k, v in (row.get("quantities") or {}).items()}
        quantities = {k: int(v) for k, v in quantities.items() if str(v).strip() not in ("", "0", "None") and int(v) != 0}

        if not quantities:
            continue   # an all-blank row contributes nothing; not an error

        if not bsid:
            problems.append({"rsno": rsno, "item": label, "kind": "unbound",
                             "message": f"“{label}” isn’t linked to a catalog product yet."})
            continue

        product = LOOKUPS.product(bsid)
        if product is None:
            problems.append({"rsno": rsno, "item": label, "kind": "unknown_product",
                             "message": f"“{label}” is linked to a product ({bsid}) that is no longer in the catalog."})
            continue

        missing_sizes = []
        for size, qty in quantities.items():
            if qty < 0:
                problems.append({"rsno": rsno, "item": label, "kind": "negative_qty", "size": size,
                                 "message": f"“{label}” size {size}: quantity can’t be negative."})
                continue
            if qty > QTY_MAX:
                problems.append({"rsno": rsno, "item": label, "kind": "qty_too_large", "size": size,
                                 "message": f"“{label}” size {size}: {qty} exceeds the maximum ({QTY_MAX})."})
                continue
            detid = product.sizes.get(str(size))
            if detid is None:
                missing_sizes.append(str(size))
                continue
            existing = by_detid.get(detid)
            if existing is not None:
                existing.qty += qty
                merged.append({"detid": detid, "size": size, "item": label,
                               "message": f"“{label}” size {size} was added to an earlier row for the same product."})
                continue
            line = ResolvedLine(rsno=rsno, detid=detid, qty=qty, item_label=label, size=str(size))
            by_detid[detid] = line
            lines.append(line)

        if missing_sizes:
            available = ", ".join(_sorted_sizes(product.sizes.keys()))
            problems.append({
                "rsno": rsno, "item": label, "kind": "size_not_in_product",
                "sizes": missing_sizes,
                "message": (f"“{label}” has quantities for size(s) {', '.join(_sorted_sizes(missing_sizes))}, "
                            f"which {product.label} doesn’t come in (catalog sizes: {available})."),
            })

    return ResolveResult(lines=lines, problems=problems, merged=merged)


def insert_order(header: dict, lines: list[ResolvedLine]) -> int:
    """Writes one OrderMaster row and its OrderDetails rows in a single
    transaction, returning the allocated OrderNo.

    OrderNo is not an identity column in this schema (checked against
    INFORMATION_SCHEMA) -- the existing application allocates MAX+1 itself.
    That is a race by nature, so the MAX is taken under (UPDLOCK, HOLDLOCK),
    which holds a range lock for the life of the transaction and serializes
    concurrent allocations instead of letting two sessions pick the same
    number.
    """
    if not lines:
        raise ValueError("Nothing to insert -- no quantities on any row.")

    ref_no = (header.get("ref_no") or "").strip()[:REFNO_MAX_LEN]
    order_dt = parse_date(header.get("order_dt")) or date.today()
    ref_dt = parse_date(header.get("ref_dt")) or order_dt
    head_id = int(header["head_id"])
    company_id = int(header.get("company_id") or 1)
    user_id = int(header.get("user_id") or 0)
    order_source = (header.get("order_source") or "Local")[:50]

    conn = connect()
    try:
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute("SELECT ISNULL(MAX(OrderNo), 0) + 1 FROM OrderMaster WITH (UPDLOCK, HOLDLOCK)")
        order_no = int(cur.fetchone()[0])

        cur.execute(
            """
            INSERT INTO OrderMaster
                (OrderNo, OrderDt, RefNo, RefDt, HeadID, IsClose, OrderSource,
                 OnlineOrderId, WebUserId, UpdateDate, CompanyID, UserID)
            VALUES (?, ?, ?, ?, ?, 0, ?, 0, 0, GETDATE(), ?, ?)
            """,
            order_no, order_dt, ref_no, ref_dt, head_id, order_source, company_id, user_id,
        )

        cur.fast_executemany = True
        cur.executemany(
            "INSERT INTO OrderDetails (OrderNo, DetID, Qty, RSno, Sno, IsClosed) VALUES (?, ?, ?, ?, ?, 0)",
            [(order_no, ln.detid, ln.qty, ln.rsno, sno) for sno, ln in enumerate(lines, start=1)],
        )
        conn.commit()
        return order_no
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = True
        conn.close()


def order_summary(order_no: int) -> dict:
    """Read-back of what was actually written, shown on the success screen
    -- the point of the upload is the DB row, so the confirmation should
    come from the DB, not from what the app believes it sent."""
    conn = connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.OrderNo, m.OrderDt, m.RefNo, m.HeadID, b.BuyerName, b.City,
                   (SELECT COUNT(*) FROM OrderDetails d WHERE d.OrderNo = m.OrderNo),
                   (SELECT SUM(CAST(d.Qty AS int)) FROM OrderDetails d WHERE d.OrderNo = m.OrderNo)
            FROM OrderMaster m LEFT JOIN BUYER b ON b.BuyerID = m.HeadID
            WHERE m.OrderNo = ?
            """,
            order_no,
        )
        row = cur.fetchone()
        if row is None:
            return {}
        return {
            "order_no": int(row[0]),
            "order_dt": row[1].strftime("%d/%m/%Y") if row[1] else "",
            "ref_no": row[2] or "",
            "head_id": int(row[3]) if row[3] is not None else None,
            "buyer_name": row[4] or "",
            "buyer_city": row[5] or "",
            "detail_rows": int(row[6] or 0),
            "total_qty": int(row[7] or 0),
        }
    finally:
        conn.close()
