/* Order Form Review — client.
 *
 * Three ideas hold this file together:
 *
 * 1. The server's payload is a *starting point*, not the truth. The moment a
 *    page's rows are adopted into the local model they belong to the
 *    reviewer; later polls only update that page's status, never its data.
 *    That's what lets extraction of page 3 finish while page 1 is being
 *    edited without silently reverting the edits.
 *
 * 2. Every automatic reading is editable and every correction is local until
 *    "Upload to database". Nothing is written to SQL Server before that one
 *    click, and the click is validated server-side first.
 *
 * 3. The pipeline's known failure modes have direct, one-click repairs
 *    (see HISTORY.md): a whole row read one column off (← →), a block of
 *    rows attributed one row off (shift ↑/↓ from here), a product whose real
 *    sizes aren't on the printed grid at all (add the product's own sizes as
 *    columns). Those are the repairs, not free-text re-entry.
 */

'use strict';

/* ══════════════════════════════════════════════ utilities ═══════════════ */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (k === 'dataset') Object.assign(node.dataset, v);
    else if (v === true) node.setAttribute(k, '');
    else node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const text = await res.text();
  let body;
  try { body = text ? JSON.parse(text) : null; } catch { body = { detail: text }; }
  if (!res.ok) {
    const err = new Error((body && (body.detail || body.message)) || `${res.status} ${res.statusText}`);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

function toast(message, kind = '') {
  const node = el('div', { class: `toast ${kind}`.trim(), text: message });
  $('#toast-area').append(node);
  setTimeout(() => {
    node.style.transition = 'opacity .3s';
    node.style.opacity = '0';
    setTimeout(() => node.remove(), 300);
  }, kind === 'bad' ? 6000 : 3000);
}

/** Sizes sort numerically first, then letter sizes alphabetically — the
 *  order they appear across a real form's printed header row. */
function sizeSort(a, b) {
  const na = /^\d+$/.test(a), nb = /^\d+$/.test(b);
  if (na && nb) return Number(a) - Number(b);
  if (na !== nb) return na ? -1 : 1;
  return a.localeCompare(b);
}

function isoToday() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

/** The pipeline writes dates as DD/MM/YYYY (schema.py normalizes to that);
 *  <input type="date"> needs YYYY-MM-DD. Returns '' for anything else so a
 *  half-read date leaves the field empty rather than wrong. */
function formDateToIso(text) {
  const m = /^(\d{1,2})\/(\d{1,2})\/(\d{2,4})$/.exec((text || '').trim());
  if (!m) return '';
  let [, d, mo, y] = m;
  if (y.length === 2) y = `20${y}`;
  const iso = `${y}-${mo.padStart(2, '0')}-${d.padStart(2, '0')}`;
  return Number.isNaN(Date.parse(iso)) ? '' : iso;
}

/* ══════════════════════════════════════════════ state ═══════════════════ */

const state = {
  sid: null,
  status: 'idle',
  images: [],
  pages: [],              // local, editable model
  adopted: new Set(),     // page names whose rows are now owned by the client
  order: { party_name: '', ref_no: '', ref_dt: '', order_dt: isoToday(), buyer: null, buyer_note: '' },
  touched: new Set(),     // order fields the reviewer changed by hand
  activePage: null,
  operators: [],
  companies: [],
  operatorId: null,
  companyId: null,
  dbOk: false,
  polling: null,
  uploading: 0,      // uploads in flight, for the progress bar
};

const LS_KEY = 'orderFormReview:v1';

function persist() {
  if (!state.sid) return;
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({
      sid: state.sid,
      pages: state.pages,
      adopted: [...state.adopted],
      order: state.order,
      touched: [...state.touched],
      activePage: state.activePage,
      operatorId: state.operatorId,
      companyId: state.companyId,
    }));
  } catch { /* quota or private mode — persistence is a convenience, not a requirement */ }
}

function loadPersisted() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

function clearPersisted() {
  try { localStorage.removeItem(LS_KEY); } catch { /* ignore */ }
}

const savePersist = debounce(persist, 400);

/* ══════════════════════════════════════════════ bootstrap ═══════════════ */

async function bootstrap() {
  const dbStatus = $('#db-status');
  try {
    const info = await api('/api/bootstrap');
    state.dbOk = info.db_ok;
    state.operators = info.users || [];
    state.companies = info.companies || [];

    if (info.db_ok) {
      dbStatus.className = 'pill pill-ok';
      dbStatus.textContent = `${info.catalog_size.toLocaleString()} products`;
      dbStatus.title = `${info.catalog_size} catalog products · ${info.buyer_count} active buyers`;
    } else {
      dbStatus.className = 'pill pill-bad';
      dbStatus.textContent = 'no database';
      dbStatus.title = info.db_error;
      showBanner('error', `Can’t reach the product catalog: ${info.db_error} Review still works, but orders can’t be uploaded until this is fixed.`);
    }
  } catch (err) {
    dbStatus.className = 'pill pill-bad';
    dbStatus.textContent = 'offline';
    showBanner('error', `Couldn’t start up: ${err.message}`);
  }

  fillSelect($('#operator-select'), state.operators, 'user_id', 'name', 'Select…');
  fillSelect($('#company-select'), state.companies, 'company_id', 'name', '');

  const saved = loadPersisted();
  const savedOperator = saved?.operatorId ?? Number(localStorage.getItem(`${LS_KEY}:operator`) || 0);
  const savedCompany = saved?.companyId ?? Number(localStorage.getItem(`${LS_KEY}:company`) || 0);
  if (savedOperator) $('#operator-select').value = String(savedOperator);
  if (savedCompany) $('#company-select').value = String(savedCompany);
  else if (state.companies.length) $('#company-select').value = String(state.companies[0].company_id);
  state.operatorId = Number($('#operator-select').value) || null;
  state.companyId = Number($('#company-select').value) || null;

  if (saved?.sid) restoreSession(saved);
}

function fillSelect(select, rows, idKey, labelKey, placeholder) {
  select.textContent = '';
  if (placeholder !== undefined && placeholder !== '') {
    select.append(el('option', { value: '', text: placeholder }));
  }
  for (const row of rows) {
    select.append(el('option', { value: String(row[idKey]), text: row[labelKey] || `#${row[idKey]}` }));
  }
}

function showBanner(kind, message, action) {
  const area = $('#banner-area');
  const banner = el('div', { class: `banner banner-${kind}` }, el('span', { text: message }));
  if (action) banner.append(el('button', { class: 'btn btn-ghost', text: action.label, onclick: action.onClick }));
  banner.append(el('button', { class: 'icon-btn', text: '✕', onclick: () => banner.remove() }));
  area.append(banner);
}

async function restoreSession(saved) {
  try {
    const server = await api(`/api/sessions/${saved.sid}`);
    state.sid = saved.sid;
    state.pages = saved.pages || [];
    state.adopted = new Set(saved.adopted || []);
    state.order = { ...state.order, ...(saved.order || {}) };
    state.touched = new Set(saved.touched || []);
    state.activePage = saved.activePage;
    applyServerSession(server, { adoptNew: true });
    showView('review');
    renderAll();
    startPolling();
    toast('Picked up where you left off');
  } catch {
    clearPersisted();
  }
}

/* ══════════════════════════════════════════════ views ═══════════════════ */

function showView(name) {
  $('#view-upload').hidden = name !== 'upload';
  $('#view-review').hidden = name !== 'review';
  $('#view-done').hidden = name !== 'done';
  $('#new-order-btn').hidden = name === 'upload';
}

/* ══════════════════════════════════════════════ uploading ═══════════════ */

async function startSession(files) {
  if (!files.length) return;
  const form = new FormData();
  for (const file of files) form.append('files', file);
  const modelSelect = $('#model-select');
  if (modelSelect && modelSelect.value) form.append('model', modelSelect.value);

  showView('review');
  // The POST itself takes real time on phone photos, and until it returns
  // there are no pages to render — so stand in for them straight away
  // rather than showing an empty review screen.
  state.uploading += 1;
  renderProgress();
  renderUploadPlaceholder(files.length);
  try {
    const session = await api('/api/sessions', { method: 'POST', body: form });
    state.sid = session.sid;
    state.pages = [];
    state.adopted = new Set();
    state.touched = new Set();
    state.order = { party_name: '', ref_no: '', ref_dt: '', order_dt: isoToday(), buyer: null, buyer_note: '' };
    applyServerSession(session, { adoptNew: true });
    renderAll();
    startPolling();
    persist();
  } catch (err) {
    showBanner('error', `Upload failed: ${err.message}`);
    showView('upload');
  } finally {
    state.uploading -= 1;
    renderProgress();
  }
}

function renderUploadPlaceholder(count) {
  const container = $('#pages-container');
  container.textContent = '';
  container.append(el('article', { class: 'card' },
    el('div', { class: 'page-status' },
      el('div', { class: 'spinner' }),
      count === 1 ? 'Uploading the photo…' : `Uploading ${count} photos…`)));
}

async function addPages(files) {
  if (!files.length || !state.sid) return;
  const form = new FormData();
  for (const file of files) form.append('files', file);
  state.uploading += 1;
  renderProgress();
  try {
    const session = await api(`/api/sessions/${state.sid}/images`, { method: 'POST', body: form });
    applyServerSession(session, { adoptNew: true });
    renderAll();
    startPolling();
  } catch (err) {
    showBanner('error', `Couldn’t add those pages: ${err.message}`);
  } finally {
    state.uploading -= 1;
    renderProgress();
  }
}

/* ══════════════════════════════════════════════ polling ═════════════════ */

function startPolling() {
  stopPolling();
  state.polling = setInterval(async () => {
    if (!state.sid) return stopPolling();
    try {
      const session = await api(`/api/sessions/${state.sid}`);
      const changed = applyServerSession(session, { adoptNew: true });
      if (changed) renderAll();
      if (!session.images.some(i => i.status === 'queued' || i.status === 'running')) stopPolling();
    } catch {
      // Without this, a lost connection just stops the "reading the
      // form" spinner silently mid-page with no explanation -- looks
      // exactly like the app has frozen, not like something failed.
      stopPolling();
      showBanner('error', 'Lost connection while checking for results.', { label: 'Retry', onClick: startPolling });
    }
  }, 2500);
}

function stopPolling() {
  if (state.polling) clearInterval(state.polling);
  state.polling = null;
}

/** Folds a server session into the local model.
 *  Rows are taken from the server ONLY for pages not yet adopted; an adopted
 *  page's data belongs to the reviewer from that point on. */
function applyServerSession(session, { adoptNew }) {
  let changed = false;
  const before = JSON.stringify(state.images);
  state.images = session.images || [];
  state.status = session.status;
  if (before !== JSON.stringify(state.images)) changed = true;

  const payload = session.payload || {};
  const serverPages = payload.pages || [];

  // Drop local pages the server no longer has (a page was removed).
  const serverNames = new Set(serverPages.map(p => p.name));
  if (state.pages.some(p => !serverNames.has(p.name))) {
    state.pages = state.pages.filter(p => serverNames.has(p.name));
    changed = true;
  }

  for (const sp of serverPages) {
    let local = state.pages.find(p => p.name === sp.name);
    if (!local) {
      local = { name: sp.name, status: sp.status, message: sp.message, headers: [], rows: [], notes: [], collapsed: false };
      state.pages.push(local);
      changed = true;
    }
    if (local.status !== sp.status || local.message !== sp.message) changed = true;
    local.status = sp.status;
    local.message = sp.message;
    local.seller_name = sp.seller_name || '';
    local.party_check = sp.party_check || null;
    local.template_learning = sp.template_learning || null;

    if (sp.status === 'done' && adoptNew && !state.adopted.has(sp.name)) {
      local.headers = [...sp.headers];
      local.rows = sp.rows.map(r => ({ ...r, quantities: { ...r.quantities } }));
      local.notes = sp.notes || [];
      state.adopted.add(sp.name);
      changed = true;
    }
  }

  state.pages.sort((a, b) => {
    const ia = state.images.findIndex(i => i.name === a.name);
    const ib = state.images.findIndex(i => i.name === b.name);
    return ia - ib;
  });

  // Order-level fields, only where the reviewer hasn't taken over.
  const order = payload.order || {};
  if (!state.touched.has('party_name') && order.party_name) state.order.party_name = order.party_name;
  if (!state.touched.has('ref_no') && order.order_no) state.order.ref_no = order.order_no;
  if (!state.touched.has('ref_dt') && order.order_date) {
    const iso = formDateToIso(order.order_date);
    if (iso) state.order.ref_dt = iso;
  }
  if (!state.touched.has('buyer') && order.buyer) {
    if (order.buyer.selected) state.order.buyer = order.buyer.selected;
    state.order.buyer_candidates = order.buyer.candidates || [];
    state.order.buyer_note = order.buyer.note || '';
  }

  if (!state.activePage || !state.pages.some(p => p.name === state.activePage)) {
    const firstDone = state.pages.find(p => p.status === 'done') || state.pages[0];
    state.activePage = firstDone ? firstDone.name : null;
    changed = true;
  }
  if (changed) savePersist();
  return changed;
}

/* ══════════════════════════════════════════════ rendering ═══════════════ */

/** The bar under the topbar is on whenever work is outstanding — an upload
 *  in flight, or any page still queued/running. Without it, a reviewer who
 *  has scrolled away from the page being read sees a completely static
 *  screen for the minute it takes. */
function renderProgress() {
  const busy = state.uploading > 0 || state.images.some(i => i.status === 'queued' || i.status === 'running');
  $('#progress-bar').hidden = !busy;
}

function renderAll() {
  renderProgress();
  renderPageTabs();
  renderOrderCard();
  renderPages();
  renderTotals();
  showViewer(state.activePage);
  savePersist();
}

function renderPageTabs() {
  const tabs = $('#page-tabs');
  tabs.textContent = '';
  for (const page of state.pages) {
    const img = state.images.find(i => i.name === page.name);
    const status = img ? img.status : page.status;
    const tab = el('button', {
      class: `page-tab${page.name === state.activePage ? ' active' : ''}`,
      type: 'button',
      onclick: () => { state.activePage = page.name; renderPageTabs(); showViewer(page.name); },
      title: img ? img.original_name : page.name,
    },
      el('span', { class: `dot ${status}` }),
      page.name);
    tabs.append(tab);
  }
}

function renderOrderCard() {
  $('#party-name').value = state.order.party_name || '';
  $('#ref-no').value = state.order.ref_no || '';
  $('#ref-dt').value = state.order.ref_dt || '';
  $('#order-dt').value = state.order.order_dt || isoToday();

  const note = $('#party-note');
  if (state.order.buyer_note) {
    note.hidden = false;
    note.textContent = `⚠️ ${state.order.buyer_note}`;
  } else {
    note.hidden = true;
  }

  const chip = $('#order-summary-chip');
  chip.textContent = state.order.buyer ? state.order.buyer.label : 'buyer not selected';

  mountBuyerPicker();
}

function renderPages() {
  const container = $('#pages-container');
  container.textContent = '';
  for (const page of state.pages) container.append(renderPageCard(page));
}

function renderPageCard(page) {
  const img = state.images.find(i => i.name === page.name);
  const status = img ? img.status : page.status;

  const card = el('article', { class: `card page-card${page.collapsed ? ' collapsed' : ''}`, dataset: { page: page.name } });

  const head = el('header', { class: 'card-head' },
    el('button', { class: 'collapse-btn', type: 'button', html: '<svg viewBox="0 0 24 24"><path d="m6 9 6 6 6-6"/></svg>' }),
    el('h2', { text: page.name }),
    el('span', { class: 'chip', text: statusChipText(status, page) }),
    el('span', { class: 'head-spacer' }),
    el('button', {
      class: 'tool-btn', type: 'button', text: 'Show photo',
      onclick: (e) => { e.stopPropagation(); state.activePage = page.name; renderPageTabs(); showViewer(page.name); },
    }),
    status === 'error' ? el('button', {
      class: 'tool-btn', type: 'button', text: 'Retry',
      onclick: async (e) => { e.stopPropagation(); await retryPage(page.name); },
    }) : null,
    el('button', {
      class: 'tool-btn', type: 'button', text: 'Remove page',
      onclick: async (e) => { e.stopPropagation(); await removePage(page.name); },
    }),
  );
  head.addEventListener('click', (e) => {
    if (e.target.closest('button') && !e.target.closest('.collapse-btn')) return;
    page.collapsed = !page.collapsed;
    card.classList.toggle('collapsed', page.collapsed);
    savePersist();
  });
  card.append(head);

  const body = el('div', { class: 'card-body' });
  if (status !== 'done') {
    body.append(renderPageStatus(page, status, img));
  } else {
    body.append(renderPageToolbar(page), buildTableRegion(page), renderPageFoot(page));
    if (page.notes && page.notes.length) {
      body.append(el('div', { class: 'page-notes' },
        el('strong', { text: 'Notes on this page: ' }),
        el('ul', {}, ...page.notes.map(n => el('li', { text: n })))));
    }
    if (page.template_learning && page.template_learning.matched) {
      const tl = page.template_learning;
      const times = tl.occurrences === 1 ? 'once' : `${tl.occurrences} times`;
      const corrected = tl.corrections_applied
        ? `, ${tl.corrections_applied} correction${tl.corrections_applied === 1 ? '' : 's'} applied automatically from past orders`
        : '';
      body.append(el('div', { class: 'page-notes' },
        el('strong', { text: '📋 Seen before: ' }),
        el('span', { text: `this template has been reviewed ${times} before${corrected}.` })));
    }
  }
  card.append(body);
  return card;
}

function statusChipText(status, page) {
  if (status === 'queued') return 'waiting…';
  if (status === 'running') return 'reading…';
  if (status === 'error') return 'failed';
  const cells = page.rows.reduce((n, r) => n + Object.values(r.quantities).filter(v => v > 0).length, 0);
  return `${page.rows.length} rows · ${cells} cells`;
}

function renderPageStatus(page, status, img) {
  if (status === 'error') {
    return el('div', { class: 'page-status' },
      el('div', { class: 'note note-warn', text: `Couldn’t read this page: ${img?.message || page.message || 'unknown error'}` }),
      el('button', { class: 'btn btn-ghost', type: 'button', text: 'Try again', onclick: () => retryPage(page.name) }));
  }
  const since = img && img.started_at ? Date.parse(img.started_at) : null;
  return el('div', { class: 'page-status' },
    el('div', { class: 'spinner' }),
    el('div', {},
      status === 'queued' ? 'Waiting for the reader…' : 'Reading the form — about a minute.',
      status === 'running' && since
        ? el('span', { class: 'elapsed', dataset: { since: String(since) }, text: ' 0s' })
        : null));
}

/** One shared ticker for every "reading…" counter on the page. Polling only
 *  re-renders when something actually changed, so without this the elapsed
 *  time would sit frozen at whatever it was when the page last rendered. */
setInterval(() => {
  for (const node of $$('.elapsed')) {
    const secs = Math.max(0, Math.round((Date.now() - Number(node.dataset.since)) / 1000));
    node.textContent = ` ${secs}s`;
  }
}, 1000);

/* ─────────────────────────────── toolbar ─────────────────────────────── */

function renderPageToolbar(page) {
  const bar = el('div', { class: 'page-toolbar' });

  bar.append(el('button', {
    class: 'tool-btn', type: 'button', text: '+ size column',
    onclick: (e) => openAddSizePopover(e.currentTarget, page),
  }));
  bar.append(el('button', {
    class: 'tool-btn', type: 'button', text: '+ row',
    onclick: () => { addRow(page); afterEdit(page); },
  }));
  bar.append(el('span', { class: 'sep' }));
  bar.append(el('button', {
    class: 'tool-btn', type: 'button', text: 'Hide empty sizes',
    title: 'Remove size columns no row on this page uses',
    onclick: () => { pruneEmptySizes(page); afterEdit(page); },
  }));
  return bar;
}

/** The grid is its own scroll region (that's what makes the size-header row
 *  and item column freeze), so its horizontal scrollbar sits at the *bottom*
 *  of a viewport-tall table — you'd have to scroll to the last row just to
 *  reach sideways. This builds the table region with a second, mirrored
 *  scrollbar pinned above it, so the far-right size columns are always one
 *  drag away from wherever you are in the rows. */
function buildTableRegion(page) {
  const wrap = el('div', { class: 'table-wrap' }, renderGrid(page));
  const spacer = el('div', { class: 'hscroll-spacer' });
  const mirror = el('div', { class: 'hscroll-mirror' }, spacer);

  let syncing = false;
  const link = (from, to) => from.addEventListener('scroll', () => {
    if (syncing) return;             // each scroll sets the other, which would bounce back
    syncing = true;
    to.scrollLeft = from.scrollLeft;
    requestAnimationFrame(() => { syncing = false; });
  });
  link(mirror, wrap);
  link(wrap, mirror);

  // Width has to come from the laid-out table, so it's measured after the
  // region is in the document; re-measured on resize since the split pane
  // and the image-pane collapse both change the available width.
  const resize = () => {
    const table = wrap.querySelector('table');
    if (!table) return;
    spacer.style.width = `${table.scrollWidth}px`;
    mirror.classList.toggle('needed', table.scrollWidth > wrap.clientWidth + 1);
  };
  requestAnimationFrame(resize);
  if (window.ResizeObserver) new ResizeObserver(resize).observe(wrap);

  return el('div', { class: 'table-region' }, mirror, wrap);
}

function renderPageFoot(page) {
  const totals = pageTotals(page);
  return el('div', { class: 'page-foot' },
    el('span', { class: 'chip', text: `${totals.rows} rows` }),
    el('span', { class: 'chip', text: `${totals.cells} filled cells` }),
    el('span', { class: 'chip', text: `${totals.qty.toLocaleString()} pieces` }),
    totals.unbound ? el('span', { class: 'pill pill-bad', text: `${totals.unbound} row(s) not linked to a product` }) : null,
  );
}

/* ─────────────────────────────── the grid ────────────────────────────── */

function renderGrid(page) {
  const table = el('table', { class: 'grid' });

  // Column widths live in the stylesheet, keyed off these classes. The table
  // is table-layout: fixed, so this <colgroup> is what actually sizes the
  // grid — see the comment on table.grid in styles.css.
  table.append(el('colgroup', {},
    el('col', { class: 'c-idx' }),
    el('col', { class: 'c-item' }),
    ...page.headers.map(() => el('col', { class: 'c-size' })),
    el('col', { class: 'c-total' }),
    el('col', { class: 'c-printed' }),
    el('col', { class: 'c-actions' })));

  const headRow = el('tr', {},
    el('th', { class: 'col-idx', text: '#' }),
    el('th', { class: 'col-item', text: 'Item / product' }),
  );
  for (const size of page.headers) {
    headRow.append(el('th', { class: 'col-size', dataset: { size } },
      el('div', { class: 'hdr' },
        el('span', { class: 'size-text', text: size }),
        el('button', {
          class: 'kill', type: 'button', title: `Remove the ${size} column`,
          onclick: () => { removeSize(page, size); afterEdit(page); },
        }, '✕'))));
  }
  headRow.append(el('th', { class: 'col-total', text: 'Total' }));
  // "Form", not "On form": the header text is what actually sets this
  // column's width (white-space: nowrap), and every pixel here is a pixel
  // of horizontal scrolling. The tooltip carries the full meaning.
  headRow.append(el('th', { class: 'col-printed', text: 'Form', title: 'The row total printed or circled on the form itself, as read by the pipeline' }));
  headRow.append(el('th', { class: 'col-actions', text: '' }));
  table.append(el('thead', {}, headRow));

  const tbody = el('tbody');
  page.rows.forEach((row, index) => tbody.append(renderRow(page, row, index)));
  table.append(tbody);

  const totals = pageTotals(page);
  const footRow = el('tr', {},
    el('td', { class: 'label', colspan: '2', text: 'Column totals' }));
  for (const size of page.headers) {
    const sum = page.rows.reduce((n, r) => n + (Number(r.quantities[size]) || 0), 0);
    footRow.append(el('td', { text: sum || '' }));
  }
  footRow.append(el('td', { text: totals.qty.toLocaleString() }), el('td', {}), el('td', {}));
  table.append(el('tfoot', {}, footRow));
  return table;
}

function renderRow(page, row, index) {
  const tr = el('tr', { class: 'row', dataset: { uid: row.uid } });
  tr.append(el('td', { class: 'cell-idx', text: String(index + 1) }));
  tr.append(renderItemCell(page, row));

  for (const size of page.headers) {
    tr.append(renderQtyCell(page, row, size));
  }

  const total = rowTotal(row);
  const mismatch = row.printed_total != null && total !== row.printed_total;
  tr.append(el('td', { class: `cell-total${mismatch ? ' mismatch' : ''}`, dataset: { role: 'row-total' } },
    el('strong', { text: total ? total.toLocaleString() : '' })));
  tr.append(el('td', {
    class: 'cell-printed',
    title: row.printed_total != null ? 'The row total written on the form' : 'No row total on the form',
    text: row.printed_total != null ? String(row.printed_total) : '—',
  }));
  tr.append(el('td', { class: 'cell-actions' }, renderRowTools(page, row, index)));

  applyRowState(tr, page, row);
  return tr;
}

function renderItemCell(page, row) {
  const cell = el('td', { class: 'cell-item' });

  // One line by default. The product binding sits on a second line only in
  // "show products" mode — carrying it on every row all the time doubled
  // the height of a row that otherwise needs 30px, which on a 20-row form
  // is more than a screenful of scrolling spent on text nobody is reading
  // during the quantity-checking pass.
  const flags = el('span', { class: 'item-flags', dataset: { role: 'flags' } });
  cell.append(el('div', { class: 'item-top' },
    el('input', {
      class: 'item-name-input', type: 'text', value: row.item || '',
      placeholder: 'Item name',
      oninput: (e) => { row.item = e.target.value; savePersist(); },
    }),
    flags));
  paintItemFlags(flags, page, row);

  const boundLine = el('div', { class: 'bound-line', dataset: { role: 'bound' } });
  cell.append(boundLine);
  paintBoundLine(boundLine, page, row);

  return cell;
}

/** What stays visible on the row's single line: anything demanding action.
 *  A correctly-bound row shows nothing here — absence is the "fine" signal,
 *  which keeps the grid quiet and makes the exceptions obvious. */
function paintItemFlags(node, page, row) {
  node.textContent = '';
  const mismatch = liveSizeMismatch(row);
  if (!row.product) {
    node.append(el('button', {
      class: 'chip-action chip-bad', type: 'button', text: 'link',
      title: 'Not linked to a catalog product yet — click to choose',
      onclick: (e) => openProductPicker(e.currentTarget, page, row),
    }));
  } else if (mismatch.length) {
    node.append(el('button', {
      class: 'chip-action chip-warn', type: 'button', text: 'sizes',
      title: `${row.product.label} doesn’t come in size ${mismatch.join(', ')}`,
      onclick: (e) => openProductPicker(e.currentTarget, page, row),
    }));
  }
  for (const [kind, glyph, text] of rowMarkers(row)) {
    node.append(el('span', { class: `marker marker-${kind}`, text: glyph, title: text }));
  }
}

/** Flags and catalog notes used to render as paragraphs inside the item
 *  cell, which made every flagged row three or four lines tall — on a
 *  20-row form that is most of the scrolling. They are markers now: the row
 *  still carries its coloured left edge and its highlighted cells, and the
 *  full text is one hover away. Nothing is lost, it just stops shouting. */
function rowMarkers(row) {
  const markers = [];
  if (row.struck_out) {
    markers.push(['struck', '⌫', 'Struck through on the form, so its quantities were cleared. Delete the row, or re-enter them if that was wrong.']);
  }
  if (row.flag && row.flag.note) {
    markers.push(['warn', '⚠', friendlyFlag(row.flag)]);
  }
  const dbNote = dbNoteText(row.db_note);
  if (dbNote) markers.push(['info', 'ℹ', dbNote]);
  return markers;
}

function paintBoundLine(line, page, row) {
  line.textContent = '';
  if (row.product) {
    line.append(el('span', { class: 'bound-name', text: row.product.label, title: `${row.product.label} · catalog sizes ${row.product.size_list.join(', ')}` }));
    // A binding whose product can't cover the sizes this row ordered is
    // still shown (it names the exact bad cell), but never as "auto" —
    // that tag is an invitation to skip checking, and this one needs it.
    const mismatch = liveSizeMismatch(row);
    if (mismatch.length) line.append(el('span', { class: 'tag tag-check', text: 'check sizes', title: `This product doesn’t come in size ${mismatch.join(', ')}` }));
    else if (row.auto_bound) line.append(el('span', { class: 'tag tag-auto', text: 'auto' }));
    line.append(el('button', { class: 'link-btn', type: 'button', text: 'change', onclick: (e) => openProductPicker(e.currentTarget, page, row) }));
  } else {
    line.append(el('span', { class: 'bound-name none', text: 'not linked to a product' }));
    line.append(el('button', { class: 'link-btn', type: 'button', text: 'choose', onclick: (e) => openProductPicker(e.currentTarget, page, row) }));
  }
  if (row.type) line.append(el('span', { class: 'tag tag-style', text: row.type }));
}

/** Sizes this row orders that its bound product doesn't come in, computed
 *  live rather than read from the payload's `size_mismatch` — the reviewer
 *  is editing both sides of that comparison (quantities and the product),
 *  so a value captured at extraction time goes stale on the first edit. */
function liveSizeMismatch(row) {
  if (!row.product) return [];
  return Object.keys(row.quantities)
    .filter(size => Number(row.quantities[size]) > 0 && !(size in row.product.sizes))
    .sort(sizeSort);
}

function renderQtyCell(page, row, size) {
  const value = row.quantities[size];
  const flagged = row.flag && Array.isArray(row.flag.sizes) && row.flag.sizes.includes(size);
  const cell = el('td', { class: 'cell-qty', dataset: { size, uid: row.uid } });
  const input = el('input', {
    type: 'number', min: '0', step: '1', inputmode: 'numeric',
    value: value ? String(value) : '',
    dataset: { size, uid: row.uid },
    oninput: (e) => onQtyInput(page, row, size, e.target),
    onfocus: (e) => e.target.select(),
    onkeydown: onQtyKeydown,
  });
  cell.append(input);
  if (flagged) cell.classList.add('flagged');
  paintQtyCell(cell, page, row, size);
  return cell;
}

/** A cell is "invalid" when its product simply doesn't come in that size —
 *  that's the one client-side check that maps exactly to something the
 *  upload will refuse, so it's worth showing before the reviewer gets there. */
function paintQtyCell(cell, page, row, size) {
  const value = Number(row.quantities[size]) || 0;
  cell.classList.toggle('filled', value > 0);
  const bad = value > 0 && row.product && !(size in row.product.sizes);
  cell.classList.toggle('invalid', Boolean(bad));
  cell.title = bad ? `${row.product.label} doesn’t come in size ${size} (catalog: ${row.product.size_list.join(', ')})` : '';
}

function applyRowState(tr, page, row) {
  const hasQty = Object.values(row.quantities).some(v => Number(v) > 0);
  tr.classList.toggle('unbound', hasQty && !row.product);
  tr.classList.toggle('flagged', Boolean(row.flag));
}

function renderRowTools(page, row, index) {
  const wrap = el('div', { class: 'row-tools' });
  wrap.append(el('button', {
    class: 'icon-btn', type: 'button', title: 'Move this row’s quantities one size column left', text: '←',
    onclick: () => { shiftRowHorizontally(page, row, -1); afterEdit(page); },
  }));
  wrap.append(el('button', {
    class: 'icon-btn', type: 'button', title: 'Move this row’s quantities one size column right', text: '→',
    onclick: () => { shiftRowHorizontally(page, row, +1); afterEdit(page); },
  }));
  wrap.append(el('button', {
    class: 'icon-btn', type: 'button', title: 'More actions', text: '⋯',
    onclick: (e) => openRowMenu(e.currentTarget, page, row, index),
  }));
  return wrap;
}

/* ══════════════════════════════════════════════ editing ═════════════════ */

function onQtyInput(page, row, size, input) {
  const raw = input.value.trim();
  if (raw === '') delete row.quantities[size];
  else {
    const n = Math.max(0, Math.round(Number(raw) || 0));
    if (n === 0) delete row.quantities[size];
    else row.quantities[size] = n;
  }
  const cell = input.closest('td');
  paintQtyCell(cell, page, row, size);
  const tr = input.closest('tr');
  applyRowState(tr, page, row);
  const boundLine = tr.querySelector('[data-role="bound"]');
  if (boundLine) paintBoundLine(boundLine, page, row);
  const totalCell = tr.querySelector('[data-role="row-total"]');
  const total = rowTotal(row);
  totalCell.firstChild.textContent = total ? total.toLocaleString() : '';
  totalCell.classList.toggle('mismatch', row.printed_total != null && total !== row.printed_total);
  refreshFooterTotals(page);
  renderTotals();
  savePersist();
}

/** Arrow keys / Enter move between quantity cells the way a spreadsheet
 *  does — a reviewer entering a dense table shouldn't have to reach for the
 *  mouse or tab through the product pickers between rows. */
function onQtyKeydown(e) {
  const key = e.key;
  if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Enter'].includes(key)) return;
  const input = e.target;
  const cell = input.closest('td');
  const tr = input.closest('tr');
  const table = tr.closest('table');
  const cellIndex = Array.from(tr.children).indexOf(cell);
  const rows = Array.from(table.querySelectorAll('tbody tr'));
  const rowIndex = rows.indexOf(tr);

  const atStart = input.selectionStart === 0 && input.selectionEnd === 0;
  const atEnd = input.selectionStart === input.value.length && input.selectionEnd === input.value.length;
  if (key === 'ArrowLeft' && !atStart) return;
  if (key === 'ArrowRight' && !atEnd) return;

  let target = null;
  if (key === 'ArrowUp') target = rows[rowIndex - 1]?.children[cellIndex];
  else if (key === 'ArrowDown' || key === 'Enter') target = rows[rowIndex + 1]?.children[cellIndex];
  else if (key === 'ArrowLeft') target = tr.children[cellIndex - 1];
  else if (key === 'ArrowRight') target = tr.children[cellIndex + 1];

  const next = target?.querySelector?.('input[type="number"]');
  if (next) {
    e.preventDefault();
    next.focus();
    next.select();
  }
}

/** Column shift: the pipeline's single most common residual error is a whole
 *  row read one column off (see CLAUDE.md's "Known limitations"). Shifting
 *  within the page's own header order — rather than by numeric size — is
 *  what makes this correct on forms whose columns aren't evenly spaced. */
function shiftRowHorizontally(page, row, direction) {
  const headers = page.headers;
  const next = {};
  for (const [size, qty] of Object.entries(row.quantities)) {
    const at = headers.indexOf(size);
    if (at === -1) { next[size] = qty; continue; }   // a size not on the grid stays put
    const to = at + direction;
    if (to < 0 || to >= headers.length) continue;    // pushed off the edge: dropped, visibly
    next[headers[to]] = qty;
  }
  row.quantities = next;
}

/** Vertical shift: a block of rows attributed one row too high/low. Every
 *  row from `fromIndex` down moves its quantities to its neighbour, so the
 *  reviewer clicks once on the first row that's wrong rather than re-typing
 *  everything below it. Item names and product bindings stay put — it is
 *  the quantities that drifted, not the row labels. */
function shiftQuantitiesVertically(page, fromIndex, direction) {
  const rows = page.rows;
  if (direction < 0) {
    for (let i = fromIndex; i < rows.length - 1; i++) rows[i].quantities = rows[i + 1].quantities;
    rows[rows.length - 1].quantities = {};
  } else {
    for (let i = rows.length - 1; i > fromIndex; i--) rows[i].quantities = rows[i - 1].quantities;
    rows[fromIndex].quantities = {};
  }
}

function addSize(page, size) {
  size = String(size).trim().toUpperCase();
  if (!size) return false;
  if (page.headers.includes(size)) { toast(`Size ${size} is already a column`); return false; }
  page.headers.push(size);
  page.headers.sort(sizeSort);
  return true;
}

function removeSize(page, size) {
  const used = page.rows.filter(r => Number(r.quantities[size]) > 0);
  if (used.length && !confirm(`Size ${size} has quantities on ${used.length} row(s). Remove the column and those quantities?`)) return;
  page.headers = page.headers.filter(s => s !== size);
  for (const row of page.rows) delete row.quantities[size];
}

function pruneEmptySizes(page) {
  const before = page.headers.length;
  page.headers = page.headers.filter(size => page.rows.some(r => Number(r.quantities[size]) > 0));
  const removed = before - page.headers.length;
  toast(removed ? `Removed ${removed} empty size column(s)` : 'Every size column is in use');
}

function addRow(page, afterIndex = null) {
  const row = {
    uid: `${page.name}#new${Date.now()}${Math.random().toString(36).slice(2, 6)}`,
    page: page.name, item: '', type: '', quantities: {},
    printed_total: null, struck_out: false, flag: null, db_note: null,
    bsid: '', product: null, auto_bound: false, candidates: [],
  };
  if (afterIndex === null) page.rows.push(row);
  else page.rows.splice(afterIndex + 1, 0, row);
  return row;
}

/** Adds the bound product's own catalog sizes as columns.
 *
 *  This is the direct fix for the case in CLAUDE.md where a product's real
 *  sizes have no relation to the form's printed grid at all (the kids' line
 *  sized 35–55 on a form printed 45–105): the pipeline has nowhere to put
 *  those numbers, so the reviewer needs the columns to exist before the
 *  quantities can be entered. */
function addProductSizes(page, row) {
  if (!row.product) { toast('Link this row to a product first', 'bad'); return; }
  const added = row.product.size_list.filter(s => !page.headers.includes(s));
  if (!added.length) { toast('Every size this product comes in is already a column'); return; }
  for (const size of added) page.headers.push(size);
  page.headers.sort(sizeSort);
  toast(`Added size column(s): ${added.join(', ')}`);
}

function afterEdit(page) {
  const card = $(`.page-card[data-page="${CSS.escape(page.name)}"]`);
  if (card) {
    const body = card.querySelector('.card-body');
    body.textContent = '';
    body.append(renderPageToolbar(page), buildTableRegion(page), renderPageFoot(page));
    if (page.notes && page.notes.length) {
      body.append(el('div', { class: 'page-notes' },
        el('strong', { text: 'Notes on this page: ' }),
        el('ul', {}, ...page.notes.map(n => el('li', { text: n })))));
    }
    if (page.template_learning && page.template_learning.matched) {
      const tl = page.template_learning;
      const times = tl.occurrences === 1 ? 'once' : `${tl.occurrences} times`;
      const corrected = tl.corrections_applied
        ? `, ${tl.corrections_applied} correction${tl.corrections_applied === 1 ? '' : 's'} applied automatically from past orders`
        : '';
      body.append(el('div', { class: 'page-notes' },
        el('strong', { text: '📋 Seen before: ' }),
        el('span', { text: `this template has been reviewed ${times} before${corrected}.` })));
    }
  }
  renderTotals();
  savePersist();
}

function refreshFooterTotals(page) {
  const card = $(`.page-card[data-page="${CSS.escape(page.name)}"]`);
  if (!card) return;
  const foot = card.querySelector('tfoot tr');
  if (foot) {
    page.headers.forEach((size, i) => {
      const sum = page.rows.reduce((n, r) => n + (Number(r.quantities[size]) || 0), 0);
      const td = foot.children[i + 1];
      if (td) td.textContent = sum || '';
    });
    const totals = pageTotals(page);
    const last = foot.children[page.headers.length + 1];
    if (last) last.textContent = totals.qty.toLocaleString();
  }
  const pageFoot = card.querySelector('.page-foot');
  if (pageFoot) pageFoot.replaceWith(renderPageFoot(page));
}

/* ─────────────────────────────── row menu ────────────────────────────── */

let openMenu = null;

function closeMenus() {
  if (openMenu) { openMenu.remove(); openMenu = null; }
}
document.addEventListener('click', (e) => {
  if (openMenu && !openMenu.contains(e.target)) closeMenus();
}, true);

function openRowMenu(anchor, page, row, index) {
  closeMenus();
  const items = [
    ['↑  Shift quantities up from this row', () => { shiftQuantitiesVertically(page, index, -1); afterEdit(page); }],
    ['↓  Shift quantities down from this row', () => { shiftQuantitiesVertically(page, index, +1); afterEdit(page); }],
    ['⇅  Swap quantities with the row above', () => {
      if (index === 0) return toast('This is already the first row');
      const above = page.rows[index - 1];
      [above.quantities, row.quantities] = [row.quantities, above.quantities];
      afterEdit(page);
    }],
    ['⊞  Add this product’s sizes as columns', () => { addProductSizes(page, row); afterEdit(page); }],
    ['＋ Insert a blank row below', () => { addRow(page, index); afterEdit(page); }],
    ['⌫  Clear this row’s quantities', () => { row.quantities = {}; afterEdit(page); }],
    ['🗑  Delete this row', () => {
      const total = rowTotal(row);
      if (total && !confirm(`Delete “${row.item || 'this row'}” (${total} pieces)?`)) return;
      page.rows.splice(index, 1);
      afterEdit(page);
    }],
  ];

  const menu = el('div', { class: 'picker-menu' },
    ...items.map(([label, fn]) => el('div', {
      class: 'picker-option',
      onclick: () => { closeMenus(); fn(); },
    }, el('span', { class: 'opt-main', text: label }))));
  positionMenu(menu, anchor, 260);
  openMenu = menu;
  document.body.append(menu);
}

function positionMenu(menu, anchor, width) {
  const rect = anchor.getBoundingClientRect();
  menu.style.width = `${width}px`;
  menu.style.left = `${Math.min(rect.left, window.innerWidth - width - 12)}px`;
  const below = window.innerHeight - rect.bottom;
  if (below > 220) menu.style.top = `${rect.bottom + 4}px`;
  else menu.style.bottom = `${window.innerHeight - rect.top + 4}px`;
}

function openAddSizePopover(anchor, page) {
  closeMenus();
  const input = el('input', {
    class: 'control', type: 'text', placeholder: 'e.g. 110 or XL',
    style: 'width:100%',
    onkeydown: (e) => {
      if (e.key === 'Enter') { commit(); }
      if (e.key === 'Escape') closeMenus();
    },
  });
  const commit = () => {
    const value = input.value;
    closeMenus();
    if (addSize(page, value)) afterEdit(page);
  };
  const menu = el('div', { class: 'picker-menu', style: 'padding:10px' },
    el('div', { class: 'field' },
      el('span', { class: 'field-label', text: 'Add a size column' }),
      input,
      el('span', { class: 'field-help', text: 'For a size the form has but the reader missed.' })),
    el('button', { class: 'btn btn-primary', type: 'button', text: 'Add column', style: 'width:100%;margin-top:8px', onclick: commit }));
  positionMenu(menu, anchor, 240);
  openMenu = menu;
  document.body.append(menu);
  input.focus();
}

/* ══════════════════════════════════════════════ pickers ═════════════════ */

/** A search-as-you-type combobox over a server endpoint.
 *  Used for both products and buyers — the two fields where picking the
 *  right one out of thousands by scrolling would be hopeless. */
function openSearchPicker({ anchor, endpoint, width = 420, seed = '', preload = [], onPick, footer }) {
  closeMenus();
  const input = el('input', { class: 'control picker-input', type: 'text', placeholder: 'Type to search…', value: seed });
  const list = el('div');
  const menu = el('div', { class: 'picker-menu' },
    el('div', { style: 'padding:4px 4px 8px' }, input),
    list,
    footer || null);
  positionMenu(menu, anchor, width);
  openMenu = menu;
  document.body.append(menu);
  input.focus();
  input.select();

  let active = 0;
  let results = [];

  function paint() {
    list.textContent = '';
    if (!results.length) {
      list.append(el('div', { class: 'picker-empty', text: 'No matches' }));
      return;
    }
    results.forEach((row, i) => {
      list.append(el('div', {
        class: `picker-option${i === active ? ' active' : ''}`,
        onclick: () => { closeMenus(); onPick(row); },
        // Only toggles the 'active' class -- must NOT rebuild the list
        // (no full paint()) on hover. A real mouse click is mousemove →
        // mouseenter → mousedown → mouseup → click; if mouseenter tears
        // down and recreates every option div, the element the cursor is
        // about to click no longer exists by the time the click fires,
        // and the click silently lands on whatever's underneath instead
        // (confirmed live: a synthetic click dispatched with no mouse
        // movement selects correctly every time; a real mouse click does
        // not). Keyboard nav (ArrowUp/ArrowDown) still calls the full
        // paint() below since there's no DOM element in-flight to lose.
        onmouseenter: (e) => {
          const prev = list.querySelector('.picker-option.active');
          if (prev) prev.classList.remove('active');
          active = i;
          e.currentTarget.classList.add('active');
        },
      },
        el('div', { class: 'opt-main' }, row.label),
        el('div', { class: 'opt-sub', text: row.sub || '' })));
    });
  }

  const search = debounce(async (q) => {
    try {
      const data = await api(`${endpoint}?q=${encodeURIComponent(q)}&limit=25`);
      results = (data.results || []).map(decorate);
      active = 0;
      paint();
    } catch (err) {
      list.textContent = '';
      list.append(el('div', { class: 'picker-empty', text: err.message }));
    }
  }, 160);

  function decorate(row) {
    if (row.bsid) {
      return { ...row, sub: `${row.bstyle ? `style ${row.bstyle} · ` : ''}sizes ${row.size_list.join(', ') || '—'}` };
    }
    return { ...row, sub: [row.city, row.state, row.code].filter(Boolean).join(' · ') };
  }

  results = preload.map(decorate);
  paint();
  if (seed) search(seed);

  input.addEventListener('input', () => search(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { active = Math.min(active + 1, results.length - 1); paint(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { active = Math.max(active - 1, 0); paint(); e.preventDefault(); }
    else if (e.key === 'Enter') { if (results[active]) { closeMenus(); onPick(results[active]); } e.preventDefault(); }
    else if (e.key === 'Escape') closeMenus();
  });
}

function openProductPicker(anchor, page, row) {
  openSearchPicker({
    anchor,
    endpoint: '/api/products',
    width: 460,
    seed: `${row.item || ''} ${row.type || ''}`.trim(),
    preload: row.candidates || [],
    footer: row.product ? el('div', { style: 'padding:6px 4px 2px;border-top:1px solid var(--border);margin-top:4px' },
      el('button', {
        class: 'link-btn', type: 'button', text: 'Unlink this row',
        onclick: () => { closeMenus(); row.product = null; row.bsid = ''; row.auto_bound = false; afterEdit(page); },
      })) : null,
    onPick: (product) => {
      row.product = product;
      row.bsid = product.bsid;
      row.auto_bound = false;
      if (product.bstyle) row.type = product.bstyle;
      afterEdit(page);
      const missing = Object.keys(row.quantities).filter(s => Number(row.quantities[s]) > 0 && !(s in product.sizes));
      if (missing.length) {
        toast(`${product.label} doesn’t come in size ${missing.join(', ')} — fix those cells before uploading`, 'bad');
      }
    },
  });
}

function mountBuyerPicker() {
  const host = $('#buyer-picker');
  host.textContent = '';
  const buyer = state.order.buyer;
  const button = el('button', {
    class: 'control', type: 'button',
    style: 'text-align:left;width:100%;cursor:pointer',
    text: buyer ? buyer.label : 'Search for the buyer…',
  });
  button.style.color = buyer ? '' : 'var(--text-faint)';
  button.addEventListener('click', () => openSearchPicker({
    anchor: button,
    endpoint: '/api/buyers',
    width: 420,
    seed: state.order.party_name || '',
    preload: state.order.buyer_candidates || [],
    onPick: (b) => {
      state.order.buyer = b;
      state.touched.add('buyer');
      renderOrderCard();
      renderTotals();
      savePersist();
    },
  }));
  host.append(button);

  const help = $('#buyer-help');
  if (buyer) help.textContent = '';
  else if (state.order.party_name) help.textContent = `Form says “${state.order.party_name}”.`;
  else help.textContent = '';
}

/* ══════════════════════════════════════════════ notes text ══════════════ */

function friendlyFlag(flag) {
  const sizes = (flag.sizes || []).slice().sort(sizeSort).join(', ');
  const note = flag.note || '';
  // Substring, not startsWith: a template-memory correction is always
  // merged in LAST (extract_ollama_cloud.py applies it after every other
  // flag-producing stage), so its note is never the first segment
  // _merge_flag() assembled once another mechanism already flagged the
  // same row -- startsWith would silently miss it in that case. Applies
  // to the two existing prefixes too, which had the identical latent gap.
  if (note.includes('Hybrid OCR+VLM:') || note.includes('Catalog check:') || note.includes('Template memory:')) return `⚠️ ${note}`;
  switch (flag.status) {
    case 'unresolved': return `⚠️ Check sizes ${sizes} against the photo — two automatic checks disagreed and neither could be confirmed.`;
    case 'resolved': return `⚠️ Worth a quick look at sizes ${sizes} — two checks disagreed and the one matching the row total or catalog was used.`;
    case 'auto_corrected': return `⚠️ Verify sizes ${sizes} — the values looked shifted and were corrected against the product catalog.`;
    case 'template_corrected': return `✓ Sizes ${sizes} auto-corrected from a previously confirmed reading of this exact form template — quick check recommended.`;
    case 'unverified': return 'This row’s quantities weren’t double-checked — verify against the photo.';
    default: return note;
  }
}

function dbNoteText(note) {
  if (!note || !note.match) return '';
  const parts = [];
  if (note.type_filled_from_catalog) parts.push(`style “${note.type_filled_from_catalog}” filled in from the catalog`);
  if (note.resolved_sizes) parts.push(`letter sizes resolved: ${Object.entries(note.resolved_sizes).map(([k, v]) => `${k}=${v}`).join(', ')}`);
  if (note.unresolved_sizes) parts.push(`couldn’t resolve size letter(s): ${note.unresolved_sizes.join(', ')}`);
  if (note.style_mismatch) parts.push(`catalog styles for this product: ${note.style_mismatch.join(', ')}`);
  if (note.code_not_in_catalog) parts.push(`product code “${note.code_not_in_catalog}” isn’t in the catalog`);
  return parts.length ? `ℹ️ ${parts.join(' · ')}` : '';
}

/* ══════════════════════════════════════════════ totals ══════════════════ */

function rowTotal(row) {
  return Object.values(row.quantities).reduce((n, v) => n + (Number(v) || 0), 0);
}

function pageTotals(page) {
  let cells = 0, qty = 0, unbound = 0;
  for (const row of page.rows) {
    const rowQty = rowTotal(row);
    qty += rowQty;
    cells += Object.values(row.quantities).filter(v => Number(v) > 0).length;
    if (rowQty > 0 && !row.product) unbound++;
  }
  return { rows: page.rows.length, cells, qty, unbound };
}

function allRows() {
  return state.pages.flatMap(p => p.rows.map(r => ({ page: p, row: r })));
}

function renderTotals() {
  let rows = 0, cells = 0, qty = 0;
  for (const page of state.pages) {
    const t = pageTotals(page);
    rows += t.rows; cells += t.cells; qty += t.qty;
  }
  $('#total-rows').textContent = rows.toLocaleString();
  $('#total-cells').textContent = cells.toLocaleString();
  $('#total-qty').textContent = qty.toLocaleString();
  renderValidationSummary(localProblems());
}

/** Header-level blockers — things the grid itself can't show, so they're
 *  the client's own responsibility to report. */
function headerProblems() {
  const problems = [];
  if (!state.order.buyer) problems.push('Pick the buyer.');
  if (!state.operatorId) problems.push('Pick the operator (top right).');
  if (!allRows().some(({ row }) => rowTotal(row) > 0)) problems.push('No row has any quantity yet.');
  return problems;
}

/** Row-level pre-check, deliberately mirroring exactly what
 *  db.resolve_lines() can also refuse — so "Check" never promises something
 *  the upload then rejects, and the two never contradict each other. The
 *  server remains the authority; this only makes the same finding visible
 *  without a round trip. */
function rowProblems() {
  const problems = [];
  for (const { page, row } of allRows()) {
    const total = rowTotal(row);
    if (total > 0 && !row.product) {
      problems.push(`${page.name}: “${row.item || 'a row'}” isn’t linked to a product.`);
      continue;
    }
    const bad = liveSizeMismatch(row);
    if (bad.length) problems.push(`${page.name}: ${row.product.label} doesn’t come in size ${bad.join(', ')}.`);
  }
  return problems;
}

function localProblems() {
  return [...headerProblems(), ...rowProblems()];
}

function renderValidationSummary(problems) {
  const box = $('#validation-summary');
  box.textContent = '';
  const submit = $('#submit-btn');
  if (!problems.length) {
    box.append(el('span', { class: 'clean', text: '✓ Ready to upload' }));
    submit.disabled = false;
    return;
  }
  submit.disabled = false;   // still clickable: the server is the authority, and its message is more specific
  box.append(el('span', { class: 'problem', text: `${problems.length} thing(s) to fix — ${problems[0]}`, title: problems.join('\n') }));
}

/* ══════════════════════════════════════════════ image viewer ════════════ */

const viewer = { scale: 1, x: 0, y: 0, rotation: 0, natural: { w: 0, h: 0 }, page: null };

function showViewer(pageName) {
  const img = $('#viewer-img');
  const empty = $('#viewer-empty');
  if (!pageName || !state.sid) {
    img.removeAttribute('src');
    img.style.display = 'none';
    empty.hidden = false;
    return;
  }
  const src = `/api/sessions/${state.sid}/images/${encodeURIComponent(pageName)}/file`;
  empty.hidden = true;
  img.style.display = '';
  if (viewer.page !== pageName) {
    viewer.page = pageName;
    viewer.rotation = 0;
    img.src = src;
    img.onload = () => {
      viewer.natural = { w: img.naturalWidth, h: img.naturalHeight };
      fitViewer();
    };
  }
}

function applyViewer() {
  const img = $('#viewer-img');
  img.style.transform = `translate(${viewer.x}px, ${viewer.y}px) scale(${viewer.scale}) rotate(${viewer.rotation}deg)`;
  img.style.transformOrigin = viewer.rotation % 180 === 0 ? '0 0' : '0 0';
  $('#zoom-label').textContent = `${Math.round(viewer.scale * 100)}%`;
}

function fitViewer() {
  const pane = $('#viewer');
  const rect = pane.getBoundingClientRect();
  const rotated = viewer.rotation % 180 !== 0;
  const w = rotated ? viewer.natural.h : viewer.natural.w;
  const h = rotated ? viewer.natural.w : viewer.natural.h;
  if (!w || !h) return;
  const scale = Math.min(rect.width / w, rect.height / h) * 0.98;
  viewer.scale = scale;
  // Centre, accounting for the fact that rotation happens around the
  // top-left corner: after rotating, the image's visual box moves, so the
  // offset has to compensate before centring.
  const offset = rotationOffset(scale);
  viewer.x = (rect.width - w * scale) / 2 + offset.x;
  viewer.y = (rect.height - h * scale) / 2 + offset.y;
  applyViewer();
}

function rotationOffset(scale) {
  const { w, h } = viewer.natural;
  switch (((viewer.rotation % 360) + 360) % 360) {
    case 90: return { x: h * scale, y: 0 };
    case 180: return { x: w * scale, y: h * scale };
    case 270: return { x: 0, y: w * scale };
    default: return { x: 0, y: 0 };
  }
}

function zoomBy(factor, cx, cy) {
  const pane = $('#viewer').getBoundingClientRect();
  const px = cx === undefined ? pane.width / 2 : cx;
  const py = cy === undefined ? pane.height / 2 : cy;
  const next = Math.min(12, Math.max(0.05, viewer.scale * factor));
  const k = next / viewer.scale;
  viewer.x = px - (px - viewer.x) * k;
  viewer.y = py - (py - viewer.y) * k;
  viewer.scale = next;
  applyViewer();
}

function setupViewer() {
  const pane = $('#viewer');
  let dragging = false, lastX = 0, lastY = 0;

  pane.addEventListener('pointerdown', (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    pane.setPointerCapture(e.pointerId);
    pane.classList.add('grabbing');
  });
  pane.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    viewer.x += e.clientX - lastX;
    viewer.y += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    applyViewer();
  });
  const stop = (e) => {
    dragging = false;
    pane.classList.remove('grabbing');
    if (e.pointerId !== undefined && pane.hasPointerCapture?.(e.pointerId)) pane.releasePointerCapture(e.pointerId);
  };
  pane.addEventListener('pointerup', stop);
  pane.addEventListener('pointercancel', stop);
  pane.addEventListener('dblclick', fitViewer);
  pane.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = pane.getBoundingClientRect();
    zoomBy(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - rect.left, e.clientY - rect.top);
  }, { passive: false });

  $$('[data-viewer]').forEach(btn => btn.addEventListener('click', () => {
    switch (btn.dataset.viewer) {
      case 'zoom-in': zoomBy(1.25); break;
      case 'zoom-out': zoomBy(1 / 1.25); break;
      case 'fit': fitViewer(); break;
      case 'rotate': viewer.rotation = (viewer.rotation + 90) % 360; fitViewer(); break;
      case 'full': openLightbox(); break;
      case 'collapse': toggleImagePane(false); break;
    }
  }));
  $('#image-rail').addEventListener('click', () => toggleImagePane(true));

  window.addEventListener('resize', debounce(() => { if (viewer.page) fitViewer(); }, 200));
}

/** Hiding the photo gives the grid the whole window — the useful move once
 *  a page has been checked and only quantities are being typed. Remembered,
 *  because a reviewer who works that way works that way every time. */
function toggleImagePane(show) {
  const review = $('#view-review');
  const hidden = show === undefined ? !review.classList.contains('image-hidden') : !show;
  review.classList.toggle('image-hidden', hidden);
  $('#image-rail').hidden = !hidden;
  try { localStorage.setItem(`${LS_KEY}:imageHidden`, hidden ? '1' : ''); } catch { /* ignore */ }
  if (!hidden && viewer.page) requestAnimationFrame(fitViewer);
}

function openLightbox() {
  if (!viewer.page || !state.sid) return;
  const box = $('#lightbox');
  const img = $('#lightbox-img');
  // The full-resolution original, not the display copy — full screen is
  // exactly where a reviewer is trying to read a faint digit.
  img.src = `/api/sessions/${state.sid}/images/${encodeURIComponent(viewer.page)}/file?original=true`;
  img.style.transform = `rotate(${viewer.rotation}deg)`;
  box.hidden = false;
}

/* ══════════════════════════════════════════════ splitter ════════════════ */

function setupSplitter() {
  const splitter = $('#splitter');
  const review = $('#view-review');
  let dragging = false;

  const saved = Number(localStorage.getItem(`${LS_KEY}:split`) || 0);
  if (saved > 15 && saved < 85) review.style.setProperty('--image-pane-w', `${saved}%`);

  splitter.addEventListener('pointerdown', (e) => {
    dragging = true;
    splitter.setPointerCapture(e.pointerId);
    splitter.classList.add('dragging');
    document.body.style.userSelect = 'none';
  });
  splitter.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const rect = review.getBoundingClientRect();
    const pct = Math.min(80, Math.max(18, ((e.clientX - rect.left) / rect.width) * 100));
    review.style.setProperty('--image-pane-w', `${pct}%`);
  });
  splitter.addEventListener('pointerup', (e) => {
    dragging = false;
    splitter.classList.remove('dragging');
    document.body.style.userSelect = '';
    splitter.releasePointerCapture(e.pointerId);
    const pct = parseFloat(review.style.getPropertyValue('--image-pane-w'));
    if (pct) localStorage.setItem(`${LS_KEY}:split`, String(pct));
    if (viewer.page) fitViewer();
  });
}

/* ══════════════════════════════════════════════ pages i/o ═══════════════ */

async function removePage(name) {
  if (!confirm(`Remove page “${name}” from this order?`)) return;
  try {
    const session = await api(`/api/sessions/${state.sid}/images/${encodeURIComponent(name)}`, { method: 'DELETE' });
    state.pages = state.pages.filter(p => p.name !== name);
    state.adopted.delete(name);
    applyServerSession(session, { adoptNew: true });
    renderAll();
  } catch (err) {
    toast(err.message, 'bad');
  }
}

async function retryPage(name) {
  try {
    state.adopted.delete(name);   // take the fresh reading when it lands
    const session = await api(`/api/sessions/${state.sid}/images/${encodeURIComponent(name)}/retry`, { method: 'POST' });
    applyServerSession(session, { adoptNew: true });
    renderAll();
    startPolling();
  } catch (err) {
    toast(err.message, 'bad');
  }
}

/* ══════════════════════════════════════════════ submit ══════════════════ */

function buildSubmitBody() {
  const rows = [];
  for (const page of state.pages) {
    for (const row of page.rows) {
      const quantities = {};
      for (const [size, qty] of Object.entries(row.quantities)) {
        const n = Number(qty) || 0;
        if (n > 0) quantities[size] = n;
      }
      if (!Object.keys(quantities).length) continue;
      rows.push({ bsid: row.bsid || '', item: row.item || '', quantities, uid: row.uid || '' });
    }
  }
  return {
    header: {
      head_id: state.order.buyer ? state.order.buyer.buyer_id : null,
      ref_no: state.order.ref_no || '',
      order_dt: state.order.order_dt || isoToday(),
      ref_dt: state.order.ref_dt || state.order.order_dt || isoToday(),
      company_id: state.companyId || 1,
      user_id: state.operatorId,
      order_source: 'Local',
    },
    rows,
  };
}

async function runCheck({ quiet } = {}) {
  // Header problems are the client's own; row problems come back from the
  // server, whose messages are the more specific of the two.
  const header = headerProblems();
  renderValidationSummary([...header, ...rowProblems()]);
  try {
    const result = await api(`/api/sessions/${state.sid}/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildSubmitBody()),
    });
    const problems = [...header, ...result.problems.map(p => p.message)];
    renderValidationSummary(problems);
    if (!quiet) {
      if (!problems.length) toast(`Ready: ${result.detail_rows} detail rows, ${result.total_qty.toLocaleString()} pieces`, 'good');
      else toast(`${problems.length} thing(s) still to fix`, 'bad');
    }
    return { ok: result.ok && !header.length, problems, result };
  } catch (err) {
    if (!quiet) toast(err.message, 'bad');
    return { ok: false, problems: [...header, err.message] };
  }
}

async function submitOrder() {
  const submit = $('#submit-btn');
  const check = await runCheck({ quiet: true });
  if (!check.ok) {
    renderValidationSummary(check.problems);
    toast(check.problems[0] || 'Some rows can’t be uploaded yet', 'bad');
    return;
  }

  const body = buildSubmitBody();
  const pieces = body.rows.reduce((n, r) => n + Object.values(r.quantities).reduce((a, b) => a + b, 0), 0);
  const buyer = state.order.buyer.label;
  if (!confirm(`Upload this order to the database?\n\nBuyer: ${buyer}\nDetail rows: ${check.result.detail_rows}\nPieces: ${pieces.toLocaleString()}\n\nThis writes to OrderMaster and OrderDetails and can’t be undone from here.`)) return;

  submit.disabled = true;
  submit.textContent = 'Uploading…';
  try {
    const res = await api(`/api/sessions/${state.sid}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    showDone(res);
    clearPersisted();
  } catch (err) {
    const problems = err.body?.problems?.map(p => p.message) || [err.message];
    renderValidationSummary(problems);
    toast(problems[0], 'bad');
  } finally {
    submit.disabled = false;
    submit.textContent = 'Upload to database';
  }
}

function showDone(res) {
  const summary = res.summary || {};
  $('#done-order-no').textContent = res.order_no;
  const details = $('#done-details');
  details.textContent = '';
  const pairs = [
    ['Buyer', summary.buyer_name ? `${summary.buyer_name}${summary.buyer_city ? ` — ${summary.buyer_city}` : ''}` : '—'],
    ['Ref. no', summary.ref_no || '—'],
    ['Order date', summary.order_dt || '—'],
    ['Detail rows', String(summary.detail_rows ?? '—')],
    ['Total pieces', (summary.total_qty ?? 0).toLocaleString()],
  ];
  for (const [k, v] of pairs) {
    details.append(el('dt', { text: k }), el('dd', { text: v }));
  }
  const merged = $('#done-merged');
  if (res.merged && res.merged.length) {
    merged.hidden = false;
    merged.textContent = `Note: ${res.merged.length} quantity(ies) were added into an earlier row for the same product+size, because a size can only appear once per order.`;
  } else {
    merged.hidden = true;
  }
  stopPolling();
  showView('done');
}

function newOrder() {
  if (!$('#view-review').hidden && !confirm('Start a new order? Anything not uploaded will be lost.')) return;
  stopPolling();
  clearPersisted();
  state.sid = null;
  state.pages = [];
  state.adopted = new Set();
  state.touched = new Set();
  state.images = [];
  state.activePage = null;
  state.order = { party_name: '', ref_no: '', ref_dt: '', order_dt: isoToday(), buyer: null, buyer_note: '' };
  $('#file-input').value = '';
  viewer.page = null;
  showView('upload');
}

/* ══════════════════════════════════════════════ wiring ══════════════════ */

function setupEvents() {
  const fileInput = $('#file-input');
  const dropzone = $('#dropzone');

  fileInput.addEventListener('change', () => startSession(Array.from(fileInput.files || [])));
  $('#more-files').addEventListener('change', (e) => {
    addPages(Array.from(e.target.files || []));
    e.target.value = '';
  });

  ['dragenter', 'dragover'].forEach(ev => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach(ev => dropzone.addEventListener(ev, (e) => {
    e.preventDefault(); dropzone.classList.remove('dragover');
  }));
  dropzone.addEventListener('drop', (e) => {
    const files = Array.from(e.dataTransfer?.files || []).filter(f => f.type.startsWith('image/'));
    if (files.length) startSession(files);
  });

  $('#operator-select').addEventListener('change', (e) => {
    state.operatorId = Number(e.target.value) || null;
    localStorage.setItem(`${LS_KEY}:operator`, e.target.value);
    renderTotals();
    savePersist();
  });
  $('#company-select').addEventListener('change', (e) => {
    state.companyId = Number(e.target.value) || null;
    localStorage.setItem(`${LS_KEY}:company`, e.target.value);
    savePersist();
  });

  for (const [id, key] of [['ref-no', 'ref_no'], ['ref-dt', 'ref_dt'], ['order-dt', 'order_dt'], ['party-name', 'party_name']]) {
    $(`#${id}`).addEventListener('input', (e) => {
      state.order[key] = e.target.value;
      state.touched.add(key);
      savePersist();
    });
  }

  $('#order-card .card-head').addEventListener('click', (e) => {
    if (e.target.closest('button') && !e.target.closest('.collapse-btn')) return;
    $('#order-card').classList.toggle('collapsed');
  });

  $('#check-btn').addEventListener('click', () => runCheck({}));
  $('#submit-btn').addEventListener('click', submitOrder);
  $('#new-order-btn').addEventListener('click', newOrder);
  $('#done-new-btn').addEventListener('click', () => { state.sid = null; newOrder(); });

  $('#lightbox').addEventListener('click', (e) => {
    if (e.target.id === 'lightbox' || e.target.closest('.lightbox-close')) $('#lightbox').hidden = true;
  });

  document.addEventListener('keydown', (e) => {
    if (!$('#lightbox').hidden && e.key === 'Escape') { $('#lightbox').hidden = true; return; }
    if (e.target.matches('input, select, textarea')) return;
    if (e.key === '+' || e.key === '=') zoomBy(1.25);
    else if (e.key === '-') zoomBy(1 / 1.25);
    else if (e.key === '0') fitViewer();
    else if (e.key === 'f' || e.key === 'F') openLightbox();
    else if (e.key === 'h' || e.key === 'H') toggleImagePane();
  });

  // Only warn while there is unsaved review work — after a successful
  // upload the rows are in the database and closing the tab loses nothing.
  window.addEventListener('beforeunload', (e) => {
    if (!$('#view-review').hidden && state.sid && state.pages.some(p => p.rows.length)) {
      e.preventDefault();
      e.returnValue = '';
    }
  });
}

setupEvents();
setupViewer();
setupSplitter();
try {
  if (localStorage.getItem(`${LS_KEY}:imageHidden`)) toggleImagePane(false);
} catch { /* ignore */ }
bootstrap();
