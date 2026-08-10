/* Atik Toplama Planlama - arayuz mantigi (FAZ 3)
 *
 * Burada HESAP YOKTUR. Rota, yakit, fizibilite ve zorunlu ziyaret karari
 * sunucudan gelir (Evaluator + Simulator). Bu dosya yalnizca gosterir ve
 * kullanici eylemlerini API'ye tasir.
 */

const VEHICLE_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf'];

const S = {
  boot: null,       // /api/bootstrap
  plan: null,       // hesaplanmis plan
  markers: null,    // Leaflet katmani - konteynerler
  routes: null,     // Leaflet katmani - rota cizgileri
  map: null,
};

const $ = (id) => document.getElementById(id);
const fmt = (n, d = 0) => Number(n).toLocaleString('tr-TR', {
  minimumFractionDigits: d, maximumFractionDigits: d,
});

/* --------------------------------------------------------------- yardimcilar */

function toast(msg, kind = '') {
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  el.textContent = msg;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), kind === 'err' ? 8000 : 4000);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail || detail; } catch { /* metin degil */ }
    throw new Error(detail);
  }
  return res.status === 204 ? null : res.json();
}

function busy(btn, on, label) {
  btn.disabled = on;
  btn.innerHTML = on ? `<span class="spinner"></span>${label}` : label;
}

function stat(k, v, sub) {
  return `<div class="stat"><div class="k">${k}</div>
    <div class="v">${v}${sub ? `<small>${sub}</small>` : ''}</div></div>`;
}

/* ------------------------------------------------------------------- durum */

function renderState(st) {
  $('lastDate').textContent = st.last_date ? `son işlem ${st.last_date}` : 'henüz işlem yok';
  const cap = S.boot.hygiene_cap_days;
  $('stateStats').innerHTML =
    stat('Ort. doluluk', '%' + fmt(st.mean_fill_pct, 0)) +
    stat('Dolu (&gt;%80)', fmt(st.full_count)) +
    stat(`Hijyen sınırı (${cap}g)`, fmt(st.near_hygiene)) +
    stat('En uzun bekleme', fmt(st.max_days_waiting), ' gün');

  if (!st.history.length) {
    $('historyWrap').innerHTML =
      '<p class="muted" style="padding:14px">Henüz uygulanmış bir gün yok.</p>';
    return;
  }
  const rows = st.history.map((h) => `<tr>
      <td>${h.tarih}</td><td>${h.cozucu}</td>
      <td class="num">${fmt(h.durak)}</td>
      <td class="num">${fmt(h.yakit_l, 1)}</td>
      <td class="num">${fmt(h.mesafe_km, 1)}</td>
      <td class="num">${fmt(h.toplanan_l)}</td></tr>`).join('');
  $('historyWrap').innerHTML = `<table><thead><tr>
      <th>Tarih</th><th>Çözücü</th><th class="num">Durak</th>
      <th class="num">Yakıt L</th><th class="num">km</th><th class="num">Toplanan L</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

/* ----------------------------------------------------------------- doluluk */

function renderFill(f) {
  const box = $('fillInfo');
  if (!f) { box.classList.add('hidden'); $('btnSolve').disabled = true; return; }
  box.classList.remove('hidden');
  const warn = (f.warnings || []).map(
    (w) => `<div class="note note-warn">${w}</div>`).join('');
  box.innerHTML = `<div class="note note-ok">
      <b>${f.source}</b><br>
      toplam ${fmt(f.total_l)} L · ortalama %${fmt(f.mean_pct, 0)} ·
      hacmi aşan ${fmt(f.over_capacity)} konteyner
    </div>${warn}`;
  $('btnSolve').disabled = false;
}

/* -------------------------------------------------------------------- plan */

function renderPlan(p) {
  S.plan = p;
  const k = p.kpi;

  $('kpiStats').innerHTML =
    `<div class="stat hero"><div class="k">Yakıt</div>
       <div class="v">${fmt(k.fuel_l, 1)}<small>L</small></div></div>` +
    stat('CO₂', fmt(k.co2_kg, 0), ' kg') +
    stat('Durak', fmt(k.stops)) +
    stat('Atlanan', fmt(k.skipped)) +
    stat('Mesafe', fmt(k.distance_km, 1), ' km');

  $('kpiDetail').innerHTML = `<p class="muted" style="margin:0">
      <b>Yakıt kalemleri:</b> seyahat ${fmt(k.fuel_travel_l, 1)} L ·
      dur-kalk ${fmt(k.fuel_stop_l, 1)} L ·
      sıkıştırma ${fmt(k.fuel_compaction_l, 1)} L &nbsp;|&nbsp;
      toplanan ${fmt(k.collected_l)} L ·
      ortalama toplama doluluğu %${fmt(k.mean_fill_pct, 0)} ·
      vardiya kullanımı %${fmt(k.shift_util_pct, 0)} ·
      bölge-içi ${fmt(k.intra_km, 1)} km
    </p>`;

  const notes = [];
  if (!p.feasible) {
    notes.push(`<div class="note note-danger"><b>Plan uygulanabilir değil.</b><br>${
      p.violations.slice(0, 6).map((v) => '• ' + v).join('<br>')}</div>`);
  }
  if (k.overflow_events) {
    notes.push(`<div class="note note-warn">
      ${fmt(k.overflow_events)} konteyner hacmini aşmış durumda
      (${fmt(k.overflow_l)} L). Bunlar zorunlu ziyaret edilir.</div>`);
  }
  $('planNotes').innerHTML = notes.join('');

  $('btnApply').disabled = !p.feasible;
  const dl = $('btnStops');
  dl.style.pointerEvents = 'auto';
  dl.style.opacity = '1';

  drawPlan(p);
  renderStops(p);
  $('mapHint').textContent =
    `${p.solver_name} · ${p.routes.length} araç · ${fmt(k.stops)} durak`;
}

function renderStops(p) {
  if (!p.stops.length) {
    $('stopsWrap').innerHTML =
      '<p class="muted" style="padding:14px">Bu planda durak yok.</p>';
    return;
  }
  const rows = p.stops.map((s) => {
    const dump = s.container_id < 0;
    return `<tr class="${dump ? 'dump' : ''}">
      <td><span class="dot" style="background:${
        VEHICLE_COLORS[(s.vehicle - 1) % VEHICLE_COLORS.length]}"></span> ${s.vehicle}</td>
      <td class="num">${s.order}</td>
      <td>${dump ? 'DÖKÜM SAHASI' : '#' + s.container_id}</td>
      <td class="num">${dump ? '-' : fmt(s.fill_l)}</td>
      <td class="num">${dump ? '-' : '%' + fmt(s.fill_pct, 0)}</td>
      <td>${dump ? '' : `<span class="pill ${s.must_visit ? 'pill-must' : 'pill-opt'}">${
        s.must_visit ? 'zorunlu' : 'opsiyonel'}</span>`}</td>
      <td class="num">${fmt(s.leg_m)}</td>
      <td class="num">${fmt(s.truck_load_l)}</td></tr>`;
  }).join('');
  $('stopsWrap').innerHTML = `<table><thead><tr>
      <th>Araç</th><th class="num">Sıra</th><th>Konteyner</th>
      <th class="num">Doluluk L</th><th class="num">%</th><th>Tür</th>
      <th class="num">Bacak m</th><th class="num">Kamyon yükü L</th>
    </tr></thead><tbody>${rows}</tbody></table>`;
}

/* -------------------------------------------------------------------- harita */

function initMap() {
  S.map = L.map('map', { zoomControl: true, attributionControl: true });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap katkıda bulunanlar',
  }).addTo(S.map);
  S.markers = L.layerGroup().addTo(S.map);
  S.routes = L.layerGroup().addTo(S.map);

  const b = S.boot;
  const pts = b.containers.map((c) => [c.lat, c.lon]);
  S.map.fitBounds(L.latLngBounds(pts).pad(0.08));
  drawIdle();
}

function depotDumpMarkers() {
  const b = S.boot;
  L.circleMarker(b.depot, {
    radius: 7, color: '#fff', weight: 2, fillColor: '#1c5ba8', fillOpacity: 1,
  }).bindTooltip('Garaj').addTo(S.markers);
  L.circleMarker(b.dump, {
    radius: 7, color: '#fff', weight: 2, fillColor: '#7d3c98', fillOpacity: 1,
  }).bindTooltip('Döküm sahası').addTo(S.markers);
}

function drawIdle() {
  S.markers.clearLayers();
  S.routes.clearLayers();
  S.boot.containers.forEach((c) => {
    L.circleMarker([c.lat, c.lon], {
      radius: 3.5, color: '#b9c0c8', weight: 1, fillColor: '#b9c0c8', fillOpacity: .7,
    }).bindTooltip(`#${c.id}`).addTo(S.markers);
  });
  depotDumpMarkers();
}

function drawPlan(p) {
  S.markers.clearLayers();
  S.routes.clearLayers();

  const must = new Set(p.must_visit_ids);
  const visited = new Set(p.visited_ids);

  S.boot.containers.forEach((c) => {
    const on = visited.has(c.id);
    const color = !on ? '#b9c0c8' : (must.has(c.id) ? '#c0392b' : '#0f7a5a');
    L.circleMarker([c.lat, c.lon], {
      radius: on ? 5 : 3,
      color: '#fff', weight: on ? 1.5 : 0.5,
      fillColor: color, fillOpacity: on ? .95 : .55,
    }).bindTooltip(
      `#${c.id} - ${on ? (must.has(c.id) ? 'zorunlu' : 'ziyaret') : 'atlandı'}`
    ).addTo(S.markers);
  });

  p.routes.forEach((r) => {
    const col = VEHICLE_COLORS[(r.vehicle - 1) % VEHICLE_COLORS.length];
    L.polyline(r.coords, { color: col, weight: 2.5, opacity: .75 })
      .bindTooltip(`Araç ${r.vehicle}`).addTo(S.routes);
  });

  depotDumpMarkers();

  $('vehLegend').innerHTML = p.routes.map((r) =>
    `<span><i class="bar" style="background:${
      VEHICLE_COLORS[(r.vehicle - 1) % VEHICLE_COLORS.length]}"></i> araç ${r.vehicle}</span>`
  ).join('');
}

/* ------------------------------------------------------------------ eylemler */

async function refreshState() { renderState(await api('/api/state')); }

function wire() {
  // sekmeler
  $('fillTabs').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    [...$('fillTabs').children].forEach((x) => x.classList.toggle('on', x === b));
    $('tab-csv').classList.toggle('hidden', b.dataset.tab !== 'csv');
    $('tab-sim').classList.toggle('hidden', b.dataset.tab !== 'sim');
  });

  $('csvFile').addEventListener('change', (e) => {
    const f = e.target.files[0];
    $('csvFake').textContent = f ? `📄 ${f.name}` : '📄 CSV seç veya buraya sürükle';
    $('btnUpload').disabled = !f;
  });

  $('btnUpload').addEventListener('click', async () => {
    const f = $('csvFile').files[0]; if (!f) return;
    const fd = new FormData(); fd.append('file', f);
    busy($('btnUpload'), true, 'Yükleniyor');
    try {
      renderFill(await api('/api/fill/upload', { method: 'POST', body: fd }));
      toast(`${S.boot.n_containers} konteyner okundu.`, 'ok');
    } catch (err) { toast(err.message, 'err'); renderFill(null); }
    busy($('btnUpload'), false, 'Yükle');
  });

  $('btnSimulate').addEventListener('click', async () => {
    busy($('btnSimulate'), true, 'Üretiliyor');
    try {
      renderFill(await api('/api/fill/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seed: Number($('simSeed').value) }),
      }));
      toast('Bir günlük üretim eklendi.', 'ok');
    } catch (err) { toast(err.message, 'err'); }
    busy($('btnSimulate'), false, 'Bir gün üret');
  });

  $('btnSolve').addEventListener('click', async () => {
    busy($('btnSolve'), true, 'Rota hesaplanıyor');
    try {
      renderPlan(await api('/api/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          solver: $('solver').value,
          skip_lambda: Number($('lambda').value),
          time_limit_sec: Number($('tlimit').value),
        }),
      }));
      toast('Plan hazır.', 'ok');
    } catch (err) { toast(err.message, 'err'); }
    busy($('btnSolve'), false, 'Bugünü çöz');
  });

  $('btnApply').addEventListener('click', async () => {
    busy($('btnApply'), true, 'Kaydediliyor');
    try {
      renderState(await api('/api/apply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ collection_date: $('applyDate').value }),
      }));
      S.plan = null;
      renderFill(null);
      $('planNotes').innerHTML = '';
      $('kpiDetail').innerHTML =
        '<p class="muted" style="margin:0">Plan uygulandı. Yeni gün için doluluk girin.</p>';
      drawIdle();
      toast('Kaydedildi. Yarın kaldığı yerden devam eder.', 'ok');
    } catch (err) { toast(err.message, 'err'); }
    busy($('btnApply'), false, 'Uygula ve kaydet');
  });

  $('btnReset').addEventListener('click', async () => {
    if (!confirm('Tüm doluluk ve bekleme sayaçları sıfırlanacak, geçmiş silinecek. Emin misiniz?')) return;
    try {
      renderState(await api('/api/reset', { method: 'POST' }));
      S.plan = null; renderFill(null); drawIdle();
      $('planNotes').innerHTML = '';
      toast('Durum sıfırlandı.', 'ok');
    } catch (err) { toast(err.message, 'err'); }
  });
}

/* ------------------------------------------------------------------ baslangic */

(async function main() {
  try {
    S.boot = await api('/api/bootstrap');
  } catch (err) {
    document.body.innerHTML =
      `<div style="padding:40px;font:14px system-ui;color:#c0392b">
         Sunucuya bağlanılamadı: ${err.message}</div>`;
    return;
  }

  $('regionLabel').textContent =
    `${S.boot.region} · ${S.boot.n_containers} konteyner · ${S.boot.n_vehicles} araç`;
  $('hashLabel').textContent = S.boot.config_hash;
  $('lambda').value = S.boot.default_lambda;
  $('applyDate').valueAsDate = new Date();
  $('solver').innerHTML = S.boot.solvers.map(
    (s) => `<option value="${s.code}">${s.name} - ${s.note}</option>`).join('');

  initMap();
  wire();
  await refreshState();

  // Sayfa yenilendiyse sunucudaki gecici durumu geri yukle
  const f = await api('/api/fill');
  if (f) renderFill(f);
  const p = await api('/api/plan');
  if (p) renderPlan(p);
})();
