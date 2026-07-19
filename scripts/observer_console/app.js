// Observer Console — frontend
// ===========================================

const COLORS = {
  accent: '#7cc5ff',
  signal: '#c4b5fd',
  warn: '#fbbf77',
  good: '#86efac',
  danger: '#fb7185',
  phaseBase: '#7cc5ff',
  phasePerturb: '#fbbf77',
  phaseReask: '#c4b5fd',
  grid: '#222634',
  text: '#e8ecf3',
  textDim: '#9aa3b5',
  textMuted: '#5f6678',
  bg: '#0a0b0f',
};

const PLOTLY_CONFIG = {
  displayModeBar: false,
  responsive: true,
  staticPlot: false,
};

const BASE_LAYOUT = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  margin: { t: 12, r: 14, b: 30, l: 44 },
  font: { family: 'JetBrains Mono, monospace', color: COLORS.textDim, size: 10 },
  xaxis: {
    gridcolor: COLORS.grid,
    zerolinecolor: COLORS.grid,
    linecolor: COLORS.grid,
    tickfont: { color: COLORS.textMuted },
    title: { text: '', font: { size: 10 } },
  },
  yaxis: {
    gridcolor: COLORS.grid,
    zerolinecolor: COLORS.grid,
    linecolor: COLORS.grid,
    tickfont: { color: COLORS.textMuted },
  },
  showlegend: false,
  hoverlabel: { bgcolor: '#141720', bordercolor: '#222634', font: { color: COLORS.text } },
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  view: 'launch',
  mode: 'observe',
  registry: { models: [], default_model: null },
  job: { status: 'idle' },
  events: [],          // live event buffer for current run
  archive: [],
  archiveFilter: 'all',
  compareSelection: new Set(), // Set<run_id>
  sse: null,
  chartsInitialized: false,
};

const COMPARE_PALETTE = [
  '#7cc5ff', '#fbbf77', '#c4b5fd', '#86efac', '#fb7185',
  '#a3e3ff', '#ffd89c', '#ddd0ff', '#bff4cd', '#ffb4c0',
];

const HYSTERESIS_PHASES = [
  'clean_exposure',
  'perturbed_exposure',
  'clean_recovery',
  'perturbed_recovery',
];

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const fmtNum = (v, digits = 3) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (typeof v !== 'number') return String(v);
  const abs = Math.abs(v);
  if (abs === 0) return '0';
  if (abs >= 1000) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(digits);
  if (abs >= 0.001) return v.toFixed(digits);
  return v.toExponential(2);
};

const fmtTime = (ts) => {
  const d = new Date(ts * 1000);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  const hours = d.getHours().toString().padStart(2, '0');
  const mins = d.getMinutes().toString().padStart(2, '0');
  if (sameDay) return `${hours}:${mins}`;
  return `${d.getMonth() + 1}/${d.getDate()} ${hours}:${mins}`;
};

const relativeTime = (ts) => {
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
};

// ---------------------------------------------------------------------------
// View routing
// ---------------------------------------------------------------------------

function switchView(name) {
  state.view = name;
  $$('.view').forEach((el) => el.classList.toggle('is-active', el.dataset.view === name));
  $$('.nav-btn').forEach((el) => el.classList.toggle('is-active', el.dataset.view === name));

  if (name === 'archive') {
    fetchArchive();
  } else if (name === 'live') {
    if (!state.chartsInitialized) initLiveCharts();
  } else if (name === 'compare') {
    renderCompareView();
  }
}

function switchMode(mode) {
  state.mode = mode;
  $$('.mode-tab').forEach((el) => el.classList.toggle('is-active', el.dataset.mode === mode));
  $$('.mode-options').forEach((el) => {
    el.hidden = !el.dataset.for.split(',').includes(mode);
  });
  $$('[data-only-for]').forEach((el) => {
    el.hidden = !el.dataset.onlyFor.split(',').includes(mode);
  });
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function fetchModels() {
  const res = await fetch('/api/models');
  const data = await res.json();
  state.registry = data;
  const sel = $('#f-model');
  sel.innerHTML = '';
  (data.models || []).forEach((m) => {
    const opt = document.createElement('option');
    opt.value = m.key;
    opt.textContent = `${m.key}  ·  ${m.hf_id}`;
    if (m.key === data.default_model) opt.selected = true;
    sel.appendChild(opt);
  });
  updateModelHint();
  sel.addEventListener('change', updateModelHint);
}

function updateModelHint() {
  const sel = $('#f-model');
  const m = (state.registry.models || []).find((x) => x.key === sel.value);
  $('#f-model-hint').textContent = m ? m.hf_id : '';
}

async function fetchArchive() {
  const res = await fetch('/api/runs');
  const data = await res.json();
  state.archive = data.runs || [];
  renderArchive();
}

async function fetchRunDetail(runId) {
  const res = await fetch('/api/runs/' + encodeURIComponent(runId));
  return res.json();
}

async function launchRun(payload) {
  const res = await fetch('/api/launch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: 'request failed' }));
    throw new Error(err.error || 'request failed');
  }
  return res.json();
}

async function stopRun() {
  await fetch('/api/stop', { method: 'POST' });
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------

function connectSSE() {
  if (state.sse) state.sse.close();
  const es = new EventSource('/api/stream');
  state.sse = es;

  es.addEventListener('hello', (e) => {
    const data = JSON.parse(e.data);
    if (data.job) applyJobSnapshot(data.job);
  });

  es.addEventListener('job_started', (e) => {
    const job = JSON.parse(e.data);
    applyJobSnapshot(job);
    resetLiveView(job);
    switchView('live');
  });

  es.addEventListener('log', (e) => {
    const { line } = JSON.parse(e.data);
    appendLog(line);
  });

  es.addEventListener('run_detected', (e) => {
    const data = JSON.parse(e.data);
    state.job.run_id = data.run_id;
    state.job.run_dir = data.run_dir;
    updateLiveSub();
  });

  es.addEventListener('event', (e) => {
    const ev = JSON.parse(e.data);
    handleTokenEvent(ev);
  });

  es.addEventListener('frame', (e) => {
    const { phase, frame } = JSON.parse(e.data);
    handleHysteresisFrame(phase, frame);
  });

  es.addEventListener('job_finished', (e) => {
    const data = JSON.parse(e.data);
    state.job.status = data.status;
    state.job.run_id = data.run_id || state.job.run_id;
    updateStatusUI();
    if (data.summary) handleFinalSummary(data.summary);
  });

  es.onerror = () => {
    setTimeout(() => {
      if (!state.sse || state.sse.readyState === EventSource.CLOSED) connectSSE();
    }, 1500);
  };
}

function applyJobSnapshot(job) {
  state.job = job;
  updateStatusUI();
  if (job.status === 'running' && job.logs) {
    $('#log-feed').textContent = job.logs.join('\n') || 'Waiting for output…';
  }
}

function updateStatusUI() {
  const pill = $('#status-pill');
  const status = state.job.status || 'idle';
  pill.dataset.status = status;
  pill.textContent = status;

  const meta = $('#status-meta');
  if (status === 'idle') {
    meta.textContent = 'No run in progress';
  } else {
    const lines = [
      state.job.mode ? `mode: ${state.job.mode}` : null,
      state.job.model ? `model: ${state.job.model}` : null,
      state.job.run_id ? `run: ${state.job.run_id.slice(-14)}` : null,
      state.job.event_count ? `events: ${state.job.event_count}` : null,
    ].filter(Boolean);
    meta.textContent = lines.join('\n');
  }

  const btnStop = $('#btn-stop');
  const btnLaunch = $('#btn-launch');
  const isActive = status === 'starting' || status === 'running';
  btnStop.disabled = !isActive;
  btnLaunch.disabled = isActive;

  const badge = $('#live-badge');
  if (isActive) {
    badge.hidden = false;
    badge.textContent = state.job.event_count || '•';
  } else {
    badge.hidden = true;
  }

  updateLiveSub();
}

function updateLiveSub() {
  const sub = $('#live-sub');
  if (!state.job || state.job.status === 'idle') {
    sub.textContent = 'Waiting for a run…';
    return;
  }
  const parts = [];
  parts.push(state.job.mode || 'run');
  if (state.job.model) parts.push('· ' + state.job.model);
  if (state.job.run_id) parts.push('· ' + state.job.run_id);
  if (state.job.status) parts.push('· ' + state.job.status);
  sub.textContent = parts.join(' ');
}

// ---------------------------------------------------------------------------
// Live view
// ---------------------------------------------------------------------------

function resetLiveView(job) {
  state.events = [];
  $('#log-feed').textContent = 'Starting…';
  $('#token-stream').innerHTML = '';
  $('#meta-tokens').textContent = '0 tokens';
  $('#meta-divergence').textContent = '—';
  $('#meta-entropy').textContent = '—';
  $('#meta-hidden').textContent = '—';
  $('#hysteresis-card').hidden = job.mode !== 'hysteresis';
  HYSTERESIS_PHASES.forEach((phase) => {
    const id = phase.replaceAll('_', '-');
    $(`#phase-output-${id}`).textContent = '—';
    $(`#phase-status-${id}`).textContent = 'pending';
    $(`#phase-status-${id}`).classList.remove('is-done');
  });
  $('#hysteresis-regime').textContent = '—';

  // Clear charts
  initLiveCharts(true);
}

function appendLog(line) {
  const feed = $('#log-feed');
  if (feed.textContent === 'Waiting for output…' || feed.textContent === 'Starting…') {
    feed.textContent = '';
  }
  feed.textContent += (feed.textContent ? '\n' : '') + line;
  feed.scrollTop = feed.scrollHeight;
}

function initLiveCharts(reset = false) {
  const divergenceTrace = [{
    x: [],
    y: [],
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: COLORS.accent, width: 2, shape: 'spline', smoothing: 0.5 },
    marker: { size: 5, color: COLORS.accent },
    fill: 'tozeroy',
    fillcolor: 'rgba(124, 197, 255, 0.12)',
    hovertemplate: 't=%{x}<br>div=%{y:.4f}<extra></extra>',
  }];

  const divergenceLayout = {
    ...BASE_LAYOUT,
    margin: { t: 8, r: 14, b: 28, l: 52 },
    yaxis: { ...BASE_LAYOUT.yaxis, title: { text: 'divergence', font: { size: 10, color: COLORS.textMuted } } },
  };

  Plotly.react('chart-divergence', divergenceTrace, divergenceLayout, PLOTLY_CONFIG);

  const entropyTrace = [{
    x: [], y: [],
    type: 'scatter', mode: 'lines',
    line: { color: COLORS.signal, width: 2, shape: 'spline', smoothing: 0.5 },
    hovertemplate: 't=%{x}<br>H=%{y:.3f}<extra></extra>',
  }];
  Plotly.react('chart-entropy', entropyTrace, BASE_LAYOUT, PLOTLY_CONFIG);

  const hiddenTrace = [{
    x: [], y: [],
    type: 'scatter', mode: 'lines',
    line: { color: COLORS.warn, width: 2, shape: 'spline', smoothing: 0.5 },
    hovertemplate: 't=%{x}<br>‖h‖=%{y:.2f}<extra></extra>',
  }];
  Plotly.react('chart-hidden', hiddenTrace, BASE_LAYOUT, PLOTLY_CONFIG);

  // Spectral bands: 6 stacked area traces
  const bandColors = ['#7cc5ff', '#9ec3ff', '#b8b4f9', '#c4b5fd', '#d4a7f0', '#e89cd5'];
  const bandTraces = bandColors.map((color, i) => ({
    x: [],
    y: [],
    type: 'scatter',
    mode: 'lines',
    stackgroup: 'one',
    groupnorm: 'percent',
    name: `band ${i + 1}`,
    line: { width: 0.5, color },
    fillcolor: color,
    hovertemplate: `band ${i + 1}: %{y:.1f}%<extra></extra>`,
  }));
  Plotly.react('chart-bands', bandTraces, {
    ...BASE_LAYOUT,
    margin: { t: 8, r: 14, b: 28, l: 52 },
    yaxis: { ...BASE_LAYOUT.yaxis, range: [0, 100], ticksuffix: '%' },
  }, PLOTLY_CONFIG);

  state.chartsInitialized = true;
}

function handleTokenEvent(ev) {
  state.events.push(ev);
  state.job.event_count = state.events.length;
  updateStatusUI();

  const t = ev.t ?? state.events.length - 1;
  const div = ev.diagnostics?.local_prediction_error ?? ev.diagnostics?.divergence;
  const entropy = ev.diagnostics?.spectral?.spectral_entropy;
  const hiddenNorm = ev.hidden_post_norm ?? ev.hidden_pre_norm;
  const bands = ev.diagnostics?.spectral?.band_fracs || [];

  // Token text
  const tokenText = ev.consumed_token_text ?? ev.token_text;
  if (tokenText) {
    const span = document.createElement('span');
    span.className = 'token';
    span.textContent = tokenText;
    $('#token-stream').appendChild(span);
    $('#meta-tokens').textContent = `${state.events.length} tokens`;
    const stream = $('#token-stream');
    stream.scrollTop = stream.scrollHeight;
  }

  // Meta readouts
  if (typeof div === 'number') $('#meta-divergence').textContent = `latest: ${fmtNum(div, 4)}`;
  if (typeof entropy === 'number') $('#meta-entropy').textContent = `latest: ${fmtNum(entropy, 3)}`;
  if (typeof hiddenNorm === 'number') $('#meta-hidden').textContent = `latest: ${fmtNum(hiddenNorm, 1)}`;

  // Charts — use extendTraces for efficient streaming
  try {
    if (typeof div === 'number') {
      Plotly.extendTraces('chart-divergence', { x: [[t]], y: [[div]] }, [0]);
    }
    if (typeof entropy === 'number') {
      Plotly.extendTraces('chart-entropy', { x: [[t]], y: [[entropy]] }, [0]);
    }
    if (typeof hiddenNorm === 'number') {
      Plotly.extendTraces('chart-hidden', { x: [[t]], y: [[hiddenNorm]] }, [0]);
    }
    if (bands.length === 6) {
      const xs = bands.map(() => [t]);
      const ys = bands.map((v) => [v * 100]);
      Plotly.extendTraces('chart-bands', { x: xs, y: ys }, [0, 1, 2, 3, 4, 5]);
    }
  } catch (err) {
    // Chart might not be initialized yet; ignore.
  }
}

function handleHysteresisFrame(phase, frame) {
  const id = phase.replaceAll('_', '-');
  $(`#phase-status-${id}`).textContent = 'done';
  $(`#phase-status-${id}`).classList.add('is-done');
  const output = frame?.output || frame?.text || '(no output captured)';
  $(`#phase-output-${id}`).textContent = output;
}

function handleFinalSummary(summary) {
  if (!summary) return;
  if (summary.mode === 'hysteresis') {
    renderHysteresisMetricsChart(summary);
    const regime = summary.metrics?.regime;
    if (regime) $('#hysteresis-regime').textContent = `regime: ${regime}`;

    // Fetch all four matched trace outputs after the run completes.
    if (state.job.run_id) {
      fetchRunDetail(state.job.run_id).then((detail) => {
        const outs = detail.outputs || {};
        Object.entries(outs).forEach(([phase, txt]) => {
          if (txt) handleHysteresisFrame(phase, { text: txt });
        });
      }).catch(() => {});
    }
  }
}

function renderHysteresisMetricsChart(summary) {
  const metrics = summary.metrics || {};
  const exposure = metrics.distance_curve_exposure || [];
  const recovery = metrics.distance_curve_recovery || [];
  const traces = [{
    x: exposure.map((point) => point.global_decision_index),
    y: exposure.map((point) => point.distance),
    name: 'exposure distance',
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: COLORS.warn, width: 2 },
  }, {
    x: recovery.map((point) => point.global_decision_index),
    y: recovery.map((point) => point.distance),
    name: 'recovery distance',
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: COLORS.signal, width: 2 },
  }];

  Plotly.react('chart-hysteresis-metrics', traces, {
    ...BASE_LAYOUT,
    showlegend: true,
    legend: { orientation: 'h', x: 0, y: -0.15, font: { color: COLORS.textDim, size: 10 } },
    margin: { t: 10, r: 14, b: 50, l: 44 },
  }, PLOTLY_CONFIG);
}

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------

function renderArchive() {
  const grid = $('#archive-grid');
  const count = $('#archive-count');
  const filtered = state.archive.filter((r) => state.archiveFilter === 'all' || r.mode === state.archiveFilter);
  count.textContent = `${state.archive.length} total · ${filtered.length} shown`;

  if (!filtered.length) {
    grid.innerHTML = '<div class="empty-state">No runs match this filter.</div>';
    return;
  }

  grid.innerHTML = '';
  filtered.forEach((run) => grid.appendChild(buildRunCard(run)));
}

function buildRunCard(run) {
  const card = document.createElement('div');
  card.className = 'run-card';
  if (state.compareSelection.has(run.id)) card.classList.add('is-selected');
  card.addEventListener('click', (e) => {
    if (e.target.closest('.run-card-check')) return;
    openRunDetail(run.id);
  });

  const check = document.createElement('div');
  check.className = 'run-card-check';
  check.textContent = '✓';
  check.title = 'Add to compare';
  check.addEventListener('click', (e) => {
    e.stopPropagation();
    toggleCompareSelection(run.id);
    card.classList.toggle('is-selected', state.compareSelection.has(run.id));
  });
  card.appendChild(check);

  // Advisory badge (top-right corner, to the left of the compare check)
  const advisoryFlag = run.headline?.advisory_flag;
  if (advisoryFlag) {
    const tone = advisoryFlagTone(advisoryFlag);
    const badge = document.createElement('div');
    badge.className = `run-card-advisory advisory-flag advisory-flag-${tone}`;
    badge.title = `advisory: ${advisoryFlag}`;
    badge.innerHTML = `<span class="advisory-dot"></span>${escapeHTML(advisoryFlag)}`;
    card.appendChild(badge);
  }

  const head = document.createElement('div');
  head.className = 'run-card-head';
  const modeLabel = run.mode === 'sweep' ? `sweep · ${run.sweep_mode || '?'}` : run.mode;
  head.innerHTML = `
    <span class="run-mode" data-mode="${run.mode}">${modeLabel}</span>
    <span class="run-time" title="${new Date(run.updated_at * 1000).toLocaleString()}">${relativeTime(run.updated_at)}</span>
  `;

  const prompt = document.createElement('div');
  prompt.className = 'run-prompt';
  prompt.textContent = run.prompt || '(no prompt)';

  const metrics = document.createElement('div');
  metrics.className = 'run-metrics';
  const h = run.headline || {};
  const tiles = [];
  if (h.model) tiles.push(['model', h.model]);
  if (run.mode === 'observe') {
    if (h.tokens != null) tiles.push(['tokens', h.tokens]);
    if (h.avg_divergence != null) tiles.push(['avg div', fmtNum(h.avg_divergence, 3)]);
  } else if (run.mode === 'hysteresis') {
    if (h.regime) tiles.push(['regime', h.regime, h.regime === 'elastic' ? 'good' : 'warn']);
    if (h.recovery != null) tiles.push(['recovery', fmtNum(h.recovery, 3)]);
    if (h.drift != null) tiles.push(['drift', fmtNum(h.drift, 2)]);
  } else if (run.mode === 'stress') {
    if (h.regime) tiles.push(['regime', h.regime]);
    if (h.token_match_rate != null) tiles.push(['match', fmtNum(h.token_match_rate, 3)]);
  } else if (run.mode === 'control') {
    if (h.avg_raw_div_mean != null) tiles.push(['avg div', fmtNum(h.avg_raw_div_mean, 3)]);
    if (h.avg_score_mean != null) tiles.push(['score', fmtNum(h.avg_score_mean, 3)]);
  } else if (run.mode === 'propagation') {
    if (h.injection_relative != null) tiles.push(['inject rel', fmtNum(h.injection_relative, 3)]);
    if (h.final_norm_relative != null) tiles.push(['final rel', fmtNum(h.final_norm_relative, 3)]);
    if (h.mean_logit_kl != null) tiles.push(['logit KL', fmtNum(h.mean_logit_kl, 4)]);
  } else if (run.mode === 'sweep') {
    if (h.n_seeds != null) tiles.push(['seeds', `${h.n_ok ?? h.n_seeds}/${h.n_seeds}`, 'good']);
    if (h.avg_divergence != null) tiles.push(['avg div', `${fmtNum(h.avg_divergence, 3)} ± ${fmtNum(h.avg_divergence_std, 3)}`, 'accent']);
    if (h['metrics.recovery'] != null) tiles.push(['recovery', `${fmtNum(h['metrics.recovery'], 3)} ± ${fmtNum(h['metrics.recovery_std'], 3)}`, 'accent']);
  }
  if (h.device) tiles.push(['device', h.device]);

  tiles.forEach(([label, value, cls]) => {
    const tile = document.createElement('div');
    tile.className = 'run-metric';
    tile.innerHTML = `
      <span class="run-metric-label">${label}</span>
      <span class="run-metric-value ${cls || ''}">${value}</span>
    `;
    metrics.appendChild(tile);
  });

  card.appendChild(head);
  card.appendChild(prompt);
  card.appendChild(metrics);
  return card;
}

function toggleCompareSelection(runId) {
  if (state.compareSelection.has(runId)) {
    state.compareSelection.delete(runId);
  } else {
    if (state.compareSelection.size >= 10) {
      showLaunchMsg('Max 10 runs in compare.', 'error');
      return;
    }
    state.compareSelection.add(runId);
  }
  updateCompareSelectionUI();
}

function updateCompareSelectionUI() {
  const n = state.compareSelection.size;
  const badge = $('#compare-badge');
  const cnt = $('#compare-count');
  const goto = $('#btn-goto-compare');
  const clear = $('#btn-clear-selection');
  if (n > 0) {
    badge.hidden = false;
    badge.textContent = n;
    cnt.textContent = n;
    goto.hidden = false;
    clear.hidden = false;
  } else {
    badge.hidden = true;
    goto.hidden = true;
    clear.hidden = true;
  }
}

async function openRunDetail(runId) {
  const detail = await fetchRunDetail(runId);
  const wrap = $('#run-detail');
  const body = $('#run-detail-body');
  wrap.hidden = false;
  $('#archive-grid').hidden = true;
  wrap.scrollIntoView({ behavior: 'smooth', block: 'start' });

  body.innerHTML = '';

  if (detail.mode === 'sweep') {
    renderSweepDetail(body, detail);
    return;
  }

  // Header
  const summary = detail.summary || {};
  const config = detail.config?.config || detail.config || {};
  const mode = detail.mode;

  const head = document.createElement('div');
  head.className = 'detail-head';
  head.innerHTML = `
    <div>
      <h2><span class="run-mode" data-mode="${mode}">${mode}</span> · ${config.model || summary.model_id || '?'}</h2>
      <div class="detail-prompt">${escapeHTML(config.prompt || config.original_question || '')}</div>
      <div class="detail-id">${detail.id}</div>
    </div>
  `;
  body.appendChild(head);

  // Metrics tiles
  body.appendChild(buildMetricsRow(mode, summary, detail));

  // Advisory card (if present)
  renderAdvisory(body, detail);

  // Mode-specific charts
  if (mode === 'observe' && detail.events) {
    body.appendChild(sectionTitle('Divergence trajectory'));
    body.appendChild(buildChartDiv('detail-chart-div'));
    body.appendChild(sectionTitle('Spectral entropy'));
    body.appendChild(buildChartDiv('detail-chart-ent'));
    body.appendChild(sectionTitle('Hidden-state norm'));
    body.appendChild(buildChartDiv('detail-chart-hid'));

    setTimeout(() => renderObserveCharts(detail.events), 40);
  }

  if (mode === 'hysteresis') {
    body.appendChild(sectionTitle('Matched exposure and recovery distance'));
    body.appendChild(buildChartDiv('detail-chart-hyst'));
    body.appendChild(sectionTitle('Matched trace outputs'));
    body.appendChild(buildPhaseOutputs(detail.outputs || {}));

    setTimeout(() => {
      renderHysteresisDetailCharts(summary);
    }, 40);
  }

  if (mode === 'propagation') {
    body.appendChild(sectionTitle('Same-context downstream propagation'));
    body.appendChild(buildChartDiv('detail-chart-propagation'));
    setTimeout(() => renderPropagationDetailChart(summary), 40);
  }

  if (mode === 'control' && detail.events) {
    body.appendChild(sectionTitle('Controller signal'));
    body.appendChild(buildChartDiv('detail-chart-ctrl'));
    setTimeout(() => renderControlCharts(detail.events), 40);
  }

  if (mode === 'stress' || mode === 'branchpoints') {
    const results = detail.results || {};
    if (Object.keys(results).length) {
      body.appendChild(sectionTitle('Stress results'));
      const pre = document.createElement('pre');
      pre.className = 'log-feed';
      pre.textContent = JSON.stringify(results.metrics || results, null, 2);
      body.appendChild(pre);
    }
  }

  if (mode === 'branchpoints' && detail.events) {
    body.appendChild(sectionTitle('Local counterfactual rows'));
    const pre = document.createElement('pre');
    pre.className = 'log-feed';
    pre.textContent = JSON.stringify(detail.events, null, 2);
    body.appendChild(pre);
  }

  if (detail.output) {
    body.appendChild(sectionTitle('Generated output'));
    const pre = document.createElement('pre');
    pre.className = 'log-feed';
    pre.textContent = detail.output;
    body.appendChild(pre);
  }
}

function closeRunDetail() {
  $('#run-detail').hidden = true;
  $('#archive-grid').hidden = false;
}

function renderSweepDetail(body, detail) {
  const sweep = detail.sweep || {};
  const desc = sweep.describe || {};
  const agg = sweep.aggregate || {};

  const head = document.createElement('div');
  head.className = 'detail-head';
  head.innerHTML = `
    <div>
      <h2><span class="run-mode" data-mode="sweep">sweep</span> · ${sweep.mode || '?'} × ${sweep.n_seeds || 0} seeds</h2>
      <div class="detail-prompt">${escapeHTML(desc.prompt || '')}</div>
      <div class="detail-id">${detail.id}</div>
    </div>
  `;
  body.appendChild(head);

  // Key aggregated numbers
  const keyMetrics = ['avg_divergence', 'metrics.recovery', 'metrics.drift', 'metrics.hysteresis', 'tokens'];
  const row = document.createElement('div');
  row.className = 'metrics-row';
  for (const k of keyMetrics) {
    const a = agg[k];
    if (!a) continue;
    const tile = document.createElement('div');
    tile.className = 'metric-tile';
    const mean = fmtNum(a.mean, 4);
    const std = a.stdev != null ? ` ± ${fmtNum(a.stdev, 3)}` : '';
    tile.innerHTML = `
      <div class="metric-tile-label">${k}</div>
      <div class="metric-tile-value accent">${mean}</div>
      <div class="metric-tile-sub">${std} · n=${a.n} · [${fmtNum(a.min, 3)}, ${fmtNum(a.max, 3)}]</div>
    `;
    row.appendChild(tile);
  }
  body.appendChild(row);

  body.appendChild(sectionTitle('All aggregated scalars'));
  const aggWrap = document.createElement('div');
  aggWrap.className = 'sweep-aggregate';
  const sortedKeys = Object.keys(agg).sort();
  for (const k of sortedKeys) {
    const a = agg[k];
    const r = document.createElement('div');
    r.className = 'sweep-aggregate-row';
    r.innerHTML = `
      <span class="label">${k}</span>
      <span class="val">${fmtNum(a.mean, 4)}</span>
      <span class="sub">± ${a.stdev != null ? fmtNum(a.stdev, 4) : '—'}</span>
      <span class="sub">min ${fmtNum(a.min, 3)}</span>
      <span class="sub">max ${fmtNum(a.max, 3)}</span>
    `;
    aggWrap.appendChild(r);
  }
  body.appendChild(aggWrap);

  body.appendChild(sectionTitle('Per-seed runs'));
  const list = document.createElement('div');
  list.className = 'sweep-aggregate';
  for (const r of (sweep.per_run || [])) {
    const row2 = document.createElement('div');
    row2.className = 'sweep-aggregate-row';
    const status = r.error ? `error: ${r.error}` : r.status || 'ok';
    row2.innerHTML = `
      <span class="label">seed ${r.seed}</span>
      <span class="val">${r.run_id || '—'}</span>
      <span class="sub">${status}</span>
      <span class="sub"></span>
      <span class="sub"></span>
    `;
    if (r.run_id) {
      row2.style.cursor = 'pointer';
      row2.addEventListener('click', () => openRunDetail(r.run_id));
    }
    list.appendChild(row2);
  }
  body.appendChild(list);
}

async function renderCompareView() {
  const empty = $('#compare-empty');
  const body = $('#compare-body');
  const ids = [...state.compareSelection];
  if (!ids.length) {
    empty.hidden = false;
    body.hidden = true;
    return;
  }
  empty.hidden = true;
  body.hidden = false;

  const res = await fetch('/api/compare?ids=' + encodeURIComponent(ids.join(',')));
  const data = await res.json();
  const runs = data.runs || [];

  // Legend
  const legend = $('#compare-legend');
  legend.innerHTML = '';
  runs.forEach((r, i) => {
    const color = COMPARE_PALETTE[i % COMPARE_PALETTE.length];
    const item = document.createElement('div');
    item.className = 'legend-item';
    const h = r.headline || {};
    const tag = h.model ? `${r.mode} · ${h.model}` : r.mode;
    item.innerHTML = `
      <div class="legend-swatch" style="background: ${color}"></div>
      <div>
        <div class="legend-label">${tag}</div>
        <div class="legend-label" style="font-size: 10px; color: var(--text-muted)">${r.id}</div>
      </div>
    `;
    const rm = document.createElement('span');
    rm.className = 'legend-remove';
    rm.textContent = '×';
    rm.addEventListener('click', () => {
      state.compareSelection.delete(r.id);
      updateCompareSelectionUI();
      renderCompareView();
      renderArchive();
    });
    item.appendChild(rm);
    legend.appendChild(item);
  });

  // Build traces per metric
  const metrics = [
    { id: 'compare-chart-divergence', key: 'divergence', title: 'divergence' },
    { id: 'compare-chart-entropy', key: 'entropy', title: 'entropy' },
    { id: 'compare-chart-hidden', key: 'hidden', title: '‖h‖' },
  ];

  for (const m of metrics) {
    const traces = runs.map((r, i) => {
      const color = COMPARE_PALETTE[i % COMPARE_PALETTE.length];
      const series = r.series || [];
      return {
        x: series.map((p) => p.t),
        y: series.map((p) => p[m.key]),
        name: r.id.slice(-14),
        type: 'scatter',
        mode: 'lines',
        line: { color, width: 2, shape: 'spline', smoothing: 0.4 },
        hovertemplate: `${r.id.slice(-14)}<br>t=%{x}<br>${m.title}=%{y:.4f}<extra></extra>`,
      };
    });
    Plotly.react(m.id, traces, {
      ...BASE_LAYOUT,
      showlegend: false,
      margin: { t: 10, r: 14, b: 30, l: 52 },
      yaxis: { ...BASE_LAYOUT.yaxis, title: { text: m.title, font: { size: 10, color: COLORS.textMuted } } },
      xaxis: { ...BASE_LAYOUT.xaxis, title: { text: 'token', font: { size: 10, color: COLORS.textMuted } } },
    }, PLOTLY_CONFIG);
  }
}

function sectionTitle(text) {
  const el = document.createElement('div');
  el.className = 'detail-section-title';
  el.textContent = text;
  const wrap = document.createElement('div');
  wrap.className = 'detail-section';
  wrap.appendChild(el);
  return wrap;
}

function buildChartDiv(id) {
  const div = document.createElement('div');
  div.id = id;
  div.className = 'detail-chart';
  return div;
}

function buildMetricsRow(mode, summary, detail) {
  const row = document.createElement('div');
  row.className = 'metrics-row';

  const tiles = [];
  const runtime = summary.runtime || {};
  tiles.push(['device', runtime.device || '—']);
  tiles.push(['dtype', runtime.resolved_dtype || '—']);

  if (mode === 'observe') {
    tiles.push(['tokens', summary.tokens ?? '—', 'accent']);
    tiles.push(['avg divergence', fmtNum(summary.avg_divergence, 4), 'accent']);
    const health = summary.diagnostics_health || {};
    tiles.push(['degraded steps', health.degraded_steps ?? '—', health.degraded_steps ? 'warn' : 'good']);
  }

  if (mode === 'hysteresis') {
    const m = summary.metrics || {};
    tiles.push(['regime', m.regime || '—', m.regime === 'elastic' ? 'good' : 'warn']);
    tiles.push([
      'initial effect',
      fmtNum(m.initial_intervention_distance ?? m.direct_effect_peak, 3),
      'warn',
    ]);
    tiles.push([
      'active peak',
      fmtNum(m.active_window_peak_distance ?? m.direct_effect_peak, 3),
      'warn',
    ]);
    tiles.push(['propagated', fmtNum(m.propagated_distance, 3), 'signal']);
    tiles.push(['residual', fmtNum(m.residual_distance, 3), 'signal']);
    tiles.push(['recovery', fmtNum(m.recovery, 3), 'accent']);
  }

  if (mode === 'control') {
    tiles.push(['avg div', fmtNum(summary.avg_raw_div_mean, 4), 'accent']);
    tiles.push(['avg score', fmtNum(summary.avg_score_mean, 4), 'accent']);
  }

  if (mode === 'stress') {
    const r = detail.results || {};
    const m = r.metrics || {};
    tiles.push(['regime', m.regime || '—', 'warn']);
    tiles.push(['match rate', fmtNum(m.token_match_rate, 3), 'accent']);
    tiles.push(['recovery', fmtNum(m.recovery_ratio, 3)]);
  }

  if (mode === 'branchpoints') {
    const m = summary.metrics || {};
    const validity = summary.protocol_validity || {};
    tiles.push(['local rows', validity.rows ?? '—', 'accent']);
    tiles.push(['argmax flip rate', fmtNum(m.argmax_flip_rate, 3), 'warn']);
    tiles.push(['sampled flip rate', fmtNum(m.sampled_flip_rate, 3), 'signal']);
  }

  if (mode === 'propagation') {
    const m = summary.metrics || {};
    const validity = summary.protocol_validity || {};
    const firstLayer = String((summary.capture_layers || [])[0] ?? '');
    const injection = m.layer_summary?.[firstLayer] || {};
    const terminal = m.terminal_summary?.final_norm || {};
    tiles.push(['paired rows', validity.rows ?? '—', 'accent']);
    tiles.push([
      'injection rel.',
      fmtNum(injection.delta_relative_clean?.mean, 3),
      'warn',
    ]);
    tiles.push([
      'final norm rel.',
      fmtNum(terminal.delta_relative_clean?.mean, 3),
      'signal',
    ]);
    tiles.push(['mean logit KL', fmtNum(m.mean_logit_kl, 4), 'accent']);
    tiles.push(['argmax flip rate', fmtNum(m.argmax_flip_rate, 3), 'warn']);
  }

  tiles.forEach(([label, value, cls]) => {
    const tile = document.createElement('div');
    tile.className = 'metric-tile';
    tile.innerHTML = `
      <div class="metric-tile-label">${label}</div>
      <div class="metric-tile-value ${cls || ''}">${value}</div>
    `;
    row.appendChild(tile);
  });

  return row;
}

// ---------------------------------------------------------------------------
// Advisory card
// ---------------------------------------------------------------------------

const ADVISORY_FLAG_TONE = {
  'no-op': 'danger',
  'runaway': 'danger',
  'persistent-effect': 'warn',
  'noise-absorbed': 'warn',
  'nominal': 'good',
  'good-recovery': 'good',
};

function advisoryFlagTone(flag) {
  return ADVISORY_FLAG_TONE[flag] || 'neutral';
}

function primaryAdvisoryTone(flags) {
  // Pick the most severe tone among the flags (danger > warn > good > neutral).
  const order = { danger: 3, warn: 2, good: 1, neutral: 0 };
  let best = 'neutral';
  for (const f of flags || []) {
    const t = advisoryFlagTone(f);
    if (order[t] > order[best]) best = t;
  }
  return best;
}

function renderAdvisory(body, detail) {
  const advisory = detail.summary?.advisory || detail.results?.advisory;
  if (!advisory || advisory.error) return;

  const flags = Array.isArray(advisory.flags) ? advisory.flags : [];
  const tone = primaryAdvisoryTone(flags);

  const card = document.createElement('div');
  card.className = `advisory-card advisory-tone-${tone}`;

  // Header: summary_line + flags
  const head = document.createElement('div');
  head.className = 'advisory-head';
  const summaryLine = document.createElement('div');
  summaryLine.className = `advisory-summary advisory-summary-${tone}`;
  summaryLine.textContent = advisory.summary_line || 'Advisory';
  head.appendChild(summaryLine);

  if (flags.length) {
    const flagWrap = document.createElement('div');
    flagWrap.className = 'advisory-flags';
    flags.forEach((f) => {
      const chip = document.createElement('span');
      chip.className = `advisory-flag advisory-flag-${advisoryFlagTone(f)}`;
      chip.textContent = f;
      flagWrap.appendChild(chip);
    });
    head.appendChild(flagWrap);
  }
  card.appendChild(head);

  // Observations
  const obs = Array.isArray(advisory.observations) ? advisory.observations : [];
  if (obs.length) {
    const title = document.createElement('div');
    title.className = 'advisory-subtitle';
    title.textContent = 'Observations';
    card.appendChild(title);
    const ul = document.createElement('ul');
    ul.className = 'advisory-list';
    obs.forEach((line) => {
      const li = document.createElement('li');
      li.textContent = line;
      ul.appendChild(li);
    });
    card.appendChild(ul);
  }

  // Likely causes (only if non-empty)
  const causes = Array.isArray(advisory.likely_causes) ? advisory.likely_causes : [];
  if (causes.length) {
    const title = document.createElement('div');
    title.className = 'advisory-subtitle';
    title.textContent = 'Likely causes';
    card.appendChild(title);
    const ul = document.createElement('ul');
    ul.className = 'advisory-list';
    causes.forEach((line) => {
      const li = document.createElement('li');
      li.textContent = line;
      ul.appendChild(li);
    });
    card.appendChild(ul);
  }

  // Next actions
  const actions = Array.isArray(advisory.next_actions) ? advisory.next_actions : [];
  if (actions.length) {
    const title = document.createElement('div');
    title.className = 'advisory-subtitle';
    title.textContent = 'Next actions';
    card.appendChild(title);
    const chipRow = document.createElement('div');
    chipRow.className = 'advisory-actions';
    actions.forEach((action) => {
      if (!action || typeof action !== 'object') return;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'advisory-action-chip';
      btn.textContent = action.label || '(unnamed action)';
      btn.addEventListener('click', () => {
        applyAdvisoryAction(detail, action);
      });
      chipRow.appendChild(btn);
    });
    card.appendChild(chipRow);
  }

  // Confidence badge
  if (advisory.confidence) {
    const badge = document.createElement('span');
    const conf = String(advisory.confidence).toLowerCase();
    badge.className = `advisory-confidence advisory-confidence-${conf}`;
    badge.textContent = `confidence: ${conf}`;
    card.appendChild(badge);
  }

  body.appendChild(card);
}

function applyAdvisoryAction(detail, action) {
  const params = action?.params || {};
  const mode = detail.mode;
  const cfg = detail.config?.config || detail.config || {};

  // Switch to Launch tab
  try { switchView('launch'); } catch (e) { /* fall through */ }
  if (mode && [
    'observe',
    'hysteresis',
    'stress',
    'branchpoints',
    'propagation',
    'control',
  ].includes(mode)) {
    try { switchMode(mode); } catch (e) { /* ignore */ }
  }

  // Merge current config + suggested params, then fill the form.
  const merged = { ...cfg, ...params };
  try {
    applyConfigToForm(mode, merged);
  } catch (err) {
    console.warn('advisory: could not auto-fill form', err, merged);
  }
  console.log('advisory action applied:', { mode, params, merged });

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function setInputValue(selector, value) {
  const el = $(selector);
  if (!el || value === undefined || value === null) return;
  if (el.type === 'checkbox') {
    el.checked = Boolean(value);
  } else {
    el.value = String(value);
  }
}

function applyConfigToForm(mode, cfg) {
  // Common fields
  setInputValue('#f-model', cfg.model);
  setInputValue('#f-prompt', cfg.prompt ?? cfg.original_question);
  setInputValue('#f-max-tokens', cfg.max_tokens);
  setInputValue('#f-seed', cfg.seed);
  setInputValue('#f-probe-layers', cfg.probe_layers);

  if (mode === 'stress' || mode === 'branchpoints' || mode === 'propagation') {
    setInputValue('#f-layer', cfg.layer);
    setInputValue('#f-intervention', cfg.intervention_type);
    setInputValue('#f-magnitude', cfg.magnitude);
    setInputValue('#f-absolute-magnitude', cfg.absolute_magnitude);
    if (mode === 'stress') {
      setInputValue('#f-start', cfg.start);
      setInputValue('#f-duration', cfg.duration);
    }
  }
  if (mode === 'hysteresis') {
    setInputValue('#f-perturbation-mode', cfg.perturbation_mode);
    setInputValue('#f-noise-layer', cfg.noise_layer);
    setInputValue('#f-noise-magnitude', cfg.noise_magnitude);
    setInputValue('#f-noise-start', cfg.noise_start);
    setInputValue('#f-noise-duration', cfg.noise_duration);
    setInputValue('#f-recovery-tokens', cfg.recovery_tokens);
  }
  if (mode === 'control') {
    setInputValue('#f-measure-layer', cfg.measure_layer);
    setInputValue('#f-act-layer', cfg.act_layer);
    setInputValue('#f-control-type', cfg.intervention_type);
    setInputValue('#f-shadow', cfg.shadow);
  }
  try { updateModelHint(); } catch (e) { /* ignore */ }
}

function renderObserveCharts(events) {
  const ts = events.map((e, i) => e.t ?? i);
  const div = events.map((e) => e.diagnostics?.divergence ?? null);
  const ent = events.map((e) => e.diagnostics?.spectral?.spectral_entropy ?? null);
  const hid = events.map((e) => e.hidden_post_norm ?? e.hidden_pre_norm ?? null);

  Plotly.react('detail-chart-div', [{
    x: ts, y: div,
    type: 'scatter', mode: 'lines+markers',
    line: { color: COLORS.accent, width: 2, shape: 'spline', smoothing: 0.5 },
    marker: { size: 4, color: COLORS.accent },
    fill: 'tozeroy', fillcolor: 'rgba(124, 197, 255, 0.12)',
    hovertemplate: 't=%{x}<br>div=%{y:.4f}<extra></extra>',
  }], { ...BASE_LAYOUT, margin: { t: 8, r: 14, b: 30, l: 52 } }, PLOTLY_CONFIG);

  Plotly.react('detail-chart-ent', [{
    x: ts, y: ent,
    type: 'scatter', mode: 'lines',
    line: { color: COLORS.signal, width: 2, shape: 'spline', smoothing: 0.5 },
    hovertemplate: 't=%{x}<br>H=%{y:.3f}<extra></extra>',
  }], BASE_LAYOUT, PLOTLY_CONFIG);

  Plotly.react('detail-chart-hid', [{
    x: ts, y: hid,
    type: 'scatter', mode: 'lines',
    line: { color: COLORS.warn, width: 2, shape: 'spline', smoothing: 0.5 },
    hovertemplate: 't=%{x}<br>‖h‖=%{y:.2f}<extra></extra>',
  }], BASE_LAYOUT, PLOTLY_CONFIG);
}

function renderHysteresisDetailCharts(summary) {
  const metrics = summary.metrics || {};
  const exposure = metrics.distance_curve_exposure || [];
  const recovery = metrics.distance_curve_recovery || [];

  Plotly.react('detail-chart-hyst', [{
    x: exposure.map((point) => point.global_decision_index),
    y: exposure.map((point) => point.distance),
    name: 'exposure distance',
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: COLORS.warn, width: 2 },
  }, {
    x: recovery.map((point) => point.global_decision_index),
    y: recovery.map((point) => point.distance),
    name: 'recovery distance',
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: COLORS.signal, width: 2 },
  }], {
    ...BASE_LAYOUT,
    showlegend: true,
    legend: { orientation: 'h', x: 0, y: -0.2, font: { color: COLORS.textDim } },
    margin: { t: 10, r: 14, b: 60, l: 44 },
  }, PLOTLY_CONFIG);
}

function renderPropagationDetailChart(summary) {
  const metrics = summary.metrics || {};
  const layerSummary = metrics.layer_summary || {};
  const layers = Object.keys(layerSummary)
    .map((value) => Number(value))
    .sort((a, b) => a - b);
  const labels = layers.map((layer) => `L${layer}`);
  const relative = layers.map(
    (layer) => layerSummary[String(layer)]?.delta_relative_clean?.mean ?? null,
  );
  const amplification = layers.map(
    (layer) => layerSummary[String(layer)]?.amplification_vs_injection?.mean ?? null,
  );
  const terminal = metrics.terminal_summary?.final_norm || {};
  if (summary.terminal_capture?.available) {
    labels.push('final norm');
    relative.push(terminal.delta_relative_clean?.mean ?? null);
    amplification.push(terminal.amplification_vs_injection?.mean ?? null);
  }

  Plotly.react('detail-chart-propagation', [{
    x: labels,
    y: relative,
    name: 'relative delta',
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: COLORS.signal, width: 2 },
    marker: { size: 5 },
    hovertemplate: '%{x}<br>relative=%{y:.5f}<extra></extra>',
  }, {
    x: labels,
    y: amplification,
    name: 'absolute amplification',
    type: 'scatter',
    mode: 'lines+markers',
    line: { color: COLORS.warn, width: 2, dash: 'dot' },
    marker: { size: 5 },
    yaxis: 'y2',
    hovertemplate: '%{x}<br>amp=%{y:.5f}<extra></extra>',
  }], {
    ...BASE_LAYOUT,
    showlegend: true,
    legend: { orientation: 'h', x: 0, y: -0.2, font: { color: COLORS.textDim } },
    margin: { t: 10, r: 52, b: 60, l: 52 },
    yaxis: {
      ...BASE_LAYOUT.yaxis,
      title: { text: 'relative delta', font: { size: 10 } },
    },
    yaxis2: {
      title: { text: 'amplification', font: { size: 10 } },
      overlaying: 'y',
      side: 'right',
      gridcolor: 'rgba(0,0,0,0)',
      tickfont: { color: COLORS.warn },
    },
  }, PLOTLY_CONFIG);
}

function renderControlCharts(events) {
  const ts = events.map((e, i) => e.t ?? i);
  const div = events.map((e) => e.diagnostics?.divergence ?? e.raw_divergence ?? null);
  const score = events.map((e) => e.score ?? null);

  const traces = [
    {
      x: ts, y: div,
      name: 'divergence',
      type: 'scatter', mode: 'lines',
      line: { color: COLORS.accent, width: 2 },
      hovertemplate: 't=%{x}<br>div=%{y:.4f}<extra></extra>',
    },
  ];
  if (score.some((v) => v !== null)) {
    traces.push({
      x: ts, y: score,
      name: 'score',
      type: 'scatter', mode: 'lines',
      line: { color: COLORS.warn, width: 2, dash: 'dot' },
      yaxis: 'y2',
      hovertemplate: 't=%{x}<br>score=%{y:.4f}<extra></extra>',
    });
  }
  Plotly.react('detail-chart-ctrl', traces, {
    ...BASE_LAYOUT,
    showlegend: true,
    legend: { orientation: 'h', x: 0, y: -0.2, font: { color: COLORS.textDim } },
    margin: { t: 10, r: 44, b: 50, l: 52 },
    yaxis2: { overlaying: 'y', side: 'right', gridcolor: 'transparent', tickfont: { color: COLORS.textMuted } },
  }, PLOTLY_CONFIG);
}

function buildPhaseOutputs(outputs) {
  const wrap = document.createElement('div');
  wrap.className = 'detail-outputs';
  HYSTERESIS_PHASES.forEach((phase) => {
    const box = document.createElement('div');
    box.className = 'detail-output-box';
    const color = phase.startsWith('clean') ? COLORS.phaseBase : COLORS.phasePerturb;
    box.innerHTML = `
      <div class="detail-output-label" data-phase="${phase}">
        <span class="phase-dot" style="background: ${color}"></span>
        ${phase.replaceAll('_', ' ').toUpperCase()}
      </div>
      <pre>${escapeHTML(outputs[phase] || '(not captured)')}</pre>
    `;
    wrap.appendChild(box);
  });
  return wrap;
}

function escapeHTML(s) {
  if (!s) return '';
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

// ---------------------------------------------------------------------------
// Form handling
// ---------------------------------------------------------------------------

function buildPayload() {
  const payload = {
    mode: state.mode,
    model: $('#f-model').value,
    prompt: $('#f-prompt').value,
    max_tokens: parseInt($('#f-max-tokens').value, 10),
    seed: parseInt($('#f-seed').value, 10),
    probe_layers: $('#f-probe-layers').value,
  };
  const seedsRaw = $('#f-seeds').value.trim();
  if (seedsRaw) payload.seeds = seedsRaw;

  if (
    state.mode === 'stress'
    || state.mode === 'branchpoints'
    || state.mode === 'propagation'
  ) {
    payload.layer = parseInt($('#f-layer').value, 10);
    payload.intervention_type = $('#f-intervention').value;
    payload.magnitude = parseFloat($('#f-magnitude').value);
    payload.absolute_magnitude = $('#f-absolute-magnitude').checked;
    if (state.mode === 'stress') {
      payload.start = parseInt($('#f-start').value, 10);
      payload.duration = parseInt($('#f-duration').value, 10);
    }
  }
  if (state.mode === 'hysteresis') {
    payload.perturbation_mode = $('#f-perturbation-mode').value;
    payload.noise_layer = parseInt($('#f-noise-layer').value, 10);
    payload.noise_magnitude = parseFloat($('#f-noise-magnitude').value);
    payload.noise_start = parseInt($('#f-noise-start').value, 10);
    payload.noise_duration = parseInt($('#f-noise-duration').value, 10);
    payload.recovery_tokens = parseInt($('#f-recovery-tokens').value, 10);
  }
  if (state.mode === 'control') {
    payload.measure_layer = parseInt($('#f-measure-layer').value, 10);
    payload.act_layer = parseInt($('#f-act-layer').value, 10);
    payload.intervention_type = $('#f-control-type').value;
    payload.shadow = $('#f-shadow').checked;
  }
  return payload;
}

function showLaunchMsg(text, kind = 'ok') {
  const el = $('#launch-msg');
  el.textContent = text;
  el.className = 'launch-msg ' + (kind === 'error' ? 'is-error' : 'is-ok');
  el.hidden = false;
  setTimeout(() => { el.hidden = true; }, 4000);
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

function bindUI() {
  $$('.nav-btn').forEach((btn) => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });

  $$('.mode-tab').forEach((btn) => {
    btn.addEventListener('click', () => switchMode(btn.dataset.mode));
  });

  $('#launch-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      await launchRun(buildPayload());
      showLaunchMsg('Run launched — switching to live view.');
    } catch (err) {
      showLaunchMsg('Failed: ' + err.message, 'error');
    }
  });

  $('#btn-stop').addEventListener('click', async () => {
    try { await stopRun(); } catch {}
  });

  $('#btn-clear-logs').addEventListener('click', () => {
    $('#log-feed').textContent = '';
  });

  $('#btn-refresh-archive').addEventListener('click', fetchArchive);
  $('#btn-close-detail').addEventListener('click', closeRunDetail);

  $('#btn-goto-compare').addEventListener('click', () => switchView('compare'));
  $('#btn-clear-selection').addEventListener('click', () => {
    state.compareSelection.clear();
    updateCompareSelectionUI();
    renderArchive();
  });
  $('#btn-compare-clear').addEventListener('click', () => {
    state.compareSelection.clear();
    updateCompareSelectionUI();
    renderCompareView();
    renderArchive();
  });

  $$('.filter-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      state.archiveFilter = chip.dataset.filter;
      $$('.filter-chip').forEach((c) => c.classList.toggle('is-active', c === chip));
      renderArchive();
    });
  });
}

async function main() {
  bindUI();
  switchMode('observe');
  try { await fetchModels(); } catch (e) { console.warn('models', e); }
  try { await fetchArchive(); } catch (e) { console.warn('archive', e); }
  initLiveCharts();
  connectSSE();
}

main();
