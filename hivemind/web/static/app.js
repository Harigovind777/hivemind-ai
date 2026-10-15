'use strict';

/* ------------------------------------------------------------------ helpers */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else if (key.startsWith('on')) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

async function api(method, path, body) {
  let res;
  try {
    res = await fetch(path, {
      method,
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (err) {
    $('#offline').hidden = false;
    throw err;
  }
  let data = null;
  try { data = await res.json(); } catch { /* empty body */ }
  if (!res.ok) {
    const err = new Error((data && data.error) || `${res.status} ${res.statusText}`);
    err.status = res.status;
    err.data = data || {};
    throw err;
  }
  return data;
}

const fmtInt = (n) => Number(n || 0).toLocaleString('en-US');
const fmtPct = (x) => `${Math.round(x * 1000) / 10}%`;
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function fmtDuration(s) {
  s = Math.round(s || 0);
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
}

function fmtDate(epoch) {
  return new Date(epoch * 1000).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

/* ------------------------------------------------------------------ vocabulary */

const ROLE_NAMES = {
  research: 'Research agent', coding: 'Coding agent', testing: 'Testing agent', security: 'Security agent',
  documentation: 'Documentation agent', ethics: 'Ethics agent', coordinator: 'Coordinator',
};
const AGENT_ROLES = ['research', 'coding', 'testing', 'security', 'documentation', 'ethics'];
const STRATEGIES = [
  ['solo', 'Solo agent'], ['first', 'First candidate'], ['vote', 'Majority vote'], ['debate', 'Debate'],
  ['tests_only', 'Trust the tests'], ['arbitration_select', 'Arbitration, select only'], ['arbitration', 'Arbitration with repair'],
];
const HINTS = [
  'write the clearest straightforward solution.',
  'work through the docstring examples by hand first, then implement exactly the rule they imply.',
  'list the edge cases first (empty input, zero, negatives, duplicates, boundaries), then implement.',
  'consider an alternative algorithm to the obvious one and use whichever is more robust.',
];
const CLAUDE_MODELS = ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5', 'claude-fable-5-1'];
const EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'];

function describePhase(phase) {
  const [p, a, b] = String(phase).split(':');
  switch (p) {
    case 'solo': return ['Solo baseline', 'Works alone, with no brief, no tests and no teammates. This is the single-agent baseline.'];
    case 'brief': return ['Research brief', 'States the input/output contract, the edge cases and any ambiguity. Every candidate sees this brief.'];
    case 'candidate': return [`Candidate ${Number(a) + 1}`, `Writes a solution using the brief. Its approach hint: ${HINTS[Number(a) % HINTS.length]}`];
    case 'tests': return ['Team tests', 'Writes test cases from the specification alone, without seeing any code.'];
    case 'arbiter': return ['Ruling on disputed cases', 'Sees the specification and the real outputs, shuffled and without vote counts, and decides which output each case requires.'];
    case 'repair': return ['Repair', 'The chosen candidate still failed a verified case, so it gets one chance to fix it. The fix is kept only if it passes more verified cases.'];
    case 'debate': return [`Debate, candidate ${Number(b) + 1}`, "For the debate baseline: reads the teammates' code, points out bugs and revises. Nothing is run."];
    case 'judge': return ['Debate verdict', 'Picks the debate winner from code and arguments alone, without running anything.'];
    case 'review': return [`${ROLE_NAMES[a] || a} review`, 'Reads the final answer chosen by arbitration.'];
    case 'draft': return ['Spec draft', 'Turns your request into a function signature, a docstring and examples.'];
    case 'classify': return ['Reading your message', 'Decides whether you asked a question or asked for a change.'];
    case 'question': return ['Answer', "Answers from the round's evidence."];
    default: return [String(phase), ''];
  }
}

const who = (role, label) => h('span', { class: 'who', style: `--role: var(--r-${role}, var(--muted))` }, label || ROLE_NAMES[role] || role);
const chip = (passed) => h('span', { class: `chip ${passed ? 'pass' : 'fail'}` }, passed ? 'pass' : 'fail');

/* ------------------------------------------------------------------ transcript entries */

function renderReply(text) {
  const box = h('div', { class: 'reply' });
  if (!text || !String(text).trim()) {
    box.append(h('p', { class: 'prose' }, 'No reply.'));
    return box;
  }
  String(text).split('```').forEach((part, i) => {
    if (i % 2 === 1) {
      box.append(h('pre', { class: 'code' }, part.replace(/^[\w+-]*\n/, '').replace(/\s+$/, '')));
    } else if (part.trim()) {
      box.append(h('div', { class: 'prose' }, part.trim().replace(/\*\*(.+?)\*\*/g, '$1').replace(/^#{1,6}\s+/gm, '')));
    }
  });
  return box;
}

function entryHead(left, what, meta) {
  return h('header', { class: 'entry-head' }, left, h('span', { class: 'what' }, what), meta ? h('span', { class: 'meta' }, meta) : null);
}

function turnEntry(t, pending = false) {
  const [title, note] = describePhase(t.phase);
  const meta = [];
  if (t.model) meta.push(t.model);
  if (pending) meta.push('working');
  else {
    if (t.output_tokens != null) meta.push(`${fmtInt((t.input_tokens || 0) + (t.output_tokens || 0))} tokens`);
    if (t.latency_s != null) meta.push(`${Number(t.latency_s).toFixed(1)} s`);
  }
  const el = h('article', { class: `entry${pending ? ' pending' : ''}${t.error ? ' failed' : ''}`, dataset: { phase: t.phase } },
    entryHead(who(t.role), title, meta.join(', ')),
    note ? h('p', { class: 'note' }, note) : null);
  if (!pending) {
    el.append(t.error ? h('div', { class: 'reply' }, h('p', {}, `The model call failed: ${t.error}`)) : renderReply(t.response));
    if (t.prompt) el.append(h('details', {}, h('summary', {}, 'Show the prompt it received'), h('pre', { class: 'code' }, t.prompt)));
  }
  return el;
}

function problemEntry(e) {
  return h('article', { class: 'entry system' },
    entryHead(who('none', 'Problem'), `${e.task_id}, ${e.entry_point}`, e.config ? `config ${e.config}` : null),
    h('pre', { class: 'code' }, e.spec));
}

const fact = (label, value) => h('span', {}, `${label} `, h('b', {}, String(value)));

function evidenceEntry(e) {
  return h('article', { class: 'entry system' },
    entryHead(who('none', 'Sandbox'), 'Every candidate ran on every test case'),
    h('div', { class: 'facts' },
      fact('Candidates', e.candidates), fact('Test cases', e.cases), fact('From the docstring', e.spec_cases),
      fact('From the Testing agent', e.tester_cases), fact('Different behaviours', e.distinct_behaviours),
      fact('Flagged by the security scan', e.unsafe_candidates)),
    h('p', { class: 'note' }, e.distinct_behaviours > 1
      ? 'The candidates disagree. Where their outputs differ, the Coordinator rules on what the specification requires.'
      : 'All candidates behaved the same way. Any dispute left is between them and the Testing agent.'));
}

function rulingEntry(rulings) {
  const cases = rulings.map((r) => h('div', { class: 'case' },
    h('code', { class: 'call' }, r.call),
    h('div', { class: 'exhibits' }, r.options.map((option, j) => h('span', { class: `exhibit${option === r.choice ? ' chosen' : ''}` },
      h('span', { class: 'letter' }, String.fromCharCode(65 + j)), h('span', {}, option)))),
    h('p', { class: 'verdict' }, r.choice == null ? 'The Coordinator answered unsure, so this case is not used.' : '')));
  return h('article', { class: 'entry ruling' },
    entryHead(who('coordinator'), 'Rulings'),
    h('p', { class: 'note' }, 'Each option is a real output. The filled option is the one the Coordinator chose; it never saw which agent produced an output or how many agreed.'),
    h('div', { class: 'cases' }, cases));
}

function markReference(entry, rulings) {
  $$('.case', entry).forEach((el, i) => {
    const r = rulings[i];
    const verdict = $('.verdict', el);
    if (!r || r.reference_agrees == null) return;
    verdict.className = `verdict ${r.reference_agrees ? 'agrees' : 'differs'}`;
    verdict.textContent = r.reference_agrees
      ? "Matches the benchmark's reference answer (checked after the run)."
      : "Differs from the benchmark's reference answer (checked after the run).";
  });
}

function outcomeEntry(outcome, savedTo) {
  const findings = outcome.review && outcome.review.static_security_findings;
  return h('article', { class: 'entry system' },
    entryHead(who('none', 'Result'), 'Graded on the hidden tests no agent saw'),
    h('ul', { class: 'outcomes' }, STRATEGIES.map(([key, label]) =>
      h('li', {}, h('span', {}, label), chip(outcome.answers[key].passed)))),
    h('p', { class: 'note' },
      `Arbitration's answer came from ${outcome.answers.arbitration.source}. `,
      outcome.candidate_passed.some(Boolean) ? 'At least one candidate was correct.' : 'No candidate was correct.'),
    findings ? h('p', { class: 'note' }, findings.length ? `Static security scan: ${findings.join('; ')}.` : 'Static security scan found nothing risky.') : null,
    savedTo ? h('p', { class: 'hint' }, `Saved to ${savedTo}`) : null);
}

function systemEntry(title, text, failed) {
  return h('article', { class: `entry system${failed ? ' failed' : ''}` }, entryHead(who('none', title), ''), h('div', { class: 'reply' }, h('p', {}, text)));
}

/* ------------------------------------------------------------------ state */

const state = {
  status: null, tasks: [], configs: [], live: null, runReady: false,
  jobId: null, jobTimer: null, formBase: null, selectedConfig: null, backends: null,
};

/* ------------------------------------------------------------------ status bar */

async function refreshStatus() {
  try {
    state.status = await api('GET', '/api/status');
    $('#offline').hidden = true;
  } catch {
    return;
  }
  const s = state.status;
  const items = [];
  items.push(s.ollama.reachable
    ? ['ok', `Ollama running, ${plural(s.ollama.models.length, 'model')}`]
    : ['bad', 'Ollama not running']);
  items.push(s.claude.available ? ['ok', `Claude credentials from ${s.claude.source}`] : ['', 'No Claude credentials']);
  if (s.current) {
    items.push(['busy', s.current.kind === 'live' ? `Running ${s.current.task_id}`
      : s.current.kind === 'chat' ? 'Chat round running'
        : `Experiment ${s.current.done} of ${s.current.total}`]);
  }
  $('#health').replaceChildren(...items.map(([cls, text]) => h('span', { class: cls }, text)));

  // Readiness depends on Ollama and Claude, so check it again when either comes or goes.
  const backends = JSON.stringify([s.ollama.reachable, s.ollama.models, s.claude.available]);
  if (state.backends !== null && backends !== state.backends) {
    checkRunConfig();
    checkExpConfig();
  }
  state.backends = backends;

  const cur = s.current;
  if (cur && cur.kind === 'live' && (!state.live || state.live.id !== cur.id)) attachLive(cur.id);
  if (cur && cur.kind === 'experiment' && state.jobId !== cur.id) followJob(cur.id);
  updateRunButtons();
  updateExpButton();
}

/* ------------------------------------------------------------------ tabs */

const TABS = ['chat', 'run', 'agents', 'experiments', 'results'];

function showTab() {
  const tab = TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'chat';
  for (const name of TABS) $(`#tab-${name}`).hidden = name !== tab;
  $$('.tabs a').forEach((a) => (a.dataset.tab === tab ? a.setAttribute('aria-current', 'page') : a.removeAttribute('aria-current')));
  if (tab === 'results') loadResults();
  if (tab === 'agents' && !state.formBase && state.configs.length) openConfig(state.selectedConfig || defaultConfig());
  if (tab === 'chat') showChatTab();
}

/* ------------------------------------------------------------------ configs (shared) */

function defaultConfig() {
  return (state.configs.find((c) => c.name === 'pilot-local') || state.configs[0] || {}).name;
}

function modelsLine(c) {
  if (!c || c.errors && c.errors.file) return '';
  let text = `Agents use ${c.agents_model}; the Coordinator uses ${c.coordinator_model}.`;
  if (c.role_overrides && c.role_overrides.length) text += ` Own models for ${c.role_overrides.map((r) => ROLE_NAMES[r] || r).join(', ')}.`;
  return text;
}

async function loadConfigs() {
  state.configs = await api('GET', '/api/configs');
  for (const sel of [$('#run-config'), $('#exp-config'), $('#chat-config')]) {
    const keep = sel.value;
    sel.replaceChildren(...state.configs.map((c) =>
      h('option', { value: c.name }, c.name + (Object.keys(c.errors || {}).length ? ' (has errors)' : ''))));
    sel.value = state.configs.some((c) => c.name === keep) ? keep : (defaultConfig() || '');
  }
  renderConfigList();
}

function showFixes(list, { problems = [], errors = {} }, okText) {
  const items = [
    ...Object.entries(errors).map(([field, message]) => `${field}: ${message}. Fix it on the Agents tab.`),
    ...problems,
  ];
  list.classList.toggle('ok', !items.length && !!okText);
  list.replaceChildren(...(items.length ? items : okText ? [okText] : []).map((t) => h('li', {}, t)));
  list.hidden = !items.length && !okText;
  return !items.length;
}

/* ------------------------------------------------------------------ Run tab */

const STEPS = [
  ['solo', 'Solo baseline'], ['brief', 'Research brief'], ['candidates', 'Candidates'], ['tests', 'Team tests'],
  ['sandbox', 'Sandbox run'], ['arbiter', 'Arbitration'], ['repair', 'Repair'], ['debate', 'Debate baseline'],
  ['judge', 'Debate verdict'], ['review', 'Reviews'],
];
const PHASE_STEP = { solo: 'solo', brief: 'brief', candidate: 'candidates', tests: 'tests', arbiter: 'arbiter', repair: 'repair', debate: 'debate', judge: 'judge', review: 'review' };
const stepOf = (phase) => PHASE_STEP[String(phase).split(':')[0]];

function resetPipeline(review) {
  $('#pipeline').replaceChildren(...STEPS.filter(([id]) => review || id !== 'review').map(([id, name]) =>
    h('li', { dataset: { step: id, state: 'waiting' } }, h('span', { class: 'step-name' }, name), h('span', { class: 'step-detail' }, ''))));
}

function setStep(id, stepState, detail) {
  const li = $(`#pipeline li[data-step="${id}"]`);
  if (!li) return;
  if (stepState === 'working') {
    for (const other of $$('#pipeline li[data-state="working"]')) if (other !== li) other.dataset.state = 'done';
  }
  li.dataset.state = stepState;
  if (detail != null) $('.step-detail', li).textContent = detail;
}

function finishPipeline(status) {
  for (const li of $$('#pipeline li')) {
    if (li.dataset.state === 'working') li.dataset.state = status === 'done' ? 'done' : 'failed';
    else if (li.dataset.state === 'waiting') {
      li.dataset.state = 'skipped';
      const detail = $('.step-detail', li);
      if (!detail.textContent) detail.textContent = status === 'done' ? 'not needed' : 'not reached';
    }
  }
}

function setRunState(text) { $('#run-state').textContent = text; }

function appendEntry(el) {
  const nearBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
  $('#feed').append(el);
  if (nearBottom) el.scrollIntoView({ block: 'end', behavior: reducedMotion ? 'auto' : 'smooth' });
}

function attachLive(id) {
  if (state.live && state.live.es) state.live.es.close();
  const live = { id, lastSeq: 0, pending: new Map(), finished: false, rulingEntry: null, candidates: null };
  state.live = live;
  live.es = new EventSource(`/api/live/${encodeURIComponent(id)}/events`);
  live.es.onmessage = (msg) => handleEvent(live, JSON.parse(msg.data));
  updateRunButtons();
}

function handleEvent(live, e) {
  if (state.live !== live || e.seq <= live.lastSeq) return;
  live.lastSeq = e.seq;
  switch (e.type) {
    case 'started': {
      live.candidates = e.candidates;
      resetPipeline(e.review);
      $('#feed').replaceChildren(problemEntry(e));
      setRunState(`Running ${e.task_id} with ${e.config}.`);
      break;
    }
    case 'turn_started': {
      const step = stepOf(e.phase);
      let detail = `${ROLE_NAMES[e.role] || e.role}, ${e.model}`;
      if (step === 'candidates') detail = `${describePhase(e.phase)[0]}${live.candidates ? ` of ${live.candidates}` : ''}, ${e.model}`;
      setStep(step, 'working', detail);
      const el = turnEntry(e, true);
      live.pending.set(e.phase, el);
      appendEntry(el);
      break;
    }
    case 'turn': {
      const el = turnEntry(e);
      const pending = live.pending.get(e.phase);
      if (pending) { pending.replaceWith(el); live.pending.delete(e.phase); } else appendEntry(el);
      if (e.error) setStep(stepOf(e.phase), 'working', 'a model call failed');
      break;
    }
    case 'evidence': {
      setStep('tests', 'done');
      setStep('sandbox', 'done', plural(e.distinct_behaviours, 'different behaviour'));
      appendEntry(evidenceEntry(e));
      break;
    }
    case 'rulings': {
      if (e.disputes) {
        setStep('arbiter', 'done', plural(e.disputes, 'disputed case'));
        live.rulingEntry = rulingEntry(e.rulings);
        appendEntry(live.rulingEntry);
      } else {
        setStep('arbiter', 'skipped', 'no disputes');
        appendEntry(systemEntry('Arbitration', 'No disputed cases: every program that ran agreed with the verified expectations, so there was nothing to rule on.'));
      }
      break;
    }
    case 'outcome': {
      finishPipeline('done');
      if (live.rulingEntry) markReference(live.rulingEntry, e.outcome.diagnostics.rulings);
      appendEntry(outcomeEntry(e.outcome, e.saved_to));
      break;
    }
    case 'error':
      appendEntry(systemEntry('The run stopped', e.message, true));
      break;
    case 'done': {
      live.es.close();
      live.finished = true;
      for (const el of live.pending.values()) el.remove();
      live.pending.clear();
      finishPipeline(e.status);
      setRunState({ done: 'Finished.', cancelled: 'Cancelled.', failed: 'Stopped with an error.' }[e.status] || e.status);
      refreshStatus();
      break;
    }
    default:
      break;
  }
  updateRunButtons();
}

function taskFromInput() {
  const value = $('#run-task').value.trim();
  if (!value) return null;
  const first = value.split(/\s+/)[0];
  const lower = first.toLowerCase();
  return state.tasks.find((t) => t.task_id.toLowerCase() === lower || t.entry_point === first)
    || state.tasks.find((t) => t.task_id.toLowerCase() === `humaneval/${lower}`) || null;
}

function updateTaskSummary() {
  const task = taskFromInput();
  const value = $('#run-task').value.trim();
  $('#run-task-summary').textContent = task
    ? `${task.task_id}, ${task.entry_point}: ${task.summary}`
    : value ? 'No problem with that number or function name.' : '';
  updateRunButtons();
}

async function checkRunConfig() {
  const name = $('#run-config').value;
  const c = state.configs.find((x) => x.name === name);
  $('#run-config-models').textContent = modelsLine(c);
  if (!name) { state.runReady = false; updateRunButtons(); return; }
  try {
    state.runReady = showFixes($('#run-problems'), await api('GET', `/api/configs/${encodeURIComponent(name)}/readiness`));
  } catch (err) {
    state.runReady = showFixes($('#run-problems'), { problems: [err.message] });
  }
  updateRunButtons();
}

function updateRunButtons() {
  const live = state.live;
  const liveRunning = !!(live && !live.finished);
  const current = state.status && state.status.current;
  const busyElsewhere = !!(current && current.kind !== 'live');
  $('#run-start').disabled = liveRunning || busyElsewhere || !state.runReady || !taskFromInput();
  $('#run-cancel').hidden = !liveRunning;
  if (busyElsewhere && !liveRunning) {
    setRunState(current.kind === 'chat'
      ? 'A chat round is running. Runs start again when it finishes or is cancelled on the Chat tab.'
      : 'An experiment is running. Runs start again when it finishes or is cancelled on the Experiments tab.');
  } else if (!liveRunning && /^(An experiment|A chat round)/.test($('#run-state').textContent)) setRunState('');
}

$('#run-task').addEventListener('input', updateTaskSummary);
$('#run-config').addEventListener('change', checkRunConfig);
$('#run-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const task = taskFromInput();
  if (!task) return;
  try {
    const { id } = await api('POST', '/api/live', {
      config: $('#run-config').value, task_id: task.task_id, review: $('#run-review').checked,
    });
    attachLive(id);
  } catch (err) {
    showFixes($('#run-problems'), { problems: (err.data && err.data.problems) || [err.message], errors: (err.data && err.data.errors) || {} });
  }
  refreshStatus();
});
$('#run-cancel').addEventListener('click', async () => {
  if (!state.live) return;
  setRunState('Cancelling once the current model call finishes.');
  try { await api('POST', `/api/live/${encodeURIComponent(state.live.id)}/cancel`); } catch (err) { setRunState(err.message); }
});

/* ------------------------------------------------------------------ Agents tab */

function field(id, label, control, opts = {}) {
  return h('div', { class: `field${opts.wide ? ' wide' : ''}`, dataset: opts.provider ? { provider: opts.provider } : {} },
    h('label', { for: id }, label), control, opts.extra || null, h('p', { class: 'err', hidden: true }));
}

function modelFields(path) {
  const id = `cf-${path.replace(/\./g, '-')}`;
  const wrap = h('div', { class: 'model-fields', dataset: { model: path } },
    field(`${id}-provider`, 'Provider',
      h('select', { id: `${id}-provider`, dataset: { path: `${path}.provider`, key: 'provider' } },
        h('option', { value: 'ollama' }, 'Ollama, on this computer'), h('option', { value: 'anthropic' }, 'Claude, through the API'))),
    field(`${id}-model`, 'Model',
      h('input', { id: `${id}-model`, list: `${id}-models`, spellcheck: 'false', dataset: { path: `${path}.model`, key: 'model' } }),
      { wide: true, extra: h('datalist', { id: `${id}-models` }) }),
    field(`${id}-temperature`, 'Temperature',
      h('input', { id: `${id}-temperature`, type: 'number', step: '0.1', min: '0', max: '2', dataset: { path: `${path}.temperature`, key: 'temperature', num: '1' } }),
      { provider: 'ollama' }),
    field(`${id}-effort`, 'Effort',
      h('select', { id: `${id}-effort`, dataset: { path: `${path}.effort`, key: 'effort' } },
        h('option', { value: '' }, 'Model default'), EFFORTS.map((e) => h('option', { value: e }, e))),
      { provider: 'anthropic' }),
    field(`${id}-max-tokens`, 'Max tokens per reply',
      h('input', { id: `${id}-max-tokens`, type: 'number', min: '1', dataset: { path: `${path}.max_tokens`, key: 'max_tokens', num: '1' } })));
  $(`#${id}-provider`, wrap).addEventListener('change', () => syncProvider(wrap, true));
  return wrap;
}

function syncProvider(wrap, changed) {
  const provider = $('[data-key="provider"]', wrap).value;
  for (const f of $$('.field[data-provider]', wrap)) f.hidden = f.dataset.provider !== provider;
  const models = provider === 'anthropic' ? CLAUDE_MODELS : ((state.status && state.status.ollama.models) || []);
  $('datalist', wrap).replaceChildren(...models.map((m) => h('option', { value: m })));
  if (!changed) return;
  const model = $('[data-key="model"]', wrap);
  const tokens = $('[data-key="max_tokens"]', wrap);
  if (provider === 'anthropic' && !model.value.startsWith('claude')) { model.value = 'claude-opus-5-5'; tokens.value = '16000'; }
  if (provider === 'ollama' && model.value.startsWith('claude')) { model.value = models[0] || ''; tokens.value = '1024'; }
}

function readModel(wrap) {
  const provider = $('[data-key="provider"]', wrap).value;
  const section = { provider, model: $('[data-key="model"]', wrap).value.trim() };
  for (const f of $$('.field', wrap)) {
    if (f.dataset.provider && f.dataset.provider !== provider) continue;
    const input = $('[data-key]', f);
    const key = input.dataset.key;
    if (key === 'provider' || key === 'model' || input.value.trim() === '') continue;
    section[key] = input.dataset.num ? Number(input.value) : input.value;
  }
  return section;
}

function writeModel(wrap, section) {
  section = section || { provider: 'ollama', model: '' };
  $('[data-key="provider"]', wrap).value = section.provider === 'anthropic' ? 'anthropic' : 'ollama';
  for (const input of $$('[data-key]', wrap)) {
    if (input.dataset.key !== 'provider') input.value = section[input.dataset.key] ?? '';
  }
  syncProvider(wrap, false);
}

function buildConfigForm() {
  $('#cf-agents').append(modelFields('agents'));
  $('#cf-coordinator').append(modelFields('coordinator'));
  for (const role of AGENT_ROLES) {
    const mode = h('select', { id: `cf-role-${role}`, 'aria-label': `${ROLE_NAMES[role]} model` },
      h('option', { value: 'team' }, 'Same as the agent team'), h('option', { value: 'own' }, 'Its own model'));
    const fields = modelFields(`roles.${role}`);
    fields.hidden = true;
    mode.addEventListener('change', () => {
      fields.hidden = mode.value !== 'own';
      if (!fields.hidden && !$('[data-key="model"]', fields).value) writeModel(fields, readModel($('#cf-agents .model-fields')));
    });
    $('#cf-roles').append(h('div', { class: 'role-row', dataset: { role } }, who(role), h('div', { class: 'role-body' }, mode, fields)));
  }
  $('#cf-file').addEventListener('input', () => { $('#cf-file-echo').textContent = $('#cf-file').value || 'name'; });
  $('#cf-run-name').addEventListener('input', () => { $('#cf-run-echo').textContent = $('#cf-run-name').value || 'name'; });
}

const PIPELINE_FIELDS = [['candidates', '#cf-candidates'], ['debate_rounds', '#cf-debate'], ['n_tasks', '#cf-n-tasks'],
  ['sample_seed', '#cf-sample-seed'], ['seed', '#cf-seed'], ['workers', '#cf-workers']];

function fillConfigForm(name, config) {
  state.formBase = config;
  state.selectedConfig = name;
  $('#cf-file').value = name;
  $('#cf-file-echo').textContent = name;
  $('#cf-run-name').value = (config.run && config.run.name) || name;
  $('#cf-run-echo').textContent = $('#cf-run-name').value;
  for (const [key, sel] of PIPELINE_FIELDS) $(sel).value = config.run && config.run[key] != null ? config.run[key] : '';
  $('#cf-repair').checked = !config.run || config.run.repair !== false;
  writeModel($('#cf-agents .model-fields'), config.agents);
  writeModel($('#cf-coordinator .model-fields'), config.coordinator);
  for (const row of $$('#cf-roles .role-row')) {
    const own = config.roles && config.roles[row.dataset.role];
    $('select', row).value = own ? 'own' : 'team';
    const fields = $('.model-fields', row);
    fields.hidden = !own;
    writeModel(fields, own || null);
  }
  renderConfigList();
}

function collectConfig() {
  const base = state.formBase || {};
  const run = { ...(base.run || {}), name: $('#cf-run-name').value.trim() };
  for (const [key, sel] of PIPELINE_FIELDS) {
    const value = $(sel).value.trim();
    if (value === '') delete run[key]; else run[key] = Number(value);
  }
  run.repair = $('#cf-repair').checked;
  const config = { run, agents: readModel($('#cf-agents .model-fields')), coordinator: readModel($('#cf-coordinator .model-fields')) };
  const roles = {};
  for (const row of $$('#cf-roles .role-row')) {
    if ($('select', row).value === 'own') roles[row.dataset.role] = readModel($('.model-fields', row));
  }
  if (Object.keys(roles).length) config.roles = roles;
  return config;
}

function clearFormErrors() {
  for (const input of $$('#config-form [aria-invalid]')) input.removeAttribute('aria-invalid');
  for (const err of $$('#config-form .err')) { err.hidden = true; err.textContent = ''; }
  $('#cf-problems').hidden = true;
}

function showFormErrors(errors, problems = []) {
  const leftover = [...problems];
  for (const [path, message] of Object.entries(errors || {})) {
    const input = $(`#config-form [data-path="${CSS.escape(path)}"]`);
    const err = input && input.closest('.field') && $('.err', input.closest('.field'));
    if (input && err && !input.closest('[hidden]')) {
      input.setAttribute('aria-invalid', 'true');
      err.textContent = message;
      err.hidden = false;
    } else {
      leftover.push(`${path}: ${message}`);
    }
  }
  showFixes($('#cf-problems'), { problems: leftover });
  const first = $('#config-form [aria-invalid="true"]');
  if (first) first.focus();
}

async function openConfig(name) {
  if (!name) return;
  clearFormErrors();
  $('#cf-saved').textContent = '';
  try {
    const body = await api('GET', `/api/configs/${encodeURIComponent(name)}`);
    fillConfigForm(name, body.config);
    if (Object.keys(body.errors).length) showFormErrors(body.errors);
    else checkFormReadiness(name);
  } catch (err) {
    showFixes($('#cf-problems'), { problems: [err.message] });
  }
}

async function checkFormReadiness(name) {
  try {
    const r = await api('GET', `/api/configs/${encodeURIComponent(name)}/readiness`);
    showFixes($('#cf-problems'), r, 'Ready to run on this computer.');
  } catch (err) {
    showFixes($('#cf-problems'), { problems: [err.message] });
  }
}

function renderConfigList() {
  $('#config-list').replaceChildren(...state.configs.map((c) => h('li', {},
    h('button', {
      type: 'button', 'aria-current': c.name === state.selectedConfig ? 'true' : 'false',
      onclick: () => { location.hash = '#agents'; openConfig(c.name); },
    },
    h('span', { class: 'name' }, c.name),
    h('span', { class: 'sub' }, Object.keys(c.errors || {}).length ? 'Has errors' : `${c.agents_model} with ${c.coordinator_model}`)))));
}

$('#config-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  clearFormErrors();
  const name = $('#cf-file').value.trim();
  if (name !== state.selectedConfig && state.configs.some((c) => c.name === name)
      && !window.confirm(`configs/${name}.toml already exists. Replace it?`)) return;
  try {
    await api('PUT', `/api/configs/${encodeURIComponent(name)}`, { config: collectConfig() });
    $('#cf-saved').textContent = `Saved configs/${name}.toml`;
    state.selectedConfig = name;
    state.formBase = collectConfig();
    await loadConfigs();
    checkFormReadiness(name);
    checkRunConfig();
    updateExpForm();
  } catch (err) {
    if (err.data && err.data.errors) showFormErrors(err.data.errors);
    else showFormErrors({ file: err.message });
  }
});

/* ------------------------------------------------------------------ Experiments tab */

async function updateExpForm() {
  const name = $('#exp-config').value;
  if (!name) return;
  try {
    const body = await api('GET', `/api/configs/${encodeURIComponent(name)}`);
    const run = body.config.run || {};
    $('#exp-n').value = run.n_tasks || 164;
    $('#exp-where').textContent = `Results go to runs/${run.name}. Problems already there are skipped, so launching again continues a stopped run.`;
  } catch (err) {
    state.expReady = showFixes($('#exp-problems'), { problems: [err.message] });
    updateExpButton();
    return;
  }
  await checkExpConfig();
}

async function checkExpConfig() {
  const name = $('#exp-config').value;
  if (!name) return;
  try {
    state.expReady = showFixes($('#exp-problems'), await api('GET', `/api/configs/${encodeURIComponent(name)}/readiness`));
  } catch (err) {
    state.expReady = showFixes($('#exp-problems'), { problems: [err.message] });
  }
  updateExpButton();
}

function updateExpButton() {
  const current = state.status && state.status.current;
  $('#exp-start').disabled = !!current || !state.expReady;
}

function followJob(id) {
  state.jobId = id;
  clearInterval(state.jobTimer);
  pollJob();
  state.jobTimer = setInterval(pollJob, 2000);
}

async function pollJob() {
  const id = state.jobId;
  let p;
  try { p = await api('GET', `/api/experiments/${encodeURIComponent(id)}`); } catch { return; }
  if (id !== state.jobId) return;
  renderJob(p);
  if (p.status !== 'running' && p.status !== 'pending') {
    clearInterval(state.jobTimer);
    refreshStatus();
  }
}

function renderJob(p) {
  const labels = { pending: 'Starting', running: 'Running', done: 'Finished', failed: 'Stopped with an error', cancelled: 'Cancelled' };
  const runName = p.out_dir.split(/[\\/]/).pop();
  const log = h('pre', { class: 'log' }, p.log.join('\n') || 'Waiting for output.');
  const panel = $('#exp-progress');
  panel.replaceChildren(
    h('div', { class: 'entry-head' }, h('h2', {}, `Experiment with ${p.config}`), h('span', { class: 'meta' }, labels[p.status] || p.status)),
    h('div', { class: 'facts' }, fact('Problems done', `${p.done} of ${p.total}`), fact('Elapsed', fmtDuration(p.elapsed_s)),
      fact('Agents', p.models.agents), fact('Coordinator', p.models.coordinator)),
    h('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': String(p.total), 'aria-valuenow': String(p.done) },
      h('span', { style: `width: ${p.total ? (100 * p.done) / p.total : 0}%` })),
    p.tasks.length ? h('div', { class: 'task-grid', 'aria-label': 'Arbitration result per problem' },
      p.tasks.map((t) => h('span', {
        class: `chip ${t.passed.arbitration ? 'pass' : 'fail'}`,
        title: STRATEGIES.map(([k, label]) => `${label}: ${t.passed[k] ? 'pass' : 'fail'}`).join('\n'),
      }, `${t.task_id.replace('HumanEval/', '#')} ${t.passed.arbitration ? 'pass' : 'fail'}`))) : null,
    h('details', { open: p.status !== 'done' }, h('summary', {}, 'Runner output'), log),
    h('div', { class: 'actions' },
      p.status === 'running' ? h('button', { type: 'button', onclick: cancelJob }, 'Cancel') : null,
      p.done && p.status !== 'running' ? h('button', { type: 'button', class: 'primary', onclick: () => { location.hash = '#results'; openRun(runName); } }, 'Open results') : null,
      p.status === 'running' ? h('span', { class: 'hint' }, `Writing to runs/${runName}`) : null));
  log.scrollTop = log.scrollHeight;
}

async function cancelJob() {
  try { await api('POST', `/api/experiments/${encodeURIComponent(state.jobId)}/cancel`); } catch { /* shown on next poll */ }
  pollJob();
}

$('#exp-config').addEventListener('change', updateExpForm);
$('#exp-form').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  try {
    const { id } = await api('POST', '/api/experiments', { config: $('#exp-config').value, n_tasks: Number($('#exp-n').value) });
    followJob(id);
  } catch (err) {
    showFixes($('#exp-problems'), { problems: (err.data && err.data.problems) || [err.message], errors: (err.data && err.data.errors) || {} });
  }
  refreshStatus();
});

/* ------------------------------------------------------------------ Results tab */

async function loadResults() {
  let data;
  try { data = await api('GET', '/api/runs'); } catch { return; }
  $('#run-list').replaceChildren(...(data.runs.length ? data.runs.map((r) => h('li', {},
    h('button', { type: 'button', dataset: { run: r.name }, onclick: () => openRun(r.name) },
      h('span', { class: 'name' }, r.name),
      h('span', { class: 'sub' }, r.error ? r.error
        : `Arbitration ${r.pass.arbitration} of ${r.tasks}, ${(r.models || []).join(', ')}, ${fmtDate(r.updated)}`)))) : [h('li', { class: 'hint' }, 'No experiments yet.')]));
  $('#demo-list').replaceChildren(...(data.demos.length ? data.demos.map((d) => h('li', {},
    h('button', { type: 'button', dataset: { demo: d.name }, onclick: () => openDemo(d.name) },
      h('span', { class: 'name' }, d.name.replace('_', '/')),
      h('span', { class: 'sub' }, fmtDate(d.updated))))) : [h('li', { class: 'hint' }, 'No live runs yet.')]));
}

function markCurrent(attr, value) {
  for (const b of $$('#run-list button, #demo-list button')) b.setAttribute('aria-current', b.dataset[attr] === value ? 'true' : 'false');
}

async function openRun(name) {
  const view = $('#result-view');
  let d;
  try { d = await api('GET', `/api/runs/${encodeURIComponent(name)}`); } catch (err) {
    view.replaceChildren(h('ul', { class: 'fixes' }, h('li', {}, err.message)));
    return;
  }
  markCurrent('run', name);
  const s = d.summary;
  const e = s.evidence;
  const replay = h('div', { class: 'feed' });
  const buttons = d.transcripts.map((stem) => h('button', {
    type: 'button',
    onclick: async (ev) => {
      for (const b of $$('button', ev.currentTarget.parentElement)) b.setAttribute('aria-current', 'false');
      ev.currentTarget.setAttribute('aria-current', 'true');
      const tr = await api('GET', `/api/runs/${encodeURIComponent(name)}/transcripts/${encodeURIComponent(stem)}`);
      replay.replaceChildren(...tr.turns.map((t) => turnEntry(t)));
    },
  }, stem.replace('_', '/')));
  view.replaceChildren(
    h('div', { class: 'result-head' }, h('h2', {}, name),
      h('p', { class: 'hint' }, `${plural(s.n_tasks, 'problem')}, ${s.models.map((m) => `${m.model} via ${m.provider}`).join('; ')}`)),
    h('section', { class: 'panel' }, h('h3', {}, 'Strategies on the same candidates'),
      h('div', { class: 'table-wrap' }, h('table', {},
        h('thead', {}, h('tr', {}, h('th', {}, 'Strategy'), h('th', { class: 'num' }, 'Passed'), h('th', { class: 'num' }, 'Pass rate'),
          h('th', {}, '95% interval'), h('th', { class: 'num' }, 'Calls per problem'), h('th', { class: 'num' }, 'Tokens per problem'))),
        h('tbody', {}, STRATEGIES.map(([key, label]) => {
          const r = s.strategies[key];
          return h('tr', { class: key === 'arbitration' ? 'is-arb' : null },
            h('td', {}, label), h('td', { class: 'num' }, `${r.passed} of ${r.n}`), h('td', { class: 'num' }, fmtPct(r.pass_rate)),
            h('td', {}, `${fmtPct(r.ci95[0])} to ${fmtPct(r.ci95[1])}`), h('td', { class: 'num' }, r.calls_per_task.toFixed(1)),
            h('td', { class: 'num' }, fmtInt(Math.round(r.tokens_per_task))));
        })))),
      h('p', { class: 'hint' }, `At least one candidate was correct on ${s.oracle_passed} of ${s.n_tasks} problems: the ceiling for strategies that only pick a candidate.`)),
    h('section', { class: 'panel' }, h('h3', {}, 'Evidence'),
      h('dl', { class: 'stats' },
        [['Problems where candidates disagreed', e.tasks_with_disagreement], ['Problems that needed a ruling', e.tasks_with_disputes],
          ['Disputed cases', e.disputed_cases], ['Rulings matching the reference', `${e.rulings_agreeing_with_reference} of ${e.rulings_checked}`],
          ['Testing-agent expectations correct', `${e.tester_expectations_correct} of ${e.tester_expectations_judged}`],
          ['Repairs kept', `${e.repairs_accepted} of ${e.repairs_attempted}`], ['Unsure rulings', e.rulings_unsure],
          ['Flagged by the security scan', e.unsafe_candidates]]
          .map(([k, v]) => h('div', {}, h('dt', {}, k), h('dd', {}, String(v)))))),
    h('section', { class: 'panel' }, h('h3', {}, 'Replay a problem'),
      buttons.length ? h('div', { class: 'task-buttons' }, buttons) : h('p', { class: 'hint' }, 'This run saved no transcripts.'), replay),
    d.report_md ? h('details', { class: 'panel' }, h('summary', {}, 'Full report (report.md)'), h('pre', { class: 'report' }, d.report_md)) : null);
}

async function openDemo(name) {
  const view = $('#result-view');
  let d;
  try { d = await api('GET', `/api/demos/${encodeURIComponent(name)}`); } catch (err) {
    view.replaceChildren(h('ul', { class: 'fixes' }, h('li', {}, err.message)));
    return;
  }
  markCurrent('demo', name);
  const feed = h('div', { class: 'feed' });
  for (const t of d.transcript.turns) {
    feed.append(turnEntry(t));
    if (t.phase === 'arbiter' && d.outcome.diagnostics.rulings.length) {
      const entry = rulingEntry(d.outcome.diagnostics.rulings);
      markReference(entry, d.outcome.diagnostics.rulings);
      feed.append(entry);
    }
  }
  feed.append(outcomeEntry(d.outcome));
  view.replaceChildren(h('div', { class: 'result-head' }, h('h2', {}, d.outcome.task_id),
    h('p', { class: 'hint' }, 'A live run, replayed from runs/demos.')), feed);
}

/* ------------------------------------------------------------------ start */

async function init() {
  buildConfigForm();
  window.addEventListener('hashchange', showTab);
  await refreshStatus();
  try {
    state.tasks = await api('GET', '/api/tasks');
    $('#task-list').replaceChildren(...state.tasks.map((t) => h('option', { value: t.task_id, label: `${t.entry_point}: ${t.summary}` })));
  } catch { /* the offline banner explains */ }
  if (!$('#run-task').value) $('#run-task').value = 'HumanEval/2';
  try { await loadConfigs(); } catch { /* offline */ }
  updateTaskSummary();
  checkRunConfig();
  updateExpForm();
  showTab();
  setInterval(refreshStatus, 4000);
}

document.addEventListener('DOMContentLoaded', init);  // after chat.js has loaded too
