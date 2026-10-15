'use strict';

/* The Chat tab: give the team your own problem in plain English and talk about the answer.
   A chat is its event list. The page loads the whole thread once, replays it, then follows
   /api/chats/<id>/events for new events. Uses the helpers and entry builders in app.js. */

const CHAT_PHASES = new Set(['draft', 'classify', 'question']);
const CHAT_BUSY_TEXT = {
  message: 'The team is reading your message…',
  reinterpret: 'The team is reading your message again…',
  round: 'The team is working on the problem. Each step appears below as it happens.',
};
const CHAT_IDLE_TEXT = 'Enter sends, Shift+Enter starts a new line.';

const chat = {
  id: null, es: null, lastSeq: 0, busy: null, currentRound: null, started: false,
  pending: new Map(), // phase -> pending bubble
  specs: new Map(), // spec id -> spec card
  notes: new Map(), // message id -> "read as" note
  rounds: new Map(), // round number -> { box, steps }
  ran: new Set(), // spec ids a round has used
};

// ChatAnswer counts candidates from 0; the round's bubbles (and people) count from 1.
const displaySource = (source) => source.replace(/candidate (\d+)/, (_, n) => `candidate ${Number(n) + 1}`);

const chatPath = (suffix = '') => `/api/chats/${encodeURIComponent(chat.id)}${suffix}`;

function setChatState(text) { $('#chat-state').textContent = text || CHAT_IDLE_TEXT; }

function showChatProblems(items) { showFixes($('#chat-problems'), { problems: items }); }

function problemsOf(err) {
  if (err.status === 409) return [`The team can't start yet: ${err.message}.`];
  return (err.data && err.data.problems) || [err.message];
}

/* ------------------------------------------------------------------ list */

async function loadChatList() {
  let chats;
  try { chats = await api('GET', '/api/chats'); } catch { return; }
  $('#chat-list').replaceChildren(...(chats.length ? chats.map((c) => h('li', {},
    h('button', { type: 'button', dataset: { chat: c.id }, onclick: () => openChat(c.id) },
      h('span', { class: 'name' }, c.title || 'New chat'),
      h('span', { class: 'sub' }, `${c.config}, ${fmtDate(c.updated)}${c.busy ? ', working' : ''}`))))
    : [h('li', { class: 'hint' }, 'No chats yet.')]));
  markChatList();
}

function markChatList() {
  for (const b of $$('#chat-list button')) b.setAttribute('aria-current', b.dataset.chat === chat.id ? 'true' : 'false');
}

async function newChat() {
  showChatProblems([]);
  try {
    const { id } = await api('POST', '/api/chats', { config: $('#chat-config').value });
    await loadChatList();
    await openChat(id);
    return id;
  } catch (err) {
    showChatProblems(problemsOf(err));
    return null;
  }
}

async function showChatTab() {
  if (chat.started) return;
  chat.started = true;
  await loadChatList();
  let last = null;
  try { last = localStorage.getItem('hivemind.chat'); } catch { /* storage blocked */ }
  if (last && $(`#chat-list button[data-chat="${CSS.escape(last)}"]`)) openChat(last);
  else updateComposer();
}

/* ------------------------------------------------------------------ thread */

async function openChat(id) {
  if (chat.es) chat.es.close();
  Object.assign(chat, { id, es: null, lastSeq: 0, busy: null, currentRound: null });
  for (const map of [chat.pending, chat.specs, chat.notes, chat.rounds]) map.clear();
  chat.ran.clear();
  try { localStorage.setItem('hivemind.chat', id); } catch { /* storage blocked */ }
  showChatProblems([]);
  setChatState('');
  markChatList();
  const thread = $('#chat-thread');
  thread.replaceChildren();
  let whole;
  try { whole = await api('GET', chatPath()); } catch (err) {
    thread.replaceChildren(h('ul', { class: 'fixes' }, h('li', {}, err.message)));
    return;
  }
  if (chat.id !== id) return;
  thread.append(h('header', { class: 'chat-head' }, h('h2', {}, whole.title || 'New chat'),
    h('p', { class: 'hint' }, `Config ${whole.config}. Describe a Python function in plain English; once the team has answered, ask about it or ask for a change.`)));
  for (const e of whole.events) handleChatEvent(e, false);
  if (chat.busy) setChatState(CHAT_BUSY_TEXT[chat.busy]);
  window.scrollTo({ top: document.documentElement.scrollHeight });
  chat.es = new EventSource(chatPath(`/events?after=${chat.lastSeq}`));
  chat.es.onmessage = (msg) => handleChatEvent(JSON.parse(msg.data), true);
  updateComposer();
  $('#chat-input').focus();
}

function place(el, parent, live) {
  const target = parent || $('#chat-thread');
  const nearBottom = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
  target.append(el);
  if (live && nearBottom) el.scrollIntoView({ block: 'end', behavior: reducedMotion ? 'auto' : 'smooth' });
}

function roundSteps(phase) {
  const round = chat.rounds.get(chat.currentRound);
  return round && !CHAT_PHASES.has(phase) ? round.steps : null;
}

function handleChatEvent(e, live) {
  if (e.seq <= chat.lastSeq) return;
  chat.lastSeq = e.seq;
  const add = (el, parent) => place(el, parent, live);
  switch (e.type) {
    case 'message':
      add(h('div', { class: 'bubble user', dataset: { message: e.id } }, h('p', { class: 'prose' }, e.text)));
      break;
    case 'action':
      chat.busy = e.action;
      for (const b of $$('.retry button', $('#chat-thread'))) b.disabled = true;
      if (live) setChatState(CHAT_BUSY_TEXT[e.action] || 'Working…');
      break;
    case 'classified':
      showReading(e, add);
      break;
    case 'turn_started': {
      const el = turnEntry(e, true);
      chat.pending.set(e.phase, el);
      add(el, roundSteps(e.phase));
      break;
    }
    case 'turn': {
      const pending = chat.pending.get(e.phase);
      chat.pending.delete(e.phase);
      if (CHAT_PHASES.has(e.phase) && !e.error) { // shown by their own spec, note or answer
        if (pending) pending.remove();
        break;
      }
      const el = turnEntry(e);
      if (pending) pending.replaceWith(el); else add(el, roundSteps(e.phase));
      break;
    }
    case 'spec':
      add(specCard(e));
      break;
    case 'spec_superseded':
      setSpecState(e.spec_id, 'superseded');
      break;
    case 'round_started': {
      chat.ran.add(e.spec_id);
      setSpecState(e.spec_id, 'sent', e.spec);
      chat.currentRound = e.round;
      const steps = h('div', { class: 'feed' });
      const box = h('details', { class: 'round', open: true },
        h('summary', {}, `Round ${e.round}: how the team got there`), steps);
      chat.rounds.set(e.round, { box, steps });
      add(box);
      break;
    }
    case 'evidence':
      add(evidenceEntry(e), roundSteps('evidence'));
      break;
    case 'rulings':
      add(e.disputes ? rulingEntry(e.rulings)
        : systemEntry('Arbitration', 'No disputed cases: every program that ran agreed with the verified expectations, so there was nothing to rule on.'),
      roundSteps('rulings'));
      break;
    case 'answer': {
      const round = chat.rounds.get(e.round);
      if (round) round.box.open = false;
      chat.currentRound = null;
      add(answerCard(e.round, e.answer));
      break;
    }
    case 'question_answer':
      add(h('article', { class: 'entry' }, entryHead(who('coordinator'), 'Answer', `about round ${e.round}`),
        renderReply(e.text)));
      break;
    case 'error':
      add(systemEntry('Something went wrong', e.message, true));
      break;
    case 'done': {
      chat.busy = null;
      chat.currentRound = null;
      for (const el of chat.pending.values()) el.remove();
      chat.pending.clear();
      if (e.status !== 'done') add(retryNote(e));
      if (live) { setChatState(''); loadChatList(); }
      break;
    }
    default:
      break;
  }
  updateComposer();
}

function showReading(e, add) {
  const other = e.reading === 'question' ? 'change' : 'question';
  const ran = [...chat.specs.values()].some((c) => c.dataset.message === e.message_id && chat.ran.has(c.dataset.spec));
  const button = h('button', { type: 'button', class: 'link', dataset: { locked: String(ran) },
    onclick: () => reinterpret(e.message_id) }, `treat as a ${other} instead`);
  const note = h('p', { class: 'reading hint' }, e.automatic ? `Read as a ${e.reading}` : `You chose: ${e.reading}`, ' · ', button);
  const old = chat.notes.get(e.message_id);
  chat.notes.set(e.message_id, note);
  if (old) old.replaceWith(note); else add(note);
}

function specCard(e) {
  const box = h('textarea', { class: 'code-edit', spellcheck: 'false', 'aria-label': 'Spec for the team',
    rows: String(Math.min(24, e.text.split('\n').length + 1)) });
  box.value = e.text;
  const problems = h('ul', { class: 'fixes', hidden: !e.error },
    e.error ? h('li', {}, `${e.error}. Fix the spec by hand, then press Go.`) : null);
  const go = h('button', { type: 'button', class: 'primary go', dataset: { locked: 'false' } }, 'Go');
  const card = h('article', { class: 'entry spec-card', dataset: { spec: e.id, message: e.message_id, state: 'draft' } },
    entryHead(who('research'), 'Spec for the team', e.entry_point ? `${e.entry_point}, ${plural(e.examples, 'example')}` : null),
    h('p', { class: 'note' }, 'This is what the team will solve. Edit anything, then press Go.'),
    box,
    !e.error && e.examples === 0
      ? h('p', { class: 'note' }, "This spec has no >>> examples, so the team will rely only on its own tests and the Coordinator's rulings.")
      : null,
    problems,
    h('div', { class: 'actions' }, go, h('span', { class: 'hint state' })));
  go.addEventListener('click', () => startRound(e.id, box, problems));
  chat.specs.set(e.id, card);
  return card;
}

function setSpecState(id, specState, text) {
  const card = chat.specs.get(id);
  if (!card) return;
  card.dataset.state = specState;
  const box = $('textarea', card);
  if (text != null) box.value = text;
  box.readOnly = true;
  $('.go', card).dataset.locked = 'true';
  $('.state', card).textContent = specState === 'sent' ? 'Sent to the team.' : 'Set aside: you asked a question instead.';
  $('.note', card).textContent = specState === 'sent' ? 'The team solved this spec.' : 'This draft was set aside.';
  const note = chat.notes.get(card.dataset.message);
  if (note && specState === 'sent') $('button', note).dataset.locked = 'true';
}

function verifiedLine(a) {
  if (!a.verified_total) {
    return 'No verified cases: the spec had no usable examples and the Testing agent wrote no usable tests, so nothing checked this answer.';
  }
  const parts = [];
  if (a.spec_cases) parts.push(`${a.spec_cases} from your examples`);
  if (a.ruled_cases) parts.push(`${a.ruled_cases} ruled on by the Coordinator`);
  if (a.undisputed_cases) parts.push(`${a.undisputed_cases} no one disputed`);
  let head;
  if (a.verified_passed !== a.verified_total) head = `Passes ${a.verified_passed} of ${plural(a.verified_total, 'verified case')}`;
  else head = a.verified_total === 1 ? 'Passes its 1 verified case' : `Passes all ${a.verified_total} verified cases`;
  return `${head}: ${parts.join(', ')}.`;
}

function answerCard(round, a) {
  const flagged = a.unsafe.filter(Boolean).length;
  return h('article', { class: `entry answer-card${a.working ? '' : ' failed'}` },
    entryHead(who('coordinator', 'Team answer'), `Round ${round}`,
      a.repaired ? `${displaySource(a.source)}, kept because it passes more verified cases` : displaySource(a.source)),
    a.working ? null : h('p', { class: 'note' }, 'No candidate produced working code. This is the best attempt, and the failures below show why. Send a message to change the problem.'),
    h('p', { class: 'verified' }, verifiedLine(a)),
    h('pre', { class: 'code' }, a.code || '(no code)'),
    a.failures.length ? h('div', { class: 'failures' }, h('h4', {}, 'Verified cases it still fails'),
      h('ul', {}, a.failures.map((f) => h('li', {}, h('code', {}, f.call), ` should give ${f.expected}, gives ${f.got}`)))) : null,
    a.rulings.length ? h('div', { class: 'rulings' }, h('h4', {}, "The Coordinator's rulings"),
      h('ul', {}, a.rulings.map((r) => h('li', {}, h('code', {}, r.call),
        r.choice == null ? ': unsure, so the case was not used' : ` should give ${r.choice}`)))) : null,
    flagged ? h('p', { class: 'note' }, `${plural(flagged, 'candidate')} flagged by the security scan.`) : null);
}

function retryNote(e) {
  const text = { failed: 'This step stopped with an error.', cancelled: 'Cancelled.',
    interrupted: 'The server stopped before the team finished.' }[e.status] || e.status;
  return h('div', { class: 'retry actions' }, h('span', { class: 'hint' }, text),
    h('button', { type: 'button', onclick: retryChat }, 'Retry'));
}

function updateComposer() {
  const busy = !!chat.busy;
  $('#chat-send').disabled = busy;
  $('#chat-cancel').hidden = !busy;
  for (const b of $$('.reading button, .spec-card .go', $('#chat-thread'))) b.disabled = busy || b.dataset.locked === 'true';
}

/* ------------------------------------------------------------------ actions */

async function sendChat(ev) {
  ev.preventDefault();
  const input = $('#chat-input');
  const text = input.value.trim();
  if (!text || chat.busy) return;
  if (!chat.id && !(await newChat())) return;
  showChatProblems([]);
  try {
    await api('POST', chatPath('/messages'), { text });
    input.value = '';
  } catch (err) {
    showChatProblems(problemsOf(err));
  }
}

async function startRound(specId, box, problems) {
  problems.hidden = true;
  try {
    await api('POST', chatPath('/go'), { spec: box.value, spec_id: specId });
  } catch (err) {
    problems.replaceChildren(...problemsOf(err).map((t) => h('li', {}, t)));
    problems.hidden = false;
    if (err.data && err.data.line) focusLine(box, err.data.line);
  }
}

function focusLine(box, line) {
  const lines = box.value.split('\n');
  const start = lines.slice(0, line - 1).reduce((n, l) => n + l.length + 1, 0);
  box.focus();
  box.setSelectionRange(start, start + (lines[line - 1] || '').length);
}

async function reinterpret(messageId) {
  showChatProblems([]);
  try { await api('POST', chatPath('/reinterpret'), { message_id: messageId }); } catch (err) { showChatProblems(problemsOf(err)); }
}

async function retryChat(ev) {
  const button = ev.currentTarget;
  button.disabled = true;
  showChatProblems([]);
  try {
    await api('POST', chatPath('/retry'));
  } catch (err) {
    button.disabled = false; // refused (busy, not ready): the person can try again later
    showChatProblems(problemsOf(err));
  }
}

$('#chat-form').addEventListener('submit', sendChat);
$('#chat-input').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && !ev.shiftKey && !ev.isComposing) {
    ev.preventDefault();
    $('#chat-form').requestSubmit();
  }
});
$('#chat-new').addEventListener('click', newChat);
$('#chat-cancel').addEventListener('click', async () => {
  setChatState('Cancelling once the current model call finishes.');
  try { await api('POST', chatPath('/cancel')); } catch (err) { setChatState(err.message); }
});
