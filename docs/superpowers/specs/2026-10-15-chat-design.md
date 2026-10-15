# HiveMind chat: design

Status: approved in conversation 2026-10-15. Owner: Harigovind.

## Goal

A Chat tab in the web interface where you give the team your own coding problem
in plain English and talk with it about the answer. The team solves it with the
same evidence-based pipeline as a live run. Follow-up messages either ask about
the answer or change the problem, and the chat works out which.

## Decisions

- **Input:** plain English. The Research agent drafts a Python spec (signature,
  docstring, `>>>` examples); you see it, may edit it, and press Go. Nothing
  runs on a spec you have not seen.
- **Follow-ups:** after a round has an answer, the Coordinator reads each new
  message as a QUESTION (answered from that round's evidence) or a CHANGE (the
  Research agent drafts an updated spec for a new round). Every reply says how
  the message was read and offers a one-click switch to the other reading.
- **Rounds skip the research baselines:** no solo agent, no debate, no debate
  verdict. A round is brief → candidates → tests → sandbox → rulings → repair.
  These baselines exist to compare strategies in experiments; skipping them
  roughly halves a round's time on a local model.
- **No grade:** your problem has no hidden tests, so the answer reports the
  verified cases it passes (your examples plus Coordinator rulings) instead of
  pass/fail.
- **Placement:** a new Chat tab, first in the bar and the default tab.
- **Jobs:** a team round is a job under the one-job-at-a-time rule. Drafting a
  spec, classifying a message and answering a question are not; they run during
  an experiment (slower, as they share the model).
- **Stack:** unchanged. Standard library, SSE, plain HTML/CSS/JS, no new
  dependencies.

## Screens

**Chat tab.** Left: saved chats (newest first), a New chat button and a config
picker. Right: the thread, with a composer at the bottom (Enter sends,
Shift+Enter is a newline; disabled while the team is working).

Thread entries:

| Entry | Content |
|---|---|
| Your message | Right-aligned bubble |
| Pending agent | "Research agent is writing…" style bubble while a call runs |
| Spec card | Drafted spec in an editable code box; Go; a warning when it has no `>>>` examples; the parser's error (with line) when an edit is not valid |
| Team round | Agent bubbles colour-coded by role (brief, candidates, tests, sandbox evidence, rulings, repair) streaming in live, folded into "Show how the team got there" once the round ends |
| Answer card | Final code, its source (candidate n or repair), "Passes 7 of 7 verified cases: 3 from your examples, 4 ruled on by the Coordinator", the rulings |
| Coordinator answer | Reply to a question, headed "Read as a question · treat as a change instead" |
| Error | Red bubble with the reason and Retry |

Status strip: a running team round shows as "Chat round running".

## Architecture

Changes to the existing core (backward compatible):

- `coordinator.py`: the shared middle of `solve()` (brief, candidates, tests,
  sandbox, arbitration, repair) moves into a private method. `solve()` calls the
  solo baseline, then that method, then debate and grading, so its calls, their
  order, seeds, transcript and outcome are unchanged. New `answer(task,
  on_event=None, cancel=None) -> (ChatAnswer, Transcript)` runs only the middle.
  `ChatAnswer`: `code`, `source`, `candidates`, `unsafe`, `verified_total`,
  `verified_passed`, `spec_cases`, `ruled_cases`, `rulings`,
  `distinct_behaviours`, `repaired`. No grading and no reference diagnostics.
- `web/live.py`: the event history (append with `seq`, `events_after`,
  `wait_events`) moves into a small `EventLog` class; `LiveRun` uses it with no
  change in behaviour.

New modules:

| Module | Job |
|---|---|
| `hivemind/chat.py` | Prompts and parsing for the chat-only calls; spec validation; building a `CodeTask` from a spec. No web code |
| `hivemind/web/chats.py` | A chat thread: messages, persistence to `runs/chats/<id>.json`, one background action at a time, events through an `EventLog` |

`hivemind/chat.py`:

- `draft_spec(team, message, current_spec=None) -> Draft`: the Research agent
  writes a spec in one ```python block. `validate_spec(text)` parses it with
  `ast`: optional imports, then exactly one function whose body is only its
  docstring; its name is the entry point. If invalid, one retry with the error;
  if still invalid, the raw reply goes into the spec card with the error for
  hand-editing.
- `classify(team, message, round) -> "question" | "change"`: the Coordinator
  answers with one word. An unclear reply counts as a question. `round` is
  always the latest round that has an answer (its spec, `ChatAnswer` and the
  questions asked about it so far).
- `answer_question(team, message, round) -> str`: the Coordinator answers from
  the round's spec, final code, evidence summary, rulings and the earlier
  questions in that round.
- Reinterpret: question → change drafts an updated spec from the message;
  change → question answers it and marks that message's unsent spec card as
  superseded. A spec that already ran cannot be reinterpreted.
- `custom_task(spec, task_id) -> CodeTask`: `task_id` is
  `chat/<chat id>/<round>`; hidden tests and reference solution are empty.

Chat model calls go through `Agent.ask` with phases `draft`, `classify` and
`question`, so they stream as `turn_started`/`turn` events like any other turn
and are saved with the thread.

A chat uses its config's team and Coordinator models (including per-role
models), `candidates`, `repair` and `seed`. `debate_rounds`, `n_tasks`,
`sample_seed` and `workers` do not apply.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/chats` | Saved chats: id, title (first message, cut to 60 characters), config, updated |
| `POST /api/chats` `{config}` → `{id}` | New chat |
| `GET /api/chats/<id>` | The whole thread |
| `GET /api/chats/<id>/events` | SSE stream; replays history on connect, then live events |
| `POST /api/chats/<id>/messages` `{text}` | Draft a spec, or classify then answer/draft |
| `POST /api/chats/<id>/go` `{spec}` | Validate the spec and start a team round |
| `POST /api/chats/<id>/reinterpret` `{message}` | Redo a classified message with the other reading |
| `POST /api/chats/<id>/cancel` | Stop at the next model-call boundary |

New event types: `message` (your text), `classified`, `spec`, `answer`,
`question_answer`, plus the existing `turn_started`, `turn`, `evidence`,
`rulings`, `error`, `done`.

`runs/chats/` is added to `.gitignore`, like `runs/demos/`.

## Errors

- Ollama down, model not pulled, or Claude without a key: the readiness check
  runs before each action; the chat shows the fix in a red bubble with Retry.
- Go while another job runs: HTTP 409; the chat says the round waits for the
  experiment or live run to finish or be cancelled.
- A second action in the same chat while one runs: HTTP 409; the composer is
  disabled meanwhile.
- Invalid spec on Go: HTTP 400 with the parser's message and line; nothing runs.
- No candidate produces working code: the answer card says so and shows the best
  attempt and its failures.
- Cancel: stops before the next model call; the round is marked cancelled.
- Reload mid-round: the stream replays from history. Server restart mid-round:
  the round is marked interrupted, with Retry.
- Chat ids are generated by the server and checked with the existing safe-name
  rule, so no request reaches files outside `runs/chats/`.
- Your problems run in the existing sandbox and static security scan.

## Out of scope

Renaming or deleting chats from the page; problems needing more than one
function, classes, or a language other than Python; the Security,
Documentation and Ethics reviews in chat rounds.

## Testing

Test-first, offline, with `ScriptedLLM`:

- spec validation (good spec, prose around the block, a body, two functions,
  no docstring, syntax error) and the one retry;
- classify parsing, including an unclear reply;
- `answer()` choosing the candidate the rulings support, repair, and no working
  candidate;
- `solve()` unchanged: same phases in the same order and the same outcome on a
  scripted task;
- `EventLog` and `LiveRun` behaviour unchanged (existing tests);
- chat persistence round-trip and the interrupted state after a restart;
- every new endpoint on a real server on a spare port, including SSE, the
  one-job rule (409), an invalid spec (400) and reinterpret.

Then a headless-browser run on the local model: type a problem, check the draft,
Go, watch the round, ask a question, make a change; no console errors. README
gains a Chat section.
