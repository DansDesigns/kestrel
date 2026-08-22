# Kestrel

An agentic harness for llama.cpp that fits the context window you actually have,
runs a single model across several machines, and keeps what it learns between
sessions.

---

## Contents

1. [Rationale](#1-rationale)
2. [Design principles](#2-design-principles)
3. [The context budget](#3-the-context-budget)
4. [Tool-calling dialects](#4-tool-calling-dialects)
5. [Working memory: the task checklist](#5-working-memory-the-task-checklist)
6. [Long-term memory](#6-long-term-memory)
7. [Personality](#7-personality)
8. [Reasoning models](#8-reasoning-models)
9. [Skills](#9-skills)
10. [Model management](#10-model-management)
11. [Backend acquisition](#11-backend-acquisition)
12. [Distributed inference](#12-distributed-inference)
13. [Speech](#13-speech)
14. [The interface](#14-the-interface)
15. [Headless operation](#15-headless-operation)
16. [Installation](#16-installation)
17. [Configuration reference](#17-configuration-reference)
18. [Project layout](#18-project-layout)
19. [Verification status and known limitations](#19-verification-status-and-known-limitations)

---

## 1. Rationale

Most agentic harnesses assume a large context window. They typically require
something in the region of 64,000 tokens before they will run at all, and that
requirement is not a consequence of the task being hard. It is a consequence of
how the prompt is assembled.

A conventional harness sends the model, on every single turn:

- the complete tool registry, rendered as JSON Schema;
- a personality or identity file, in full;
- every installed skill definition;
- the entire conversation transcript, unpruned;
- and the complete output of every tool call made so far.

None of this is individually unreasonable. Together they establish a floor that
rules out most consumer hardware. A 7B model quantised to fit in 8 GB of VRAM,
loaded with an 8,192-token window, cannot run such a harness — not because it
lacks the capability, but because the scaffolding does not fit.

Kestrel starts from the opposite assumption. The context window is a fixed
resource to be allocated deliberately, and every component of the prompt must
justify its share. The practical floor is approximately 2,048 tokens. The harness
is comfortable from 4,096 upward.

A second concern follows from the first. If the window is small, the transcript
will be compacted, and anything the agent needs to remember must live somewhere
other than the conversation. Kestrel therefore treats durable state — the task
checklist and the long-term memory store — as first-class components with their
own budget allocations, rather than as text that happens to be in the history.

Kestrel also removes the dependency on a separate model-management application.
It discovers, inspects, downloads and loads GGUF models itself, and exposes the
llama.cpp backend directly rather than through an abstraction layer.

---

## 2. Design principles

Five principles govern the implementation. They are stated here because they
explain most of the decisions documented in the sections that follow.

**Everything is budgeted.** No component of the prompt is sent at full fidelity
unless it fits. Tool descriptions, skill listings, persona text, recalled
memories and the transcript each have an allowance, and each is rendered at
whatever verbosity that allowance permits.

**Bulk data goes to disk, and context gets a reference.** Tool output larger than
the preview allowance is written to a file. The model receives a head-and-tail
excerpt plus a path, and pages through the remainder with the ordinary file tools
if it needs to. A 40,000-line log therefore costs a fixed and small number of
tokens.

**Generation headroom is reserved before anything else is allocated.** The classic
small-context failure is a tool call truncated mid-JSON because the prompt
consumed the whole window. The output reserve is subtracted first; input
components divide what remains.

**State that must survive compaction does not live in the transcript.** The
checklist and the memory store are re-rendered into every prompt from their own
persistent sources. Summarising the transcript therefore does not lose them.

**Degradation is graceful and visible.** A missing tokeniser endpoint falls back
to a self-correcting character estimate. A missing FTS5 extension falls back to
substring search. A broken memory database disables memory and reports it rather
than terminating the session. Where the harness makes a compromise, it says so.

---

## 3. The context budget

### 3.1 Profiles

On connection, Kestrel queries the server for the context length the model was
**loaded** with — not the length the GGUF metadata claims it supports, which is
frequently much larger and not what governs behaviour. That figure selects a
profile:

| Profile | Window | Tool listing | Skills named | Tool-output preview |
|---|---|---|---|---|
| `nano` | ≤ 4,096 | signatures only | 8 | 180 tokens |
| `small` | ≤ 16,384 | signatures and parameter notes | 20 | 500 tokens |
| `standard` | ≤ 65,536 | full descriptions | 48 | 1,200 tokens |
| `large` | > 65,536 | full, with extended guidance | 128 | 3,000 tokens |

The profile can be overridden. Forcing `nano` on a large window is a useful way
to confirm that a workflow survives a tight budget before depending on it.

### 3.2 Allocation

The window is divided into five allocations. Measured values:

| Window | System | Memory | Checklist | Transcript | Output reserve |
|---|---|---|---|---|---|
| 2,048 | 473 | 129 | 84 | 994 | 368 |
| 4,096 | 947 | 258 | 167 | 1,987 | 737 |
| 8,192 | 1,580 | 553 | 320 | 4,265 | 1,474 |
| 32,768 | 5,496 | 2,442 | 320 | 18,612 | 5,898 |

The checklist allocation is deliberately small and capped, because a plan is
short regardless of how much room is available. It has a floor as well as a
ceiling: it must fit at `nano`, since it is precisely the state that compaction
would otherwise destroy.

### 3.3 What the system prompt actually costs

The allowances above are ceilings. Actual measured consumption at 4,096 tokens,
with ten tools and two skills installed:

| Configuration | Cost |
|---|---|
| Text dialect, `nano` verbosity | **295 tokens** |
| Native dialect, same tools | 456 tokens, plus 859 tokens of JSON Schema |

The comparison is the reason the dialect selection in §4 exists. At small window
sizes, JSON Schema is not affordable.

### 3.4 Output spooling

Tool results are compared against the preview allowance for the profile. Results
exceeding it are written to `.kestrel/artifacts/` and replaced in context by an
excerpt and a pointer.

Measured: a shell command producing 23,893 characters of output consumed **233
tokens** of context. The complete output remained available on disk and in the
interface.

The excerpt retains both head and tail by default, because errors and stack
traces appear at the end of output and a head-only truncation would routinely
discard the informative part.

### 3.5 Compaction

When the transcript exceeds its allowance, the oldest turns are removed and
folded into a rolling digest, injected as a short system note. The digest is
produced by the model where the window can afford the generation, and by an
extractive fallback where it cannot — the fallback never fails and never requires
a model call, which matters when every generation is expensive.

Turn boundaries are respected: a tool result is never separated from the
assistant message that requested it.

Measured behaviour in a 2,048-token window running a twelve-step tool-using task:
peak input 1,516 tokens across 8 compactions, with no overflow at any point.

---

## 4. Tool-calling dialects

Kestrel speaks two protocols and selects between them automatically.

**Native.** The model emits OpenAI-style `tool_calls`. This requires a chat
template containing a tools branch, and costs the full JSON Schema of every tool
in the prompt.

**Text.** The model emits a delimited block:

```
<tool>
{"name": "read_file", "arguments": {"path": "notes.md"}}
</tool>
```

This costs roughly one-fifth of the prompt tokens that schemas require, works
with any instruction-tuned model regardless of its template, and permits a stop
sequence on the closing tag so generation terminates the moment the call is
complete rather than continuing to the token limit.

Under `auto`, Kestrel selects the text dialect below 16,384 tokens, or whenever
the server rejects a `tools` payload. The capability probe runs once at
connection, so a template without tool support degrades cleanly at startup
instead of failing mid-task.

The text parser is deliberately tolerant. Small models routinely wrap the block
in a code fence, omit the closing tag, leave a trailing comma, or emit
`parameters` where the specification says `arguments`. All of these are accepted.
Recovering a malformed call is considerably cheaper than spending a step
rejecting it.

---

## 5. Working memory: the task checklist

The agent maintains a plan, and that plan is re-rendered into every prompt.

```
Plan (1/5 done) — add retry field:
1. [x] Read the config parser
2. [>] Add the new field   <- you are here
3. [ ] Write a test
4. [ ] Run the suite
5. [ ] Update the docs
```

That block costs 47 tokens.

**Why it appears in every prompt.** The transcript is the first thing compaction
discards. The checklist is not part of the transcript; it is rendered fresh each
turn from a persistent source. It is therefore the one element of task state
guaranteed to be present, however long the task runs and however aggressively
history has been summarised.

**Seeding.** For any request substantial enough to warrant it, Kestrel asks the
model to decompose the task before work begins, so the checklist reflects the
model's own reading of the request. Single-action requests, and windows too small
to spare the generation, are skipped; the tools remain available in both cases.

**Maintenance.** The model updates the plan with `plan`, `todo` and `plan_add`.
Steps carry four states — pending, in progress, done, blocked — and a blocked step
may carry a short note recording why.

**The checklist decides when the turn is over.** A model that answers the first
step and stops — "I have read the spec, let me know if you want me to continue" —
has not finished the task, it has narrated one step of it. While steps remain
open, a prose reply is treated as a status update: the plan state is put back to
the model and the work carries on. `finish` is refused twice while steps are
still open, with the count of what remains, so ending the turn is a decision
rather than a slip.

Three guards keep that from becoming a monologue: three prose replies in a row
end it, a genuine question ends it (one that actually ends in a question mark and
asks for a decision, not "let me know if you want me to continue"), and the step
limit ends it regardless. Every tool result carries a one-line plan status, so
the model is never more than one message from knowing where it is.

**Reasoning is separated as it streams.** A model may put its thinking in a
field of its own, or inline in the content as `<think>…</think>`. The second is
the awkward one: a tag can arrive split across chunks — `<thi` then `nk>` — and
matching each chunk alone never sees it, so markup lands in the reply while
pieces of the reply are filed as thinking. Text is therefore held back until it
cannot be the start of a tag. Verified with the response delivered one character
at a time, and at every chunk size in between.

**A refused tool call says how to call it.** `todo needs a 'id' argument. Call it
as todo(id, status="done", note?). You passed: nothing.` A model told only what
is missing has to guess the rest, and guesses the same way again next time.

**Drift correction.** Models reliably begin a task by maintaining the checklist
and then quietly stop. After three steps without a change, a single-line reminder
is appended to the next tool result. It costs about a dozen tokens and arrives at
the point of action, which is where it is most likely to be acted upon.

**Overflow.** When a plan grows beyond its allowance, the rendering keeps the
current step and its immediate neighbours and summarises the remainder by count.
A twelve-step plan renders in 76 tokens.

**Steps are editable in place.** Double-click one, or select it and press F2. A
plan is a working document, and correcting a word in it should not mean retyping
it through a dialog.

**A plan you wrote is instructions, not a suggestion.** The model is told
explicitly that an existing checklist may have been written by hand, to work
through it in order rather than replace it, and to mark steps done only with
evidence. Each turn opens with a line naming the step to start from. Verified:
a three-step plan typed in by hand survived a turn intact, was followed in
order, and the model's own alternative plan was never substituted.

Completed steps carry a green tick, and a step longer than the panel wraps onto
as many lines as it needs rather than being cut off.

Expanding anything — a trace or a picture — holds the view where it was. The
document changes height when it is rebuilt, so the position is restored
proportionally rather than by pixel offset; finishing at the end instead reads as
the window jumping away, with the thing you expanded now off screen, which looks
exactly like the click having done nothing. Navigation is refused outright too:
QTextBrowser treats a clicked link as a document to load, and ours are commands
rather than destinations.

**Thinking blocks collapse.** Expanding one redraws the transcript from its own
record — and a reply still streaming is not in that record, because it has not
finished. Rather than trying to carry the live text across a rebuild, the
rebuild waits: clicking while the model is writing is remembered and applied the
moment the turn ends. Nothing touches the document while tokens are arriving,
which also protects a palette change made mid-reply.

 A trace is the most interesting thing in the
transcript when something has gone wrong and the least interesting when it has
not, so it shows its opening with the whole of it one click away.

The composer is one line to start with and grows with what is typed, to a
maximum of eight. A box that opens four lines tall implies a paragraph is
wanted, when most messages are a sentence; past eight it is a document, and
taking the window for it would push away the conversation being replied to.

**Sub-steps write themselves.** The two-level plan is only useful if something
fills the second level, and a small model will not reliably do that while also
doing the work. Kestrel knows exactly what happened, so it records it: every
tool that runs is logged as a sub-step of the stage in flight.

```
1. [x] Check the existing files
   1a. [x] list_dir .
   1b. [x] read_file readme.txt
2. [x] Create main.py
   2a. [x] write_file main.py
```

`plan_add` follows the same rule: a note about what is being done attaches to
the stage in flight unless `new_stage` is set. Without that default every
observation becomes another stage, and a plan of five becomes a list of
twenty-one — which is a log, not a plan.

These are a record of what happened, not a list of what must happen, so they do
not decide when a stage is finished — otherwise a stage would close the moment
its first tool ran. Progress counts planned work only.

**Stages and sub-steps.** A plan has two levels. A stage is a piece of work with
an end — looking at what already exists is one stage however many files are
opened; writing each file is a stage of its own. What happens inside a stage is
recorded as sub-steps, labelled 1a, 1b and so on:

```
1. [x] Check the existing file structure
   1a. [x] Found 4 files, checking their state
   1b. [x] Confirmed there is no main.py yet
2. [>] Create main.py
   2a. [>] Write the game loop
3. [ ] Create player.py
```

This gives a small model somewhere to record progress without inventing a
top-level step for every action, and closing all of a stage's sub-steps closes
the stage. Progress counts the work, not the headings.

**A plan is never destroyed by a call that produces nothing.** Auto-planning
streams the checklist in line by line, then parses the finished reply to confirm
it — and a reasoning model can put the entire plan in its trace and return an
empty answer, so that confirming parse finds nothing. The plan built while
streaming is the better record and is never discarded in favour of a worse one.
`set_plan` likewise ignores an empty or garbled set of steps rather than
emptying the list first and asking questions afterwards. Between them these were
what made a plan appear, vanish, and leave the model with nothing to work
through.

**A plan under way is not replaced.** Calling `plan` again with steps already
closed is refused, with the current plan returned and a pointer to `plan_add`
and `todo` instead — models rewrite the checklist while reasoning, and every
closed step goes with it. `replace=true` is there for when the task genuinely
changed.

**Setting a step by hand.** *Working*, *Done* and *To do* move the selected step
between states. The model sets these as it goes; setting them yourself matters
when it gets one wrong, or when the work happened outside Kestrel. Each change
rewrites PLAN.md.

**The canvas** has two tabs. **Model** is what the model writes to; **Your
files** is what you load into it — text, code, Word, Excel, PowerPoint, PDF,
OpenDocument or an image. They are separate because they have different owners:
an import cannot destroy what the model is midway through writing, and the model
cannot overwrite the file you loaded for it to look at. It reads that one with
`canvas_read(source="user")`.

Office formats are read without needing a library installed — a `.docx` is a zip
of XML, and stripping the tags gives the words in order, which is what a model
needs. Where a proper library is present it is used instead. An image cannot be
read by a text model, so what is recorded is its format and dimensions, and you
are told so plainly rather than left wondering why it was ignored.

Attached pictures show as thumbnails beside their filenames above the composer —
a filename tells you a picture is there, a picture tells you which one.

**+ beside the composer** attaches files to the next message, with the same
readers. The text travels with the message rather than becoming a task in
itself.

**The canvas** is a shared editor on the right rail — shared because the model
writes to it too. A model asked to produce code in a chat reply has to reproduce
the whole thing every time it changes a line, which is slow, expensive in
context and error-prone. Instead it reads the canvas with line numbers and edits
a span of them:

| Tool | Does |
|---|---|
| `canvas_read` | The buffer, line-numbered |
| `canvas_write` | Replace it entirely |
| `canvas_append` | Add to the end |
| `canvas_edit` | Replace lines *start..end* |
| `canvas_save` | Write it to a file in the workspace |

**Code in a reply is moved to the canvas.** Telling a model where to put code
only works when it listens, and some do not — leaving a wall of source in the
chat that cannot be edited, run or saved while the canvas beside it sits empty.
So the harness moves it: fenced blocks are lifted out of the finished reply into
the canvas, and the reply keeps its explanation with a line saying where the code
went.

Judgement is applied rather than a blanket rule. A two-line snippet quoted
mid-sentence is part of the explanation and stays; so does anything in a `text`
or `output` fence, which is usually a paste of what a command printed. Several
blocks in one reply are kept in order with a rule between them. Nothing is moved
if the model already wrote it to the canvas itself.

A model that decides the canvas must be an XML tag — writing the file inside
`<canvas>…</canvas>` and then calling `canvas_save` on an empty buffer — has its
intent taken at face value: the block is read as the write it was meant to be,
including when the reply was cut off before the closing tag.

The prompt gives the model the whole sequence — read, write or edit, save — and
tells it plainly never to paste a file into a reply or build one up with
`write_file`. Neither can be edited while it works, and reproducing a file to
change one line spends the context the task needs. As a backstop, code written
with `write_file` is mirrored into the canvas anyway: the result belongs on
screen whether or not the model took the intended route. Edits from either side appear on the
other immediately.

**The canvas** is a scratch editor. Working on code through a chat
window means pasting it in, reading a reply and pasting it back; the canvas keeps
one buffer both sides can see. **Check** sends exactly what is on screen for
review, **Save** writes it into the project folder, and **Copy** takes the lot.

**Writing a plan yourself.** *Add step…* takes one step per line. A model that
will not decompose a task — small ones often will not — should not leave the
checklist unusable, and hand-written steps behave exactly like generated ones.
When every step is closed the plan collapses to a single line, so a finished
checklist does not sit on screen implying there is still work to do.

**Editing while it runs.** The Plan tab can remove a step, mark one done without
running it, clear completed steps, or discard the plan entirely. Edits take
effect on the model's next step, because the checklist is re-rendered into every
prompt. Step numbers are never reassigned — the model refers to them by number,
and shifting them underneath it makes it close the wrong one.

**Pausing.** *Pause after this step* holds at the next step boundary. The check
sits between steps rather than mid-generation: pausing part-way through a tool
call would abandon it half-written, whereas a step boundary is a clean place to
stop, edit the plan, and resume.

**Repetition.** Small and non-chat models get stuck repeating themselves. Two
guards apply. An identical tool call — same name, same arguments — is not run a
third time; the model is told the result will not change and asked to do
something else, and a fourth attempt ends the turn. Separately, immediately
repeated lines and sentences are folded into one before an answer is displayed
or spoken, which is what a completion-only model does when handed a greeting.

**The plan belongs to the conversation.** Starting a new conversation starts
from an empty checklist; reopening an old one restores its plan exactly as it
was left, and the conversation list shows how far each got. Switching workspace
folder switches to that folder's checklist, for the same reason memory is scoped
that way — a plan from another project is worse than no plan.

**Persistence.** The plan is written to `.kestrel/todo.json` in the workspace, so
it survives a restart as well as compaction. The Plan tab displays exactly the
block being sent to the model, rather than an independently derived progress
indicator.

---

## 6. Long-term memory

A transcript is not memory. Compaction discards it, and a new session begins
empty. Long-term memory is the component that persists.

### 6.1 Storage

SQLite with the FTS5 full-text index. There is **no embedding model**, and this
is a deliberate choice rather than an omission: a second model resident in memory
is precisely the wrong trade on hardware that is already constrained, and keyword
retrieval over a few hundred short factual statements performs well. Where a
Python build lacks FTS5, retrieval falls back to substring matching
automatically.

### 6.2 Retrieval

Recall is scored rather than merely matched. Four signals contribute:

| Signal | Rationale |
|---|---|
| Text relevance (BM25) | Does this memory concern the current request? |
| Stated importance | Explicit weighting, 1–5, set when stored |
| Recency | Recent knowledge is more likely to remain accurate |
| Usage count | Memories that have proved useful before tend to again |

Pinned memories bypass scoring entirely and are always included. The selected set
is rendered into a compact block sized to the memory allowance, so recall can
never crowd out the conversation.

### 6.3 What gets stored

Two filters, because a memory store that fills with noise is worse than none: it
costs context on every turn and recalls irrelevance.

**Durability.** Everything is checked before it is stored, and anything that
will not still be true in a month is refused — clock times, dates, weekdays,
command results, directory listings, "the current X", code blocks, and text too
short or too numeric to mean anything later. The capture prompt asks for the
same restraint, but a prompt instruction is advice and a filter is not: "the
current time is 14:32" slips past the first and never the second.

**Provenance.** Tool output is no longer shown to the capture pass at all — only
that a tool ran, never what it returned. Feeding results in is what filled memory
with the date, the contents of a directory, and the text of whatever was read.

The `remember` tool is told *why* something was refused rather than having it
silently dropped, so the model stops attempting it.

### 6.4 Writing

Two paths. The agent has explicit `remember`, `recall` and `forget` tools. And on
task completion, Kestrel asks the model which parts of the exchange are worth
retaining — restricted to durable material such as preferences, environment facts,
decisions and procedures that worked, and explicitly excluding file contents and
transient output.

Restating a known fact raises its importance rather than creating a duplicate;
agents repeat themselves constantly, and an unbounded store is worse than none.
The store prunes its least useful unpinned entries once it exceeds a configured
size.

Automatic capture is skipped below a 6,000-token window, where the additional
generation costs more than the memories are worth. The explicit tools continue to
function.

### 6.5 Scope and inspection

Starting a new session clears the **project** tier and leaves the other two
alone: neither the machine's setup nor what you have said about yourself stopped
being true because a conversation ended.

**Three kinds of memory**, kept apart because they have different lives:

| Tier | Holds | Lives |
|---|---|---|
| **Project** | This codebase — its layout, its decisions, what has been tried | With the folder |
| **Global** | Where tools live, how projects are laid out, what Kestrel can do | Everywhere |
| **Personal** | The person — their name, how they like to work | Everywhere |

A project fact dies with the project; what Kestrel knows about the machine does
not, and what it knows about you belongs to you rather than to a folder. Global
and personal memories are therefore stored without a scope — tying them to the
folder they happened to be learnt in is exactly what made them invisible
everywhere else.

The model picks a tier when it remembers something, and one is guessed when it
does not: automatic capture cannot ask, and filing everything under the project
is how "the user is called Dan" ends up unknown in the next folder. A wrong
guess is correctable in the Memory tab, which has a sub-tab per tier; a store
that predates the distinction is filed by the same guess on first open rather
than dumped into one tier and left wrong.

Recall demands more than a shared word. A single matching term only counts when
it is distinctive rather than one appearing in half the store — otherwise a note
about the clock surfaces on every prompt that happens to mention time. What
survives that is then measured against the best match and anything scoring below
45% of it is dropped, because a weak match propped up by recency and importance
is still weak and costs context on every turn. **Clean up** in the Memory tab
deletes entries the durability filter would refuse today, for stores that predate
it.

Recall requires a genuine match: a memory has to contain something that was
actually asked about, not merely rank highest among poor candidates. Ranking
alone will happily return the least-bad memory in the store, which is how
unrelated things surface.

Memory is scoped per workspace by default, or may be configured as a single pool
shared across all workspaces. The Memory tab lists, searches, edits, pins,
deletes, imports and exports the store, so its contents are never opaque.

---

## 7. Personality

A persona file re-sent on every turn is a recurring cost. Two thousand tokens of
background, charged again at every step, on a window that may be four thousand
tokens wide, is not a viable arrangement.

Kestrel separates the two things such a file conflates.

**Voice** — how the agent sounds — is small and belongs in every prompt.
**Background** — history, detail, characterisation — is large, rarely
load-bearing, and belongs behind a tool call.

A persona is therefore compiled once into tiers, and the window determines which
tier is sent:

| Tier | Applied at | Contents | Measured cost |
|---|---|---|---|
| 0 | `nano` | name and voice, one line | 34 tokens |
| 1 | `small` | voice plus two rules | 63 tokens |
| 2 | `standard` | voice, traits, all rules | 118 tokens |
| 3 | `large` | everything distilled | 118 tokens |

The complete file — 196 tokens in the bundled example — is never injected. The
model retrieves it with the `persona` tool if the detail becomes relevant,
following the same progressive-disclosure pattern as skills.

Personas are Markdown files with `name`, `voice`, `traits` and `rules` in
frontmatter. A file lacking frontmatter is distilled heuristically: the first
paragraph becomes the voice, bullet lines become rules. An existing SOUL.md can
therefore be used unmodified.

Because the injected form is small and derived, changing personality is a cheap
operation rather than a renegotiation of the prompt budget — choosing one in the
Persona tab applies from the next message, with no reconnection and the
conversation so far kept. The Status tab shows which is active, and the bundled
personas are found relative to the package, so a launch from a menu shortcut
finds them just as a launch from the project folder does.

---

## 8. Reasoning models

Reasoning models present the sharpest constraint in a small window. A model may
spend two thousand tokens deliberating before emitting its first tool call, and a
tool call truncated by the token limit wastes the entire step.

Kestrel accepts a reasoning trace in all three forms it arrives in: a separate
`reasoning_content` field, an inline `<think>` block, or an unterminated trace
left behind when generation is cut short. All three are normalised identically.

Four behaviours follow.

**The reserve expands.** Enabling reasoning enlarges the generation allowance
automatically; setting an explicit budget sizes it precisely. At 8,192 tokens the
output reserve grows from 1,474 to 3,242, and the transcript allowance contracts
correspondingly so the window remains balanced.

**Traces are never resent.** Past reasoning is displayed but excluded from the
history submitted on subsequent turns. This is a pure saving: the model has
already acted on that reasoning, and re-reading it changes nothing.

**Traces collapse.** The trace streams live, then folds into a single summary line
with its token count once the answer begins.

**Runaway reasoning is detected.** If a model consumes its entire generation
budget deliberating and produces nothing, Kestrel reports that explicitly and
directs the user to the budget control, rather than presenting an empty reply.

Controls cover mode (`auto`, `on`, `off`), a token budget, the server's
`--reasoning-format`, reasoning effort, and whether traces are displayed at all.

---

## 8a. Projects and conversations

**Switching project moves everything with it.** Clearing the transcript is not
enough: the checklist, the thinking log, the memory scope and the file sandbox
are each tied to a folder, and leaving any of them attached to the previous one
is how a new project arrives already carrying another project's plan, reasoning
and recollections. All five are rebound together.

Thought recall is narrower still — per task, not per project. Reasoning about
last week's installer is not context for today's question, so `THOUGHTS.md` is
written under task headings and only the current task's lines are recalled.

A **project** is a folder named `YYYY-MM-DD name` inside the workspace root, so
projects sort chronologically wherever they are listed — in Kestrel, in Explorer,
in a terminal — without anything needing to read metadata. Each holds its own
files, memory scope, checklist and saved conversations.

Two files are kept in the project folder for people rather than for the program:

**PLAN.md** — the checklist as a document, rewritten on every change:

```markdown
# write the cluster split into my notes

Updated 2026-08-16 23:10 · 2 of 3 done

- [x] Read the existing notes file
- [x] Write the cluster split into notes.md
- [ ] Verify the file reads back
```

**HANDOVER.md** — what a competent colleague would write on a card before going
home: where the work is, what has been done, what is next, which files were
touched, and what is in the way. Written when the context is compacted — the
moment the detail stops being available — and when a turn ends with the
checklist still open.

It is loaded at the start of a conversation, so a full context window or a
deliberate fresh start no longer loses the thread: the next conversation in the
same project opens knowing what the last one was doing. It is a summary and says
so, telling the model to check anything it is unsure of rather than assume. A
note older than three days is not resumed silently.

**THOUGHTS.md** — one line of reasoning per step. A reasoning trace is discarded
after the turn that produced it, which is why a small model can think the same
thought at step 2, step 5 and step 9. A dozen summarised lines cost a few dozen
tokens and answer the question that actually prevents it: what have I already
considered? Repetitions are counted rather than appended, and a thought reached
three times is marked as such to the model and in the file. Recall is narrowed
twice — to this conversation and to this task within it — because reasoning from
another conversation is someone else's working, however recent. The file keeps
everything, grouped by conversation and then by task.

Both are plain markdown, so the state of a piece of work is legible without
Kestrel running. Switching folder switches all four together — an agent that
recalls one project's decisions while working on another is worse than one that
recalls nothing. The Projects tab is a path field and a Browse button; the folder
is created if it does not exist.

**Conversations** are saved after every turn rather than on exit, so a crash or a
closed window costs nothing. Each is listed with its opening line, turn count and
age, and reopening one restores the transcript into the running agent. A session
saved on a large context window reopens correctly on a small one, because the
history is re-fitted to the current budget on the next turn.

Both live under the Projects tab.

---

## 8b. Downloading models

Downloading is not something to watch: a 40 GB file runs for an hour while the
application is used for something else. It therefore has its own window, opened
from **Downloads** in the top bar, and the transfers belong to the application
rather than to the window — closing it stops nothing, and reopening shows them
still running.

Searching filters on Hugging Face's `gguf` tag first and, when that returns
nothing, repeats the search without it — a repository holding GGUF files is not
always tagged as one. A failure is shown in the results list rather than only in
the status line, with the HTTP code and a note on the likely cause, and a **Test**
button reports what the network actually does instead of leaving an empty list to
be interpreted.

The window has a second tab listing the models already on disk, with their sizes
and a running total — downloading without seeing what you already have is how a
20 GB file gets fetched twice. Models can be deleted from there as well as from
the Models panel.

**An unfinished download survives a restart.** The queue is written alongside the
`.part` files, so a transfer paused when Kestrel closed reappears — still paused,
because a download that resumes itself on launch is a surprise — with Resume
ready. Verified: paused at 1 MB, restarted, resumed, and the finished file was
the right size.

Several run at once, two by default and up to four. Each can be **paused and
resumed**: pausing keeps the bytes already on disk in a `.part` file, and
resuming asks the server for the rest with a range request. Cancelling is the
only thing that discards them. A transfer interrupted by a closed laptop or a
dropped connection resumes the same way, because the partial file *is* the
state.

Verified against a server that honours ranges: paused at 1,048,576 of 6,000,000
bytes, resumed with a request for exactly that offset, and the finished file was
byte-identical to the original.

---

## 8c. A team on one model

Two switches in **Settings → Agent**. *Several agents sharing one model* turns
the team off entirely, leaving a single assistant with no roles, delegation or
whiteboard. *Minimal prompt* goes further, sending the task and the tools and
nothing else — no team, canvas, memory, handover or thinking log — which exists
to answer one question: if the output is still wrong with everything optional
removed, the fault is below the prompt, in the weights, the cache or the
template, and no amount of rewording will reach it.

**One identity, said once.** A persona and a role both want to open with "You
are …", and a prompt that says it four times — as Quartermaster, as Lead, as the
one the user talks to, as one of several agents — is asking the model to decide
which it believes. The name, the job and the manner are assembled into a single
statement: *"You are Lead, speaking as Quartermaster. Your speciality is …"*. A
persona's own identity line is stripped, because what is wanted from it here is
its manner, not a competing claim about who the model is. With a team, a
session-wide persona no longer applies at all — a character belongs to a role.

**Roles and personas are not the same thing**, and both are worth having. A
persona is who Kestrel is *to you* — tone, manner, background — applied across
the session, with its own tiering so the full character costs nothing until the
model asks for it. A role is what someone *does*: a speciality, its own
conversation, a place work can be handed to.

A persona belongs to whoever is wearing it, so there is no separate Persona tab:
each role picks one in the **Agents** tab, or none, and **Personas…** there opens
the editor for writing and importing them. A role with no persona is just the
job. Each role also carries a short
brief — what the job is, not what the character is like, which is the persona's
business.

The usual way to build several agents is to give each its own model, which on a
laptop means loading none of them. Kestrel does the opposite: one set of weights
stays loaded and the **role** is swapped around it — briefing, conversation and
tools. Switching costs a prompt rebuild rather than a model load, so a four-agent
team runs on the hardware that runs one agent.

The default team is Lead, Builder, Reviewer and Scribe. Deliberately few: every
agent is another conversation to keep straight, and a team of nine on one model
spends its time waiting rather than working. Add your own in the **Agents** tab
with a name, a speciality and how they should behave.

**Each keeps its own conversation.** That is the point of the separation — a
reviewer that has read every keystroke of the implementation is not a reviewer,
it is the same context wearing a different hat. Clicking a name switches who
answers and brings up their history.

Two things are shared:

| | |
|---|---|
| **Whiteboard** | A folder in the project any agent can read and write. Work is handed over as files, which outlive the conversation and can be read by a person. |
| **Mailboxes** | `agent_send` leaves a message delivered at the start of the recipient's next turn. Asynchronous on purpose: they share one model and take turns, so anything else would misrepresent what is happening. |

**Delegation is the normal way work moves.** You talk to the Lead; it hands
pieces out with `delegate` and waits. The specialist picks up its own
conversation, does the work, reports back in a sentence or two, and the Lead
reads that and decides what happens next. On one model this is a real handover
rather than a parallel call — which is why the reply says what it did and where,
rather than pasting the work back.

Four limits keep it from becoming a machine for talking to itself:

- a specialist cannot delegate onward — it would become a middle manager, and on
  one model that is a queue with extra steps
- four handovers per turn, then it must answer with what it has
- delegating to yourself, or to a name that does not exist, is refused with the
  list of who does
- the specialist is told the task in full, because it genuinely cannot see the
  Lead's conversation

A heuristic offers an opinion when a task obviously suits someone — writing code
to Builder, checking it to Reviewer, writing it up to Scribe — and abstains when
it is unclear, because a confident wrong routing sends work to someone unequipped
for it and the Lead never finds out. It is an opinion, not an instruction.

Handovers appear in the transcript as they happen: `→ handing to Builder: …`
then `← Builder: …`.

Tools: `delegate`, `agent_list`, `agent_send`, `whiteboard_write`,
`whiteboard_read`.

---

### 8d. The tool listing shrinks with the window

A model cannot call a tool it has not been told exists — a folder path would not
do, because it would need a tool to look in the folder. But it does not need
every parameter of every tool in front of it. At the tightest budget only the
names are listed and `tool_help(name)` fetches the arguments, which takes the
listing from 346 tokens to 86.

That matters more than it sounds. The system prompt has an allowance, and when
it exceeds it the prompt is cut — truncating the tool list mid-entry and leaving
the model with half an instruction. The reply is then empty or nonsense, which
looks like a broken model rather than a prompt cut in two. Kestrel now says so
when it happens, and with names-only listing a 28-tool build fits a 4,096-token
context with room to spare.

---

## 9. Skills

Kestrel implements the open [agentskills.io](https://agentskills.io/specification)
format — the same `SKILL.md` layout used by Hermes, Claude Code, Codex CLI and
others. Existing skills work without modification.

```
skill-name/
  SKILL.md      YAML frontmatter (name, description) followed by a Markdown body
  scripts/      executables the body may invoke
  references/   documentation loaded only when required
  assets/       templates and data
```

Access is progressively disclosed across three levels:

1. **Name and description** appear in the system prompt — roughly 25 tokens per
   skill, trimmed harder at smaller profiles.
2. **The body** is retrieved with `skill_open` when the model judges the skill
   relevant.
3. **Bundled files** are reached with the ordinary file and shell tools.

The following directories are scanned by default, so skills installed for another
harness are picked up automatically:

```
~/.config/kestrel/skills     ~/.kestrel/skills
~/.hermes/skills             ~/.config/hermes/skills
~/.claude/skills             ~/.config/agent-skills
./skills                     ./.claude/skills
```

Frontmatter is parsed with PyYAML where available, and by a built-in fallback
parser where it is not.

---

## 10. Model management

Kestrel discovers, inspects, downloads and loads GGUF models directly. A separate
model-management application is not required.

### 10.1 Inspection before loading

The catalogue reads each file's GGUF header — architecture, quantisation, trained
context length, layer count, parameter count — without loading the model. The
header occupies the first few hundred kilobytes, so inspecting a 40 GB file costs
one short read.

The bundled chat template is also parsed, yielding two facts otherwise only
discoverable by trial:

- whether the template contains a tools branch, and therefore whether native tool
  calling will work or the text dialect will be used;
- whether the template supports reasoning.

Selecting a model estimates the memory required at the configured context length
and compares it against the pooled memory of the configured cluster.

Directories used by LM Studio and `huggingface-cli` are scanned by default, so an
existing library appears without configuration. Sharded models are listed once,
by their first part.

### 10.2 Acquisition

The Download tab searches Hugging Face for GGUF repositories, lists the files
within one, and streams a download with progress reporting and resumption. A
token may be configured for gated repositories.

### 10.3 Runtime parameters

Under **Params → Runtime**. The complete llama.cpp load-time surface is exposed:

| Category | Flags |
|---|---|
| Capacity | `-c`, `-ngl`, `-t`, `-tb` |
| GPU split | `-ngl` resolved automatically by default (see below) |
| Batching | `-b`, `-ub`, `-np`, `--no-cont-batching` |
| Attention and cache | `-fa`, `--cache-type-k`, `--cache-type-v`, `-nkvo` |
| Memory residency | `--no-mmap`, `--mlock` |
| Multi-device | `-sm`, `-mg`, `-ts` |
| Positional scaling | `--rope-scaling`, `--rope-freq-base`, `--rope-freq-scale`, `--yarn-orig-ctx` |
| Templating | `--chat-template`, `--chat-template-file`, `--jinja` |
| Reasoning | `--reasoning-format`, `--reasoning-budget` |

A live command preview shows the exact command line that will be executed.
Nothing is concealed behind the interface.

### 10.2a Models that can see

A vision model is usually published as two files: the language weights and an
`mmproj` projector holding the image encoder. Loading only the first gives a
model that quietly cannot see, which is indistinguishable from a text model —
which is why one could be reported as text-only.

Kestrel looks for a projector beside the model and passes it as `--mmproj`. The
pairing requires a real name match rather than any `mmproj` in the folder, since
handing llama.cpp an encoder belonging to different weights is worse than
finding none. The Models tab says whether a model reads images and which
projector it found; a known vision family with no projector beside it is
reported as needing one rather than being silently downgraded.

The projector is passed as `--mmproj` alongside `-m`, so the encoder loads with
the weights rather than the language model coming up alone.

Attached pictures appear in the transcript as previews, 260px wide, with the
filename and dimensions beneath and a link to expand them to the full width of
the column. A thumbnail is the right default — an attached screenshot is context
for a question rather than the subject of the page — but a thumbnail of a
screenshot is unreadable, so one click gives the whole thing. They are shown
whether or not the model can read them: what was attached belongs in the record
of the conversation either way.

With a projector loaded, attached images are sent as images: the message becomes
a list of parts rather than a string, which is what llama.cpp expects and what
the composer, the worker signal, the agent and the context manager all now carry
unchanged. An encoded image is counted as roughly 300 tokens rather than by the
length of its base64, or a single screenshot would reserve megabytes of budget
and compact the conversation away to make room. Without one they
are described — dropping them silently is how a model ends up answering about a
picture it was never shown.

### 10.1a When the output is nonsense rather than the load

A model that loads and then answers with word salad, or drifts into another
language, is usually missing its chat template rather than broken. Merged and
converted models often carry one that is absent or truncated; the server falls
back to a generic template, the model never sees the turn markers it was trained
on, and it answers as a base model would. Chinese from a Qwen build is the
classic shape of it, and it looks exactly like a bad quantisation.

Kestrel checks the template when a model loads — missing, too short to be real,
not a template at all, or truncated mid-loop — and says so rather than leaving
the output to be interpreted. **Chat template** in Params → Runtime supplies one
by name (`chatml` suits most Qwen builds), which is passed to llama-server as
`--chat-template`. The Models tab reports the state of it before loading.

---

### 10.2a Comparing two models

When one model of a family loads and another does not, the answer is almost
always a single field — a wider vocabulary, twice the layers, a quantisation
that is not what the filename claims — and it is invisible until the two are put
side by side. **Compare with loaded** in the Models tab does that against
whichever model last started successfully, and a failed load adds the same
comparison to the error.

---

### 10.2b Why two models of the same size behave differently

**The vocabulary matters as much as the weights.** llama.cpp allocates an output
buffer of batch x vocabulary x 4 bytes, which has nothing to do with how large
the model is:

| Model | Vocabulary | Output buffer at batch 2048 |
|---|---|---|
| Gemma 4 26B-A4B | 262,144 | 2.00 GB |
| Qwen3 27B | 151,936 | 1.16 GB |

Nearly a gigabyte of difference between two 15 GB models. It is why a Gemma can
fail to load while a larger Qwen succeeds, and why the error mentions compute
buffers rather than the model. Kestrel reads the vocabulary and sizes the batch
to it before the first attempt, saying what it did and why; the Models tab shows
the vocabulary and the buffer it implies.

The weights are only part of what a model needs. The KV cache scales with
layers, key/value heads and context length, and varies enormously between
models of similar file size:

| Model | Weights | KV at 10k ctx | Total |
|---|---|---|---|
| Qwen3 30B-A3B (48 layers, 4 KV heads) | 17.2 GB | 0.47 GB | 18.0 GB |
| A 27B MoE (62 layers, 8 KV heads) | 16.2 GB | 2.42 GB | 18.9 GB |

The smaller file needs more memory. That is why one loads and the other does
not, and why the answer is context length rather than a smaller download.

**Three different things can fail to allocate**, and they have different fixes:

| Message | What ran out | Fix |
|---|---|---|
| `failed to allocate compute pp buffers` | The compute buffer | A smaller batch |
| Allocation failures naming the model or backend buffer | Weights or KV cache | Less offload, smaller cache, less context |
| `unable to allocate backend buffer` | The device | Fewer layers on it |

The compute buffer is sized by the batch, not by the model or the context, so a
model can fail here while fitting in memory comfortably — and a larger model
with a narrower batch will load where a smaller one did not. Kestrel names this
case specifically rather than reporting a generic allocation failure.

**The cache does not have to live on the GPU.** `--no-kv-offload` keeps it in
system RAM while every layer of weights stays on the device, which is usually
the right trade: the weights are read constantly and benefit most from being
there, and system RAM is the larger pool. For a mixture-of-experts model
`--cpu-moe` goes further — only a fraction of the experts run per token, so the
idle ones are the cheapest thing to leave behind. Both are switches in
Params → Runtime.

**The recovery ladder** applies these in order of what they cost, retrying after
each failure:

1. reduce the batch and micro-batch, which is what sizes the compute buffer
2. keep the KV cache in system RAM
3. keep the mixture-of-experts weights on the CPU
4. use an 8-bit KV cache, halving it
5. halve the GPU offload — repeatedly
6. run entirely on the CPU
7. halve the context

An 8-bit KV cache sits near the end rather than early: it halves the memory but
is the change most likely to cost output quality, and on some Vulkan builds it
combines badly with flash attention to produce nonsense rather than an error —
so it is applied with flash attention switched off.

**Settings are remembered per model.** What fits is a property of the model and
the machine, not of the session: a model that needed flash attention off and a
batch of 512 will need them again tomorrow. Those settings are saved against the
model's filename when it loads successfully and put back when it is next
selected, so nobody has to rediscover them by watching it fail.

**The embedding and output tensors are counted separately.** They are not part
of the per-layer stack — llama.cpp keeps the output tensor on the device even at
partial offload — so treating the file as an even pile of layers spreads them
across it and gets the arithmetic wrong. On a 152k vocabulary that is 1.2 GB; on
Gemma's 262k it is 2.1 GB, in a budget where four hundred megabytes decides
whether a model loads at all.

**Headroom is reserved rather than filled.** The driver needs memory of its own
for compiled shaders, attention scratch and the output buffer. Kestrel now works
out how much — the output buffer is knowable as batch x vocabulary x 4, the rest
is a fixed reserve that is larger when flash attention is on, since it compiles
more shaders — and keeps it free before deciding how many layers fit.

This is not wasted space. On integrated graphics every allocation comes from the
same system RAM, so a layer left on the CPU costs little beyond the compute units
it would have used, while a pipeline that cannot be built costs the entire load.
The arithmetic on a 15 GB shared budget with Gemma 4 26B:

| Batch | Flash attention | Reserved | Layers on the GPU |
|---|---|---|---|
| 2048 | auto | 3.4 GB | 40 of 62 |
| 512 | auto | 1.5 GB | 46 of 62 |
| 512 | off | 1.1 GB | 47 of 62 |
| 128 | off | 0.6 GB | 49 of 62 |

Nine more layers from the same budget, by not letting the driver run out.

**Flash attention is its own failure.** Vulkan compiles a compute pipeline for
it at load, and that needs device memory of its own; when the layers have taken
it all, the pipeline fails with a message naming a shader —
`Compute pipeline creation failed for flash_attn_f32_f16_aligned` — which reads
like a driver fault rather than the memory pressure it is. Kestrel recognises it
and turns flash attention off first, ahead of everything else in the ladder.

**What recovery changed is recorded and shown.** These settings stay, because
the model needs them to load at all, but an 8-bit cache or an offload of zero
costs quality or speed and nobody would guess a week later that they were still
on. The Status tab lists them, a note appears when a model loads with them
active, and **Reset to defaults** in Params → Runtime undoes exactly the ones
recovery set.

Context is last because it is the only one that changes what the model can
actually do; everything above it costs speed and nothing else. When the ladder
runs out, Kestrel says what it tried and that a smaller quantisation is the next
step, rather than reporting a bare failure.

Kestrel checks before loading and, when a model will not fit, says what it needs
and offers the largest context that would. If a load still fails once nothing is
on the GPU, it halves the context and tries again rather than stopping. The
Models tab shows the split — weights and cache separately — so the cause is
visible before anything is attempted.

Estimates account for grouped-query attention: a model with 40 query heads and 8
key/value heads needs a fifth of the cache its embedding width alone suggests.
Ignoring that overstates the requirement several times over on exactly the
models people are running now.

### 10.3a Splitting a model between GPU and system RAM

llama.cpp puts on the GPU whatever it is told to, and **fails the allocation if
it does not fit** — it does not spill the remainder to system RAM by itself. A
default of "all layers" therefore turns any model larger than the card into a
model that will not load at all, which is wrong for integrated graphics and for
large models on modest cards alike.

GPU layers default to **auto**: Kestrel reads the model's layer count and size,
measures the memory the device actually reports, subtracts the KV cache for the
configured context and a margin for compute buffers, and offloads the layers
that fit. The rest runs from system RAM. For a 22 GB model of 48 layers at 4096
context:

| Device | Result |
|---|---|
| No GPU detected | `-ngl 0` — entirely on the CPU |
| 2 GB integrated | `-ngl 2` |
| 8 GB | `-ngl 14` |
| 16 GB | `-ngl 29` |
| 24 GB | `-ngl 44` |

**Params → Runtime** has a slider for it. The bar shows what each side holds in
gigabytes as the handle moves — graphics memory on the left, system RAM on the
right — with a dashed mark at the point the device runs out, since crossing that
is what turns a slow load into a failed one. On integrated graphics both sides
are the same physical memory, and the caption says so.

Integrated adapters report a token amount as dedicated memory — a UHD 620 says
1 GB while Windows shares 15.9 GB of system RAM with it — so both figures are
shown wherever GPU memory appears, and the larger one is what sizes the offload.
Where a driver reports something implausible, **GPU memory budget** overrides it
outright.

Auto sets the largest split that fits; the number can also be typed. 0 is CPU
only. There is no requirement to fit the model on the GPU: whatever is not
offloaded runs from system RAM, which is how a 15 GB model runs on a machine
with a 2 GB adapter.

### 10.4 Sampling parameters

Applied per request, requiring no reload: temperature, top-k, top-p, min-p,
typical-p, repeat penalty and window, presence and frequency penalties, Mirostat
v1 and v2 with tau and eta, DRY, XTC, and seed.

Four presets — deterministic, precise, balanced, creative — are provided because
raw sampler defaults are tuned for creative generation rather than tool use, and a
harness benefits from a lower temperature than a chat session.

---

## 11. Backend acquisition

Kestrel requires `llama-server`. The Backend tab scans `PATH` and the conventional
installation locations, reports the build number of whatever it finds, and offers
two routes when nothing is present.

**Download an official build.** Release assets are selected by scoring their names
against the detected platform, architecture and accelerator, rather than by
matching a fixed filename pattern. llama.cpp's asset naming changes between
releases; a hardcoded pattern would fail silently at some future point.

**Build from source.** Clones and compiles with `-DGGML_RPC=ON`, so the machine
can additionally serve as a cluster worker.

One case deserves explicit mention. llama.cpp publishes no Linux CUDA or HIP
prebuilt. Selecting CUDA on Linux would otherwise yield a CPU build, which
presents later as an unexplained hundredfold slowdown. Kestrel states this plainly
and directs the user to the source build.

---

## 12. Distributed inference

Kestrel uses llama.cpp's own RPC backend rather than reimplementing distribution.
Each worker runs `rpc-server`; the head node runs `llama-server` with `--rpc`, and
the scheduler distributes weights and KV cache across every backend it can reach.

### 12.1 Workers

The worker finds `rpc-server` through the same search the rest of Kestrel uses
and shares its configuration file, so whatever the installer put in place is
picked up without being told where it went. `--install` will build it on that
machine if it has none.

```bash
./node.sh --mem 8192            # contribute 8 GB
./node.sh --mem 24576 --cache   # contribute 24 GB, cache tensors locally
node.bat --mem 8192             # Windows
```

This starts `rpc-server` bound to all interfaces and broadcasts a UDP discovery
beacon.

### 12.2 Head node

The Cluster tab discovers workers, reports reachability and latency, and
constructs the launch command. Workers appear with the memory they have
contributed.

### 12.3 Proportional distribution

llama.cpp divides layers evenly by default, which strands capacity as soon as
machines differ. Given the memory each worker contributes, Kestrel computes the
split:

```
workstation 24,576 MB  +  laptop 8,192 MB   →   --tensor-split 0.750,0.250
```

### 12.4 Operational constraints

Three points, all inherited from llama.cpp rather than introduced by Kestrel, and
all of which commonly cause confusion:

- **Every machine must run the same llama.cpp build.** The ggml wire format
  changes between releases; mismatched versions hang during handshake or fail
  during loading. Pin one tag across the cluster.
- **RPC pools memory, not compute.** If a model already fits on one machine,
  adding workers makes generation slower, because each layer boundary crossing a
  machine incurs a network round trip. Distribute to run a model that would
  otherwise not fit — not for throughput.
- **The RPC protocol provides no authentication or encryption.** Use it on a
  trusted network or a VPN. Do not expose a worker port to the internet.

The bundled `llama-cluster-tuning` skill covers diagnosis in more depth and
includes a script for deriving split proportions.

---

## 13. Speech

Kestrel can be spoken to and can speak back. Because it is a local-first tool,
the arrangement is deliberate rather than incidental: speech is the component
where the convenient path is a cloud API, and the cost of taking that path is
that everything the user says and everything the agent replies leaves the
machine.

**Local engines are the default, and network engines are inert until permitted.**
They appear in the engine list so their existence is discoverable, but they
cannot be selected — and `auto` will not choose one — until *Permit network
speech services* is switched on. Selecting a network engine while the permission
is off silently falls back to the best local engine rather than failing or
leaking.

### 13.1 Speech output

| Engine | Kind | Notes |
|---|---|---|
| Piper | local | Small ONNX neural voices. The best offline quality per megabyte, and the reason a local default is reasonable. |
| System voices (pyttsx3) | local | SAPI5 on Windows, NSSpeechSynthesizer on macOS, espeak on Linux. |
| macOS `say` | local | Built in; no installation. |
| espeak-ng | local | Robotic, tiny, present almost everywhere. The guaranteed fallback. |
| OpenAI-compatible endpoint | network | Any `/v1/audio/speech` service. |
| ElevenLabs | network | Voice list fetched from the account. |

Ten Piper voices across six languages can be downloaded from the Speech tab,
which fetches both the ONNX model and its JSON sidecar. Voice and rate are
selectable; a test button plays a sample.

**Playback happens in-process where it can.** Windows ships none of ffplay,
aplay or sox, so the fallback was spawning PowerShell to play each clip — most
of a second per sentence, and the main reason speech lagged behind a quick
model. `winsound` is used there instead, `sounddevice` elsewhere, and external
players only when neither is available. The same applies to streaming: with no
external player, Piper's raw output is fed straight to the sound card rather
than the feature being unavailable.

**One warm process per session.** Loading the voice model is most of the cost of
a short utterance — seconds to load, a fraction of a second to speak. Piper is
therefore started once and kept for the session rather than per reply. Measured
against a 1.5 s model load: first utterance 1.62 s, every one after it 0.12 s.
Changing voice or speed starts a new process, because it must.

Voice quality is the other half: a `low` voice synthesises several times faster
than a `medium` one and starts quicker, which matters most on a laptop.

**One warm process per reply.** Per-call startup dominates short utterances —
about 0.25 s of process spawn and model load against 0.2 s of actual synthesis.
Piper is therefore started once for the turn with `--output-raw` piped straight
into a player, which removes the startup, the file writes and the gaps between
clips together.

Sentences remain the unit rather than words. Synthesising word by word would pay
the remaining per-call cost nineteen times over for a twenty-word reply — 4.9 s
against 0.7 s measured — and neural voices compute prosody across a whole
clause, so cutting at word boundaries gives every word the flat intonation of a
word spoken alone. It buys about 0.06 s of first-audio latency for that.

**Synthesis and playback are a pipeline** where streaming is unavailable. Rendering and playing in one loop
makes every gap between sentences a whole synthesis; running them as two stages
means the next sentence is already rendered when the current one ends. Measured
with a 300 ms synthesis: first audio at 0.42 s, then no gap at all between
sentences. The first fragment is also cut at a comma rather than a full stop,
because it alone decides how long the silence before speech is.

**Speech starts a sentence in, not a reply in.** The answer is split on sentence
boundaries and each is synthesised as it completes, overlapping speech with
generation instead of waiting for both. In a two-second reply the first sentence
is spoken half a second in.

Agent output is not read verbatim. Fenced code, tables, shell lines and URLs do
not read aloud usefully, so they are removed rather than narrated, and markdown
emphasis is stripped. By default only the final answer is spoken, not the tool
activity leading to it.

### 13.2 Speech input

| Engine | Kind | Notes |
|---|---|---|
| whisper.cpp | local | The natural counterpart to llama.cpp — same ggml runtime, same quantised-model story. |
| faster-whisper | local | CTranslate2; convenient if you already have it. |
| Vosk | local | Lightweight, good on very modest hardware. |
| OpenAI-compatible endpoint | network | Any `/v1/audio/transcriptions` service. |

Seven whisper.cpp models from 75 MB to 1.6 GB can be downloaded from the same
tab. Recording captures 16 kHz mono, which is what every engine here expects, and
uses `sounddevice` where it is installed, falling back to `ffmpeg`, `arecord` or
`sox`. Playback similarly falls back across `ffplay`, `afplay`, `aplay`, `paplay`
and `sox`. Where neither is present, the controls say so instead of failing when
pressed.

**Dictation is continuous, not a fixed recording.** Pressing Dictate starts
listening and words appear in the composer as they are recognised; pressing it
again stops. Recording a block and transcribing it afterwards means the whole
utterance *plus* the whole transcription pass before a single word shows —
seconds of apparent deafness.

Vosk streams natively and emits partial results while you are still speaking, so
it is used when installed. Everything else is fed overlapping two-second chunks,
each transcribed as it closes, which puts words a chunk behind your voice rather
than an utterance behind it. Partial results are re-sent in full as they grow, so
the previous partial is replaced rather than appended.

---

## 14. The interface

PySide6, presented as an instrument panel rather than a chat window.

**Collapsing.** A thin strip sits between each side panel and the centre, with a
chevron at its midpoint. Clicking anywhere along it collapses that panel and
gives its width to the transcript; clicking again restores it to the width it
had. The whole strip is the target rather than a small button, and the chevron
points the way the panel will move.

The generation rate is labelled `gen tok/s`, because llama.cpp's own log reports
prompt-processing speed in the same unit and the two differ by an order of
magnitude — 26 against 0.7 on a loaded laptop. A bare figure invites the reading
that one of them is wrong.

**Scaling.** A panel cannot be dragged narrower than its contents need. The
floor is re-measured when a page is first built — panels are created on demand,
and measuring before that finds an empty widget — and taken from the widest page
opened so far rather than fixed, since a
tab holding a wide form needs more than one holding a list — and it is applied
while dragging rather than as a widget minimum, which would stop the panel
collapsing to its rail. Panel contents stretch to whatever width they are given. Table
columns size to their content except for the one carrying the most text, which
takes the remainder — fixed pixel widths look deliberate at the size they were
chosen for and wrong at every other.

**Start-up is staged.** The interface comes up first and is usable immediately;
only then does the backend start, and only then the connection to it. Each stage
runs after the window is on screen and reports itself in the top bar, so a slow
step is visible rather than looking like a stall.

Nothing during start-up walks the filesystem without a bound. Counting the files
in a project stops at 250 entries or 350 milliseconds and says `250+` rather than
pretending to be exact — an unbounded count is fine for a project folder and
catastrophic for a home directory, where it walks the whole drive before the
window can open.

**Fonts that cannot be used.** Windows ships several bitmap-only faces —
Fixedsys, Terminal, MS Sans Serif and friends — that have no outlines for Qt to
instantiate. Rendering a font list in its own faces asks the system to load
every one of them, which logs a DirectWrite failure per font and takes real
time. Those families are excluded from the pickers and from font resolution.

**Choosing a workspace root.** The root is where projects are created and where
the agent's file tools are confined. Keep it a directory of its own: pointing it
at a home directory or a drive root makes every folder in it a project and
scopes the agent's sandbox to all of it. Change it under Projects → Change…

**Opening the window.** llama.cpp is started when the application opens, with
**no model loaded** — the backend running and waiting, so choosing a model is the
only remaining step. Reading tens of gigabytes into memory unasked at launch is
the wrong default; that happens when a model is chosen.

Panels are built the first time their tab is shown. Start-up used to be the sum
of every panel's constructor, and those constructors read directories, open
databases and probe for binaries. Deferring them makes opening the window
independent of how slow any one panel is, and most panels are never opened in a
given session. Measured here: 134 ms to a usable window, against 430 ms when
everything was built eagerly — and the gap widens on a machine with a large
model library, since that scanning is exactly what is now deferred.

**Loading a model.** This is the longest wait in the application and the one
that most looks like a crash: llama.cpp maps tens of gigabytes and the window
has nothing to show meanwhile. An overlay reports the stage and the server's own
most recent output — `load_tensors: offloading 48 repeating layers to GPU` and
so on — so the wait is legibly progress.

Everything that probes the network does so on a worker thread. Asking whether a
port answers takes milliseconds when the answer is a refusal and five seconds
when a firewall drops the packets instead, and doing that on the GUI thread is
indistinguishable from the application hanging. Measured on an unroutable
endpoint: worst GUI stall 13 ms, against five seconds per call before.

**Waiting.** Generation begins with a silence that can run to tens of seconds on
a large model — prompt ingestion, then reasoning before the first visible token.
An idle window during that period is indistinguishable from a hung one, so a
pulsing indicator reports what is happening: *thinking*, *reasoning*, *running
grep*, *writing*. Start-up narrates itself the same way, over a splash, because
opening the application means scanning model directories, reading skill folders
and probing speech engines — some of which shell out to other programs.

**Top bar.** Present on every page, above the tab stack rather than inside it:
the wordmark, a status line, and the actions that belong to the session as a
whole — new conversation, settings, downloads, and the light/dark switch.

**Composer row.** The controls that act on the message being written sit with
it: following, continue, speak, dictate, stop and send.

**Where settings live.** Two places, split by what they configure. **Settings**
(top bar) holds the application: appearance, agent behaviour, endpoint and
binaries, and the folders searched for skills and models. **Params** (side panel)
holds the model: sampling, reasoning and load-time flags, next to the model they
apply to.

Models can be deleted from the Local tab, sharded ones in full — removing only
the listed file would leave the other parts as orphans that no longer load and
are not obvious to find.

**Every reply carries its own actions** — retry, read aloud, copy, fork from
here. Retry rewinds to just before the prompt that produced *that* reply — found by
walking back through the history rather than remembered when the reply arrived,
since a remembered map goes stale as soon as a conversation is reopened, forked
or retried. Tool results are skipped in that walk: in the text dialect they are
user messages too. Fork branches a new conversation containing everything up to that
reply, switches to it, and leaves the original saved — trying another direction
should not cost the one that got you there.

**The backend dies with Kestrel.** Asking a child process to stop only works if
Kestrel gets the chance to ask, and a crash or a kill from Task Manager does
not. On Windows llama.cpp is started inside a job object marked kill-on-close,
so the operating system enforces it however Kestrel ends. A clean exit also
stops it explicitly, from the close event, `aboutToQuit` and `atexit` alike.

**Closing the window stops the server.** A backend Kestrel started belongs to
Kestrel; leaving it running holds the port and the model's memory after the
window has gone. One adopted from elsewhere is left alone.

**Continue** picks the task back up: the open step if the checklist has one,
otherwise from wherever the reply stopped. A turn can end with work outstanding —
stopped by hand, cut short by the step limit, or a reply that trailed away — and
retyping the request loses the thread. The button carries a mark when the
checklist still has open steps.

**Stop stops the speech too.** A cancelled reply should not carry on being read
out, so the queue is dropped and whatever is making noise is killed rather than
being allowed to finish the sentence in hand.

**The transcript reads as a conversation.** Your messages sit right in one
colour, Kestrel's left in another, and reasoning appears as a quieter italic
bubble between them — visibly an aside rather than part of the reply.

**Settings were once in the side panel, not a dialog.** Having half the options in a
popup and half in the panels meant knowing which half a given one was in. They
are grouped by when they take effect: **Params** holds sampling and reasoning
(next message), runtime (needs a reload), agent behaviour and appearance
(immediate); **Backend** holds the endpoint and the binary; each subsystem's tab
holds its own.

**Collapsing.** Clicking the active tab icon collapses that panel to its icon
rail; clicking any icon reopens it at that tab. The rail is the control, so
there are no separate handles to find — and the splitter holds three panes and
therefore exactly two drag handles, one per side.

**Left column.** Status readouts; the model browser, downloader and runtime
parameters; sampling and reasoning controls; the cluster; installed skills; the
memory store; persona selection; and backend discovery.

**Centre.** Transcript and composer.

**What the transcript shows.** Results, not workings. A skill that fetches the
time shows the time — not the instructions it read, the command it ran, or the
exit status. Everything is kept in full in the Activity tab, and *Show tool
arguments and raw output* on the Status tab puts it back inline.

**Tools** lists everything the model can call this session, with what each one
reaches — read, writes, or runs commands — and its full signature and
parameters. The system prompt tells the model; this tells you.

**A chime marks the end of a task**, since a long one is worth walking away from
and a finished run looks much like a stalled one at a glance. Settings → Agent
switches it off, or replaces it: the bundled chime, anything the platform
already provides, or a file of your own in wav, mp3, flac or ogg.

**Prompt** shows the system prompt exactly as sent, with its size in characters,
words and approximate tokens. It can be edited and saved: an override is used
verbatim and survives a restart, because a prompt you edited and Kestrel then
appended to is not the one you tested. Revert returns to the assembled version.

**Right column.** A matching icon rail: the live task checklist; an activity tree
recording every tool call with its complete untruncated output; the
`llama-server` log; and a system monitor showing CPU, memory and GPU. The monitor
samples only while it is on screen — polling a vendor GPU tool spawns a process
every couple of seconds, which is not worth doing for a panel nobody is looking
at. Reading GPU counters spawns a process, so the first sample takes a second or
two. Until it lands the monitor says it is looking rather than that there is
nothing — reporting no GPU while still checking is simply wrong.

psutil is installed by default; without it the same figures come from `/proc` on
Linux and the Win32 API on Windows.

GPU readings come from `nvidia-smi` or `rocm-smi` where those exist. They do not
exist for integrated graphics — which is what most laptops actually run
llama.cpp on — so on Windows the readings come from the same performance
counters Task Manager uses, covering Intel and AMD integrated parts, and on
Linux from `/sys/class/drm`. Sampling happens on a background thread: reading
those counters spawns a process and takes the better part of a second, which is
not something to make the interface wait for.

**The bottom strip** shows CPU, RAM, GPU, video memory and temperatures under
the context gauge, in space that was empty — the gauge itself is neither moved
nor resized. CPU temperature comes from the platform where it offers one; Windows
generally does not without a driver, so it is left out there rather than guessed.

**Contrast.** Dimmed text sits at better than 6:1 against its background in both
palettes, and a selected row has its own background *and* its own text colour —
inheriting the unselected colour is how a highlighted row ends up unreadable.

**Table columns** behave as tables should: drag a divider and that column
changes while the others stay put. More than one stretching section is what
makes a table feel wrong — Qt keeps them in proportion, so widening one visibly
shrinks the rest. Exactly one section stretches, the last, which takes the
remaining width rather than whatever is left over. A divider handle sits on a column's right edge, so a
last column sized to its contents can be clipped with no way to drag it wider.

**Colour.** Two palettes, each with a surface **tint** (slate, blue, teal,
green, orange, red, violet, graphite) and an **accent** for buttons and
highlights (amber, blue, teal, green, orange, red, violet, pink) — 128
combinations in all. They are derived rather than hand-written: one neutral
palette per mode, mixed a little way towards a hue. Writing a dozen full
palettes by hand would drift out of step the moment a role changed.

Only surfaces take the tint. Text and the status colours are left alone, since
tinting those is how an interface becomes hard to read. Each accent is adjusted
until its own label can be read on it — a mid-tone fill sits too close to both
black and white, and a button label at 2.3:1 is decoration rather than writing.
Every combination clears 4.5:1.

**Themes.** Two palettes, switchable from the top bar and remembered between
sessions. Light is a paper-and-bronze daylight face rather than an inverted dark
theme — inverting a dark palette produces glare and washes the accent colour out,
so the two are designed separately from the same set of role names. Switching
rebuilds the transcript in the new palette: inline HTML does not follow a
stylesheet change, so history would otherwise be left stranded in the old
colours.

**Along the bottom** is the context gauge: the entire window drawn as a single
bar, divided into system prompt, recalled memory, checklist, transcript, free
space, and the generation reserve that is never spent on input. It is the element
the design is organised around — when the transcript band advances rightward, a
compaction is approaching, and this is visible before it occurs rather than
afterward.

---

## 15. Headless operation

The same agent loop without Qt, for use over SSH and for verifying a configuration
before opening the interface.

```bash
./run.sh --cli                                       # interactive
./run.sh --cli "summarise the files in src"          # single task
./run.sh --cli --model ~/models/qwen3-8b-q4_k_m.gguf "audit this repository"
./run.sh --cli --think off --preset precise "..."
./run.sh --cli --memories                            # print the memory store
```

| Flag | Effect |
|---|---|
| `--url`, `--config`, `--workspace` | Endpoint, configuration file, working directory |
| `--model`, `--serve` | Load a model, managing `llama-server` for the session |
| `--dialect`, `--profile` | Force a tool dialect or context profile |
| `--think`, `--think-budget` | Reasoning mode and token cap |
| `--preset`, `--temp` | Sampling |
| `--no-memory`, `--memories` | Disable long-term memory; print its contents |
| `--yes` | Skip approval prompts |

---

## 16. Installation

```bash
bash install.sh     # Linux, macOS, WSL
install.bat         # Windows
```

Use `bash install.sh` rather than `./install.sh` on first run: extracting the
archive does not preserve the executable bit. The installer restores it on the
remaining scripts.

The installer creates a virtual environment in `.venv` and installs PySide6,
requests and PyYAML. Python 3.10 or later is required.

### llama.cpp

The installer handles the backend as part of the same run. It looks for an
existing installation first, reports what it finds, and asks what you want:

```
llama.cpp — the inference backend Kestrel drives

  No llama.cpp installation was found on this machine.
  Install it now? [Y/n]

  How would you like it installed?
    1) Download an official prebuilt build   (fast, recommended)
    2) Compile from source                   (slower; needed for CUDA or ROCm on Linux)
  Choice [1]:

  Which accelerator? This machine looks like: cuda
    1) auto   — detect and choose (cuda)   (recommended)
    2) cpu    — no GPU acceleration
    3) cuda   — NVIDIA
    4) vulkan — any modern GPU, including Intel and older AMD
    5) hip    — AMD ROCm
    6) metal  — Apple silicon
    7) sycl   — Intel oneAPI
  Choice [1]:

  The RPC backend lets this machine join or host a cluster, so a model
  too large for one machine can be spread across several.
  Include RPC support? [Y/n]
```

Every question has a default, so pressing Enter throughout is a complete
installation.

### When llama.cpp is already installed

Finding a binary on disk is not the same as having a working backend. The
installer runs whatever it finds before believing in it, which catches the
failures that otherwise look like nothing happened: a CUDA build on a machine
with no CUDA runtime, a half-finished extraction, an architecture mismatch, a
missing shared library.

```
  Found: /usr/bin/llama-server (b4820)
  rpc-server:        not present — clustering unavailable

  !! This installation does not work:
     error while loading shared libraries: libcudart.so.12

  What would you like to do?
    1) Install a fresh copy for Kestrel to use (leaves yours untouched)
    2) Keep this installation and carry on
    3) Nothing for now — decide later in the Backend tab
  Choice [1]:
```

The default follows the diagnosis: a working installation defaults to *keep*, a
broken one defaults to *replace*.

What option 1 does depends on who installed it. If Kestrel did, it is removed
and reinstalled. If it came from a package manager, Homebrew, or your own build,
it is **left alone** — deleting someone's system packages is not an installer's
business — and a fresh copy is placed in Kestrel's own directory, which takes
precedence. The installer says which of the two is happening.

Removal is confined to Kestrel's own directory in every case:

```bash
bash install.sh --reinstall           remove Kestrel's copy, install again
bash install.sh --uninstall-llama     remove Kestrel's copy and stop
```

The Backend tab has the same controls: a *Remove Kestrel's copy* button, enabled
only when there is something of Kestrel's to remove, and a *Remove existing copy
first* checkbox that is pre-ticked when the current installation is detected as
broken.

### Windows backend archives

Recent llama.cpp Windows releases split the binaries from the base runtime: a
backend archive such as `llama-bNNNNN-bin-win-vulkan-x64.zip` contains the
executables and `ggml-vulkan.dll`, but the base `ggml` and `llama` DLLs live in
the CPU package. Extracting only the backend archive produces files that exist,
look correct, and fail to start with exit code `3221225781` —
`0xC0000135 STATUS_DLL_NOT_FOUND`, which Windows reports as a bare number.

Kestrel verifies the binary immediately after unpacking. If it fails on Windows,
the matching CPU package is fetched and overlaid to supply the base runtime, and
verification runs again. Exit codes in that family are translated into a
sentence rather than shown as a number.

### The unified `llama` binary

Recent llama.cpp builds ship a single `llama` dispatcher with subcommands, so the
server is started as `llama serve` rather than as a separate `llama-server`
executable. Passing `-m` straight to the dispatcher produces
`error: unknown command '-m'`, which reads like a rejected flag rather than a
missing subcommand. Kestrel detects which style a binary is from its name and
builds the command accordingly, preferring a dedicated `llama-server` when both
are present.

### Starting and adopting a server

Kestrel starts the backend itself when nothing is serving, so opening the
application does not require having remembered to start one. If a server is
already responding, it is left alone and used as it is.

The distinction matters more than it sounds. Starting a second server on an
occupied port produces a process that exits immediately — while the health check
succeeds, because the server that was already there answers it. The result looks
like a successful load and behaves like nothing happened. Kestrel therefore
checks the port before spawning, and treats the process exiting as the
authoritative answer rather than the health probe:

- **Kestrel's own server is running** — it is stopped and replaced. Loading a
  different model is the ordinary reason to do that, so it happens rather than
  being refused. Stopping waits for the process to exit *and* for the port to be
  released; terminating is a request, not an outcome, and a server mid-load can
  take seconds to notice.
- **Port free** — start, wait for health, connect.
- **Port held** — Kestrel names the process holding it and offers three ways
  out: stop it and start fresh, connect to it as it is, or move to a different
  port. Every route to a busy port ends at the same dialog, whether it came from
  loading a model, starting the backend, or a failure reported by llama.cpp
  itself. **Restart server** on the Status tab does the same thing without
  waiting for an error. Most often this
  is a llama-server left behind by a previous run: a stranded process that keeps
  answering is otherwise a dead end, since a model cannot be loaded into a server
  Kestrel did not start.
- **Port held and not answering** — refuse, and say what to stop or which port to
  change.
- **Process exits during startup** — report the exit code and the last lines of
  its output, rather than reporting success.

### If the chosen accelerator will not run

An installer's purpose is a working backend, not a particular one. If a GPU
build still fails after the overlay — a missing Vulkan runtime, absent drivers,
a CUDA build on a machine without the CUDA runtime — Kestrel says so and
automatically installs a CPU build instead, which has no driver or runtime
dependencies. The accelerator can be revisited from the Backend tab once the
missing runtime is in place. Pass `--no-fallback` to suppress this.

A path that fails verification is never recorded in the configuration, so a
failed install cannot leave the next run believing the machine is already set
up.

The same choices are available as flags for scripted or unattended installs, in
which case nothing is asked:

```bash
bash install.sh --no-llama                   dependencies only
bash install.sh --llama --backend vulkan     prebuilt, chosen accelerator
bash install.sh --llama --source --no-rpc    build without clustering
bash install.sh --yes                        accept every default
```

`install.bat` takes the same options. Prompts are skipped automatically when the
installer is not attached to a terminal, so piping it into a shell will not hang.

| Choice | Meaning |
|---|---|
| Prebuilt vs source | A release download is fast; a source build is slower but is the only way to get CUDA or ROCm acceleration on Linux, where no prebuilt exists. |
| `auto` accelerator | Inspects the machine — Metal on macOS, CUDA if `nvidia-smi` is present, HIP if ROCm is, Vulkan if `vulkaninfo` is, otherwise CPU. |
| RPC on or off | On by default. Off produces a smaller build that can neither host nor join a cluster. |

Because llama.cpp publishes no Linux CUDA or HIP prebuilt, requesting one there
yields a CPU build. Kestrel says so plainly rather than letting it present later
as an unexplained slowdown, and points at the source build. Prebuilt archives do
not always contain `rpc-server`; when RPC was requested and the binary is absent,
that is reported too, since the Cluster tab would otherwise fail with no
explanation.

The Backend tab in the interface offers the same choices later, and
`python -m kestrel.setup_backend` is the same logic on the command line.

---

## 17. Configuration reference

Configuration is stored as JSON in the platform configuration directory and is
fully editable from the interface. The principal sections:

| Section | Governs |
|---|---|
| `runtime` | Load-time llama.cpp flags (§10.3) |
| `sampling` | Per-request sampler settings (§10.4) |
| `thinking` | Reasoning mode, budget, format, display (§8) |
| `memory` | Enablement, automatic capture, recall count, scope, size limit (§6) |
| `speech` | Engines, voice, rate, dictation, and the network permission (§13) |
| `theme` | `dark` or `light` |
| `llama_backend`, `llama_with_rpc` | Accelerator and RPC choice for installation (§11) |
| `nodes` | Cluster workers (§12) |
| `skills_dirs`, `persona_dirs`, `model_dirs` | Search paths |
| `tool_dialect`, `profile_override` | Dialect and profile selection (§3, §4) |
| `approval` | `always`, `safe` (writes and shell only), or `never` |
| `plan_driven`, `bell_on_finish` | Keep working until the checklist closes; chime when it does (Settings → Agent) |

### Tools

`list_dir` · `read_file` · `write_file` · `edit_file` · `find_files` · `grep` ·
`shell` · `skill_find` · `skill_open` · `plan` · `todo` · `plan_add` ·
`remember` · `recall` · `forget` · `persona` · `finish`

File tools are confined to the workspace. Shell commands execute within it. The
approval policy is switchable mid-session.

---

## 18. Project layout

```
kestrel/
  config.py     settings model and persistence
  llm.py        llama.cpp HTTP client, streaming, capability probing
  tokens.py     token accounting with a self-correcting offline estimator
  context.py    profiles, allocation, clipping, rolling compaction
  prompts.py    tiered system prompt construction
  gguf.py       GGUF header parsing without model loading
  models.py     local catalogue, Hugging Face search and download
  runtime.py    runtime, sampling and reasoning parameters; command construction
  reasoning.py  reasoning trace extraction and normalisation
  todo.py       the task checklist
  memory.py     long-term memory store
  persona.py    personality tiers
  skills.py     SKILL.md discovery and parsing
  llamacpp.py   backend discovery and installation
  setup_backend.py  installer entry point used by install.sh / install.bat
  speech.py     text-to-speech and speech-to-text engines
  cluster.py    node discovery, probing, proportional split, process supervision
  agent.py      the loop: both dialects, spooling, planning, memory, reasoning
  node.py       cluster worker entry point
  cli.py        headless mode
  ui/           Qt interface
```

Dependencies: PySide6 (LGPL), requests, PyYAML. SQLite is part of the standard
library. Speech engines are optional and detected, never bundled. There is no
framework, no vendor SDK, and no telemetry.

---

## 18. Support and thanks

**Settings → About** carries the version, what Kestrel is, and two links:

- **Forum** — https://alternitech.freeforums.net/ for questions, problems and
  suggestions.
- **Donate** — https://alternitech.square.site/product/donation/6

Kestrel is free and stays that way; the donation page is there for anyone who
finds it useful enough to put something back.

---

## 18a. Updates

**Settings → Updates** compares the version published at
`github.com/dansdesigns/kestrel/version.txt` with the one installed and reports
the difference. **A tie between two versions written differently is not a tie** — it means the
comparison could not read one of them, and the update is offered anyway.
Offering redundantly is the safe way to be wrong; refusing strands the machine.

**Reinstall the published version** in Settings → Updates installs whatever is
published regardless of the numbers. It exists because a machine whose version
comparison is broken cannot be fixed by an update it will not offer itself.

Versions are compared as numbers with as many parts as they have: `1.0` is
`1.0.0`, `1.10` is above `1.2`, and a leading `v` is ignored. Requiring exactly
three parts meant `1.0` and `0.15` both failed to parse and fell back to zero —
so every comparison between them was a tie, and a two-part version could be
reported as ahead of a newer one.

The installed version is read from `version.txt` when there is one — that file
belongs to whoever maintains the checkout, and Kestrel only ever reports it,
never writes it. Without one it falls back to the package constant.

When an update is available an **Install the update** button appears. A git
checkout is pulled, which keeps your history. Any other copy is replaced from
the published archive:

1. downloaded and unpacked into a temporary folder
2. checked that it actually contains Kestrel
3. the current copy backed up to the system temporary directory — not beside the
   program, which is what left a litter of folders next to it
4. files overwritten in place — the folder structure is preserved and **nothing
   is deleted**, so notes, scratch files and modules of your own survive an
   update
5. a restart offered, which launches the new copy before closing this one

Nothing is written into the program folder until a complete, verified copy
exists elsewhere, so an update that fails partway leaves the working
installation alone. `.git`, `.venv` and your settings, memory, conversations and
projects are never touched.

---

## 19. Verification status and known limitations

Kestrel was developed against a mock llama.cpp server reproducing the `/health`,
`/props`, `/tokenize` and streaming `/v1/chat/completions` endpoints. The
following were verified by execution:

| Area | Result |
|---|---|
| Agent loop, text dialect, 4,096 tokens | 5-step task, 295-token system prompt, 2,475 tokens free at completion |
| Compaction, 2,048 tokens | 12 steps, 8 compactions, peak input 1,516; no overflow |
| Agent loop, native dialect, 32,768 tokens | Completed; no compaction required |
| Dialect fallback | Server rejecting `tools` correctly downgraded to text |
| Output spooling | 23,893 characters retained in 233 tokens |
| GGUF parsing | Architecture, quantisation, context, layers; tool and reasoning detection |
| Memory store | Retrieval scoring, deduplication, budgeted injection, capture parsing |
| Checklist | Seeding, state transitions, drift reminder, overflow clipping, persistence |
| Persona | Tier compilation; distillation from unstructured SOUL.md |
| Reasoning | All three trace forms; reserve expansion; exclusion from history |
| Cluster | UDP discovery round trip, TCP probing, proportional split, command construction |
| Backend asset selection | Correct across Linux, Windows and macOS on x64 and arm64 |
| Broken-install detection | Missing shared library, illegal instruction, non-zero exit, hang, bad permissions, vanished path |
| Managed-install detection | Kestrel's own directory distinguished from system, Homebrew and user builds |
| Removal | Confined to Kestrel's directory; idempotent when there is nothing to remove |
| Sharded model sizes | Every shard counted, not just the first |
| Chat-template compatibility | Helper calls carry a system message; same-role turns merged |
| Windows exit-code translation | `0xC0000135` and related statuses reported as sentences |
| CPU-package overlay | Correct companion archive selected for every Windows backend |
| Accelerator fallback | A failing GPU build retried as CPU, ending with a working backend |
| Unified binary | `llama serve` built correctly, including Windows paths parsed off-Windows |
| Server adoption | Existing endpoint reused; dead port refused; instant exit reported with its output |
| Installer detection fragment | Empty fields survive; `%`, `!`, quotes and redirection neutralised |
| Installer decision paths | Working keep, broken managed replace, broken system install-alongside, decline, flags-only, install failure |
| Speech engine selection | Detection, auto-selection, and the network permission gate |
| Speech text cleanup | Code fences, tables, links and emphasis removed before synthesis |
| Backend installer | Option parsing, detection, version reporting, exit codes |
| Themes | Palette switch and transcript rebuild, both directions |
| Interface | Driven end to end offscreen through complete tasks, in both palettes |

**Not verified.** Two paths could not be exercised in the development environment
and warrant an early check:

- **Hugging Face downloads and GitHub release retrieval.** The build environment
  had no route to either host. The selection and URL-construction logic beneath
  them is tested; the network transfer itself is not.
- **Speech synthesis and transcription against real engines.** Piper,
  whisper.cpp and the rest are not installable in the build environment, so
  detection, selection, command construction and the permission gate were tested
  against stand-in binaries; no audio has actually been produced or recorded.
- **Loading a real GGUF through a real `llama-server`.** Every code path was
  exercised against the mock, but no actual model has been loaded. Expect minor
  friction with a specific model's chat template; `--dialect text` is the reliable
  fallback if native tool calling misbehaves.

The command preview under Models → Runtime displays precisely what would be
executed, and is the fastest way to confirm the flags suit a particular llama.cpp
build before committing to a load.
