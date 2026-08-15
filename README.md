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

**Drift correction.** Models reliably begin a task by maintaining the checklist
and then quietly stop. After three steps without a change, a single-line reminder
is appended to the next tool result. It costs about a dozen tokens and arrives at
the point of action, which is where it is most likely to be acted upon.

**Overflow.** When a plan grows beyond its allowance, the rendering keeps the
current step and its immediate neighbours and summarises the remainder by count.
A twelve-step plan renders in 76 tokens.

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

A **workspace** is a folder with its own files, memory scope, checklist and
saved conversations. Switching folder switches all four together — an agent that
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

The Models tab shows the split for the selected model before loading it. Setting
the value explicitly overrides this: 0 for CPU only, 999 for everything on the
GPU.

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

**Scaling.** A panel cannot be dragged narrower than its contents need. The
floor is measured from the widest page opened so far rather than fixed, since a
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
the wordmark, a status line, the speech-output toggle, dictation, and the
light/dark switch.

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

**Closing the window stops the server.** A backend Kestrel started belongs to
Kestrel; leaving it running holds the port and the model's memory after the
window has gone. One adopted from elsewhere is left alone.

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
there are no separate handles to find.

**Left column.** Status readouts; the model browser, downloader and runtime
parameters; sampling and reasoning controls; the cluster; installed skills; the
memory store; persona selection; and backend discovery.

**Centre.** Transcript and composer.

**What the transcript shows.** Results, not workings. A skill that fetches the
time shows the time — not the instructions it read, the command it ran, or the
exit status. Everything is kept in full in the Activity tab, and *Show tool
arguments and raw output* on the Status tab puts it back inline.

**Right column.** A matching icon rail: the live task checklist; an activity tree
recording every tool call with its complete untruncated output; the
`llama-server` log; and a system monitor showing CPU, memory and GPU. The monitor
samples only while it is on screen — polling a vendor GPU tool spawns a process
every couple of seconds, which is not worth doing for a panel nobody is looking
at. psutil is installed by default; without it the same figures come from `/proc` on
Linux and the Win32 API on Windows.

GPU readings come from `nvidia-smi` or `rocm-smi` where those exist. They do not
exist for integrated graphics — which is what most laptops actually run
llama.cpp on — so on Windows the readings come from the same performance
counters Task Manager uses, covering Intel and AMD integrated parts, and on
Linux from `/sys/class/drm`. Sampling happens on a background thread: reading
those counters spawns a process and takes the better part of a second, which is
not something to make the interface wait for.

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
- **Port held and answering** — offer to use that server as it is. A model cannot
  be loaded into a server Kestrel did not start, so this is presented as a
  choice rather than done silently.
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

## 18a. Updates

**Settings → Updates** compares the version published at
`github.com/dansdesigns/kestrel/version.txt` with the one installed and reports
the difference. It does not update itself: this is a local-first tool that people
modify, and replacing their files from the network without asking would be a poor
trade for saving them a `git pull`. Settings, memory and conversations live
outside the program folder, so re-running the installer over a fresh copy keeps
them.

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
