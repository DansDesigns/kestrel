---
name: working-in-tight-context
description: Strategies for completing tasks when the context window is small (under 16k tokens). Use when the profile is nano or small, when files are too large to read whole, or when a previous attempt ran out of room and got compacted.
license: MIT
metadata:
  author: kestrel
  version: "1.0"
---

# Working in tight context

A small window is a budget problem, not a capability problem. Spend it on the
few hundred tokens that actually decide the next action.

## Look before you read

Never open a file to find out whether it is relevant. Narrow first:

1. `find_files` to locate candidates by name.
2. `grep` for the symbol, string, or error you care about — it returns one line
   per hit with a line number.
3. `read_file` with `offset` set to that line number and `limit` around 60.

A grep plus a targeted read costs perhaps 400 tokens. Reading a 2,000-line file
costs 25,000 and will be truncated anyway.

## Let output go to disk

Long tool output is saved to a file automatically and only previewed inline. If
the preview is enough, move on. If it is not, `read_file` the saved path in
pages rather than re-running the command.

## Carry state in files, not in the conversation

The transcript is the first thing to be compacted away. Anything you will need
in ten steps belongs in a file:

- Write findings to `notes.md` as you go.
- Keep a short plan in `plan.md` and tick items off by editing it.
- Re-read either one when you need to reorient.

This survives compaction; a message twenty turns back does not.

## One step per turn

On a small window, batching several tool calls into one reply usually means the
generation gets cut off mid-JSON. Make one call, read the result, decide again.

## Say when you are stuck

If two different approaches have failed, stop and describe what you tried and
what you would need. That is a better answer than a third attempt that fills the
window with the same error.
