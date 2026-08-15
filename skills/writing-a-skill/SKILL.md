---
name: writing-a-skill
description: Create a new skill for Kestrel, or fix one that is being ignored. Use when the user asks to add, write, make or edit a skill, or wonders why a skill they wrote never triggers.
license: MIT
metadata:
  author: kestrel
  version: "1.0"
---

# Writing a skill

A skill is a folder with a `SKILL.md` in it. Only the name and description are
loaded into the prompt; the body is read when the skill is opened. That split is
the whole design, and it dictates how to write both parts.

## Where to put it

Write to the first folder Kestrel is watching — `skill_find` on any term names
the folders in use, and the Skills tab lists them. A folder dropped in is picked
up within a second; there is nothing to restart.

## The frontmatter is the part that matters

```
---
name: lowercase-hyphenated
description: What it does, and when to use it. Include the words someone would
  use when they need it.
---
```

The description is the only thing the model sees before deciding whether to open
the skill. Write it as a trigger, not a title:

- Poor: `Database helper`
- Good: `Query and migrate the project's SQLite database. Use when asked to
  inspect tables, add a column, or write a migration.`

Name and description are both required. A file without them is skipped, and the
Skills tab says why.

## The body

Written for a model that has just opened it mid-task and needs to act:

1. **When to use this** — the situations that should trigger it, and the near
   misses that should not.
2. **Steps** — numbered, with the exact commands and paths. Say how to tell each
   one worked.
3. **Notes** — the things that are easy to get wrong: a flag people forget, an
   error message and what it actually means, a value that must match elsewhere.

Keep it concrete. Anything the model could have guessed is wasted space; write
down what it could not.

## Bundled files

    skill-name/
      SKILL.md
      scripts/      run these with the shell tool
      references/   read these with read_file when the body says to
      assets/       templates and data

Put long material in `references/` and point at it from the body rather than
inlining it — the body is loaded whole, so length there costs context on the
turn it is opened.

## Steps to create one

1. `list_dir` the target skills folder to see what is already there and avoid a
   name clash.
2. `write_file` the folder's `SKILL.md` with frontmatter and body.
3. Add `scripts/` or `references/` only if there is something to put in them.
4. `skill_find` with a phrase from the description to confirm it is now
   discoverable. If it does not appear, the frontmatter is malformed.

## Why a skill gets ignored

Almost always the description. If it does not contain the words the user would
say, the model never opens it. Rewrite it around the request, not the subject.
