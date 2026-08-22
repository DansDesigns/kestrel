"""Checklist tools.

Two levels: stages, and sub-steps within a stage. Checking the project's files
is one stage however many things are looked at; writing each file is a stage of
its own. That shape matches how the work actually divides, and it gives a small
model somewhere to record progress without inventing new top-level steps for
every action it takes.
"""
from __future__ import annotations

from . import DANGER_SAFE, Param, Tool, ToolResult
from ..todo import DONE, STATUSES


def register(reg, provider) -> None:
    """`provider` returns the live TodoList."""

    GREETINGS = {"hi", "hello", "hey", "yo", "morning", "afternoon", "evening",
                 "thanks", "thank you", "ok", "okay", "sure", "cheers", "test"}

    def plan(steps: str, title: str = "", replace: bool = False) -> ToolResult:
        tl = provider()
        # A plan for a greeting is not a plan, and the model that writes one
        # then works through it — reading files and hunting skills for a task
        # nobody set.
        asked = " ".join(str(title or "").lower().split()).strip(" .!?")
        if asked in GREETINGS:
            return ToolResult(
                f'"{title}" does not need a plan. Answer it and call finish.',
                ok=False)
        # A plan already under way is not replaced on a whim. Models rewrite it
        # mid-task while reasoning, which throws away every step already closed
        # and leaves the work looking untouched.
        if tl.items and not tl.complete and not replace:
            done, total = tl.progress
            return ToolResult(
                f"There is already a plan, {done}/{total} done. Do not replace "
                "it — add to it with plan_add, or update a step with todo. "
                "Pass replace=true only if the task itself has changed.\n"
                + tl.render(), ok=False)
        tl.set_plan(steps, title)
        if not tl.items:
            return ToolResult("That plan had no steps in it. Give one step per line.",
                              ok=False)
        return ToolResult(f"Plan set ({len(tl.items)} stages). Start with step 1.\n"
                          + tl.render())

    def todo(id: str, status: str = "done", note: str = "") -> ToolResult:
        tl = provider()
        item = tl.by_label(str(id))
        if item is None:
            return ToolResult(f"No step {id}. Current plan:\n"
                              + (tl.render() or "(empty)"), ok=False)
        tl.update(item.id, status, note)
        done, total = tl.progress
        label = tl.label_for(item)
        stage = tl.stage_of(item)
        extra = ""
        if stage is not None and stage is not item and stage.status == DONE:
            extra = f" Stage {tl.label_for(stage)} is now complete."
        tail = "\nEverything is closed — call finish." if tl.complete else ""
        return ToolResult(f"Step {label} -> {item.status} "
                          f"({done}/{total} done).{extra}{tail}")

    def plan_read() -> ToolResult:
        tl = provider()
        if not tl.items:
            return ToolResult("There is no plan yet.")
        return ToolResult(tl.render())

    def plan_add(text: str, under: str = "", new_stage: bool = False) -> ToolResult:
        tl = provider()
        parent = 0
        if under:
            stage = tl.by_label(str(under))
            if stage is None:
                return ToolResult(f"No stage {under}. Current plan:\n" + tl.render(),
                                  ok=False)
            parent = stage.parent or stage.id
        elif not new_stage:
            # Default to a sub-step of the stage in flight. Without this every
            # note about what just happened becomes a new stage, and a plan of
            # five turns into a list of twenty-one — which is a log, not a plan.
            current = tl.current
            stage = tl.stage_of(current) if current else None
            parent = stage.id if stage else 0
        item = tl.add(text, parent=parent)
        where = "as a new stage" if not item.parent else f"under {tl.label_for(stage)}"
        return ToolResult(f"Added {tl.label_for(item)} {where}: {item.text}")

    reg.add(Tool("plan", "Write the checklist for this task, one stage per line.",
                 [Param("steps", "string", "Stages, one per line.", required=True),
                  Param("title", "string", "Short name for the task."),
                  Param("replace", "boolean",
                        "Only to discard an existing plan when the task changed.")],
                 plan, DANGER_SAFE,
                 detail="Do this once, before starting. Three to seven stages is "
                        "usually right. A stage is a piece of work with an end: "
                        "checking the project's files is one stage; writing each "
                        "file is a stage. Record what happens inside a stage with "
                        "plan_add(under=...) rather than adding more stages."))
    reg.add(Tool("todo", "Update a step as you go.",
                 [Param("id", "string", "Step label, e.g. 2 or 2a.", required=True),
                  Param("status", "string", f"One of: {', '.join(STATUSES)}.",
                        default="done"),
                  Param("note", "string", "Short note, e.g. why it is blocked.")],
                 todo, DANGER_SAFE,
                 detail="Mark a step doing when you start it and done when you "
                        "have checked it works. Closing every sub-step closes "
                        "its stage automatically."))
    reg.add(Tool("plan_read", "Read the whole checklist.",
                 [], plan_read, DANGER_SAFE,
                 detail="Only a one-line summary is in your prompt. Read this "
                        "when you need the detail — before starting a stage, or "
                        "when picking up work you did not begin."))
    reg.add(Tool("plan_add", "Add a step: a sub-step of what you are doing, or "
                             "a new stage.",
                 [Param("text", "string", "The new step.", required=True),
                  Param("under", "string",
                        "Stage label to put it under, e.g. 2. Defaults to the "
                        "stage you are on."),
                  Param("new_stage", "boolean",
                        "True only for work the plan is genuinely missing.")],
                 plan_add, DANGER_SAFE,
                 detail="Notes about what you are doing become sub-steps of the "
                        "stage you are on — 2a, 2b and so on. A new stage is for "
                        "work nobody anticipated, not for each action you take."))
