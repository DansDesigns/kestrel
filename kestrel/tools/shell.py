"""Shell tool. Gated by the approval policy in the registry."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import DANGER_EXEC, Param, Tool, ToolResult

MAX_CAPTURE = 200_000


def register(reg, workspace: Path) -> None:
    ws = Path(workspace)

    def run(command: str, timeout: int = 120) -> ToolResult:
        timeout = max(1, min(int(timeout), 1800))
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env["KESTREL_WORKSPACE"] = str(ws)
        try:
            proc = subprocess.run(
                command, shell=True, cwd=str(ws), env=env, timeout=timeout,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors="replace",
            )
        except subprocess.TimeoutExpired:
            return ToolResult(f"Command hit the {timeout}s timeout and was killed. "
                              "Run it in the background or raise the timeout.", ok=False)
        except OSError as e:
            return ToolResult(f"Could not start the command: {e}", ok=False)
        out = (proc.stdout or "")[:MAX_CAPTURE]
        status = "exit 0" if proc.returncode == 0 else f"exit {proc.returncode}"
        body = out.strip() or "(no output)"
        return ToolResult(f"[{status}]\n{body}", full=out, ok=proc.returncode == 0)

    shell_name = "cmd.exe" if sys.platform == "win32" else "sh"
    reg.add(Tool(
        "shell", f"Run a shell command in the workspace ({shell_name}).",
        [Param("command", "string", "The command line.", required=True),
         Param("timeout", "integer", "Seconds before it is killed.", default=120)],
        run, DANGER_EXEC,
        detail="stdout and stderr come back together. Long output is saved to a file "
               "and previewed; read the rest with read_file.",
    ))
