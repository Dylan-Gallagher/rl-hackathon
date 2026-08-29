"""Hardened ReplCTFEnv for Random-Crypto tasks (anti-cheat, README §1.5).

The stock ReplCTFEnv injects ``flag.txt`` containing the real flag, which makes
``cat flag.txt`` a 1-step "solve". For real traces we subclass it in our own
package (contracts are read-only) and:

- never write ``flag.txt`` (``ls`` shows only ``challenge.py``);
- verification is still exact-match against the static flag from the Task
  (``submit <flag>`` or the flag appearing in snippet stdout).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from contracts.env.base import Obs, capped_output
from contracts.env.repl import ReplCTFEnv

if TYPE_CHECKING:  # pragma: no cover - typing only
    from contracts.task import Task

_LS = "challenge.py"

_HEREDOC_RE = re.compile(r"^<<\s*['\"]?(\w+)['\"]?\s*\n", re.S)


def prep_python_code(code: str) -> str:
    """Normalize model-written snippets the stock repl chokes on.

    Accepts ``-c "<code>"``, bare quoted ``'<code>'``, and heredocs
    (``<< 'EOF'\n<code>\nEOF``) and returns the raw python to run.
    """
    code = code.strip()
    if code.startswith("-c "):
        code = code[3:].strip()
    m = _HEREDOC_RE.match(code)
    if m:
        marker = m.group(1)
        body = code[m.end():]
        lines = body.rstrip().splitlines()
        if lines and lines[-1].strip().strip("'\"") == marker:
            lines = lines[:-1]
        code = "\n".join(lines)
    if len(code) >= 2 and code[0] == code[-1] and code[0] in "\"'" and code.count(code[0]) == 2:
        code = code[1:-1]
    return code.strip()


class HardReplCTFEnv(ReplCTFEnv):
    """ReplCTFEnv without the flag file shortcut."""

    def __init__(self, task: "Task", python_bin: str = "python3",
                 snippet_timeout_s: float = ReplCTFEnv.SNIPPET_TIMEOUT_S):
        super().__init__(task, python_bin=python_bin, snippet_timeout_s=snippet_timeout_s)

    async def reset(self, seed: Optional[int] = None) -> Obs:
        self._dir = Path(tempfile.mkdtemp(prefix="rcenv-"))
        self._flag = self.task.flag.format  # static: verified, never exposed
        (self._dir / "challenge.py").write_text(
            "# challenge (see task prompt)\n" + self.task.prompt + "\n", encoding="utf-8")
        self._outputs = []
        self._steps = 0
        self._solved = False
        obs = Obs(output=_LS, done=False, step=0,
                  metadata={"sandbox_id": self._dir.name})
        self._outputs.append(_LS)
        return obs

    async def step(self, action: str) -> Obs:
        if self._dir is None:
            raise RuntimeError("call reset() before step()")
        self._steps += 1
        cmd = action.strip()
        if cmd == "ls":
            out = _LS
        elif cmd in ("cat flag.txt", "cat ~/flag.txt"):
            out = "cat: flag.txt: No such file or directory"
        elif cmd == "cat challenge.py":
            out = (self._dir / "challenge.py").read_text(encoding="utf-8")
        elif cmd.startswith("submit "):
            candidate = cmd[len("submit "):].strip()
            self._solved = self._solved or (candidate == self._flag)
            out = "Correct!" if self._solved else "Wrong flag."
        elif cmd.startswith(("python ", "python3 ")):
            out = await self._run_snippet(prep_python_code(cmd.split(" ", 1)[1]))
        else:
            out = f"sh: {cmd.split()[0]}: command not found"
        out = capped_output(out, self.MAX_OUTPUT_CHARS)
        self._outputs.append(out)
        return Obs(output=out, done=self._solved, step=self._steps,
                   metadata={"exit_code": 0})
