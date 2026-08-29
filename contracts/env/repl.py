"""ReplCTFEnv — subprocess python REPL for random-crypto style tasks.

No docker required: python snippets run via ``python3 -I -c <code>`` in a
throwaway subprocess with a per-episode temp cwd; the flag is injected as a
generated task file in that directory; ``ls``/``cat`` are simulated over it.

SECURITY NOTE: this is NOT a security boundary. README §1.5 egress/anti-cheat
rules still apply at infrastructure level (Daytona network limits). Only run
against trusted task definitions.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from contracts.env.base import CTFEnv, Obs, capped_output
from contracts.flag import DEFAULT_FLAG_PATTERN, new_flag, scan_for_flags, seeded_flag

if TYPE_CHECKING:  # pragma: no cover - typing only
    from contracts.task import Task

_LS_LINE = "challenge.py  flag.txt"


class ReplCTFEnv(CTFEnv):
    """Local REPL environment for ``launch == 'repl'`` tasks.

    Actions:
    - ``python <code>`` or ``python3 <code>`` → run ``python3 -I -c <code>``
      in the episode temp dir, capture stdout+stderr, cap output.
    - ``ls`` → simulated listing of the episode dir.
    - ``cat flag.txt`` / ``cat challenge.py`` → file contents.
    - ``submit <flag>`` → explicit verification (outside the "sandbox").

    Deterministic flags: ``reset(seed)`` uses ``random.Random(seed)``-derived
    uuids when ``task.flag.mode == 'generated'`` so tests are reproducible.
    """

    SNIPPET_TIMEOUT_S = 30
    MAX_OUTPUT_CHARS = 4000

    def __init__(self, task: "Task", python_bin: str = "python3",
                 snippet_timeout_s: float = SNIPPET_TIMEOUT_S):
        self.task = task
        self.python_bin = python_bin
        self.snippet_timeout_s = snippet_timeout_s
        self._dir: Path | None = None
        self._flag: str | None = None
        self._outputs: list[str] = []
        self._steps = 0
        self._solved = False

    async def reset(self, seed: int | None = None) -> Obs:
        """Create a fresh temp dir, write challenge.py + injected flag.txt."""
        self._dir = Path(tempfile.mkdtemp(prefix="replctf-"))
        if self.task.flag.mode == "generated":
            self._flag = seeded_flag(seed, self.task) if seed is not None else new_flag(self.task)
        else:
            self._flag = self.task.flag.format
        (self._dir / "flag.txt").write_text(self._flag + "\n", encoding="utf-8")
        (self._dir / "challenge.py").write_text(
            "# challenge (see task prompt)\n" + self.task.prompt + "\n",
            encoding="utf-8")
        self._outputs = []
        self._steps = 0
        self._solved = False
        obs = Obs(output=_LS_LINE, done=False, step=0,
                  metadata={"sandbox_id": self._dir.name})
        self._outputs.append(_LS_LINE)
        return obs

    async def step(self, action: str) -> Obs:
        if self._dir is None:
            raise RuntimeError("call reset() before step()")
        self._steps += 1
        cmd = action.strip()
        if cmd == "ls":
            out = _LS_LINE
        elif cmd in ("cat flag.txt", "cat challenge.py"):
            out = (self._dir / cmd.split()[1]).read_text(encoding="utf-8")
        elif cmd.startswith("submit "):
            candidate = cmd[len("submit "):].strip()
            self._solved = self._solved or (candidate == self._flag)
            out = "Correct!" if self._solved else "Wrong flag."
        elif cmd.startswith(("python ", "python3 ")):
            code = cmd.split(" ", 1)[1]
            out = await self._run_snippet(code)
        else:
            out = f"sh: {cmd.split()[0]}: command not found"
        out = capped_output(out, self.MAX_OUTPUT_CHARS)
        self._outputs.append(out)
        return Obs(output=out, done=self._solved, step=self._steps,
                   metadata={"exit_code": 0})

    async def _run_snippet(self, code: str) -> str:
        """Run ``python3 -I -c <code>`` in the episode dir with timeout + cap."""
        proc = await asyncio.create_subprocess_exec(
            self.python_bin, "-I", "-c", code,
            cwd=str(self._dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self._dir)},
        )
        try:
            raw, _ = await asyncio.wait_for(proc.communicate(), timeout=self.snippet_timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            return f"TimeoutError: snippet exceeded {self.snippet_timeout_s}s"
        return raw.decode("utf-8", errors="replace")

    def solved(self) -> bool:
        """Regex scan of accumulated outputs for the injected flag."""
        if self._solved:
            return True
        if self._flag is None:
            return False
        pattern = (DEFAULT_FLAG_PATTERN if self.task.flag.mode == "generated"
                   else None)
        if pattern is None:
            return any(self._flag in o for o in self._outputs)
        return any(self._flag in scan_for_flags(o, pattern)
                   for o in self._outputs)

    async def close(self) -> None:
        """Best-effort temp dir removal."""
        if self._dir is not None:
            import shutil
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None
