"""MockCTFEnv — in-process fake shell for tests and demos.

No subprocess, no docker, fully deterministic under ``seed``. Not a security
boundary of any kind; it exists so runners/tests can exercise the env protocol
end-to-end without infrastructure.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from contracts.env.base import CTFEnv, Obs, capped_output
from contracts.flag import DEFAULT_FLAG_PATTERN, new_flag, scan_for_flags, seeded_flag

if TYPE_CHECKING:  # pragma: no cover - typing only
    from contracts.task import Task

_LISTING = "flag.txt  challenge.py  README"


class MockCTFEnv(CTFEnv):
    """Fake shell implementing the §3.3 protocol.

    Supported commands (deterministic):
    - ``ls``            -> file listing
    - ``cat flag.txt``  -> prints the injected per-episode flag
    - ``cat README``    -> prints the task prompt
    - anything else     -> a plausible shell error

    ``solved()`` = regex scan of accumulated outputs + explicit submit via
    ``submit <flag>`` (verified outside the "sandbox", i.e. in this process).
    """

    def __init__(self, task: "Task", max_output_chars: int = 4000):
        self.task = task
        self.max_output_chars = max_output_chars
        self._flag: str | None = None
        self._outputs: list[str] = []
        self._steps = 0
        self._solved = False
        self._rng = random.Random()

    async def reset(self, seed: int | None = None) -> Obs:
        """Fresh episode: new deterministic RNG, inject a fresh flag if mode allows."""
        self._rng = random.Random(seed)
        if self.task.flag.mode == "generated":
            self._flag = seeded_flag(seed, self.task) if seed is not None else new_flag(self.task)
        else:
            self._flag = self.task.flag.format
        self._outputs = []
        self._steps = 0
        self._solved = False
        obs = Obs(output=_LISTING, done=False, step=0,
                  metadata={"sandbox_id": f"mock-{self.task.task_id}-{seed}"})
        self._outputs.append(_LISTING)
        return obs

    async def step(self, action: str) -> Obs:
        self._steps += 1
        cmd = action.strip()
        if cmd == "ls":
            out = _LISTING
        elif cmd == "cat flag.txt":
            out = self._flag or "cat: flag.txt: No such file or directory"
        elif cmd in ("cat README", "cat README.txt"):
            out = self.task.prompt
        elif cmd.startswith("submit "):
            candidate = cmd[len("submit "):].strip()
            self._solved = self._solved or (candidate == self._flag)
            out = "Correct!" if self._solved else "Wrong flag."
        else:
            out = f"sh: {cmd.split()[0]}: command not found"
        out = capped_output(out, self.max_output_chars)
        self._outputs.append(out)
        return Obs(output=out, done=self._solved, step=self._steps,
                   metadata={"exit_code": 0})

    def solved(self) -> bool:
        """Scan accumulated observations for the flag; explicit submits count too."""
        if self._solved:
            return True
        if self._flag is None:
            return False
        pattern = (DEFAULT_FLAG_PATTERN if self.task.flag.mode == "generated"
                   else None)
        if pattern is None:
            # static flags: scan for the literal flag string
            return any(self._flag in o for o in self._outputs)
        return any(self._flag in scan_for_flags(o, pattern)
                   for o in self._outputs)

    async def close(self) -> None:
        """Nothing to tear down for the mock."""
