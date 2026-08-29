"""DockerCTFEnv — local docker fallback backend (pure subprocess, no SDK).

Per-step command execution via ``docker exec`` against a detached container
started from the task image. Untestable in this environment (no docker
binary) — the Mock/Repl envs are the testable paths.

NOTE (§1.5): local docker gives NO egress lock; use DaytonaCTFEnv for
training/eval where the egress lock matters.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from typing import TYPE_CHECKING

from contracts.env.base import CTFEnv, Obs, capped_output
from contracts.flag import DEFAULT_FLAG_PATTERN, new_flag, scan_for_flags

if TYPE_CHECKING:  # pragma: no cover - typing only
    from contracts.task import Task

logger = logging.getLogger(__name__)


class DockerCTFEnv(CTFEnv):
    """CTFEnv over a local docker container.

    Lifecycle: ``reset`` → ``docker run -d --rm -i <image>``; ``step`` →
    ``docker exec <cid> sh -lc <action>`` (output capped); ``close`` →
    ``docker kill <cid>``. Raises RuntimeError with a clear message if the
    docker binary is missing.
    """

    STEP_TIMEOUT_S = 120
    MAX_OUTPUT_CHARS = 4000

    def __init__(self, task: "Task", docker_bin: str = "docker"):
        self.task = task
        self.docker_bin = docker_bin
        self._cid: str | None = None
        self._flag: str | None = None
        self._outputs: list[str] = []
        self._steps = 0
        self._solved = False

    def _require_docker(self) -> None:
        if shutil.which(self.docker_bin) is None:
            raise RuntimeError(
                f"docker binary '{self.docker_bin}' not found on PATH; "
                "install docker or use MockCTFEnv/ReplCTFEnv/DaytonaCTFEnv")

    async def _run(self, *args: str, timeout: float | None = None) -> tuple[int, str]:
        self._require_docker()
        proc = await asyncio.create_subprocess_exec(
            self.docker_bin, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
        try:
            raw, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout or 60)
        except asyncio.TimeoutError:
            proc.kill()
            return 124, "docker: command timed out"
        return proc.returncode or 0, raw.decode("utf-8", errors="replace")

    async def reset(self, seed: int | None = None) -> Obs:
        """Start a fresh container; inject a per-episode flag if mode allows."""
        if not self.task.env.image:
            raise RuntimeError(f"task {self.task.task_id} has no env image")
        if self.task.flag.mode == "generated":
            self._flag = new_flag(self.task)
            # Injection assumes /flag.txt; challenge images may override.
            rc, out = await self._run("run", "-d", "--rm", "-i",
                                      "-e", f"CTF_FLAG={self._flag}",
                                      self.task.env.image)
        else:
            self._flag = self.task.flag.format
            rc, out = await self._run("run", "-d", "--rm", "-i",
                                      self.task.env.image)
        if rc != 0:
            raise RuntimeError(f"docker run failed: {out}")
        self._cid = out.strip().splitlines()[-1].strip()
        if self._flag and self.task.flag.mode == "generated":
            await self._run("exec", self._cid, "sh", "-lc",
                            f"printf '%s\\n' '{self._flag}' > /flag.txt")
        rc, out = await self._run("exec", self._cid, "sh", "-lc", "ls")
        obs = Obs(output=capped_output(out, self.MAX_OUTPUT_CHARS), done=False,
                  step=0, metadata={"sandbox_id": self._cid})
        self._outputs = [obs.output]
        self._steps = 0
        self._solved = False
        return obs

    async def step(self, action: str) -> Obs:
        if self._cid is None:
            raise RuntimeError("call reset() before step()")
        self._steps += 1
        rc, out = await self._run("exec", self._cid, "sh", "-lc", action,
                                  timeout=self.STEP_TIMEOUT_S)
        out = capped_output(out, self.MAX_OUTPUT_CHARS)
        self._outputs.append(out)
        if action.strip().startswith("submit "):
            candidate = action.strip()[len("submit "):]
            self._solved = self._solved or (candidate == self._flag)
            if self._solved:
                out = "Correct!"
                self._outputs[-1] = out
        return Obs(output=out, done=self._solved, step=self._steps,
                   metadata={"exit_code": rc, "sandbox_id": self._cid})

    def solved(self) -> bool:
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
        """Kill the container (best effort)."""
        if self._cid is not None:
            try:
                await self._run("kill", self._cid, timeout=30)
            except Exception:  # pragma: no cover - best effort
                logger.warning("failed to kill container %s", self._cid)
            self._cid = None
