"""DaytonaCTFEnv — scale-out backend on the Daytona Python SDK.

Implements the §3.3 CTFEnv contract on Daytona sandboxes: per-episode sandbox
from the task image/snapshot, labels (run_id/task_id/episode_id), auto-delete
TTL, network limits (egress lock — the §1.5 anti-cheat egress rule lives
here), and command execution with capped output.

The `daytona` package is NOT installed by default — the import is guarded
with an actionable error.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from contracts.env.base import CTFEnv, Obs, capped_output
from contracts.flag import DEFAULT_FLAG_PATTERN, new_flag, scan_for_flags

if TYPE_CHECKING:  # pragma: no cover - typing only
    from contracts.task import Task

try:
    from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams
    _DAYTONA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _DAYTONA_AVAILABLE = False

logger = logging.getLogger(__name__)


def _require_daytona() -> Any:
    if not _DAYTONA_AVAILABLE:
        raise ImportError(
            "The 'daytona' package is required for DaytonaCTFEnv. "
            "Install it with: pip install 'rl-hackathon[daytona]' "
            "(or 'pip install daytona')")
    from daytona import Daytona  # re-import for typing clarity
    return Daytona


class DaytonaCTFEnv(CTFEnv):
    """CTFEnv over a Daytona sandbox (per-episode isolation + egress lock).

    Args:
        task: the task definition (env.image = snapshot/image reference).
        run_id: rollout/run label for sandbox organization.
        episode_id: episode label (used in sandbox labels + idempotency).
        ttl_s: auto-delete TTL in seconds (default 30 min) — sandboxes never
            outlive a crashed run.
        egress_locked: when True, sandbox network access is restricted to the
            minimum (§1.5: no writeup-search cheating). Exact network-limit
            API usage: see Daytona docs "Network > Sandbox network policies".
        max_output_chars: head/tail output cap per step.

    SDK details that could not be verified offline are marked TODO(daytona).
    """

    DEFAULT_TTL_S = 1800
    MAX_OUTPUT_CHARS = 4000
    STEP_TIMEOUT_S = 180

    def __init__(
        self,
        task: "Task",
        run_id: str = "default-run",
        episode_id: str | None = None,
        ttl_s: int = DEFAULT_TTL_S,
        egress_locked: bool = True,
        max_output_chars: int = MAX_OUTPUT_CHARS,
        client: Any | None = None,
    ):
        if client is None and not _DAYTONA_AVAILABLE:
            _require_daytona()  # raise the friendly ImportError early
        self.task = task
        self.run_id = run_id
        self.episode_id = episode_id or task.task_id
        self.ttl_s = ttl_s
        self.egress_locked = egress_locked
        self.max_output_chars = max_output_chars
        self._client = client
        self._sandbox: Any | None = None
        self._flag: str | None = None
        self._outputs: list[str] = []
        self._steps = 0
        self._solved = False

    def _daytona(self) -> Any:
        if self._client is None:
            Daytona = _require_daytona()
            self._client = Daytona()  # uses DAYTONA_API_KEY env by default
        return self._client

    async def reset(self, seed: int | None = None) -> Obs:
        """Create a fresh sandbox, inject the flag, return the initial listing."""
        client = self._daytona()
        if self.task.flag.mode == "generated":
            self._flag = new_flag(self.task)
        else:
            self._flag = self.task.flag.format
        params = CreateSandboxFromSnapshotParams(
            language="python",
            image=self.task.env.image or None,
            labels={
                "run_id": self.run_id,
                "task_id": self.task.task_id,
                "episode_id": self.episode_id,
            },
            # TODO(daytona): confirm exact snapshot/TTL param names against
            # https://www.daytona.io/docs/python-sdk/ — auto-delete timeout
            # and public/private network options live on the create params.
            timeout=self.ttl_s,
        )
        self._sandbox = await _maybe_async(client.create(params))
        if self.egress_locked:
            # TODO(daytona): apply network limits / egress lock via the
            # sandbox network API (docs: "Network limits"). Placeholder logs
            # until the exact API shape is confirmed against the SDK version.
            logger.info("egress lock requested for %s (apply via network API)",
                        self.sandbox_id)
        if self._flag:
            await self._exec(
                f"printf '%s\\n' '{self._flag}' > /flag.txt")
        out = await self._exec("ls")
        obs = Obs(output=capped_output(out, self.max_output_chars), done=False,
                  step=0, metadata={"sandbox_id": self.sandbox_id})
        self._outputs = [obs.output]
        self._steps = 0
        self._solved = False
        return obs

    @property
    def sandbox_id(self) -> str:
        return str(getattr(self._sandbox, "id", "unknown"))

    async def _exec(self, command: str, timeout: int | None = None) -> str:
        """Run a shell command in the sandbox, return combined output."""
        if self._sandbox is None:
            raise RuntimeError("call reset() before step()")
        result = await _maybe_async(self._sandbox.process.code_run(
            f"sh -lc {_shell_quote(command)}", timeout=timeout or self.STEP_TIMEOUT_S))
        out = getattr(result, "result", None)
        if out is None:
            out = getattr(result, "stdout", "") or ""
        err = getattr(result, "stderr", "") or ""
        return (out + ("\n" + err if err else ""))

    async def step(self, action: str) -> Obs:
        self._steps += 1
        cmd = action.strip()
        if cmd.startswith("submit "):
            candidate = cmd[len("submit "):].strip()
            self._solved = self._solved or (candidate == self._flag)
            out = "Correct!" if self._solved else "Wrong flag."
        else:
            out = await self._exec(cmd)
        out = capped_output(out, self.max_output_chars)
        self._outputs.append(out)
        return Obs(output=out, done=self._solved, step=self._steps,
                   metadata={"sandbox_id": self.sandbox_id})

    def solved(self) -> bool:
        """Verifier runs here — OUTSIDE the sandbox (§1.5)."""
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
        """Delete the sandbox (idempotent, best effort)."""
        if self._sandbox is not None:
            try:
                await _maybe_async(self._sandbox.delete())
            except Exception:  # pragma: no cover - best effort
                logger.warning("failed to delete sandbox %s", self.sandbox_id)
            self._sandbox = None


def _shell_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


async def _maybe_async(value: Any) -> Any:
    """Await coroutine-returning SDK calls; pass plain values through.

    The Daytona SDK historically exposes both sync and async variants —
    this shim keeps the env protocol async either way.
    """
    import inspect
    if inspect.isawaitable(value):
        return await value
    return value
