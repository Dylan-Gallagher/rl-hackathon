"""CTFEnv protocol (README section 3.3) and the shared backend-agnostic engine.

Concrete backends (Docker CLI, Daytona SDK, REPL subprocess) only implement
sandbox lifecycle primitives; flag injection, output capping, timeouts, step
limits, flag scanning/verification and idempotent close live here.
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from ctf_gym.contracts import Obs, Task, _check_relpath
from ctf_gym.verifier import (
    DEFAULT_HEAD_BYTES,
    DEFAULT_TAIL_BYTES,
    VerifierRecord,
    cap_output,
    generate_flag,
    solved_by_scan,
)

log = logging.getLogger(__name__)

STEP_TIMEOUT_S = 120
PYTHON_HEREDOC = "python3 - <<'CTFGYM_EOF'\n%s\nCTFGYM_EOF"


@dataclass
class EpisodeContext:
    run_id: str
    task_id: str
    episode_id: str
    seed: int


class CTFEnv(ABC):
    """Minimal env protocol from the shared contracts."""

    async def reset(self, seed: Optional[int] = None) -> Obs: ...
    async def step(self, action: str) -> Obs: ...
    def solved(self) -> bool: ...
    async def close(self) -> None: ...


class SandboxError(RuntimeError):
    pass


class BaseCTFEnv(CTFEnv, ABC):
    """Backend-agnostic episode engine.

    Security/robustness guarantees implemented here:
      - fresh sandbox per episode, labeled run_id/task_id/episode_id
      - default-deny egress at backend level (docker `--network none`,
        Daytona `network_block_all=True`)
      - generated per-episode flag injected via safe file-placeholder
        replacement (unsafe env-var mode must be explicitly enabled)
      - capped head+tail tool output
      - per-action and per-episode timeouts
      - flag verification/scanning outside the sandbox
      - idempotent close
    """

    def __init__(
        self,
        task: Task,
        verifier: VerifierRecord,
        *,
        run_id: str,
        episode_id: str,
        seed: Optional[int] = None,
        assets_dir: Optional[str] = None,
        head_bytes: int = DEFAULT_HEAD_BYTES,
        tail_bytes: int = DEFAULT_TAIL_BYTES,
        step_timeout_s: float = STEP_TIMEOUT_S,
        allow_env_flag: bool = False,
    ) -> None:
        task.validate()
        self.task = task
        self.verifier = verifier
        self.ctx = EpisodeContext(
            run_id=run_id, task_id=task.task_id, episode_id=episode_id,
            seed=seed if seed is not None else 0,
        )
        self.assets_dir = assets_dir
        self.head_bytes = head_bytes
        self.tail_bytes = tail_bytes
        self.step_timeout_s = float(step_timeout_s)
        self.allow_env_flag = allow_env_flag
        self._flag: Optional[str] = None
        self._solved = False
        self._verified_flag: Optional[str] = None
        self._observations: list[str] = []
        self._step_count = 0
        self._closed = False
        self._started = False
        self.sandbox_id: Optional[str] = None

    # ---- backend primitives -------------------------------------------------
    @abstractmethod
    async def _start_sandbox(self, env_vars: dict[str, str]) -> str:
        """Start a fresh, egress-locked sandbox; return its id."""

    @abstractmethod
    async def _exec(self, command: str, timeout: Optional[float] = None) -> tuple[int, str]:
        """Run command in sandbox; return (exit_code, output)."""

    @abstractmethod
    async def _stop_sandbox(self) -> None:
        """Destroy the sandbox (best-effort; must be idempotent)."""

    async def _upload(self, local_path: str, dst: str) -> None:
        """Upload a file into the sandbox at relative dst. Optional."""
        raise SandboxError(f"{type(self).__name__} does not support asset upload")

    # ---- helpers for backends ----------------------------------------------
    @staticmethod
    def safe_dst(dst: str, base: str = "/root/challenge") -> str:
        """Defend against path traversal: dst must be relative and clean."""
        _check_relpath(dst)
        if "\x00" in dst:
            raise SandboxError("null byte in destination path")
        parts = [p for p in dst.replace("\\", "/").split("/") if p not in ("", ".")]
        return "/".join([base.rstrip("/")] + parts)

    def wrap_action(self, action: str) -> str:
        if self.task.env.launch == "repl":
            return PYTHON_HEREDOC % action.rstrip("\n")
        return action

    # ---- lifecycle ----------------------------------------------------------
    async def reset(self, seed: Optional[int] = None) -> Obs:
        if seed is not None:
            self.ctx.seed = seed
        if self._started:
            await self.close()
        env_vars: dict[str, str] = {}
        if self.task.flag.mode == "generated":
            self._flag = generate_flag(self.task.flag.format)
        else:
            self._flag = self.verifier.expected_flag()
        if self.task.flag.injection.mode == "env":
            if not self.allow_env_flag:
                raise SandboxError(
                    "env-var flag injection is unsafe (agent can read `env`); "
                    "pass allow_env_flag=True to opt in explicitly"
                )
            env_vars[self.task.flag.injection.var] = self._flag
        self.sandbox_id = await self._start_sandbox(env_vars)
        self._started = True
        self._solved = False
        self._verified_flag = None
        self._observations = []
        self._step_count = 0
        # upload assets, injecting the flag into the designated placeholder file
        if self.assets_dir:
            for asset in self.task.assets:
                await self._upload_asset(asset)
        intro = (
            f"Sandbox {self.sandbox_id} ready. Challenge files are under /root/challenge. "
            f"Max steps: {self.task.horizon.max_steps}."
        )
        return Obs(step=0, content=intro, sandbox_id=self.sandbox_id)

    async def _upload_asset(self, rel: str) -> None:
        import os

        src = os.path.join(self.assets_dir, rel)  # type: ignore[arg-type]
        if not os.path.isfile(src):
            raise SandboxError(f"missing asset {rel!r} in {self.assets_dir!r}")
        inj = self.task.flag.injection
        if inj.mode == "file" and rel == inj.path:
            with open(src, "r", encoding="utf-8", errors="surrogateescape") as f:
                data = f.read()
            if inj.placeholder not in data:
                raise SandboxError(f"placeholder {inj.placeholder!r} missing in asset {rel!r}")
            import tempfile

            with tempfile.NamedTemporaryFile("w", suffix=".flagged", delete=False,
                                             encoding="utf-8", errors="surrogateescape") as tf:
                tf.write(data.replace(inj.placeholder, self._flag or ""))
                tmp = tf.name
            try:
                await self._upload(tmp, rel)
            finally:
                os.unlink(tmp)
        else:
            await self._upload(src, rel)

    async def step(self, action: str) -> Obs:
        if not self._started:
            raise SandboxError("call reset() before step()")
        if self._step_count >= self.task.horizon.max_steps:
            return Obs(step=self._step_count, content="step limit reached", done=True,
                       sandbox_id=self.sandbox_id)
        if self._solved:
            return Obs(step=self._step_count, content="episode already solved", done=True,
                       sandbox_id=self.sandbox_id)
        self._step_count += 1
        command = self.wrap_action(action)
        try:
            exit_code, output = await asyncio.wait_for(
                self._exec(command, timeout=self.step_timeout_s), timeout=self.step_timeout_s + 5.0
            )
        except asyncio.TimeoutError:
            return Obs(step=self._step_count,
                       content=f"action timed out after {self.step_timeout_s}s",
                       exit_code=None, sandbox_id=self.sandbox_id)
        capped, truncated = cap_output(output, self.head_bytes, self.tail_bytes)
        self._observations.append(capped)
        found = solved_by_scan([capped], self.verifier, self._flag if self.task.flag.mode == "generated" else None)
        if found:
            self._solved = True
            self._verified_flag = found
        return Obs(step=self._step_count, content=capped, exit_code=exit_code,
                   truncated=truncated, sandbox_id=self.sandbox_id,
                   done=self._solved or self._step_count >= self.task.horizon.max_steps)

    def submit_flag(self, candidate: str) -> bool:
        """Explicit external flag submission (agent declares a flag)."""
        from ctf_gym.verifier import verify_flag

        ok = verify_flag(candidate, self.verifier,
                         self._flag if self.task.flag.mode == "generated" else None)
        if ok:
            self._solved = True
            self._verified_flag = candidate
        return ok

    def solved(self) -> bool:
        return self._solved

    @property
    def verified_flag(self) -> Optional[str]:
        return self._verified_flag

    @property
    def observations(self) -> Sequence[str]:
        return list(self._observations)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._started:
            try:
                await self._stop_sandbox()
            except Exception as e:  # close must be best-effort and idempotent
                log.warning("sandbox teardown failed for %s: %s", self.sandbox_id, e)
            finally:
                self._started = False


COMPOSE_WARNING = (
    "compose/DinD launches run a docker daemon inside the sandbox; they are a "
    "compatibility path only and are NOT suitable for strict anti-cheat locked "
    "evaluation unless the challenge is flattened into a single image."
)


async def compose_up_in_sandbox(exec_fn, compose_file: str = "docker-compose.yml") -> tuple[int, str]:
    """Best-effort `docker compose up` inside a sandbox (compat path).

    ``exec_fn`` is an async callable(command) -> (exit_code, output), e.g. a
    Daytona sandbox's process.exec. See COMPOSE_WARNING.
    """
    log.warning(COMPOSE_WARNING)
    return await exec_fn(f"cd /root/challenge && docker compose -f {compose_file} up -d")
