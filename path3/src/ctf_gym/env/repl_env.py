"""Local REPL backend for random-crypto style tasks (no docker needed).

Runs python actions in an isolated `python3 -I` subprocess (isolated mode,
cwd inside a per-episode scratch dir, no shell). This is a dev/CI convenience
backend: it is NOT a strong security boundary (no namespace isolation), so use
it only for trusted, procedurally generated tasks like Random-Crypto.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from typing import Optional

from ctf_gym.env.base import BaseCTFEnv, SandboxError


class ReplEnv(BaseCTFEnv):
    def __init__(self, *args, python_binary: str = "python3", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.python_binary = python_binary
        self._scratch: Optional[str] = None

    async def _start_sandbox(self, env_vars: dict[str, str]) -> str:
        self._scratch = tempfile.mkdtemp(prefix=f"ctfgym-{self.ctx.episode_id}-")
        for k, v in env_vars.items():
            # not used in safe mode; env injection raises earlier unless opted in
            os.environ[k] = v
        return f"local:{os.path.basename(self._scratch)}"

    async def _exec(self, command: str, timeout: Optional[float] = None) -> tuple[int, str]:
        # BaseCTFEnv wraps repl actions in a heredoc; strip it and run directly.
        code = command
        marker = "python3 - <<'CTFGYM_EOF'\n"
        if command.startswith(marker) and command.endswith("\nCTFGYM_EOF"):
            code = command[len(marker):-len("\nCTFGYM_EOF")]
        proc = await asyncio.create_subprocess_exec(
            self.python_binary, "-I", "-c", code,
            cwd=self._scratch,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout or 120.0)
        except asyncio.TimeoutError:
            proc.kill()
            return 124, f"python action timed out after {timeout}s"
        return proc.returncode or 0, out.decode("utf-8", errors="replace")

    async def _upload(self, local_path: str, dst: str) -> None:
        target = self.safe_dst(dst, base=self._scratch or tempfile.gettempdir())
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(local_path, target)

    async def _stop_sandbox(self) -> None:
        if self._scratch and os.path.isdir(self._scratch):
            shutil.rmtree(self._scratch, ignore_errors=True)
        self._scratch = None
