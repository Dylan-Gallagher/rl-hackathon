"""Docker CLI backend (local dev fallback). No native Docker SDK dependency.

Each episode runs `docker run -d --network none --label ...` and execs actions
with `docker exec`. Requires the `docker` binary on PATH.
"""

from __future__ import annotations

import asyncio
import shlex
from typing import Optional

from ctf_gym.env.base import BaseCTFEnv, SandboxError


class DockerEnv(BaseCTFEnv):
    def __init__(self, *args, container_prefix: str = "ctfgym", docker_binary: str = "docker", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.container_prefix = container_prefix
        self.docker_binary = docker_binary

    async def _run_cli(self, *args: str, timeout: float = 60.0) -> tuple[int, str]:
        proc = await asyncio.create_subprocess_exec(
            self.docker_binary, *args,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise SandboxError(f"docker CLI timed out: {args[:3]}")
        return proc.returncode or 0, out.decode("utf-8", errors="replace").strip()

    async def _start_sandbox(self, env_vars: dict[str, str]) -> str:
        name = f"{self.container_prefix}-{self.ctx.episode_id}"[:63].rstrip("-")
        args = [
            "run", "-d", "--rm",
            "--name", name,
            "--network", "none",  # default-deny egress
            "--label", f"ctfgym.run_id={self.ctx.run_id}",
            "--label", f"ctfgym.task_id={self.ctx.task_id}",
            "--label", f"ctfgym.episode_id={self.ctx.episode_id}",
        ]
        for k, v in env_vars.items():
            args += ["-e", f"{k}={v}"]
        args += [self.task.env.image, "sleep", "infinity"]
        rc, out = await self._run_cli(*args)
        if rc != 0:
            raise SandboxError(f"docker run failed: {out}")
        self._container_name = name
        return out.splitlines()[-1]

    async def _exec(self, command: str, timeout: Optional[float] = None) -> tuple[int, str]:
        # `timeout` handled by BaseCTFEnv.wait_for; docker exec has no native cap.
        # shlex-join the wrapper so heredocs/multiline actions survive exec argv.
        return await self._run_cli(
            "exec", self._container_name, "sh", "-c", command,
            timeout=(timeout or self.step_timeout_s) + 10.0,
        )

    async def _upload(self, local_path: str, dst: str) -> None:
        target = self.safe_dst(dst)
        await self._run_cli("exec", self._container_name, "mkdir", "-p",
                            shlex.quote("/".join(target.split("/")[:-1])))
        rc, out = await self._run_cli("cp", local_path, f"{self._container_name}:{target}", timeout=120.0)
        if rc != 0:
            raise SandboxError(f"docker cp failed: {out}")

    async def _stop_sandbox(self) -> None:
        await self._run_cli("rm", "-f", self._container_name, timeout=60.0)
