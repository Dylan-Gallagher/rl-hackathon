"""Daytona sandbox wrapper.

Each challenge attempt gets its own ephemeral sandbox created from the prebuilt
toolchain snapshot. The harness runs privileged setup as root (plant /flag,
install the challenge). The AGENT's bash tool runs as the unprivileged 'hacker'
user via runuser, so it cannot read the root-only /flag and must exploit the
challenge. Commands are passed base64-encoded to avoid any quoting issues.
"""
from __future__ import annotations
import base64
import os
import random
import time
from daytona import (Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams)

SNAPSHOT = "alloy-ctf-toolchain-v1"
AGENT_USER = "hacker"
CHALLENGE_DIR = "/challenge"


def client() -> Daytona:
    return Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))


class Sandbox:
    def __init__(self, daytona: Daytona, ttl_minutes: int = 20,
                 create_retries: int = 8):
        self.d = daytona
        # The account tier caps total mem/disk across running sandboxes, and
        # sandbox deletion is async, so concurrent create can transiently exceed
        # the cap. Retry with backoff+jitter until a slot frees up.
        last = None
        for i in range(create_retries):
            try:
                self._sb = daytona.create(
                    CreateSandboxFromSnapshotParams(snapshot=SNAPSHOT,
                                                    ttl_minutes=ttl_minutes),
                    timeout=180)
                return
            except Exception as e:  # noqa: BLE001
                last = e
                if "limit exceeded" in str(e).lower() or "quota" in str(e).lower():
                    time.sleep(min(30, 4 * (i + 1)) + random.uniform(0, 3))
                    continue
                raise
        raise last

    @property
    def id(self) -> str:
        return self._sb.id

    # ---- privileged (root) --------------------------------------------------
    def root_exec(self, command: str, cwd: str | None = None,
                  timeout: int = 120) -> tuple[int, str]:
        r = self._sb.process.exec(command, cwd=cwd, timeout=timeout)
        return r.exit_code, (r.result or "")

    def upload_bytes(self, data: bytes, dst: str) -> None:
        self._sb.fs.upload_file(data, dst)

    # ---- unprivileged (agent) ----------------------------------------------
    def agent_exec(self, command: str, timeout: int = 90) -> str:
        """Run a command as the non-root agent user inside CHALLENGE_DIR."""
        b64 = base64.b64encode(command.encode()).decode()
        wrapped = (f"echo {b64} | base64 -d | "
                   f"runuser -u {AGENT_USER} -- env TERM=xterm HOME=/home/{AGENT_USER} "
                   f"bash -lc 'cd {CHALLENGE_DIR} 2>/dev/null; bash -s'")
        try:
            r = self._sb.process.exec(wrapped, timeout=timeout)
            out = r.result or ""
            if r.exit_code != 0 and not out.strip():
                out = f"(command exited {r.exit_code} with no output)"
            return out
        except Exception as e:  # noqa: BLE001
            return f"[sandbox exec error/timeout] {e}"

    def destroy(self) -> None:
        try:
            self.d.delete(self._sb)
        except Exception:  # noqa: BLE001
            pass
