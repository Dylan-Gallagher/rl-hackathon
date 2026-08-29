"""Daytona sandbox wrapper for one challenge attempt.

Boots from the prebuilt solver snapshot, uploads the challenge's files into
/challenge, makes the flagCheck oracle executable, exposes a shell-exec tool,
and verifies a submitted flag against the stored sha256.
"""
from __future__ import annotations
import hashlib, os, posixpath, time
from pathlib import Path
import config
from daytona import Daytona, DaytonaConfig, CreateSandboxFromSnapshotParams

MAX_OUTPUT_CHARS = 6000

_client = None


def _get_client():
    """Lazily build the Daytona client so the module imports without creds set."""
    global _client
    if _client is None:
        _client = Daytona(DaytonaConfig(api_key=config.DAYTONA_API_KEY))
    return _client


class ChallengeSandbox:
    def __init__(self, manifest_entry: dict):
        self.m = manifest_entry
        self.sb = None

    def start(self, max_wait_tries: int = 12):
        # The account has tight total mem/disk caps and delete() frees space
        # asynchronously, so a create can transiently fail with "limit
        # exceeded" while sibling sandboxes are still tearing down. Treat that
        # as back-pressure and retry rather than failing the trajectory.
        last = None
        for i in range(max_wait_tries):
            try:
                self.sb = _get_client().create(
                    CreateSandboxFromSnapshotParams(
                        snapshot=config.SOLVER_SNAPSHOT,
                        ephemeral=True,
                        auto_stop_interval=15,
                        auto_delete_interval=0,
                    ),
                    timeout=300,
                )
                self._provision()
                return self
            except Exception as e:  # noqa: BLE001
                last = e
                if "limit exceeded" in str(e).lower():
                    time.sleep(15)
                    continue
                raise
        raise last

    def _provision(self):
        cdir = Path(self.m["dir"])
        # create any needed subdirectories under /challenge
        needed_dirs = sorted({posixpath.dirname(f) for f in self.m["upload_files"] if posixpath.dirname(f)})
        if needed_dirs:
            self.sb.process.exec("mkdir -p " + " ".join(f"/challenge/{d}" for d in needed_dirs), timeout=30)
        for rel in self.m["upload_files"]:
            local = cdir / rel
            if not local.exists():
                continue
            self.sb.fs.upload_file(local.read_bytes(), f"/challenge/{rel}")
        # make the flagCheck oracle (and any ELF) executable
        chk = self.m.get("flag_check_file")
        if chk:
            self.sb.process.exec(f"chmod +x '/challenge/{chk}' 2>/dev/null; true", timeout=30)
        self.sb.process.exec("chmod -R +rX /challenge 2>/dev/null; true", timeout=30)

    def exec(self, command: str, timeout: int | None = None) -> dict:
        timeout = timeout or config.COMMAND_TIMEOUT
        t0 = time.time()
        try:
            r = self.sb.process.exec(command, cwd="/challenge", timeout=timeout)
            out = r.result or ""
            exit_code = r.exit_code
        except Exception as e:  # noqa: BLE001
            out = f"[command error: {e}]"
            exit_code = -1
        truncated = False
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {len(out)} chars total]"
            truncated = True
        return {"output": out, "exit_code": exit_code, "duration": time.time() - t0, "truncated": truncated}

    def verify_flag(self, flag: str) -> bool:
        target = self.m["sha256_flag"].lower()
        candidates = {flag, flag.strip(), flag.strip().strip('"').strip("'")}
        for c in candidates:
            if hashlib.sha256(c.encode()).hexdigest().lower() == target:
                return True
        return False

    def stop(self):
        if self.sb is not None:
            for _ in range(3):
                try:
                    self.sb.delete()
                    break
                except Exception:
                    time.sleep(3)
            self.sb = None
