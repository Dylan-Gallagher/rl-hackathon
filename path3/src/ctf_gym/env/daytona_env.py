"""Daytona SDK backend (pinned to the Daytona 0.207 Python SDK API).

Verified against daytona-sdk 0.207.x signatures:
  - AsyncDaytona.create(params, timeout=...) with
    CreateSandboxFromImageParams / CreateSandboxFromSnapshotParams accepting
    env_vars, labels, ttl_minutes, network_block_all and image/snapshot.
  - client.delete(sandbox, timeout=..., wait=...)
  - AsyncSandbox.process.exec(command, cwd=None, env=None, timeout=None)
  - AsyncSandbox.fs.upload_file(src, dst)
  - sandbox.update_network_settings / sandbox.id / client.close()

The SDK is imported lazily; core package and tests run without it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from ctf_gym.env.base import BaseCTFEnv, SandboxError

log = logging.getLogger(__name__)

DAYTONA_SDK_VERSION = "0.207.*"
MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.5


def _import_daytona():
    try:
        from daytona_sdk import AsyncDaytona, DaytonaConfig  # noqa: F401
    except ImportError as e:
        raise SandboxError(
            f"daytona-sdk is required for the Daytona backend: pip install 'daytona-sdk=={DAYTONA_SDK_VERSION}' ({e})"
        ) from e
    return AsyncDaytona, DaytonaConfig


def _import_params():
    from daytona_sdk import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams

    return CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams


async def _retry_idempotent(fn, *, what: str, attempts: int = MAX_RETRIES):
    """Retry idempotent Daytona lifecycle operations with backoff."""
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            return await fn()
        except Exception as e:  # noqa: BLE001 - SDK raises varied errors
            last = e
            wait = RETRY_BACKOFF_S * (2 ** i)
            log.warning("%s failed (attempt %d/%d): %s; retrying in %.1fs",
                        what, i + 1, attempts, e, wait)
            await asyncio.sleep(wait)
    raise SandboxError(f"{what} failed after {attempts} attempts: {last}")


class DaytonaEnv(BaseCTFEnv):
    """One fresh Daytona sandbox per episode.

    ``image`` may be either a Docker image reference (image mode) or
    ``snapshot:<snapshot_id>`` (snapshot mode). Egress is denied with
    network_block_all=True; TTL ensures abandoned sandboxes self-delete.
    """

    def __init__(self, *args, api_key: Optional[str] = None, server_url: Optional[str] = None,
                 org_id: Optional[str] = None, ttl_minutes: int = 60, create_timeout: float = 300.0,
                 client: Any = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._api_key = api_key
        self._server_url = server_url
        self._org_id = org_id
        self.ttl_minutes = ttl_minutes
        self.create_timeout = create_timeout
        self._client = client  # injectable for tests
        self._owns_client = client is None
        self._sandbox: Any = None

    def _get_client(self):
        if self._client is None:
            AsyncDaytona, DaytonaConfig = _import_daytona()
            if not (self._api_key and self._server_url):
                raise SandboxError(
                    "Daytona backend needs DAYTONA_API_KEY and DAYTONA_SERVER_URL "
                    "(set env vars or pass api_key=/server_url=)"
                )
            self._client = AsyncDaytona(
                DaytonaConfig(api_key=self._api_key, server_url=self._server_url, org_id=self._org_id)
            )
        return self._client

    async def _start_sandbox(self, env_vars: dict[str, str]) -> str:
        client = self._get_client()
        labels = {
            "run_id": self.ctx.run_id,
            "task_id": self.ctx.task_id,
            "episode_id": self.ctx.episode_id,
        }
        image = self.task.env.image
        base = dict(
            env_vars=env_vars or None,
            labels=labels,
            ttl_minutes=self.ttl_minutes,
            network_block_all=True,  # default-deny egress
        )
        if image.startswith("snapshot:"):
            FromSnapshot, _ = _import_params()
            params = FromSnapshot(snapshot=image.split("snapshot:", 1)[1], **base)
        else:
            _, FromImage = _import_params()
            params = FromImage(image=image, **base)

        async def create():
            return await client.create(params, timeout=self.create_timeout)

        self._sandbox = await _retry_idempotent(create, what=f"daytona create {self.task.task_id}")
        return self._sandbox.id

    async def _exec(self, command: str, timeout: Optional[float] = None) -> tuple[int, str]:
        response = await self._sandbox.process.exec(command, cwd="/root/challenge", timeout=timeout)
        return response.exit_code, response.result or ""

    async def _upload(self, local_path: str, dst: str) -> None:
        target = self.safe_dst(dst)
        await self._sandbox.fs.upload_file(local_path, target)

    async def _stop_sandbox(self) -> None:
        if self._sandbox is None:
            return
        client = self._get_client()

        async def delete():
            return await client.delete(self._sandbox, timeout=60.0, wait=True)

        await _retry_idempotent(delete, what=f"daytona delete {self.sandbox_id}")
        self._sandbox = None
        if self._owns_client and self._client is not None:
            try:
                await self._client.close()
            except Exception as e:  # best-effort
                log.warning("daytona client close failed: %s", e)
            finally:
                self._client = None
