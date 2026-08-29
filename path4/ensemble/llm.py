"""LLM clients for the Path 4 ensemble layer.

Two clients with the same call interface:

- :class:`ChatClient` — raw httpx against any OpenAI-compatible
  ``/v1/chat/completions`` endpoint (LiteLLM proxy per ``contracts/models.yaml``
  is the intended server). Retries 429/5xx with exponential backoff.
- :class:`MockChatClient` — deterministic scripted responses, the offline path
  for tests and the CLI ``--mock`` demo.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any, Sequence

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:4000/v1"

#: Backoff (seconds) for retry attempts: base * 2**attempt +- jitter.
_RETRY_BASE_DELAY = 1.0
_RETRY_JITTER = 0.25


class ChatClient:
    """Minimal async OpenAI-compatible chat client (httpx, no SDK dep)."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "alloy",
        timeout: float = 120.0,
        retries: int = 3,
        transport: Any | None = None,
    ):
        self.transport = transport  # injectable for tests (httpx.MockTransport)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "sk-no-key"
        self.model = model
        self.timeout = timeout
        self.retries = retries

    async def chat(self, messages: Sequence[dict[str, str]], model: str | None = None) -> dict[str, Any]:
        """Send a chat completion; return ``{content, model, tokens_in, tokens_out}``.

        ``model`` overrides the client default per call (per-turn alloy routing).
        The returned ``model`` is the SERVED model name when the provider echoes
        one (LiteLLM does), else the requested name.
        """
        requested = model or self.model
        payload = {"model": requested, "messages": list(messages)}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        delay = _RETRY_BASE_DELAY
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.retries + 1):
                try:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                except httpx.HTTPError as e:  # network errors -> retry
                    last_err = e
                else:
                    if resp.status_code == 200:
                        return self._parse(resp.json(), requested)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        last_err = httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}", request=resp.request, response=resp
                        )
                    else:
                        resp.raise_for_status()  # 4xx other than 429: fail fast
                if attempt == self.retries:
                    break
                await asyncio.sleep(delay + random.uniform(0, _RETRY_JITTER))
                delay *= 2
        assert last_err is not None
        raise last_err

    @staticmethod
    def _parse(data: dict[str, Any], requested: str) -> dict[str, Any]:
        choice = (data.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content")) or ""
        usage = data.get("usage") or {}
        served = data.get("model") or requested
        return {
            "content": content,
            "model": served,
            "tokens_in": int(usage.get("prompt_tokens") or 0),
            "tokens_out": int(usage.get("completion_tokens") or 0),
        }


#: Default mock script: solves the mock/repl env challenge (ls -> cat flag.txt).
DEFAULT_MOCK_SCRIPT = [
    "Let me start by listing the files.\n```bash\nls\n```",
    "There's a flag file; reading it.\n```bash\ncat flag.txt\n```",
    "Got it. FLAG: <flag>",
]


class MockChatClient:
    """Deterministic scripted chat client — offline path for tests/demo.

    Scripts are keyed by the requested model name (each racing policy may get
    its own script); ``default`` is used for models without a script. Within a
    script, responses are keyed by call index; once exhausted the LAST entry
    repeats forever (e.g. a stuck policy that keeps re-running ``ls``).

    Set ``CTF_MOCK_LLM=1`` (or pass this client explicitly) to run the agent
    loop fully offline.
    """

    def __init__(
        self,
        scripts: dict[str, Sequence[str]] | None = None,
        default: Sequence[str] | None = None,
    ):
        self.scripts = {k: list(v) for k, v in (scripts or {}).items()}
        self.default = list(default) if default is not None else list(DEFAULT_MOCK_SCRIPT)
        self.calls: list[tuple[str, str]] = []  # (requested_model, content) audit log

    def reset(self) -> None:
        """Restart all per-model script indices (per-episode reset).

        A single shared client (e.g. the CLI builds one for all tasks) must
        replay its script from the top on every episode, not run off the end.
        """
        self.calls.clear()

    async def chat(self, messages: Sequence[dict[str, str]], model: str | None = None) -> dict[str, Any]:
        requested = model or "mock"
        script = self.scripts.get(requested, self.default)
        idx = sum(1 for m, _ in self.calls if m == requested)
        content = script[min(idx, len(script) - 1)] if script else ""
        self.calls.append((requested, content))
        return {
            "content": content,
            "model": requested,
            "tokens_in": sum(len(m.get("content", "")) for m in messages) // 4,
            "tokens_out": len(content) // 4,
        }
