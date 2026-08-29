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
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        model_base_urls: dict[str, str] | None = None,
    ):
        self.transport = transport  # injectable for tests (httpx.MockTransport)
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "sk-no-key"
        self.model = model
        self.timeout = timeout
        self.retries = retries
        # Optional request extensions (per-call ``model`` may override base URL).
        self.max_tokens = max_tokens
        self.extra_body = dict(extra_body) if extra_body else None
        self.model_base_urls = {m: u.rstrip("/") for m, u in (model_base_urls or {}).items()}

    async def chat(self, messages: Sequence[dict[str, str]], model: str | None = None) -> dict[str, Any]:
        """Send a chat completion; return ``{content, model, tokens_in, tokens_out}``.

        ``model`` overrides the client default per call (per-turn alloy routing).
        The returned ``model`` is the SERVED model name when the provider echoes
        one (LiteLLM does), else the requested name.
        """
        requested = model or self.model
        payload = self._payload(requested, messages)
        base = self.model_base_urls.get(requested, self.base_url)

        def _bearer() -> str:
            key = self.api_key
            if callable(key):
                return str(key())
            return str(key)

        delay = _RETRY_BASE_DELAY
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
            url = f"{base}/chat/completions"
            empty_retry_used = False
            attempt = -1  # incremented BEFORE each retry gate; -1 -> first pass
            while attempt < self.retries:
                attempt += 1
                headers = {"Authorization": f"Bearer {_bearer()}"}
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                except httpx.HTTPError as e:  # network errors -> retry
                    last_err = e
                else:
                    if resp.status_code == 401:
                        inv = getattr(self.api_key, "invalidate", None)
                        if callable(inv):
                            inv()
                        last_err = httpx.HTTPStatusError(
                            "HTTP 401", request=resp.request, response=resp
                        )
                    elif resp.status_code == 200:
                        parsed = self._parse(resp.json(), requested)
                        # Hybrid reasoners (GLM-4.5+) may return empty
                        # message.content with everything in reasoning_content;
                        # retry ONCE with a larger max_tokens budget.
                        if not parsed["content"] and not empty_retry_used:
                            empty_retry_used = True
                            payload = dict(payload)
                            payload["max_tokens"] = max(
                                int(payload.get("max_tokens") or 0), 3000
                            )
                            continue  # empty-content retry: does not consume a retry slot
                        return parsed
                    elif resp.status_code == 429 or resp.status_code >= 500:
                        last_err = httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}", request=resp.request, response=resp
                        )
                    elif resp.status_code == 401:
                        pass  # last_err set; retry with refreshed bearer
                    else:
                        resp.raise_for_status()  # other 4xx: fail fast
                if attempt == self.retries:
                    break
                await asyncio.sleep(delay + random.uniform(0, _RETRY_JITTER))
                delay *= 2
        assert last_err is not None
        raise last_err

    def _payload(self, requested: str, messages: Sequence[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": requested, "messages": list(messages)}
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.extra_body:
            payload.update(self.extra_body)
        return payload

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
