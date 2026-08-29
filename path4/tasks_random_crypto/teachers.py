"""Teacher LLM clients backed by `pi auth` (Grok OAuth) + ZAI_API_KEY (GLM).

Never writes tokens to disk. `pi auth print-bearer-token` refreshes OAuth.
"""

from __future__ import annotations

import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import httpx

from path4.ensemble.llm import ChatClient

ZAI_BASE = os.environ.get("ZAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
XAI_BASE = "https://api.x.ai/v1"
THINKING_OFF = {"thinking": {"type": "disabled"}}


class CachedPiSecret:
    """Callable credential from `pi auth` with a TTL cache. Never written to disk."""

    def __init__(self, provider: str, kind: str = "print-bearer-token", ttl_s: float = 240.0):
        self.provider = provider
        self.kind = kind  # print-bearer-token | print-api-key
        self.ttl_s = ttl_s
        self._token = ""
        self._fetched = 0.0

    def invalidate(self) -> None:
        self._token = ""
        self._fetched = 0.0

    def __call__(self) -> str:
        now = time.time()
        if self._token and (now - self._fetched) < self.ttl_s:
            return self._token
        out = subprocess.check_output(
            ["pi", "auth", self.kind, "--provider", self.provider],
            text=True,
            timeout=30,
        ).strip()
        if not out:
            raise RuntimeError(f"pi auth {self.kind} empty for {self.provider}")
        self._token = out
        self._fetched = now
        return self._token


class CachedPiBearer(CachedPiSecret):
    def __init__(self, provider: str, ttl_s: float = 240.0):
        super().__init__(provider, kind="print-bearer-token", ttl_s=ttl_s)


def _load_dotenv() -> None:
    p = Path(__file__).resolve().parents[2] / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


class AnthropicChatClient:
    """Native Anthropic Messages API (same chat() shape as ChatClient)."""

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-5",
                 timeout: float = 180.0, retries: int = 4, max_tokens: int = 2048):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or ""
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.max_tokens = max_tokens

    async def chat(self, messages: Sequence[dict[str, str]], model: str | None = None) -> dict[str, Any]:
        system, conv = [], []
        for m in messages:
            role, content = m.get("role") or "user", m.get("content") or ""
            if role == "system":
                system.append(content)
            elif role == "assistant":
                conv.append({"role": "assistant", "content": content})
            else:
                if conv and conv[-1]["role"] == "user":
                    conv[-1]["content"] += "\n" + content
                else:
                    conv.append({"role": "user", "content": content})
        if not conv or conv[0]["role"] != "user":
            conv.insert(0, {"role": "user", "content": "(continue)"})
        payload: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": self.max_tokens,
            "messages": conv,
        }
        if system:
            payload["system"] = "\n\n".join(system)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        delay = 1.0
        last: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.retries + 1):
                try:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages", json=payload, headers=headers)
                except httpx.HTTPError as e:
                    last = e
                else:
                    if resp.status_code == 200:
                        data = resp.json()
                        blocks = data.get("content") or []
                        text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
                        usage = data.get("usage") or {}
                        return {
                            "content": text,
                            "model": data.get("model") or payload["model"],
                            "tokens_in": int(usage.get("input_tokens") or 0),
                            "tokens_out": int(usage.get("output_tokens") or 0),
                        }
                    if resp.status_code in (429, 500, 502, 503, 529):
                        last = httpx.HTTPStatusError(
                            f"HTTP {resp.status_code}", request=resp.request, response=resp)
                    else:
                        resp.raise_for_status()
                if attempt == self.retries:
                    break
                await __import__("asyncio").sleep(delay + random.uniform(0, 0.25))
                delay *= 2
        assert last is not None
        raise last


class RoutingChatClient:
    """Route `chat(model=...)` to the matching teacher client by model prefix."""

    def __init__(self, routes: list[tuple[str, Any]], default: Any):
        self.routes = routes
        self.default = default

    async def chat(self, messages: Sequence[dict[str, str]], model: str | None = None) -> dict[str, Any]:
        m = model or ""
        for prefix, client in self.routes:
            if m.startswith(prefix):
                return await client.chat(messages, model=model)
        return await self.default.chat(messages, model=model)


def make_teacher_client() -> RoutingChatClient:
    _load_dotenv()
    # GLM-5.x lives on the *coding* plan (pi auth zai-coding-cn), not the
    # pay-as-you-go ZAI_API_KEY (that one 429s / has no 5.x pack).
    # 5.3 always-thinks: do NOT send thinking.disabled (API 400).
    glm5 = ChatClient(
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
        api_key=CachedPiSecret("zai-coding-cn", kind="print-api-key", ttl_s=600),
        model="glm-5.3-flash",
        timeout=180.0,
        retries=4,
        max_tokens=3000,
        extra_body=None,
    )
    glm4 = ChatClient(
        base_url=ZAI_BASE,
        api_key=os.environ.get("ZAI_API_KEY"),
        model="glm-4.5-air",
        timeout=180.0,
        retries=4,
        max_tokens=2048,
        extra_body=THINKING_OFF,
    )
    grok = ChatClient(
        base_url=XAI_BASE,
        api_key=CachedPiBearer("xai"),
        model="grok-4.6",
        timeout=180.0,
        retries=4,
        max_tokens=2048,
        extra_body=THINKING_OFF,
    )
    claude = AnthropicChatClient(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        model="claude-sonnet-4-5",
        timeout=180.0,
        retries=4,
        max_tokens=2048,
    )
    # Prefix order: glm-5* before glm* so 5.x does not hit the 4.x client.
    return RoutingChatClient(
        [("claude", claude), ("glm-5", glm5), ("glm", glm4), ("grok", grok)],
        default=glm5,
    )
