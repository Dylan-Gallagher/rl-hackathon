"""Provider adapters.

Each adapter takes the SHARED neutral Transcript + a tool spec and produces a
completion, returning neutral assistant blocks (text + tool_call). The critical
property: an adapter serializes *the entire history* — including assistant turns
and tool calls originally produced by the OTHER model — into its own wire
format, with no marker that another model authored them. So each model believes
the whole conversation is its own work. That is the alloy mechanism.

Two adapters:
  * AnthropicAdapter  -> api.anthropic.com /v1/messages   (Claude)
  * OpenAICompatAdapter -> OpenAI-style /chat/completions  (GLM / Zhipu via z.ai)
"""
from __future__ import annotations
import json
import time
from typing import Any
import requests

from .transcript import Transcript, new_tool_call_id


class ProviderError(RuntimeError):
    pass


# --- tool spec: provider-neutral, translated per provider ------------------
# tools = [{"name","description","parameters": <json schema>}]

def _retry(fn, *, tries=6, base=2.0):
    last = None
    for i in range(tries):
        try:
            return fn()
        except ProviderError as e:
            last = e
            # retry on transient HTTP (429/5xx) AND connection/tunnel blips
            msg = str(e)
            if not any(c in msg for c in ("429", "500", "502", "503", "529",
                                          "overloaded", "timeout", "Timeout",
                                          "Connection", "connection", "Max retries",
                                          "Connection refused", "reset")):
                raise
            time.sleep(base * (2 ** i))
    raise last


class AnthropicAdapter:
    name_default = "claude-opus-4-8"
    provider = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int = 4096,
                 base_url: str = "https://api.anthropic.com"):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")

    # neutral -> anthropic messages
    def _messages(self, t: Transcript) -> list[dict]:
        out = []
        for m in t.messages:
            if m["role"] == "assistant":
                blocks = []
                for b in m["content"]:
                    if b["type"] == "text":
                        if b["text"].strip():
                            blocks.append({"type": "text", "text": b["text"]})
                    elif b["type"] == "tool_call":
                        blocks.append({"type": "tool_use", "id": b["id"],
                                       "name": b["name"], "input": b["arguments"]})
                if not blocks:
                    blocks = [{"type": "text", "text": "(continuing)"}]
                out.append({"role": "assistant", "content": blocks})
            else:  # user
                blocks = []
                for b in m["content"]:
                    if b["type"] == "text":
                        blocks.append({"type": "text", "text": b["text"]})
                    elif b["type"] == "tool_result":
                        blocks.append({"type": "tool_result",
                                       "tool_use_id": b["id"],
                                       "content": b["output"],
                                       "is_error": b["is_error"]})
                out.append({"role": "user", "content": blocks})
        return out

    def _tools(self, tools: list[dict]) -> list[dict]:
        return [{"name": t["name"], "description": t["description"],
                 "input_schema": t["parameters"]} for t in tools]

    def complete(self, t: Transcript, tools: list[dict]) -> list[dict]:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": t.system,
            "messages": self._messages(t),
            "tools": self._tools(tools),
        }

        def call():
            try:
                r = requests.post(f"{self.base_url}/v1/messages",
                                  headers={"x-api-key": self.api_key,
                                           "anthropic-version": "2023-06-01",
                                           "content-type": "application/json"},
                                  json=body, timeout=180)
            except requests.exceptions.RequestException as e:
                raise ProviderError(f"anthropic connection: {e}")
            if r.status_code != 200:
                raise ProviderError(f"anthropic {r.status_code}: {r.text[:300]}")
            return r.json()

        data = _retry(call)
        blocks = []
        for b in data.get("content", []):
            if b["type"] == "text":
                blocks.append({"type": "text", "text": b["text"]})
            elif b["type"] == "tool_use":
                blocks.append({"type": "tool_call", "id": b["id"],
                               "name": b["name"], "arguments": b["input"]})
        return blocks


class OpenAICompatAdapter:
    """OpenAI-compatible chat.completions (used for GLM / Zhipu via z.ai)."""
    provider = "openai_compat"

    def __init__(self, api_key: str, model: str, max_tokens: int = 4096,
                 base_url: str = "https://api.z.ai/api/paas/v4",
                 sampling: dict | None = None):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        # pinned sampling (temperature/top_p/top_k/repetition_penalty/...) merged
        # into every request body — used to hold the serving stack fixed for the
        # self-hosted baseline. Empty for hosted providers (no behavior change).
        self.sampling = dict(sampling or {})

    # neutral -> openai messages
    def _messages(self, t: Transcript) -> list[dict]:
        out = [{"role": "system", "content": t.system}]
        for m in t.messages:
            if m["role"] == "assistant":
                text = "\n".join(b["text"] for b in m["content"]
                                 if b["type"] == "text")
                tcs = []
                for b in m["content"]:
                    if b["type"] == "tool_call":
                        tcs.append({"id": b["id"], "type": "function",
                                    "function": {"name": b["name"],
                                                 "arguments": json.dumps(b["arguments"])}})
                msg = {"role": "assistant", "content": text or None}
                if tcs:
                    msg["tool_calls"] = tcs
                out.append(msg)
            else:  # user
                texts = [b for b in m["content"] if b["type"] == "text"]
                trs = [b for b in m["content"] if b["type"] == "tool_result"]
                if texts:
                    out.append({"role": "user",
                                "content": "\n".join(b["text"] for b in texts)})
                for b in trs:  # each tool_result -> a tool-role message
                    content = b["output"]
                    if b["is_error"]:
                        content = f"[ERROR]\n{content}"
                    out.append({"role": "tool", "tool_call_id": b["id"],
                                "content": content})
        return out

    def _tools(self, tools: list[dict]) -> list[dict]:
        return [{"type": "function",
                 "function": {"name": t["name"],
                              "description": t["description"],
                              "parameters": t["parameters"]}}
                for t in tools]

    def complete(self, t: Transcript, tools: list[dict]) -> list[dict]:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._messages(t),
            "tools": self._tools(tools),
            "tool_choice": "auto",
        }
        body.update(self.sampling)

        def call():
            try:
                r = requests.post(f"{self.base_url}/chat/completions",
                                  headers={"Authorization": f"Bearer {self.api_key}",
                                           "Content-Type": "application/json"},
                                  json=body, timeout=180)
            except requests.exceptions.RequestException as e:
                raise ProviderError(f"openai_compat connection: {e}")
            if r.status_code != 200:
                raise ProviderError(f"openai_compat {r.status_code}: {r.text[:300]}")
            return r.json()

        data = _retry(call)
        choice = data["choices"][0]["message"]
        blocks = []
        if choice.get("content"):
            blocks.append({"type": "text", "text": choice["content"]})
        for tc in choice.get("tool_calls") or []:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc["function"]["arguments"]}
            cid = tc.get("id") or new_tool_call_id()
            blocks.append({"type": "tool_call", "id": cid,
                           "name": tc["function"]["name"], "arguments": args})
        return blocks
