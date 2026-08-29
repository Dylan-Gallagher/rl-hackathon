"""Provider-neutral transcript.

Every model in an alloy reads and writes the SAME transcript. We store messages
in a provider-agnostic shape and let each provider adapter serialize the whole
history into its own wire format on every call. That is what lets GLM see
Claude's earlier tool calls as if GLM had made them, and vice versa.

Neutral message shapes (a message is a dict with a "role" and typed "content"
blocks):

  {"role": "system",    "text": "..."}
  {"role": "user",      "content": [ {"type":"text","text":...},
                                     {"type":"tool_result","id":...,
                                      "output":..., "is_error":bool} ]}
  {"role": "assistant", "content": [ {"type":"text","text":...},
                                     {"type":"tool_call","id":...,
                                      "name":..., "arguments":{...}} ],
                        "model": "<which model emitted this>"}

Tool *calls* live on assistant turns; tool *results* live on the following user
turn. Ids link a call to its result. Adapters translate this to/from Anthropic
and OpenAI-compatible (GLM) formats.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional
import itertools

_counter = itertools.count(1)


def new_tool_call_id() -> str:
    return f"call_{next(_counter):06d}"


@dataclass
class Transcript:
    system: str = ""
    messages: list[dict] = field(default_factory=list)

    # ---- writers -----------------------------------------------------------
    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user",
                              "content": [{"type": "text", "text": text}]})

    def add_assistant(self, blocks: list[dict], model: str) -> None:
        """blocks: list of {"type":"text",...} and/or {"type":"tool_call",...}"""
        self.messages.append({"role": "assistant", "content": blocks,
                              "model": model})

    def add_tool_results(self, results: list[dict]) -> None:
        """results: list of {"id","output","is_error"} -> one user turn."""
        content = [{"type": "tool_result", "id": r["id"],
                    "output": r["output"], "is_error": r.get("is_error", False)}
                   for r in results]
        self.messages.append({"role": "user", "content": content})

    # ---- readers -----------------------------------------------------------
    def last_assistant(self) -> Optional[dict]:
        for m in reversed(self.messages):
            if m["role"] == "assistant":
                return m
        return None

    def pending_tool_calls(self) -> list[dict]:
        """tool_call blocks from the most recent assistant turn."""
        a = self.last_assistant()
        if not a:
            return []
        return [b for b in a["content"] if b["type"] == "tool_call"]

    def assistant_text(self, msg: dict) -> str:
        return "\n".join(b["text"] for b in msg["content"]
                         if b["type"] == "text")
