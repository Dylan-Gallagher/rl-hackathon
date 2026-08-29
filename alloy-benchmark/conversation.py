"""The alloy mechanism, isolated.

A single continuous conversation is stored ONCE in a provider-neutral canonical
form. At each turn we render the full canonical history into whichever provider's
wire format we are about to call, and we parse the reply straight back into the
neutral form. The canonical form has NO slot for provider name, model name,
response id, stop reason, thinking/signature, reasoning content or usage — so
none of those can ever be carried across a turn boundary. Every assistant turn,
no matter which model produced it, is rebuilt identically for the next model,
which therefore sees the whole thread as if it had written every turn itself.

Canonical shape:
    system : str
    messages : list[turn]
        turn = {"role": "user"|"assistant", "content": [part, ...]}
        part (assistant): {"type":"text","text":str}
                          {"type":"tool_use","id":str,"name":str,"input":dict}
        part (user):      {"type":"text","text":str}
                          {"type":"tool_result","id":str,"output":str,"is_error":bool}

Tool-call ids are OUR OWN synthetic ids ("call_0001", ...), assigned by this
module when an assistant turn is appended. Provider-generated ids are discarded
at the door.
"""
from __future__ import annotations
import json
from typing import Any


class Conversation:
    def __init__(self, system: str):
        self.system = system
        self.messages: list[dict] = []
        self._id_counter = 0

    # -- building the thread -------------------------------------------------
    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": [{"type": "text", "text": text}]})

    def add_tool_results(self, results: list[dict]) -> None:
        """results: [{"id","output","is_error"}]"""
        content = [
            {"type": "tool_result", "id": r["id"], "output": r["output"], "is_error": r.get("is_error", False)}
            for r in results
        ]
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, parts: list[dict]) -> list[dict]:
        """parts come from a parser as neutral text/tool_use dicts WITHOUT ids.
        Assign our own synthetic ids to tool_use parts, append the turn, and
        return the list of tool_use parts (now with ids) for execution."""
        clean: list[dict] = []
        tool_uses: list[dict] = []
        for p in parts:
            if p["type"] == "text":
                if p.get("text", "").strip() != "":
                    clean.append({"type": "text", "text": p["text"]})
            elif p["type"] == "tool_use":
                cid = f"call_{self._id_counter:04d}"
                self._id_counter += 1
                part = {"type": "tool_use", "id": cid, "name": p["name"], "input": p["input"]}
                clean.append(part)
                tool_uses.append(part)
        if not clean:
            # A model may return an empty completion; keep the thread valid with
            # a non-empty placeholder (empty text blocks are rejected by the APIs).
            clean.append({"type": "text", "text": "(no output)"})
        self.messages.append({"role": "assistant", "content": clean})
        return tool_uses

    # -- rendering to provider wire formats ---------------------------------
    def to_anthropic(self) -> tuple[str, list[dict]]:
        out: list[dict] = []
        for turn in self.messages:
            blocks: list[dict] = []
            for p in turn["content"]:
                if p["type"] == "text":
                    if p["text"] != "":
                        blocks.append({"type": "text", "text": p["text"]})
                elif p["type"] == "tool_use":
                    blocks.append({"type": "tool_use", "id": p["id"], "name": p["name"], "input": p["input"]})
                elif p["type"] == "tool_result":
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": p["id"],
                        "content": p["output"],
                        "is_error": p["is_error"],
                    })
            if not blocks:
                blocks.append({"type": "text", "text": "(no output)"})
            out.append({"role": turn["role"], "content": blocks})
        return self.system, out

    def to_openai(self) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": self.system}]
        for turn in self.messages:
            if turn["role"] == "assistant":
                text_parts = [p["text"] for p in turn["content"] if p["type"] == "text" and p["text"] != ""]
                tool_calls = [
                    {
                        "id": p["id"],
                        "type": "function",
                        "function": {"name": p["name"], "arguments": json.dumps(p["input"])},
                    }
                    for p in turn["content"] if p["type"] == "tool_use"
                ]
                msg: dict[str, Any] = {"role": "assistant"}
                msg["content"] = ("\n".join(text_parts)) if text_parts else None
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                out.append(msg)
            else:  # user turn: split text vs tool_result into separate messages
                text_parts = [p for p in turn["content"] if p["type"] == "text"]
                tool_results = [p for p in turn["content"] if p["type"] == "tool_result"]
                for p in text_parts:
                    out.append({"role": "user", "content": p["text"]})
                for p in tool_results:
                    out.append({"role": "tool", "tool_call_id": p["id"], "content": p["output"]})
        return out


# --- parsers: provider reply -> neutral parts (no ids, no provider fields) ---
def parse_anthropic_response(resp) -> list[dict]:
    parts: list[dict] = []
    for block in resp.content:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            parts.append({"type": "tool_use", "name": block.name, "input": dict(block.input)})
        # thinking/redacted_thinking blocks (should not occur; thinking is off) are dropped
    return parts


def parse_openai_response(resp) -> list[dict]:
    parts: list[dict] = []
    msg = resp.choices[0].message
    if getattr(msg, "content", None):
        parts.append({"type": "text", "text": msg.content})
    for tc in (getattr(msg, "tool_calls", None) or []):
        try:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            args = {"_raw_arguments": tc.function.arguments}
        parts.append({"type": "tool_use", "name": tc.function.name, "input": args})
    return parts
