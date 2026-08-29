"""Robust parsing of model output into terminal commands, and flag extraction.

Terminal-command action protocol (no native function calling across providers):
the model replies with a single fenced ```bash block (or a `CMD:`-prefixed
line); anything else is treated as a no-op comment turn. We never trust model
self-reports of solving — only verified flags count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ctf_gym.verifier import FLAG_SCAN_RE


@dataclass
class ParsedAction:
    command: Optional[str]
    declared_flags: list[str] = field(default_factory=list)
    raw: str = ""


BASH_BLOCK_RE = re.compile(r"```(?:bash|sh|shell|python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
CMD_PREFIX_RE = re.compile(r"^\s*(?:CMD|COMMAND|ACTION)\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def parse_action(model_output: str) -> ParsedAction:
    """Extract the terminal command a model wants to run.

    Priority: fenced code block -> CMD: prefix -> single bare command line.
    Refuses obvious conversational text; returns command=None for no-op.
    """
    raw = model_output or ""
    blocks = BASH_BLOCK_RE.findall(raw)
    if blocks:
        # last block wins (models often restate the final command)
        cmd = blocks[-1].strip()
        if cmd:
            return ParsedAction(command=cmd, raw=raw)
    m = CMD_PREFIX_RE.search(raw)
    if m:
        return ParsedAction(command=m.group(1).strip(), raw=raw)
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    bare = [l for l in lines if not l.startswith(("#", "//"))]
    if len(bare) == 1 and re.match(r"^[A-Za-z_./][\w./@%+=:,-]*", bare[0]) and len(bare[0]) <= 500:
        return ParsedAction(command=bare[0], raw=raw)
    return ParsedAction(command=None, raw=raw)


def extract_declared_flags(model_output: str) -> list[str]:
    """Flag-shaped strings the model claims to have found (still must verify)."""
    return list(dict.fromkeys(FLAG_SCAN_RE.findall(model_output or "")))


def extract_flags(text: str) -> list[str]:
    return list(dict.fromkeys(FLAG_SCAN_RE.findall(text or "")))
