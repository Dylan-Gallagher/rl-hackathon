"""Flag generation, verification, and observation scanning (README §1.5, §3.3).

Anti-cheat rules implemented here:
- ``new_flag`` produces a fresh ``flag{uuid4}`` per episode; it is injected
  only where the challenge needs it.
- ``verify_flag`` runs OUTSIDE the sandbox (called by the env backend /
  orchestrator, never by agent code inside the sandbox).
- ``scan_for_flags`` lets envs auto-detect solves by regex-scanning
  observations for the flag pattern.
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from contracts.task import Task

#: Default pattern for generated flags: flag{<36-char uuid4 hex/dash>}.
DEFAULT_FLAG_PATTERN = r"flag\{[0-9a-fA-F-]{36}\}"

_GENERATED_RE = re.compile(DEFAULT_FLAG_PATTERN)


def new_flag(task: "Task | None" = None) -> str:
    """Return a fresh ``flag{uuid4}``.

    If ``task`` is given and has a custom static-format template
    (``flag.format`` other than ``flag{uuid4}`` with a ``{uuid4}`` hole), the
    uuid is substituted into it; otherwise the default pattern is used.
    """
    u = str(uuid.uuid4())
    template = getattr(task, "flag", None)
    fmt = getattr(template, "format", None) if template else None
    if fmt and "{uuid4}" in fmt and fmt != "flag{uuid4}":
        return fmt.replace("{uuid4}", u)
    return f"flag{{{u}}}"


def seeded_flag(seed: int, task: "Task | None" = None, task_id: str | None = None) -> str:
    """Deterministic flag for a given seed (reproducible tests/episodes).

    Task identity (``task_id`` from the param or from ``task``) is mixed into
    the derived seed so two different tasks replayed with the same numeric
    seed never collide on the same flag. Same task + same seed is stable.
    """
    import random
    tid = task_id or (task.task_id if task is not None else "")
    rnd = random.Random(f"flag:{seed}:{tid}")
    u = str(uuid.UUID(int=rnd.getrandbits(128), version=4))
    fmt = getattr(getattr(task, "flag", None), "format", None) if task else None
    if fmt and "{uuid4}" in fmt and fmt != "flag{uuid4}":
        return fmt.replace("{uuid4}", u)
    return f"flag{{{u}}}"


def verify_flag(found: str, expected: str, mode: str, pattern: str | None = None) -> bool:
    """Verify a candidate flag against the expected one.

    Modes:
    - ``exact``: string equality (after strip).
    - ``regex``: currently requires EXACT equality with the concrete
      per-episode flag AND, when a ``pattern`` is supplied, that the found
      value conforms to the pattern. There is no pattern-only verification
      today: the pattern check cannot accept a flag the expected value does
      not equal-match (the branch below is behaviorally exact-equality).
      Future work may support flags verified by pattern conformance alone
      (where the concrete flag is not known up front).
    - ``script``: verification script executed by the env backend OUTSIDE the
      sandbox (see README §1.5). This function does NOT execute scripts —
      env backends that support script verification must run the task's
      verifier script themselves in trusted infrastructure.
    """
    found = (found or "").strip()
    expected = (expected or "").strip()
    if mode == "exact":
        return found == expected
    if mode == "regex":
        if found == expected:
            return True
        if pattern:
            try:
                return re.fullmatch(pattern, found) is not None and found == expected
            except re.error:
                return False
        return False
    if mode == "script":
        # Executed by the env backend outside the sandbox; nothing to do here.
        return found == expected
    return False


def scan_for_flags(text: str, pattern: str | None = None) -> list[str]:
    """Regex-scan an observation for flag strings (auto-solve detection).

    Default pattern is :data:`DEFAULT_FLAG_PATTERN`; pass ``pattern`` to scan
    for a custom flag format. Returns unique matches in order of appearance.
    """
    if not text:
        return []
    rx = _GENERATED_RE if pattern is None else re.compile(pattern)
    seen: list[str] = []
    for m in rx.findall(text):
        if m not in seen:
            seen.append(m)
    return seen
