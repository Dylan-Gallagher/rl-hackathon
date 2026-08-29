"""Env protocol per README §3.3 — the CTF gym abstraction shared by all paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from contracts.task import Task


@dataclass
class Obs:
    """Observation returned by ``reset``/``step``.

    ``metadata`` is backend-specific free-form info (e.g. exit codes, sandbox id).
    """

    output: str
    done: bool
    step: int = 0
    metadata: dict[str, Any] | None = field(default=None)


class CTFEnv(ABC):
    """Abstract CTF environment (README §3.3).

    Contract:
    - ``reset``: fresh sandbox, inject per-episode flag (§1.5), return initial obs.
    - ``step``: execute a shell-style action; cap output (head/tail policy).
    - ``solved``: flag verifier + regex scan of accumulated observations
      (verification happens OUTSIDE the sandbox).
    - ``close``: destroy the sandbox.
    """

    @abstractmethod
    async def reset(self, seed: int | None = None) -> Obs:
        """Start a fresh episode."""

    @abstractmethod
    async def step(self, action: str) -> Obs:
        """Execute one action; return a capped observation."""

    @abstractmethod
    def solved(self) -> bool:
        """True iff a verified flag was found (verifier outside sandbox)."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down the sandbox."""


def capped_output(text: str, max_chars: int = 4000) -> str:
    """Cap long output with a head/tail policy: keep head 60% / tail 40%.

    Returns ``text`` unchanged if it fits. Otherwise:
    ``<head>...[truncated N chars]...<tail>`` where N is the dropped char count.
    """
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    marker = f"...[truncated {len(text) - max_chars} chars]..."
    head = max_chars * 3 // 5
    tail = max_chars - head
    return text[:head] + marker + (text[-tail:] if tail else "")
