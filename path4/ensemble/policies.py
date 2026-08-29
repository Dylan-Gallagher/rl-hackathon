"""Policies — which model serves each turn (README §1.1 per-turn alloy pattern).

A policy is a routing rule, not an agent: it answers "which model for this
turn?" and knows nothing about the conversation. Models never learn about each
other — each thinks it wrote all prior assistant turns.

Canonical policy strings (``Transcript.policy``):
- ``solo:MODEL``
- ``alloy:M1:w1,M2:w2`` (weights optional, default equal)
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class Policy(Protocol):
    """Routing rule: pick the model for a given turn."""

    def name(self) -> str:
        """Canonical policy string: 'solo:MODEL' or 'alloy:M1:w1,M2:w2'."""

    async def pick_model(self, turn: int, rng: random.Random) -> str:
        """Choose the model serving this turn (weighted random for alloys)."""


class SoloPolicy(BaseModel):
    """One model every turn."""

    model_config = {"frozen": True}

    model: str

    def name(self) -> str:
        return f"solo:{self.model}"

    async def pick_model(self, turn: int, rng: random.Random) -> str:
        return self.model

    def model_for_client(self) -> str:
        return self.model


class AlloyMember(BaseModel):
    model_config = {"frozen": True}

    model: str
    weight: float = Field(default=1.0, gt=0)


class AlloyPolicy(BaseModel):
    """Per-turn weighted-random routing across models (XBow pattern)."""

    model_config = {"frozen": True}

    members: list[AlloyMember]

    def name(self) -> str:
        parts = []
        for m in self.members:
            if m.weight == 1.0:
                parts.append(m.model)
            else:
                parts.append(f"{m.model}:{m.weight:g}")
        return "alloy:" + ",".join(parts)

    async def pick_model(self, turn: int, rng: random.Random) -> str:
        return rng.choices(
            [m.model for m in self.members],
            weights=[m.weight for m in self.members],
        )[0]

    def model_for_client(self) -> str:
        """Alloys route per call; the client-level default is unused."""
        return ""


def parse_policy(spec: str) -> Policy:
    """Parse ``'solo:X'`` or ``'alloy:A:0.6,B:0.4'`` into a Policy."""
    spec = spec.strip()
    if spec.startswith("solo:"):
        model = spec[len("solo:"):].strip()
        if not model:
            raise ValueError(f"empty solo model in policy {spec!r}")
        return SoloPolicy(model=model)
    if spec.startswith("alloy:"):
        body = spec[len("alloy:"):].strip()
        if not body:
            raise ValueError(f"empty alloy in policy {spec!r}")
        members: list[AlloyMember] = []
        for part in body.split(","):
            bits = [b.strip() for b in part.split(":") if b.strip()]
            if not bits:
                raise ValueError(f"empty alloy member in policy {spec!r}")
            weight = float(bits[1]) if len(bits) > 1 else 1.0
            members.append(AlloyMember(model=bits[0], weight=weight))
        if len(members) < 2:
            raise ValueError(f"alloy needs >=2 members: {spec!r}")
        return AlloyPolicy(members=members)
    raise ValueError(f"unrecognized policy spec {spec!r} (want 'solo:X' or 'alloy:A:0.6,B:0.4')")
