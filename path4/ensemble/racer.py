"""Racing harness — k policies race per challenge, first verified flag wins.

Each episode runs on its OWN fresh env from ``env_factory(task)``. Episodes
gather concurrently; the first one whose env reports ``solved()`` sets a shared
event that cancels the others at their next turn boundary (§1.1 racing-alloy
pattern, ctf-agent style). Optional shared-findings bus carries HINTS ONLY —
any ``flag{...}`` payload is filtered out before delivery (anti-cheat).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from contracts.env.base import CTFEnv
from contracts.task import Task
from contracts.transcript import Transcript, TranscriptMessage, episode_id_for, write_transcript

from path4.ensemble.agent import ChatLike, run_episode
from path4.ensemble.policies import Policy

logger = logging.getLogger(__name__)

#: Anything shaped like a flag is scrubbed from bus messages (test-enforced).
_FLAG_RE = re.compile(r"flag\{[^}\n]*\}")
_FLAG_REPLACEMENT = "flag{[REDACTED]}"


def scrub_flags(text: str) -> str:
    """Remove any flag-shaped substring (anti-cheat: the bus never carries flags)."""
    return _FLAG_RE.sub(_FLAG_REPLACEMENT, text)


class Inbox:
    """Per-episode mailbox for teammate findings (drained by the agent loop)."""

    def __init__(self) -> None:
        self._items: list[str] = []
        self._read = 0

    def push(self, note: str) -> None:
        self._items.append(note)

    def drain(self) -> list[str]:
        out = self._items[self._read:]
        self._read = len(self._items)
        return out


class FindingsBus:
    """Minimal async pub/sub for non-flag hints between racing policies."""

    def __init__(self) -> None:
        self.inboxes: dict[str, Inbox] = {}
        self.published: list[tuple[str, str]] = []  # audit log (already scrubbed)

    def inbox_for(self, policy_name: str) -> Inbox:
        if policy_name not in self.inboxes:
            self.inboxes[policy_name] = Inbox()
        return self.inboxes[policy_name]

    async def publish(self, policy_name: str, note: str) -> None:
        note = scrub_flags(note)
        self.published.append((policy_name, note))
        for name, inbox in self.inboxes.items():
            if name != policy_name:
                inbox.push(note)


@dataclass
class RaceResult:
    task_id: str
    race_id: str
    winner_policy: str | None
    solved: bool
    episodes: list[Transcript] = field(default_factory=list)
    wall_time: float = 0.0
    summary_path: Path | None = None


def make_env_factory(
    kind: str,
) -> Callable[[Task], CTFEnv]:
    """Map an env kind to a factory producing FRESH envs per episode."""
    kind = kind.lower()

    def factory(task: Task) -> CTFEnv:
        if kind == "mock":
            from contracts.env.mock import MockCTFEnv

            return MockCTFEnv(task)
        if kind == "repl":
            from contracts.env.repl import ReplCTFEnv

            return ReplCTFEnv(task)
        if kind == "docker":
            from contracts.env.docker import DockerCTFEnv

            return DockerCTFEnv(task)
        if kind == "daytona":
            from contracts.env.daytona import DaytonaCTFEnv

            return DaytonaCTFEnv(task)
        raise ValueError(f"unknown env kind {kind!r} (mock|repl|docker|daytona)")

    return factory


async def race(
    task: Task,
    policies: list[Policy],
    env_factory: Callable[[Task], CTFEnv],
    chat_client: ChatLike,
    max_steps: int = 40,
    findings_bus: bool = False,
    race_id: str | None = None,
    out_dir: str | Path | None = None,
    timeout_s: float | None = None,
    seed: int | None = None,
) -> RaceResult:
    """Race k policies on one task; first verified flag wins.

    Writes one transcript JSONL per episode plus ``summary.json`` into
    ``out_dir`` (when given). Each episode gets its own fresh env and its own
    seeded rng derived from ``(race_id, policy, task)``.
    """
    race_id = race_id or uuid.uuid4().hex[:8]
    stop_event = asyncio.Event()
    bus = FindingsBus() if findings_bus else None
    winner: str | None = None
    winner_lock = asyncio.Lock()

    async def run_one(policy: Policy, policy_seed: int | None) -> Transcript:
        nonlocal winner
        inbox = bus.inbox_for(policy.name()) if bus else None
        on_finding = bus.publish if bus else None

        async def guarded_on_finding(policy_name: str, note: str) -> None:
            if on_finding is not None:
                await on_finding(policy_name, note)

        try:
            t = await run_episode(
                task,
                policy,
                chat_client,
                env_factory(task),
                max_steps=max_steps,
                race_id=race_id,
                on_finding=guarded_on_finding,
                findings_inbox=inbox,
                stop_event=stop_event,
                timeout_s=timeout_s,
                seed=policy_seed,
            )
        except Exception as e:  # a failing policy must not kill the race
            logger.warning("episode crashed for %s on %s: %s", policy.name(), task.task_id, e)
            t = Transcript(
                task_id=task.task_id,
                episode_id=episode_id_for(
                    f"{race_id + ':' if race_id else ''}{policy.name()}", task.task_id, 0
                ),
                policy=policy.name(),
                split=task.split,
                solved=False,
                category=task.category,
                messages=[TranscriptMessage(turn=0, role="assistant", content=f"error: {e}")],
                error=str(e),
            )
        if t.solved:
            async with winner_lock:
                if winner is None:
                    winner = policy.name()
            stop_event.set()
        return t

    started = time.monotonic()
    # Distinct per-policy seeds so racing policies never share a derived flag.
    episodes = list(
        await asyncio.gather(
            *(run_one(p, None if seed is None else seed + i) for i, p in enumerate(policies))
        )
    )
    wall_time = time.monotonic() - started

    result = RaceResult(
        task_id=task.task_id,
        race_id=race_id,
        winner_policy=winner,
        solved=winner is not None,
        episodes=episodes,
        wall_time=round(wall_time, 3),
    )

    if out_dir is not None:
        result.summary_path = write_race_artifacts(result, out_dir)
    return result


def write_race_artifacts(result: RaceResult, out_dir: str | Path) -> Path:
    """Write per-episode transcripts + ``summary.json``; return summary path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per_policy: dict[str, dict[str, Any]] = {}
    for t in result.episodes:
        # mode='w': a re-run with the same race_id OVERWRITES per-episode files
        # instead of appending duplicate lines (scoreboard double-count).
        write_transcript(t, out / f"{t.episode_id}.jsonl", mode="w")
        per_policy[t.policy] = {
            "solved": t.solved,
            "steps": t.steps,
            "wall_time": getattr(t, "wall_time_ext", None),
            "cancelled": getattr(t, "cancelled", False),
        }
    summary = {
        "race_id": result.race_id,
        "task_id": result.task_id,
        "winner": result.winner_policy,
        "solved": result.solved,
        "wall_time": result.wall_time,
        "policies": per_policy,
    }
    path = out / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


# Re-export so callers can build ids consistently with run_episode.
__all__ = [
    "RaceResult",
    "FindingsBus",
    "Inbox",
    "race",
    "scrub_flags",
    "make_env_factory",
    "write_race_artifacts",
    "episode_id_for",
]
