"""GRPO rewards + DAPO-style dynamic sampling filter (README §1.2, Path 4 step 2).

Pure python; no torch/veRL needed. The binary flag reward is the whole reward
function — flags self-verify, so Pass@k (not reward shaping) is the metric.
"""

from __future__ import annotations

from dataclasses import dataclass


def flag_reward(solved: bool, flags_found: list[str] | None = None) -> float:
    """Binary flag-as-reward: 1.0 iff the episode verified a flag.

    ``flags_found`` is accepted for transcript-shaped call sites
    (``flag_reward(t.solved, t.flags_found)``); an empty/missing list with
    ``solved=True`` still scores 1.0 — solvedness is the env verifier's word.
    """
    return 1.0 if solved else 0.0


@dataclass
class GroupStats:
    """Per-group reward summary (a group = G rollouts of one task)."""

    mean: float
    std: float
    all_zero: bool
    all_one: bool
    variance: float


def reward_group(rewards: list[float]) -> GroupStats:
    """Aggregate one GRPO group. Empty group -> zeros (safe to filter out)."""
    n = len(rewards)
    if n == 0:
        return GroupStats(0.0, 0.0, True, False, 0.0)
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    return GroupStats(mean=mean, std=variance ** 0.5, all_zero=all(r == 0.0 for r in rewards),
                      all_one=all(r == 1.0 for r in rewards), variance=variance)


def dapo_filter(groups: list[list[float]]) -> list[bool]:
    """DAPO-style dynamic sampling: keep_mask dropping degenerate groups.

    All-zero groups (nobody solved) and all-one groups (everybody solved)
    carry zero advantage variance -> zero gradient -> wasted GPU. Drop them
    and resample instead (README Path 4 step 2, DAPO dynamic sampling).
    """
    keep = []
    for g in groups:
        stats = reward_group(g)
        keep.append(not (stats.all_zero or stats.all_one))
    return keep
