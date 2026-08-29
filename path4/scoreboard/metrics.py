"""Pure-python aggregation of canonical transcripts into scoreboard metrics.

No I/O policy of its own beyond scanning helpers — ``aggregate()`` takes plain
``list[Transcript]`` (plus optional race summaries) so it is trivially testable.

Metric definitions
------------------
Pass@k (k > 1): the *unbiased* combinatorial estimator (HumanEval-style), averaged
per task over tasks with at least k episodes for the policy:

    pass@k(task) = 1 - C(n - c, k) / C(n, k)

where n = episode count for (policy, task), c = solved count. This equals the
probability that k episodes drawn uniformly without replacement from the n
observed episodes contain at least one solve. Tasks with n < k are **excluded**
from the average (guard: no meaningful unbiased estimate exists; documented
choice — the alternative, "any of first k", would silently use fewer than k
samples and bias downward on hard tasks).

Pass@1: solves / episodes over all episodes (the raw solve rate).

Maj@k: a task counts as solved iff strictly more than k/2 of its *first k*
episodes (ordered by episode_id) are solved; averaged over tasks with >= k
episodes. Excluded otherwise (same guard as Pass@k).

first-solve-time: NOT computed — the transcript schema carries no wall-clock
timing, only step/token counts. Noted rather than invented.

Categories come from the optional ``category`` extension field on Transcript;
missing -> 'unknown'.

Race wins: optional. If the scanned tree contains ``summary.json`` files in the
path4/ensemble format (``{"race_id", "task_id", "winner", ...}``), each file is
one race and ``winner`` (a policy name) gets +1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from contracts import Transcript, iter_transcripts

DEFAULT_KS: tuple[int, ...] = (1, 4, 8)
UNKNOWN_CATEGORY = "unknown"


def scan_transcripts(root: str | Path) -> list[Transcript]:
    """Recursively read every ``*.jsonl`` under ``root`` into Transcripts.

    ``contracts.iter_transcripts`` is non-recursive (top-level ``*.jsonl`` of a
    dir), so we walk the tree ourselves and hand each file to it — schemas stay
    owned by contracts, only discovery is local.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    out: list[Transcript] = []
    for fp in sorted(root.rglob("*.jsonl")):
        out.extend(iter_transcripts(fp))
    return out


def scan_race_summaries(root: str | Path) -> list[dict[str, Any]]:
    """Recursively find ensemble ``summary.json`` race files under ``root``.

    A JSON file is treated as a race summary iff it has ``race_id``, ``task_id``
    and ``winner`` keys (path4/ensemble format). Anything else is ignored.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    races: list[dict[str, Any]] = []
    for fp in sorted(root.rglob("summary.json")):
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(obj, dict) and "race_id" in obj and "task_id" in obj and "winner" in obj:
            races.append(obj)
    return races


def _category_of(t: Transcript) -> str:
    cat = getattr(t, "category", None)
    return cat if isinstance(cat, str) and cat else UNKNOWN_CATEGORY


def _pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k; ``None`` when k > n (cannot estimate)."""
    if n <= 0 or k > n:
        return None
    if k <= 0:
        return None
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def _maj_at_k(outcomes: list[bool], k: int) -> bool | None:
    """Maj@k over the first k episodes; ``None`` if fewer than k exist."""
    if len(outcomes) < k:
        return None
    first = outcomes[:k]
    return sum(1 for s in first if s) * 2 > k  # strictly more than k/2 solved


def aggregate(
    transcripts: Iterable[Transcript],
    races: Iterable[dict[str, Any]] | None = None,
    ks: Iterable[int] = DEFAULT_KS,
) -> dict[str, Any]:
    """Aggregate transcripts into the scoreboard summary payload.

    Returns a JSON-friendly dict with one entry per policy (deterministic
    ordering: policy name; categories sorted within each policy).
    """
    ks = tuple(sorted({k for k in ks if k >= 1}))
    ts = list(transcripts)
    races = list(races or [])

    # group episodes per (policy, task)
    per_task: dict[tuple[str, str], list[Transcript]] = {}
    for t in ts:
        per_task.setdefault((t.policy, t.task_id), []).append(t)

    policies: dict[str, dict[str, Any]] = {}

    def _policy(name: str) -> dict[str, Any]:
        return policies.setdefault(
            name,
            {
                "policy": name,
                "episodes": 0,
                "solves": 0,
                "solve_rate": 0.0,
                "tasks": 0,
                "pass_at_k": {str(k): None for k in ks},
                "maj_at_k": {str(k): None for k in ks},
                "categories": {},
                "race_wins": 0,
                "avg_steps_solved": None,
                "avg_steps_unsolved": None,
                "avg_tokens_solved": None,
                "avg_tokens_unsolved": None,
            },
        )

    # (policy, task) level outcomes, sorted deterministically per task
    for (policy, task_id) in sorted(per_task):
        eps = sorted(per_task[(policy, task_id)], key=lambda t: t.episode_id)
        p = _policy(policy)
        p["tasks"] += 1
        for k in ks:
            if k == 1:
                continue  # Pass@1 handled at episode level below
            vals = [_pass_at_k(len(eps), sum(1 for e in eps if e.solved), k)]
            vals = [v for v in vals if v is not None]
            if vals:
                acc = p["pass_at_k"][str(k)]
                if acc is None:
                    acc = []
                    p["pass_at_k"][str(k)] = acc
                acc.append(vals[0])
            m = _maj_at_k([e.solved for e in eps], k)
            if m is not None:
                acc = p["maj_at_k"][str(k)]
                if acc is None:
                    acc = []
                    p["maj_at_k"][str(k)] = acc
                acc.append(1.0 if m else 0.0)

    # episode-level stats
    for t in ts:
        p = _policy(t.policy)
        p["episodes"] += 1
        p["solves"] += 1 if t.solved else 0
        cat = _category_of(t)
        c = p["categories"].setdefault(
            cat, {"category": cat, "episodes": 0, "solves": 0, "solve_rate": 0.0}
        )
        c["episodes"] += 1
        c["solves"] += 1 if t.solved else 0

    # finalize: average the per-task lists, compute rates
    for name, p in policies.items():
        if p["episodes"]:
            p["solve_rate"] = round(p["solves"] / p["episodes"], 6)
        for k in ks:
            if k == 1:
                p["pass_at_k"]["1"] = round(p["solve_rate"], 6)
            else:
                for key in ("pass_at_k", "maj_at_k"):
                    acc = p[key][str(k)]
                    if isinstance(acc, list):
                        p[key][str(k)] = round(sum(acc) / len(acc), 6)
        for cat in p["categories"]:
            c = p["categories"][cat]
            c["solve_rate"] = round(c["solves"] / c["episodes"], 6) if c["episodes"] else 0.0
        p["categories"] = {c: p["categories"][c] for c in sorted(p["categories"])}

    # steps/tokens solved vs unsolved
    solved_steps: dict[str, list[int]] = {}
    unsolved_steps: dict[str, list[int]] = {}
    solved_tok: dict[str, list[int]] = {}
    unsolved_tok: dict[str, list[int]] = {}
    for t in ts:
        d_s = solved_steps if t.solved else unsolved_steps
        d_t = solved_tok if t.solved else unsolved_tok
        d_s.setdefault(t.policy, []).append(t.steps)
        d_t.setdefault(t.policy, []).append(t.tokens_in + t.tokens_out)

    def _avg(xs: list[int]) -> float | None:
        return round(sum(xs) / len(xs), 2) if xs else None

    for name in policies:
        p = policies[name]
        p["avg_steps_solved"] = _avg(solved_steps.get(name, []))
        p["avg_steps_unsolved"] = _avg(unsolved_steps.get(name, []))
        p["avg_tokens_solved"] = _avg(solved_tok.get(name, []))
        p["avg_tokens_unsolved"] = _avg(unsolved_tok.get(name, []))

    # race wins
    race_total = 0
    for r in races:
        winner = r.get("winner")
        if isinstance(winner, str) and winner:
            _policy(winner)["race_wins"] += 1
            race_total += 1

    return {
        "ks": list(ks),
        "episodes": len(ts),
        "races": race_total,
        "policies": [policies[n] for n in sorted(policies)],
    }


def render_table(summary: dict[str, Any]) -> "rich.table.Table":  # noqa: F821
    """Rich console table for CLI use (imported lazily-friendly)."""
    from rich.table import Table

    ks = summary.get("ks", list(DEFAULT_KS))
    t = Table(title=f"CTF Scoreboard — {summary['episodes']} episodes", header_style="bold cyan")
    t.add_column("policy", style="bold")
    t.add_column("eps", justify="right")
    t.add_column("solve rate", justify="right")
    for k in ks:
        t.add_column(f"Pass@{k}", justify="right")
    for k in ks:
        if k > 1:
            t.add_column(f"Maj@{k}", justify="right")
    t.add_column("race W", justify="right")
    t.add_column("categories", overflow="fold")

    def fmt(v):
        return "—" if v is None else f"{v:.3f}"

    for p in sorted(summary["policies"], key=lambda p: -(p["pass_at_k"].get(str(max(ks)), 0) or 0)):
        cats = "  ".join(
            f"{c}:{v['solves']}/{v['episodes']}" for c, v in p["categories"].items()
        )
        t.add_row(
            p["policy"],
            str(p["episodes"]),
            fmt(p["solve_rate"]),
            *[fmt(p["pass_at_k"][str(k)]) for k in ks],
            *[fmt(p["maj_at_k"][str(k)]) for k in ks if k > 1],
            str(p["race_wins"]),
            cats,
        )
    return t
