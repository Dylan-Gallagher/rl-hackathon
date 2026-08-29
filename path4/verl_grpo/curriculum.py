"""Curriculum from EI pass-rate bands (README Path 4 step 2; Path 3 EI output).

Input: EI stats as either ``{"task_id": {"attempts": int, "solves": int}}`` or
``[{"task_id": ..., "attempts": ..., "solves": ...}, ...]``. Output: tasks in
the learnable band (default 5%–40% pass rate), sorted by distance from the
20% sweet spot (enough signal for GRPO group variance, not hopeless).
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_BAND = (0.05, 0.40)
SWEET_SPOT = 0.20


def load_stats(stats: dict | list | str | Path) -> list[dict]:
    """Normalize EI stats (dict / list / JSON path) to [{task_id, attempts, solves}]."""
    if isinstance(stats, (str, Path)):
        stats = json.loads(Path(stats).read_text())
    if isinstance(stats, dict):
        return [{"task_id": tid, **v} for tid, v in stats.items()]
    return list(stats)


def pass_rate(entry: dict) -> float | None:
    """Pass rate, or None if 0 attempts (division-safe: unmeasured ≠ 0%)."""
    attempts = int(entry.get("attempts", 0))
    if attempts <= 0:
        return None
    return int(entry.get("solves", 0)) / attempts


def curriculum(entries: list[dict], band: tuple[float, float] = DEFAULT_BAND,
               sweet_spot: float = SWEET_SPOT) -> list[dict]:
    """Keep tasks with band_low <= pass_rate <= band_high; sort by |rate - sweet_spot|.

    Boundary values are INCLUSIVE (5% and 40% both learnable). Zero-attempt
    tasks are excluded (unmeasured). Ties broken by task_id for determinism.
    """
    lo, hi = sorted(band)
    kept = []
    for e in entries:
        rate = pass_rate(e)
        if rate is None or not (lo <= rate <= hi):
            continue
        kept.append({**e, "pass_rate": rate, "distance": abs(rate - sweet_spot)})
    kept.sort(key=lambda e: (e["distance"], e["task_id"]))
    return kept


def print_table(entries: list[dict], band: tuple[float, float] = DEFAULT_BAND) -> None:
    table = Table(title=f"GRPO curriculum: learnable band {band[0]:.0%}–{band[1]:.0%} "
                        f"(closest to {SWEET_SPOT:.0%} first)")
    table.add_column("task_id")
    table.add_column("solves/attempts", justify="right")
    table.add_column("pass_rate", justify="right")
    table.add_column("|rate − 20%|", justify="right")
    for e in entries:
        table.add_row(e["task_id"], f"{e.get('solves', 0)}/{e.get('attempts', 0)}",
                      f"{e['pass_rate']:.2f}", f"{e['distance']:.2f}")
    console.print(table)


def run(stats_input, out: str | None = None, band: tuple[float, float] = DEFAULT_BAND) -> list[dict]:
    """Load -> filter+sort -> optional JSON dump -> rich table. Returns the list."""
    entries = curriculum(load_stats(stats_input), band=band)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(entries, indent=2) + "\n")
        console.print(f"[green]wrote[/] {len(entries)} tasks -> {out}")
    print_table(entries, band=band)
    return entries


if __name__ == "__main__":  # tiny manual entry point
    import argparse

    ap = argparse.ArgumentParser(
        description="GRPO curriculum from EI pass-rate stats (learnable band, "
                    "closest-to-20% first)."
    )
    ap.add_argument("stats", help="EI stats: JSON path, dict, or list")
    ap.add_argument("out", nargs="?", default=None,
                    help="optional output JSON path for the curriculum")
    args = ap.parse_args()
    run(args.stats, args.out)
