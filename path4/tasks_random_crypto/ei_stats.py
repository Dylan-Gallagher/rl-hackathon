"""Build runs/ei_stats.json from ALL train-trace episodes (solved + unsolved)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from contracts.transcript import iter_transcripts


def build(traces_dir: str | Path, out: str | Path = "runs/ei_stats.json") -> dict:
    attempts: dict[str, int] = defaultdict(int)
    solves: dict[str, int] = defaultdict(int)
    for t in iter_transcripts(Path(traces_dir)):
        attempts[t.task_id] += 1
        solves[t.task_id] += int(bool(t.solved))
    stats = {tid: {"attempts": attempts[tid], "solves": solves[tid]} for tid in sorted(attempts)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    s = build(sys.argv[1] if len(sys.argv) > 1 else "runs/traces/train",
              sys.argv[2] if len(sys.argv) > 2 else "runs/ei_stats.json")
    tot_a = sum(v["attempts"] for v in s.values())
    tot_s = sum(v["solves"] for v in s.values())
    print(f"ei_stats: {len(s)} tasks, {tot_s}/{tot_a} solves")
