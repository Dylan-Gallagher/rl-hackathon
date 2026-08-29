"""Build the cold-start SFT dataset from alloy success traces.

Input: a transcripts dir/file/glob (contracts.iter_transcripts, README §3.2).
Output: JSONL SFT records::

    {"task_id": ..., "episode_id": ..., "messages": [{"role","content"}, ...],
     "mask": [true/false, ...]}

``mask[i]`` is True only for assistant messages — the loss is computed on
assistant turns only (README Path 2 step 5: "mask non-assistant turns").

Filters, in order:
1. ``solved == true`` only (we distill successes).
2. ``steps <= max_steps`` (default 40 = task-horizon default in §3.1; longer
   episodes are dropped — exposure bias on long horizons, README §1.3).
3. Provider-artifact strip: lines starting with provider thought-leak markers
   like ``assistant&nbsp;thought`` are removed (simple, documented below).
4. Per-message char cap via ``contracts.capped_output`` (head/tail policy).
5. Dedup identical ``(task_id, sha256(messages))``.

CLI (typer): see ``python -m path4.coldstart.build_dataset --help``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import random
import re
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from contracts import capped_output, iter_transcripts  # schemas stay owned by contracts

app = typer.Typer(add_completion=False, help="Build cold-start SFT dataset from success traces.")
console = Console()

# Provider thought-leak artifacts seen in cross-provider transcripts: lines
# that begin with 'assistant'/'system' glued to '&nbsp;' / U+00A0 / space and
# then 'thought' (e.g. "assistant&nbsp;thought: maybe try rot13"). Simple
# line-prefix rule — deliberately conservative, documented and unit-tested.
_ARTIFACT_RE = re.compile(r"^\s*(assistant|system)(?:&nbsp;|\u00a0|\s)+thought\b", re.IGNORECASE)

# Transcript extension field (README §3.2 allows extras) used for per-category stats.
_CATEGORY_FIELD = "category"


def strip_artifacts(content: str) -> str:
    """Drop provider thought-leak lines; collapse the resulting gaps."""
    kept = [ln for ln in content.splitlines() if not _ARTIFACT_RE.match(ln)]
    return "\n".join(kept).strip()


def clean_message(content: str, max_chars: int) -> str:
    """Artifact strip + char cap (head/tail via contracts.capped_output)."""
    return capped_output(strip_artifacts(content), max_chars)


def record_hash(task_id: str, messages: list[dict]) -> str:
    """Stable hash over (task_id, messages) for dedup."""
    payload = json.dumps({"task_id": task_id, "messages": messages},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def discover_transcripts(paths_or_dir: str | Path) -> list[Path]:
    """Resolve a dir/file/glob into a deduped list of ``*.jsonl`` files.

    ``contracts.iter_transcripts`` is non-recursive (top-level ``*.jsonl`` of a
    dir), but race runs write transcripts one level deeper
    (``runs/live/<task_id>/<episode_id>.jsonl``). So when given a directory we
    walk the tree ourselves (``rglob``) and hand each file to it — mirroring
    ``path4/scoreboard/metrics.py``. File and glob args pass through as-is;
    dedupe in case a glob and rglob could both match.
    """
    p = Path(paths_or_dir)
    if p.is_dir():
        files = sorted(p.rglob("*.jsonl"))
    else:
        files = sorted(Path(x) for x in glob.glob(str(p))) or [p]
    return list(dict.fromkeys(files))


def build_records(
    transcripts_dir: str | Path,
    max_steps: int = 40,
    max_chars: int = 4000,
) -> tuple[list[dict], dict]:
    """Filter/clean solved transcripts into SFT records + stats dict."""
    records: list[dict] = []
    seen: set[str] = set()
    stats = {
        "episodes_in": 0,
        "episodes_unsolved": 0,
        "episodes_too_long": 0,
        "episodes_deduped": 0,
        "episodes_out": 0,
        "messages": 0,
        "assistant_messages": 0,
        "turns_total": 0,
        "by_category": {},
    }
    for fp in discover_transcripts(transcripts_dir):
        for t in iter_transcripts(fp):
            stats["episodes_in"] += 1
            if not t.solved:
                stats["episodes_unsolved"] += 1
                continue
            if t.steps > max_steps:
                stats["episodes_too_long"] += 1
                continue
            messages = [{"role": m.role, "content": clean_message(m.content, max_chars)}
                        for m in t.messages]
            if not any(m["role"] == "assistant" for m in messages):
                continue  # nothing to learn from
            h = record_hash(t.task_id, messages)
            if h in seen:
                stats["episodes_deduped"] += 1
                continue
            seen.add(h)
            mask = [m["role"] == "assistant" for m in messages]
            records.append({"task_id": t.task_id, "episode_id": t.episode_id,
                            "messages": messages, "mask": mask})
            stats["messages"] += len(messages)
            stats["assistant_messages"] += sum(mask)
            stats["turns_total"] += t.steps
            cat = getattr(t, _CATEGORY_FIELD, None)
            if cat:
                stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
    stats["episodes_out"] = len(records)
    return records, stats


def split_records(records: list[dict], val_frac: float, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Shuffle-split into (train, val). Empty val when val_frac <= 0."""
    if val_frac <= 0:
        return list(records), []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_frac)) if records else 0
    n_val = min(n_val, len(shuffled))
    return shuffled[n_val:], shuffled[:n_val]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def print_stats(stats: dict) -> None:
    console.print(f"[bold]episodes:[/] in={stats['episodes_in']} "
                  f"unsolved={stats['episodes_unsolved']} "
                  f"too_long={stats['episodes_too_long']} "
                  f"deduped={stats['episodes_deduped']} "
                  f"out={stats['episodes_out']}")
    if stats["episodes_out"]:
        console.print(f"[bold]messages:[/] {stats['messages']} "
                      f"(assistant={stats['assistant_messages']}, "
                      f"{100 * stats['assistant_messages'] / max(1, stats['messages']):.1f}%) "
                      f"avg turns/episode={stats['turns_total'] / stats['episodes_out']:.1f}")
    if stats["by_category"]:
        table = Table(title="episodes by category (Transcript extension field)")
        table.add_column("category")
        table.add_column("episodes", justify="right")
        for cat, n in sorted(stats["by_category"].items()):
            table.add_row(cat, str(n))
        console.print(table)


@app.command()
def main(
    transcripts_dir: str = typer.Argument(..., help="Dir/file/glob of §3.2 JSONL transcripts"),
    out: str = typer.Option("path4/coldstart/data/sft_train.jsonl", "--out", "-o",
                            help="Output train JSONL (val gets .val suffix)"),
    max_steps: int = typer.Option(40, help="Episode step cap (§3.1 horizon default 40)"),
    max_chars: int = typer.Option(4000, help="Max chars per message (capped_output head/tail)"),
    val_frac: float = typer.Option(0.05, min=0.0, max=0.5, help="Validation split fraction"),
    seed: int = typer.Option(0, help="Split RNG seed"),
):
    """Build the cold-start SFT dataset from alloy success traces."""
    records, stats = build_records(transcripts_dir, max_steps=max_steps, max_chars=max_chars)
    if not records:
        console.print("[red]no SFT records produced[/] — check transcripts dir/filtering; "
                      f"got {stats['episodes_in']} episodes, 0 records out")
        raise SystemExit(1)
    train, val = split_records(records, val_frac, seed=seed)
    write_jsonl(train, out)
    if val:
        write_jsonl(val, str(Path(out).with_suffix("")) + ".val.jsonl")
    print_stats(stats)
    console.print(f"[green]wrote[/] {len(train)} train -> {out}"
                  + (f" and {len(val)} val -> {Path(out).with_suffix('')}.val.jsonl" if val else ""))


if __name__ == "__main__":
    app()
