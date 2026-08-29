"""Transcript format per README §3.2 — the universal artifact.

One JSON object per episode, one message per element of ``messages``; written
as a single JSONL line (one episode per line). Everything downstream (SFT
datasets, EI filtering, eval metrics, GRPO reward attribution) reads this.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TaskSplit = Literal["train", "eval"]

_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_FORMAT_STYLE_RE = re.compile(r"\{[^}]*\}")  # strip '{...}' template holes


class TranscriptMessage(BaseModel):
    """One turn of an episode transcript.

    ``model`` is the LiteLLM model *name* that produced an assistant turn
    (e.g. ``anthropic/claude-sonnet-4`` or the ``alloy`` group), ``None`` for
    tool/user/observation turns.
    """

    turn: int
    role: str
    content: str
    model: str | None = None


class Transcript(BaseModel):
    """One full episode (README §3.2, field-for-field).

    Optional extension fields (e.g. ``category``, ``race_id``, ``started_at``)
    are allowed via ``model_config extra='allow'`` — readers must treat unknown
    fields as optional extensions, never required.
    """

    model_config = {"extra": "allow"}

    task_id: str
    episode_id: str
    policy: str  # 'solo:MODEL' | 'alloy:w1,w2' | 'race:k'
    split: TaskSplit = "train"
    messages: list[TranscriptMessage] = Field(default_factory=list)
    solved: bool = False
    steps: int = 0
    flags_found: list[str] = Field(default_factory=list)
    sandbox_id: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0


def write_transcript(t: Transcript, path: str | Path) -> None:
    """Append one transcript as a single JSONL line to ``path``.

    JSON-safe by construction (json.dumps) and one-line (no newlines in the
    serialized payload; content newlines are escaped by the encoder).
    """
    line = t.model_dump_json()
    assert "\n" not in line  # model_dump_json never emits raw newlines
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def iter_transcripts(paths_or_dir: str | Path) -> Iterator[Transcript]:
    """Yield Transcripts from ``*.jsonl`` files.

    ``paths_or_dir`` may be a directory (all ``*.jsonl`` inside), a single
    file, or a glob pattern. Malformed lines are skipped with a log warning.
    """
    p = Path(paths_or_dir)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    elif p.is_file():
        files = [p]
    else:
        files = sorted(Path().glob(str(paths_or_dir)))
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield Transcript.model_validate(json.loads(line))
                except Exception:
                    logger.warning("skipping malformed transcript %s:%d", fp, lineno)


def episode_id_for(policy: str, task_id: str, idx: int) -> str:
    """Build a filename-safe episode id: ``{policy}:{task_id}:{idx}``.

    Sanitized so the id can be used directly in sandbox labels / filenames.
    """
    raw = f"{policy}:{task_id}:{idx}"
    # Drop format-template holes ('{uuid4}') before sanitizing characters.
    raw = _FORMAT_STYLE_RE.sub("", raw)
    return _SANITIZE_RE.sub("-", raw).strip("-")
