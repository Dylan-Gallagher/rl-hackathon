"""Task schema per README §3.1.

Field names and order must match the shared contracts EXACTLY — every path
imports this model, so changes here are breaking API changes.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TaskSource = Literal["nyuctf", "random-crypto", "custom"]
TaskCategory = Literal["pwn", "rev", "crypto", "web", "forensics", "misc"]
LaunchMode = Literal["supervisor", "compose", "repl", "none"]
FlagMode = Literal["generated", "static"]
FlagVerify = Literal["regex", "exact", "script"]
TaskSplit = Literal["train", "eval"]


class TaskEnv(BaseModel):
    """Where and how the challenge executes."""

    image: str = ""
    launch: LaunchMode = "none"


class TaskFlag(BaseModel):
    """Flag provisioning and verification policy.

    mode=generated -> fresh ``flag{uuid4}`` injected per episode (§1.5 anti-cheat).
    mode=static    -> the fixed flag stored (verbatim) in ``format``.
    verify=script  -> verification script is executed by the env backend OUTSIDE
    the sandbox (docstring contract; see contracts/flag.py).
    """

    mode: FlagMode = "generated"
    verify: FlagVerify = "exact"
    format: str = "flag{uuid4}"


class Horizon(BaseModel):
    """Episode budget."""

    max_steps: int = Field(default=40, ge=1)
    timeout_s: int = Field(default=1800, ge=1)


class Task(BaseModel):
    """A single CTF task (README §3.1, field-for-field)."""

    task_id: str
    source: TaskSource
    category: TaskCategory
    env: TaskEnv = TaskEnv()
    flag: TaskFlag = TaskFlag()
    prompt: str = ""
    horizon: Horizon = Horizon()
    split: TaskSplit = "train"

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Build a Task from a plain dict (tolerant of JSON-loaded input)."""
        return cls.model_validate(data)

    @classmethod
    def load(cls, path: str | Path) -> "Task":
        """Load a Task from a JSON file."""
        return cls.model_validate(json.loads(Path(path).read_text()))


def load_tasks(dir_or_glob: str | Path) -> list[Task]:
    """Load tasks from a directory of ``*.json`` files or a glob pattern."""
    p = str(dir_or_glob)
    if Path(p).is_dir():
        paths = sorted(glob.glob(f"{p}/*.json"))
    else:
        paths = sorted(glob.glob(p))
    tasks = [Task.load(fp) for fp in paths if fp.endswith(".json")]
    logger.debug("loaded %d tasks from %s", len(tasks), dir_or_glob)
    return tasks
