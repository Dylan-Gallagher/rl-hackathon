"""Random-Crypto CSV -> deterministic repl tasks.

Upstream columns (exact): input,hint,flag,archetype,subtype,difficulty.
Public task JSON never contains the plaintext flag; only its SHA-256 digest
(the exact flag goes to the external verifier record). Task IDs are a
deterministic function of the row content, so the same CSV always produces the
same registry (stable episode bookkeeping across runs).

Also provides an adapter that shells out to the upstream MIT-licensed
generator for infinite fresh tasks (optional; requires the upstream repo).
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

from ctf_gym.contracts import (
    EnvSpec,
    FlagInjection,
    FlagSpec,
    Horizon,
    Task,
    ValidationError,
    secure_verifier_write,
    sha256_hex,
)
from ctf_gym.verifier import make_verifier_record

CSV_COLUMNS = ("input", "hint", "flag", "archetype", "subtype", "difficulty")


class ConverterError(ValueError):
    pass


@dataclass
class ConvertedTask:
    task: Task
    static_flag: str

    def write(self, out_dir: str, verifiers_dir: Optional[str] = None) -> str:
        """Write <out_dir>/<task_id>.json (public) and verifier sidecar (0600)."""
        os.makedirs(out_dir, exist_ok=True)
        if verifiers_dir is None:
            verifiers_dir = os.path.join(os.path.dirname(out_dir.rstrip("/")) or ".", "verifiers")
        os.makedirs(verifiers_dir, exist_ok=True)
        public_path = os.path.join(out_dir, f"{self.task.task_id}.json")
        with open(public_path, "w", encoding="utf-8") as f:
            json.dump(self.task.to_public_dict(), f, indent=2, sort_keys=True)
            f.write("\n")
        make_verifier_record(self.task.task_id, self.task.flag,
                             static_flag=self.static_flag).write(
            os.path.join(verifiers_dir, f"{self.task.task_id}.json"))
        return public_path


def task_id_for_row(row: dict[str, str], split: str, index: int) -> str:
    digest_material = json.dumps(
        {c: row.get(c, "") for c in CSV_COLUMNS}, sort_keys=True, ensure_ascii=False
    )
    return f"rc-{split}-{sha256_hex(digest_material)[:16]}-{index:05d}"


def build_prompt(row: dict[str, str]) -> str:
    lines = [
        "You are solving a cryptography CTF challenge.",
        "Interact by writing Python code; each action is executed as a Python program",
        "in an isolated REPL with no network access. Print the flag when you find it.",
        "",
        f"Challenge input:\n{row['input']}",
    ]
    if row.get("hint"):
        lines.append(f"\nHint: {row['hint']}")
    return "\n".join(lines)


def row_to_task(row: dict[str, str], split: str, index: int,
                image: str = "ctf-gym/python-repl:latest",
                max_steps: int = 30, timeout_s: int = 900) -> tuple[Task, str]:
    missing = [c for c in ("input", "flag") if not row.get(c)]
    if missing:
        raise ConverterError(f"CSV row missing required column(s): {missing}")
    flag = row["flag"].strip()
    task_id = task_id_for_row(row, split, index)
    task = Task(
        task_id=task_id,
        source="random-crypto",
        category="crypto",
        env=EnvSpec(image=image, launch="repl"),
        flag=FlagSpec(
            mode="static",
            verify="exact",
            format="flag{...}",
            sha256=sha256_hex(flag),
            injection=FlagInjection(mode="none"),  # repl: flag computed, never placed in sandbox
        ),
        prompt=build_prompt(row),
        horizon=Horizon(max_steps=max_steps, timeout_s=timeout_s),
        split=split,
        metadata={
            "archetype": row.get("archetype", ""),
            "subtype": row.get("subtype", ""),
            "difficulty": row.get("difficulty", ""),
        },
    )
    task.validate()
    return task, flag


def convert_csv(csv_path: str, split: str = "train", out_dir: Optional[str] = None,
                verifiers_dir: Optional[str] = None, image: str = "ctf-gym/python-repl:latest",
                limit: Optional[int] = None) -> list[ConvertedTask]:
    """Read a Random-Crypto CSV and write public tasks + external verifier records.

    NOTE: repl tasks do not inject the flag into the sandbox at all; the exact
    flag exists only in the host-side verifier record, so there is nothing to
    leak inside the sandbox.
    """
    if split not in ("train", "eval"):
        raise ConverterError(f"split must be train|eval, got {split!r}")
    converted: list[ConvertedTask] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or any(c not in reader.fieldnames for c in CSV_COLUMNS):
            raise ConverterError(
                f"CSV must have exact columns {CSV_COLUMNS}, got {reader.fieldnames}"
            )
        for i, row in enumerate(reader):
            if limit is not None and len(converted) >= limit:
                break
            task, flag = row_to_task(row, split=split, index=i, image=image)
            converted.append(ConvertedTask(task=task, static_flag=flag))
    if out_dir:
        for ct in converted:
            ct.write(out_dir, verifiers_dir)
    return converted


def run_upstream_generator(repo_path: str, out_csv: str, n: int, difficulty: str = "easy",
                           extra_args: Optional[Iterable[str]] = None) -> str:
    """Adapter: run the upstream MIT-licensed Random-Crypto generator.

    Requires a local clone of the upstream repo; we do not vendor it.
    """
    if not os.path.isdir(repo_path):
        raise ConverterError(
            f"upstream generator repo not found at {repo_path}; clone "
            "https://github.com/aielte-research/Random-Crypto first"
        )
    cmd = [sys.executable, os.path.join(repo_path, "generate.py"),
           "-n", str(n), "-d", difficulty, "-o", out_csv]
    if extra_args:
        cmd += list(extra_args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise ConverterError(f"upstream generator failed ({proc.returncode}): {proc.stderr[-2000:]}")
    if not os.path.isfile(out_csv):
        raise ConverterError(f"generator did not produce {out_csv}")
    return out_csv


def load_registry(tasks_dir: str) -> list[Task]:
    tasks = []
    for name in sorted(os.listdir(tasks_dir)):
        if name.endswith(".json"):
            with open(os.path.join(tasks_dir, name), encoding="utf-8") as f:
                tasks.append(Task.from_dict(json.load(f)))
    for t in tasks:
        t.validate()
    return tasks
