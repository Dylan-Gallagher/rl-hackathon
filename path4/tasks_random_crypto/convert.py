"""Convert Random-Crypto CSV rows to contracts Task JSON (README §3.1).

- TRAIN pool: 60 ``easy`` tasks from non_verified_challenges_5000.
- EVAL: exactly 20 tasks from verified_challenges_50 (LOCKED — never regenerated).
- Archetypes restricted to stdlib-solvable ones (pure python3: classical_cipher,
  hash, prng, rsa). Selection is deterministic (seeded shuffle).

Usage: python -m path4.tasks_random_crypto.convert
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CH_5000 = REPO / "path1/random_crypto/challenges/non_verified_challenges_5000/all_challenges.csv"
CH_50 = REPO / "path1/random_crypto/challenges/verified_challenges_50/all_challenges.csv"
OUT_TRAIN = Path(__file__).resolve().parent / "tasks" / "train"
OUT_EXTRA = Path(__file__).resolve().parent / "tasks" / "train_extra"
OUT_EVAL = Path(__file__).resolve().parent / "tasks" / "eval"

#: Solvable with pure python3 (hashlib/itertools/sympy-free math).
SOLVABLE_ARCHETYPES = {"classical_cipher", "hash", "prng", "rsa"}

N_TRAIN = 60
N_EXTRA = 300
N_EVAL = 20
MAX_STEPS = 12
TIMEOUT_S = 600

SUFFIX = (
    "The flag is embedded as flag{...}; solve the challenge with python and "
    "verify by running `submit flag{...}`.\n"
    "Shell supports ONLY these commands:\n"
    "- `python <code>` — run a ONE-LINE python3 snippet (executed as python3 -c <code>); "
    "prefer simple one-liners without nested quotes\n"
    "- `ls`\n"
    "- `cat challenge.py`\n"
    "- `submit flag{...}` — verify the final flag"
)


def _row_prompt(row: dict) -> str:
    q = (row.get("question") or "").strip()
    ct = (row.get("ciphertext") or "").strip()
    parts = [q]
    if ct and ct not in q:
        parts.append(f"Ciphertext: {ct}")
    parts.append(SUFFIX)
    return "\n\n".join(parts)


def _task_dict(row: dict, task_id: str, split: str) -> dict:
    return {
        "task_id": task_id,
        "source": "random-crypto",
        "category": "crypto",
        "env": {"image": "", "launch": "repl"},
        "flag": {"mode": "static", "verify": "exact", "format": row["flag"].strip()},
        "prompt": _row_prompt(row),
        "horizon": {"max_steps": MAX_STEPS, "timeout_s": TIMEOUT_S},
        "split": split,
    }


def _load(path: Path, easy_only: bool) -> list[dict]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for r in rows if r["archetype"] in SOLVABLE_ARCHETYPES and r["flag"].strip()]
    if easy_only:
        rows = [r for r in rows if r["difficulty"] == "easy"]
    return rows


def build(out_train: Path = OUT_TRAIN, out_eval: Path = OUT_EVAL) -> dict:
    train_rows = _load(CH_5000, easy_only=True)
    eval_rows = _load(CH_50, easy_only=False)
    rng = random.Random(20260829)
    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    train_rows = train_rows[:N_TRAIN]
    eval_rows = eval_rows[:N_EVAL]
    assert len(train_rows) == N_TRAIN, f"only {len(train_rows)} train candidates"
    assert len(eval_rows) == N_EVAL, f"only {len(eval_rows)} eval candidates"

    out_train.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(train_rows):
        tid = f"rc-train-{i:03d}-{r['archetype']}-{r['subtype']}"
        (out_train / f"{tid}.json").write_text(
            json.dumps(_task_dict(r, tid, "train"), indent=2))
    # EVAL: lock — write only once.
    lock = out_eval.parent / "EVAL_LOCKED.md"
    if lock.exists():
        return {"written": 0, "note": "eval already locked"}
    out_eval.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(eval_rows):
        tid = f"rc-eval-{i:03d}-{r['archetype']}-{r['subtype']}"
        (out_eval / f"{tid}.json").write_text(
            json.dumps(_task_dict(r, tid, "eval"), indent=2))
    lock.write_text(
        f"EVAL SET LOCKED {N_EVAL} tasks; do not regenerate/delete. "
        "Selection: verified_challenges_50, archetypes "
        f"{sorted(SOLVABLE_ARCHETYPES)}, seed 20260829.\n")
    return {"train": len(train_rows), "eval": len(eval_rows)}


def build_extra(n: int = N_EXTRA, out_extra: Path = OUT_EXTRA) -> dict:
    """Next `n` easy tasks after the original 60-train slice (same shuffle seed)."""
    train_rows = _load(CH_5000, easy_only=True)
    rng = random.Random(20260829)
    rng.shuffle(train_rows)
    extra = train_rows[N_TRAIN:N_TRAIN + n]
    out_extra.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(extra):
        tid = f"rc-extra-{i:03d}-{r['archetype']}-{r['subtype']}"
        (out_extra / f"{tid}.json").write_text(
            json.dumps(_task_dict(r, tid, "train"), indent=2))
    return {"extra": len(extra), "out": str(out_extra)}


if __name__ == "__main__":
    print(build())
    print(build_extra())
