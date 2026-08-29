"""Dataset inspection helpers with no ML dependencies."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

REQUIRED_COLUMNS = {"question", "necessary_info", "flag", "difficulty"}


@dataclass(frozen=True)
class DatasetAudit:
    path: Path
    rows: int
    difficulties: dict[str, int]
    columns: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "rows": self.rows,
            "difficulties": self.difficulties,
            "columns": list(self.columns),
        }


def _read_rows(path: str | Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS.difference(columns)
        if missing:
            raise ValueError(
                f"Dataset {csv_path} is missing required columns: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)
    return rows, columns


def audit_dataset(path: str | Path) -> DatasetAudit:
    rows, columns = _read_rows(path)
    difficulties = Counter((row.get("difficulty") or "").strip() for row in rows)
    return DatasetAudit(
        path=Path(path).resolve(),
        rows=len(rows),
        difficulties=dict(sorted(difficulties.items())),
        columns=columns,
    )


def select_rows(
    rows: Iterable[Mapping[str, str]], difficulties: Iterable[str]
) -> list[Mapping[str, str]]:
    """Select rows by difficulty; ``all`` explicitly disables filtering."""
    selected = {item.strip() for item in difficulties}
    if not selected:
        raise ValueError("At least one difficulty or 'all' is required")
    if "all" in selected:
        if len(selected) != 1:
            raise ValueError("'all' cannot be combined with named difficulties")
        return list(rows)
    return [row for row in rows if (row.get("difficulty") or "").strip() in selected]


def assert_reproduction_datasets(train_path: str | Path, eval_path: str | Path) -> dict[str, DatasetAudit]:
    train = audit_dataset(train_path)
    evaluation = audit_dataset(eval_path)
    errors: list[str] = []
    if train.rows != 5000:
        errors.append(f"training CSV has {train.rows} rows; expected 5000")
    if train.difficulties.get("easy", 0) != 1700:
        errors.append(
            f"training CSV has {train.difficulties.get('easy', 0)} easy rows; expected 1700"
        )
    if evaluation.rows != 50:
        errors.append(f"verified evaluation CSV has {evaluation.rows} rows; expected 50")
    if evaluation.difficulties.get("easy", 0) != 17:
        errors.append(
            f"verified evaluation CSV has {evaluation.difficulties.get('easy', 0)} easy rows; expected 17"
        )
    if evaluation.difficulties.get("not defined", 0) != 33:
        errors.append(
            "verified evaluation CSV has "
            f"{evaluation.difficulties.get('not defined', 0)} 'not defined' rows; expected 33"
        )
    if errors:
        raise ValueError("Dataset audit failed: " + "; ".join(errors))
    return {"train": train, "eval": evaluation}
