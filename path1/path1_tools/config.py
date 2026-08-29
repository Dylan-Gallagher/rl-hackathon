"""TOML configuration loading for the Path 1 pipeline."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

SUPPORTED_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    validate_config(config)
    return config, root


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"run", "eval", "train", "report"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError("Missing config sections: " + ", ".join(sorted(missing)))

    run = config["run"]
    model_id = run.get("model_id")
    if model_id != SUPPORTED_MODEL:
        raise ValueError(
            f"This reproduction supports only {SUPPORTED_MODEL!r}; got {model_id!r}. "
            "The agent masking code relies on Llama 3.1 chat markers."
        )
    if not run.get("train_data") or not run.get("eval_data") or not run.get("output_root"):
        raise ValueError("run.train_data, run.eval_data, and run.output_root are required")

    eval_config = config["eval"]
    if eval_config.get("difficulties") != ["all"]:
        raise ValueError("Paper reproduction evaluation must set difficulties = ['all']")
    if int(eval_config.get("k", 0)) < 1:
        raise ValueError("eval.k must be at least 1")

    train = config["train"]
    if train.get("difficulties") != ["easy"]:
        raise ValueError("Paper reproduction training must set difficulties = ['easy']")
    group_size = int(train.get("group_size", 0))
    if group_size not in {8, 16}:
        raise ValueError("train.group_size must be 8 or 16")
    batch_size = int(train.get("per_device_train_batch_size", group_size))
    if batch_size % group_size:
        raise ValueError(
            "train.per_device_train_batch_size must be divisible by train.group_size"
        )
    if int(train.get("max_steps", 0)) < 1:
        raise ValueError("train.max_steps must be at least 1")


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
