from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from path4.coldstart.train_sft import (
    IGNORE_INDEX,
    app,
    build_labels,
    char_tokenizer_len,
    dry_run_stats,
    labels_to_flat,
    token_counts,
)

RUNNER = CliRunner()


def _record():
    return {
        "task_id": "t0", "episode_id": "e0",
        "messages": [
            {"role": "user", "content": "hello"},          # 5 chars
            {"role": "assistant", "content": "run ls"},    # 6
            {"role": "tool", "content": "flag.txt"},       # 8
            {"role": "assistant", "content": "done"},      # 4
        ],
        "mask": [False, True, False, True],
    }


def test_token_counts_with_fake_tokenizer():
    r = _record()
    assert token_counts(r["messages"], char_tokenizer_len) == [5, 6, 8, 4]


def test_build_labels_masks_non_assistant():
    r = _record()
    labels = build_labels(r["messages"], r["mask"], char_tokenizer_len)
    assert labels[0] == [IGNORE_INDEX] * 5
    assert labels[2] == [IGNORE_INDEX] * 8
    # assistant spans carry global positions 5..10 and 14..17
    assert labels[1] == list(range(5, 11))
    assert labels[3] == list(range(19, 23))


def test_labels_flat_learned_fraction():
    flat = labels_to_flat(build_labels(_record()["messages"], _record()["mask"],
                                       char_tokenizer_len))
    assert len(flat) == 23
    assert sum(1 for v in flat if v != IGNORE_INDEX) == 10  # only assistant tokens


def test_all_masked_is_all_ignore():
    msgs = [{"role": "tool", "content": "out"}]
    labels = build_labels(msgs, [False], char_tokenizer_len)
    assert labels == [[IGNORE_INDEX] * 3]


def test_module_imports_without_torch():
    import importlib
    import sys
    assert "torch" not in sys.modules and "trl" not in sys.modules
    importlib.import_module("path4.coldstart.train_sft")


def test_dry_run_cli(tmp_path):
    ds = tmp_path / "ds.jsonl"
    ds.write_text(json.dumps(_record()) + "\n")
    result = RUNNER.invoke(app, ["--dataset", str(ds), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "dry-run OK" in result.output


def test_dry_run_stats():
    stats = dry_run_stats([_record()])
    assert stats["records"] == 1
    assert stats["tokens"] == 23
    assert stats["learned_tokens"] == 10
    assert stats["longest_record_tokens"] == 23
