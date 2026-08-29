from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from path4.coldstart.build_dataset import (
    app,
    build_records,
    clean_message,
    record_hash,
    split_records,
)

FIXTURES = Path(__file__).parent / "fixtures"
TRANSCRIPTS = FIXTURES / "transcripts.jsonl"


def test_fixture_contains_expected_episodes():
    lines = TRANSCRIPTS.read_text().strip().splitlines()
    assert len(lines) == 5  # solved-alloy, unsolved, too-long, duplicate, long-message
    recs = [json.loads(x) for x in lines]
    assert sum(r["solved"] for r in recs) == 4  # A, C(too-long), D(dup), E
    # mixed roles + per-assistant-turn models (§3.2 alloy pattern)
    first = recs[0]
    models = {m["model"] for m in first["messages"] if m["role"] == "assistant"}
    assert len(models) >= 2


def test_build_records_filters_and_masks():
    records, stats = build_records(TRANSCRIPTS, max_steps=40, max_chars=4000)
    # unsolved (B), too-long (C) dropped; duplicate (D) deduped -> A and E remain
    assert [r["episode_id"] for r in records] == ["alloy-rc-001", "solo-gem-002"]
    assert stats["episodes_in"] == 5
    assert stats["episodes_unsolved"] == 1
    assert stats["episodes_too_long"] == 1
    assert stats["episodes_deduped"] == 1
    assert stats["episodes_out"] == 2
    assert stats["by_category"] == {"crypto": 2}
    # mask: only assistant messages
    for r in records:
        assert r["mask"] == [m["role"] == "assistant" for m in r["messages"]]
    a = records[0]
    assert a["mask"] == [False, True, False, True, False, True, False]


def test_artifact_stripping():
    dirty = "assistant&nbsp;thought: hmm\nRun: cat flag.txt\nassistant\u00a0thought: more"
    assert clean_message(dirty, 4000) == "Run: cat flag.txt"


def test_char_cap_uses_contracts_capped_output():
    out = clean_message("X" * 9000, 4000)
    assert len(out) < 5000
    assert "[truncated" in out


def test_dedup_hash_is_content_sensitive():
    msgs = [{"role": "assistant", "content": "hi"}]
    h1 = record_hash("t", msgs)
    assert h1 == record_hash("t", list(msgs))
    assert h1 != record_hash("t2", msgs)
    assert h1 != record_hash("t", [{"role": "assistant", "content": "bye"}])


def test_split_records_deterministic():
    recs = [{"task_id": str(i), "episode_id": str(i), "messages": [], "mask": []}
            for i in range(10)]
    tr1, va1 = split_records(recs, 0.2, seed=0)
    tr2, va2 = split_records(recs, 0.2, seed=0)
    assert (tr1, va1) == (tr2, va2)
    assert len(va1) == 2 and len(tr1) == 8
    assert split_records(recs, 0.0)[1] == []


@pytest.mark.parametrize("args", [
    ["--help"],
    [str(TRANSCRIPTS), "--out", "does_not_matter.jsonl", "--max-steps", "40"],
])
def test_cli(tmp_path, args):
    runner = CliRunner()
    if args[0] == "--help":
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        return
    out = tmp_path / "sft_train.jsonl"
    args[2] = str(out)
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    lines = out.read_text().strip().splitlines()
    # 2 kept records, val_frac 0.05 -> 1 train + 1 val
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert {"task_id", "episode_id", "messages", "mask"} <= set(rec)
    val = tmp_path / "sft_train.val.jsonl"
    assert val.exists() and len(val.read_text().strip().splitlines()) >= 1
