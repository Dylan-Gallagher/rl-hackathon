"""Tests for the Random-Crypto -> Task JSON converter."""

from __future__ import annotations

import json

from contracts.task import load_tasks

from path4.tasks_random_crypto.convert import SUFFIX, build


def test_build_counts_and_schema(tmp_path) -> None:
    out = tmp_path / "train"
    out_eval = tmp_path / "eval"
    res = build(out, out_eval)
    assert res["train"] == 60 and res["eval"] == 20
    train = load_tasks(out)
    eval_ = load_tasks(out_eval)
    assert len(train) == 60 and len(eval_) == 20
    ids = {t.task_id for t in train}
    assert len(ids) == 60  # unique ids
    for t in train + eval_:
        assert t.source == "random-crypto" and t.category == "crypto"
        assert t.env.launch == "repl"
        assert t.flag.mode == "static" and t.flag.verify == "exact"
        assert t.flag.format.startswith("flag{") and t.flag.format.endswith("}")
        assert SUFFIX in t.prompt
    assert all(t.split == "train" for t in train)
    assert all(t.split == "eval" for t in eval_)


def test_eval_lock_prevents_regeneration(tmp_path) -> None:
    out, out_eval = tmp_path / "train", tmp_path / "eval"
    build(out, out_eval)
    first = sorted(p.name for p in out_eval.glob("*.json"))
    payload = json.loads((out_eval / first[0]).read_text())
    payload["flag"] = "flag{tampered}"
    (out_eval / first[0]).write_text(json.dumps(payload))
    res = build(out, out_eval)
    assert res == {"written": 0, "note": "eval already locked"}
    assert "flag{tampered}" in (out_eval / first[0]).read_text()
