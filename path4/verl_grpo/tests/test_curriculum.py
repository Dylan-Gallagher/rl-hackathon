from __future__ import annotations

from path4.verl_grpo.curriculum import curriculum, load_stats, pass_rate, run


def _entries():
    return [
        {"task_id": "a", "attempts": 8, "solves": 0},    # 0% -> out
        {"task_id": "b", "attempts": 8, "solves": 1},    # 12.5% -> in, |rate-.2|=.075
        {"task_id": "c", "attempts": 8, "solves": 8},    # 100% -> out
        {"task_id": "d", "attempts": 10, "solves": 2},   # 20% -> in, distance 0 -> first
        {"task_id": "e", "attempts": 100, "solves": 4},  # 4% -> below band
        {"task_id": "f", "attempts": 10, "solves": 4},   # 40% boundary -> inclusive
        {"task_id": "g", "attempts": 10, "solves": 0},   # 0 solves but measured 0% -> out
        {"task_id": "h"},                                 # 0 attempts -> out (unmeasured)
    ]


def test_load_stats_formats(tmp_path):
    as_dict = {"a": {"attempts": 8, "solves": 1}}
    as_list = [{"task_id": "a", "attempts": 8, "solves": 1}]
    p = tmp_path / "s.json"
    p.write_text(__import__("json").dumps(as_dict))
    assert load_stats(as_dict) == load_stats(as_list) == load_stats(p)


def test_pass_rate_division_safety():
    assert pass_rate({"attempts": 0, "solves": 0}) is None
    assert pass_rate({"attempts": 4, "solves": 1}) == 0.25
    assert pass_rate({}) is None


def test_band_boundaries_inclusive():
    band = (0.05, 0.40)
    assert pass_rate({"attempts": 20, "solves": 1}) == 0.05  # exact low edge
    kept = {e["task_id"] for e in curriculum(
        [{"task_id": "lo", "attempts": 20, "solves": 1},
         {"task_id": "hi", "attempts": 10, "solves": 4}], band=band)}
    assert kept == {"lo", "hi"}


def test_curriculum_filters_and_sort():
    out = curriculum(_entries())
    assert [e["task_id"] for e in out] == ["d", "b", "f"]  # closest-to-20% first
    assert out[0]["distance"] == 0.0


def test_run_writes_json_and_prints(tmp_path, capsys):
    p = tmp_path / "cur.json"
    out = run(_entries(), out=str(p))
    assert [e["task_id"] for e in out] == ["d", "b", "f"]
    assert p.exists()
    assert "task_id" in capsys.readouterr().out


def test_custom_band():
    out = curriculum(_entries(), band=(0.0, 0.5))
    ids = [e["task_id"] for e in out]
    assert "a" in ids and "g" in ids and "c" not in ids
